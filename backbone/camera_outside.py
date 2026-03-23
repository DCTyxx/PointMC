import math
from dataclasses import dataclass, field

import torch
from pykdtree.kdtree import KDTree

from utils.cutils import grid_subsampling
from utils.subsample import fps_sample

import torch.nn.functional as F
import numpy as np



def points_scaler(xyz, scale=1.):
    """
    :param xyz: [B, N, 3]
    :param scale: float, scale factor, by default 1.0, which means scale points xyz into [0, 1]
    :return: [B, N, 3]]
    """
    mi, ma = xyz.min(dim=1, keepdim=True)[0], xyz.max(dim=1, keepdim=True)[0]
    xyz = (xyz - mi) / (ma - mi + 1e-12)
    return xyz * scale



@dataclass
class CameraOptions(dict):
    # virtual camera numbers 
    n_cameras: int = 8
    # camera field of view in degree along y-axis. 
    cam_fovy: float = 60.0
    # camera field size, [width, height] 
    cam_field_size: list = field(default_factory=list)
    # use shelter points 
    use_shelter: bool = False
    cam_sampler: str = 'fps'
    cam_gen_method: str = 'cycle'

    @classmethod
    def default(cls, n_cameras=8):
        return CameraOptions(
            n_cameras=n_cameras,
            cam_fovy=120.0,
            cam_field_size=[512, 512],
            use_shelter=False,
            cam_sampler='fps',
            cam_gen_method='cycle'
        )

    def __str__(self):
        return f'''CameraOptions(
            n_cameras={self.n_cameras},
            cam_fovy={self.cam_fovy},
            cam_field_size={self.cam_field_size},
            use_shelter={self.use_shelter},
            cam_sampler={self.cam_sampler},
            cam_gen_method={self.cam_gen_method})'''


class CameraPoints(object):
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device

    def __set_attr__(self, key, value):
        assert not self.__is_attr_exists__(key)
        self.__dict__[key] = value

    def __update_attr__(self, key, value):
        self.__dict__[key] = value

    def __get_attr__(self, key):
        if not self.__is_attr_exists__(key):
            return None
        return self.__dict__[key]

    def __del_attr__(self, key):
        if not self.__is_attr_exists__(key):
            return
        self.__dict__.pop(key)

    def __is_attr_exists__(self, key):
        return key in self.__dict__.keys()

    def keys(self):
        return self.__dict__.keys()

    def to_cuda(self, device=None, non_blocking=True):
        keys = self.keys()
        for key in keys:
            item = self.__get_attr__(key)
            if isinstance(item, torch.Tensor):
                if device is not None:
                    item = item.to(device, non_blocking=non_blocking)
                else:
                    item = item.cuda(non_blocking=non_blocking)
            if isinstance(item, list):
                for i in range(len(item)):
                    if isinstance(item[i], torch.Tensor):
                        if device is not None:
                            item[i] = item[i].to(device, non_blocking=non_blocking)
                        else:
                            item[i] = item[i].cuda(non_blocking=non_blocking)
            self.__update_attr__(key, item)
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda'

    @property
    def layer_idx(self):
        return self.__get_attr__('layer_idx')

    @property
    def idx_ds(self):
        return self.__get_attr__('idx_ds')

    @property
    def idx_us(self):
        return self.__get_attr__('idx_us')

    @property
    def idx_group(self):
        return self.__get_attr__('idx_group')

    @property
    def pts_list(self):
        return self.__get_attr__('pts_list')

    @property
    def p(self):
        return self.__get_attr__('p')

    @property
    def f(self):
        return self.__get_attr__('f')

    @property
    def y(self):
        return self.__get_attr__('y')

    @property
    def f_cam(self):
        return self.__get_attr__('f_cam')

    @property
    def cameras(self):
        cameras = self.__get_attr__('orbit_cameras')
        return cameras

    @property
    def uv(self):
        item = self.__get_attr__('uv')
        return item

    @property
    def depths(self):
        item = self.__get_attr__('depths')
        return item

    @property
    def visible(self):
        item = self.__get_attr__('visible')
        return item

    @property
    def cam_intr(self):
        return self.__get_attr__('cam_intr')

    @property
    def cam_extr(self):
        return self.__get_attr__('cam_extr')


def generate_spiral_cameras(xyz, n_cameras=16, radius=2.0):

    center = torch.mean(xyz, dim=0)  

    phi = (1 + np.sqrt(5)) / 2  
    indices = torch.arange(n_cameras, device=xyz.device)
    
    y = 1 - (indices / float(n_cameras - 1)) * 2  
    radius_at_y = torch.sqrt(1 - y * y)  
    
    theta = 2 * np.pi * indices / phi 
    
    x = torch.cos(theta) * radius_at_y
    z = torch.sin(theta) * radius_at_y
    
    camera_positions = torch.stack([x, y, z], dim=1) * radius
    camera_positions = camera_positions + center 

    def build_view_matrix(position):
        forward = center - position 
        forward = forward / torch.norm(forward)
        up = torch.tensor([0,1,0], dtype=torch.float32, device=xyz.device)  
        right = torch.cross(up, forward)
        right = right / torch.norm(right)
        up = torch.cross(forward, right)
        return torch.stack([right, up, forward], dim=0)  

    rotation_matrices = torch.stack([build_view_matrix(pos) for pos in camera_positions])  # [n_cameras,3,3]
    
    points_centered = xyz.unsqueeze(0) - camera_positions.unsqueeze(1)  # [n_cameras,N,3]
    camera_coords = torch.bmm(points_centered, rotation_matrices.transpose(1, 2))  # [n_cameras,N,3]
    depths = camera_coords[..., 2]      
    
    return camera_positions, depths


def project_points_to_camera(xyz, camera_positions, image_width=512, image_height=512, focal_length=120):
    n_cameras = camera_positions.shape[0]
    center = torch.mean(xyz, dim=0)
    
    cx = image_width / 2
    cy = image_height / 2
    fx = fy = focal_length
    
    forward = center.unsqueeze(0) - camera_positions  # [n_cameras, 3]
    forward = F.normalize(forward, dim=1)  # 归一化
    
    global_up = torch.tensor([0, 1, 0], dtype=torch.float32, device=xyz.device).expand(n_cameras, -1)
    right = torch.cross(global_up, forward, dim=1)
    right = F.normalize(right, dim=1)
    up = torch.cross(forward, right, dim=1)
    
    rotation_matrices = torch.stack([right, up, forward], dim=2)
    points_expanded = xyz.unsqueeze(1)  # [N, 1, 3]
    
    cam_pos_expanded = camera_positions.unsqueeze(0)  # [1, n_cameras, 3]
    points_centered = points_expanded - cam_pos_expanded
    
    rotation_matrices_t = rotation_matrices.transpose(1, 2)
    points_cam = torch.einsum('nci,cij->ncj', points_centered, rotation_matrices_t)
    
    depths = points_cam[:, :, 2]
    
    z_mask = depths > 0
    safe_z = torch.where(z_mask, depths, torch.ones_like(depths))
    
    x_proj = (points_cam[:, :, 0] * fx / safe_z) + cx
    y_proj = (points_cam[:, :, 1] * fy / safe_z) + cy
    pixel_coords = torch.stack([x_proj, y_proj], dim=-1)
    
    valid_mask = (x_proj >= 0) & (x_proj < image_width) & \
                 (y_proj >= 0) & (y_proj < image_height) & \
                 z_mask
    

    return pixel_coords, depths, valid_mask


class CameraHelper:
    def __init__(self,
                 opt: CameraOptions = None,
                 batch_size: int = 8,
                 device: str = 'cuda',
                 **kwargs):
        if opt is None:
            opt = CameraOptions.default()
        self.opt = opt
        self.batch_size = batch_size
        self.device = device
        self.cam_points = CameraPoints(batch_size=self.batch_size, device=self.device)

    def init_points(self):
        self.cam_points = CameraPoints(batch_size=self.batch_size, device=self.device)

    def to(self, device):
        self.device = device
        return self

    @torch.no_grad()
    def projects(self, xyz, scale=1., cam_batch=128):
        """
        :param xyz: [N, 3]
        :param scale: xyz scale factor
        :param cam_batch: projection batch size of camera
        :return: [N, 3]
        """
        assert len(xyz.shape) == 2
        n_cameras = self.opt.n_cameras

        if scale > 0:
            # recommend to use scaler
            xyz_scaled = points_scaler(xyz.unsqueeze(0), scale=scale).squeeze(0)
        else:
            xyz_scaled = xyz

        camera_positions, _ = generate_spiral_cameras(xyz_scaled,n_cameras=n_cameras*2,radius=0.5)

        _, _, visible_camera = project_points_to_camera(xyz_scaled, camera_positions)



        self.cam_points.__update_attr__('visible', visible_camera.unsqueeze(0).float())
        self.cam_points.__update_attr__('f_cam', visible_camera.float())
        self.cam_points.__update_attr__('camera_positions', camera_positions.unsqueeze(0))


def calc_distance_scaler(full_p):
    ps, _ = fps_sample(full_p.unsqueeze(0), 2, random_start_point=True)
    ps = ps.squeeze(0)
    p0, p1 = ps[0], ps[1]
    scaler = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)
    return scaler


def make_cam_points(cam_points, ks, grid_size=None, n_samples=None, up_sample=True, alpha=0.) -> CameraPoints:
    assert (grid_size is not None and n_samples is not None) is False
    assert (grid_size is None and n_samples is None) is False
    n_layers = len(ks)
    full_p = cam_points.p
    full_visible = cam_points.visible.squeeze().byte()
    scaler = calc_distance_scaler(full_p)

    full_p = full_p.contiguous()
    full_visible = full_visible.contiguous()
    visible = full_visible
    p = full_p

    idx_ds = []
    idx_us = []
    idx_group = []
    idx_group_small = []
    idx_gs_group = []
    
    for i in range(n_layers):
        if i > 0:
            if grid_size is not None:
                gsize = grid_size[i-1]
                if p.is_cuda:
                    ds_idx = grid_subsampling(p.detach().cpu(), gsize)
                else:
                    ds_idx = grid_subsampling(p, gsize)
            else:
                _, ds_idx = fps_sample(p.unsqueeze(0), n_samples[i-1])
                ds_idx = ds_idx.squeeze(0)
            

            ds_idx = torch.sort(ds_idx)[0]
            p = p[ds_idx]
            visible = visible[ds_idx]
            idx_ds.append(ds_idx)

        # knn group by kdtree
        k = ks[i]

        kdt = KDTree(p.numpy(), visible.numpy())
        _, idx = kdt.query(p.numpy(), visible.numpy(), k=k, alpha=alpha, scaler=scaler)
        idx_small = idx[:,:k//4]
        idx_group.append(torch.from_numpy(idx).long())
        idx_group_small.append(torch.from_numpy(idx_small).long())
        
        if i > 0 and up_sample:
            _, us_idx = kdt.query(full_p.numpy(), full_visible.numpy(), k=1, alpha=alpha, scaler=scaler)
            idx_us.append(torch.from_numpy(us_idx).long())

    cam_points.__update_attr__('idx_ds', idx_ds)
    cam_points.__update_attr__('idx_us', idx_us)
    cam_points.__update_attr__('idx_group', idx_group)
    cam_points.__update_attr__('idx_group_small', idx_group_small)
    cam_points.__update_attr__('idx_gs_group', idx_gs_group)
    return cam_points


def merge_cam_points(cam_points_list, up_sample=True) -> CameraPoints:
    assert len(cam_points_list) > 0
    new_cam_points = CameraPoints(batch_size=len(cam_points_list),
                                  device=cam_points_list[0].device)

    p_all = []
    f_cam_all = []
    f_all = []
    camera_positions_all = [cam_points_list[0].camera_positions]
    rgb_all = []
    y_all = []
    idx_ds_all = []
    idx_us_all = []
    idx_group_all = []
    idx_group_small_all = []
    pts_all = []
    n_layers = len(cam_points_list[0].idx_group)
    pts_per_layer = [0] * n_layers


    for i in range(len(cam_points_list)):
        cam_points = cam_points_list[i]
        p_all.append(cam_points.p)
        f_cam_all.append(cam_points.f_cam)
        f_all.append(cam_points.f)
        rgb = cam_points.__get_attr__('rgb')
        if rgb is not None:
            rgb_all.append(rgb)
        y_all.append(cam_points.y)

        idx_ds = cam_points.idx_ds
        idx_us = cam_points.idx_us
        idx_group = cam_points.idx_group
        idx_group_small = cam_points.idx_group_small
        pts = []

        for layer_idx in range(n_layers):
            if layer_idx < len(idx_ds):
                idx_ds[layer_idx].add_(pts_per_layer[layer_idx])
                if up_sample:
                    idx_us[layer_idx].add_(pts_per_layer[layer_idx + 1])
            idx_group[layer_idx].add_(pts_per_layer[layer_idx])
            idx_group_small[layer_idx].add_(pts_per_layer[layer_idx])


            pts.append(idx_group[layer_idx].shape[0])

        idx_ds_all.append(idx_ds)
        idx_us_all.append(idx_us)
        idx_group_all.append(idx_group)
        idx_group_small_all.append(idx_group_small)
        pts_all.append(pts)
        pts_per_layer = [pt + idx.shape[0] for (pt, idx) in zip(pts_per_layer, idx_group)]

    p = torch.cat(p_all, dim=0)
    new_cam_points.__update_attr__('p', p)

    f_cam = torch.cat(f_cam_all, dim=0)
    new_cam_points.__update_attr__('f_cam', f_cam)

    camera_positions = torch.cat(camera_positions_all, dim=0)
    new_cam_points.__update_attr__('camera_positions', camera_positions)

    f = torch.cat(f_all, dim=0)
    new_cam_points.__update_attr__('f', f)

    if len(rgb_all) > 0:
        rgb = torch.cat(rgb_all, dim=0)
        new_cam_points.__update_attr__('rgb', rgb)

    y = torch.cat(y_all, dim=0)
    new_cam_points.__update_attr__('y', y)

    # layer_idx is [1, 2, 3] when CamPoint layers = 4
    idx_ds = [torch.cat(idx, dim=0) for idx in zip(*idx_ds_all)]
    new_cam_points.__update_attr__('idx_ds', idx_ds)

    # layer_idx is [2, 1, 0] when CamPoint layers = 4
    idx_us = [torch.cat(idx, dim=0) for idx in zip(*idx_us_all)]
    new_cam_points.__update_attr__('idx_us', idx_us)

    # layer_idx is [0, 1, 2, 3] when CamPoint layers = 4
    idx_group = [torch.cat(idx, dim=0) for idx in zip(*idx_group_all)]
    new_cam_points.__update_attr__('idx_group', idx_group)


    idx_group_small = [torch.cat(idx, dim=0) for idx in zip(*idx_group_small_all)]
    new_cam_points.__update_attr__('idx_group_small', idx_group_small)

    # batch_size * layer_idx is [0, 1, 2, 3] when CamPoint layers = 4
    pts_list = torch.tensor(pts_all, dtype=torch.int64)
    pts_list = pts_list.view(-1, n_layers).transpose(0, 1).contiguous()
    new_cam_points.__update_attr__('pts_list', pts_list)
    return new_cam_points


