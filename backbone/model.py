import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from timm.models.layers import DropPath
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint

from backbone.camera_outside import CameraPoints
from backbone.mamba_ssm.models import MambaConfig
from backbone.modules import InvResMLP, GlobalPerceptionModule, LocalAggregation, get_activation


def get_biase(p_group, biase_fn='n2f'):
    distance_large = torch.norm(p_group, dim=-1) # [N, K]
    if biase_fn == 'n2f':
        biase = distance_large.unsqueeze(-1) # [N, K, 1]
    elif biase_fn == 'f2n':
        max_distance = torch.max(distance_large, dim=-1)[0] # [N]
        biase = max_distance.unsqueeze(-1) / distance_large # [N, K, 1]
        biase = torch.clamp(biase, max=32)
        biase = biase.unsqueeze(-1)
    else:
        biase = torch.ones_like(distance_large)
    return biase.reshape(-1, 1)


class LocalFusionModule(nn.Module):
    def __init__(self,
                 layer_index=0,
                 in_channels=4,
                 channel_list=[64, 128, 256, 512],
                 bn_momentum=0.02,
                 use_cp=False,
                 act='gelu',
                 ):
        super().__init__()
        self.use_cp = use_cp
        self.layer_index = layer_index
        is_head = self.layer_index == 0
        self.is_head = is_head
        self.in_channels = in_channels if is_head else channel_list[layer_index - 1]
        self.out_channels = channel_list[layer_index]

        embed_in_channels = 3 + self.in_channels if is_head else 3
        embed_hidden_channels = channel_list[0] // 2 if is_head else channel_list[0] // 4
        embed_out_channels = self.out_channels if is_head else channel_list[0] // 2

        self.embed_act = get_activation(act)
        
        self.embed = nn.Sequential(
            nn.Linear(embed_in_channels, embed_hidden_channels // 2, bias=False),
            nn.BatchNorm1d(embed_hidden_channels // 2, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(embed_hidden_channels // 2, embed_hidden_channels, bias=False),
            nn.BatchNorm1d(embed_hidden_channels, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(embed_hidden_channels, embed_out_channels, bias=False),
        )

        self.proj = nn.Identity() if is_head else nn.Linear(embed_out_channels, self.out_channels, bias=False)

        self.bn = nn.BatchNorm1d(self.out_channels, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, 0.8 if is_head else 0.2)

        self.embed_small = nn.Sequential(
            nn.Linear(embed_in_channels, embed_hidden_channels // 2, bias=False),
            nn.BatchNorm1d(embed_hidden_channels // 2, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(embed_hidden_channels // 2, embed_out_channels, bias=False),
        )

        self.proj_small = nn.Identity() if is_head else nn.Linear(embed_out_channels, self.out_channels, bias=False) 
        
        self.bn_small = nn.BatchNorm1d(self.out_channels, momentum=bn_momentum)
        nn.init.constant_(self.bn_small.weight, 0.2)


        self.alpha = nn.Parameter(torch.ones(1),requires_grad=True)*0.05
        
        self.beta = nn.Parameter(torch.ones(1),requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1),requires_grad=True)

        
    def forward(self, p, f, group_idx, group_idx_small=None, biase_fn=None, alpha=1):
        assert len(f.shape) == 2

        p_group = p[group_idx] - p.unsqueeze(1)

        biase_large = get_biase(p_group, biase_fn)

        if self.is_head:
            f_group = f[group_idx]
            f_group = torch.cat([p_group, f_group], dim=-1).view(-1, 3 + self.in_channels)

            f_group = torch.mul(f_group, biase_large)
        else:
            f_group = p_group.view(-1, 3)
            f_group = torch.mul(f_group, biase_large)

        N, K = group_idx.shape # 索引的分组的维度

        def embed_fn(x):
            embedded = self.embed(x)
            reshaped = embedded.view(N, K, -1)
            pooled, _ = reshaped.max(dim=1)
            return pooled

        f_group = embed_fn(f_group) if not self.use_cp else checkpoint(embed_fn, f_group, use_reentrant=False)
        f_group = self.proj(f_group) 

        f_group = self.bn(f_group) 


        if group_idx_small is None:
            group_idx_small = group_idx[:, :group_idx.shape[1]//4]
        else:
            group_idx_small = group_idx_small

        group_small_shape = group_idx_small.shape
        p_group_small = p[group_idx_small] - p.unsqueeze(1)

        biase_small = get_biase(p_group_small, biase_fn)
        
        if self.is_head:
            f_group_small = f[group_idx_small]
            f_group_small = torch.cat([p_group_small, f_group_small], dim=-1).view(-1, 3 + self.in_channels)
            f_group_small = torch.mul(f_group_small, biase_small)
        else:
            f_group_small = p_group_small.view(-1, 3)
            f_group_small = torch.mul(f_group_small, biase_small)

        def embed_small_fn(x):
            embedded = self.embed_small(x)
            reshaped = embedded.view(N, group_small_shape[1], -1)
            pooled, _ = reshaped.max(dim=1)
            return pooled
        
        f_group_small = embed_small_fn(f_group_small) if not self.use_cp else checkpoint(embed_small_fn, f_group_small, use_reentrant=False)
        f_group_small = self.proj_small(f_group_small)

        
        f_group_small = self.bn_small(f_group_small)

        
        f = f_group + alpha * f_group_small if self.is_head else f + f_group + alpha * f_group_small 
        return f

class PointMC(nn.Module):
    def __init__(self,
                 layer_index=0,
                 in_channels=4,
                 channel_list=[64, 128, 256, 512],
                 head_channels=256,
                 mamba_blocks=[1, 1, 1, 1],
                 res_blocks=[4, 4, 8, 4],
                 mlp_ratio=2.,
                 bn_momentum=0.02,
                 drop_paths=None,
                 head_drops=None,
                 mamba_config=MambaConfig().default(),
                 hybrid_args={'hybrid': False},
                 cam_opts=None,
                 use_cp=False,
                 diff_factor=40.,
                 diff_std=None,
                 task_type='segsem',
                 biase_fn='origin',
                 alpha=1,
                 batch_size=32,
                 **kwargs
                 ):
        super().__init__()

        self.task_type = task_type.lower()
        assert self.task_type in ['segsem', 'segpart', 'cls']
        assert cam_opts is not None
        self.use_cp = use_cp
        self.layer_index = layer_index
        is_head = self.layer_index == 0
        self.is_head = is_head
        is_tail = self.layer_index == len(channel_list) - 1
        self.is_tail = is_tail
        self.in_channels = in_channels if is_head else channel_list[layer_index - 1]
        self.channel_list = channel_list
        self.out_channels = channel_list[layer_index]
        self.head_channels = head_channels
        self.diff_factor = diff_factor
        self.alpha = alpha
        self.biase_fn = biase_fn
        self.batch_size = batch_size

        if not is_head:
            self.skip_proj = nn.Sequential(
                nn.Linear(self.in_channels, self.out_channels, bias=False),
                nn.BatchNorm1d(self.out_channels, momentum=bn_momentum)
            )
            self.la = LocalAggregation(self.in_channels, self.out_channels, bn_momentum, 0.3)
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.local_module = LocalFusionModule(
            layer_index=layer_index,
            in_channels=in_channels,
            channel_list=channel_list,
            bn_momentum=bn_momentum,
            use_cp=use_cp,
        )

        self.res_mlp = InvResMLP(
            channels=self.out_channels,
            res_blocks=res_blocks[layer_index],
            mlp_ratio=mlp_ratio,
            bn_momentum=bn_momentum,
            drop_path=drop_paths[layer_index],
        )

        mamba_config.n_layer = mamba_blocks[layer_index]
        self.global_module = GlobalPerceptionModule(
            layer_index=layer_index,
            channels=self.out_channels,
            mamba_config=mamba_config,
            hybrid_args=hybrid_args,
            bn_momentum=bn_momentum,
            cam_opts=cam_opts,
        )

        if self.task_type != 'cls':
            self.post_proj = nn.Sequential(
                nn.BatchNorm1d(self.out_channels, momentum=bn_momentum),
                nn.Linear(self.out_channels, head_channels, bias=False),
            )
            nn.init.constant_(self.post_proj[0].weight, (channel_list[0] / self.out_channels) ** 0.5)
            self.head_drop = DropPath(head_drops[layer_index])

        if diff_std is not None:
            self.use_diff = True
            self.diff_std = 1 / diff_std[layer_index]
            self.diff_head = nn.Sequential(
                nn.Linear(self.out_channels, 32, bias=False),
                nn.BatchNorm1d(32, momentum=bn_momentum),
                nn.GELU(),
                nn.Linear(32, 3, bias=False),
            )
        else:
            self.use_diff = False

        if not is_tail:
            self.sub = PointMC(
                layer_index=layer_index + 1,
                in_channels=in_channels,
                channel_list=channel_list,
                head_channels=head_channels,
                mamba_blocks=mamba_blocks,
                res_blocks=res_blocks,
                mlp_ratio=mlp_ratio,
                bn_momentum=bn_momentum,
                drop_paths=drop_paths,
                head_drops=head_drops,
                mamba_config=mamba_config,
                hybrid_args=hybrid_args,
                cam_opts=cam_opts,
                use_cp=use_cp,
                diff_factor=diff_factor,
                diff_std=diff_std,
                task_type=task_type,
            )

    def diff_loss(self, p, f, group_idx, d_sub):
        # use the grouped features to predict 3d position
        if not self.training:
            return None
        N, K = group_idx.shape
        rand_group = torch.randint(0, K, (N, 1), device=p.device)
        rand_group_idx = torch.gather(group_idx, 1, rand_group).squeeze(1)
        rand_p_group = p[rand_group_idx] - p
        rand_p_group.mul_(self.diff_std)
        rand_f_group = f[rand_group_idx] - f
        rand_f_group = self.diff_head(rand_f_group)
        diff = nn.functional.mse_loss(rand_f_group, rand_p_group)
        if d_sub is not None:
            diff = diff + d_sub
        return diff

    def forward(self, p, f, f_cam, cam_points: CameraPoints):
        
        if self.is_head:
            p = p.mul_(self.diff_factor)

        if not self.is_head:
            idx = cam_points.idx_ds[self.layer_index - 1]
            p = p[idx]
            f_cam = f_cam[idx]
            pre_group_idx = cam_points.idx_group[self.layer_index - 1]
            f = self.skip_proj(f)[idx] + self.la(f.unsqueeze(0), pre_group_idx.unsqueeze(0)).squeeze(0)[idx]

        group_idx = cam_points.idx_group[self.layer_index] 
        
        group_idx_small = cam_points.idx_group_small[self.layer_index]
        
        if group_idx_small is None:
            f_local = self.local_module(p, f, group_idx, biase_fn=self.biase_fn, alpha=self.alpha)
        else:
            f_local = self.local_module(p, f, group_idx, group_idx_small, biase_fn=self.biase_fn, alpha=self.alpha)


        # local aggregation and propagation 
        pts = cam_points.pts_list[self.layer_index].tolist()

        f_local = self.res_mlp(f_local.unsqueeze(0), group_idx.unsqueeze(0), pts) if not self.use_cp \
            else checkpoint(self.res_mlp.forward, f_local.unsqueeze(0), group_idx.unsqueeze(0), pts, use_reentrant=False)
        f_local = f_local.squeeze(0)

        f_global = self.global_module(f_local, f_cam)
        f = f_global + f_local

        f_sub, diff = self.sub(p, f, f_cam, cam_points) if not self.is_tail else (None, None)

        f_out = f
        if self.task_type != 'cls':
            f_out = self.post_proj(f_out)
            if not self.is_head:
                us_idx = cam_points.idx_us[self.layer_index - 1]
                f_out = f_out[us_idx]
            f_out = self.head_drop(f_out)
            f_out = f_sub + f_out if f_sub is not None else f_out
        else:
            f_out = f_sub if f_sub is not None else f_out

        diff = self.diff_loss(p, f, group_idx, diff)
        return f_out, diff


class SegSemHead(nn.Module):
    def __init__(self,
                 backbone: PointMC,
                 num_classes=13,
                 bn_momentum=0.02,
                 **kwargs
                 ):
        super().__init__()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.BatchNorm1d(backbone.head_channels, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(backbone.head_channels, num_classes),
        )

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, cam_points: CameraPoints):
        p = cam_points.p
        f = cam_points.f
        f_cam = cam_points.f_cam

        f, diff = self.backbone(p, f, f_cam, cam_points)
        if self.training:
            return self.head(f), diff
        return self.head(f)


class SegPartHead(nn.Module):
    def __init__(self,
                 backbone: PointMC,
                 num_classes=50,
                 shape_classes=16,
                 bn_momentum=0.1,
                 **kwargs
                 ):
        super().__init__()
        self.backbone = backbone
        self.shape_classes = shape_classes

        self.proj = nn.Sequential(
            nn.Linear(shape_classes, 64, bias=False),
            nn.BatchNorm1d(64, momentum=bn_momentum),
            nn.Linear(64, backbone.head_channels, bias=False)
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(backbone.head_channels, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(backbone.head_channels, num_classes),
        )

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, cam_points: CameraPoints, shape):
        p = cam_points.p
        f = cam_points.f
        f_cam = cam_points.f_cam

        f, diff = self.backbone(p, f, f_cam, cam_points)
        shape = nn.functional.one_hot(shape, self.shape_classes).float()
        shape = self.proj(shape)
        shape = repeat(shape, 'b c -> b n c', n=f.shape[0]//cam_points.batch_size)
        shape = rearrange(shape, 'b n c -> (b n) c')
        f = f + shape
        if self.training:
            return self.head(f), diff
        return self.head(f)


class ClsHead(nn.Module):
    def __init__(self,
                 backbone: PointMC,
                 num_classes=13,
                 bn_momentum=0.1,
                 cls_type='mean_max',
                 **kwargs
                 ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.cls_type = cls_type
        assert cls_type in ['mean_max', 'max', 'mean']

        self.proj = nn.Sequential(
            nn.BatchNorm1d(backbone.channel_list[-1], momentum=bn_momentum),
            nn.Linear(backbone.channel_list[-1], backbone.head_channels),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(backbone.head_channels, 512, bias=False),
            nn.BatchNorm1d(512, momentum=bn_momentum),
            nn.GELU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256, momentum=bn_momentum),
            nn.GELU(),
            nn.Dropout(.5),
            nn.Linear(256, num_classes)
        )

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __get_cls(self, f):
        if self.cls_type == 'mean_max':
            return f.max(1)[0] + f.mean(1)[0]
        elif self.cls_type == 'max':
            return f.max(1)[0]
        elif self.cls_type == 'mean':
            return f.mean(1)[0]
        else:
            return f.max(1)[0] + f.mean(1)[0]

    def forward(self, cam_points: CameraPoints):
        p = cam_points.p 
        f = cam_points.f 
        f_cam = cam_points.f_cam

        f, diff = self.backbone(p, f, f_cam, cam_points)
        f = self.proj(f)
        f = f.view(cam_points.batch_size, -1, self.backbone.head_channels)
        f = self.__get_cls(f).squeeze(1)
        f = self.head(f)
        f = f.view(-1, self.num_classes)
        if self.training:
            return f, diff
        return f
