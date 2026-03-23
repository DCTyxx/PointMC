import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint
from backbone.camera_outside import CameraOptions
from backbone.mamba_ssm.custom.order import Order
from backbone.mamba_ssm.models import MambaConfig, MixerModel
from utils.cutils import knn_edge_maxpooling
from pointnet2_ops import pointnet2_utils


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class LocalFusionModule(nn.Module):
    
    def __init__(self,
                 layer_index=0,
                 in_channels=4,
                 channel_list=[64, 128, 256, 512],
                 bn_momentum=0.02,
                 use_cp=False,
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

    def forward(self, p, f, group_idx):

        assert len(f.shape) == 2

        p_group = p[group_idx] - p.unsqueeze(1)

        if self.is_head:
            f_group = f[group_idx]
            f_group = torch.cat([p_group, f_group], dim=-1).view(-1, 3 + self.in_channels)
        else:
            f_group = p_group.view(-1, 3)

        N, K = group_idx.shape 

        embed_fn = lambda x: self.embed(x).view(N, K, -1).max(dim=1)[0]


        f_group = embed_fn(f_group) if not self.use_cp else checkpoint(embed_fn, f_group, use_reentrant=False)
        f_group = self.proj(f_group)
        f_group = self.bn(f_group) 

        f = f_group if self.is_head else f_group + f 
        return f


class LocalGrouper(nn.Module):
    def __init__(self, in_channels, out_channels, groups, kneighbors, use_xyz=True, normalize="anchor",batch_size=1, activation='relu', **kwargs):

        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.in_channels = in_channels
        self.out_channels = out_channels
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, in_channels + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, in_channels + add_channel]))

        self.transfer = ConvBNReLU1D(in_channels*2, out_channels, bias=True, activation=activation)

        self.batch_size = batch_size

    def forward(self, xyz, points, feature_camera):

        B, N, C = xyz.shape
        S = self.groups 

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]
        new_camera = index_points(feature_camera, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]	
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta # ([32, 512, 32, 128]

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1) # ([32, 512, 32, 128+128])
        b, n, s, d = new_points.size()  # torch.Size([32, 512, 32, 6])
        new_points = new_points.permute(0, 1, 3, 2)
        new_points = new_points.reshape(-1, d, s)

        new_points = self.transfer(new_points)
        batch_size, _, _ = new_points.size()
        new_points = F.adaptive_max_pool1d(new_points, 1).view(batch_size, -1)
        new_points = new_points.reshape(b, n, -1)
        return new_xyz, new_points, new_camera


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)



class LocalAggregation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bn_momentum: float = 0.1,
                 init_weight: float = 0.,
                 **kwargs
                 ):
        super().__init__()

        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init_weight)

    def forward(self, f, group_idx):
        """
        :param f: [B, N, channels]
        :param group_idx: [B, N, K]
        :return: [B, N, channels]
        """
        assert len(f.shape) == 3
        assert len(group_idx.shape) == 3
        B, N, C = f.shape
        f = self.proj(f)
        f = knn_edge_maxpooling(f, group_idx, self.training)
        f = self.bn(f.view(B * N, -1)).view(B, N, -1)
        return f


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 output_channels,
                 bn_momentum,
                 init_weight=0.,
                 **kwargs
                 ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, output_channels, bias=False),
            nn.BatchNorm1d(output_channels, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init_weight)

    def forward(self, f):
        """
        :param f: [B, N, in_channels]
        :return: [B, N, output_channels]
        """
        assert len(f.shape) == 3
        B, N, C = f.shape
        f = self.mlp(f.view(B * N, -1)).view(B, N, -1)
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 channels, 
                 res_blocks, 
                 mlp_ratio, 
                 bn_momentum, 
                 drop_path, 
                 **kwargs
                 ):
        super().__init__()
        self.res_blocks = res_blocks 
        hidden_channels = round(channels * mlp_ratio) 
        self.mlp = MLP(channels, hidden_channels, channels, bn_momentum, 0.2) 

        self.blocks = nn.ModuleList([
            LocalAggregation(
                channels,
                channels,
                bn_momentum,
            )
            for _ in range(res_blocks)
        ])

        self.mlps = nn.ModuleList([
            MLP(
                channels,
                hidden_channels,
                channels,
                bn_momentum,
            )
            for _ in range(res_blocks // 2)
        ])

        self.drop_paths = nn.ModuleList([
            DropPath(d) for d in drop_path
        ])

        self.use_drop = [d > 0. for d in drop_path]


    def drop_path(self, f, block_index, pts):
        if not self.use_drop[block_index] or not self.training:
            return f
        return torch.cat([self.drop_paths[block_index](x) for x in torch.split(f, pts, dim=1)], dim=1)


    def forward(self, f, group_idx, pts):

        assert len(f.shape) == 3
        assert len(group_idx.shape) == 3
        assert pts is not None

        f = f + self.drop_path(self.mlp(f), 0, pts)

        for i in range(self.res_blocks):
            f = f + self.drop_path(self.blocks[i](f, group_idx), i, pts) 
            if i % 2 == 1:
                f = f + self.drop_path(self.mlps[i // 2](f), i, pts) 
        return f


def make_hybrid_idx(n_layer, hybrid_type, ratio) -> list:
    assert hybrid_type in ['pre', 'post']
    n_hybrid = int(n_layer * ratio)
    if n_hybrid == 0 and ratio > 0:
        n_hybrid = 1

    attn_idx_left, attn_idx_right = 0, 0
    if hybrid_type == 'post':
        attn_idx_left, attn_idx_right = n_layer - n_hybrid, n_layer
    elif hybrid_type == 'pre':
        attn_idx_left, attn_idx_right = 0, n_hybrid
    return [i for i in range(attn_idx_left, attn_idx_right)]


def self_adapt_heads(d_model: int) -> int:
    min_heads = 8
    min_head_dims = 64
    num_heads = math.ceil(d_model / min_head_dims)
    num_heads |= num_heads >> 1
    num_heads |= num_heads >> 2
    num_heads |= num_heads >> 4
    num_heads |= num_heads >> 8
    num_heads |= num_heads >> 16
    num_heads += 1
    return max(min_heads, num_heads)


def make_scan_perm(scan_method, base):
    N = base.shape[0]
    if scan_method == 'random' or scan_method == 'random2':
        perm = torch.randperm(N)
    elif scan_method == 'flip':
        perm = torch.arange(N - 1, -1, step=-1, dtype=torch.long, device=base.device)
    else:
        perm = None
    return perm


def create_mixer(
        config: MambaConfig,
        d_model: int,
        hybrid_args: dict,
        scan_twice: bool,
):
    config.d_model = d_model
    n_layer = config.n_layer
    assert n_layer > 0
    ssm_cfg = config.ssm_cfg
    attn_cfg = config.attn_cfg
    d_intermediate = config.d_intermediate
    rms_norm = config.rms_norm
    residual_in_fp32 = config.residual_in_fp32

    hybrid = hybrid_args.get('hybrid', False)
    hybrid_type = hybrid_args.get('type')
    hybrid_ratio = hybrid_args.get('ratio', 0.)

    if attn_cfg.get('self_adapt_heads', False):
        num_heads = self_adapt_heads(d_model)
    else:
        num_heads = attn_cfg.get('num_heads', self_adapt_heads(d_model))

    expand = ssm_cfg.get('expand', 2)
    ssm_cfg.d_state = min(d_model * expand, 1024)
    ssm_cfg.headdim = min(d_model // num_heads, 192)
    attn_layer_idx = [] if not hybrid else make_hybrid_idx(n_layer, hybrid_type, hybrid_ratio)

    min_heads = max(d_model // ssm_cfg.headdim, 8)
    if num_heads < min_heads:
        logging.warning(f'num heads {num_heads} < min heads {min_heads}, '
                        f'will replace with {min_heads}, d_model={d_model}')
        num_heads = min_heads
    attn_cfg.num_heads = num_heads
    attn_cfg = attn_cfg if hybrid else None

    return MixerModel(
        d_model=d_model,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        ssm_cfg=ssm_cfg,
        attn_layer_idx=attn_layer_idx,
        attn_cfg=attn_cfg,
        rms_norm=rms_norm,
        residual_in_fp32=residual_in_fp32,
        scan_twice=scan_twice,
    )


class GlobalPerceptionModule(nn.Module):

    def __init__(self,
                 layer_index: int,
                 channels: int,
                 mamba_config: MambaConfig,
                 hybrid_args: dict,
                 bn_momentum: float,
                 cam_opts: CameraOptions,
                 batch_size: int = 32,
                 **kwargs,
                 ):
        super().__init__()
        self.layer_index = layer_index
        self.mamba_config = mamba_config
        init_std = math.sqrt(1 / channels / 3)
        cpe = torch.empty([cam_opts.n_cameras * 2, 16], dtype=torch.float32)
        nn.init.trunc_normal_(cpe, mean=0, std=init_std)
        self.cpe = nn.Parameter(cpe)


        self.batch_size = batch_size


        self.mlp = MLP(16, 32, channels, bn_momentum)


        scan_method = mamba_config.scan_method
        self.scan_method = scan_method
        assert scan_method in ['', 'random', 'random2', 'flip']
        if scan_method == 'random' or scan_method == '':
            scan_twice = False
        else:
            scan_twice = True
        if mamba_config.n_layer <= 0:
            mamba_config.n_layer = 1
        self.mixer = create_mixer(mamba_config, channels, hybrid_args, scan_twice)
        self.bn = nn.BatchNorm1d(channels, momentum=bn_momentum)

    def forward(self, f, cvf):
        """
        :param f: [N, channels] 特征
        :param cvf: [N, n_cameras] 相机看到的内容
        :return: [N, channels] 特征
        """
        assert len(f.shape) == 2

        
        cpe = cvf @ self.cpe / cvf.shape[-1]
        cpe = self.mlp(cpe.unsqueeze(0))

        
        perm = make_scan_perm(self.scan_method, f)
        if perm is not None:
            order = Order(perm.unsqueeze(0))
        else:
            order = None

        f = f.unsqueeze(0)
        B, N, C = f.shape
        
        f = f + self.mixer(input_ids=f, order=order, pos_embed=cpe)
        f = self.bn(f.view(B * N, -1))
        return f