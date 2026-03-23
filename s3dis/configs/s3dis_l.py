import __init__

import torch

from backbone.camera_outside import CameraOptions
from backbone.mamba_ssm.models import MambaConfig
from utils.config import EasyConfig


class S3disConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'S3disConfig'
        self.k = [24, 24, 24, 24]
        self.grid_size = [0.08, 0.16, 0.32]
        self.voxel_max = 30000
        self.cam_opts = CameraOptions.default(n_cameras=128)
        self.alpha = 0.1
        if self.alpha == 0:
            self.cam_opts.n_cameras = 1  # whatever


class S3disWarmupConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'S3disWarmupConfig'
        self.k = [24, 24, 24, 24]
        self.grid_size = [0.08, 0.16, 0.32]
        self.voxel_max = 30000
        self.cam_opts = CameraOptions.default(n_cameras=128)
        self.alpha = 0.1
        if self.alpha == 0:
            self.cam_opts.n_cameras = 1  # whatever


class ModelConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelConfig'
        self.train_cfg = S3disConfig()
        self.warmup_cfg = S3disWarmupConfig()
        self.num_classes = 13
        self.bn_momentum = 0.02
        drop_path = 0.1
        backbone_cfg = EasyConfig()
        backbone_cfg.name = 'CamPointModelConfig'
        backbone_cfg.in_channels = 4
        backbone_cfg.channel_list = [128, 256, 384, 512]
        backbone_cfg.head_channels = 256
        backbone_cfg.mamba_blocks = [1, 1, 1, 1]
        backbone_cfg.res_blocks = [4, 4, 8, 4]
        backbone_cfg.mlp_ratio = 2.
        backbone_cfg.bn_momentum = self.bn_momentum
        drop_rates = torch.linspace(0., drop_path, sum(backbone_cfg.res_blocks)).split(backbone_cfg.res_blocks)
        backbone_cfg.drop_paths = [d.tolist() for d in drop_rates]
        backbone_cfg.head_drops = torch.linspace(0., 0.15, len(backbone_cfg.res_blocks)).tolist()
        backbone_cfg.mamba_config = MambaConfig.default()
        backbone_cfg.hybrid_args = {'hybrid': False}  # whether hybrid mha, {'hybrid': True, 'type': 'post', 'ratio': 0.5}
        backbone_cfg.cam_opts = self.train_cfg.cam_opts
        backbone_cfg.diff_factor = 40.
        backbone_cfg.diff_std = [1.6, 3.2, 6.4, 12.8]
        # backbone_cfg.diff_std = None
        self.backbone_cfg = backbone_cfg
