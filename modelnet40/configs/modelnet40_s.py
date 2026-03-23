import __init__

import torch

from backbone.camera_outside import CameraOptions
from backbone.mamba_ssm.models import MambaConfig
from utils.config import EasyConfig


class ModelNet40Config(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelNet40Config'
        self.k = [32, 32, 32]
        self.n_samples = [1024, 256, 64]
        self.num_points = 1024
        self.sequence_length = [1024, 1024, 256]

        self.weight = [889/625, 889/106, 889/515, 889/173, 889/572, 889/335, 889/64, 889/197, 1, 889/167, 889/79, 889/137, 889/200, 889/109, 889/200, 889/149, 889/171, 889/155, 889/145, 889/124, 889/149, 889/284, 889/465, 889/200, 889/88, 889/231, 889/239, 889/104, 889/115, 889/128, 889/680, 889/124, 889/90, 889/392, 889/163, 889/344, 889/267, 889/475, 889/87, 889/103]
        self.n_cameras = 8
        self.cam_opts = CameraOptions.default(n_cameras=self.n_cameras)
        self.alpha = 0.1
        if self.alpha == 0:
            self.cam_opts.n_cameras = 1


class ModelConfig(EasyConfig):
    def __init__(self):
        super().__init__()
        self.name = 'ModelConfig'
        self.train_cfg = ModelNet40Config()
        self.num_classes = 40
        self.bn_momentum = 0.1
        drop_path = 0.15
        backbone_cfg = EasyConfig()
        backbone_cfg.name = 'CamPointModelConfig'
        backbone_cfg.in_channels = 4
        backbone_cfg.channel_list = [96, 192, 384]
        backbone_cfg.head_channels = 2048
        backbone_cfg.mamba_blocks = [1, 1, 1]
        backbone_cfg.res_blocks = [4, 4, 4]
        backbone_cfg.mlp_ratio = 2.
        backbone_cfg.bn_momentum = self.bn_momentum
        drop_rates = torch.linspace(0., drop_path, sum(backbone_cfg.res_blocks)).split(backbone_cfg.res_blocks)
        backbone_cfg.drop_paths = [d.tolist() for d in drop_rates]
        backbone_cfg.mamba_config = MambaConfig.default()
        backbone_cfg.hybrid_args = {'hybrid': False}  
        backbone_cfg.cam_opts = self.train_cfg.cam_opts
        backbone_cfg.diff_factor = 40.
        backbone_cfg.diff_std = [2.8, 5.3, 10]
        backbone_cfg.weight = self.train_cfg.weight
        backbone_cfg.n_cameras = self.train_cfg.n_cameras
        backbone_cfg.sequence_length = self.train_cfg.sequence_length
        self.backbone_cfg = backbone_cfg
