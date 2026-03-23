import __init__

import math
import random
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from backbone.camera_outside import CameraOptions, CameraHelper, make_cam_points, merge_cam_points, fps_sample



class ScanObjectNN(Dataset):
    def __init__(self,
                 dataset_dir: Path,
                 train=True,
                 warmup=False,
                 num_points=1024,
                 k=[24, 24, 24],
                 n_samples=[1024, 256, 64],
                 alpha=0.,
                 batch_size=32,
                 data_name = 'PB_T50_RS',
                 cam_opts: CameraOptions = CameraOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)

        if data_name == 'PB_T50_RS':
            self.data_path = dataset_dir.joinpath(
                f"main_split/{'training' if train else 'test'}_objectdataset_augmentedrot_scale75.h5")
        elif data_name == 'OBJ_ONLY':
            self.data_path = dataset_dir.joinpath(
                f"main_split_nobg/{'training' if train else 'test'}_objectdataset.h5")
        elif data_name == 'OBJ_BG':
            self.data_path = dataset_dir.joinpath(
                f"main_split/{'training' if train else 'test'}_objectdataset.h5")
        else:
            raise ValueError(f"Invalid data name: {data_name}")
        

        self.train = train
        self.warmup = warmup
        self.num_points = num_points
        self.k = k
        self.n_samples = n_samples
        self.alpha = alpha
        self.batch_size = batch_size
        self.cam_opts = cam_opts


        f = h5py.File(self.data_path, 'r')
        self.datas = torch.from_numpy(f['data'][:]).float()
        self.label = torch.from_numpy(f['label'][:]).long()
        f.close()

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, idx):
        xyz = self.datas[idx]
        label = self.label[idx]

        if self.train:
            angle = random.random() * 2 * math.pi
            cos, sin = math.cos(angle), math.sin(angle)
            rotmat = torch.tensor([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
            scale = torch.rand((3,)) * 0.2 + 0.9
            rotmat = torch.diag(scale) @ rotmat
            xyz = xyz @ rotmat
            xyz = xyz[torch.randperm(xyz.shape[0])]

        label = label.unsqueeze(0)
        if self.train:
            xyz, _ = fps_sample(xyz.unsqueeze(0), self.num_points)
        else:
            xyz, _ = fps_sample(xyz.unsqueeze(0), self.num_points + 200)
            xyz = xyz[:, torch.randperm(self.num_points + 200)[:self.num_points]]
        

        xyz = xyz.squeeze(0)

        xyz -= xyz.min(dim=0)[0]
        height = xyz[:, 2:] * 4
        height -= height.min(dim=0, keepdim=True)[0]
        if self.train:
            height += torch.empty((1, 1)).uniform_(-0.2, 0.2) * 4
        feature = torch.cat([xyz, height], dim=-1)

        cam_helper = CameraHelper(self.cam_opts, batch_size=self.batch_size, device=xyz.device)
        cam_helper.projects(xyz, scale=0) 

        cam_helper.cam_points.__update_attr__('p', xyz) 
        cam_helper.cam_points = make_cam_points(
            cam_helper.cam_points, 
            ks = self.k, 
            grid_size = None,
            n_samples = self.n_samples, 
            up_sample=False, 
            alpha=self.alpha
        )
        cam_helper.cam_points.__update_attr__('y', label) 
        cam_helper.cam_points.__update_attr__('f', feature)
        return cam_helper.cam_points


def scanobjectnn_collate_fn(batch):
    cam_list = list(batch)
    new_cam_points = merge_cam_points(cam_list, up_sample=False)
    return new_cam_points
