import __init__

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from backbone.camera_outside import CameraOptions, CameraHelper, make_cam_points, merge_cam_points

from utils.subsample import trunc_sample


class ModelNet40(Dataset):
    def __init__(self,
                 dataset_dir: Path,
                 train=True,
                 warmup=False,
                 num_points=1024,
                 k=[20, 20, 20],# 这里的k相当于knn的k近邻
                 n_samples=[1024, 256, 64],
                 alpha=0.,
                 batch_size=32,
                 cam_opts: CameraOptions = CameraOptions.default(),
                 ):
        dataset_dir = Path(dataset_dir)

        self.data_paths = dataset_dir.glob(f"ply_data_{'train' if train else 'test'}*.h5")
        self.train = train
        self.warmup = warmup
        self.num_points = num_points
        self.k = k
        self.n_samples = n_samples
        self.alpha = alpha
        self.batch_size = batch_size
        self.cam_opts = cam_opts

        datas, label = [], []
        for p in self.data_paths:
            f = h5py.File(p, 'r')
            datas.append(torch.from_numpy(f['data'][:]).float())
            label.append(torch.from_numpy(f['label'][:]).long())
            f.close()
        self.datas = torch.cat(datas)
        self.label = torch.cat(label)

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, idx):
        xyz = self.datas[idx]
        label = self.label[idx]

        # 随机缩放
        if self.train:
            scale = torch.rand((3,)) * (3/2 - 2/3) + 2/3
            xyz = xyz * scale
            xyz = xyz[torch.randperm(xyz.shape[0])]

        # 随机采样
        xyz, _ = trunc_sample(xyz.unsqueeze(0), self.num_points)
        xyz = xyz.squeeze(0)

        xyz -= xyz.min(dim=0)[0]
        height = xyz[:, 2:] * 4
        height -= height.min(dim=0, keepdim=True)[0]
        if self.train:
            height += torch.empty((1, 1)).uniform_(-0.2, 0.2) * 4
        feature = torch.cat([xyz, height], dim=-1)

        cam_helper = CameraHelper(self.cam_opts, batch_size=self.batch_size, device=xyz.device)
        # 进行映射 
        cam_helper.projects(xyz)
        cam_helper.cam_points.__update_attr__('p', xyz)
        cam_helper.cam_points = make_cam_points(
            cam_points = cam_helper.cam_points, 
            ks= self.k, 
            grid_size=None,
            n_samples=self.n_samples, 
            up_sample=False, 
            alpha=self.alpha
        )
        cam_helper.cam_points.__update_attr__('y', label)
        cam_helper.cam_points.__update_attr__('f', feature)
        return cam_helper.cam_points


def modelnet40_collate_fn(batch):
    cam_list = list(batch)
    new_cam_points = merge_cam_points(cam_list, up_sample=False)
    return new_cam_points
