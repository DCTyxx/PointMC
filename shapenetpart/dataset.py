import __init__

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from backbone.camera_outside import CameraOptions, CameraHelper, make_cam_points, merge_cam_points
from utils.subsample import random_sample


def get_ins_mious(pred, target, shape, class_parts, multihead=False):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    ins_mious = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        parts = class_parts[int(shape[shape_idx])]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part
            target_part = target[shape_idx] == part
            I = torch.logical_and(pred_part, target_part).sum()
            U = torch.logical_or(pred_part, target_part).sum()
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    return ins_mious


class ShapeNetPartNormalTest(Dataset):
    classes = ['airplane', 'bag', 'cap', 'car',
               'chair', 'earphone', 'guitar', 'knife',
               'lamp', 'laptop', 'motorbike', 'mug',
               'pistol', 'rocket', 'skateboard', 'table']
    shape_classes = len(classes)
    num_classes = 50
    class_parts = {
        'earphone': [16, 17, 18], 
        'motorbike': [30, 31, 32, 33, 34, 35],
        'rocket': [41, 42, 43], 
        'car': [8, 9, 10, 11],
        'laptop': [28, 29], 
        'cap': [6, 7],
        'skateboard': [44, 45, 46], 
        'mug': [36, 37],
        'guitar': [19, 20, 21], 
        'bag': [4, 5],
        'lamp': [24, 25, 26, 27], 
        'table': [47, 48, 49],
        'airplane': [0, 1, 2, 3], 
        'pistol': [38, 39, 40],
        'chair': [12, 13, 14, 15], 
        'knife': [22, 23]
    }

    class2parts = []
    for i, cls in enumerate(classes):
        idx = class_parts[cls]
        class2parts.append(idx)
    class_segs = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    part_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

    def __init__(self,
                 presample_path,
                 k=[20, 20, 20, 20],
                 n_samples=[2048, 512, 192, 64],
                 alpha=0.,
                 batch_size=8,
                 cam_opts: CameraOptions = CameraOptions.default(),
                 ) -> None:
        super().__init__()
        self.k = k
        self.n_samples = n_samples
        self.alpha = alpha
        self.batch_size = batch_size
        self.cam_opts = cam_opts
        self.xyz, self.norm, self.shape, self.seg = torch.load(presample_path)

    def __len__(self):
        return self.xyz.shape[0]

    @classmethod
    def get_classes(cls):
        return cls.classes

    def __getitem__(self, idx):
        xyz = self.xyz[idx]
        norm = self.norm[idx]
        shape = self.shape[idx]
        seg = self.seg[idx]

        xyz -= xyz.min(dim=0)[0]
        height = xyz[:, 2:] * 4
        height -= height.min(dim=0, keepdim=True)[0]
        norm = torch.cat([norm, height], dim=-1)

        cam_helper = CameraHelper(self.cam_opts, batch_size=self.batch_size, device=xyz.device)
        cam_helper.projects(xyz)
        cam_helper.cam_points.__update_attr__('p', xyz)
        cam_helper.cam_points = make_cam_points(
            cam_helper.cam_points, 
            self.k, 
            None,
            self.n_samples, 
            up_sample=True, 
            alpha=self.alpha
        )
        cam_helper.cam_points.__update_attr__('f', norm)
        cam_helper.cam_points.__update_attr__('y', seg)
        return cam_helper.cam_points, shape


class ShapeNetPartNormal(Dataset):
    classes = ['airplane', 'bag', 'cap', 'car',
               'chair', 'earphone', 'guitar', 'knife',
               'lamp', 'laptop', 'motorbike', 'mug',
               'pistol', 'rocket', 'skateboard', 'table']
    shape_classes = len(classes)
    num_classes = 50
    class_parts = {
        'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35],
        'rocket': [41, 42, 43], 'car': [8, 9, 10, 11],
        'laptop': [28, 29], 'cap': [6, 7],
        'skateboard': [44, 45, 46], 'mug': [36, 37],
        'guitar': [19, 20, 21], 'bag': [4, 5],
        'lamp': [24, 25, 26, 27], 'table': [47, 48, 49],
        'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40],
        'chair': [12, 13, 14, 15], 'knife': [22, 23]}

    class2parts = []
    for i, cls in enumerate(classes):
        idx = class_parts[cls]
        class2parts.append(idx)
    class_segs = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    part_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

    def __init__(self,
                 dataset_dir: Path,
                 train=True,
                 presample=False,
                 voxel_max=2048,
                 k=[20, 20, 20, 20],
                 n_samples=[2048, 512, 192, 64],
                 alpha=0.,
                 batch_size=8,
                 cam_opts: CameraOptions = CameraOptions.default(),
                 ):
        self.train = train
        self.presample = presample
        self.k = k
        self.n_samples = n_samples
        self.alpha = alpha
        self.batch_size = batch_size
        self.cam_opts = cam_opts
        self.root = dataset_dir
        self.npoints = voxel_max
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if train:
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            else:
                fns = [fn for fn in fns if fn[0:-4] in test_ids]

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

    def __len__(self):
        return len(self.datapath)

    @classmethod
    def get_classes(cls):
        return cls.classes

    def __getitem__(self, idx):
        fn = self.datapath[idx]
        cat = self.datapath[idx][0]
        cls = self.classes[cat]
        data = np.loadtxt(fn[1]).astype(np.float32)
        xyz = data[:, 0:3]
        norm = data[:, 3:6]
        seg = data[:, -1]
        xyz = torch.from_numpy(xyz).float()
        norm = torch.from_numpy(norm).float()
        seg = torch.from_numpy(seg).long()

        if self.train:
            xyz, mask_idx = random_sample(xyz.unsqueeze(0), self.npoints)
            xyz = xyz.squeeze(0)

            mask_idx = mask_idx.squeeze(0)
            norm = norm[mask_idx]
            seg = seg[mask_idx]
            # mask
            mask = mask_idx == 0
            mask[0].fill_(False)
            seg[mask] = 255

            scale = torch.rand((3,)) * 0.4 + 0.8
            xyz *= scale
            if random.random() < 0.2:
                norm.fill_(0.)
            else:
                norm *= scale[[1, 2, 0]] * scale[[2, 0, 1]]
                norm = torch.nn.functional.normalize(norm, p=2, dim=-1, eps=1e-8)

            jitter = torch.empty_like(xyz).normal_(std=0.001)
            xyz += jitter

        if not self.presample:
            xyz -= xyz.min(dim=0)[0]
            height = xyz[:, 2:] * 4
            height -= height.min(dim=0, keepdim=True)[0]
            if self.train:
                height += torch.empty((1, 1), device=xyz.device).uniform_(-0.1, 0.1) * 4
            norm = torch.cat([norm, height], dim=-1)

        cam_helper = CameraHelper(self.cam_opts, batch_size=self.batch_size, device=xyz.device)
        cam_helper.cam_points.__update_attr__('p', xyz)
        if not self.presample:
            cam_helper.projects(xyz)
            cam_helper.cam_points = make_cam_points(
                cam_helper.cam_points, 
                self.k, 
                None,
                self.n_samples, 
                up_sample=True, 
                alpha=self.alpha
            )
        cam_helper.cam_points.__update_attr__('f', norm)
        cam_helper.cam_points.__update_attr__('y', seg)
        return cam_helper.cam_points, cls


def shapenetpart_collate_fn(batch):
    cam_list, shape = list(zip(*batch))
    new_cam_points = merge_cam_points(cam_list)
    shape = torch.tensor(shape, dtype=torch.long)
    new_cam_points.__update_attr__('shape', shape)
    return new_cam_points
