import __init__

import argparse
import logging
import os
import sys
import numpy as np
import torch

from glob import glob
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backbone.model import SegSemHead, PointMC
from backbone.camera_outside import merge_cam_points
from s3dis.configs import model_configs
from s3dis.dataset import S3DIS, s3dis_test_collate_fn
from utils.config import EasyConfig
from utils.logger import setup_logger_dist, format_dict, format_list
from utils.metrics import Timer, AverageMeter, Metric
from utils.misc import set_random_seed, cal_model_params, cal_model_flops, load_state
from utils.visual_utils import write_obj
from utils.tools import prepare_exp


def cal_flops(cfg, model):
    cfg.model_cfg.train_cfg.voxel_max = 1024
    ds = S3DIS(
        dataset_dir=cfg.dataset,
        area=cfg.test_area,
        loop=1,
        train=True,
        warmup=False,
        voxel_max=cfg.model_cfg.train_cfg.voxel_max,
        k=cfg.model_cfg.train_cfg.k,
        grid_size=cfg.model_cfg.train_cfg.grid_size,
        alpha=cfg.model_cfg.train_cfg.alpha,
        batch_size=1,
        cam_opts=cfg.model_cfg.train_cfg.cam_opts
    )
    cam_points = merge_cam_points([ds[0]])
    cam_points.to_cuda(non_blocking=True)
    flops, macs, params = cal_model_flops(model, inputs=(cam_points,))
    logging.info(f'FLOPs = {flops / 1e9:.4f}G, Macs = {macs / 1e9:.4f}G, Params = {params / 1e6:.4f}')


def save_vis_results(cfg, file_name, xyz, rgb, label, pred):
    rgb = rgb.cpu().numpy().squeeze() / 255.0

    label = label.long()
    gt = label.cpu().numpy().squeeze()
    gt = cfg.cmap[gt, :]

    pred = pred.argmax(1)
    pred = pred.cpu().numpy().squeeze()
    pred = cfg.cmap[pred, :]

    write_obj(xyz, rgb, f'{cfg.vis_root}/rgb-{file_name}.txt')
    write_obj(xyz, gt, f'{cfg.vis_root}/gt-{file_name}.txt')
    write_obj(xyz, pred, f'{cfg.vis_root}/pred-{file_name}.txt')


def warmup(cfg, model, test_loader):
    steps_per_epoch = 10
    pbar = tqdm(enumerate(test_loader), total=steps_per_epoch, desc='Warmup')
    i = 0
    for idx, (cam_points, full_nn, full_lbl) in pbar:
        cam_points.to_cuda(non_blocking=True)
        with autocast():
            model(cam_points)
        pbar.set_description(f"Warmup [{idx}/{steps_per_epoch}]")

        i += 1
        if i > steps_per_epoch:
            return


@torch.no_grad()
def main(cfg):
    torch.cuda.set_device(cfg.device)
    set_random_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True

    logging.info(f'Config:\n{cfg.__str__()}')

    test_ds = S3DIS(
        dataset_dir=cfg.dataset,
        area=cfg.test_area,
        loop=cfg.test_loop,
        train=False,
        warmup=False,
        voxel_max=cfg.model_cfg.train_cfg.voxel_max,
        k=cfg.model_cfg.train_cfg.k,
        grid_size=cfg.model_cfg.train_cfg.grid_size,
        alpha=cfg.model_cfg.train_cfg.alpha,
        batch_size=1,
        cam_opts=cfg.model_cfg.train_cfg.cam_opts,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        collate_fn=s3dis_test_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    cfg.cmap = np.array(test_ds.cmap).astype(np.float32) / 255.

    backbone = PointMC(
        **cfg.model_cfg.backbone_cfg,
        batch_size=cfg.batch_size,
        alpha=cfg.alpha,
    ).to('cuda')

    model = SegSemHead(
        backbone=backbone,
        num_classes=cfg.model_cfg.num_classes,
        bn_momentum=cfg.model_cfg.bn_momentum,
    ).to('cuda')

    model_size, trainable_model_size = cal_model_params(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    logging.info('Number of trainable params: %.4f M' % (trainable_model_size / 1e6))

    model.eval()
    cal_flops(cfg, model)

    if cfg.ckpt == '':
        logging.warning(f'Checkpoint path is empty, quit now...')
        return
    load_state(model, cfg.ckpt)

    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()
    m = Metric(cfg.num_classes)
    steps_per_epoch = len(test_loader)

    cum = 0
    cnt = 0
    current_full_lbl = None

    pbar = tqdm(enumerate(test_loader), total=test_loader.__len__(), desc='Testing')

    logging.info(f'Start testing {cfg.exp_name} on area {cfg.test_area}...')

    for idx, (cam_points, full_nn, full_lbl) in pbar:
        cam_points.to_cuda(non_blocking=True)
        full_nn = full_nn.cuda(non_blocking=True)

        target = cam_points.y
        timer.record(f'I{idx}')
        with autocast():
            pred = model(cam_points)
        time_cost = timer.record(f'I{idx}')
        timer_meter.update(time_cost)
        cum = cum + pred[full_nn]
        cnt += 1
        current_full_lbl = full_lbl

        if cnt % cfg.test_loop == 0:
            current_full_lbl = current_full_lbl.cuda(non_blocking=True)
            m.update(cum, current_full_lbl)
            cum = 0
            cnt = 0
            current_full_lbl = None

        pbar.set_description(f"Testing [{idx}/{steps_per_epoch}] mACC {m.calc_macc():.4f}")
        

        if writer is not None and idx % cfg.metric_freq == 0:
            writer.add_scalar('time_cost_avg', timer_meter.avg, idx)
            writer.add_scalar('time_cost', time_cost, idx)

        if cfg.vis and cnt == 0:
            save_vis_results(
                cfg,
                f's3dis-{idx // cfg.test_loop}',
                cam_points.p,
                cam_points.__get_attr__('rgb'),
                cam_points.y,
                pred,
            )


        if len(target.shape) > 1:
            target = target.max(dim=1)[1]


        pred = pred.max(dim=1)[1].unsqueeze(0) == target

        target = target.unsqueeze(0) == target
        tmp_c = target.sum(dim=1)

        intersection = (pred & target).sum(dim=1)
        union = (pred | target).sum(dim=1)
        acc = intersection / tmp_c * 100
        iou = intersection / union * 100

        logging.info(f'[{idx}/{steps_per_epoch}] '
                     + f'mACC: {acc.mean():.4f}, '
                     + f'mIoU: {iou.mean():.4f}')
    

    logging.info(f'Finish testing {cfg.exp_name} on area {cfg.test_area}.')


    acc, macc, miou, iou = m.calc()
    test_info = {
        'miou': miou,
        'macc': macc,
        'oa': acc,
        'time_cost_avg': f"{timer_meter.avg:.8f}s",
    }
    logging.info(f'Summary:'
                 + f'\ntest: \n{format_dict(test_info)}'
                 + f'\nious: \n{format_list(S3DIS.get_classes(), iou)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('s3dis testing')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='s3dis')
    parser.add_argument('--ckpt', type=str, required=False, default='')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1, 10000))
    parser.add_argument('--model_size', type=str, required=False, default='l', choices=['l'])

    # for dataset
    parser.add_argument('--dataset', type=str, required=False, default='/mnt/data/0/xxyu/data/s3dis')
    parser.add_argument('--test_loop', type=int, required=False, default=12)
    parser.add_argument('--test_area', type=str, required=False, default='5')
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--num_workers', type=int, required=False, default=16)

    # for test
    parser.add_argument("--metric_freq", type=int, required=False, default=1)

    # for vis
    parser.add_argument("--vis", action='store_true')

    # for model
    parser.add_argument("--use_cp", action='store_true')
    parser.add_argument("--alpha", type=float, required=False, default=1)
    parser.add_argument("--device", type=int, required=False, default=2)

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load_args(args)

    model_cfg = model_configs[cfg.model_size]
    cfg.model_cfg = model_cfg
    cfg.model_cfg.backbone_cfg.use_cp = cfg.use_cp
    if cfg.use_cp:
        cfg.model_cfg.backbone_cfg.bn_momentum = 1 - (1 - cfg.model_cfg.bn_momentum) ** 0.5

    cfg.num_classes = 13
    cfg.ignore_index = -100

    prepare_exp(cfg, exp_root='exp-test')
    main(cfg)
