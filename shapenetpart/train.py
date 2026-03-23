import __init__

import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backbone.model import SegPartHead, PointMC

from shapenetpart.configs import model_configs
from shapenetpart.dataset import ShapeNetPartNormal, shapenetpart_collate_fn, get_ins_mious, \
    ShapeNetPartNormalTest
from utils.config import EasyConfig
from utils.logger import setup_logger_dist, format_dict, format_list
from utils.metrics import Timer, AverageMeter
from utils.misc import set_random_seed, load_state, save_state, resume_state, cal_model_params
from utils.tools import prepare_exp


def train(cfg, model, train_loader, optimizer, scheduler, scaler, epoch, scheduler_steps):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), desc='Train')
    loss_meter = AverageMeter()
    diff_meter = AverageMeter()
    steps_per_epoch = len(train_loader)
    for idx, cam_points in pbar:
        lam = scheduler_steps / (epoch * steps_per_epoch)
        lam = 3e-3 ** lam * 0.25
        scheduler.step(scheduler_steps)
        scheduler_steps += 1
        cam_points.to_cuda(non_blocking=True)
        shape = cam_points.__get_attr__('shape')
        target = cam_points.y
        target = target.flatten()
        with autocast():
            pred, diff = model(cam_points, shape)
            loss = F.cross_entropy(pred, target, label_smoothing=cfg.ls, ignore_index=cfg.ignore_index)

        optimizer.zero_grad(set_to_none=True)
        if cfg.use_amp:
            scaler.scale(loss + diff * lam).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss + diff * lam
            loss.backward()
            optimizer.step()
        loss_meter.update(loss)
        diff_meter.update(diff)
        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                             + f"Loss {loss_meter.avg:.4f} "
                             + f"Diff {diff_meter.avg:.4f}")
    return loss_meter.avg, diff_meter.avg, scheduler_steps


def validate(cfg, model, val_loader, epoch, **kwargs):
    class2parts = kwargs.get('class2parts', None)
    model.eval()
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32, device="cuda")
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32, device="cuda")
    ins_miou_list = []
    for idx, cam_points in pbar:
        cam_points.to_cuda(non_blocking=True)
        shape = cam_points.__get_attr__('shape')
        target = cam_points.y
        B, N = cfg.batch_size, target.shape[0] // cfg.batch_size
        with autocast():
            pred = model(cam_points, shape)
            pred = pred.max(dim=1)[1].view(B, N)
            target = target.view(B, N)
        ins_mious = get_ins_mious(pred, target, shape, class2parts)
        ins_miou_list += ins_mious
        for shape_idx in range(shape.shape[0]):  # sample_idx
            gt_label = int(shape[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[gt_label] += ins_mious[shape_idx]
            cls_nums[gt_label] += 1

        pbar.set_description(f"Val Epoch [{epoch}/{cfg.epochs}]")
    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    for cat_idx in range(16):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    ins_miou = ins_mious_sum / count
    cls_miou = torch.mean(cls_mious)
    cls_mious = [round(cm, 2) for cm in cls_mious.tolist()]
    return ins_miou, cls_miou, cls_mious


def main(cfg):
    torch.cuda.set_device(cfg.device)
    set_random_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True

    logging.info(f'Config:\n{cfg.__str__()}')
    train_ds = ShapeNetPartNormal(
        dataset_dir=cfg.dataset,
        train=True,
        voxel_max=cfg.model_cfg.train_cfg.voxel_max,
        k=cfg.model_cfg.train_cfg.k,
        n_samples=cfg.model_cfg.train_cfg.n_samples,
        alpha=cfg.model_cfg.train_cfg.alpha,
        batch_size=cfg.batch_size,
        cam_opts=cfg.model_cfg.train_cfg.cam_opts
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        collate_fn=shapenetpart_collate_fn,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        ShapeNetPartNormalTest(
            presample_path=cfg.presample_path,
            k=cfg.model_cfg.train_cfg.k,
            n_samples=cfg.model_cfg.train_cfg.n_samples,
            alpha=cfg.model_cfg.train_cfg.alpha,
            batch_size=cfg.batch_size,
            cam_opts=cfg.model_cfg.train_cfg.cam_opts
        ),
        batch_size=cfg.batch_size,
        collate_fn=shapenetpart_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last = True,
        num_workers=cfg.num_workers,
    )

    backbone = PointMC(
        **cfg.model_cfg.backbone_cfg,
        task_type='segpart',
        batch_size=cfg.batch_size,
        alpha=cfg.alpha,
    ).to('cuda')


    logging.info('backbone: %s' % backbone)

    model = SegPartHead(
        backbone=backbone,
        num_classes=cfg.model_cfg.num_classes,
        shape_classes=cfg.shape_classes,
        bn_momentum=cfg.model_cfg.bn_momentum,
    ).to('cuda')

    model_size, trainable_model_size = cal_model_params(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    logging.info('Number of trainable params: %.4f M' % (trainable_model_size / 1e6))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
    scaler = GradScaler()
    start_epoch = 1
    best_epoch = 0
    best_ins_miou = 0
    best_cls_miou = 0

    if cfg.mode == 'resume':
        cfg.ckpt = cfg.last_ckpt_path
        model_dict = resume_state(model, cfg.ckpt, optimizer=optimizer, scaler=scaler)
        start_epoch = model_dict['last_epoch'] + 1
        best_epoch = model_dict['best_epoch']
        best_ins_miou = model_dict['best_ins_miou']
        best_cls_miou = model_dict['best_cls_miou']
        logging.info(f"Resume model from {cfg.ckpt}, best_ins_miou={best_ins_miou:.4f}, best_cls_miou={best_cls_miou:.4f}, best_epoch={best_epoch}, start_epoch={start_epoch}")

    if cfg.mode == 'finetune':
        model_dict = load_state(model, cfg.ckpt, optimizer=optimizer, scaler=scaler)
        best_epoch = model_dict['best_epoch']
        best_ins_miou = model_dict['best_ins_miou']
        best_cls_miou = model_dict['best_cls_miou']
        logging.info(f"Finetune model from {cfg.ckpt}, best_ins_miou={best_ins_miou:.4f}, best_cls_miou={best_cls_miou:.4f}, best_epoch={best_epoch}, start_epoch={start_epoch}")

    steps_per_epoch = len(train_loader)
    scheduler_steps = steps_per_epoch * (start_epoch - 1)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=cfg.epochs * steps_per_epoch,
                                  lr_min=cfg.lr / 10000,
                                  cycle_decay=cfg.lr_decay,
                                  warmup_t=cfg.warmup_epochs * steps_per_epoch,
                                  warmup_lr_init=cfg.lr / 20)

    val_ins_miou, val_cls_miou, val_ious = 0., 0., []
    writer = SummaryWriter(log_dir=cfg.exp_dir)
    timer = Timer(dec=1)
    timer_meter = AverageMeter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        timer.record(f'E{epoch}_start')
        train_loss, train_diff, scheduler_steps = train(
            cfg, model, train_loader, optimizer, scheduler, scaler, epoch, scheduler_steps,
        )
        lr = optimizer.param_groups[0]['lr']
        time_cost = timer.record(f'E{epoch}_end')
        timer_meter.update(time_cost)
        logging.info(
            f'@E{epoch} train:    '
            + f'loss={train_loss:.4f} diff={train_diff:.4f} lr={lr:.6f}')

        is_best_ins, is_best_cls = False, False
        if epoch % cfg.val_freq == 0 or epoch >= cfg.epochs:
            with torch.no_grad():
                val_ins_miou, val_cls_miou, val_ious = validate(
                    cfg, model, val_loader, epoch, class2parts=train_ds.class2parts
                )
            logging.info(f'@E{epoch} val:      '
                         + f'ins_miou={val_ins_miou:.4f} cls_miou={val_cls_miou:.4f}')
            if val_ins_miou > best_ins_miou:
                logging.info(f'@E{epoch} new best: ins miou {best_ins_miou:.4f} => {val_ins_miou:.4f}')
                is_best_ins = True
                best_ins_miou = val_ins_miou
            if val_cls_miou > best_cls_miou:
                logging.info(f'@E{epoch} new best: cls miou {best_cls_miou:.4f} => {val_cls_miou:.4f}')
                is_best_cls = True
                best_cls_miou = val_cls_miou
            if not is_best_ins or not is_best_cls:
                logging.info(f'@E{epoch} cur best: ins miou {best_ins_miou:.4f}, cls miou {best_cls_miou:.4f}')
        if is_best_ins or is_best_cls:
            train_info = {
                'loss': train_loss,
                'diff': train_diff,
                'lr': f"{lr:.6f}",
                'time_cost': f"{time_cost:.2f}s",
                'time_cost_avg': f"{timer_meter.avg:.2f}s",
            }
            val_info = {
                'ins_miou': val_ins_miou,
                'cls_miou': val_cls_miou,
            }
            logging.info(f'@E{epoch} summary:'
                         + f'\ntrain: \n{format_dict(train_info)}'
                         + f'\nval: \n{format_dict(val_info)}'
                         + f'\nious: \n{format_list(ShapeNetPartNormal.get_classes(), val_ious)}')
            best_epoch = epoch
            if is_best_ins:
                save_state(cfg.best_small_ins_ckpt_path, model=model)
                save_state(cfg.best_ins_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                           best_epoch=best_epoch, last_epoch=epoch, best_ins_miou=best_ins_miou,
                           best_cls_miou=best_cls_miou)
            if is_best_cls:
                save_state(cfg.best_small_cls_ckpt_path, model=model)
                save_state(cfg.best_cls_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                           best_epoch=best_epoch, last_epoch=epoch, best_ins_miou=best_ins_miou,
                           best_cls_miou=best_cls_miou)
        save_state(cfg.last_ckpt_path, model=model, optimizer=optimizer, scaler=scaler,
                   best_epoch=best_epoch, last_epoch=epoch, best_ins_miou=best_ins_miou, best_cls_miou=best_cls_miou)
        if writer is not None and (epoch % cfg.val_freq == 0 or epoch >= cfg.epochs):
            writer.add_scalar('best_ins_miou', best_ins_miou, epoch)
            writer.add_scalar('best_cls_miou', best_cls_miou, epoch)
            writer.add_scalar('val_ins_miou', val_ins_miou, epoch)
            writer.add_scalar('val_cls_miou', val_cls_miou, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_diff', train_diff, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('time_cost_avg', timer_meter.avg, epoch)
            writer.add_scalar('time_cost', time_cost, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('shapenetpart training')
    # for prepare
    parser.add_argument('--exp', type=str, required=False, default='shapenetpart')
    parser.add_argument('--mode', type=str, required=False, default='train', choices=['train', 'finetune', 'resume'])
    parser.add_argument('--ckpt', type=str, required=False, default='')
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1, 10000))
    parser.add_argument('--model_size', type=str, required=False, default='l',
                        choices=['l'])

    # for dataset
    parser.add_argument('--dataset', type=str, required=False, default='')
    parser.add_argument('--presample', type=str, required=False, default='')
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--num_workers', type=int, required=False, default=16)

    # for train
    parser.add_argument('--epochs', type=int, required=False, default=400)
    parser.add_argument("--warmup_epochs", type=int, required=False, default=30)
    parser.add_argument("--lr", type=float, required=False, default=2e-3)
    parser.add_argument("--lr_decay", type=float, required=False, default=1.)
    parser.add_argument("--decay", type=float, required=False, default=0.05)
    parser.add_argument("--ls", type=float, required=False, default=0.2)
    parser.add_argument("--no_amp", action='store_true')

    # for validate
    parser.add_argument('--val_freq', type=int, required=False, default=1)

    # for model
    parser.add_argument("--use_cp", action='store_true')
    parser.add_argument("--step1", type=int, required=False, default=300)
    parser.add_argument("--step2", type=int, required=False, default=400)


    parser.add_argument("--add_info", type=str, required=True, default='')
    parser.add_argument("--device", type=int, required=False, default=0)
    parser.add_argument("--alpha", type=float, required=False, default=0.2)
    parser.add_argument('--biase_fn', type=str, required=False, default='origin') 

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load_args(args)

    model_cfg = model_configs[cfg.model_size]
    cfg.model_cfg = model_cfg
    cfg.model_cfg.backbone_cfg.use_cp = cfg.use_cp
    if cfg.use_cp:
        cfg.model_cfg.backbone_cfg.bn_momentum = 1 - (1 - cfg.model_cfg.bn_momentum) ** 0.5
    cfg.presample_path = os.path.join(cfg.dataset, cfg.presample)

    if cfg.mode == 'finetune':
        assert cfg.ckpt != ''
    cfg.use_amp = not cfg.no_amp

    # shapenetpart
    cfg.num_classes = 50
    cfg.shape_classes = 16
    cfg.ignore_index = 255

    prepare_exp(cfg)
    main(cfg)
