import os
import sys

from glob import glob
from utils.logger import setup_logger_dist


def prepare_exp(cfg, exp_root='exp'):
    if exp_root == 'exp':
        exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
        if cfg.mode == 'resume':
            exp_id -= 1
        cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
        cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
        cfg.log_path = f'{cfg.exp_dir}/train.log'

        if cfg.exp == 'shapenetpart':
            cfg.best_small_ins_ckpt_path = f'{cfg.exp_dir}/best_small_ins.ckpt'
            cfg.best_small_cls_ckpt_path = f'{cfg.exp_dir}/best_small_cls.ckpt'
            cfg.best_ins_ckpt_path = f'{cfg.exp_dir}/best_ins.ckpt'
            cfg.best_cls_ckpt_path = f'{cfg.exp_dir}/best_cls.ckpt'
            cfg.last_ckpt_path = f'{cfg.exp_dir}/last.ckpt'

        if cfg.exp == 's3dis' or cfg.exp == 'scanobjectnn':
            cfg.best_small_ckpt_path = f'{cfg.exp_dir}/best_small.ckpt'
            cfg.best_ckpt_path = f'{cfg.exp_dir}/best.ckpt'
            cfg.last_ckpt_path = f'{cfg.exp_dir}/last.ckpt'

        os.makedirs(cfg.exp_dir, exist_ok=True)
        setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)

    else:
        exp_root = 'exp-test'
        exp_id = len(glob(f'{exp_root}/{cfg.exp}-*'))
        cfg.exp_name = f'{cfg.exp}-{exp_id:03d}'
        cfg.exp_dir = f'{exp_root}/{cfg.exp_name}'
        cfg.log_path = f'{cfg.exp_dir}/test.log'

        if hasattr(cfg, 'vis_root'):
            cfg.vis_root = 'visual'
            os.makedirs(cfg.vis_root, exist_ok=True)

        os.makedirs(cfg.exp_dir, exist_ok=True)
        setup_logger_dist(cfg.log_path, 0, name=cfg.exp_name)
        logfile = open(cfg.log_path, "a", 1)
        sys.stdout = logfile
