import time
import logging
import torch

from utils.misc import ObjDict


class Timer:
    def __init__(self, dec=1):
        self.dec = dec
        self.rec = ObjDict()
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    def record(self, desc):
        rec_time = time.time()
        self.rec[desc] = rec_time - self.start_time
        self.start_time = rec_time
        return self.rec[desc]*self.dec

    def record_desc(self, desc):
        return f'{desc}: {self.record(desc)}'

    def __str__(self):
        desc = ''
        for k, v in self.rec.items():
            desc += f'{k}: {v*self.dec}\n'
        return desc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        if hasattr(val, 'item'):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metric():
    def __init__(self, num_classes=13, device='cuda'):
        self.n = num_classes
        self.label = torch.arange(num_classes, dtype=torch.int64, device=device).unsqueeze(1)
        self.device = device
        self.reset()

    def reset(self):
        # pred == label == i for i in 0...num_classes
        self.intersection = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        # pred == i or label == i
        self.union = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        # label == i
        self.count = torch.zeros((self.n,), dtype=torch.int64, device=self.device)
        #
        self.acc = 0.
        self.macc = 0.
        self.iou = [0.] * self.n
        self.miou = 0.

    def update(self, pred, label):

        if len(label.shape) > 1:
            label = label.max(dim=1)[1]

        pred = pred.max(dim=1)[1].unsqueeze(0) == self.label

        label = label.unsqueeze(0) == self.label
        self.tmp_c = label.sum(dim=1)
        self.count += self.tmp_c 
        self.intersection += (pred & label).sum(dim=1)
        self.union += (pred | label).sum(dim=1)


    def update_few_shot(self, intersection, union, count):
        self.intersection += intersection
        self.union += union
        self.count += count
        

    def calc_macc(self):
        macc = self.intersection / self.count * 100
        macc = macc.mean()
        return macc
    

    def calc_miou(self):
        iou = self.intersection / self.union * 100
        iou = iou.mean()
        return iou

    def calc(self, digits=4):
        acc = self.intersection.sum() / self.count.sum() * 100
        self.acc = round(acc.item(), digits)
        macc = self.calc_macc()
        self.macc = round(macc.item(), digits)
        iou = self.intersection / self.union * 100
        self.iou = [round(i.item(), digits) for i in iou]
        miou = iou.mean()
        self.miou = round(miou.item(), digits)
        return acc, macc, miou, iou

