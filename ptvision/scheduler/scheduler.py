import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


def warmup_lr(current_step, steps_per_epoch, warmup_epochs, base_lr, warmup_init_lr, optim):
    warmup_steps = steps_per_epoch * warmup_epochs
    if current_step <= warmup_steps:
        if current_step == warmup_steps:
            lr = base_lr
        else:
            lr_inc = (base_lr - warmup_init_lr) / warmup_steps
            lr = warmup_init_lr + lr_inc * current_step
        for param_group in optim.param_groups:
            param_group['lr'] = lr


class WarmupCosine(_LRScheduler):
    def __init__(self, optimizer, max_epoch, steps_per_epoch, warmup_epochs=0, warmup_init_lr=0.0, eta_min=0.0):
        self.optimizer = optimizer
        self.tot_steps = max_epoch * steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.cosine_steps = self.tot_steps - self.warmup_steps
        self.warmup_init_lr = warmup_init_lr
        self.eta_min = eta_min
        self.step_cnt = 0
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))
        if self.warmup_steps > 0:
            self.lr_inc = [(x - warmup_init_lr) / self.warmup_steps for x in self.base_lrs]

    def get_lr(self):
        if self.step_cnt < self.warmup_steps:
            lrs = [self.warmup_init_lr + x * (self.step_cnt + 1) for x in self.lr_inc]
        else:
            cur_step = self.step_cnt + 1 - self.warmup_steps
            lrs = [self.eta_min + (x - self.eta_min) * (1 + math.cos(math.pi * cur_step / self.cosine_steps)) / 2 for x in self.base_lrs]
        self.step_cnt += 1
        return lrs

    def step(self):
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr


class WarmupStep(_LRScheduler):
    def __init__(self, optimizer, max_epoch, steps_per_epoch,
                 warmup_epochs=0, warmup_init_lr=0.0, lr_epochs=[30,60,80], lr_gamma=0.1):
        self.optimizer = optimizer
        self.tot_steps = max_epoch * steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.warmup_init_lr = warmup_init_lr
        self.lr_steps = [int(x * steps_per_epoch) for x in lr_epochs]
        self.lr_gamma = lr_gamma
        self.step_cnt = 0
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))
        self.lr_inc = [(x - warmup_init_lr) / self.warmup_steps for x in self.base_lrs]

    def get_lr(self):
        if self.step_cnt < self.warmup_steps:
            lrs = [self.warmup_init_lr + x * (self.step_cnt + 1) for x in self.lr_inc]
        else:
            if self.step_cnt not in self.lr_steps:
                pass
            else:
                self.base_lrs = [x * self.lr_gamma for x in self.base_lrs]
            lrs = self.base_lrs
        self.step_cnt += 1
        return lrs

    def step(self):
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr


class WarmupPoly(_LRScheduler):
    def __init__(self, optimizer, max_epoch, steps_per_epoch,
                 warmup_epochs=0, warmup_init_lr=0.0, poly_pow=2):
        self.optimizer = optimizer
        self.tot_steps = max_epoch * steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.warmup_init_lr = warmup_init_lr
        self.poly_pow = poly_pow
        self.step_cnt = 0
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))
        self.lr_inc = [(x - warmup_init_lr) / self.warmup_steps for x in self.base_lrs] if self.warmup_steps > 0 else 0

    def get_lr(self):
        if self.step_cnt < self.warmup_steps:
            lrs = [self.warmup_init_lr + x * (self.step_cnt + 1) for x in self.lr_inc]
        else:
            base = (1.0 - 1.0 * (self.step_cnt - self.warmup_steps) / (self.tot_steps - self.warmup_steps))
            lrs = [x * base ** self.poly_pow for x in self.base_lrs]
        self.step_cnt += 1
        return lrs

    def step(self):
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
