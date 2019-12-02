import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.optimizer import required


def Adam(params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-8, adam_w_mode=True,
                 weight_decay=0., amsgrad=False, set_grad_none=True):
    try:
        from apex.optimizers import FusedAdam
        return FusedAdam(params, lr, bias_correction, betas, eps, adam_w_mode, weight_decay,
                         amsgrad, set_grad_none)
    except ImportError:
        return optim.Adam(params, lr, betas, eps, weight_decay, amsgrad)


def SGD(params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False,
        wd_after_momentum=False, materialize_master_grads=True):
    try:
        from apex.optimizers import FusedSGD
        return FusedSGD(params, lr, momentum, dampening, weight_decay, nesterov, wd_after_momentum,
                        materialize_master_grads)
    except ImportError:
        return optim.SGD(params, lr, momentum, dampening, weight_decay, nesterov)


class DecayingCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    """Cosine Annealing with warm restarts and learning rate decay.

    The decay_factor is multiple to learning rate at the start of each
    restart. When the decay_factor is less than 1, each restart begins
    with a smaller learning rate than before.

    Slightly modified from PyTorch's CosineAnnealingWarmRestart class.
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay_factor=1., last_epoch=-1):
        self.decay_factor = decay_factor
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        if self.T_mult == 1:
            n = epoch // self.T_0
        else:
            n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr * (self.decay_factor ** n)
