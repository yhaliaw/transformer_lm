import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_dropout(module, prob, x, *args, **kwargs):
    if module.training and torch.rand(1)[0] >= prob:
        return module(x, *args, **kwargs)
    else:
        return x
