import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    """Gaussian error linear unit.

    See "Gaussian Error Linear Units (GELUs)":
    https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1. + torch.erf(x / math.sqrt(2.)))


def approx_gelu(x):
    """Approximated gaussian error linear unit.

    See "Gaussian Error Linear Units (GELUs)":
    https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
