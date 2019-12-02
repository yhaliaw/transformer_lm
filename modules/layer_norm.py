import torch
import torch.nn as nn


def LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True):
    try:
        from apex.normalization import FusedLayerNorm
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    except ImportError:
        return nn.LayerNorm(normalized_shape, eps, elementwise_affine)