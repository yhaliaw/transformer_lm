import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_norm import LayerNorm
from .multihead_attention import MultiheadAttention
from .utils import Linear


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed forward layer in transformer."""

    def __init__(self, model_dim, inner_dim, bias=True, dropout=0.,
                 activation=F.relu):
        super().__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.dropout = dropout
        self.activation = activation

        # Parameters
        self.input_linear = Linear(model_dim, inner_dim, bias=bias)
        self.output_linear = Linear(inner_dim, model_dim, bias=bias)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.activation(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.output_linear(x)
        return x


class TransformerLayer(nn.Module):
    """A single layer in transformer.

    See "Attention is all you need" for details:
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, model_dim, num_head, inner_dim, dropout, attn_dropout=0., head_dim=None,
                 bias=False, activation=F.relu):
        super().__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.inner_dim = inner_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.bias = bias
        self.activation = activation

        # Parameters
        self.self_attn_layer_norm = LayerNorm(model_dim, elementwise_affine=True)
        self.self_attn = MultiheadAttention(
            model_dim, num_head, dropout=attn_dropout, head_dim=head_dim
        )
        self.position_wise_layer_norm = LayerNorm(model_dim, elementwise_affine=True)
        self.position_wise = PositionWiseFeedForward(
            model_dim, inner_dim, bias=bias, dropout=dropout, activation=activation
        )

    def forward(self, x, self_attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attn_mask=self_attn_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.position_wise_layer_norm(x)
        x = self.position_wise(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        return x


# Unused
class Sublayer(nn.Module):
    """Sublayer in transformer.

    In "Attention is all you need" paper, each sublayer is post-process
    with dropout -> residual connection -> layer norm.
    The Tensor2Tensor GitHub repository recommends pre-process with
    layer norm and post-process with dropout -> residual connection.
    The default is the Tensor2Tensor version as it is more stable.
    Set 'norm_after' to true for the paper version.
    """

    def __init__(self, module, model_dim, dropout=0.1, norm_before=True):
        super().__init__()
        self.module = module
        self.model_dim = model_dim
        self.dropout = dropout
        self.norm_before = norm_before

        # Parameters
        self.layer_norm = LayerNorm(model_dim)

    def forward(self, x, *args, **kwargs):
        if self.norm_before:
            return x + F.dropout(self.module(self.layer_norm(x), *args, **kwargs),
                                 self.dropout, training=self.training)
        else:
            return self.layer_norm(x + F.dropout(self.module(x, *args, **kwargs),
                                                 self.dropout, training=self.training))
