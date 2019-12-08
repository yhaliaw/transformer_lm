import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding for transformer.

    See "Attention is all you need" for details:
    https://arxiv.org/abs/1706.03762

    In the paper, the sine and cosine embedding are interlaced. This
    implementation concatenate them as do many other implementations.
    """

    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim

        dim = model_dim // 2
        freq = math.log(10000) / (dim - 1)
        freq = torch.exp(-freq * torch.arange(dim, dtype=torch.float))
        self.register_buffer('freq', freq)

    def forward(self, position):
        # position: [pos]
        # self.freq: [dim]

        # Outer product
        # emb = [pos x dim]
        emb = torch.ger(position.type(self.freq.dtype), self.freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
