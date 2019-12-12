import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ConstituentAttention(nn.Module):
    """Constituent attention as described in Tree Transformer.

    Modified from official implementation at
    https://github.com/yaushian/Tree-Transformer/blob/master/attention.py.

    See "Tree Transformer: Integrating Tree Structures into
    Self-Attention" for details: https://arxiv.org/abs/1909.06639
    """

    def __init__(self, embed_dim, proj_dim, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.bias = bias

        # Parameters
        self.proj_weight = Parameter(torch.Tensor(self.proj_dim * 2, self.embed_dim))
        if bias:
            self.proj_bias = Parameter(torch.Tensor(self.proj_dim * 2))
        else:
            self.register_parameter('proj_bias', None)

        self.reset_parameters()

    def forward(self, context, prior, padding_mask=None):
        # Convert to batch first
        # context: [seq x batch x embed_dim] -> [batch x seq x embed_dim]
        device = context.device
        context = context.transpose(1, 0)
        batch_size, seq_len, _ = context.size()

        # Apply linear transform.
        # [batch x seq x embed_dim] -> [batch x seq x proj_dim]
        query, key = F.linear(context, self.proj_weight, self.proj_bias).chunk(2, dim=2)

        # Dot product between all query and key
        # score: [batch x seq x seq]
        score = torch.bmm(query, key.permute(0, 2, 1))
        score = score / self.embed_dim  # Scale by dimension.

        upper_diag = torch.ones(seq_len - 1).diag(1).type(torch.bool).to(device)
        diag = torch.ones(seq_len).diag(0).type(torch.bool).to(device)
        lower_diag = torch.ones(seq_len - 1).diag(-1).type(torch.bool).to(device)
        # Mask out non-neighboring dot product.
        # mask: [batch x seq x seq]
        # score: [batch x seq x seq]
        if padding_mask is not None:
            mask = ~padding_mask & upper_diag + lower_diag
        else:
            mask = upper_diag + lower_diag
        score = score.masked_fill(~mask, -1e9)

        # formula 6
        # neighbor_attn: [batch x seq x seq]
        neighbor_attn = F.softmax(score, dim=-1)
        # formula 7
        neighbor_attn = neighbor_attn * neighbor_attn.transpose(-2, -1) + 1e-9
        neighbor_attn = neighbor_attn.sqrt()
        # hierarchical constraint
        neighbor_attn = prior + (1 - prior) * neighbor_attn

        upper_tri = torch.ones((batch_size, seq_len, seq_len)).triu().type(context.dtype).to(device)
        # Build rest of attention matrix.
        # log_prob: [batch x seq x seq]
        log_prob = torch.log(neighbor_attn + 1e-9).masked_fill(~upper_diag, 0)
        log_prob = torch.bmm(log_prob, upper_tri)
        # Compute upper half.
        # c_attn: [batch x seq x seq]
        c_attn = torch.bmm(upper_tri, log_prob)
        c_attn = c_attn.exp().masked_fill(diag, 0).masked_fill(upper_tri == 0, 0)
        # Copy to lower half.
        c_attn = c_attn + c_attn.transpose(-2, -1) + neighbor_attn.masked_fill(diag == 0, 1e-9)

        return c_attn, neighbor_attn

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_weight)
        if self.proj_bias is not None:
            nn.init.constant_(self.proj_bias, 0.)


# The official implementation.
# Retrieved from: https://github.com/yaushian/Tree-Transformer/blob/master/attention.py
# The LayerNorm is commented out as it is handled elsewhere.
class GroupAttention(nn.Module):
    def __init__(self, d_model, dropout=0.8):
        super(GroupAttention, self).__init__()
        self.d_model = 256.
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        # self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, eos_mask, prior):
        batch_size, seq_len = context.size()[:2]

        # context = self.norm(context)

        a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), 1))
        b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0))
        c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), -1))
        tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0))

        # mask = eos_mask & (a+c) | b
        mask = eos_mask & (a + c)

        key = self.linear_key(context)
        query = self.linear_query(context)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model

        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1)
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(-2, -1) + 1e-9)
        neibor_attn = prior + (1. - prior) * neibor_attn

        t = torch.log(neibor_attn + 1e-9).masked_fill(a == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - b) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b == 0, 1e-9)

        return g_attn, neibor_attn


# TODO replace with unit test
if __name__ == '__main__':
    # 5 batch, 10 seq, 512 dim
    x = torch.normal(mean=0., std=1., size=(5, 10, 256))
    mask = torch.ones((5, 10), dtype=torch.int)
    mask[:, 7:] = 0
    mask = mask[:, None, :]
    prior = torch.zeros(1)

    group_attn = GroupAttention(256)  # Official implementation
    constituent_attn = ConstituentAttention(256, 256)

    group_attn.linear_query.weight, group_attn.linear_key.weight = [nn.Parameter(i) for i in constituent_attn.proj_weight.chunk(2, dim=0)]
    group_attn.linear_query.bias, group_attn.linear_key.bias = [nn.Parameter(i) for i in constituent_attn.proj_bias.chunk(2, dim=0)]

    output1, attn1 = group_attn(x, mask, prior)
    output1, attn1 = group_attn(x, mask, attn1)
    mask = (mask == 0)
    output2, attn2 = constituent_attn(x.permute(1,0,2), prior, mask)
    output2, attn2 = constituent_attn(x.permute(1,0,2), attn2, mask)

    assert torch.equal(output1, output2)