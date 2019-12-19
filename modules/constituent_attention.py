import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConstituentAttention(nn.Module):
    """Constituent attention as described in Tree Transformer.

    See "Tree Transformer: Integrating Tree Structures into
    Self-Attention" for details: https://arxiv.org/abs/1909.06639
    """

    def __init__(self, embed_dim, proj_dim, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.bias = bias

        # Parameters
        self.create_parameters()

        self.reset_parameters()

    def create_parameters(self):
        self.proj_weight = nn.Parameter(torch.Tensor(self.proj_dim * 2, self.embed_dim))
        if self.bias:
            self.proj_bias = nn.Parameter(torch.Tensor(self.proj_dim * 2))
        else:
            self.register_parameter('proj_bias', None)

    def neighbor_attention(self, context, padding_mask=None):
        dtype = context.dtype
        device = context.device
        batch_size, seq_len, _ = context.size()

        # Apply linear transform.
        # [batch x seq x embed_dim] -> [batch x seq x proj_dim]
        query, key = F.linear(context, self.proj_weight, self.proj_bias).chunk(2, dim=2)

        # Scaled dot product. Formula 5.
        # forward, backward: [batch x seq x proj_dim]
        # Forward is s_i,i+1, backward is s_i,i-1
        forward_query = query[:, :-1, :]
        forward_key = key[:, 1:, :]
        backward_query = query[:, 1:, :]
        backward_key = key[:, :-1, :]
        forward = (forward_query * forward_key).sum(dim=-1) / self.embed_dim
        backward = (backward_query * backward_key).sum(dim=-1) / self.embed_dim

        # Softmax. Formula 6.
        score = torch.ones((batch_size, seq_len, 2), dtype=dtype, device=device) * -math.inf
        score[:, torch.arange(seq_len - 1), 0] = forward
        score[:, torch.arange(1, seq_len), 1] = backward
        if padding_mask is not None:
            score[:, :, 0] = score[:, :, 0].masked_fill(padding_mask.roll(shifts=-1, dims=1) == 1, -math.inf)
        prob = F.softmax(score, dim=-1)

        # Average in log scale. Formula 7.
        shift_prob = prob[:, :, 1].roll(shifts=-1)
        prob = (prob[:, :, 0] * shift_prob + 1e-6).sqrt()
        return prob

    def forward(self, context, prior, padding_mask=None):
        dtype = context.dtype
        device = context.device

        # Convert to batch first
        # context: [seq x batch x embed_dim] -> [batch x seq x embed_dim]
        context = context.permute(1, 0, 2)
        batch_size, seq_len, _ = context.size()
        # Convert to torch.int as a hack to get around torch.roll() not
        # supported for torch.bool on GPU.
        if padding_mask is not None:
            padding_mask = padding_mask.view(batch_size, seq_len).type(torch.int)

        # Neighboring attention.
        neighbor_attn = self.neighbor_attention(context, padding_mask)

        # Hierarchical constraint. Formula 8.
        neighbor_attn = prior + (1 - prior) * neighbor_attn

        # Compute the rest of log probability.
        upper_tri = torch.ones((seq_len, seq_len), dtype=dtype, device=device).triu_()[None, :, :]
        constituent_attn = torch.zeros((batch_size, seq_len, seq_len), dtype=dtype, device=device)
        log_prob = torch.log(neighbor_attn)
        if padding_mask is not None:
            log_prob = log_prob.masked_fill(padding_mask.roll(shifts=-1, dims=1) == 1, 0)
        # Compute the upper half
        constituent_attn[:, torch.arange(seq_len - 1), torch.arange(1, seq_len)] = log_prob[:, :-1]
        constituent_attn = torch.matmul(constituent_attn, upper_tri)
        constituent_attn = torch.matmul(upper_tri, constituent_attn)
        # Copy to lower half.
        constituent_attn = constituent_attn + constituent_attn.permute(0, 2, 1)
        constituent_attn = constituent_attn.exp()

        # Mask out self
        diag = torch.ones(seq_len, device=device).diag(0)
        constituent_attn = constituent_attn.masked_fill(diag == 1, 0)

        if padding_mask is not None:
            constituent_attn.masked_fill_(
                (padding_mask[:, :, None] == 1) | (padding_mask[:, None, :] == 1), 0)
        return constituent_attn, neighbor_attn

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_weight)
        if self.proj_bias is not None:
            nn.init.constant_(self.proj_bias, 0.)

    def extra_repr(self):
        return f"(dot_product_attn){self.embed_dim}, {self.proj_dim}, bias={self.bias}"


class RecurrentConstituentAttention(ConstituentAttention):
    """Constituent attention with recurrence.

    Assumes padding are at the end of the sequence.
    """

    def __init__(self, embed_dim, proj_dim, bias=True, attn_type=None):
        self.attn_type = attn_type
        if self.attn_type == 'recurrent_dot_product':
            self.attn = self.recurrent_dot_product
        else:
            raise NotImplementedError
        super().__init__(embed_dim, proj_dim, bias)

    def create_parameters(self):
        if self.attn_type == 'recurrent_dot_product':
            self.forward_lstm = nn.LSTM(
                self.embed_dim, self.proj_dim, bias=self.bias, batch_first=True
            )
            self.backward_lstm = nn.LSTM(
                self.embed_dim, self.proj_dim, bias=self.bias, batch_first=True
            )
            self.proj_weight = nn.Parameter(torch.Tensor(self.proj_dim * 2, self.embed_dim))
            if self.bias:
                self.proj_bias = nn.Parameter(torch.Tensor(self.proj_dim * 2))
            else:
                self.register_parameter('proj_bias', None)

    def reset_parameters(self):
        if self.attn_type == 'recurrent_dot_product':
            nn.init.xavier_uniform_(self.proj_weight)
            if self.proj_bias is not None:
                nn.init.constant_(self.proj_bias, 0.)

    def neighbor_attention(self, context, padding_mask=None):
        return self.attn(context, padding_mask)

    def recurrent_dot_product(self, context, padding_mask=None):
        dtype = context.dtype
        device = context.device
        batch_size, seq_len, _ = context.size()

        length = (padding_mask == 0).sum(dim=-1)

        # Apply LSTM
        # [seq x batch x embed_dim] -> [seq x batch x proj_dim]
        forward_context = context
        backward_context = context.flip(dims=[-1])
        forward_context = pack_padded_sequence(forward_context, length, batch_first=True, enforce_sorted=False)
        backward_context = pack_padded_sequence(backward_context, length, batch_first=True, enforce_sorted=False)
        forward, _ = self.forward_lstm(forward_context)
        backward, _ = self.backward_lstm(backward_context)
        forward, _ = pad_packed_sequence(forward, batch_first=True, total_length=seq_len)
        backward, _ = pad_packed_sequence(backward, batch_first=True, total_length=seq_len)

        # Apply linear transform.
        # [batch x seq x embed_dim] -> [batch x seq x proj_dim]
        backward_query, forward_key = F.linear(forward, self.proj_weight, self.proj_bias).chunk(2, dim=2)
        forward_query, backward_key = F.linear(backward, self.proj_weight, self.proj_bias).chunk(2, dim=2)

        # Scaled dot product. Formula 5.
        # forward, backward: [batch x seq x proj_dim]
        # Forward is s_i,i+1, backward is s_i,i-1
        forward_query = forward_query[:, :-1, :]
        forward_key = forward_key[:, 1:, :]
        backward_query = backward_query[:, 1:, :]
        backward_key = backward_key[:, :-1, :]
        forward = (forward_query * forward_key).sum(dim=-1) / self.embed_dim
        backward = (backward_query * backward_key).sum(dim=-1) / self.embed_dim

        # Softmax. Formula 6.
        score = torch.ones((batch_size, seq_len, 2), dtype=dtype, device=device) * -math.inf
        score[:, torch.arange(seq_len - 1), 0] = forward
        score[:, torch.arange(1, seq_len), 1] = backward
        if padding_mask is not None:
            score[:, :, 0] = score[:, :, 0].masked_fill(padding_mask.roll(shifts=-1, dims=1) == 1, -math.inf)
        prob = F.softmax(score, dim=-1)

        # Average in log scale. Formula 7.
        shift_prob = prob[:, :, 1].roll(shifts=-1)
        prob = (prob[:, :, 0] * shift_prob + 1e-6).sqrt()
        return prob


# The official implementation.
# Retrieved from: https://github.com/yaushian/Tree-Transformer/blob/master/attention.py
# Changes:
# Fix the hardcoded self.d_model: self.d_model = 256  ->  self.d_model = d_model
# The LayerNorm is commented out as it is handled elsewhere.
# The .cuda() is changed to .to(context.device), so it works on both CPU and GPU.
class GroupAttention(nn.Module):
    def __init__(self, d_model, dropout=0.8):
        super(GroupAttention, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        # self.linear_output = nn.Linear(d_model, d_model)
        # self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, eos_mask, prior):
        batch_size, seq_len = context.size()[:2]

        # context = self.norm(context)

        a = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), 1)).to(context.device)
        b = torch.from_numpy(np.diag(np.ones(seq_len, dtype=np.int32), 0)).to(context.device)
        c = torch.from_numpy(np.diag(np.ones(seq_len - 1, dtype=np.int32), -1)).to(context.device)
        tri_matrix = torch.from_numpy(np.triu(np.ones([seq_len, seq_len], dtype=np.float32), 0)).to(context.device)

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


class OriginalConstituentAttention(nn.Module):
    """Wrapper for official implementation."""

    def __init__(self, embed_dim, proj_dim, bias=True):
        assert embed_dim == proj_dim, "Different projection dimension not supported."
        assert bias, "No bias is not supported."
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.bias = bias
        self.grp_attn = GroupAttention(embed_dim)

    def forward(self, context, prior, padding_mask):
        mask = padding_mask.type(torch.int)
        context = context.permute(1, 0, 2)  # Convert to batch first.
        weight, prior = self.grp_attn(context, mask, prior)
        return weight, prior

    def extra_repr(self):
        return f"{self.embed_dim}, {self.proj_dim}, bias={self.bias}"


# TODO replace with unit test
if __name__ == '__main__':
    # 5 batch, 10 seq, 512 dim
    x = torch.normal(mean=0., std=1., size=(5, 10, 256))
    mask = torch.ones((5, 10), dtype=torch.int)
    mask[:3, 5:] = 0
    mask = mask[:, None, :]
    prior = torch.zeros(1)

    group_attn = GroupAttention(256)  # Official implementation
    constituent_attn = ConstituentAttention(256, 256)

    group_attn.linear_query.weight, group_attn.linear_key.weight = [nn.Parameter(i) for i in constituent_attn.proj_weight.chunk(2, dim=0)]
    group_attn.linear_query.bias, group_attn.linear_key.bias = [nn.Parameter(i) for i in constituent_attn.proj_bias.chunk(2, dim=0)]

    output1, attn1 = group_attn(x, mask, prior)
    for _ in range(1):
        output1, attn1 = group_attn(x, mask, attn1)
    mask = (mask == 0)
    output2, attn2 = constituent_attn(x.permute(1, 0, 2), prior, mask)
    for _ in range(1):
        output2, attn2 = constituent_attn(x.permute(1, 0, 2), attn2, mask)

    assert torch.equal(output1, output2)
