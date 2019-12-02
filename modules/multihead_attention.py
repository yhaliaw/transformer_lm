import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from modules.utils import Linear


class MultiheadAttention(nn.Module):
    """Multi-Head Attention in transformer.

    See "Attention is all you need" for details:
    https://arxiv.org/abs/1706.03762

    For behavior in the paper, set output_dim to model dimension and
    leave head_dim, query_dim, key_dim, value_dim to default values.

    For process_attn_fn hook function, the input is the attention
    weights and the expected output is the modified attention weights.
    e.g,

    def process_attn_fn(attn_weights):
        # Do stuff
        return attn_weights

    Args:
        output_dim: Output dimension.
        num_head: Number of projection heads.
        dropout: Dropout rate for attention weights.
        bias: Add bias to query, key, value projection and linear layer.
        head_dim: Dimension of each projection head.
        query_dim: Dimension of query input.
        key_dim: Dimension of key input.
        value_dim: Dimension of value input.
    """

    def __init__(self, output_dim, num_head, dropout=0., bias=True, head_dim=None,
                 query_dim=None, key_dim=None, value_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.num_head = num_head
        self.dropout = dropout
        self.bias = bias
        assert head_dim is not None or output_dim % num_head == 0
        self.head_dim = output_dim // num_head if head_dim is None else head_dim
        self.query_dim = output_dim if query_dim is None else query_dim
        self.key_dim = output_dim if key_dim is None else key_dim
        self.value_dim = self.key_dim if value_dim is None else value_dim
        self.same_dim = self.query_dim == self.key_dim == self.value_dim

        self.scale = 1 / (self.head_dim ** 0.5)

        # Parameters
        if self.same_dim:
            self.proj_weight = Parameter(torch.Tensor(self.head_dim * self.num_head * 3, self.query_dim))
            if bias:
                self.proj_bias = Parameter(torch.Tensor(self.head_dim * self.num_head * 3))
            else:
                self.register_parameter('proj_bias', None)
        else:
            self.query_weight = Parameter(torch.Tensor(self.head_dim * self.num_head, self.query_dim))
            self.key_weight = Parameter(torch.Tensor(self.head_dim * self.num_head, self.key_dim))
            self.value_weight = Parameter(torch.Tensor(self.head_dim * self.num_head, self.value_dim))
            if bias:
                self.query_bias = Parameter(torch.Tensor(self.head_dim * num_head))
                self.key_bias = Parameter(torch.Tensor(self.head_dim * num_head))
                self.value_bias = Parameter(torch.Tensor(self.head_dim * num_head))
            else:
                self.register_parameter('query_bias', None)
                self.register_parameter('key_bias', None)
                self.register_parameter('value_bias', None)
        self.linear = Linear(self.head_dim * self.num_head, self.output_dim, bias=bias)

        self.reset_parameters()

    def forward(self, query, key=None, value=None, attn_mask=None, attn_hook=None):
        """Compute multi-head attention.

        Args:
            query: Embedding for query in sequence first format.
            key: Embedding for key. Leave as 'None' for self attention.
            value: Embedding for value.
            attn_mask: Mask for attention. 'True' are masked out.
            attn_hook: Hook function to process attention weights.
        Returns: Output of multi-head attention.
        """
        self_attn = key is None
        key = query if key is None else key
        value = key if value is None else value
        query_len, batch_size, _ = query.size()
        key_len = key.size(0)
        # Input shape: [seq x batch x dim]

        # Projection for each head.
        # [seq x batch x dim] -> [seq x batch x head x proj_dim]
        query, key, value = self.projection(query, key, value, self_attn=self_attn)

        # Reshape for batch matrix multiplication.
        # [seq x batch x head x proj_dim] -> [batch * head x k_seq x proj_dim]
        query = query.reshape(query_len, batch_size * self.num_head, self.head_dim).permute(1, 0, 2)
        key = key.reshape(key_len, batch_size * self.num_head, self.head_dim).permute(1, 0, 2)
        value = value.reshape(key_len, batch_size * self.num_head, self.head_dim).permute(1, 0, 2)

        # Matrix multiplication of query and key.
        # attn_score: [batch * head x q_seq x k_seq]
        attn_score = torch.bmm(query, key.permute(0, 2, 1))

        # Scale scores.
        attn_score = attn_score * self.scale

        # Apply mask.
        # attn_mask: [batch x q_seq x k_seq]
        # attn_score: [batch * head x q_seq x k_seq]
        if attn_mask is not None:
            attn_score = attn_score.view(batch_size, self.num_head, query_len, key_len)
            attn_score.masked_fill_(attn_mask[:, None, :, :], -math.inf)
            attn_score = attn_score.view(batch_size * self.num_head, query_len, key_len)

        # Apply softmax.
        # attn_score/attn_weight: [batch * head x q_seq x k_seq]
        attn_weight = F.softmax(attn_score, dim=-1)

        # Optional: hook function to process the attention weights.
        if attn_hook is not None:
            attn_weight = attn_hook(attn_weight)
        # Optional: Dropout on attention.
        if not self.dropout == 0.:
            attn_weight = F.dropout(attn_weight, self.dropout, self.training)

        # Matrix Multiplication of attention weights and value
        # attn_weight: [batch * head x q_seq x k_seq]
        # value: [batch * head x k_seq x proj_dim]
        # x: [batch * head x q_seq x proj_dim]
        x = torch.bmm(attn_weight, value)

        # Concat the heads.
        # x: [batch * head x q_seq x proj_dim] -> [q_seq x batch x dim]
        x = x.permute(1, 0, 2)
        x = x.reshape(query_len, batch_size, self.num_head, self.head_dim)
        x = x.reshape(query_len, batch_size, self.num_head * self.head_dim)

        # Apply linear layer.
        # x: [q_seq x batch x dim] -> [q_seq x batch x dim]
        x = self.linear(x)
        return x

    def projection(self, query, key, value, self_attn=False):
        query_len, batch_size, _ = query.size()
        key_len = key.size(0)
        if self_attn:
            assert self.same_dim
            query, key, value = F.linear(query, self.proj_weight, self.proj_bias).chunk(3, dim=-1)
        elif self.same_dim:
            key_len = key.size(0)
            query_weight, key_weight, value_weight = self.proj_weight.chunk(3, dim=0)
            query_bias, key_bias, value_bias = self.proj_bias.chunk(3, dim=0)
            query = F.linear(query, query_weight, query_bias)
            key = F.linear(key, key_weight, key_bias)
            value = F.linear(value, value_weight, value_bias)
        else:
            key_len = key.size(0)
            query = F.linear(query, self.query_weight, self.query_bias)
            key = F.linear(key, self.key_weight, self.key_bias)
            value = F.linear(value, self.value_weight, self.value_bias)
        query = query.view(query_len, batch_size, self.num_head, self.head_dim)
        key = key.view(key_len, batch_size, self.num_head, self.head_dim)
        value = value.view(key_len, batch_size, self.num_head, self.head_dim)
        return query, key, value

    def reset_parameters(self):
        if self.same_dim:
            nn.init.xavier_uniform_(self.proj_weight)
            if self.proj_bias is not None:
                nn.init.constant_(self.proj_bias, 0.)
        else:
            nn.init.xavier_uniform_(self.query_weight)
            nn.init.xavier_uniform_(self.key_weight)
            nn.init.xavier_uniform_(self.value_weight)
            if self.bias:
                nn.init.constant_(self.query_bias, 0.)
                nn.init.constant_(self.key_bias, 0.)
                nn.init.constant_(self.value_bias, 0.)

    def extra_repr(self):
        return f"output_dim={self.output_dim}, num_head={self.num_head}, " \
               f"head_dim={self.head_dim}, bias={self.bias}"


# TODO replace with unit test?
if __name__ == '__main__':
    # Test implementation is same as PyTorch implementation of F.multi_head_attention_forward.
    from modules.utils import subsequent_mask
    sub_mask = subsequent_mask(10)
    padding_mask = torch.zeros(5, 10, dtype=torch.bool)
    padding_mask[0][7:] = 1
    padding_mask[1][1:] = 1
    padding_mask[2][8:] = 1
    padding_mask[3][3:] = 1
    padding_mask[4][6:] = 1

    # 10 seq, 5 batch, 512 dim
    x = torch.normal(mean=0., std=1., size=(10, 5, 512))

    attn = MultiheadAttention(512, 8)
    self_attn = MultiheadAttention(512, 8)
    nn.init.normal_(attn.proj_bias, 0, 0.25)

    self_attn.proj_weight = attn.proj_weight
    self_attn.proj_bias = attn.proj_bias
    self_attn.linear = attn.linear

    mask = padding_mask[:, None, :] | sub_mask[None, :, :]
    output = attn(x, x, x, attn_mask=mask)  # Without self attention speed up.
    self_output = self_attn(x, attn_mask=mask)  # with self attention speed up.

    attn_mask = torch.zeros_like(sub_mask, dtype=torch.float32)
    attn_mask = attn_mask.masked_fill(sub_mask, -math.inf)
    torch_output, _ = F.multi_head_attention_forward(
        x, x, x, 512, 8, attn.proj_weight, attn.proj_bias,
        bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.,
        out_proj_weight=attn.linear.weight, out_proj_bias=attn.linear.bias,
        training=False, key_padding_mask=padding_mask, attn_mask=attn_mask
    )

    assert torch.equal(output, self_output)
    assert torch.equal(torch_output, output)
