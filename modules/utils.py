import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def subsequent_mask(size, device):
    """Mask for masked self attention."""
    return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), 1)


class Linear(nn.Module):
    """Linear layer."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def extra_repr(self):
        return f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"


class SoftmaxLayer(nn.Module):
    """Log softmax with linear layer."""

    def __init__(self, vocab_size, hidden_dim, padding_idx, bias):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx
        self.bias = bias

        # Parameters
        self.linear = TiedEmbedding(vocab_size, hidden_dim, padding_idx, bias, embed_init=False)

    def forward(self, x, target):
        """Calculate the mean negative log likelihood of targets."""
        target = target.reshape(-1)
        log_prob = self.log_prob(x)
        nll = F.nll_loss(log_prob.view(-1, self.vocab_size), target,
                         ignore_index=self.padding_idx, reduction='sum')
        return nll

    def log_prob(self, x):
        """Calculate the log prob of all vocab."""
        x = self.linear(x, linear=True)
        log_prob = F.log_softmax(x, dim=-1, dtype=torch.float32)
        return log_prob


class TiedLinear(nn.Module):
    """Linear layer that can be transposed."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.in_bias = Parameter(torch.Tensor(in_features))
            self.out_bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('in_bias', None)
            self.register_parameter('out_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias:
            nn.init.constant_(self.in_bias, 0.)
            nn.init.constant_(self.out_bias, 0.)

    def forward(self, x, transpose=False):
        weight = self.weight if not transpose else self.weight.t()
        bias = self.out_bias if not transpose else self.in_bias
        return F.linear(x, weight, bias)

    def extra_repr(self):
        return f"{self.in_features}, {self.out_features}, bias={self.bias}"


class TiedEmbedding(nn.Module):
    """Embedding layer that can be transposed as a linear layer."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, bias=True,
                 embed_init=True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            assert padding_idx >= 0
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings
        self.padding_idx = padding_idx
        self.embed_init = embed_init  # Embedding method for weight initialization
        self.bias = bias

        # Parameters
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        if bias:
            self.linear_bias = Parameter(torch.Tensor(num_embeddings))
        else:
            self.register_parameter('linear_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.embed_init:
            nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.weight[self.padding_idx].fill_(0)
        else:
            nn.init.xavier_uniform_(self.weight)
        if self.linear_bias is not None:
            nn.init.constant_(self.linear_bias, 0.)

    def forward(self, x, linear=False):
        return F.linear(x, self.weight) if linear else \
            F.embedding(x, self.weight, self.padding_idx)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.embedding_dim}, " \
               f"bias={self.linear_bias is not None}, padding_idx={self.padding_idx}"
