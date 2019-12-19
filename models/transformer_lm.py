import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.activation import gelu
from modules.positional_embedding import SinusoidalPositionalEmbedding
from modules.transformer_layer import TransformerLayer
from modules.adaptive_input import AdaptiveInput
from modules.adaptive_softmax import AdaptiveSoftmax
from modules.utils import subsequent_mask, SoftmaxLayer, TiedEmbedding


class TransformerLanguageModel(nn.Module):
    """Transformer for language modeling."""

    def __init__(self, vocab, args):
        super().__init__()
        self.vocab_size = len(vocab)
        self.padding_idx = vocab.pad_idx
        self.model_dim = args.embed_dim
        self.num_head = args.num_head
        self.head_dim = args.head_dim
        self.inner_dim = args.inner_dim
        self.dropout = args.dropout
        self.attn_dropout = args.attn_dropout
        self.num_layer = args.num_layer
        self.tied_layer = args.tied_layer  # Share all weight between layers
        self.input_cutoff = args.input_cutoff
        self.input_factor = args.input_factor
        self.softmax_cutoff = args.softmax_cutoff
        self.softmax_factor = args.softmax_factor
        self.activation = gelu if args.activation == 'gelu' else F.relu
        # No masked self attention for masked language model.
        self.self_attn_mask = False if args.task == 'masked_lm' else True
        self.bias = True

        self.position_embedding = SinusoidalPositionalEmbedding(self.model_dim)

        # Parameters
        if args.adaptive_input:
            self.embedding = AdaptiveInput(self.vocab_size, self.model_dim, self.input_cutoff,
                                           self.input_factor, self.padding_idx)
        else:
            self.embedding = TiedEmbedding(self.vocab_size, self.model_dim, self.padding_idx)
        self.layer = nn.ModuleList([])
        if self.tied_layer:
            transformer_layer = TransformerLayer(
                self.model_dim, self.num_head, self.inner_dim, self.dropout, self.attn_dropout,
                self.head_dim, self.bias, self.activation
            )
            self.layer.extend([transformer_layer for _ in range(self.num_layer)])
        else:
            self.layer.extend([
                TransformerLayer(self.model_dim, self.num_head, self.inner_dim, self.dropout,
                                 self.attn_dropout, self.head_dim, self.bias, self.activation)
                for _ in range(self.num_layer)
            ])
        if args.adaptive_softmax:
            self.adaptive_softmax_dropout = args.adaptive_softmax_dropout
            self.softmax = AdaptiveSoftmax(self.vocab_size, self.model_dim, self.softmax_cutoff,
                                           self.softmax_factor, self.padding_idx,
                                           self.adaptive_softmax_dropout)
        else:
            self.softmax = SoftmaxLayer(self.vocab_size, self.model_dim, self.padding_idx, self.bias)

        # Tied parameters
        if args.tied_adaptive_proj:
            for i in range(len(self.softmax.projection)):
                if self.softmax.projection[i] is not None:
                    self.softmax.projection[i] = self.embedding.projection[i]
        if args.tied_adaptive_embed:
            for i in range(len(self.softmax.linear)):
                self.softmax.linear[i] = self.embedding.embedding[i]
        if args.tied_embed:
            self.softmax.linear = self.embedding

    def extract_feature(self, x):
        # x: [seq x batch]
        seq_len, batch_size = x.size()

        # Padding mask
        # padding_mask: [batch x seq]
        padding_mask = x.eq(self.padding_idx)
        padding_mask = padding_mask.permute(1, 0)
        if self.self_attn_mask:  # Mask out subsequent positions.
            # self_attn_mask: [seq x seq]
            self_attn_mask = subsequent_mask(seq_len, x.device)
            # mask: [batch x seq x seq]
            # The first token can not be padding.
            mask = padding_mask[:, None, :] | self_attn_mask[None, :, :]
        else:
            mask = padding_mask[:, None, :]
        x = self.embedding(x) * (self.model_dim ** 0.5)
        x = x + self.position_embedding(torch.arange(1, seq_len + 1, device=x.device))[:, None, :]
        x = F.dropout(x, self.dropout, training=self.training)
        for encoder_layer in self.layer:
            x = encoder_layer(x, mask)
        return x

    def log_prob(self, x):
        x = self.extract_feature(x)
        log_prob = self.softmax.log_prob(x)
        return log_prob

    def forward(self, x, target):
        x = self.extract_feature(x)
        nll = self.softmax(x, target)
        return nll
