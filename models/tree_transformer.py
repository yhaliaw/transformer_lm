import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_lm import TransformerLanguageModel
from modules.activation import gelu
from modules.layer_norm import LayerNorm
from modules.multihead_attention import MultiheadAttention
from modules.constituent_attention import ConstituentAttention
from modules.transformer_layer import PositionWiseFeedForward, Sublayer


class TreeTransformerLayer(nn.Module):

    def __init__(self, model_dim, num_head, inner_dim, dropout, attn_dropout=0., head_dim=None,
                 bias=False, activation=gelu):
        super().__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.dropout = dropout
        self.bias = bias

        # Parameters
        self.layer_norm = LayerNorm(model_dim)
        self.constituent = ConstituentAttention(model_dim, model_dim, bias)
        self.multi_head = MultiheadAttention(model_dim, num_head, attn_dropout, bias, head_dim)
        position_wise = PositionWiseFeedForward(
            model_dim, inner_dim, bias=bias, dropout=dropout, activation=activation
        )
        self.position_wise = Sublayer(position_wise, model_dim, dropout)

    @staticmethod
    def apply_weight(attn_weight, weight):
        return attn_weight * weight

    def forward(self, x, prior, padding_mask):
        residual = x
        x = self.layer_norm(x)
        weight, prior = self.constituent(x, prior, padding_mask)
        x = self.multi_head(x, attn_mask=padding_mask,
                            attn_hook=lambda attn: self.apply_weight(attn, weight))
        x = residual + F.dropout(x, self.dropout, training=self.training)

        x = self.position_wise(x)
        return x, prior


class TreeTransformer(TransformerLanguageModel):

    def __init__(self, vocab, args):
        assert args.task == 'masked_lm'
        super().__init__(vocab, args)

        # Parameters
        # Replace transformer layer with tree transformer layer.
        self.layer = nn.ModuleList([])
        self.layer.extend([
            TreeTransformerLayer(self.model_dim, self.num_head, self.inner_dim, self.dropout,
                                 self.attn_dropout, self.head_dim, self.bias, self.activation)
            for _ in range(self.num_layer)
        ])

    def extract_feature(self, x):
        # x: [seq x batch]
        seq_len, batch_size = x.size()

        # Padding mask
        # padding_mask: [batch x seq]
        padding_mask = x.eq(self.padding_idx)
        padding_mask = padding_mask.permute(1, 0)[:, None, :]

        x = self.embedding(x) * (self.model_dim ** 0.5)
        x = x + self.position_embedding(torch.arange(1, seq_len + 1, device=x.device))[:, None, :]
        x = F.dropout(x, self.dropout, training=self.training)
        prior = 0
        for encoder_layer in self.layer:
            x, prior = encoder_layer(x, prior, padding_mask)
        return x
