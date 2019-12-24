import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer_layer import TransformerLayer
from models.transformer_lm import TransformerLanguageModel


class SingleLayerTransformer(TransformerLanguageModel):

    def __init__(self, vocab, args):
        self.layer_dropout = args.layer_dropout
        self.eval_num_layer = args.num_layer if args.eval_num_layer is None else args.eval_num_layer
        super().__init__(vocab=vocab, args=args)

    def create_layer(self):
        # Replace the transformer layers
        self.layer = TransformerLayer(
            self.model_dim, self.num_head, self.inner_dim, self.dropout, self.attn_dropout,
            self.head_dim, self.bias, self.activation
        )

    def transformer_stack(self, x, mask):
        if self.training:
            rand = torch.rand(self.num_layer)
            for i in range(self.num_layer):
                if rand[i] >= self.layer_dropout:
                    x = self.layer(x, mask)
        else:
            for _ in range(self.eval_num_layer):
                x = self.layer(x, mask)
        return x
