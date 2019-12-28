import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer_layer import TransformerLayer
from models.transformer_lm import TransformerLanguageModel


class SingleLayerTransformer(TransformerLanguageModel):
    """Transformer with a single layer with stochastic depth training.

    The number of layer (--num-layer) can be changed during eval or
    resumed training.
    Stochastic depth called LayerDrop in "Reducing Transformer Depth on
    Demand with Structured Dropout".
    """

    def __init__(self, vocab, args):
        self.eval_num_layer = args.num_layer if args.eval_num_layer is None else args.eval_num_layer
        super().__init__(vocab=vocab, args=args)

    def create_layer(self):
        self.layer = TransformerLayer(
            self.model_dim, self.num_head, self.inner_dim, self.dropout, self.attn_dropout,
            self.layer_dropout, self.head_dim, self.bias, self.activation
        )

    def transformer_stack(self, x, mask):
        for i in range(self.num_layer):
            x = self.layer(x, mask)
        return x


class LayerPermuteTransformer(TransformerLanguageModel):

    def __init__(self, vocab, args):
        self.pool_size = args.pool_size
        super().__init__(vocab=vocab, args=args)

    def create_layer(self):
        self.layer = nn.ModuleList([nn.ModuleList([]) for _ in range(len(self.pool_size))])
        for i in range(len(self.pool_size)):
            self.layer[i].extend([
                TransformerLayer(
                    self.model_dim, self.num_head, self.inner_dim, self.dropout, self.attn_dropout,
                    self.layer_dropout, self.head_dim, self.bias, self.activation)
                for _ in range(self.pool_size[i])
            ])

    def transformer_stack(self, x, mask):
        for i in range(len(self.pool_size)):
            if self.training:
                order = torch.randperm(self.pool_size[i])
            else:
                order = list(range(self.pool_size[i]))

            for idx in order:
                x = self.layer[i][idx](x, mask)
        return x


class LayerPoolTransformer(TransformerLanguageModel):

    def __init__(self, vocab, args):
        assert len(args.pool_size) == len(args.pool_depth)
        self.pool_size = args.pool_size
        self.pool_depth = args.pool_depth
        super().__init__(vocab=vocab, args=args)

    def create_layer(self):
        self.layer_pool = [nn.ModuleList([]) for _ in range(len(self.pool_size))]
        for i in range(len(self.pool_size)):
            self.layer_pool[i].extend([
                TransformerLayer(self.model_dim, self.num_head, self.inner_dim, self.dropout,
                                 self.attn_dropout, self.head_dim, self.bias, self.activation)
                for _ in range(self.pool_size[i])
            ])

    def transformer_stack(self, x, mask):
        for depth in self.pool_depth:
            pass  #TODO


