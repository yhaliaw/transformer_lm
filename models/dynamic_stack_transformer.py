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


class SharedLayerTransformer(TransformerLanguageModel):

    def __init__(self, vocab, args):
        self.pool_depth = args.pool_depth
        super().__init__(vocab=vocab, args=args)

    def create_layer(self):
        self.layer = nn.ModuleList([])
        self.layer.extend([
            TransformerLayer(
                self.model_dim, self.num_head, self.inner_dim, self.dropout, self.attn_dropout,
                self.layer_dropout, self.head_dim, self.bias, self.activation)
            for _ in range(len(self.pool_depth))
        ])

    def transformer_stack(self, x, mask):
        for transformer_layer, depth in zip(self.layer, self.pool_depth):
            for _ in range(depth):
                x = transformer_layer(x, mask)
        return x


class LayerPermuteTransformer(TransformerLanguageModel):

    def __init__(self, vocab, args):
        self.pool_size = args.pool_size
        self.permute_ensemble = args.permute_ensemble
        self.permute_random = args.permute_random
        self.permute_order = args.permute_order
        if self.permute_order is not None:
            self.permute_order = list(map(lambda x: [int(i) for i in x.split(',')], self.permute_order))
        super().__init__(vocab=vocab, args=args)

    def create_layer(self):
        self.layer = nn.ModuleList([nn.ModuleList([]) for _ in range(len(self.pool_size))])
        for i in range(len(self.pool_size)):
            self.layer[i].extend([
                TransformerLayer(
                    self.model_dim, self.num_head, self.inner_dim, self.dropout, self.attn_dropout,
                    self.layer_dropout, self.head_dim, self.bias, self.activation
                )
                for _ in range(self.pool_size[i])
            ])

    def transformer_stack(self, x, mask):
        if not self.training and self.permute_ensemble:
            return self.ensemble_stack(x, mask)

        for i in range(len(self.pool_size)):
            if self.training or self.permute_random:
                order = torch.randperm(self.pool_size[i])
            elif self.permute_order is not None:
                order = self.permute_order[i]
            else:
                order = list(range(self.pool_size[i]))

            for idx in order:
                x = self.layer[i][idx](x, mask)
        return x

    def ensemble_stack(self, x, mask):
        with torch.no_grad():
            ensemble_layer = TransformerLayer(
                self.model_dim, self.num_head, self.inner_dim, self.dropout, self.attn_dropout,
                self.layer_dropout, self.head_dim, self.bias, self.activation
            ).to(x.device).type(x.dtype)
            for layer_pool in self.layer:
                num = len(layer_pool)
                state_dict = list(map(lambda x: dict(x.named_parameters()), layer_pool))
                for name, param in ensemble_layer.named_parameters():
                    param = sum(map(lambda x: x[name], state_dict)) / num  # average
                for _ in range(num):
                    x = ensemble_layer(x, mask)
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


