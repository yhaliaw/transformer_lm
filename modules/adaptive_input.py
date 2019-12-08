import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import TiedLinear, TiedEmbedding


class AdaptiveInput(nn.Module):
    """Adaptive input embeddings.

    The vocab must be ordered by usage frequency.

    See "Adaptive Input Representations for Neural Language Modeling"
    for details: https://arxiv.org/abs/1809.10853
    """

    def __init__(self, vocab_size, embed_dim, cutoff, factor=4, padding_idx=None, bias=False,
                 output_dim=None):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            for i in reversed(range(len(cutoff))):
                if cutoff[i] > vocab_size:
                    cutoff = cutoff[:i + 1]
                    cutoff[i] = vocab_size
            print(f"Re-adjust adaptive input cutoff to {cutoff}.")

        if output_dim is None:
            output_dim = embed_dim

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.cutoff = [0] + cutoff
        self.factor = factor
        self.bias = bias
        self.output_dim = output_dim

        # Parameters
        self.embedding = nn.ModuleList()
        self.projection = nn.ModuleList()
        # TransposedEmbedding and TransposedLinear allows for tied
        # projection and embedding.
        for i in range(0, len(self.cutoff) - 1):
            size = self.cutoff[i + 1] - self.cutoff[i]
            dim = self.embed_dim // (factor ** i)
            assert dim > 0, "Invalid factor for adaptive input embedding."
            if self.cutoff[i] <= padding_idx < self.cutoff[i + 1]:
                self.padding_group = i
                self.padding_idx = padding_idx - self.cutoff[i]
                self.embedding.append(TiedEmbedding(size, dim, self.padding_idx, bias))
            else:
                self.embedding.append(TiedEmbedding(size, dim, bias=bias))
            self.projection.append(TiedLinear(dim, output_dim, bias))

    def forward(self, x):
        # x can be batch first or sequence first
        size = x.size()
        dtype = self.projection[0].weight.dtype

        self.projection[0].weight.type()
        embed = torch.zeros((*size, self.output_dim), dtype=dtype, device=x.device)

        # Convert embedding of each cluster.
        for i in range(len(self.cutoff) - 1):
            mask = (x >= self.cutoff[i]) & (x < self.cutoff[i + 1])
            if not mask.any():
                continue
            symbol_idx = x[mask] - self.cutoff[i]

            emb = self.embedding[i](symbol_idx)
            proj = self.projection[i](emb)
            embed[mask] = proj
        return embed


if __name__ == '__main__':
    from fairseq.modules.adaptive_input import AdaptiveInput as FairseqAdaptiveInput
    embed_dim = 512
    vocab_size = 8000
    cutoff = [2500, 4500]
    factor = 4
    padding_idx = 10

    # Seq len: 50, batch size: 10
    x = torch.LongTensor(50, 10).random_(0, vocab_size - 1)

    adapt = AdaptiveInput(vocab_size=vocab_size, embed_dim=embed_dim, cutoff=cutoff,
                          factor=factor, padding_idx=padding_idx)
    fair_adapt = FairseqAdaptiveInput(vocab_size, padding_idx, embed_dim, factor, embed_dim, cutoff)

    # Copy the weights over.
    for i in range(len(cutoff) + 1):
        embed_weight, proj_weight = fair_adapt.weights_for_band(i)
        adapt.embedding[i].weight = embed_weight
        adapt.projection[i].weight = proj_weight

    # Test for same output
    output = adapt(x)
    fair_output = fair_adapt(x)

    assert torch.equal(output, fair_output)
