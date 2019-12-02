import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import Linear, TiedLinear, TiedEmbedding


class AdaptiveSoftmax(nn.Module):
    """Adaptive softmax with negative log likelihood.

    The vocab must be ordered by usage frequency.
    Outputs the negative log likelihood of correct class.

    See "Efficient softmax approximation for GPUs" for details:
    https://arxiv.org/abs/1609.04309
    """

    def __init__(self, vocab_size, embed_dim, cutoff, factor=4, padding_idx=None, dropout=0.,
                 bias=False, input_dim=None):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            for i in reversed(range(len(cutoff))):
                if cutoff[i] > vocab_size:
                    cutoff = cutoff[:i + 1]
                    cutoff[i] = vocab_size
            print(f"Re-adjust adaptive softmax cutoff to {cutoff}.")

        if input_dim is None:
            input_dim = embed_dim

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.cutoff = [0] + cutoff
        self.factor = factor
        self.padding_idx = padding_idx
        self.dropout = dropout
        self.bias = bias
        self.input_dim = input_dim

        # Parameters
        self.projection = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.cluster = Linear(embed_dim, len(self.cutoff) - 2, bias=bias)
        # TransposedEmbedding and TransposedLinear allows for tied
        # projection and embedding.
        for i in range(0, len(self.cutoff) - 1):
            size = self.cutoff[i + 1] - self.cutoff[i]
            dim = embed_dim // (factor ** i)
            assert dim > 0, "Invalid factor for adaptive softmax."
            self.projection.append(TiedLinear(dim, input_dim, bias) if input_dim != dim else None)
            self.linear.append(TiedEmbedding(size, dim, bias=bias, embed_init=False))

    def forward(self, x, target):
        """Calculate the sum negative log likelihood of targets."""
        # x, target can be batch first or sequence first
        x = x.reshape(-1, self.input_dim)
        x = F.dropout(x, self.dropout, training=self.training)
        target = target.reshape(-1)
        nll = 0
        # Get log prob of a cluster each loop
        for i in range(len(self.cutoff) - 1):
            mask = (self.cutoff[i] <= target) & (target < self.cutoff[i + 1])
            if self.padding_idx is not None:
                mask = mask & (target != self.padding_idx)
            if not mask.any():
                continue
            idx = mask.nonzero().view(-1)
            target_idx = target[idx] - self.cutoff[i]
            hidden = x[idx, :]
            head = self.linear[0](hidden, linear=True)
            cluster = self.cluster(hidden)
            head_log_prob = F.log_softmax(torch.cat((head, cluster), dim=-1), dim=-1, dtype=torch.float32)
            if i == 0:
                log_prob = head_log_prob.gather(1, target_idx[:, None]).view(-1)
                nll += -log_prob.sum()
            else:
                prior = head_log_prob[:, self.cutoff[1] - 1 + i]
                proj = self.projection[i](hidden, transposed=True)
                proj = F.dropout(proj, self.dropout, training=self.training)
                logit = self.linear[i](proj, linear=True)
                tail_log_prob = prior[:, None] + F.log_softmax(logit, dim=-1, dtype=torch.float32)
                log_prob = tail_log_prob.gather(1, target_idx[:, None]).view(-1)
                nll += -log_prob.sum()
        return nll

    def log_prob(self, x):
        """Calculate the log prob of all vocab."""
        # x can be batch first or sequence first
        size = x.size()[:-1]

        x = x.reshape(-1, self.embed_dim)
        x = F.dropout(x, self.dropout, training=self.training)
        log_prob = torch.ones((x.size(0), self.vocab_size), device=x.device) * math.inf  # TODO padding = 0 or inf?
        # Get log prob of each cluster
        head = self.linear[0](x, linear=True)
        cluster = self.cluster(x)
        head_log_prob = F.log_softmax(torch.cat((head, cluster), dim=-1), dim=-1, dtype=torch.float32)

        # Head cluster
        log_prob[:, :self.cutoff[1]] = head_log_prob[:, :self.cutoff[1]]
        # Tail cluster
        for i in range(1, len(self.cutoff) - 1):
            prior = head_log_prob[:, self.cutoff[1] + i - 1]
            proj = self.projection[i](x, transposed=True)
            proj = F.dropout(proj, self.dropout, training=self.training)
            logit = self.linear[i](proj, linear=True)
            tail_log_prob = prior[:, None] + F.log_softmax(logit, dim=-1, dtype=torch.float32)
            log_prob[:, self.cutoff[i]:self.cutoff[i + 1]] = tail_log_prob
        log_prob = log_prob.reshape(*size, self.vocab_size)
        return log_prob


if __name__ == '__main__':
    from fairseq.modules.adaptive_softmax import AdaptiveSoftmax as FairseqAdaptiveSoftmax
    embed_dim = 512
    vocab_size = 8000
    cutoff = [1000, 6500]
    factor = 4
    padding_idx = 220

    # Seq len: 50, batch size: 10
    x = torch.Tensor(50, 10, embed_dim).normal_(0., 1.)
    tgt = torch.LongTensor(50, 10).random_(0, vocab_size - 1)
    tgt[5, 3] = padding_idx

    adapt = AdaptiveSoftmax(vocab_size, embed_dim, cutoff, factor=4, padding_idx=padding_idx, dropout=0.,
                 bias=False, input_dim=None)
    fair_adapt = FairseqAdaptiveSoftmax(vocab_size, embed_dim, cutoff, dropout=0., factor=4.,
                                        adaptive_inputs=None, tie_proj=False)

    # Copy weight over
    adapt.linear[0].weight = nn.Parameter(fair_adapt.head.weight[:cutoff[0]])
    adapt.cluster.weight = nn.Parameter(fair_adapt.head.weight[cutoff[0]:])
    for i in range(len(cutoff)):
        adapt.projection[i + 1].weight = nn.Parameter(fair_adapt.tail[i][0].weight.t())
        adapt.linear[i + 1].weight = fair_adapt.tail[i][2].weight

    # Test same output
    output = adapt(x, tgt)

    lprob = adapt.log_prob(x)
    fair_lprob = fair_adapt.get_log_prob(x, None)

    fair_nll = F.nll_loss(fair_lprob.view(-1, vocab_size), tgt.view(-1), ignore_index=padding_idx, reduction='sum')
    nll = F.nll_loss(lprob.view(-1, vocab_size), tgt.view(-1), ignore_index=padding_idx, reduction='sum')

    assert torch.equal(nll, fair_nll)
    assert torch.equal(nll, output), f"log_prob: {nll}    forward:{output}"
