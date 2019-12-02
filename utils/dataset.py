import math

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader


def split_list(lst, delimiter):
    """Split a list into sublist by a delimiter."""
    idx = [i + 1 for i, e in enumerate(lst) if e == delimiter]
    idx = idx if not lst and idx[-1] != len(lst) else idx[:-1]
    return [lst[i:j] for i, j in zip([0] + idx, idx + [None])]


def flatten(lst):
    """Flatten a two-dimension list to an one-dimension list."""
    return [item for sublist in lst for item in sublist]


def get_loader(data, vocab, args):
    merge = True if 'merge' in args.context_type else False
    if 'sent' in args.context_type:
        delimiter = vocab.get_idx('.')
        data = flatten([split_list(d, delimiter)for d in data])
    elif 'file' in args.context_type:
        data = [flatten(data)]

    if args.task == 'lm':
        dataset = LanguageModelIterableDataset(
            data, vocab, args.context_size, args.train_token, args.max_token,
            shuffle=args.shuffle, merge=merge, world_size=args.world_size, rank=args.rank
        )
    elif args.task == 'masked_lm':
        dataset = MaskedLanguageModelIterableDataset(
            data, vocab, args.train_token, args.max_token,
            proc_prob=args.proc_prob, mask_prob=args.mask_prob, rand_prob=args.rand_prob,
            shuffle=args.shuffle, merge=merge, world_size=args.world_size, rank=args.rank
        )
    else:
        raise NotImplementedError
    assert args.worker <= 1, "Multiple workers are not supported."
    return DataLoader(dataset, batch_size=None, num_workers=args.worker,
                      collate_fn=dataset.collate, pin_memory=args.cuda)


def get_eval_loader(data, vocab, args):
    merge = True if 'merge' in args.eval_context_type else False
    if 'sent' in args.eval_context_type:
        delimiter = vocab.get_idx('.')
        data = flatten([split_list(d, delimiter)for d in data])
    elif 'file' in args.eval_context_type:
        data = [flatten(data)]

    if args.task == 'lm':
        dataset = LanguageModelIterableDataset(
            data, vocab, args.eval_context_size, args.eval_token,
            args.eval_max_token, shuffle=False, merge=merge, world_size=1, rank=0
        )
    elif args.task == 'masked_lm':
        dataset = EvalMaskedLanguageModelIterableDataset(
            data, vocab, args.eval_token, args.eval_max_token,
            shuffle=False, merge=merge, world_size=1, rank=0
        )
    else:
        raise NotImplementedError
    assert args.worker <= 1, "Multiple workers are not supported."
    return DataLoader(dataset, batch_size=None, num_workers=args.worker,
                      collate_fn=dataset.collate, pin_memory=args.cuda)


def collate_data(data, pad_idx, left_pad=False):
    """Convert a batch of data from list to padded tensor format."""
    max_len = max(s.size(0) for s in data)
    max_len = ((max_len // 8) + 1) * 8  # Round length to multiple of 8.
    datatype = data[0].dtype
    padded_data = torch.ones((len(data), max_len), dtype=datatype).fill_(pad_idx)
    for i, s in enumerate(data):
        if not left_pad:
            padded_data[i][:len(s)] = s
        else:
            padded_data[i][max_len - len(s):] = s
    return padded_data


class LanguageModelIterableDataset(IterableDataset):
    """A IterableDataset for language model with token based batch size.

    Batch data for language modeling. Uses token count for batching,
    resulting in similar token count for each batch.

    Calling next() on this class will return a batch.

    usage:
    dataset = LanguageModelIterableDataset(params)
    dataloader = DataLoader(dataset, batch_size=None,
                            collate_fn=dataset.collate)
    for batch in dataloader:
        net_input = batch['feature']
        net_target = batch['target']
        # Happy language modeling

    Args:
        data: Two-dimensional list containing the tokens.
        vocab: The Vocab associated with the data.
        context_size: The context window length for each sample.
        max_token: Max token for training in each batch. Excludes
            tokens for context window.
        train_token: Max token for each sample in a batch. Excludes
            tokens for context window.
        world_size: The world size for distributed training. Use default
            value for CPU, or single GPU training.
        rank: The rank of process in distributed training. Use default
            value for CPU, or single GPU training.
    """

    def __init__(self, data, vocab, context_size, train_token, max_token, shuffle=False,
                 merge=False, world_size=1, rank=0, seed=0):
        assert max_token >= train_token
        self.total_target = sum([len(d) for d in data])
        self.data = data
        self.pad_idx = vocab.pad_idx
        self.bos_idx = vocab.bos_idx
        self.context_size = context_size
        self.max_token = max_token
        self.train_token = train_token
        self.shuffle = shuffle
        self.merge = merge
        self.world_size = world_size
        self.rank = rank
        self.seed = seed

    def get_data(self, *index):
        """Returns the processed data at index."""
        data = [self.bos_idx]
        for i in index:
            data += self.data[i]
        return torch.LongTensor(data)

    def create_sample(self, data):
        """Create samples of context size from a single data.

        Args:
            data: One-dimensional tensor containing a single sequence.
        Yields:
            feature: The input to produce output.
            target: The target of produced output.
            feature_len: Length of feature, including context window.
            num_target: Number of targets.
        """
        len_data = data.size(0)
        start = 0
        end = 0
        while end + 1 < len_data:
            # Extract input feature (including context window) and
            # target from data.
            end = min(len_data - 1, start + self.train_token)
            context_start = max(0, start - self.context_size)
            feature = data[context_start:end]
            target_padding = torch.ones(start - context_start, dtype=torch.long) * self.pad_idx
            target = torch.cat((target_padding, data[start + 1:end + 1]), dim=-1)
            num_target = end - start
            yield feature, target, num_target
            start = end

    def create_batch(self, indexes):
        """Create batch according to a max token count."""
        sample_lst = []
        token_count = 0
        # For splitting data among distributed processes.
        sample_num = 0
        # For merging context range.
        length = 0
        idx_lst = []
        for idx in indexes:
            if self.merge:
                # Merge context range within train_token length.
                length += len(self.data[idx])
                if length < self.train_token:
                    idx_lst.append(idx)
                    continue
                else:
                    data = self.get_data(*idx_lst)
                    idx_lst = [idx]
                    length = len(self.data[idx])
            else:
                data = self.get_data(idx)
            for sample in self.create_sample(data):
                token_count += len(sample[0])
                if token_count > self.max_token:
                    # Splitting data among multiprocess.
                    # The statement is always true for single process.
                    if sample_num % self.world_size == self.rank:
                        yield sample_lst
                        last_sample = sample_lst
                    sample_lst = [sample]
                    token_count = len(sample[0])
                    sample_num += 1
                else:
                    sample_lst.append(sample)
        if sample_num % self.world_size == self.rank:
            yield sample_lst
        # For distributed training, pad the processes with last complete sample.
        if sample_num % self.world_size < self.rank:
            yield last_sample

    def collate(self, batch):
        """Process a batch of samples to neural network input/target.

        Args:
            batch: A batch of data from create_batch().
        Returns:
            A prepared batch of input feature and target as described in
            class docstring.
            If context is all padding, e.g., each sample is a full
            sentence, then the context is removed.
        """
        feature, target, num_target = zip(*batch)
        feature = collate_data(feature, self.pad_idx)
        target = collate_data(target, self.pad_idx)

        # Convert to sequence first format.
        batch = {'feature': feature.permute(1, 0), 'target': target.permute(1, 0),
                 'num_target': num_target}
        return batch

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indexes = torch.randperm(len(self.data), generator=g).tolist()
        else:
            indexes = list(range(len(self.data)))
            # Sort by sequence length.
            # indexes.sort(key=lambda i: len(self.data[i]), reverse=True)
        return self.create_batch(indexes)

    def set_seed(self, seed: int):
        self.seed = seed


class MaskedLanguageModelIterableDataset(LanguageModelIterableDataset):

    def __init__(self, data, vocab, train_token, max_token, proc_prob=0.15, mask_prob=0.8, rand_prob=0.1,
                 shuffle=False, merge=False, world_size=1, rank=0, seed=0):
        # Context size is 0 for masked LM.
        super().__init__(data, vocab, 0, train_token, max_token, shuffle, merge, world_size, rank, seed)
        assert proc_prob <= 1
        assert mask_prob + rand_prob <= 1
        self.total_target *= proc_prob
        self.mask_idx = vocab.mask_idx
        self.vocab_size = len(vocab)
        self.proc_prob = proc_prob
        self.mask_prob = mask_prob
        self.rand_prob = rand_prob

    def create_sample(self, data):
        len_data = data.size(0)
        start = 0
        end = 0
        while end + 1 < len_data:
            # Extract input feature (including context window) and
            # target from data.
            end = min(len_data - 1, start + self.train_token)
            feature, target, num_target = self.mask_procedure(data[start:end])
            if num_target > 0:
                yield feature, target, num_target
            start = end

    def mask_procedure(self, data):
        feature = data.clone()
        target = torch.ones_like(data) * self.pad_idx
        num_target = 0
        for i in range(len(data)):
            if np.random.random() < self.proc_prob:
                num_target += 1
                target[i] = data[i]
                rand = np.random.random()
                if rand < self.mask_prob:
                    feature[i] = self.mask_idx
                elif rand < self.mask_prob + self.rand_prob:
                    rand_token = np.random.randint(0, self.vocab_size)
                    feature[i] = rand_token
        return feature, target, num_target


class EvalMaskedLanguageModelIterableDataset(LanguageModelIterableDataset):

    def __init__(self, data, vocab, train_token, max_token, shuffle=False, merge=False,
                 world_size=1, rank=0, seed=0):
        # Context size is 0 for masked LM.
        super().__init__(data, vocab, 0, train_token, max_token, shuffle, merge, world_size, rank, seed)
        self.mask_idx = vocab.mask_idx

    def create_sample(self, data):
        len_data = data.size(0)
        start = 0
        end = 0
        while end + 1 < len_data:
            # Extract input feature (including context window) and
            # target from data.
            end = min(len_data - 1, start + self.train_token)
            for i in range(end - start):
                feature = data[start:end].clone()
                target = torch.ones_like(feature) * self.pad_idx
                target[i] = feature[i]
                feature[i] = self.mask_idx
                yield feature, target, 1  # The num_target is 1
            start = end
