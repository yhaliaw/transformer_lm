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
    if args.task == 'lm':
        dataset = LanguageModelDataset(
            data, vocab, args, training=True, world_size=args.world_size, rank=args.rank
        )
    elif args.task == 'masked_lm':
        dataset = MaskedLanguageModelDataset(
            data, vocab, args, training=True, world_size=args.world_size, rank=args.rank
        )
    else:
        raise NotImplementedError
    assert args.worker <= 1, "Multiple workers are not supported."
    return DataLoader(dataset, batch_size=None, num_workers=args.worker,
                      collate_fn=dataset.collate, pin_memory=args.cuda)


def get_eval_loader(data, vocab, args):
    if args.task == 'lm':
        dataset = LanguageModelDataset(
            data, vocab, args, training=False, world_size=1, rank=0
        )
    elif args.task == 'masked_lm':
        dataset = EvalMaskedLanguageModelDataset(
            data, vocab, args, world_size=1, rank=0
        )
    else:
        raise NotImplementedError
    assert args.worker <= 1, "Multiple workers are not supported."
    return DataLoader(dataset, batch_size=None, num_workers=args.worker,
                      collate_fn=dataset.collate, pin_memory=args.cuda)


def collate_data(data, pad_idx, left_pad=False):
    """Convert a batch of data from list to padded tensor format."""
    max_len = max(s.size(0) for s in data)
    max_len = math.ceil(max_len / 8) * 8  # Round length to multiple of 8.
    datatype = data[0].dtype
    padded_data = torch.ones((len(data), max_len), dtype=datatype).fill_(pad_idx)
    for i, s in enumerate(data):
        if not left_pad:
            padded_data[i][:len(s)] = s
        else:
            padded_data[i][max_len - len(s):] = s
    return padded_data


class LanguageModelDataset(IterableDataset):
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
        max_token: Max token for training in each batch. Includes
            tokens for context window.
        train_token: Max token for each sample in a batch. Excludes
            tokens for context window.
        world_size: The world size for distributed training. Use default
            value for CPU, or single GPU training.
        rank: The rank of process in distributed training. Use default
            value for CPU, or single GPU training.
    """

    def __init__(self, data, vocab, args, training=True, world_size=1, rank=0, seed=0):
        assert args.max_token >= args.train_token
        self.total_target = sum([len(d) for d in data])
        self.pad_idx = vocab.pad_idx
        self.bos_idx = vocab.bos_idx
        self.context_size = args.context_size
        self.max_token = args.max_token
        self.train_token = args.train_token
        self.context_type = args.context_type
        self.shuffle = args.shuffle
        self.trim_data = args.trim_data
        self.min_length = args.min_length
        self.world_size = world_size
        self.rank = rank
        self.seed = seed

        if not training:
            self.shuffle = False
            self.min_length = 0

        if 'sent' in self.context_type:
            if '.' in vocab:
                delimiter = vocab.get_idx('.')
                data = flatten([split_list(d, delimiter) for d in data])
        elif 'file' in self.context_type:
            data = [flatten(data)]

        if 'merge' in self.context_type:
            merged_data = []
            length = 0
            sample = []
            for ele in data:
                if length + len(ele) < self.train_token:  # Leave one out for bos token.
                    sample += ele
                    length += len(ele)
                else:
                    merged_data.append(sample)
                    sample = ele
                    length = len(ele)
            merged_data.append(sample)
            data = merged_data

        self.sample = self.get_sample(data)

    def get_sample(self, data):
        sample = []
        for ele in data:
            for s in self.create_sample(self.process_data(ele)):
                if s[0] >= self.min_length:
                    sample.append(s)
        return sample

    def process_data(self, data):
        data = [self.bos_idx] + data
        return torch.LongTensor(data)

    def create_sample(self, data):
        """Create samples of context size from a single data.

        Args:
            data: One-dimensional tensor containing a single sequence.
        Yields:
            length: Length of feature, including context window.
            feature: The input to produce output.
            target: The target of produced output.
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
            length = end - context_start
            yield length, feature, target, num_target
            start = end

    def process_sample(self, sample):
        return sample

    def create_batch(self, idx):
        length = 0
        batch = []
        batch_num = 0
        for i in idx:
            sample = self.process_sample(self.sample[i])
            if length <= self.max_token * self.world_size:
                batch.append(sample)
                length += sample[0]
            else:
                # Split batch among ranks
                yield batch[0 + self.rank::self.world_size]

                batch = [sample]
                length = sample[0]
                batch_num += 1
        # Remaining data.
        # If not trimming data or a full sized batch, then yield the batch.
        if length == self.max_token * self.world_size or not self.trim_data:
            yield batch[0 + self.rank::self.world_size]

    def __len__(self):
        return len(self.sample)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indexes = torch.randperm(len(self.sample), generator=g).tolist()
        else:
            indexes = list(range(len(self.sample)))
            # Sort by sequence length.
            indexes.sort(key=lambda i: self.sample[i][0])
        return self.create_batch(indexes)

    def set_seed(self, seed: int):
        self.seed = seed

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
        _, feature, target, num_target = zip(*batch)  # Length is not used in training.
        feature = collate_data(feature, self.pad_idx)
        target = collate_data(target, self.pad_idx)

        # Convert to sequence first format.
        batch = {'feature': feature.permute(1, 0), 'target': target.permute(1, 0),
                 'num_target': num_target}
        return batch


class MaskedLanguageModelDataset(LanguageModelDataset):

    def __init__(self, data, vocab, args, training=True, world_size=1, rank=0, seed=0):
        # Context size is 0 for masked LM.
        assert args.context_size == 0
        assert args.proc_prob <= 1
        assert args.mask_prob + args.rand_prob <= 1
        self.mask_idx = vocab.mask_idx
        self.vocab_size = len(vocab)
        self.proc_prob = args.proc_prob
        self.mask_prob = args.mask_prob
        self.rand_prob = args.rand_prob

        super().__init__(data, vocab, args, training, world_size, rank, seed)
        self.total_target *= args.proc_prob

    def create_sample(self, data):
        len_data = data.size(0)
        start = 0
        end = 0
        while end < len_data:
            # Extract input feature (including context window) and
            # target from data.
            end = min(len_data, start + self.train_token)
            yield end - start, data[start:end]
            start = end

    def process_sample(self, sample):
        return self.mask_procedure(sample[1])

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
        return feature.size(0), feature, target, num_target


class EvalMaskedLanguageModelDataset(LanguageModelDataset):

    def __init__(self, data, vocab, args, world_size=1, rank=0, seed=0):
        # Context size is 0 for masked LM.
        assert args.context_size == 0
        self.mask_idx = vocab.mask_idx

        super().__init__(data, vocab, args, False, world_size, rank, seed)

    def create_sample(self, data):
        len_data = data.size(0)
        start = 0
        end = 0
        while end < len_data:
            # Extract input feature (including context window) and
            # target from data.
            end = min(len_data, start + self.train_token)
            yield end - start, data[start:end]
            start = end

    def create_batch(self, idx):
        length = 0
        batch = []
        batch_num = 0
        for i in idx:
            for sample in self.process_sample(self.sample[i]):
                if length <= self.max_token * self.world_size:
                    batch.append(sample)
                    length += sample[0]
                else:
                    # Split batch among ranks
                    yield batch[0 + self.rank::self.world_size]

                    batch = [sample]
                    length = sample[0]
                    batch_num += 1
        # Remaining data.
        # If not trimming data or a full sized batch, then yield the batch.
        if length == self.max_token * self.world_size or not self.trim_data:
            yield batch[0 + self.rank::self.world_size]

    def process_sample(self, sample):
        # sample[0] is length, sample[1] is the data.
        for i in range(1, sample[0]):
            feature = sample[1].clone()
            target = torch.ones_like(feature) * self.pad_idx
            target[i] = feature[i]
            feature[i] = self.mask_idx
            yield feature.size(0), feature, target, 1  # The num_target is 1
