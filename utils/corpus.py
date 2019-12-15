import os
import pickle
from collections import Counter, OrderedDict

from tqdm import tqdm


def get_special(args):
    if args.task == 'lm':
        special = ['<bos>']
    elif args.task == 'masked_lm':
        special = ['<bos>', '<mask>']
    else:
        special = ['<bos>', '<eos>']
    return special


def get_corpus(args):
    special = get_special(args)
    vocab = Vocab(args.vocab_min_count, args.vocab_max_size, args.delimiter, special)
    if not os.path.isfile(args.vocab):
        print("|  Vocab file not found.\nCounting vocabs in data...", flush=True)
        vocab.count_files(args.train)
        print(f"Counted {len(vocab.counter)} unique tokens.", flush=True)
        vocab.save(args.vocab)
        print("|  Stored the vocab file in the data directory.", flush=True)
    else:
        vocab.load(args.vocab)
    vocab.build_vocab()
    print(f"|  Vocab size of {len(vocab.symbols)} from {len(vocab.special)} special tokens and "
          f"{len(vocab.counter)} unique tokens.", flush=True)

    corpus = Corpus(vocab)
    # Load corpus from cache or process the data files.
    if args.train_cache:
        print(f"|  Cached data found: training: True, validation: {args.valid_cache}", flush=True)
        print("|  Loading data from cache...", flush=True)
        corpus.load(args.cache_dir)
    else:
        print("|  Tokenizing data...", flush=True)
        corpus.process(train_file=args.train, valid_file=args.valid)
        corpus.save(args.cache_dir)
    return corpus


def get_eval_corpus(args):
    special = get_special(args)
    vocab = Vocab(args.vocab_min_count, args.vocab_max_size, special=special)
    vocab.load(args.vocab)
    vocab.build_vocab()
    print(f"|  Vocab size of {len(vocab.symbols)} from {len(vocab.special)} special tokens and "
          f"{len(vocab.counter)} unique tokens.", flush=True)
    corpus = Corpus(vocab)
    corpus.process(valid_file=args.valid, test_file=args.test)
    print(f"|  Done.", flush=True)
    return corpus


def write_file(text, path):
    with open(path, 'w+', encoding='utf-8') as file:
        for line in text:
            file.write(line + '\n')


class Vocab(object):
    """Stores the mapping between indexes and symbols.

    Use count_files() or load() to get vocab counts. Then build_vocab to
    finalize.

    Args:
    min_count: Minimum count to include in vocab.
    max_size: Maximum vocab size.
    delimiter: The string used to split each line into tokens.
    special: List of special symbols. Must be all lower case and
        enclosed with '<>'.
    """

    def __init__(self, min_count=0, max_size=None, delimiter=None, special=[]):
        self.counter = Counter()
        self.special = ['<pad>'] + special
        self.min_count = min_count
        self.max_size = max_size
        self.delimiter = delimiter
        self.symbols = []
        self.indexes = OrderedDict()
        self.unk_idx = -1

    def count_files(self, *files):
        """Count the unique tokens in files."""
        for filename in files:
            if filename is not None:
                print(f"Counting vocab in {filename}...", flush=True)
                with open(filename, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, unit=" line"):
                        tokens = self.tokenize(line)
                        self.counter.update(tokens)

    def tokenize(self, line):
        return line.strip().split(self.delimiter)

    def save(self, path):
        with open(path, 'w+', encoding='utf-8') as file:
            for key, value in self.counter.items():
                file.write(f"{key} {value}\n")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                item = line.split()
                self.counter[item[0].strip()] = int(item[1].strip())

    def add_special(self, symbol):
        if symbol not in self.indexes:
            self.symbols.append(symbol)
            self.indexes[symbol] = len(self.symbols) - 1
            special = symbol.strip('<>').lower()
            setattr(self, f'{special}_idx', self.indexes[symbol])

    def add_symbol(self, symbol):
        if symbol not in self.indexes:
            self.symbols.append(symbol)
            self.indexes[symbol] = len(self.symbols) - 1

    def build_vocab(self):
        for symbol in self.special:
            self.add_special(symbol)
        for symbol, count in self.counter.most_common(self.max_size):
            if count < self.min_count:
                break
            self.add_symbol(symbol)
        if '<unk>' in self.indexes:
            self.unk_idx = self.indexes['<unk>']

    def get_sym(self, idx):
        return self.symbols[idx]

    def get_idx(self, sym):
        if sym in self.indexes:
            return self.indexes[sym]
        else:
            return self.unk_idx

    def to_text(self, tensor):
        return " ".join([self.symbols[idx] if idx >= 0 else '<unk>' for idx in tensor])

    def to_index(self, text):
        return [self.get_idx(sym) for sym in self.tokenize(text)]

    def __eq__(self, other):
        return self.indexes == other.indexes

    def __len__(self):
        return len(self.symbols)

    def __contains__(self, item):
        return item in self.indexes

    def __getitem__(self, idx):
        return self.get_sym(idx)


class Corpus(object):
    """Collection of a dataset and its vocab."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.train = None
        self.valid = None
        self.test = None

    def process(self, train_file=None, valid_file=None, test_file=None):
        if train_file is not None:
            with open(train_file, 'r', encoding='utf-8') as train:
                self.train = [self.vocab.to_index(line) for line in train if len(line.strip()) > 0]
        if valid_file is not None:
            with open(valid_file, 'r', encoding='utf-8') as valid:
                self.valid = [self.vocab.to_index(line) for line in valid if len(line.strip()) > 0]
        if test_file is not None:
            with open(test_file, 'r', encoding='utf-8') as test:
                self.test = [self.vocab.to_index(line) for line in test if len(line.strip()) > 0]

    def save(self, path):
        """Saves the processed dataset to a directory."""
        dataset = ['train', 'valid', 'test']
        for data_name in dataset:
            data = getattr(self, data_name)
            if data is not None:
                with open(os.path.join(path, data_name + '.bin'), 'wb') as file:
                    pickle.dump(data, file)

    def load(self, path):
        """Load processed dataset from a directory."""
        dataset = ['train', 'valid', 'test']
        for data_name in dataset:
            filename = os.path.join(path, data_name + '.bin')
            if os.path.isfile(filename):
                with open(filename, 'rb') as file:
                    setattr(self, data_name, pickle.load(file))
