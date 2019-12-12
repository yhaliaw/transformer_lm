#!/usr/bin/env python3

import os
import sys
import math

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from utils.options import test_argparser

from utils.corpus import get_eval_corpus
from utils.dataset import get_eval_loader
from models import get_model
from train import valid_model

try:
    from apex import amp
    # Apex DistributedDataParallel does not work with adaptive embedding/softmax
    apex_import = True
except ImportError:
    apex_import = False


def main():
    parser = test_argparser()
    args = parser.parse_args()

    print(f"|  Arguments:\n{' '.join(sys.argv)}")
    args = valid_options(args)
    print(f"|  Processed args:\n{args}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cuda' if args.cuda else 'cpu')

    # Get data
    corpus = get_eval_corpus(args)

    # Setup model
    print("|  Loading model...")
    model = get_model(corpus.vocab, args)
    model.to(args.device)

    if args.fp16:
        print("|  Floating point 16 precision setting:\n", end='')
        model = amp.initialize(model, opt_level='O2')
    print("|  Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    print(f"|  epoch: {checkpoint['epoch']}, step: {checkpoint['step']}")

    def evaluate(dataloader):
        model.eval()
        total_loss = 0
        total_target = 0
        total = dataloader.dataset.total_target
        progress = tqdm(desc="Evaluating", total=total, unit=" token")
        for batch in dataloader:
            # TODO debug
            f = batch['feature'].data.numpy()
            t = batch['target'].data.numpy()
            n = batch['num_target']
            vocab = corpus.vocab
            # TODO test dataloading
            num_target = sum(batch['num_target'])
            total_target += num_target

            feature = batch['feature'].to(args.device)
            target = batch['target'].to(args.device)
            loss = model(feature, target)

            # log_prob = model.log_prob(feature)
            # pad_idx = -100 if corpus.vocab.pad_idx is None else corpus.vocab.pad_idx
            # loss = F.nll_loss(log_prob.view(-1, len(corpus.vocab)), target.view(-1),
            #                   ignore_index=pad_idx, reduction='sum')

            total_loss += loss.item()
            progress.update(num_target)
        progress.close()
        return total_loss / total_target

    if args.eval_valid:
        print("|  Evaluating model on validation data...")
        valid_loader = get_eval_loader(corpus.valid, corpus.vocab, args)
        loss = evaluate(valid_loader)
        loss = loss / math.log(2)
        print(f"|  valid loss: {loss:.4f}")
        print(f"\n== Valid perplexity: {2 ** loss:.4f} ==\n")

    print("|  Evaluating model on test data...")
    test_loader = get_eval_loader(corpus.test, corpus.vocab, args)
    loss = evaluate(test_loader)
    loss = loss / math.log(2)
    print(f"|  test loss: {loss:.4f}")
    print(f"\n== Test perplexity: {2 ** loss:.4f} ==\n")


def valid_options(args):
    # Device
    if args.cuda and not torch.cuda.is_available():
        print("== WARNING: CUDA devices are not found, switching to CPU. ==")
        args.cuda = False
    if args.fp16:
        if not args.cuda:
            print("== WARNING: fp16 requires CUDA, switching to fp32. ==")
            args.fp16 = False
        if not apex_import:
            print("== WARNING: Unable to import Nvidia Apex package, fallback to fp32. ==")
            args.fp16 = False

    # Data loading
    if args.eval_max_token is None:
        args.eval_max_token = args.max_token
    if args.eval_token is None:
        args.eval_token = args.train_token
    if args.eval_context_size is None:
        args.eval_context_size = args.context_size
    if args.eval_context_type is None:
        args.eval_context_type = args.context_type
    if args.eval_min_length is None:
        args.eval_min_length = args.min_length

    args.data = os.path.abspath(args.data)
    args.checkpoint = os.path.abspath(args.checkpoint)

    assert os.path.isfile(args.checkpoint)

    args.vocab = os.path.join(args.data, args.vocab)
    args.test = os.path.join(args.data, args.test)
    args.valid = os.path.join(args.data, args.valid)
    assert os.path.isfile(args.vocab)
    assert os.path.isfile(args.test)
    if args.eval_valid:
        assert os.path.isfile(args.valid)
    else:
        args.valid = None

    # Model
    args = valid_model(args)
    return args


if __name__ == '__main__':
    main()
