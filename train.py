#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import math
import itertools
import socket
import re
from shutil import copyfile, copytree, rmtree

import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.options import train_argparser
from utils.log import Logger
from utils.corpus import get_corpus
from utils.dataset import get_loader, get_eval_loader
from utils.optim import Adam, SGD, DecayingCosineAnnealingWarmRestarts
from models import get_model, reset_parameters, count_param

try:
    from apex import amp
    apex_import = True
except ImportError:
    apex_import = False


torch.backends.cudnn.benchmark = True


def main():
    parser = train_argparser()
    args = parser.parse_args()

    torch.set_num_threads(1)

    # Set working directory.
    # Under the --path dir, there will be a cache dir, and a dir named
    # with current time to store training results.
    args = setup_dir(args)

    print(f"|  Arguments:\n{' '.join(sys.argv)}", flush=True)

    # Validate options
    args = valid_options(args)
    print(f"|  Processed args:\n{args}", flush=True)

    if args.dist:
        setup_dist(args)
    else:
        args.world_size = 1

    # Manually set the seed.
    # Needed for distributed training and reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cuda' if args.cuda else 'cpu')

    # Get data
    corpus = get_corpus(args)

    # Start training.
    if args.dist:
        mp.spawn(setup_train, args=(corpus, args,), nprocs=args.gpu_per_node)
    else:
        setup_train(0, corpus, args)  # process index is 0 for single process training.


def setup_train(i, corpus, args):
    """Setup training.

    Handles CPU, single GPU, and distributed training.

    Args:
        i: The process index. Since one process per GPU, this is also
            the GPU index. For single GPU or CPU this is set to 0.
        corpus: The corpus for training.
        args: Arguments from argparse and main().
    """
    args.device = torch.device(args.device.type, i)

    # Find rank among all processes.
    args.rank = args.node_rank * args.gpu_per_node + i

    log = Logger(i, args.tensorboard_dir)
    log.train_add_text('arguments', str(args))
    log.valid_add_text('arguments', str(args))

    if args.dist:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size,
                                rank=args.rank)
        torch.cuda.set_device(args.rank)

    # Initialize model
    log("|  Loading model...")
    model = get_model(corpus.vocab, args)
    model.to(args.device)

    args.total_param = count_param(model)
    args.layer_param = count_param(model.layer)
    string = f"|  Model:\n{model}\n"
    string += f"|  Total parameters: {args.total_param}\n"
    string += f"|  Parameters without embedding and pre-softmax linear: {args.layer_param}"
    log(string)
    log.train_add_text('arguments', string)
    log.valid_add_text('arguments', string)

    # Create optimizer and scheduler.
    optimizer, scheduler = get_optimizer_scheduler(model, args)

    if args.fp16:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2', verbosity=False
        )
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[i], find_unused_parameters=True)

    resume_step = 0
    resume_epoch = 0
    if args.checkpoint is not None:
        log("|  Loading checkpoint...")
        if args.fp16:
            resume_step, resume_epoch = load_checkpoint(args.checkpoint, args.device, model,
                                                        optimizer, scheduler, amp)
        else:
            resume_step, resume_epoch = load_checkpoint(args.checkpoint, args.device, model,
                                                         optimizer, scheduler)

        def update_dropout(module):
            if hasattr(module, 'dropout'):
                model.dropout = args.dropout
            if hasattr(module, 'attn_dropout'):
                model.attn_dropout = args.attn_dropout
        model.apply(update_dropout)
    else:
        model.apply(reset_parameters)  # Initialize parameters

    # Get DataLoader
    train_loader = get_loader(corpus.train, corpus.vocab, args)
    if args.valid is not None:
        valid_loader = get_eval_loader(corpus.valid, corpus.vocab, args)

    log(f"|  Training on {socket.gethostname()} with rank {args.rank}.", True)

    def train(step, epoch, best_loss):
        model.train()
        optimizer.zero_grad()

        train_loader.dataset.set_seed(epoch)
        log.init_epoch(step, epoch, train_loader.dataset.total_target)
        epoch_loss = 0
        epoch_num_target = 0
        for batch_num, batch in enumerate(train_loader):
            # TODO debug
            f = batch['feature'].data.numpy()
            t = batch['target'].data.numpy()
            n = batch['num_target']
            vocab = corpus.vocab
            # TODO print out data to test
            # feat = np.transpose(f)
            # for data in feat:
            #     print(vocab.to_text(data))
            # continue
            # TODO test dataloading

            num_target = sum(batch['num_target'])
            epoch_num_target += num_target
            log.num_target += num_target
            log.batch_size += len(batch['num_target'])

            feature = batch['feature'].to(args.device)
            target = batch['target'].to(args.device)

            assert (target != vocab.pad_idx).sum() == num_target  # TODO remove debug check

            loss = model(feature, target)
            assert loss.dtype == torch.float32  # TODO remove debug check
            batch_loss = loss.item()
            epoch_loss += batch_loss
            log.loss += batch_loss

            loss = loss / num_target
            loss = loss / args.update_freq

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (batch_num + 1) % args.update_freq == 0:
                step, epoch, best_loss = update(step, epoch, best_loss)
                if args.max_step is not None and step >= args.max_step:
                    break

        # Remaining batches that doesn't fit in update freq.
        if not args.trim_batch and (batch_num + 1) % args.update_freq != 0:
            step, epoch, best_loss = update(step, epoch, best_loss)
        log.end_epoch(step, epoch)
        return step, epoch_loss / epoch_num_target, best_loss

    def update(step, epoch, best_loss):
        loss_scale = 1
        if args.fp16:
            loss_scale = amp._amp_state.loss_scalers[0]._loss_scale

        # Calculate norm of gradients. For logging.
        if args.log_norm:
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                norm = param.grad.data.float().norm().item() / loss_scale
                log.train_add_scalar('norm/' + name, norm, step)

        # Clip gradient
        if args.fp16:
            norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_norm)
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        log.norm += norm
        log.clip_norm += min(args.clip_norm, norm)

        optimizer.step()
        optimizer.zero_grad()

        step += 1
        if scheduler is not None:
            if step < args.warmup_step:
                # Linear warmup
                warmup_lr = args.lr * step / args.warmup_step
                optimizer.param_groups[0]['lr'] = warmup_lr
            else:
                scheduler.step()

        if step % args.log_freq == 0:
            lr = optimizer.param_groups[0]['lr']

            log.train(step, lr, loss_scale)

        if i == 0 and args.step_per_save != 0 and step % args.step_per_save == 0:
            path = os.path.join(args.checkpoint_dir, f'checkpoint-{epoch}-{step}.pt')
            save_checkpoint(path, step, epoch, model, optimizer, scheduler,
                            amp if args.fp16 else None)
            copyfile(path, os.path.join(args.checkpoint_dir, 'checkpoint_last.pt'))
        if args.dist:
            dist.barrier()

        if args.step_per_valid != 0 and step % args.step_per_valid == 0:
            # Eval on validation data.
            if args.valid is not None:
                best_loss = validate(best_loss)
        return step, epoch, best_loss

    def evaluate():
        model.eval()
        total_loss = 0
        total_target = 0
        total = valid_loader.dataset.total_target
        if i == 0:
            progress = tqdm(desc="Evaluating", total=total, unit=' token')
        for batch in valid_loader:
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

            total_loss += loss.item()
            if i == 0:
                progress.update(num_target)
        if i == 0:
            progress.close()
        return total_loss / total_target

    def validate(best_loss):
        with torch.no_grad():
            loss = evaluate()
        log.valid(loss, step, epoch)
        if i == 0 and best_loss > loss:
            best_loss = loss
            best_path = os.path.join(args.checkpoint_dir, 'checkpoint_best.pt')
            save_checkpoint(best_path, step, epoch, model, optimizer, scheduler,
                            amp if args.fp16 else None)
        if args.dist:
            dist.barrier()

        best_loss = best_loss / math.log(2)
        log.valid_add_scalar('best loss', best_loss, step)
        log.valid_add_scalar('best ppl', 2 ** best_loss, step)
        return best_loss

    step = resume_step
    best_loss = math.inf
    # Start from epoch 1 or resume from next epoch
    for epoch in itertools.count(resume_epoch + 1):
        # Train on training data.
        step, loss, best_loss = train(step, epoch, best_loss)
        if args.max_step is not None and step >= args.max_step:
            break

        if args.epoch_per_valid != 0 and epoch % args.epoch_per_valid == 0:
            # Eval on validation data.
            if args.valid is not None:
                if args.dist:
                    dist.barrier()
                best_loss = validate(best_loss)

        if args.epoch_per_save != 0 and epoch % args.epoch_per_save == 0:
            # Saving checkpoint.
            if i == 0:
                path = os.path.join(args.checkpoint_dir, f'checkpoint-{epoch}-{step}.pt')
                save_checkpoint(path, step, epoch, model, optimizer, scheduler,
                                amp if args.fp16 else None)
                copyfile(path, os.path.join(args.checkpoint_dir, 'checkpoint_last.pt'))
            if args.dist:
                dist.barrier()

        # Delete old checkpoints.
        if i == 0 and (args.keep_step is not None or args.keep_epoch is not None):
            for filename in os.listdir(args.checkpoint_dir):
                if re.match(r'checkpoint-\d+-\d+\.pt', filename):
                    file_epoch, file_step = re.split(r'[-.]', filename)[1:3]
                    if args.keep_step is not None and int(file_step) <= (step - args.keep_step):
                        os.remove(os.path.join(args.checkpoint_dir, filename))
                    if args.keep_epoch is not None and int(file_epoch) <= (epoch - args.keep_epoch):
                        os.remove(os.path.join(args.checkpoint_dir, filename))
        if args.dist:
            dist.barrier()

        if args.max_epoch is not None and epoch >= args.max_epoch:
            break


def save_checkpoint(path, step, epoch, model, optimizer=None, scheduler=None, amp=None):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    checkpoint = {'step': step, 'epoch': epoch, 'model': model.state_dict()}
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if amp is not None:
        checkpoint['amp'] = amp.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, device, model, optimizer=None, scheduler=None, amp=None):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if amp is not None:
        amp.load_state_dict(checkpoint['amp'])
    return checkpoint['step'], checkpoint['epoch']


def get_optimizer_scheduler(model, args):
    if args.optim == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, betas=args.adam_betas, eps=args.adam_eps,
                         weight_decay=args.weight_decay)
    elif args.optim == 'nag':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay, nesterov=True)
    elif args.optim == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay, nesterov=False)
    else:
        raise NotImplementedError

    if args.scheduler == 'cosine':
        scheduler = DecayingCosineAnnealingWarmRestarts(
            optimizer, T_0=args.step_per_period, T_mult=args.period_factor, eta_min=args.min_lr,
            decay_factor=args.period_decay
        )
    elif args.scheduler == 'inv_sqrt':
        def inv_sqrt(step):
            return 1. / (step ** 0.5) if step != 0 else 1.  # The warmup is handled in training loop.
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=inv_sqrt)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.decay_factor, patience=args.patience, min_lr=args.min_lr,
            verbose=True
        )
    elif args.scheduler == 'constant':
        scheduler = None
    else:
        raise NotImplementedError

    return optimizer, scheduler


def valid_options(args):
    # Device
    if args.cuda and not torch.cuda.is_available():
        print("== WARNING: CUDA devices are not found, switching to CPU training. ==", flush=True)
        args.cuda = False
    if args.fp16:
        if not args.cuda:
            print("== WARNING: fp16 requires CUDA, switching to fp32. ==", flush=True)
            args.fp16 = False
        if not apex_import:
            print("== WARNING: Unable to import Nvidia Apex package, fallback to fp32. ==", flush=True)
            args.fp16 = False
    if args.gpu_per_node > 0:
        args.dist = True
    if args.dist:
        if not args.cuda:
            print("== WARNING: Multi-GPU training requires CUDA. ==", flush=True)
            args.dist = False
        elif args.gpu_per_node == 0:
            # Auto-detect number of GPU.
            args.gpu_per_node = torch.cuda.device_count()

    # Optimizer
    assert args.lr is not None, "A learning rate is needed."
    if args.clip_norm == 0.:
        args.clip_norm = math.inf

    # Data loading
    if args.eval_max_token is None:
        args.eval_max_token = args.max_token
    if args.eval_token is None:
        args.eval_token = args.train_token
    if args.eval_context_size is None:
        args.eval_context_size = args.context_size
    if args.eval_context_type is None:
        args.eval_context_type = args.context_type

    # Model
    args = valid_model(args)
    return args


def valid_model(args):
    # Adaptive input embedding and Adaptive softmax
    if args.cutoff is not None:
        args.input_cutoff = args.cutoff
        args.softmax_cutoff = args.cutoff
    if args.factor is not None:
        args.input_factor = args.factor
        args.softmax_factor = args.factor
    if args.tied_adaptive:
        args.tied_adaptive_proj = True
        args.tied_adaptive_embed = True
    if args.tied_adaptive_proj or args.tied_adaptive_embed:
        args.adaptive_input = True
        args.adaptive_softmax = True
        assert args.input_cutoff == args.softmax_cutoff
        assert args.input_factor == args.softmax_factor
    if args.adaptive_input:
        assert args.input_cutoff is not None
        assert args.tied_embed is False
    if args.adaptive_softmax:
        assert args.softmax_cutoff is not None
        assert args.tied_embed is False
    if args.tied_embed:
        assert args.adaptive_input is False
        assert args.adaptive_softmax is False
        assert args.tied_adaptive is False
    return args


def setup_dist(args):
    # To make DataLoader work with DistributedDataParallel (See Doc).
    mp.set_start_method('spawn')
    args.world_size = args.gpu_per_node * args.nodes
    os.environ['WORLD_SIZE'] = str(args.world_size)

    # Apply options to missing environment variables.
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(args.node_rank)
    else:
        args.rank = os.environ['RANK']
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = args.address
    else:
        args.address = os.environ['MASTER_ADDR']
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(args.port)
    else:
        args.port = os.environ['MASTER_PORT']
    return args


def setup_dir(args):

    def create_dir(path):
        if not os.path.isdir(path):
            os.makedirs(path)

    args.data = os.path.abspath(args.data)
    args.path = os.path.abspath(args.path)

    args.cache_dir = os.path.join(args.path, 'cache', args.task)
    if args.run_name is None:
        args.work_dir = os.path.join(args.path, time.strftime('%Y%m%d-%H%M%S'))
    else:
        args.work_dir = os.path.join(args.path, args.run_name)
    args.tensorboard_dir = None
    create_dir(args.cache_dir)
    create_dir(args.work_dir)
    if args.tensorboard:
        args.tensorboard_dir = os.path.join(args.work_dir, 'tensorboard')
        create_dir(args.tensorboard_dir)
    args.checkpoint_dir = os.path.join(args.work_dir, 'checkpoint')
    create_dir(args.checkpoint_dir)

    if args.checkpoint is not None:
        args.checkpoint = os.path.abspath(args.checkpoint)
        assert os.path.isfile(args.checkpoint)

    # Save current version of scripts.
    root = os.path.dirname(sys.argv[0])
    script_path = os.path.join(args.work_dir, 'scripts')
    if os.path.exists(script_path):
        rmtree(script_path)
    for directory in ('models', 'modules', 'utils'):
        copytree(os.path.join(root, directory),
                 os.path.join(script_path, directory))
    for file in ('train.py', 'eval.py'):
        copyfile(os.path.join(root, file),
                 os.path.join(script_path, file))

    # Check for data files.
    # Check if train, valid, test files exist and whether a binary version is cached.
    args.vocab = os.path.join(args.data, args.vocab)
    for name in ('train', 'valid'):
        file = os.path.join(args.data, getattr(args, name))
        setattr(args, name, file)
        cache_file = name + '_cache'
        setattr(args, cache_file, False)
        if not os.path.isfile(file):
            setattr(args, name, None)
        if os.path.isfile(os.path.join(args.cache_dir, name + '.bin')):
            setattr(args, cache_file, True)
    assert args.train is not None
    # Spawn a process to copy stdout to file.
    args.log_file = os.path.join(args.work_dir, 'log.txt')
    spawn_log_file_process(args.log_file)
    return args


def spawn_log_file_process(log_path):
    """Spawn a process to copy stdout to file."""
    tee = subprocess.Popen(['tee', '-a', log_path], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())


if __name__ == '__main__':
    main()
