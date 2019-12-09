import os
import time
import math

import numpy as np
from tqdm import tqdm


class Logger(object):

    def __init__(self, rank, tensorboard_dir=None):
        self.rank = rank
        self.tensorboard = False

        if tensorboard_dir is not None:
            try:
                from tensorboardX import SummaryWriter
                self.tensorboard = True
                self.train_writer = SummaryWriter(logdir=os.path.join(tensorboard_dir, 'train'))
                self.valid_writer = SummaryWriter(logdir=os.path.join(tensorboard_dir, 'valid'))
            except ImportError:
                self.log("== WARNING: Unable to import tensorboardX package. Not writing tensorboard logs. ==")

        self.time = 0
        self.batch_size = 0
        self.loss = 0
        self.num_target = 0
        self.norm = 0
        self.clip_norm = 0
        self.output_time = 0
        self.last_step = 0

        self.epoch_num_step = 0
        self.epoch_time = 0
        self.epoch_loss = 0
        self.epoch_num_target = 0
        self.epoch_batch_size = 0
        self.epoch_num_clip = 0

    def __call__(self, *args, **kwargs):
        return self.log(*args, **kwargs)

    def log(self, string, force=False):
        if self.rank == 0 or force:
            print(string + '\n', end='', flush=True)

    def valid_add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self.rank == 0 and self.tensorboard:
            self.valid_writer.add_scalar(tag, scalar_value, global_step, walltime)

    def valid_add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        if self.rank == 0 and self.tensorboard:
            self.valid_writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def valid_add_text(self, tag, text_string, global_step=None, walltime=None):
        if self.rank == 0 and self.tensorboard:
            self.valid_writer.add_text(tag, text_string, global_step, walltime)

    def valid(self, loss, step, epoch):
        loss = loss / math.log(2)
        self.valid_add_scalar('loss', loss, step)
        self.log(f"|  Validation at {epoch} epoch, {step} step, loss: {loss:.4f}"
                 f", perplexity: {np.square(loss):.4f}")

    def train_add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self.rank == 0 and self.tensorboard:
            self.train_writer.add_scalar(tag, scalar_value, global_step, walltime)

    def train_add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        if self.rank == 0 and self.tensorboard:
            self.train_writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def train_add_text(self, tag, text_string, global_step=None, walltime=None):
        if self.rank == 0 and self.tensorboard:
            self.train_writer.add_text(tag, text_string, global_step, walltime)

    def train(self, step, lr, loss_scale):
        end_time = time.time()
        num_step = step - self.last_step

        loss = self.loss / self.num_target / math.log(2)
        perplexity = np.square(loss)
        norm = self.norm / num_step
        clip_norm = self.clip_norm / num_step
        time_per_step = (end_time - self.time) / num_step
        token_per_step = self.num_target / num_step
        token_per_sec = self.num_target / (end_time - self.time)
        batch_size = self.batch_size / num_step
        self.train_add_scalar('loss', loss, step)
        self.train_add_scalar('time per step', time_per_step, step)
        self.train_add_scalar('token per step', token_per_step, step)
        self.train_add_scalar('batch size', batch_size, step)
        self.train_add_scalar('norm', norm, step)
        self.train_add_scalar('clip norm', clip_norm, step)
        self.train_add_scalar('learning rate', lr, step)
        self.train_add_scalar('loss scale', loss_scale, step)

        if self.rank == 0:
            string = f"{step} step, " \
                     f"size: {batch_size:.2f}, token: {token_per_step:.2f}, " \
                     f"time: {time_per_step:.2f}, tkn/s: {token_per_sec:.2f}, " \
                     f"norm: {norm:.2f}, " \
                     f"lr: {lr:.9f}, scale: 2^{math.log(loss_scale, 2):.0f}, " \
                     f"loss: {loss:.4f} ppl: {perplexity:.4f}"
            self.progress_bar.set_postfix_str(string, refresh=False)
            self.progress_bar.update(self.num_target)

        self.epoch_num_step += num_step
        self.epoch_time += end_time - self.time
        self.epoch_loss += self.loss
        self.epoch_num_target += self.num_target
        self.epoch_batch_size += self.batch_size
        self.epoch_num_clip += norm > clip_norm

        self.time = end_time
        self.loss = 0
        self.num_target = 0
        self.batch_size = 0
        self.norm = 0
        self.clip_norm = 0
        self.last_step = step

    def end_epoch(self, step, epoch):
        if self.rank == 0:
            self.progress_bar.close()

        loss = self.epoch_loss / self.epoch_num_target / math.log(2)
        perplexity = np.square(loss)
        time_per_step = self.epoch_time / self.epoch_num_step
        token_per_step = self.epoch_num_target / self.epoch_num_step
        token_per_sec = self.epoch_num_target / self.epoch_time
        batch_size = self.epoch_batch_size / self.epoch_num_step
        clip_ratio = self.epoch_num_clip / self.epoch_num_step

        string = f"|  {self.rank} rank, {epoch} epoch, {step} step, " \
                 f"size: {batch_size:.2f}, token: {token_per_step:.2f}, " \
                 f"time: {time_per_step:.2f}, tkn/s: {token_per_sec:.2f}, " \
                 f"clip: {clip_ratio:.2f}, " \
                 f"loss: {loss:.4f} ppl: {perplexity:.4f}"
        print(string, flush=True)

    def init_epoch(self, step, epoch, total):
        if self.rank == 0:
            self.progress_bar = tqdm(
                desc=f"| Epoch {epoch}", total=total, unit='',
                bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]{postfix}'
            )

        self.last_step = step
        self.time = time.time()
        self.output_time = self.time

        self.epoch_num_step = 0
        self.epoch_time = 0
        self.epoch_loss = 0
        self.epoch_num_target = 0
        self.epoch_batch_size = 0
        self.epoch_num_clip = 0
