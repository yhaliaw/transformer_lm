#!/usr/bin/env bash

set -e

ROOT=$(dirname "$(readlink -f "$BASH_SOURCE")")

if [ "$1" == "train" ]; then
  echo "Training masked language model on penn data..."
  python3 train.py --data data/penn \
  --train train.txt --valid valid.txt \
  --path workspace/penn --tensorboard \
  --task masked_lm --context-type line \
  --min-length 5 --shuffle \
  --train-token 4096 --max-token 4096 \
  --step-freq 4 \
  --optim adam --lr 0.00025 --warmup-step 0 \
  --scheduler constant \
  --clip-norm 1 \
  --num-layer 12 \
  --embed-dim 512 \
  --num-head 8 \
  --inner-dim 2048 \
  --dropout 0.1 \
  --activation gelu \
  --tied-layer \
  --cuda --fp16 \
  --run-name masked_lm_base_layer_12 \
  --max-epoch 10 \
  "${@:2}"
elif [ "$1" == "eval" ]; then
  echo "Evaluating masked language model on penn data..."
  python3 eval.py --data data/penn \
  --test test.txt \
  --path workspace/penn \
  --task masked_lm --context-type line \
  --eval-token 200 --max-token 200 \
  --num-layer 12 \
  --embed-dim 512 \
  --num-head 8 \
  --inner-dim 2048 \
  --dropout 0.1 \
  --activation gelu \
  --tied-layer \
  --cuda \
  --checkpoint workspace/penn/masked_lm_base_layer_12/checkpoint/checkpoint_best.pt \
  "${@:2}"
fi
