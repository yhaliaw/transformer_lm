#!/usr/bin/env bash

set -e

ROOT=$(dirname "$(readlink -f "$BASH_SOURCE")")

if [ "$1" == "train" ]; then
  echo "Training masked language model on penn data..."
  python3 train.py --data data/penn \
  --train train.txt --valid valid.txt \
  --path workspace/penn --tensorboard \
  --task masked_lm --context-type line \
  --train-token 400 --max-token 400 \
  --eval-token 200 --eval-max-token 200 \
  --update-freq 64 \
  --optim adam --lr 0.0005 --warmup-step 0 \
  --scheduler constant \
  --clip-norm 1 \
  --num-layer 12 \
  --embed-dim 512 \
  --num-head 8 \
  --inner-dim 2048 \
  --dropout 0.1 \
  --activation gelu \
  --tied-layer \
  --cuda \
  --run-name masked_lm_base_layer_12 \
  --epoch-per-valid 10 --keep-step 100000 --max-step 150000 \
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
