#!/usr/bin/env bash

set -e

ROOT=$(dirname "$0")

if [ "$1" == "train" ]; then
  echo "Training language model on wikitext-103 data..."
  python3 "$ROOT"/train.py --data data/wikitext-103 --worker 1 \
    --train wiki.train.tokens --valid wiki.valid.tokens \
    --path workspace/wiki103 --tensorboard \
    --task lm --context-type file \
    --context-size 0 --train-token 1024 --max-token 4096 \
    --step-freq 4 \
    --optim adam --adam-betas 0.9 0.999 --lr 0.00025 \
    --warmup-step 0 \
    --scheduler cosine --step-per-period 300000 --max-step 300000 \
    --min-lr 1e-9 \
    --clip-norm 0.1 \
    --adaptive-input \
    --adaptive-softmax \
    --cutoff 20000 40000 200000 \
    --num-layer 12 \
    --embed-dim 512 \
    --num-head 8 \
    --inner-dim 2048 \
    --dropout 0.2 \
    --attn-dropout 0.1 \
    --adaptive-softmax-dropout 0.2 \
    --cuda --fp16 \
    --run-name lm_base_layer_12 \
    "${@:2}"
elif [ "$1" == "eval" ]; then
  echo "Evaluating masked language model on wikitext-103 data..."
  python3 "$ROOT"/eval.py --data data/wikitext-103 --worker 1 \
    --test wiki.test.tokens --valid wiki.valid.tokens --eval-valid \
    --task lm --context-type file \
    --context-size 1024 --eval-token 1024 --eval-max-token 2048 \
    --adaptive-input \
    --adaptive-softmax \
    --cutoff 20000 40000 200000 \
    --num-layer 12 \
    --embed-dim 512 \
    --num-head 8 \
    --inner-dim 2048 \
    --dropout 0.1 \
    --attn-dropout 0.1 \
    --adaptive-softmax-dropout 0.2 \
    --cuda --fp16 \
    --checkpoint workspace/wiki103/lm_base_layer_12/checkpoint/checkpoint_best.pt \
    "${@:2}"
else
  echo "Specify 'train' or 'eval'."
fi