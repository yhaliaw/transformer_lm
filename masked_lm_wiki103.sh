#!/usr/bin/env bash

set -e

ROOT=$(dirname "$0")

if [ "$1" == "train" ]; then
  echo "Training masked language model on wikitext-103 data..."
  python3 "$ROOT"/train.py --data data/wikitext-103 \
    --train wiki.train.tokens --valid wiki.valid.tokens \
    --path workspace/wiki103 --tensorboard \
    --task masked_lm --context-type sent \
    --train-token 1024 --max-token 1024 \
    --update-freq 8 \
    --optim adam --adam-betas 0.9 0.98 --lr 0.00025 \
    --warmup-step 0 \
    --scheduler cosine --step-per-period 400000 \
    --period-decay 0.25 --min-lr 1e-9 \
    --clip-norm 0.5 \
    --adaptive-input \
    --adaptive-softmax \
    --tied-adaptive \
    --cutoff 20000 40000 200000 \
    --num-layer 12 \
    --embed-dim 512 \
    --num-head 8 \
    --inner-dim 2048 \
    --dropout 0.1 \
    --activation gelu \
    --cuda --fp16 \
    --run-name masked_lm_base_layer_12 \
    "${@:2}"
elif [ "$1" == "eval" ]; then
  echo "Evaluating masked language model on wikitext-103 data..."
  python3 "$ROOT"/eval.py --data data/wiki103 \
    --test test.txt \
    --task masked_lm --context-type sent \
    --eval-token 2048 --max-token 2048 \
    --adaptive-input \
    --adaptive-softmax \
    --tied-adaptive \
    --cutoff 20000 40000 200000 \
    --num-layer 12 \
    --embed-dim 512 \
    --num-head 8 \
    --inner-dim 2048 \
    --dropout 0.1 \
    --activation gelu \
    --cuda --fp16 \
    --checkpoint workspace/wiki103/masked_lm_base_layer_12/checkpoint/checkpoint_best.pt \
    "${@:2}"
else
  echo ""
fi
