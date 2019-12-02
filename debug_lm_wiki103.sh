#!/usr/bin/env bash
set -e

ROOT=$(dirname "$0")

if [ "$1" == "train" ]; then
  echo "Training language model on wikitext-103 data..."
  python3 "$ROOT"/train.py --data data/wikitext-103 --worker 1 \
    --train wiki.train.tokens --valid wiki.valid.tokens \
    --path workspace/wiki103 --tensorboard \
    --task lm --context-type file \
    --context-size 0 --train-token 300 --max-token 2400 \
    --update-freq 6 \
    --optim adam --adam-betas 0.9 0.98 --lr 0.00025 \
    --warmup-step 0 \
    --scheduler cosine --step-per-period 200000 \
    --period-decay 0.25 --min-lr 1e-9 \
    --clip-norm 0.25 \
    --adaptive-input \
    --adaptive-softmax \
    --tied-adaptive \
    --cutoff 20000 60000 \
    --num-layer 16 \
    --embed-dim 512 \
    --num-head 8 \
    --inner-dim 2048 \
    --dropout 0.1 \
    --attn-dropout 0.1 \
    --cuda --fp16 \
    --run-name lm_base_layer_16 \
    "${@:2}"
elif [ "$1" == "eval" ]; then
  echo "Evaluating masked language model on wikitext-103 data..."
  python3 "$ROOT"/eval.py --data data/wikitext-103 --worker 1 \
    --test wiki.test.tokens --valid wiki.valid.tokens --eval-valid \
    --context-type file --context-size 0 --eval-token 300 \
    --eval-max-token 2400 \
    --adaptive-input \
    --adaptive-softmax \
    --tied-adaptive \
    --cutoff 20000 60000 \
    --num-layer 16 \
    --embed-dim 512 \
    --num-head 8 \
    --inner-dim 2048 \
    --dropout 0.1 \
    --attn-dropout 0.1 \
    --cuda --fp16 \
    --checkpoint workspace/wiki103/lm_base_layer_16/checkpoint/checkpoint_best.pt \
    "${@:2}"
fi