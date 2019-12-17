#!/usr/bin/env bash

set -e

ROOT=$(dirname "$0")

if [ "$1" == "train" ]; then
  echo "Training masked language model on sentencized wikitext-103 data..."
  python3 "$ROOT"/train.py --data data/sent-wikitext-103 \
    --train wiki.train.tokens --valid wiki.valid.tokens \
    --path workspace/sent-wiki103 --tensorboard \
    --task masked_lm --context-type line \
    --train-token 2048 --max-token 2048 \
    --step-freq 8 \
    --eval-token 2048 --eval-max-token 2048 \
    --optim adam --adam-betas 0.9 0.98 --lr 0.00025 \
    --warmup-step 0 \
    --scheduler cosine \
    --step-per-period 300000 \
    --max-step 300000 \
    --min-lr 1e-9 \
    --clip-norm 0.25 \
    --adaptive-input \
    --adaptive-softmax \
    --cutoff 20000 40000 \
    --num-layer 12 \
    --embed-dim 512 \
    --num-head 8 \
    --inner-dim 2048 \
    --dropout 0.1 \
    --activation gelu \
    --arch original_tree_transformer \
    --cuda \
    --run-name masked_lm_org_tree_layer_12 \
    "${@:2}"
elif [ "$1" == "eval" ]; then
  echo "Evaluating masked language model on wikitext-103 data..."
  python3 "$ROOT"/eval.py --data data/sent-wikitext-103 \
    --eval-valid \
    --valid wiki.valid.tokens \
    --test wiki.test.tokens \
    --task masked_lm --context-type line \
    --eval-token 2048 --eval-max-token 2048 \
    --adaptive-input \
    --adaptive-softmax \
    --cutoff 20000 40000 \
    --num-layer 12 \
    --embed-dim 512 \
    --num-head 8 \
    --inner-dim 2048 \
    --dropout 0.1 \
    --activation gelu \
    --arch original_tree_transformer \
    --cuda \
    --checkpoint workspace/sent-wiki103/masked_lm_org_tree_layer_12/checkpoint/checkpoint_best.pt \
    "${@:2}"
else
  echo "Specify 'train' or 'eval'."
fi
