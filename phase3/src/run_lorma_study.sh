#!/bin/bash
# Phase 3 LoRMA study: top-8-block LoRMA, same settings as the full fine-tuning runs (ft_c256_k*), 5 held-out folds.
cd "$(dirname "$0")"
for k in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} python finetune.py --fold $k --context 256 --max_sent 256 --lr 5e-4 \
    --lorma_rank 16 --lorma_alpha 32 --lorma_top 8 --no_extract --out ../runs/lorma/top8_k$k > ../runs/lorma/top8_k$k.log 2>&1
done
