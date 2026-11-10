#!/bin/bash
# Phase 6 LoRMA study: LoRMA (rank 16) only on the three feed-forward matrices of every block, 1 epoch, then the
# adapters are folded into the weights and the first 100 dev cases are summarised (same setup as abs_1ep_lr5e5).
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-4}
python abstractive.py train --out /workspace/models/cs781_lorma/p6_ffn_r16 --epochs 1 --lr 5e-4 --lorma_rank 16 \
  --lorma_alpha 32 --lorma_ffn_only > ../runs/lorma/ffn_r16_train.log 2>&1
python abstractive.py generate --ckpt /workspace/models/cs781_lorma/p6_ffn_r16/e1.pt --split dev \
  --out ../runs/lorma/ffn_r16 --tag _n100 --limit 100 --batch 16 > ../runs/lorma/ffn_r16_eval.log 2>&1
