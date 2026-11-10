#!/bin/bash
# Phase 5 LoRMA study: rank-64 LoRMA with rank-stabilised scale (alpha/sqrt(r) = 16/8 = 2), TAPT model, 1 epoch.
# (A first try with adapter lr 5e-4 diverged after warm-up; 2e-4 is used.)
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=${GPU:-5} python ../../Phase4_CJPE/src/ft_doc.py --task lsi \
  --backbone /workspace/models/cs781_phase45/tapt_lsi_compact --text_fix --tail_tokens 512 --epochs 1 --lr 2e-4 \
  --lorma_rank 64 --lorma_alpha 16 --lorma_rs --out ../runs/lorma/rs64_e1 > ../runs/lorma/rs64_e1.log 2>&1
