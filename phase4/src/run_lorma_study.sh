#!/bin/bash
# Phase 4 LoRMA study: LoRMA (rank 16, all 7 matrices of every block) on top of the TAPT model, repaired text,
# 1 epoch, scored twice per epoch on the dev set.  (Adapter lr 5e-4 diverged and 2e-4 collapsed to one class by mid-epoch; 5e-5 is used.)
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=${GPU:-3} python ft_doc.py --task cjpe --backbone /workspace/models/cs781_phase45/tapt_cjpe_fix \
  --text_fix --epochs 1 --evals_per_epoch 2 --lr 5e-5 --lorma_rank 16 --lorma_alpha 32 \
  --out ../runs/lorma/tapt_lorma_e1 > ../runs/lorma/tapt_lorma_e1.log 2>&1
