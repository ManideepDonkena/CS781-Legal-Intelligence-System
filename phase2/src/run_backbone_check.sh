#!/bin/bash
# Compare Phase 1 backbones on the Phase 2 task with one fixed recipe: frozen backbone, BiLSTM head,
# CRF output layer, 5-fold held-out score (the recipe of k_frozencrf, which gave 54.5 raw F1 on the
# original FINAL_295M backbone).  One fold after another on a single GPU.
# usage: run_backbone_check.sh <gpu> <name> <backbone run dir>
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
D=/home/mani/cs781/Capstone/Phase2_LNER/data/Phase_2_LNER
OUT=/home/mani/cs781/Capstone/Phase2_LNER/runs
cd "$(dirname "$0")"
for i in 0 1 2 3 4; do
  [ -f $OUT/${2}_k$i/eval_spans.json ] && continue
  CUDA_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1 $PY train_ner.py --data $D --backbone $3 --head bilstm --train fold1,fold2 \
    --kfold 5:$i --epochs 30 --freeze --crf --seed 1 --eval_every 10 --out_dir $OUT/${2}_k$i > $OUT/${2}_k$i.log 2>&1
done
echo "backbone check $2 done"
