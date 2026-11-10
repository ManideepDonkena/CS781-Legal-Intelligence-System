#!/bin/bash
# Grid 6: 5-fold held-out predictions for five tagger variants (plain, augmented, CRF+augmented,
# frozen, frozen+CRF), so the second stage is trained on first-stage output from models that saw
# 80% of the labelled documents - close to the full-data models used on the test set.  Also trains
# the two full-data CRF models that have no counterpart yet.  Every run predicts fold3 too.
# GPUs 0-3, two lanes each.
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
D=/home/mani/cs781/Capstone/Phase2_LNER/data/Phase_2_LNER
M=/home/mani/cs781_phase1/Legal_LLM/runs/studies/FINAL_295M
OUT=/home/mani/cs781/Capstone/Phase2_LNER/runs
cd "$(dirname "$0")"
run() {  # gpu name args...
  local gpu=$1 name=$2; shift 2
  [ -f $OUT/$name/fold3_spans.json ] && return  # finished earlier
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY train_ner.py --data $D --backbone $M --head bilstm --save_probs \
    --train fold1,fold2 --predict fold3 --eval_every 8 --out_dir $OUT/$name "$@" > $OUT/$name.log 2>&1
}
CRFAUG="--epochs 20 --lr 2e-4 --crf --aug 0.8"
FZCRF="--epochs 30 --freeze --crf"
FT=("final_v3_ftcrfaug_s1 $CRFAUG --seed 1")
FZ=("final_v3_frozencrf_s1 $FZCRF --seed 1 --save")
for i in 0 1 2 3 4; do
  FT+=("k_ftcrfaug_k$i $CRFAUG --seed 1 --kfold 5:$i")
  FT+=("k_ftaug_k$i --epochs 24 --lr 2e-4 --aug 0.5 --seed 1 --kfold 5:$i")
  FT+=("k_ft_k$i --epochs 20 --lr 2e-4 --seed 1 --kfold 5:$i")
  FZ+=("k_frozencrf_k$i $FZCRF --seed 1 --kfold 5:$i")
  FZ+=("k_frozen_k$i --epochs 30 --freeze --seed 1 --kfold 5:$i")
done
for lane in 0 1 2 3 4 5 6 7; do
  (
    for ((j = lane; j < ${#FT[@]}; j += 8)); do run $((lane / 2)) ${FT[j]}; done
    for ((j = lane; j < ${#FZ[@]}; j += 8)); do run $((lane / 2)) ${FZ[j]}; done
  ) &
done
wait
echo "grid6 done"
