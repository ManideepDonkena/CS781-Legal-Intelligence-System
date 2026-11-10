#!/bin/bash
# Grid 4: seed ensemble members on all labelled data, plus cross-validation of mention-swap
# augmentation, the CRF layer and extra seeds.  Short runs (5-10 min), two lanes on each of GPUs 0-3;
# GPUs 4 and 5 are left free.
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
D=/home/mani/cs781/Capstone/Phase2_LNER/data/Phase_2_LNER
M=/home/mani/cs781_phase1/Legal_LLM/runs/studies/FINAL_295M
OUT=/home/mani/cs781/Capstone/Phase2_LNER/runs
cd "$(dirname "$0")"
run() {  # gpu name args...
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY train_ner.py --data $D --backbone $M --head bilstm --save_probs \
    --predict fold3 --out_dir $OUT/$name "$@" > $OUT/$name.log 2>&1
}
full() { local gpu=$1 name=$2; shift 2; run $gpu $name --train fold1,fold2 "$@"; }
cv() {  # gpu name train-fold args...
  local gpu=$1 name=$2 tr=$3; shift 3
  local ev=fold2; [ $tr = fold2 ] && ev=fold1
  run $gpu ${name}_$tr --train $tr --eval $ev --eval_every 5 "$@"
}
FT="--epochs 20 --lr 2e-4"
FZ="--epochs 30 --freeze"
( full 0 final_v2_ft_s2 $FT --seed 2; cv 0 g4_ft_crf fold1 $FT --crf; cv 0 g4_ft_s3 fold1 $FT --seed 3 ) &
( full 0 final_v2_ft_s3 $FT --seed 3; cv 0 g4_ft_crf fold2 $FT --crf; cv 0 g4_ft_s3 fold2 $FT --seed 3 ) &
( full 1 final_v2_ftfeats_s1 $FT --feats; full 1 final_v2_ft_s4 $FT --seed 4; cv 1 g4_frozen_crf fold1 $FZ --crf ) &
( full 1 final_v2_ftfeats_s2 $FT --feats --seed 2; cv 1 g4_frozen_aug fold1 $FZ --aug 0.5; cv 1 g4_frozen_aug fold2 $FZ --aug 0.5; cv 1 g4_frozen_crf fold2 $FZ --crf ) &
( cv 2 g4_ft_aug fold1 --epochs 24 --lr 2e-4 --aug 0.5; for s in 2 3 4; do full 2 final_v2_frozen_s$s $FZ --seed $s --save; done ) &
( cv 2 g4_ft_aug fold2 --epochs 24 --lr 2e-4 --aug 0.5; cv 2 g4_frozen_s2 fold1 $FZ --seed 2; cv 2 g4_frozen_s2 fold2 $FZ --seed 2 ) &
( cv 3 g4_ft_s2 fold1 $FT --seed 2; cv 3 g4_frozen_s3 fold1 $FZ --seed 3 ) &
( cv 3 g4_ft_s2 fold2 $FT --seed 2; cv 3 g4_frozen_s3 fold2 $FZ --seed 3 ) &
wait
echo "grid4 done"
