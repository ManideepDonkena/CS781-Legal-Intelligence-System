#!/bin/bash
# Grid 5: mention-swap augmentation (about +2 raw F1 in grid 4).  Three full-data models for the test
# ensemble and cross-validation seeds so the second stage can be trained on matching inputs.
# Part a runs on GPU 3, part b on GPUs 0-1 after grid 4; GPUs 4 and 5 are left alone (another user's
# server runs on GPU 4).
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
  run $gpu ${name}_$tr --train $tr --eval $ev --eval_every 6 "$@"
}
AUG="--epochs 24 --lr 2e-4 --aug 0.5"
FZA="--epochs 30 --freeze --aug 0.5"
case "$1" in
a)  # GPU 3, free once grid 4's lanes there are done
  ( full 3 final_v3_ftaug_s1 $AUG --seed 1; cv 3 g5_ftaug_p08 fold1 --epochs 24 --lr 2e-4 --aug 0.8 ) &
  ( full 3 final_v3_ftaug_s2 $AUG --seed 2; cv 3 g5_ftaug_p08 fold2 --epochs 24 --lr 2e-4 --aug 0.8 ) &
  ;;
b)  # GPUs 0 and 1, after grid 4 has finished
  ( full 0 final_v3_ftaug_s3 $AUG --seed 3; cv 0 g5_ftaug_s3 fold1 $AUG --seed 3 ) &
  ( cv 0 g5_ftaug_s2 fold1 $AUG --seed 2; cv 0 g5_ftaug_s2 fold2 $AUG --seed 2; cv 0 g5_ftaug_s3 fold2 $AUG --seed 3 ) &
  ( for s in 1 2 3; do full 1 final_v3_frozenaug_s$s $FZA --seed $s --save; done ) &
  ( for s in 2 3; do for f in fold1 fold2; do cv 1 g5_frozenaug_s$s $f $FZA --seed $s; done; done ) &
  ;;
*) echo "usage: $0 a|b"; exit 1 ;;
esac
wait
echo "grid5 part $1 done"
