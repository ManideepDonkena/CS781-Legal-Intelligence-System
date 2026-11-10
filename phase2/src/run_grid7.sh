#!/bin/bash
# Grid 7: name-marking models on a Phase 1 backbone with working attention.  For each variant:
# five held-out runs (an honest score, and the inputs the second stage is trained on) and one model
# trained on all labelled documents for the test set.  Backbone frozen, so a run needs 2-3 GB.
# usage: run_grid7.sh <backbone run dir> <prefix> <gpu> [<gpu> ...]     (one lane per GPU listed;
#        list a GPU twice for two lanes on it)
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
D=/home/mani/cs781/Capstone/Phase2_LNER/data/Phase_2_LNER
OUT=/home/mani/cs781/Capstone/Phase2_LNER/runs
cd "$(dirname "$0")"
BB=$1; P=$2; shift 2
LANES=("$@")
run() {  # gpu name args...
  local gpu=$1 name=$2; shift 2
  [ -f $OUT/$name/fold3_spans.json ] && return  # finished earlier
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY train_ner.py --data $D --backbone $BB --head bilstm --save_probs \
    --train fold1,fold2 --predict fold3 --eval_every 10 --out_dir $OUT/$name "$@" > $OUT/$name.log 2>&1
}
FZ="--epochs 30 --freeze --crf --seed 1"
JOBS=()
for v in "fzcrf|" "fzcrfaug|--aug 0.8" "fzcrfaugmix|--aug 0.8 --layer_mix"; do
  name=${v%%|*}; extra=${v#*|}
  for i in 0 1 2 3 4; do JOBS+=("k_${P}_${name}_k$i $FZ $extra --kfold 5:$i"); done
  JOBS+=("final_${P}_${name} $FZ $extra --save")
done
# held-out runs of all variants first, fold by fold, so a comparison is available early
ORDER=()
for i in 0 1 2 3 4 5; do for v in 0 1 2; do ORDER+=("${JOBS[v * 6 + i]}"); done; done
n=${#LANES[@]}
for ((lane = 0; lane < n; lane++)); do
  ( for ((j = lane; j < ${#ORDER[@]}; j += n)); do run ${LANES[lane]} ${ORDER[j]}; done ) &
done
wait
echo "grid7 $P done"
