#!/bin/bash
# One name-marking variant: five held-out runs (k_<name>_k0..4) plus one model trained on all
# labelled documents (final_<name>), spread over the GPUs listed (list a GPU twice for two lanes).
# usage: run_variant.sh <backbone run dir> <name> "<train_ner.py arguments>" <gpu> [<gpu> ...]
# e.g.   run_variant.sh $BB n2_zf "--epochs 30 --freeze --crf --aug 0.8 --layer_mix --feats" 1 1 4 4
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
D=/home/mani/cs781/Capstone/Phase2_LNER/data/Phase_2_LNER
OUT=/home/mani/cs781/Capstone/Phase2_LNER/runs
cd "$(dirname "$0")"
BB=$1; NAME=$2; ARGS=$3; shift 3
LANES=("$@")
run() {  # gpu name args...
  local gpu=$1 name=$2; shift 2
  [ -f $OUT/$name/fold3_spans.json ] && return  # finished earlier
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY train_ner.py --data $D --backbone $BB --head bilstm --save_probs \
    --train fold1,fold2 --predict fold3 --eval_every 10 --seed 1 --out_dir $OUT/$name $ARGS "$@" > $OUT/$name.log 2>&1
}
JOBS=()
for i in 0 1 2 3 4; do JOBS+=("k_${NAME}_k$i --kfold 5:$i"); done
JOBS+=("final_${NAME} --save")
n=${#LANES[@]}
for ((lane = 0; lane < n; lane++)); do
  ( for ((j = lane; j < ${#JOBS[@]}; j += n)); do run ${LANES[lane]} ${JOBS[j]}; done ) &
done
wait
echo "variant $NAME done"
