#!/bin/bash
# Data-mix study repeated with working attention (rope_float).  The earlier study (G3_base_s1,
# C5_mixNAT, DATA_clean_modern57M) was run on models whose attention was uniform, so its
# conclusions about the source mix and about cleaning may not hold.  Same 56M model and 600M-token
# budget, one run per cache, one after another on a single GPU.
# usage: studies/ropefix_data_study.sh <gpu>
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
OUT=/workspace/models/cs781_phase1_runs
cd "$(dirname "$0")/.."
for cache in bpe50k_mixA_600M bpe50k_mixNAT_600M bpe50k_mixA_clean_600M; do
  id=RF_${cache#bpe50k_}
  CUDA_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1 $PY -u studies/train_job.py --id $id --study ROPEFIX_DATA --cache $cache \
    --d-model 512 --layers 11 --heads 8 --kv-heads 2 --tokens 600000000 --lr 0.001 --seed 1 \
    --extra '{"rope_float": true}' --out-root $OUT > logs_studies/$id.log 2>&1
done
echo "data study done"
