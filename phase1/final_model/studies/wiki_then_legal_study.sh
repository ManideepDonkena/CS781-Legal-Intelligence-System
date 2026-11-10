#!/bin/bash
# Does general text first help?  56M model with working attention (rope_float), same tokenizer
# (bpe50k) and 600M-token stages, one run after another on a single GPU:
#   WIKI_56M             Wikipedia only
#   WIKI_THEN_LEGAL_56M  starts from WIKI_56M, then the legal mix          (1.2B tokens in total)
#   RF_mixA_clean_600M   third run of the data-mix study (cleaned legal mix)
#   LEGAL_2PASS_56M      legal mix only, 1.2B tokens: the equal-compute comparison
# RF_mixA_600M (legal mix only, 600M tokens) is the one-stage reference from ropefix_data_study.sh.
# usage: studies/wiki_then_legal_study.sh <gpu>
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
OUT=/workspace/models/cs781_phase1_runs
GPU=$1
cd "$(dirname "$0")/.."
while pgrep -f "train_job.py --id RF_mixNAT_600M" > /dev/null; do sleep 30; done  # the GPU's current run
job() {  # id cache extra-args...
  local id=$1 cache=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY -u studies/train_job.py --id $id --study WIKI --cache $cache \
    --d-model 512 --layers 11 --heads 8 --kv-heads 2 --lr 0.001 --seed 1 --extra '{"rope_float": true}' \
    --out-root $OUT "$@" > logs_studies/$id.log 2>&1
}
job WIKI_56M bpe50k_wiki_600M --tokens 600000000 --keep-best
job WIKI_THEN_LEGAL_56M bpe50k_mixA_600M --tokens 600000000 --keep-best --init-from $OUT/WIKI_56M/best.pt
job RF_mixA_clean_600M bpe50k_mixA_clean_600M --tokens 600000000
job LEGAL_2PASS_56M bpe50k_mixA_600M --tokens 1200000000 --keep-best
echo "wiki study done"
