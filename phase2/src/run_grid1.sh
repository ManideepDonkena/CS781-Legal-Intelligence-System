#!/bin/bash
# Grid 1: which Phase 1 backbone and which way of giving the tagger right-context works best.
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
D=/home/mani/cs781/Capstone/Phase2_LNER/data/Phase_2_LNER
R=/home/mani/cs781_phase1/Legal_LLM/runs
OUT=/home/mani/cs781/Capstone/Phase2_LNER/runs
cd "$(dirname "$0")"
run() {  # gpu name backbone extra-args...
  local gpu=$1 name=$2 bb=$3; shift 3
  for pair in "fold1 fold2" "fold2 fold1"; do
    set -- "$@"; tr=${pair% *}; ev=${pair#* }
    CUDA_VISIBLE_DEVICES=$gpu $PY train_ner.py --data $D --backbone $bb --train $tr --eval $ev \
      --out_dir $OUT/${name}_${tr} "$@" > $OUT/${name}_${tr}.log 2>&1 &
  done
}
run 0 prod_causal_linear   $R/t2_prod_train
run 1 prod_bidir_linear    $R/t2_prod_train --attention bidir
run 2 prod_repeat_linear   $R/t2_prod_train --repeat
run 3 prod_causal_bilstm   $R/t2_prod_train --head bilstm
run 4 mod295_causal_linear $R/studies/FINAL_295M
run 5 mod295_causal_bilstm $R/studies/FINAL_295M --head bilstm
wait
echo "grid1 done"
