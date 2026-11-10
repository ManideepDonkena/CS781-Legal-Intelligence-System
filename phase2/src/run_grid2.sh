#!/bin/bash
# Grid 2: longer schedule and higher learning rates, BiLSTM head throughout.
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
D=/home/mani/cs781/Capstone/Phase2_LNER/data/Phase_2_LNER
R=/home/mani/cs781_phase1/Legal_LLM/runs
OUT=/home/mani/cs781/Capstone/Phase2_LNER/runs
cd "$(dirname "$0")"
run() {  # gpu name backbone extra-args...
  local gpu=$1 name=$2 bb=$3; shift 3
  for pair in "fold1 fold2" "fold2 fold1"; do
    tr=${pair% *}; ev=${pair#* }
    CUDA_VISIBLE_DEVICES=$gpu $PY train_ner.py --data $D --backbone $bb --train $tr --eval $ev \
      --out_dir $OUT/${name}_${tr} --head bilstm --epochs 30 --eval_every 3 "$@" > $OUT/${name}_${tr}.log 2>&1 &
  done
}
run 0 g2_mod295_lr2e-4   $R/studies/FINAL_295M --lr 2e-4
run 1 g2_mod295_lr5e-4   $R/studies/FINAL_295M --lr 5e-4
run 2 g2_mod295_frozen   $R/studies/FINAL_295M --freeze
run 3 g2_prod_lr2e-4     $R/t2_prod_train --lr 2e-4
run 4 g2_prod_bidir      $R/t2_prod_train --lr 2e-4 --attention bidir
run 5 g2_prod_repeat     $R/t2_prod_train --lr 2e-4 --repeat
wait
echo "grid2 done"
