#!/bin/bash
# Grid 3: CRF output layer, position/shape features, layer mixing. Saves per-token probabilities
# for the held-out fold and the test fold so later stages can build on them.
PY=/home/mani/miniconda3/envs/phase_1_capstone/bin/python
D=/home/mani/cs781/Capstone/Phase2_LNER/data/Phase_2_LNER
R=/home/mani/cs781_phase1/Legal_LLM/runs
OUT=/home/mani/cs781/Capstone/Phase2_LNER/runs
cd "$(dirname "$0")"
one() {  # gpu name train eval backbone extra-args...
  local gpu=$1 name=$2 tr=$3 ev=$4 bb=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu $PY train_ner.py --data $D --backbone $bb --train $tr --eval $ev --predict fold3 \
    --out_dir $OUT/${name}_${tr} --head bilstm --eval_every 4 --save_probs "$@" > $OUT/${name}_${tr}.log 2>&1 &
}
both() { local gpu=$1 name=$2 bb=$3; shift 3; one $gpu $name fold1 fold2 $bb "$@"; one $gpu $name fold2 fold1 $bb "$@"; }
M=$R/studies/FINAL_295M
both 0 g3_ft_crf        $M --epochs 20 --lr 2e-4 --crf
both 1 g3_ft_feats      $M --epochs 20 --lr 2e-4 --feats
both 2 g3_ft_crf_feats  $M --epochs 20 --lr 2e-4 --crf --feats
one 3 g3_ft_crf_feats_mix fold1 fold2 $M --epochs 20 --lr 2e-4 --crf --feats --layer_mix
one 4 g3_ft_crf_feats_mix fold2 fold1 $M --epochs 20 --lr 2e-4 --crf --feats --layer_mix
one 3 g3_frozen_crf_feats_mix fold1 fold2 $M --epochs 32 --freeze --crf --feats --layer_mix
one 4 g3_frozen_crf_feats_mix fold2 fold1 $M --epochs 32 --freeze --crf --feats --layer_mix
both 5 g3_prod_crf_feats $R/t2_prod_train --epochs 24 --lr 2e-4 --crf --feats
wait
echo "grid3 done"
