#!/bin/bash
# Phase 2 adaptation + distillation study (offline, 5 held-out splits). Each line of a queue runs on one GPU.
cd "$(dirname "$0")"
B=/workspace/models/cs781_phase1_runs
T=$B/ROPEFIX_295M          # Phase 1 model (287.6M), teacher size
S=$B/ROPEFIX_112M_snap2    # smaller Phase 1 model (98.9M), student size
R=../runs/adapt
mkdir -p $R
run() { gpu=$1; name=$2; shift 2; [ -f $R/$name/log.json ] && return
        CUDA_VISIBLE_DEVICES=$gpu python train_adapt.py --out_dir $R/$name "$@" > $R/$name.log 2>&1; }
wait_for() { while [ ! -f $1 ]; do sleep 30; done; }
q_full_a() { for k in 0 1 2; do run 3 full_k$k --backbone $T --method full --lr 2e-5 --kfold 5:$k --save; done
             for k in 0 1 2 3 4; do wait_for $R/full_k$k/model.pt; run 3 kd_k$k --backbone $S --method full --lr 2e-5 --kfold 5:$k --teacher $R/full_k$k; done; }
q_full_b() { for k in 3 4; do run 5 full_k$k --backbone $T --method full --lr 2e-5 --kfold 5:$k --save; done
             for k in 0 1 2 3 4; do run 5 student_k$k --backbone $S --method full --lr 2e-5 --kfold 5:$k; done; }
q_lora()   { for k in 0 1 2 3 4; do run 0 lora_k$k --backbone $T --method lora --lr 5e-4 --kfold 5:$k; done; }
q_lorma()  { for k in 0 1 2 3 4; do run 4 lorma_k$k --backbone $T --method lorma --lr 5e-4 --kfold 5:$k; done; }
q_frozen() { for k in 0 1 2 3 4; do run 1 frozen_k$k --backbone $T --method frozen --kfold 5:$k; done; }
q_top()    { for k in 0 1 2 3 4; do run 2 top6_k$k --backbone $T --method top --top_k 6 --lr 2e-5 --kfold 5:$k; done; }
q_full_a & q_full_b & q_lora & q_lorma & q_frozen & q_top &
wait
echo ALL DONE
