#!/bin/bash
# Phase 2 check (run_backbone_check.sh) for the small Wikipedia-study models, one after another on
# one GPU, each as soon as its Phase 1 run has finished:
#   Wikipedia then legal, Wikipedia only, legal only with the same total training.
# usage: run_backbone_queue.sh <gpu>
R=/workspace/models/cs781_phase1_runs
cd "$(dirname "$0")"
while pgrep -f "run_backbone_check.sh $1 bb_rf112snap" > /dev/null; do sleep 30; done
check() {  # name run
  until [ -f $R/$2/post_eval.json ]; do sleep 60; done
  ./run_backbone_check.sh $GPU $1 $R/$2
}
GPU=$1
check bb_wikilegal56 WIKI_THEN_LEGAL_56M
check bb_wiki56 WIKI_56M
check bb_legal2pass56 LEGAL_2PASS_56M
echo "backbone queue done"
