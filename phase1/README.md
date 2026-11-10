# Phase 1: Legal Language Model

Task: train a language model (50M-300M parameters) from random weights on the course's Indian legal text and
continue held-out prompts. Score: BERTScore F1 (also ROUGE-L, BLEU-4). Codabench phase 30683.
Leaderboard entry: **upload 965325, 0.8376 (23rd)**. 33 uploads, all made 4-6 Oct, before the rotary bug was found.

## Folders

- `final_model/` - the code of the final model `ROPEFIX_295M` (287.6M parameters, used in Phases 2-7): model
  (`modern_arch.py`, with my Assignment 3 `rope.py` and `attention.py`), training file (`studies/build_cache.py`),
  training (`studies/train_job.py`), text generation (`studies/gen_modern.py`, `studies/gen_modern_mbr.py`).
  See `final_model/README.md` for the exact training command.
- `train.py`, `train_compare.py`, `src/`, `core/`, `stages/`, `configs/` - my first Phase 1 programs (design
  comparison, early models).
- `tokenizer_study.py` - the tokenizer study (normal BPE 16k/32k/50k, legal-aware BPE).

## Every upload (BERTScore F1 / ROUGE-L / BLEU-4)

The uploads came in four groups (all scores are in the report, Appendix "All Phase 1 Uploads"):

1. **Simple transformer `t2_prod_train`** (181.7M parameters, learned positions, 50k normal BPE, 200M Supreme Court
   tokens) with different ways of picking the next word: greedy 0.8016, nucleus 0.8274, min-p 0.8313, MBR 0.8336.
2. **Legal-aware tokenizer models** (192.4M): 0.7669-0.8116.
3. **Modern models** (rotary positions, still with the bug): 56M 0.8187-0.8232, 99M 0.8254, 295M 0.8263-0.8330;
   one with contrastive search 0.8005.
4. **Agreement (MBR) across the continuations of several models**: 0.8339-0.8340.

My best upload, 965325 (0.8376 / 0.2001 / 0.0498, the leaderboard entry), came on the last day. My records do not say which recipe it used, and they do not tie every one of the 33 uploads to a single command.

## Weights

GitHub release `models-v1`: `phase1_ROPEFIX_295M_weights.pt` (final model), `phase1_tokenizer.json`,
`phase1_run_config.json`.
