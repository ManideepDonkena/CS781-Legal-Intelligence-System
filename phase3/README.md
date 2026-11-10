# Phase 3: Rhetorical Roles

Task: give every sentence of a judgment one of 13 roles (Fact, Issue, Argument of each side, Statute, Precedent
relied upon, Ratio of the decision, Ruling, ...). Score: macro-F1. Codabench phase 30685.
Leaderboard entry: **S08, 0.6242 (7th)**. 8 uploads.

## Method (best recipe, S08)

1. **Sentence vectors** (`src/embed.py`): my frozen Phase 1 model `ROPEFIX_295M` reads each sentence together with up
   to 768 tokens of the text before it; the token states of 4 layers are kept.
2. **Sentence model** (`src/train_rr.py`): attention over the words of each sentence, the sentence's position in the
   document, then a BiLSTM over all sentences of the document; rare roles get extra weight (`--cw 0.5`). 5-fold
   document cross-validation for the held-back score; `--full_only` trains the final models on all labelled documents.
3. **Submission** (`src/combine.py`, data code in `src/rr.py`): averages the runs and writes `prediction.json`.

## Every upload

| Upload | Folder in `submissions/` | What it is | Held-back | Test |
|:--|:--|:--|:-:|:-:|
| S01 | `S01_att_cw_295m` | frozen 295M, word attention, extra weight for rare roles | 49.8 | 0.5979 |
| S02 | `S02_att_cw_c768` | S01 with 768 tokens of earlier text | 51.3 | 0.6259 |
| S03 | `S03_c768_shift` | S02 + shift towards common roles | - | 0.6193 |
| S04 | `S04_S02_noDissent` | S02 with `Dissent` removed (a check of the test set) | - | 0.5811 |
| S05 | `S05_S02_dissentrule` | S02 + a `Dissent` rule written after reading a test document | - | 0.6628 |
| S06 | `S06_c768_c1792_shift_rule` | two window sizes + shift + the same rule | - | 0.6658 |
| S07 | `S07_honest_plain` | models from the cross-validation parts, no rule | 52.45 | 0.5974 |
| **S08** | `S08_honest_alldata` | final models trained on all labelled documents, no rule (**leaderboard**) | 52.45 | **0.6242** |

S04-S06 used knowledge taken from the test set, so I took them off the leaderboard myself and use S08, which
follows the rules. `S01_tfidf_logreg` is the word-count baseline (36.5 held-back), not uploaded.

## LoRMA run (9 Nov, offline, not uploaded)

Adapters only in the top 8 of 23 blocks (`src/finetune.py --lorma_rank 16 --lorma_top 8`, script
`src/run_lorma_study.sh`): 47.7 against 48.0 for full fine-tuning (sentence classifier alone, 5 folds), in half
the training time. Logs: `../lorma_results/phase3/`.

## Weights

The frozen Phase 1 model `phase1_ROPEFIX_295M_weights.pt` in the GitHub release `models-v1`; the small sentence
models are retrained in a few minutes by `train_rr.py`.
