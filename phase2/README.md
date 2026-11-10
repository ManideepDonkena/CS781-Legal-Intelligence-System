# Phase 2: Legal Named Entities (L-NER)

Task: find every legal entity name (appellant, respondent, judge, court, statute, precedent, ...) in each judgment. Score: strict macro-F1 (exact string and label). Codabench phase 30684.
Leaderboard entry: **S07, 0.6783 (4th)**. 6 uploads, 5 scored (S08 stayed "Submitted" and was never scored).

## Method (best recipe, S07)

1. **Tagger** (`src/train_ner.py`, model and data code in `src/lner.py`): my Phase 1 model `ROPEFIX_112M_snap2`
   (98.9M parameters, frozen) reads 1,024-token windows; a learned mix of its layers feeds a BiLSTM and a CRF that
   picks the best label sequence. Training adds name swapping (a name is replaced by another name of the same type)
   and a "side of Versus" clue: each word is marked as being in the appellant part, the respondent part, both or
   neither of the case heading.
2. **Whole-document check** (`src/stage2.py`): every candidate string of a document gets one decision from a
   gradient-boosted classifier (features: where the string appears in the heading, evidence shared by related names,
   local patterns such as "PW-1" or ", J.", distance to counsel names), then the strings are written longest first.
3. **Submission** (`src/make_submission.py`): writes `prediction.json` in the official format.
4. **Comparing versions**: only with repeated cross-validation over four document splits (`src/stage2_cv.sh`); one
   split moves the score by about +-1.

## Every upload

| Upload | Folder in `submissions/` | What it is | Held-back | Test |
|:--|:--|:--|:-:|:-:|
| S01 | `S01_mod295_ft_bilstm` | one fine-tuned tagger on the 295M model (still with the rotary bug) + BiLSTM | 47.9 | 0.5490 |
| S02 | `S02_pool4_votes2` | four taggers voting (a string is kept if 2 of 4 agree) | - | 0.5885 |
| S05 | `S05_stage2_m6` | six taggers + whole-document check | 63.3 | 0.6438 |
| S06 | `S06_single_n2zf` | one tagger on the fixed `ROPEFIX_112M` + side clue + document check | 65.2-65.5 | 0.6757 |
| **S07** | `S07_two_n2` | S06 + a second tagger that uses only the side clue (**leaderboard**) | 67.2 | **0.6783** |
| S08 | `S08_single_wide` | S06 method on `ROPEFIX_112M_WIDE` (wider data mix) | 65.1 | not scored |

Folders `S01_mod295_frozen_bilstm`, `S03_stage2_dict` and `S04_stage2_adj` hold files that were prepared but not uploaded.

## Extra study (9 Nov, offline, not uploaded)

Fine-tuning methods and distillation on 5 held-out splits (strict macro-F1 after copying labels): frozen 57.1,
top 6 layers 58.0, full fine-tuning 58.1, LoRA 57.6, **LoRMA 58.6**, 98.9M student alone 59.9, student distilled from
the 287.6M model 58.3. Code: `src/adapters.py` (built on my Assignment 4 LoRMA functions in `src/q4_peft_lorma.py`),
`src/train_adapt.py`, `src/run_adapt_study.sh`, `src/summarize_adapt.py`; logs in `results_adapt/`.

## Weights

`phase2_lner_S07_tagger_and_backbone.tar` in the GitHub release `models-v1` (the S07 taggers and the backbone).
