# Phase 6: abstractive summarization

Deadline 3 November 2026, 23:59 IST (Codabench phase 30688, task 36369). Scored by BERTScore F1 (main), then
ROUGE-1 and ROUGE-L.

## Code (`src/`, all our own code on PyTorch and NumPy)
- `summ.py`: loads the data, holds out 400 fixed training cases as dev, splits sentences, computes ROUGE-1/2/L.
- `extractive.py`: sentence scorer (our own TF-IDF and logistic regression) and length budget; `--xfit 5` writes
  cross-fitted sentence scores for the abstractive input.
- `abstractive.py`: `train` fine-tunes our Phase 1 model `ROPEFIX_295M` (stretched to a 4,096-token context) to
  write the summary after the case's best sentences; `generate` does batched greedy decoding.
- `hybrid.py`: abstractive opening plus extracted sentences.
- `make_sub.py`: writes `submissions/<name>/<name>.zip` with `prediction.json`.
- Upload: `Capstone/Phase5_LSI/src/upload.py --task sum` (needs browser cookies in CODABENCH_CSRF / CODABENCH_SESSION).

## Reproduce the submissions
```
python extractive.py --out ../runs/ext_v1 --factor 1.1                  # dev check
python extractive.py --out ../runs/ext_v1_all --all --factor 1.1        # S01
python extractive.py --out ../runs/ext_C16_all --all --C 16 --factor 1.2  # S05
python extractive.py --out ../runs/ext_xfit5 --xfit 5                   # inputs for the language model
python abstractive.py train --all --out /workspace/models/cs781_phase6/abs_1ep_lr5e5_all --lr 5e-5 --epochs 1
python abstractive.py generate --ckpt /workspace/models/cs781_phase6/abs_1ep_lr5e5_all/e1.pt --split test \
    --out ../runs/abs_1ep_lr5e5_all                                     # S04
python hybrid.py --abs ../runs/abs_1ep_lr5e5_all/test_summaries.json --split test --keep 150 \
    --save ../runs/hybrid/test_keep150.json                             # S02 (keep 250: S03)
python abstractive.py train --out /workspace/models/cs781_phase6/abs_1ep_lr1e4 --lr 1e-4 --epochs 1   # S06
```

## Dev results (our ROUGE, no stemming; R1 / R2 / RL)
| method | 400 dev cases | 100 dev cases, full text | 100 dev cases, first 400 words |
|---|---|---|---|
| first sentences of the case | 50.9 / 24.8 / 24.5 | | |
| extractive (S01) | 59.8 / 36.8 / 33.6 | 59.0 / 35.4 / 32.8 | 57.3 / 29.0 / 31.0 |
| abstractive, lr 5e-5, 1 epoch (S04) | | 52.0 / 27.0 / 27.3 | about 54.5 / 26.0 / 29.3 |
| abstractive, lr 1e-4, 1 epoch (S06) | | 52.0 / 26.6 / 27.1 | about 54.9 / 26.2 / 29.4 |
| hybrid, 150-word opening (S02) | | 58.8 / 35.0 / 32.5 | 58.1 / 30.0 / 31.5 |
| hybrid, 250-word opening (S03) | | 58.3 / 34.1 / 31.3 | 57.3 / 28.6 / 30.2 |

"First 400 words" stands in for BERTScore, whose scoring model reads at most 512 tokens. S02 was chosen as the
leaderboard entry before any test score was seen. Findings: fine-tuning overfits after one epoch; without the
4-token no-repeat rule the model loops; the generated text is good for the first few hundred words, then drifts.
