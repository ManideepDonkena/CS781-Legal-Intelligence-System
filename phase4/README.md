# Phase 4: Case Judgement Prediction (ILDC)

Predict whether a Supreme Court appeal was accepted (1) or rejected (0) from the judgment text.
Score: macro-F1 (the mean of the F1 of "accepted" and of "rejected").

## Data facts

- Train has 37,387 rows but only 32,305 different cases: 5,082 cases appear twice (same text, same label).
  Dev has 1,988 rows = 994 cases, each listed twice. Both are de-duplicated (`src/cjpe.py`).
- Train texts keep capital letters and have no line breaks. Dev and test are all lowercase and wrapped with line
  breaks. All texts are lowercased and their spaces and line breaks collapsed.
- All splits carry an old processing glitch: "con" was replaced by "company" (e.g. "companyrt" = "court").
- Train is 41% "accepted", dev is 50%. Because of this the decision threshold must be set on dev, not left at 0.5.
- Cases are long: median about 3,400 tokens; only 22% fit in the model's 2,048-token window.
- About 10-15% of cases still state the outcome near the end ("we allow the appeal").
- **The test file still contains a `label` field**, although the task says labels were removed. `cjpe.load`
  drops it; it is never used.

## Method

1. Word-count model: TF-IDF of words and word pairs + logistic regression, written from scratch in
   `../shared/scratch_text.py` (run with `../shared/wordcount_run.py --task cjpe`). `src/baseline_tfidf.py` is the
   earlier scikit-learn version (69.2 honest; ours 69.1), kept as a record.
2. Main model (`src/ft_doc.py --task cjpe`): our own Phase 1 language model (`ROPEFIX_295M`, 295M parameters,
   trained from scratch in Phase 1) fine-tuned as a classifier on the **last 2,048 tokens** of each case.
   Its final-layer outputs are summarised three ways (attention pooling, mean, last token) and a linear layer
   gives the two classes. 2 passes over the training data, learning rate 2e-5 (head 1e-3).
3. Final prediction (`src/combine.py`): average of the class-1 probabilities of the fine-tuned runs and the
   word-count model; the threshold is the one that gives the best macro-F1 on dev.

"Honest" dev scores below choose the threshold on one half of dev and score the other half (averaged over 20
random halvings), so they do not reuse the scored cases.

## Results (dev macro-F1, honest)

| Model | Dev |
|---|---|
| Word counts, whole case | 69.2 |
| Word counts, last 1,000 words | 66.2 |
| Fine-tuned Phase 1 model, last 2,048 tokens, 2 passes (two seeds) | 80.8, 81.6 |
| Same, 3 passes (overfits: 80.9 after one pass, 79.4 at the end) | 78.8 |
| Submission P4_S01: three fine-tuned runs + library word counts | 81.5 |
| Submission P4_S02: three fine-tuned runs + our own word counts | 81.5 |
| Glitch fixed (`--text_fix`), 2 passes | 81.8 |
| Glitch fixed + task-adaptive pretraining (`tapt.py`), 2 passes (three seeds) | 82.0, 83.0, 82.2 |
| Same with learning rate 4e-5 instead of 2e-5 | 82.7 |
| Submission P4_S03: 2 adapted runs + 1 glitch-fixed run + word counts | 84.0 |
| **Submission P4_S04: all 4 adapted runs + the glitch-fixed run + word counts (replaces P4_S03)** | **84.3** |

**Improvements added on 25 October evening**
- *Undoing the "company/number" glitch* (`src/text_fix.py`). The ILDC preprocessing replaced "co" + any character
  with "company" and "no" + any character with "number" (court -> companyrt, notice -> numberice, no doubt ->
  numberdoubt); 3.3% of all words are damaged. Our Phase 1 model learned from clean Supreme Court text, where these
  spellings never occur. Each damaged word is replaced by the candidate spelling that is most common in the clean
  Phase 1 text; this restores 98% of damaged occurrences. A lone "number" (often "not") cannot be recovered and stays.
- *Task-adaptive pretraining* (`src/tapt.py`). Before fine-tuning, the Phase 1 model continues its next-word training
  for one pass on the fixed train + dev windows (never test text). Its next-word error on held-out cases fell from
  2.46 to 1.84, i.e. it learned the lowercase, OCR-damaged style of these judgments.

Tried and dropped: reading the whole case in 2,048-token chunks and combining the chunk scores
(`src/chunk_predict.py`, `src/stack.py`): the earlier chunks got weights near zero and the score fell slightly.
Averaging standardised log-odds instead of probabilities: worse.

## Rebuild the submission

```
cd src
python ft_doc.py --task cjpe --out ../runs/ft_tail2048_e2    --epochs 2 --evals_per_epoch 2
python ft_doc.py --task cjpe --out ../runs/ft_tail2048_e2_s2 --epochs 2 --evals_per_epoch 2 --seed 2
python ft_doc.py --task cjpe --out ../runs/ft_tail2048_e3_s1 --epochs 3 --evals_per_epoch 2 --seed 1
python ../../shared/wordcount_run.py --task cjpe --out tfidf_scratch
python combine.py ft_tail2048_e2:e2 ft_tail2048_e2_s2:e2 ft_tail2048_e3_s1:e3 tfidf_scratch --out ../submissions/P4_S02_lm3_tfidf_scratch

# P4_S03 (current best)
python text_fix.py                                   # builds ../data/glitch_fix.json from train + dev words
W=/workspace/models/cs781_phase45
python tapt.py --task cjpe --out $W/tapt_cjpe_fix --text_fix
python ft_doc.py --task cjpe --backbone $W/tapt_cjpe_fix --text_fix --out ../runs/ft_tapt_fix_e2    --epochs 2 --evals_per_epoch 2
python ft_doc.py --task cjpe --backbone $W/tapt_cjpe_fix --text_fix --out ../runs/ft_tapt_fix_e2_s1 --epochs 2 --evals_per_epoch 2 --seed 1
python ft_doc.py --task cjpe --text_fix --out ../runs/ft_fix_e2 --epochs 2 --evals_per_epoch 2
python combine.py ft_tapt_fix_e2:e2 ft_tapt_fix_e2_s1:e2 ft_fix_e2:e2 tfidf_scratch --out ../submissions/P4_S03_tapt_fix

# P4_S04 (current best): two more adapted runs, then average everything of the new recipe
python ft_doc.py --task cjpe --backbone $W/tapt_cjpe_fix --text_fix --out ../runs/ft_tapt_fix_e2_s2 --epochs 2 --evals_per_epoch 2 --seed 2
python ft_doc.py --task cjpe --backbone $W/tapt_cjpe_fix --text_fix --out ../runs/ft_tapt_fix_lr4e5_e2 --epochs 2 --evals_per_epoch 2 --lr 4e-5
python combine.py ft_tapt_fix_e2:e2 ft_tapt_fix_e2_s1:e2 ft_tapt_fix_e2_s2:e2 ft_tapt_fix_lr4e5_e2:e2 ft_fix_e2:e2 tfidf_scratch \
    --out ../submissions/P4_S04_tapt_fix_all
```

Each fine-tuning run takes about 50 minutes (2 passes) on one 48 GB GPU and uses about 30 GB.
Use the `cs781` conda environment. Text is turned into tokens by our own BPE encoder (`../shared/scratch_bpe.py`),
which gives exactly the same token numbers as the Phase 1 tokenizer library (checked on 31,711 texts).
