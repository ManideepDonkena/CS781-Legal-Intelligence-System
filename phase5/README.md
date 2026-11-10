# Phase 5: Legal Statute Identification

For each case (facts of an Indian criminal case with names, section numbers and act names masked as `<ENTITY>`,
`<SECTION>`, `<ACT>`), predict which of 100 statutes apply. Score: macro-F1 over the 100 statutes (primary),
micro-F1 and subset accuracy (secondary).

## Data facts

- Train 42,750 cases, dev 10,181, test 13,019. Every case has 1-31 statutes (3.7 on average).
- Even the rarest statute has 272 training cases, but the statute mix of dev differs from train
  (per-statute train/dev count ratio ranges from 2.7 to 7.4 instead of about 4.2 everywhere).
- Cases are fairly short: median about 620 tokens; 79% fit in 2,048 tokens.

## Method

Two families of models, averaged with half the weight each:

1. **Word-count family**, all written from scratch in `../shared/scratch_text.py` (no scikit-learn):
   - TF-IDF of words and word pairs + one logistic regression per statute,
   - the same with letter sequences of length 3-5 inside words,
   - nearest neighbours: similarity-weighted vote of the 30 most similar labelled cases.
   (`src/baseline_tfidf.py` and `src/knn_tfidf.py` are the earlier scikit-learn versions, kept as records; they
   give the same scores.)
2. **Fine-tuned family** (GPU): our own Phase 1 language model (`ROPEFIX_295M`, trained from scratch in Phase 1)
   fine-tuned as a 100-output classifier on the first 1,536 + last 512 tokens of each case
   (`../Phase4_CJPE/src/ft_doc.py --task lsi --tail_tokens 512`), 3-4 passes, learning rate 2e-5 (head 1e-3).

**Using dev for training without losing an honest score (cross-fitting).** Dev is split into folds (5 for the
word-count models, 2 for the fine-tuned model). Each dev fold is predicted by a model trained on train + the other
dev folds; the test set is predicted by models that saw all of dev (word counts) or by the average of the fold
models (fine-tuned). Every dev prediction therefore comes from a model that never saw that case.

**Decisions.** One threshold per statute, chosen on dev to maximise that statute's F1; a case with no statute
above its threshold gets its single most likely statute. "Honest" dev scores choose the thresholds on one half of
dev and score the other half (5 random halvings, both ways).

## Results (dev macro-F1, honest)

| Model | Train only | Train + dev (cross-fitted) |
|---|---|---|
| TF-IDF words | 41.0 | 43.2 |
| TF-IDF letters | 39.9 | 42.1 |
| Nearest neighbours | - | 38.2 |
| Fine-tuned Phase 1 model (first 1,536 + last 512 tokens, 3 passes) | 39.6 | 41.2 (second seed 40.6) |
| Same with positive cases weighted up (`--pos_weight 0.5`, 4 passes) | 39.7 | 41.2 |

The word-count rows are from our own code; the scikit-learn versions scored 43.23 / 42.05 / 38.16 (cross-fitted),
i.e. the same.

Weighting positive cases up (`--pos_weight 0.5`) gave no gain on its own (39.7 vs 39.6), but it is kept as one of
the three fine-tuned runs because a slightly different run adds variety to the average.

Tried and dropped (no gain on dev): per-statute attention pooling (39.6), first 2,048 tokens only (39.1), 5 passes (39.1), word triples (40.5), extra weight on
rare statutes in the logistic regression (40.5), a second-stage model over all statutes' scores (`src/stack.py`),
reading the whole case in chunks (+0.4 at best), averaging standardised log-odds, one shared threshold for all
statutes (-0.8), pulling thresholds toward a shared one (+0.2, noise level).

**Submission P5_S02 (`submissions/P5_S02_scratch_xf_final`): honest dev macro-F1 45.5** (47.3 with thresholds
tuned and measured on all of dev; micro-F1 56.0; 4.3 statutes per test case). It averages, with half the weight
per family, the three cross-fitted word-count models and the three cross-fitted fine-tuned pairs. The earlier
`P5_S01` (45.0) used library word-count models and train-only fine-tuned runs; P5_S02 replaces it.

## Improvements added on 25 October evening (honest dev macro-F1)

- **Single-token placeholders** (`ft_doc.py --text_fix`, `lsi.compact_masks`): `<ENTITY>`, `<SECTION>` and `<ACT>`
  each cost 5 tokens (about 280 tokens per case); they become `X`, `SECTION`, `ACT`, 1 token each.
  Fine-tuned model, training cases only: 39.6 -> 40.0.
- **Task-adaptive pretraining** (`../Phase4_CJPE/src/tapt.py --task lsi --text_fix --tail_tokens 512`): one pass
  of next-word training on the train + dev windows (never test text) before fine-tuning. Held-out next-word error
  2.55 -> 1.87. Fine-tuned model: 40.0 -> 40.3.
- **Higher learning rate** (5e-5 instead of 2e-5): 40.3 -> 41.0 (training cases only); cross-fitted pairs 42.5 and
  41.9 (the original recipe's pairs: 41.2, 40.6, 41.2). The two higher-rate pairs together reach 43.2, as good as
  the best word-count model.
- **Each statute chooses its own mix** of the two families (`combine.py --mix_split 3`): for every statute one of
  six shares (0, 0.2, ..., 1) of the fine-tuned family and a threshold are chosen on dev. 46.3 against 45.5 for
  the fixed half/half average. A three-group mix over-fitted (45.9).
- Tried without gain: extra weight on dev cases in the word-count models (`wordcount_run.py --dev_weight`), and
  pulling thresholds toward a shared one. `shift_check.py` shows the test text looks more like dev than like train.

**Submission P5_S04 (`submissions/P5_S04_mix_all`): honest dev macro-F1 46.6** (49.3 with everything tuned and
measured on all of dev; micro-F1 57.7; 4.1 statutes per test case). Per-statute mix of the word-count family
(3 cross-fitted models) and the fine-tuned family (all 7 cross-fitted pairs). Replaces P5_S02 (45.5) and P5_S03
(46.4).

## Rebuild the submission

```
cd ../shared      # word-count family, own code
python wordcount_run.py --task lsi --out tfidf_scratch_xf5 --xfit 5
python wordcount_run.py --task lsi --out tfidf_c5_scratch_xf5 --letters --ngram 5 --max_features 1000000 --xfit 5
python wordcount_run.py --task lsi --out knn30_scratch_xf5 --model knn --xfit 5
cd ../Phase4_CJPE/src   # fine-tuned family, one run per dev half (each about 70-95 minutes on one GPU)
R=../../Phase5_LSI/runs
for f in 0 1; do
  python ft_doc.py --task lsi --out $R/xf_h1536t512_e3_f$f    --epochs 3 --tail_tokens 512 --dev_fold $f
  python ft_doc.py --task lsi --out $R/xf_h1536t512_e3_s1_f$f --epochs 3 --tail_tokens 512 --dev_fold $f --seed 1
  python ft_doc.py --task lsi --out $R/xf_pw05_e4_f$f         --epochs 4 --tail_tokens 512 --dev_fold $f --pos_weight 0.5
done
cd ../../Phase5_LSI/src
python merge_xfit.py xf_h1536t512_e3    xf_h1536t512_e3_f0    xf_h1536t512_e3_f1    --tag e3
python merge_xfit.py xf_h1536t512_e3_s1 xf_h1536t512_e3_s1_f0 xf_h1536t512_e3_s1_f1 --tag e3
python merge_xfit.py xf_pw05_e4         xf_pw05_e4_f0         xf_pw05_e4_f1         --tag e4
python combine.py tfidf_scratch_xf5 tfidf_c5_scratch_xf5 knn30_scratch_xf5 \
    xf_h1536t512_e3:e3 xf_h1536t512_e3_s1:e3 xf_pw05_e4:e4 --out ../submissions/P5_S02_scratch_xf_final

# P5_S04 (current best): adapted model, single-token placeholders, two learning rates, per-statute mix
W=/workspace/models/cs781_phase45
cd ../../Phase4_CJPE/src
python tapt.py --task lsi --out $W/tapt_lsi_compact --text_fix --tail_tokens 512
for f in 0 1; do
  python ft_doc.py --task lsi --backbone $W/tapt_lsi_compact --text_fix --tail_tokens 512 --dev_fold $f --out $R/xf_tapt_compact_e3_f$f --epochs 3
  python ft_doc.py --task lsi --backbone $W/tapt_lsi_compact --text_fix --tail_tokens 512 --dev_fold $f --out $R/xf_tapt_compact_e3_s1_f$f --epochs 3 --seed 1
  python ft_doc.py --task lsi --backbone $W/tapt_lsi_compact --text_fix --tail_tokens 512 --dev_fold $f --out $R/xf_tapt_compact_lr5e5_f$f --epochs 3 --lr 5e-5
  python ft_doc.py --task lsi --backbone $W/tapt_lsi_compact --text_fix --tail_tokens 512 --dev_fold $f --out $R/xf_tapt_compact_lr5e5_s1_f$f --epochs 3 --lr 5e-5 --seed 1
done
cd ../../Phase5_LSI/src
for r in xf_tapt_compact_e3 xf_tapt_compact_e3_s1 xf_tapt_compact_lr5e5 xf_tapt_compact_lr5e5_s1; do
  python merge_xfit.py $r ${r}_f0 ${r}_f1 --tag e3
done
python combine.py tfidf_scratch_xf5 tfidf_c5_scratch_xf5 knn30_scratch_xf5 \
    xf_tapt_compact_lr5e5:e3 xf_tapt_compact_lr5e5_s1:e3 xf_tapt_compact_e3:e3 xf_tapt_compact_e3_s1:e3 \
    xf_h1536t512_e3:e3 xf_h1536t512_e3_s1:e3 xf_pw05_e4:e4 --mix_split 3 --out ../submissions/P5_S04_mix_all
```

Use the `cs781` conda environment. A fine-tuning run takes about 20-23 minutes per pass on one 48 GB GPU and
uses about 30 GB.
