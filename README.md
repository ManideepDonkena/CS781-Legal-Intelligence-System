# CS781 Capstone: Indian Legal Language Processing

**Author:** Donkena Manideep, roll no. 261040623, IIT Kanpur (dmanideep26@iitk.ac.in)
**Codabench user:** `d_261040623` (competition 18186)
**Report:** `report/CS781-CP-Manideep-Donkena-261040623-report.pdf` in the submission archive
**Model weights:** GitHub release [`models-v1`](https://github.com/ManideepDonkena/CS781-Legal-Intelligence-System/releases/tag/models-v1)

I trained a 287.6M-parameter language model from random weights on the course's Indian legal texts only (Phase 1),
then used it, together with word-count models I wrote myself, for the six downstream tasks (Phases 2-7).
All numbers below are the ones in the report; they were checked against Codabench and the run logs on 9 Nov 2026.

## Leaderboard entry per phase (hidden test set)

| Phase | Task | Main metric | Score | Rank (of 50 shown) | Uploads |
|:-:|:--|:--|:-:|:-:|:-:|
| 1 | Legal language model | BERTScore F1 | 0.8376 | 23 | 33 |
| 2 | Legal named entities (L-NER) | Strict macro-F1 | 0.6783 | 4 | 6 (5 scored) |
| 3 | Rhetorical roles | Macro-F1 | 0.6242 | 7 | 8 |
| 4 | Court judgment prediction | Macro-F1 | not uploaded | - | 0 |
| 5 | Statute identification | Macro-F1 | 0.4717 | 2  | 6 |
| 6 | Summarization | BERTScore F1 | 0.8557 | 5 | 6 |
| 7 | Prior case retrieval | Micro-F1@7 | 0.4460* | 34 | 5 |

- Phase 4: six prediction files were finished (own held-out score up to 84.3) but the phase closed before upload.
- *The Phase 7 entry uses the pretrained BGE-M3 model, which the course rules do not allow. The best upload that
  follows the rules (Phrase-BM25 only, S05) scored 0.3857.
- The Phase 1 uploads came from models that still had the rotary bug (see below); the fixed model came later.

## Folder layout

```
phase1/final_model/   code that trained ROPEFIX_295M (the model used in Phases 2-7) - see its README;
                      rope.py and attention.py are my Assignment 3 files, unchanged (with their TODOs)
phase1/               earlier Phase 1 experiments, tokenizer study, generation for the Phase 1 uploads
phase2/src/           L-NER: lner.py (tokenisation with exact offsets, side clue, CRF, tagger), train_ner.py,
                      stage2.py (whole-document check), make_submission.py;
                      adapters.py + train_adapt.py (9 Nov LoRA / LoRMA / distillation study), results_adapt/;
                      q4_peft_lorma.py = my Assignment 4 LoRMA functions, unchanged (with their TODOs)
phase3/src/           rhetorical roles: embed.py (frozen LM + 768 tokens of earlier text), train_rr.py, combine.py
phase4/src/           judgment prediction: text_fix.py (spelling repair), tapt.py, ft_doc.py, combine.py
phase5/src/           statute identification: lsi.py, ft (via phase4 ft_doc.py --task lsi), merge_xfit.py, combine.py
phase6/src/           summarization: extractive.py, abstractive.py, hybrid.py, summ.py (own ROUGE)
phase7/src/           prior case retrieval: advanced_pcr.py (Phrase-BM25), citation_ngram.py,
                      generate_top_submissions.py (final fusion), dense_bge.py (BGE-M3, see note above)
shared/               own TF-IDF / logistic regression / nearest neighbours (scratch_text.py), own BPE encoder
tests/                integrity tests (python tests/test_system_integrity.py)
notebooks/            one Kaggle notebook per phase (phase1_kaggle.ipynb ... phase7_kaggle.ipynb), made by make_notebooks.py
lorma_results/        logs of the 9 Nov LoRMA runs of Phases 3-7; summarize_lorma_phases.py collects them
```

## Code reused from my assignments

| Assignment file (unchanged, TODOs kept) | Used by | How |
|:--|:--|:--|
| A3 `rope.py`: `build_rope_cache`, `rotate_half`, `apply_rope` | `phase1/final_model/modern_arch.py` | the model's RoPE tables and the rotation of q and k |
| A3 `attention.py`: `CausalSelfAttention` | `modern_arch.CausalSelfAttention` (subclass) | its `__init__` builds q/k/v/out projections, `n_rep`, `d_head`, `attention_scale`, causal mask; `use_sdpa=False` runs its explicit scores-mask-softmax steps |
| A4 `q4_peft.py` section 4.4: `make_lorma_factors`, `lorma_reordered`, `lorma_dense`, `LORMA_VARIANT` | `phase2/src/adapters.py` (`LoRMALinear`) | factor initialisation and the adapted layer itself |

Class names follow Assignment 3 (`CausalSelfAttention`, `FeedForward`, `Block`, `DecoderLM`; the old names
`ModernAttention`, `SwiGLU`, `ModernTransformerBlock`, `ModernCausalLM` remain as aliases). What could not be shared,
and why: (1) the stored weight names (`token`, `layers`, `attn_norm`, `ffn_norm`, `norm`, `lm_head`) differ from A3's
(`tok_emb`, `blocks`, `ln1`, `ln2`, `ln_f`, `head`); renaming them would break the trained checkpoints. (2) A3's
`Block`, `FeedForward` and `DecoderLM` have other features (LayerNorm, GELU, sinusoidal positions, embeddings
scaled by sqrt(d)), so the capstone versions are new classes with the same names and structure. (3) A4's
`LoRMAConv1D` wraps GPT-2's Conv1D (weight stored [in, out]); my model uses nn.Linear ([out, in]), so
`LoRMALinear` is the same class for nn.Linear, built on the A4 functions. (4) The 287.6M model trains with
PyTorch's fused attention kernel (memory at 2,048 tokens); the explicit A3 computation gives the same loss
(3.1844 both, checked on 8 dev cases), so both are kept.

Check on 9 Nov after the change (old code vs new code, same checkpoint, 8 dev cases x 2,048 tokens):
ROPEFIX_295M loss 2.929502 vs 2.929502 (largest logit difference 4e-5); 4,096-token YaRN mode 3.4278 both;
pre-fix FINAL_295M (rotary-bug path) identical to the last digit. So no reported number changes.

## Phase 1: the language model

- **Final model `ROPEFIX_295M`:** 23 layers, width 1024, 16 query heads sharing 4 key/value heads, QK-RMSNorm,
  RMSNorm, SwiGLU, rotary positions with YaRN scaling, tied embeddings, 2,048-token context, 32,000-piece BPE
  vocabulary: **287,557,504 parameters**. AdamW, learning rate 1e-3 (warm-up 915 steps), weight decay 0.1,
  45,776 steps over 3 billion tokens, 38.7 GPU-hours. Validation loss **1.950**.
- **Rotary bug:** the cos/sin tables were stored in the integer type of the token ids, so attention was uniform.
  Fixing it lowered the validation loss from 2.95 to 1.950 (287.6M) and from 3.32 to 2.124 (98.9M).
  The models named "112M" have 98.9M parameters.
- **Tokenizer:** BPE learned only from course text with the Hugging Face `tokenizers` trainer (no ready-made
  tokenizer). From Phase 4 on, my own encoder `shared/scratch_bpe.py` gives identical output.
- **Data:** the 3B-token file repeats Supreme Court text 4-5 times; a wider mix (`ROPEFIX_112M_WIDE`) was also trained.

## Phases 2-7 in short

- **Phase 2:** frozen 98.9M model, learned layer mix, BiLSTM + CRF, name swapping, and a "side of Versus" clue
  (appellant part, respondent part, both, neither); a gradient-boosted whole-document check decides one label per
  string. Cross-validation 60.8 -> 65.0 with the document check.
  **Adaptation study (9 Nov, offline, not uploaded):** on 5 held-out splits, strict macro-F1 after copying labels:
  frozen 57.1, top-6 layers 58.0, full fine-tuning 58.1, LoRA 57.6, **LoRMA 58.6**, 98.9M student alone 59.9,
  student distilled from the 287.6M model 58.3. Run with `bash run_adapt_study.sh && python summarize_adapt.py`.
- **LoRMA in Phases 3-7 (9 Nov, offline, not uploaded; one different idea per phase):** see the table below.
- **Phase 3:** frozen 287.6M model reads each sentence with up to 768 tokens of earlier text; attention over words,
  position, and a document BiLSTM. Cross-validation 42.4 -> 49.6.
- **Phase 4:** repair of the "co"/"no" corruption (98% of damaged words fixed), TAPT, fine-tuning on the last 2,048
  tokens, averaging five runs: honest half-dev score 84.3.
- **Phase 5:** cross-fitting on dev, my own TF-IDF + the fine-tuned model, mix and threshold chosen per statute:
  honest dev 46.3, test 0.4717.
- **Phase 6:** hybrid summary: a 150-word written opening from the fine-tuned model, then picked sentences.
- **Phase 7:** Phrase-BM25 (word 1-3-grams), excluding the query itself, citation matching, score fusion.

## LoRMA in Phases 3-7 (9 Nov)

Same LoRMA code (`phase2/src/adapters.py`, built on my Assignment 4 functions), one different idea per phase,
compared with the full fine-tuning run of the same data and settings (from that run's log). Logs: `lorma_results/`.

| Phase | LoRMA idea | Trained | LoRMA | Full fine-tuning | Script |
|:-:|:--|--:|:-:|:-:|:--|
| 3 | adapters only in the top 8 of 23 blocks (no backward pass through the lower 15) | 2.3M | 47.7 | 48.0 | `phase3/src/run_lorma_study.sh` |
| 4 | LoRMA on top of the TAPT model (adapter lr 5e-5; 5e-4 diverged, 2e-4 collapsed) | 6.7M | 79.9 | 80.7 after 1 epoch (82.7 after 2) | `phase4/src/run_lorma_study.sh` |
| 5 | rank 64, rank-stabilised scale alpha/sqrt(r) | 26.9M | **38.9** | 37.5 after 1 epoch (40.3 after 3) | `phase5/src/run_lorma_study.sh` |
| 6 | feed-forward layers only, folded into W before generating | 4.8M | R-1/2/L 50.9/25.8/26.3 | 52.0/27.0/27.3 | `phase6/src/run_lorma_study.sh` |
| 7 | contrastive retriever, BM25 hard negatives | 6.7M | 25.6 (frozen 15.8) | Phrase-BM25 41.5; fused (fixed 0.3) 36.7 | `phase7/src/lorma_retriever.py` |

Scores: P3 sentence macro-F1 over 5 held-back folds; P4 dev macro-F1 at the dev median cut-off; P5 dev macro-F1 with
cut-offs tuned on the other half; P6 ROUGE on 100 dev cases (written part only); P7 dev Micro-F1@7.
`python summarize_lorma_phases.py` collects them.

## Kaggle notebooks

`notebooks/phaseN_kaggle.ipynb` (N = 1..7): set `DATA_INPUT` to the attached competition data, turn on GPU and
Internet, run all. Each one gets the code (attached dataset or this repository, which must then be public),
downloads the weights from the `models-v1` release, recreates the folder layout the code expects, then runs the
phase's main recipe and its LoRMA run. `QUICK = True` is a short smoke test. 

## Reproduce

```bash
pip install -r requirements.txt
python tests/test_system_integrity.py
```

Main commands per phase (each phase README has more detail):

- Phase 1: see `phase1/final_model/README.md`
- Phase 2: `train_ner.py`, then `stage2.py` (`stage2_cv.sh` for repeated cross-validation); study: `run_adapt_study.sh`
- Phase 3: `embed.py`, `train_rr.py --full_only`, `combine.py`
- Phases 4-5: `tapt.py`, `ft_doc.py --task cjpe|lsi [--text_fix] [--dev_fold k]`, `merge_xfit.py`, `combine.py [--mix_split]`
- Phase 6: `extractive.py --xfit 5`, `abstractive.py train|generate`, `hybrid.py --keep 150`
- Phase 7: `generate_top_submissions.py`
