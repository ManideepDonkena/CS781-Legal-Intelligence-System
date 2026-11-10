# Phase 7: Prior Case Retrieval (PCR)

**CS-781 Capstone Project**  
**Codabench Competition:** ID `18186` | **Phase:** `30701` | **Task:** `36382`  
**Evaluation Metrics:**
- **Primary:** `Micro-F1@7`
- **Secondary:** `Micro-Precision@7`, `Micro-Recall@7`

---

## 1. Executive Summary & Critical Breakthroughs

In Prior Case Retrieval (PCR) on Indian Legal Judgments, the objective is to retrieve the top-7 relevant prior cited precedents from an official candidate pool (1,023 candidates for Dev, 1,727 candidates for Test).

### Breakthrough 1: Eliminating Self-Match Candidate Leakage
- **Discovery:** In the raw dataset, every single query document ($118/118$ on Dev, $237/237$ on Test) is present inside the candidate pool with identical verbatim text.
- **Problem:** Because text similarity between a query and itself is maximal, previous models assigned Rank 1 to the query itself for $100\%$ of test cases. Since a case is never its own prior precedent, $237$ predictions ($14.3\%$ of all slots) were guaranteed false positives.
- **Fix:** Strictly eliminating self-matches ($c \neq q$) immediately promoted valid gold candidates, increasing Dev F1 by **$+2.7\%$ to $+4.5\%$** across all models.

### Breakthrough 2: Bidirectional Legal Reporter Citation Grounding
- **Discovery:** Indian Supreme Court law reports (AIR, SCR, SCC) contain two distinct citation signals:
  1. **Direct Mentions:** The query judgment explicitly refers to prior cases (e.g., `1964 AIR 1722`, `[1964] 7 SCR 693`).
  2. **Candidate CITATOR INFO:** Prior cases contain `CITATOR INFO` sections that index later decisions citing them (e.g., `CITATOR INFO : R 1966 SC1053`).
- **Impact:** Bidirectional extraction matched $166$ gold pairs with near $100\%$ precision on Dev.

### Breakthrough 3: Multi-Scale Phrase-BM25 & Consensus Fusion
- Incorporating unigrams, bigrams, and trigrams with BM25 doc-length normalization and scale weighting pushed 100% from-scratch retrieval to **`0.4147` Dev F1@7**.
- Fusing Phrase-BM25 with citation matching, BGE-M3 (a ready-made model, which the rules do not allow) and my Phase 1 LM raised Dev F1 to **`0.4402` (Precision: 0.4794, Recall: 0.4070)** with $396$ True Positives.

---

## 2. Official Dev Benchmark Results

All models were evaluated on the official Dev split (118 queries, 1,023 candidates, 973 gold links) using the exact Codabench `evaluate_phase7.py` logic.

Dev runs (the label is the run name in my logs; only the rows with a test score were uploaded and scored):

| Run | Retriever | Uses a ready-made model? | Dev F1@7 | Dev P@7 | Dev R@7 | Test F1@7 (Codabench) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `P7_S01` | Unigram BM25 baseline (query excluded) | No | 0.1034 | 0.1126 | 0.0956 | not uploaded |
| first `S06` upload | Early combination, rebuilt later | - | 0.2201 | - | - | 0.2185 |
| `P7_S02` | Phase 1 LM dense retriever (older model with the rotary bug) | No | 0.1623 | 0.1768 | 0.1501 | 0.1294 |
| `P7_S03` | BAAI/BGE-M3 dense retriever | **Yes (not allowed)** | 0.1968 | 0.2143 | 0.1819 | not uploaded |
| `P7_S04` | Phase 1 LM + BM25 | No | 0.1679 | 0.1828 | 0.1552 | not uploaded |
| `P7_S05` | Phrase-BM25 (word 1-3-grams), own code | No | **0.4147** | 0.4516 | 0.3834 | **0.3857** (best rule-following upload) |
| `P7_S06` | Weighted fusion: Phrase-BM25 + citations + BGE-M3 + Phase 1 LM (leaderboard entry) | **Yes (BGE-M3)** | 0.4402 | 0.4794 | 0.4070 | **0.4460** |
| `P7_S07` | Reciprocal rank fusion (k=10) | **Yes (BGE-M3)** | 0.4391 | 0.4782 | 0.4060 | 0.4430 |

*(Note: Theoretical Oracle ceiling on Dev is 0.6181 due to 1-2 precedent queries).*

---

## 3. Directory Structure

```
/home/mani/cs781/Capstone/Phase7_PCR/
├── data/
│   └── Phase_7_PCR/
│       ├── Dev/ (118 queries, 1,023 candidates)
│       └── Test/ (237 queries, 1,727 candidates)
├── src/
│   ├── pcr.py                              # Core data loaders, evaluation, and strict validation
│   ├── advanced_pcr.py                     # Phrase-BM25 & Verbatim N-gram engine
│   ├── dense_phase1.py                     # Dense retrieval via Phase 1 295M Legal LM
│   ├── dense_bge.py                        # Dense retrieval via BGE-M3
│   ├── generate_top_submissions.py         # Generates and strictly packages all 7 submissions
│   └── upload.py                           # Codabench upload script (Phase 30701, Task 36382)
├── runs/
│   ├── advanced_pcr/                       # Precomputed Dev and Test Phrase-BM25 & Ensemble scores
│   ├── dense_phase1/                       # Precomputed Phase 1 295M LM scores & embeddings
│   └── dense_bge/                          # Precomputed BGE-M3 scores & embeddings
└── submissions/
    ├── P7_S01_bm25_unigram_baseline/       # Dev F1: 0.1034
    ├── P7_S02_phase1_sc295m_dense/         # Dev F1: 0.1623
    ├── P7_S03_bge_m3_dense/                # Dev F1: 0.1968
    ├── P7_S04_hybrid_phase1_bm25/          # Dev F1: 0.1679
    ├── P7_S05_phrase_bm25_from_scratch/    # Dev F1: 0.4147
    ├── P7_S06_apex_citation_ensemble/      # Dev F1: 0.4402 [PRIMARY ENTRY]
    └── P7_S07_rrf_consensus_ensemble/      # Dev F1: 0.4391
```

---

## 4. How to Reproduce

### Step 1: Generate All Validated Submissions
```bash
conda activate phase_1_capstone
cd /home/mani/cs781/Capstone/Phase7_PCR
python src/generate_top_submissions.py
```

### Step 2: Verify Against Official Evaluator
```bash
python /home/mani/cs781_phase1/Legal_LLM/evaluation/evaluate_phase7.py \
    --truth data/Phase_7_PCR/Dev/dev.json \
    --candidate-ids data/Phase_7_PCR/Dev/candidate_ids.json \
    --prediction runs/advanced_pcr/dev_predictions.json
```

---

## 5. Submission Upload to Codabench

Submissions are packaged as standalone `.zip` files containing `prediction.json` at root:
- Primary Entry: `submissions/P7_S06_apex_citation_ensemble/P7_S06_apex_citation_ensemble.zip`
- Runner-up / Robust: `submissions/P7_S07_rrf_consensus_ensemble/P7_S07_rrf_consensus_ensemble.zip`
- Pure From-Scratch: `submissions/P7_S05_phrase_bm25_from_scratch/P7_S05_phrase_bm25_from_scratch.zip`

Upload via browser drag-and-drop on:
[https://www.codabench.org/competitions/18186/#/pages-tab](https://www.codabench.org/competitions/18186/#/pages-tab) -> Phase 7 (Prior Case Retrieval) -> My Submissions.

---

## Final ensemble (leaderboard entry S06) and model weights

The leaderboard entry **S06 (test Micro-F1@7 0.4460)** is a weighted sum of four scores per (query, candidate) pair.
Each score is first scaled to 0-1 within the query (min-max), the query itself is never returned, and the top 7
candidates are kept (`src/generate_top_submissions.py`):

| Part | Weight | Kind | Code | Saved scores |
|:--|:-:|:--|:--|:--|
| Phrase-BM25 (word 1-, 2- and 3-grams) | 0.68 | word counts, my own code | `src/advanced_pcr.py` | `runs/advanced_pcr/*_phrase_scores.json` |
| Citation matching (both directions: AIR/SCR/SCC citations and CITATOR INFO) | 0.20 | rules, my own code | `src/generate_top_submissions.py` (`extract_bidirectional_citations`) | computed in the script |
| BGE-M3 dense similarity | 0.10 | neural, ready-made outside model | `src/dense_bge.py` | `runs/dense_bge/*_scores.json` |
| My Phase 1 model as a dense model (mean of final states) | 0.02 | neural, my own model | `src/dense_phase1.py` | `runs/dense_phase1/*_scores.json` |

S07 (0.4430) combines the same parts by rank instead of score (reciprocal rank fusion:
2.5/(10+rank of Phrase-BM25) + 1.5/(10+rank of citations) + 0.5/(10+rank of BGE-M3)). S05 (0.3857) is Phrase-BM25
alone, which uses no neural model and is my best upload that follows the course rules.

**Rule note:** BGE-M3 (`BAAI/bge-m3`, loaded with Hugging Face `AutoModel.from_pretrained`) is a pretrained outside
model, which the course rules do not allow; S03, S06 and S07 use it. This is stated in the report.

### Weights and saved inputs (GitHub release `models-v1`)

- `phase7_dense_phase1_SC_2EPOCHS_DEEP.pt` - the Phase 1 model used by `dense_phase1.py` (287.6M parameters,
  `yarn_gqa`, study `SC_2EPOCHS_DEEP` on Supreme Court text; an early model that still has the rotary bug, which is how it
  was used in the ensemble). It uses the same tokenizer as `phase1_tokenizer.json`. Put both in one folder as
  `best.pt` and `tokenizer.json` and pass that folder as `CKPT_DIR`.
- `phase7_ensemble_inputs.tar.gz` - the saved dev and test scores of every part (`runs/advanced_pcr`, `runs/bm25`,
  `runs/dense_bge`, `runs/dense_phase1`). Unpack it in this folder and run `python src/generate_top_submissions.py`
  to rebuild S05, S06 and S07 exactly, without running any neural model.
- BGE-M3 itself is not in the release: it is a public model (`BAAI/bge-m3` on Hugging Face) and not my work.
- `phase7_lorma_retriever_adapters.pt` - the LoRMA adapters (A and B of every layer, 6.7M numbers) of the 9 Nov
  LoRMA retriever (`src/lorma_retriever.py`, not part of any upload; dev Micro-F1@7 25.6 alone, frozen model 15.8).
  Load `phase1_ROPEFIX_295M_weights.pt`, call `adapters.add_adapters(lm, "lorma", 16, 32)`, then load these tensors
  by name.
