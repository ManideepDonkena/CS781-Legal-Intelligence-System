"""Advanced Prior Case Retrieval (PCR) Engine.

Unites:
1. Phrase-BM25 (Unigrams, Bigrams, Trigrams with BM25 doc-length normalization)
2. Multi-scale Verbatim N-gram Fingerprinting (5, 6, 7, 8-grams with IDF)
3. Legal Citation & Bench Alignment
4. Dense Semantic Similarity (BGE-M3 + Pre-trained Phase 1 295M LM)
"""
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcr

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Phase_7_PCR"
RUNS_DIR = ROOT / "runs"

PAT_WORD = re.compile(r'\b[a-z0-9_]{2,}\b')
PAT_CLEAN = re.compile(r'[a-z0-9]+')
PAT_CITATION = re.compile(
    r'\b(?:\d{4}\s+)?(?:AIR|SCR|SCC|ITR|ILR|CRLJ|FLR|LLJ|COMP\s*CAS|IND\.\s*CAS\.)\s*(?:\(\d+\))?\s*\d+\b',
    re.IGNORECASE
)
PAT_BENCH = re.compile(r'BENCH:\s*([^\n]+)', re.IGNORECASE)

STOPWORDS = {
    'the', 'of', 'and', 'in', 'to', 'a', 'is', 'that', 'for', 'it', 'as', 'was', 'with', 'be', 'by', 'on',
    'not', 'this', 'are', 'or', 'from', 'at', 'which', 'an', 'had', 'have', 'were', 'been', 'there', 'what',
    'will', 'would', 'can', 'all', 'we', 'they', 'their', 'when', 'who', 'so', 'said', 'no', 'up', 'out'
}


def clean_tokens(text: str) -> list[str]:
    return PAT_CLEAN.findall(text.lower())


def extract_phrases(text_list: list[str]) -> list[str]:
    text = " ".join(text_list).lower()
    words = PAT_WORD.findall(text)
    filtered = [w for w in words if w not in STOPWORDS]
    feats = list(filtered)
    # bigrams
    for i in range(len(filtered) - 1):
        feats.append(filtered[i] + "__" + filtered[i + 1])
    # trigrams
    for i in range(len(filtered) - 2):
        feats.append(filtered[i] + "__" + filtered[i + 1] + "__" + filtered[i + 2])
    return feats


def extract_citations(text: str) -> set[str]:
    cits = set()
    for m in PAT_CITATION.finditer(text):
        cits.add(re.sub(r'\s+', ' ', m.group(0).lower()))
    return cits


def run_pipeline(split: str = "dev"):
    print("=" * 60)
    print(f"Running Advanced PCR Pipeline on split: {split.upper()}")
    print("=" * 60)

    queries, candidates = pcr.load_split(split, data_dir=DATA_DIR)
    q_ids = sorted(list(queries.keys()))
    c_ids = sorted(list(candidates.keys()))
    N_cand = len(c_ids)
    print(f"Loaded {len(q_ids)} queries, {N_cand} candidates.")

    # -------------------------------------------------------------
    # 1. Build Phrase-BM25 Index
    # -------------------------------------------------------------
    print("\n[1/4] Building Phrase-BM25 Index...")
    t0 = time.time()
    cand_tf = {}
    cand_lens = {}
    df_counts = Counter()

    for cid in c_ids:
        feats = extract_phrases(candidates[cid]["text"])
        cand_lens[cid] = len(feats)
        cand_tf[cid] = Counter(feats)
        for term in set(feats):
            df_counts[term] += 1

    avg_len = sum(cand_lens.values()) / N_cand
    k1, b = 1.5, 0.75
    # Keep terms with 1 <= df <= 250
    idf = {term: math.log((N_cand - df + 0.5) / (df + 0.5) + 1.0) for term, df in df_counts.items() if 1 <= df <= 250}

    bm25_index = defaultdict(list)
    for cid, tf_map in cand_tf.items():
        norm_factor = k1 * (1.0 - b + b * (cand_lens[cid] / avg_len))
        for term, tf in tf_map.items():
            if term in idf:
                w = (tf * (k1 + 1.0)) / (tf + norm_factor) * idf[term]
                if "__" in term:
                    w *= (2.5 if term.count("__") == 1 else 5.0)
                bm25_index[term].append((cid, w))

    print(f"  ✓ Built BM25 index with {len(bm25_index)} terms in {time.time() - t0:.1f}s")

    # Score Phrase-BM25
    print("  Scoring queries on Phrase-BM25...")
    phrase_scores = {qid: defaultdict(float) for qid in q_ids}
    for qid in q_ids:
        q_feats = Counter(extract_phrases(queries[qid]["text"]))
        for term, _ in q_feats.items():
            if term in bm25_index:
                for cid, w in bm25_index[term]:
                    phrase_scores[qid][cid] += w

    # -------------------------------------------------------------
    # 2. Multi-Scale Verbatim N-gram & Citation Index
    # -------------------------------------------------------------
    print("\n[2/4] Building Multi-Scale Verbatim N-gram & Citation Index...")
    t0 = time.time()
    ngram_index = defaultdict(list)
    ngram_df = Counter()
    ns = (5, 6, 7, 8)

    cand_cits = {cid: extract_citations(" ".join(candidates[cid]["text"][:6])) for cid in c_ids}

    cand_benches = {}
    for cid in c_ids:
        head = " ".join(candidates[cid]["text"][:5])
        bench_names = set()
        for b_str in PAT_BENCH.findall(head):
            cl = b_str.replace("<NAME>", "").replace(",", " ").strip().lower()
            for w in cl.split():
                if len(w) > 3 and w not in STOPWORDS:
                    bench_names.add(w)
        cand_benches[cid] = bench_names

    for cid in c_ids:
        text = " ".join(candidates[cid]["text"])
        tokens = clean_tokens(text)
        seen_cand = set()
        for n in ns:
            for i in range(len(tokens) - n + 1):
                ng = " ".join(tokens[i : i + n])
                if ng not in seen_cand:
                    seen_cand.add(ng)
                    ngram_index[ng].append(cid)
                    ngram_df[ng] += 1

    print(f"  ✓ Built verbatim index with {len(ngram_index)} n-grams in {time.time() - t0:.1f}s")

    # Score Verbatim N-grams & Citations
    print("  Scoring queries on Verbatim N-grams & Citations...")
    verbatim_scores = {qid: defaultdict(float) for qid in q_ids}
    for qid in q_ids:
        q_text = " ".join(queries[qid]["text"])
        q_lower = q_text.lower()
        q_tokens = clean_tokens(q_text)
        q_cits = extract_citations(q_text)

        # Citation matches
        for cid, ccits in cand_cits.items():
            overlap = q_cits & ccits
            if overlap:
                verbatim_scores[qid][cid] += len(overlap) * 50.0

        # Bench matches
        for cid, bnames in cand_benches.items():
            if any(bn in q_lower for bn in bnames):
                verbatim_scores[qid][cid] += 8.0

        # N-grams with IDF
        seen_query = set()
        for n in ns:
            scale_wt = (n ** 2.0)
            for i in range(len(q_tokens) - n + 1):
                ng = " ".join(q_tokens[i : i + n])
                if ng in seen_query:
                    continue
                seen_query.add(ng)

                if ng in ngram_index:
                    df = ngram_df[ng]
                    if df <= 30:
                        idf_val = math.log((N_cand - df + 0.5) / (df + 0.5) + 1.0)
                        val = scale_wt * idf_val
                        for cid in ngram_index[ng]:
                            verbatim_scores[qid][cid] += val

    # -------------------------------------------------------------
    # 3. Load Precomputed Dense Scores
    # -------------------------------------------------------------
    print("\n[3/4] Loading Precomputed Dense Scores (BGE-M3 & Phase 1 SC 295M LM)...")
    bge_scores = json.load(open(RUNS_DIR / "dense_bge" / f"{split}_scores.json", "r"))
    p1_scores = json.load(open(RUNS_DIR / "dense_phase1" / f"{split}_scores.json", "r"))

    # -------------------------------------------------------------
    # 4. Normalize & Ensemble Fusion
    # -------------------------------------------------------------
    print("\n[4/4] Normalizing and Computing Ensemble Consensus...")

    def normalize(d):
        out = {}
        for qid in q_ids:
            cdict = d[qid]
            vals = np.array([cdict.get(cid, 0.0) for cid in c_ids], dtype=np.float32)
            mn, mx = vals.min(), vals.max()
            denom = (mx - mn) if (mx - mn) > 1e-9 else 1.0
            out[qid] = {cid: float((cdict.get(cid, 0.0) - mn) / denom) for cid in c_ids}
        return out

    norm_phrase = normalize(phrase_scores)
    norm_verbatim = normalize(verbatim_scores)
    norm_bge = normalize(bge_scores)
    norm_p1 = normalize(p1_scores)

    # Weights tuned on Dev: Phrase-BM25: 0.50, Verbatim N-gram/Citation: 0.40, BGE-M3: 0.07, Phase1: 0.03
    ensemble_scores = {}
    for qid in q_ids:
        ensemble_scores[qid] = {
            cid: (
                0.50 * norm_phrase[qid][cid] +
                0.40 * norm_verbatim[qid][cid] +
                0.07 * norm_bge[qid][cid] +
                0.03 * norm_p1[qid][cid]
            )
            for cid in c_ids
        }

    # Top-7 predictions
    preds = {
        qid: sorted(ensemble_scores[qid].keys(), key=lambda c: ensemble_scores[qid][c], reverse=True)[:7]
        for qid in q_ids
    }

    # Evaluate if dev
    if split == "dev":
        res = pcr.evaluate_predictions(queries, preds)
        print("\n" + "=" * 60)
        print("ADVANCED PCR DEV EVALUATION METRICS:")
        print(f"Micro-F1@7:        {res['primary_metric_1']:.4f}")
        print(f"Micro-Precision@7: {res['secondary_metric_1']:.4f}")
        print(f"Micro-Recall@7:    {res['secondary_metric_2']:.4f}")
        print(f"True Positives:    {res['tp']} (FP: {res['fp']}, FN: {res['fn']})")
        print("=" * 60)

    # Save scores and predictions
    out_dir = RUNS_DIR / "advanced_pcr"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{split}_scores.json", "w") as f:
        json.dump(ensemble_scores, f)
    with open(out_dir / f"{split}_predictions.json", "w") as f:
        json.dump(preds, f, indent=2)

    # Also save separate Phrase-BM25 scores
    with open(out_dir / f"{split}_phrase_scores.json", "w") as f:
        json.dump({qid: {cid: phrase_scores[qid][cid] for cid in c_ids} for qid in q_ids}, f)

    print(f"[✓] Saved {split} outputs to {out_dir}")
    return ensemble_scores, preds


if __name__ == "__main__":
    split_arg = sys.argv[1] if len(sys.argv) > 1 else "dev"
    run_pipeline(split_arg)
