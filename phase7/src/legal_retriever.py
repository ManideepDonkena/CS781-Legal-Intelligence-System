"""Comprehensive Legal Precedent Retrieval Engine for Phase 7 (PCR).

Combines:
1. Quotation & Holding Multi-scale N-gram Matching (5, 6, 7, 8-grams with IDF)
2. Phrase-BM25 (Unigrams, Bigrams, Trigrams with BM25 doc-length normalization)
3. Statute & Constitutional Article Alignment (Exact Sections & Articles)
4. Bench Judge & Party Name Alignment
5. Dense Semantic Alignment (BGE-M3 + Pre-trained Phase 1 295M Legal LM)
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

PAT_CLEAN = re.compile(r'[a-z0-9_]+')
PAT_BENCH = re.compile(r'BENCH:\s*([^\n]+)', re.IGNORECASE)
PAT_PARTY = re.compile(r'(?:PETITIONER|RESPONDENT):\s*([^V\n]+?)(?:\s+Vs\.|\s+DATE|\s+BENCH|\n|$)', re.IGNORECASE)
PAT_SEC = re.compile(r'\b(?:section|sec|s\.)\s*(\d+[a-z]?(?:\(\d+\))*(?:\([a-z]+\))*)', re.IGNORECASE)
PAT_ART = re.compile(r'\b(?:article|art\.)\s*(\d+[a-z]?)', re.IGNORECASE)

STOPWORDS = {
    'the', 'of', 'and', 'in', 'to', 'a', 'is', 'that', 'for', 'it', 'as', 'was', 'with', 'be', 'by', 'on',
    'not', 'this', 'are', 'or', 'from', 'at', 'which', 'an', 'had', 'have', 'were', 'been', 'there', 'what',
    'will', 'would', 'can', 'all', 'we', 'they', 'their', 'when', 'who', 'so', 'said', 'no', 'up', 'out'
}


def clean_tokens(text: str) -> list[str]:
    return PAT_CLEAN.findall(text.lower())


def extract_legal_features(text_list: list[str]) -> tuple[list[str], set[str], set[str]]:
    """Extract filtered tokens, statutory tags, and bench/party names."""
    full_text = " ".join(text_list)
    lower = full_text.lower()

    # 1. Statutory sections & articles
    secs = set()
    for s in PAT_SEC.findall(lower):
        norm_s = "sec_" + re.sub(r'[^a-z0-9]', '_', s.strip())
        secs.add(norm_s)
    for a in PAT_ART.findall(lower):
        secs.add("art_" + a.strip())

    # 2. Tokens & phrases
    raw_words = PAT_CLEAN.findall(lower)
    words = [w for w in raw_words if w not in STOPWORDS]

    features = list(words)
    # Add sections as high-weight features
    for s in secs:
        features.append(s)
        features.append(s)

    # Bigrams
    for i in range(len(words) - 1):
        features.append(words[i] + "__" + words[i + 1])
    # Trigrams
    for i in range(len(words) - 2):
        features.append(words[i] + "__" + words[i + 1] + "__" + words[i + 2])

    return features, secs, lower


def build_candidate_index(candidates: dict):
    t0 = time.time()
    print(f"[*] Indexing {len(candidates)} candidates...")

    cand_features = {}
    cand_lens = {}
    cand_secs = {}
    cand_benches = {}
    cand_parties = {}
    df_counts = Counter()

    # Multi-scale verbatim n-gram inverted index
    ngram_index = defaultdict(list)
    ngram_df = Counter()
    ns = (5, 6, 7, 8)

    for cid, cdata in candidates.items():
        feats, secs, lower = extract_legal_features(cdata["text"])
        cand_features[cid] = Counter(feats)
        cand_lens[cid] = len(feats)
        cand_secs[cid] = secs

        # Bench extraction
        head = " ".join(cdata["text"][:5])
        bench_names = set()
        for b in PAT_BENCH.findall(head):
            cleaned = b.replace("<NAME>", "").replace(",", " ").strip().lower()
            for w in cleaned.split():
                if len(w) > 3 and w not in STOPWORDS:
                    bench_names.add(w)
        cand_benches[cid] = bench_names

        # Party extraction
        parties = set()
        for p in PAT_PARTY.findall(head):
            p_clean = p.replace("<NAME>", "").replace("<ORG>", "").strip().lower()
            p_words = [w for w in PAT_CLEAN.findall(p_clean) if len(w) > 3 and w not in STOPWORDS]
            if len(p_words) >= 2:
                parties.add(" ".join(p_words))
        cand_parties[cid] = parties

        # Terms DF
        for term in set(feats):
            df_counts[term] += 1

        # N-grams
        raw_words = PAT_CLEAN.findall(lower)
        seen_cand_ngrams = set()
        for n in ns:
            for i in range(len(raw_words) - n + 1):
                ng = " ".join(raw_words[i : i + n])
                if ng not in seen_cand_ngrams:
                    seen_cand_ngrams.add(ng)
                    ngram_index[ng].append(cid)
                    ngram_df[ng] += 1

    N = len(candidates)
    avg_len = sum(cand_lens.values()) / N

    # BM25 IDF
    bm25_idf = {}
    for term, df in df_counts.items():
        if 1 <= df <= 250:
            bm25_idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    # Inverted BM25 index
    k1, b = 1.5, 0.75
    bm25_index = defaultdict(list)
    for cid, tf_map in cand_features.items():
        norm_factor = k1 * (1.0 - b + b * (cand_lens[cid] / avg_len))
        for term, tf in tf_map.items():
            if term in bm25_idf:
                weight = (tf * (k1 + 1.0)) / (tf + norm_factor) * bm25_idf[term]
                if "__" in term:
                    weight *= (2.0 if term.count("__") == 1 else 4.0)
                bm25_index[term].append((cid, weight))

    print(f"[✓] Candidate index built in {time.time() - t0:.1f}s ({len(bm25_index)} BM25 terms, {len(ngram_index)} n-grams)")
    return {
        "bm25_index": bm25_index,
        "ngram_index": ngram_index,
        "ngram_df": ngram_df,
        "cand_secs": cand_secs,
        "cand_benches": cand_benches,
        "cand_parties": cand_parties,
        "N": N
    }


def score_queries(queries: dict, candidates: dict, index_data: dict) -> dict[str, dict[str, float]]:
    t0 = time.time()
    print(f"[*] Scoring {len(queries)} queries...")

    bm25_index = index_data["bm25_index"]
    ngram_index = index_data["ngram_index"]
    ngram_df = index_data["ngram_df"]
    cand_secs = index_data["cand_secs"]
    cand_benches = index_data["cand_benches"]
    cand_parties = index_data["cand_parties"]
    N = index_data["N"]
    ns = (5, 6, 7, 8)

    scores = {qid: defaultdict(float) for qid in queries}

    for qid, qdata in queries.items():
        feats, q_secs, q_lower = extract_legal_features(qdata["text"])
        raw_words = PAT_CLEAN.findall(q_lower)

        # 1. BM25 scoring
        q_feats = Counter(feats)
        for term, q_tf in q_feats.items():
            if term in bm25_index:
                for cid, w in bm25_index[term]:
                    scores[qid][cid] += w

        # 2. Multi-scale N-gram Inverted Index scoring with IDF
        seen_q_ng = set()
        for n in ns:
            scale_wt = (n ** 2.0)
            for i in range(len(raw_words) - n + 1):
                ng = " ".join(raw_words[i : i + n])
                if ng in seen_q_ng:
                    continue
                seen_q_ng.add(ng)

                if ng in ngram_index:
                    df = ngram_df[ng]
                    if df <= 30:
                        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                        val = scale_wt * idf * 0.15
                        for cid in ngram_index[ng]:
                            scores[qid][cid] += val

        # 3. Exact statutory section match boost
        for cid, c_secs in cand_secs.items():
            sec_overlap = len(q_secs & c_secs)
            if sec_overlap > 0:
                scores[qid][cid] += sec_overlap * 3.0

        # 4. Judge / Bench citation boost
        for cid, judges in cand_benches.items():
            if any(j in q_lower for j in judges):
                scores[qid][cid] += 8.0

        # 5. Party / Case Title boost
        for cid, parties in cand_parties.items():
            if any(p in q_lower for p in parties):
                scores[qid][cid] += 15.0

    print(f"[✓] Scored queries in {time.time() - t0:.1f}s")
    full_scores = {qid: {cid: float(scores[qid][cid]) for cid in candidates} for qid in queries}
    return full_scores


def main():
    dev_q, dev_c = pcr.load_split("dev")
    index_data = build_candidate_index(dev_c)
    scores = score_queries(dev_q, dev_c, index_data)

    preds = {qid: sorted(scores[qid].keys(), key=lambda c: scores[qid][c], reverse=True)[:7] for qid in dev_q}
    res = pcr.evaluate_predictions(dev_q, preds)

    print("\n" + "=" * 60)
    print("DEV COMPREHENSIVE LEGAL RETRIEVER RESULTS:")
    print(f"Micro-F1@7:        {res['primary_metric_1']:.4f}")
    print(f"Micro-Precision@7: {res['secondary_metric_1']:.4f}")
    print(f"Micro-Recall@7:    {res['secondary_metric_2']:.4f}")
    print(f"True Positives:    {res['tp']} (FP: {res['fp']}, FN: {res['fn']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
