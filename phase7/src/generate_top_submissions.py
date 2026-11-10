"""Generate top-performing, strictly validated submissions for Phase 7 (PCR).

Key innovations:
1. Strict self-match elimination (every query is in the candidate pool; predicting qid consumes rank 1 as a false positive).
2. Bidirectional legal citation grounding:
   - Extract candidate reporter citations (AIR, SCR, SCC) and match against query body
   - Extract query reporter citations and match against candidate CITATOR INFO
3. Multi-scale Phrase-BM25 (unigrams, bigrams, trigrams with length normalization and IDF pruning)
4. Dense semantic similarity (BGE-M3 + Phase 1 SC 295M Legal LM)
5. Robust linear fusion with dev-tuned weights
6. Reciprocal Rank Fusion (RRF) alternative
"""
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcr

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Phase_7_PCR"
RUNS_DIR = ROOT / "runs"
SUB_DIR = ROOT / "submissions"

PAT_AIR = re.compile(r'CITATION:\s*(\d{4})\s+AIR\s+(\d+)', re.I)
PAT_SCR = re.compile(r'(\d{4})\s+SCR\s*(?:\((\d+)\))?\s*(\d+)', re.I)
PAT_Q_AIR = re.compile(r'CITATION:\s*(\d{4})\s+AIR\s+(\d+)', re.I)


def extract_bidirectional_citations(queries, candidates):
    """Compute bidirectional citation link scores between queries and candidates."""
    q_ids = sorted(queries.keys())
    c_ids = sorted(candidates.keys())

    # Pre-extract candidate reporter citations
    cand_air = {}
    cand_scr = {}
    cand_heads = {}
    for cid in c_ids:
        head = " ".join(candidates[cid]["text"][:10])
        cand_heads[cid] = head
        m = PAT_AIR.search(head)
        if m:
            cand_air[cid] = (m.group(1), m.group(2))
        m2 = PAT_SCR.search(head)
        if m2:
            cand_scr[cid] = (m2.group(1), m2.group(2), m2.group(3))

    cit_scores = {qid: defaultdict(float) for qid in q_ids}

    for qid in q_ids:
        q_text = " ".join(queries[qid]["text"]).lower()
        q_head = " ".join(queries[qid]["text"][:6])

        # Query citation patterns for searching candidate CITATOR INFO
        mq = PAT_Q_AIR.search(q_head)
        compiled_q = []
        if mq:
            yr, pg = mq.group(1), mq.group(2)
            patterns = [
                rf"{yr}\s*sc\s*{pg}",
                rf"{yr}\s*sc{pg}",
                rf"{yr}\s+air\s+{pg}",
                rf"{yr}\s+air\s+\(sc\)\s+{pg}",
            ]
            compiled_q = [re.compile(p, re.I) for p in patterns]

        for cid, head in cand_heads.items():
            if cid == qid:
                continue

            # 1. Query citation found in candidate CITATOR INFO
            for cp in compiled_q:
                if cp.search(head):
                    cit_scores[qid][cid] += 50.0
                    break

            # 2. Candidate AIR citation found in query text
            if cid in cand_air:
                yr, pg = cand_air[cid]
                forms = [f"{yr} air {pg}", f"{yr} sc {pg}", f"air {yr} sc {pg}", f"{yr} air (sc) {pg}"]
                if any(f in q_text for f in forms):
                    cit_scores[qid][cid] += 40.0

            # 3. Candidate SCR citation found in query text
            if cid in cand_scr:
                yr, vol, pg = cand_scr[cid]
                forms = [f"{yr} scr {pg}", f"{yr} s.c.r. {pg}"]
                if vol:
                    forms.extend([
                        f"{yr} ({vol}) scr {pg}",
                        f"({yr}) {vol} scr {pg}",
                        f"[{yr}] {vol} scr {pg}",
                        f"{vol} scr {pg}",
                    ])
                if any(f in q_text for f in forms):
                    cit_scores[qid][cid] += 30.0

    return cit_scores


def normalize_scores(score_dict, q_ids, c_ids):
    """Normalize score dictionary per query to [0, 1] range, excluding self-match."""
    out = {}
    for qid in q_ids:
        vals = [score_dict[qid].get(c, 0.0) for c in c_ids if c != qid]
        mn, mx = min(vals), max(vals)
        rng = (mx - mn) if (mx - mn) > 1e-9 else 1.0
        out[qid] = {c: (score_dict[qid].get(c, 0.0) - mn) / rng for c in c_ids if c != qid}
    return out


def get_top_k_no_self(score_dict, q_ids, k=7):
    """Extract top-k candidate IDs per query, strictly excluding the query itself."""
    preds = {}
    for qid in q_ids:
        s = score_dict[qid]
        ranked = [c for c in sorted(s.keys(), key=lambda c: s[c], reverse=True) if c != qid]
        preds[qid] = ranked[:k]
    return preds


def run_pipeline():
    print("=" * 70)
    print("GENERATING ACCELERATED PCR SUBMISSIONS (Phase 7)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. EVALUATION ON DEV SPLIT
    # -------------------------------------------------------------
    print("\n[1/3] Evaluating Models on DEV Split...")
    dev_q, dev_c = pcr.load_split("dev", data_dir=DATA_DIR)
    dev_q_ids = sorted(dev_q.keys())
    dev_c_ids = sorted(dev_c.keys())

    dev_bm25 = json.load(open(RUNS_DIR / "bm25" / "dev_scores.json"))
    dev_p1 = json.load(open(RUNS_DIR / "dense_phase1" / "dev_scores.json"))
    dev_bge = json.load(open(RUNS_DIR / "dense_bge" / "dev_scores.json"))
    dev_phrase = json.load(open(RUNS_DIR / "advanced_pcr" / "dev_phrase_scores.json"))
    dev_cit = extract_bidirectional_citations(dev_q, dev_c)

    norm_dev_phrase = normalize_scores(dev_phrase, dev_q_ids, dev_c_ids)
    norm_dev_cit = normalize_scores(dev_cit, dev_q_ids, dev_c_ids)
    norm_dev_bge = normalize_scores(dev_bge, dev_q_ids, dev_c_ids)
    norm_dev_p1 = normalize_scores(dev_p1, dev_q_ids, dev_c_ids)
    norm_dev_bm25 = normalize_scores(dev_bm25, dev_q_ids, dev_c_ids)

    # Model 1: Unigram BM25 Baseline (no self)
    preds_s01_dev = get_top_k_no_self(dev_bm25, dev_q_ids)
    res_s01 = pcr.evaluate_predictions(dev_q, preds_s01_dev)
    print(f"  S01 Unigram BM25:        F1@7 = {res_s01['primary_metric_1']:.4f} (TP={res_s01['tp']})")

    # Model 2: Phase 1 295M Dense (no self)
    preds_s02_dev = get_top_k_no_self(dev_p1, dev_q_ids)
    res_s02 = pcr.evaluate_predictions(dev_q, preds_s02_dev)
    print(f"  S02 Phase 1 295M Dense:  F1@7 = {res_s02['primary_metric_1']:.4f} (TP={res_s02['tp']})")

    # Model 3: BGE-M3 Dense (no self)
    preds_s03_dev = get_top_k_no_self(dev_bge, dev_q_ids)
    res_s03 = pcr.evaluate_predictions(dev_q, preds_s03_dev)
    print(f"  S03 BGE-M3 Dense:        F1@7 = {res_s03['primary_metric_1']:.4f} (TP={res_s03['tp']})")

    # Model 4: Hybrid Phase 1 + BM25 (no self)
    dev_hybrid_scores = {}
    for qid in dev_q_ids:
        dev_hybrid_scores[qid] = {
            cid: 0.5 * norm_dev_bm25[qid][cid] + 0.5 * norm_dev_p1[qid][cid]
            for cid in dev_c_ids if cid != qid
        }
    preds_s04_dev = get_top_k_no_self(dev_hybrid_scores, dev_q_ids)
    res_s04 = pcr.evaluate_predictions(dev_q, preds_s04_dev)
    print(f"  S04 Hybrid P1+BM25:      F1@7 = {res_s04['primary_metric_1']:.4f} (TP={res_s04['tp']})")

    # Model 5: Phrase-BM25 From Scratch (no self)
    preds_s05_dev = get_top_k_no_self(dev_phrase, dev_q_ids)
    res_s05 = pcr.evaluate_predictions(dev_q, preds_s05_dev)
    print(f"  S05 Phrase-BM25 Scratch: F1@7 = {res_s05['primary_metric_1']:.4f} (TP={res_s05['tp']})")

    # Model 6: Apex Citation-Grounded Consensus Ensemble
    # Weights: Phrase-BM25: 0.68, Citations: 0.20, BGE-M3: 0.10, Phase 1 LM: 0.02
    dev_apex_scores = {}
    for qid in dev_q_ids:
        dev_apex_scores[qid] = {
            cid: (
                0.68 * norm_dev_phrase[qid][cid] +
                0.20 * norm_dev_cit[qid][cid] +
                0.10 * norm_dev_bge[qid][cid] +
                0.02 * norm_dev_p1[qid][cid]
            )
            for cid in dev_c_ids if cid != qid
        }
    preds_s06_dev = get_top_k_no_self(dev_apex_scores, dev_q_ids)
    res_s06 = pcr.evaluate_predictions(dev_q, preds_s06_dev)
    print(f"  S06 Apex Citation-Ens:   F1@7 = {res_s06['primary_metric_1']:.4f} (Prec={res_s06['secondary_metric_1']:.4f}, Rec={res_s06['secondary_metric_2']:.4f}, TP={res_s06['tp']})")

    # Model 7: Reciprocal Rank Fusion (k=10)
    dev_rrf_scores = {}
    for qid in dev_q_ids:
        ph_rank = {c: r for r, c in enumerate(sorted([c for c in dev_c_ids if c != qid], key=lambda c: dev_phrase[qid].get(c, 0.0), reverse=True))}
        ct_rank = {c: r for r, c in enumerate(sorted([c for c in dev_c_ids if c != qid], key=lambda c: dev_cit[qid].get(c, 0.0), reverse=True))}
        bg_rank = {c: r for r, c in enumerate(sorted([c for c in dev_c_ids if c != qid], key=lambda c: dev_bge[qid].get(c, 0.0), reverse=True))}
        dev_rrf_scores[qid] = {
            cid: (2.5 / (10 + ph_rank[cid]) + 1.5 / (10 + ct_rank[cid]) + 0.5 / (10 + bg_rank[cid]))
            for cid in dev_c_ids if cid != qid
        }
    preds_s07_dev = get_top_k_no_self(dev_rrf_scores, dev_q_ids)
    res_s07 = pcr.evaluate_predictions(dev_q, preds_s07_dev)
    print(f"  S07 RRF Consensus Ens:   F1@7 = {res_s07['primary_metric_1']:.4f} (TP={res_s07['tp']})")

    # -------------------------------------------------------------
    # 2. COMPUTING TEST SCORES
    # -------------------------------------------------------------
    print("\n[2/3] Processing TEST Split Data...")
    test_q, test_c = pcr.load_split("test", data_dir=DATA_DIR)
    test_q_ids = sorted(test_q.keys())
    test_c_ids = sorted(test_c.keys())
    print(f"  Loaded {len(test_q_ids)} test queries and {len(test_c_ids)} test candidates.")

    test_bm25 = json.load(open(RUNS_DIR / "bm25" / "test_scores.json"))
    test_p1 = json.load(open(RUNS_DIR / "dense_phase1" / "test_scores.json"))
    test_bge = json.load(open(RUNS_DIR / "dense_bge" / "test_scores.json"))
    test_phrase = json.load(open(RUNS_DIR / "advanced_pcr" / "test_phrase_scores.json"))

    print("  Extracting bidirectional test citations...")
    test_cit = extract_bidirectional_citations(test_q, test_c)

    norm_test_phrase = normalize_scores(test_phrase, test_q_ids, test_c_ids)
    norm_test_cit = normalize_scores(test_cit, test_q_ids, test_c_ids)
    norm_test_bge = normalize_scores(test_bge, test_q_ids, test_c_ids)
    norm_test_p1 = normalize_scores(test_p1, test_q_ids, test_c_ids)
    norm_test_bm25 = normalize_scores(test_bm25, test_q_ids, test_c_ids)

    # Compute Test Predictions
    preds_s01_test = get_top_k_no_self(test_bm25, test_q_ids)
    preds_s02_test = get_top_k_no_self(test_p1, test_q_ids)
    preds_s03_test = get_top_k_no_self(test_bge, test_q_ids)

    test_hybrid_scores = {
        qid: {cid: 0.5 * norm_test_bm25[qid][cid] + 0.5 * norm_test_p1[qid][cid] for cid in test_c_ids if cid != qid}
        for qid in test_q_ids
    }
    preds_s04_test = get_top_k_no_self(test_hybrid_scores, test_q_ids)
    preds_s05_test = get_top_k_no_self(test_phrase, test_q_ids)

    test_apex_scores = {
        qid: {
            cid: (
                0.68 * norm_test_phrase[qid][cid] +
                0.20 * norm_test_cit[qid][cid] +
                0.10 * norm_test_bge[qid][cid] +
                0.02 * norm_test_p1[qid][cid]
            )
            for cid in test_c_ids if cid != qid
        }
        for qid in test_q_ids
    }
    preds_s06_test = get_top_k_no_self(test_apex_scores, test_q_ids)

    test_rrf_scores = {}
    for qid in test_q_ids:
        ph_rank = {c: r for r, c in enumerate(sorted([c for c in test_c_ids if c != qid], key=lambda c: test_phrase[qid].get(c, 0.0), reverse=True))}
        ct_rank = {c: r for r, c in enumerate(sorted([c for c in test_c_ids if c != qid], key=lambda c: test_cit[qid].get(c, 0.0), reverse=True))}
        bg_rank = {c: r for r, c in enumerate(sorted([c for c in test_c_ids if c != qid], key=lambda c: test_bge[qid].get(c, 0.0), reverse=True))}
        test_rrf_scores[qid] = {
            cid: (2.5 / (10 + ph_rank[cid]) + 1.5 / (10 + ct_rank[cid]) + 0.5 / (10 + bg_rank[cid]))
            for cid in test_c_ids if cid != qid
        }
    preds_s07_test = get_top_k_no_self(test_rrf_scores, test_q_ids)

    # -------------------------------------------------------------
    # 3. PACKAGING SUBMISSIONS
    # -------------------------------------------------------------
    print("\n[3/3] Packaging Submissions...")
    submissions = [
        ("P7_S01_bm25_unigram_baseline", preds_s01_test, f"Dev F1: {res_s01['primary_metric_1']:.4f}"),
        ("P7_S02_phase1_sc295m_dense", preds_s02_test, f"Dev F1: {res_s02['primary_metric_1']:.4f}"),
        ("P7_S03_bge_m3_dense", preds_s03_test, f"Dev F1: {res_s03['primary_metric_1']:.4f}"),
        ("P7_S04_hybrid_phase1_bm25", preds_s04_test, f"Dev F1: {res_s04['primary_metric_1']:.4f}"),
        ("P7_S05_phrase_bm25_from_scratch", preds_s05_test, f"Dev F1: {res_s05['primary_metric_1']:.4f}"),
        ("P7_S06_apex_citation_ensemble", preds_s06_test, f"Dev F1: {res_s06['primary_metric_1']:.4f} (PRIMARY ENTRY)"),
        ("P7_S07_rrf_consensus_ensemble", preds_s07_test, f"Dev F1: {res_s07['primary_metric_1']:.4f}"),
    ]

    expected_qids = set(test_q_ids)
    valid_cids = set(test_c_ids)

    for sub_name, preds, desc in submissions:
        sub_path = SUB_DIR / sub_name
        # Strict validation
        errs = pcr.validate_predictions(preds, candidate_ids=valid_cids, expected_qids=expected_qids)
        if errs:
            print(f"[!] Validation FAILED for {sub_name}:")
            for e in errs[:3]:
                print(f"    - {e}")
            sys.exit(1)

        # Check self-match leak
        self_leaks = sum(1 for qid in preds if qid in preds[qid])
        assert self_leaks == 0, f"Error: {sub_name} contains {self_leaks} self-matches!"

        zip_file = pcr.package_submission(preds, sub_path, sub_name)
        print(f"  [✓] {sub_name:32s} | {desc} | {zip_file.name} ({zip_file.stat().st_size} bytes)")

    print("\n" + "=" * 70)
    print("ALL 7 SUBMISSIONS READY, STRICTLY VALIDATED, AND PACKAGED!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
