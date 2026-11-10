"""Generate, strictly validate, and package all 6 official submissions for Phase 7 (PCR).

Submissions generated:
- P7_S01_bm25_unigram_baseline: Unigram BM25 baseline built from scratch (Dev F1: 0.0978)
- P7_S02_phase1_sc295m_dense: Dense retriever using Phase 1 295M pre-trained Legal LM (Dev F1: 0.1490)
- P7_S03_bge_m3_dense: Dense semantic retriever using BAAI/bge-m3 (Dev F1: 0.1779)
- P7_S04_hybrid_phase1_bm25: 100% from-scratch hybrid (BM25 + Phase 1 295M LM, Dev F1: 0.1868)
- P7_S05_phrase_bm25_from_scratch: High-performance 100% From-Scratch Phrase-BM25 (Dev F1: 0.3880)
- P7_S06_apex_consensus_ensemble: Tri-representation Apex Consensus Fusion (Dev F1: 0.3724, Test Leaderboard planned entry)
"""
import json
import os
import sys
import zipfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcr

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Phase_7_PCR"
RUNS_DIR = ROOT / "runs"
SUB_DIR = ROOT / "submissions"


def top_k_from_scores(score_dict: dict[str, dict[str, float]], k: int = 7) -> dict[str, list[str]]:
    """Extract top-k candidate IDs sorted by score descending."""
    preds = {}
    for qid, cdict in score_dict.items():
        sorted_cids = sorted(cdict.keys(), key=lambda c: cdict[c], reverse=True)
        preds[qid] = sorted_cids[:k]
    return preds


def main():
    print("=" * 60)
    print("Phase 7: Prior Case Retrieval - Advanced Submission Generation")
    print("=" * 60)

    # 1. Load test metadata
    test_q, test_c = pcr.load_split("test", data_dir=DATA_DIR)
    expected_qids = set(test_q.keys())
    valid_cids = set(test_c.keys())
    print(f"Loaded {len(expected_qids)} test queries, {len(valid_cids)} test candidates.")

    # 2. Load precomputed test scores
    print("\nLoading precomputed model scores...")
    bm25_scores = json.load(open(RUNS_DIR / "bm25" / "test_scores.json", "r"))
    p1_scores = json.load(open(RUNS_DIR / "dense_phase1" / "test_scores.json", "r"))
    bge_scores = json.load(open(RUNS_DIR / "dense_bge" / "test_scores.json", "r"))
    phrase_scores = json.load(open(RUNS_DIR / "advanced_pcr" / "test_phrase_scores.json", "r"))
    apex_scores = json.load(open(RUNS_DIR / "advanced_pcr" / "test_scores.json", "r"))

    # Normalization helper
    def normalize_dict(d):
        out = {}
        for qid, cdict in d.items():
            vals = np.array(list(cdict.values()), dtype=np.float32)
            mn, mx = vals.min(), vals.max()
            denom = (mx - mn) if (mx - mn) > 1e-9 else 1.0
            out[qid] = {cid: float((v - mn) / denom) for cid, v in cdict.items()}
        return out

    norm_bm25 = normalize_dict(bm25_scores)
    norm_p1 = normalize_dict(p1_scores)

    # 3. Define submissions
    configs = [
        ("P7_S01_bm25_unigram_baseline", top_k_from_scores(bm25_scores)),
        ("P7_S02_phase1_sc295m_dense", top_k_from_scores(p1_scores)),
        ("P7_S03_bge_m3_dense", top_k_from_scores(bge_scores)),
        (
            "P7_S04_hybrid_phase1_bm25",
            top_k_from_scores({
                qid: {cid: 0.5 * norm_bm25[qid][cid] + 0.5 * norm_p1[qid][cid] for cid in norm_bm25[qid]}
                for qid in expected_qids
            })
        ),
        ("P7_S05_phrase_bm25_from_scratch", top_k_from_scores(phrase_scores)),
        ("P7_S06_apex_consensus_ensemble", top_k_from_scores(apex_scores)),
    ]

    # 4. Generate, validate, and package each submission
    print("\nGenerating, validating, and packaging submissions...")
    for sub_name, preds in configs:
        target_dir = SUB_DIR / sub_name
        # Validation check
        errors = pcr.validate_predictions(preds, candidate_ids=valid_cids, expected_qids=expected_qids)
        if errors:
            print(f"Validation FAILED for {sub_name}:")
            for err in errors[:5]:
                print(f"  - {err}")
            sys.exit(1)

        # Package submission
        zip_path = pcr.package_submission(preds, target_dir, sub_name)

        # Verify zip archive contents
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            assert "prediction.json" in namelist, f"Missing prediction.json in {zip_path}"
            assert len(namelist) == 1, f"Archive has extra files: {namelist}"

        zip_size_kb = zip_path.stat().st_size / 1024
        print(f"  ✓ {sub_name:34s} -> {zip_path.name} ({zip_size_kb:.1f} KB) [VALIDATED]")

    print("\nAll 6 submissions packaged and verified successfully!")


if __name__ == "__main__":
    main()
