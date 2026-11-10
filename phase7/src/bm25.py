"""From-scratch BM25 Retrieval Engine for Prior Case Retrieval (PCR).
Implemented using standard library, collections, and NumPy.
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pcr  # noqa: E402


class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75, include_sections: bool = True):
        self.k1 = k1
        self.b = b
        self.include_sections = include_sections
        self.candidates = {}
        self.doc_lens = {}
        self.inv_index = defaultdict(list)
        self.idf = {}
        self.cand_ids = []
        self.avgdl = 0.0

    def fit(self, candidates: dict[str, dict]):
        """Build the BM25 inverted index from the candidate documents."""
        self.candidates = candidates
        self.cand_ids = sorted(list(candidates.keys()))
        N = len(candidates)
        df = Counter()
        cand_tfs = {}

        for cid in self.cand_ids:
            toks = pcr.tokenize(candidates[cid].get("text", []), include_sections=self.include_sections)
            self.doc_lens[cid] = len(toks)
            tf = Counter(toks)
            cand_tfs[cid] = tf
            for t in tf:
                df[t] += 1

        self.avgdl = sum(self.doc_lens.values()) / max(1, N)
        self.idf = {t: math.log(1 + (N - cnt + 0.5) / (cnt + 0.5)) for t, cnt in df.items()}

        self.inv_index = defaultdict(list)
        for cid, tf in cand_tfs.items():
            l_ratio = (1.0 - self.b + self.b * (self.doc_lens[cid] / self.avgdl))
            for t, count in tf.items():
                denom = count + self.k1 * l_ratio
                weight = (count * (self.k1 + 1.0)) / denom
                self.inv_index[t].append((cid, weight))

    def score_queries(self, queries: dict[str, dict]) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
        """Score all queries and return (top_7_predictions, full_scores_dict)."""
        predictions = {}
        all_scores = {}

        for qid, qdata in queries.items():
            q_toks = Counter(pcr.tokenize(qdata.get("text", []), include_sections=self.include_sections))
            scores = defaultdict(float)

            for t, q_cnt in q_toks.items():
                if t in self.idf:
                    w = self.idf[t]
                    for cid, term_weight in self.inv_index[t]:
                        scores[cid] += w * term_weight

            all_scores[qid] = {cid: scores[cid] for cid in self.cand_ids}
            top7 = sorted(self.cand_ids, key=lambda c: scores[c], reverse=True)[:pcr.K]
            predictions[qid] = top7

        return predictions, all_scores


def main():
    parser = argparse.ArgumentParser(description="BM25 PCR Retriever")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Split to run on")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument("--out-dir", type=str, default=str(HERE.parent / "runs" / "bm25"), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading {args.split} split data...")
    queries, candidates = pcr.load_split(args.split)
    print(f"[*] Loaded {len(queries)} queries and {len(candidates)} candidates.")

    print(f"[*] Indexing candidates with BM25 (k1={args.k1}, b={args.b})...")
    t0 = time.time()
    bm25 = BM25Retriever(k1=args.k1, b=args.b)
    bm25.fit(candidates)
    print(f"[✓] Index built in {time.time() - t0:.2f}s.")

    print(f"[*] Scoring {len(queries)} queries...")
    t1 = time.time()
    preds, full_scores = bm25.score_queries(queries)
    print(f"[✓] Queries scored in {time.time() - t1:.2f}s.")

    # Save full scores and predictions
    scores_path = out_dir / f"{args.split}_scores.json"
    preds_path = out_dir / f"{args.split}_predictions.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(full_scores, f)
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    print(f"[✓] Saved scores to {scores_path} and predictions to {preds_path}.")

    if args.split == "dev":
        res = pcr.evaluate_predictions(queries, preds)
        print("\n==========================================")
        print(f"BM25 Retrieval Dev Results:")
        print(f"  Micro-F1@7:        {res['primary_metric_1']:.6f}")
        print(f"  Micro-Precision@7: {res['secondary_metric_1']:.6f}")
        print(f"  Micro-Recall@7:    {res['secondary_metric_2']:.6f}")
        print(f"  TP={res['tp']} FP={res['fp']} FN={res['fn']}")
        print("==========================================\n")


if __name__ == "__main__":
    main()
