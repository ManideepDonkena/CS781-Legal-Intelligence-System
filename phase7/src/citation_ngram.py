"""High-precision legal precedent retrieval using Citation Matching, Multi-scale N-gram Fingerprinting, and Inverted Index.
"""
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcr

PAT_CITATION = re.compile(
    r'\b(?:\d{4}\s+)?(?:AIR|SCR|SCC|ITR|ILR|CRLJ|FLR|LLJ|COMP\s*CAS|IND\.\s*CAS\.)\s*(?:\(\d+\))?\s*\d+\b',
    re.IGNORECASE
)
PAT_CLEAN = re.compile(r'[a-z0-9]+')


def clean_tokens(text: str) -> list[str]:
    return PAT_CLEAN.findall(text.lower())


def extract_citations(text: str) -> set[str]:
    cits = set()
    for m in PAT_CITATION.finditer(text):
        cits.add(re.sub(r'\s+', ' ', m.group(0).lower()))
    return cits


def build_ngram_inverted_index(candidates: dict, ns=(5, 7, 9)):
    """Build an inverted index from n-grams to candidate IDs with document frequency (DF)."""
    t0 = time.time()
    index = defaultdict(list)
    df_map = defaultdict(int)

    print(f"[*] Indexing {len(candidates)} candidates across n-grams {ns}...")
    for cid, cdata in candidates.items():
        text = " ".join(cdata["text"])
        tokens = clean_tokens(text)
        seen_cand = set()
        for n in ns:
            for i in range(len(tokens) - n + 1):
                ng = " ".join(tokens[i : i + n])
                if ng not in seen_cand:
                    seen_cand.add(ng)
                    index[ng].append(cid)
                    df_map[ng] += 1

    print(f"[✓] Built index with {len(index)} distinct n-grams in {time.time() - t0:.1f}s")
    return index, df_map


def extract_candidate_titles(candidates: dict) -> dict[str, set[str]]:
    """Extract petitioner/respondent names from candidate case headers."""
    titles = {}
    pat_party = re.compile(r'(?:PETITIONER|RESPONDENT):\s*([^V\n]+?)(?:\s+Vs\.|\s+DATE|\s+BENCH|\n|$)', re.IGNORECASE)
    for cid, cdata in candidates.items():
        header = " ".join(cdata["text"][:3])
        parties = set()
        for p in pat_party.findall(header):
            cleaned = p.replace("<NAME>", "").replace("<ORG>", "").strip().lower()
            words = [w for w in PAT_CLEAN.findall(cleaned) if len(w) > 3]
            if len(words) >= 2:
                parties.add(" ".join(words))
        titles[cid] = parties
    return titles


def score_split(queries: dict, candidates: dict, ns=(5, 7, 9), max_df=25):
    index, df_map = build_ngram_inverted_index(candidates, ns=ns)
    cand_citations = {cid: extract_citations(" ".join(candidates[cid]["text"][:5])) for cid in candidates}
    cand_titles = extract_candidate_titles(candidates)
    N_cand = len(candidates)

    scores = {qid: defaultdict(float) for qid in queries}

    print(f"[*] Scoring {len(queries)} queries against candidates...")
    t0 = time.time()
    for qid, qdata in queries.items():
        q_text = " ".join(qdata["text"])
        q_tokens = clean_tokens(q_text)
        q_cits = extract_citations(q_text)

        # 1. Exact Citation Matches (Massive Boost)
        for cid, ccits in cand_citations.items():
            overlap = q_cits & ccits
            if overlap:
                scores[qid][cid] += len(overlap) * 50.0

        # 2. Case Title / Party Matches
        q_lower = q_text.lower()
        for cid, parties in cand_titles.items():
            for party in parties:
                if party in q_lower:
                    scores[qid][cid] += 20.0
                    break

        # 3. Fast Inverted Index Multi-Scale N-grams with IDF
        seen_query = set()
        for n in ns:
            weight = (n ** 2.0)
            for i in range(len(q_tokens) - n + 1):
                ng = " ".join(q_tokens[i : i + n])
                if ng in seen_query:
                    continue
                seen_query.add(ng)

                if ng in index:
                    df = df_map[ng]
                    if df <= max_df:
                        idf = math.log((N_cand - df + 0.5) / (df + 0.5) + 1.0)
                        val = weight * idf
                        for cid in index[ng]:
                            scores[qid][cid] += val

    print(f"[✓] Scored in {time.time() - t0:.1f}s")
    # Fill zeros for any un-scored candidates
    full_scores = {qid: {cid: float(scores[qid][cid]) for cid in candidates} for qid in queries}
    return full_scores


def main():
    dev_q, dev_c = pcr.load_split("dev")
    scores = score_split(dev_q, dev_c)

    res = pcr.evaluate_predictions(
        dev_q,
        {qid: sorted(scores[qid].keys(), key=lambda c: scores[qid][c], reverse=True)[:7] for qid in dev_q}
    )
    print("\n" + "=" * 50)
    print("DEV CITATION + NGRAM INVERTED INDEX SCORES:")
    print(f"Micro-F1@7:        {res['primary_metric_1']:.4f}")
    print(f"Micro-Precision@7: {res['secondary_metric_1']:.4f}")
    print(f"Micro-Recall@7:    {res['secondary_metric_2']:.4f}")
    print(f"True Positives:    {res['tp']} (FP: {res['fp']}, FN: {res['fn']})")
    print("=" * 50)


if __name__ == "__main__":
    main()
