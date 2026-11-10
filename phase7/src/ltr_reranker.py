"""
Learning-to-Rank (LTR) Re-ranker for Phase 7 PCR.

Approach:
1. Compute multiple retrieval signals (features) for each (query, candidate) pair
2. Train a scikit-learn classifier on training data to predict relevance
3. Use the trained classifier to re-rank dev/test candidates

Features:
- Phrase-BM25 score (raw + normalized)
- Verbatim N-gram score (raw + normalized) 
- Citation overlap count
- Bench name overlap
- Statute/Article overlap
- TF-IDF cosine similarity
- Document length ratio
- BM25 rank position
- Verbatim rank position
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcr
from advanced_pcr import extract_phrases, extract_citations, clean_tokens, STOPWORDS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Phase_7_PCR"
RUNS_DIR = ROOT / "runs"


# ======================================================================
# FEATURE EXTRACTORS
# ======================================================================

PAT_CITATION = re.compile(
    r'\b(?:\d{4}\s+)?(?:AIR|SCR|SCC|ITR|ILR|CRLJ|FLR|LLJ|COMP\s*CAS|IND\.\s*CAS\.)\s*(?:\(\d+\))?\s*\d+\b',
    re.IGNORECASE
)
PAT_BENCH = re.compile(r'BENCH:\s*([^\n]+)', re.IGNORECASE)
PAT_SEC = re.compile(r'\b(?:section|sec|s\.)\s*(\d+[a-z]?)\b', re.IGNORECASE)
PAT_ART = re.compile(r'\b(?:article|art\.)\s*(\d+[a-z]?)\b', re.IGNORECASE)
PAT_YEAR = re.compile(r'\b(1[89]\d{2}|20[0-2]\d)\b')
PAT_WORD = re.compile(r'\b[a-z0-9_]{2,}\b')


def extract_features_bulk(queries, candidates, q_ids, c_ids, target_pairs):
    """
    Extract features for a set of (query, candidate) pairs.
    target_pairs: list of (qid, cid) tuples
    Returns: numpy array of shape (len(target_pairs), num_features)
    """
    N_cand = len(c_ids)
    
    # ---- Pre-compute candidate-level features ----
    print("  Pre-computing candidate features...")
    t0 = time.time()
    
    # Phrase-BM25 index
    cand_phrase_tf = {}
    cand_phrase_lens = {}
    phrase_df = Counter()
    
    for cid in c_ids:
        text = " ".join(candidates[cid]["text"]).lower()
        words = PAT_WORD.findall(text)
        filtered = [w for w in words if w not in STOPWORDS]
        feats = list(filtered)
        for i in range(len(filtered) - 1):
            feats.append(filtered[i] + "__" + filtered[i+1])
        for i in range(len(filtered) - 2):
            feats.append(filtered[i] + "__" + filtered[i+1] + "__" + filtered[i+2])
        cand_phrase_lens[cid] = len(feats)
        cand_phrase_tf[cid] = Counter(feats)
        for term in set(feats):
            phrase_df[term] += 1
    
    avg_phrase_len = sum(cand_phrase_lens.values()) / N_cand
    phrase_idf = {
        term: math.log((N_cand - df + 0.5) / (df + 0.5) + 1.0)
        for term, df in phrase_df.items() if 1 <= df <= 250
    }
    
    # BM25 inverted index
    bm25_index = defaultdict(list)
    k1, b = 1.5, 0.75
    for cid, tf_map in cand_phrase_tf.items():
        norm_factor = k1 * (1.0 - b + b * (cand_phrase_lens[cid] / avg_phrase_len))
        for term, tf in tf_map.items():
            if term in phrase_idf:
                w = (tf * (k1 + 1.0)) / (tf + norm_factor) * phrase_idf[term]
                if "__" in term:
                    w *= (2.5 if term.count("__") == 1 else 5.0)
                bm25_index[term].append((cid, w))
    
    # N-gram index (verbatim)
    ngram_index = defaultdict(set)
    ngram_df = Counter()
    ns = (5, 6, 7, 8)
    for cid in c_ids:
        tokens = clean_tokens(" ".join(candidates[cid]["text"]))
        seen = set()
        for n in ns:
            for i in range(len(tokens) - n + 1):
                ng = " ".join(tokens[i:i+n])
                if ng not in seen:
                    seen.add(ng)
                    ngram_index[ng].add(cid)
                    ngram_df[ng] += 1
    
    # Candidate citations, bench names, sections, articles, year
    cand_cits = {}
    cand_benches = {}
    cand_sections = {}
    cand_years = {}
    cand_word_lens = {}
    
    for cid in c_ids:
        full_text = " ".join(candidates[cid]["text"])
        head = " ".join(candidates[cid]["text"][:6])
        
        cand_cits[cid] = set(
            re.sub(r'\s+', ' ', m.group(0).lower()) for m in PAT_CITATION.finditer(head)
        )
        
        bench_names = set()
        for b_str in PAT_BENCH.findall(" ".join(candidates[cid]["text"][:5])):
            for w in b_str.replace("<NAME>", "").replace(",", " ").strip().lower().split():
                if len(w) > 3 and w not in STOPWORDS:
                    bench_names.add(w)
        cand_benches[cid] = bench_names
        
        secs = set(s.lower() for s in PAT_SEC.findall(full_text))
        arts = set(a.lower() for a in PAT_ART.findall(full_text))
        cand_sections[cid] = secs | arts
        
        years = PAT_YEAR.findall(head[:500])
        cand_years[cid] = int(years[0]) if years else 0
        
        cand_word_lens[cid] = len(full_text.split())
    
    # TF-IDF sparse index for cosine similarity
    # Build vocabulary from candidates
    vocab = {}
    cand_tfidf = {}
    for cid in c_ids:
        tokens = PAT_WORD.findall(" ".join(candidates[cid]["text"]).lower())
        filtered = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
        tf = Counter(filtered)
        cand_tfidf[cid] = tf
        for t in tf:
            if t not in vocab:
                vocab[t] = 0
            vocab[t] += 1
    
    # Keep only terms with reasonable df
    tfidf_idf = {
        t: math.log(N_cand / (df + 1)) for t, df in vocab.items()
        if 2 <= df <= N_cand * 0.8
    }
    
    print(f"    Features pre-computed in {time.time()-t0:.1f}s")
    
    # ---- Compute per-query scores and features ----
    print("  Computing pair-level features...")
    
    # Pre-compute query-level data
    query_phrase_feats = {}
    query_cits = {}
    query_benches_text = {}
    query_sections = {}
    query_years = {}
    query_word_lens = {}
    query_tfidf = {}
    query_ngram_tokens = {}
    
    for qid in q_ids:
        qdata = queries[qid]
        full_text = " ".join(qdata["text"])
        head = " ".join(qdata["text"][:6])
        
        text_lower = full_text.lower()
        words = PAT_WORD.findall(text_lower)
        filtered = [w for w in words if w not in STOPWORDS]
        feats = list(filtered)
        for i in range(len(filtered) - 1):
            feats.append(filtered[i] + "__" + filtered[i+1])
        for i in range(len(filtered) - 2):
            feats.append(filtered[i] + "__" + filtered[i+1] + "__" + filtered[i+2])
        query_phrase_feats[qid] = Counter(feats)
        
        query_cits[qid] = set(
            re.sub(r'\s+', ' ', m.group(0).lower()) for m in PAT_CITATION.finditer(full_text)
        )
        query_benches_text[qid] = full_text.lower()
        
        secs = set(s.lower() for s in PAT_SEC.findall(full_text))
        arts = set(a.lower() for a in PAT_ART.findall(full_text))
        query_sections[qid] = secs | arts
        
        years = PAT_YEAR.findall(head[:500])
        query_years[qid] = int(years[0]) if years else 0
        
        query_word_lens[qid] = len(full_text.split())
        
        tokens = PAT_WORD.findall(text_lower)
        filtered2 = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
        query_tfidf[qid] = Counter(filtered2)
        
        query_ngram_tokens[qid] = clean_tokens(full_text)
    
    # Pre-compute BM25 scores for all queries
    query_bm25_scores = {}
    for qid in q_ids:
        scores = defaultdict(float)
        for term in query_phrase_feats[qid]:
            if term in bm25_index:
                for cid, w in bm25_index[term]:
                    scores[cid] += w
        query_bm25_scores[qid] = scores
    
    # Pre-compute verbatim ngram scores
    query_verb_scores = {}
    for qid in q_ids:
        scores = defaultdict(float)
        q_tokens = query_ngram_tokens[qid]
        seen = set()
        for n in ns:
            scale = n ** 2.0
            for i in range(len(q_tokens) - n + 1):
                ng = " ".join(q_tokens[i:i+n])
                if ng in seen:
                    continue
                seen.add(ng)
                if ng in ngram_index:
                    df = ngram_df[ng]
                    if df <= 30:
                        idf_val = math.log((N_cand - df + 0.5) / (df + 0.5) + 1.0)
                        for cid in ngram_index[ng]:
                            scores[cid] += scale * idf_val
        query_verb_scores[qid] = scores
    
    # Pre-compute BM25 ranks
    query_bm25_ranks = {}
    for qid in q_ids:
        ranked = sorted(query_bm25_scores[qid].items(), key=lambda x: x[1], reverse=True)
        rank_map = {cid: rank for rank, (cid, _) in enumerate(ranked)}
        query_bm25_ranks[qid] = rank_map
    
    # Pre-compute verbatim ranks
    query_verb_ranks = {}
    for qid in q_ids:
        ranked = sorted(query_verb_scores[qid].items(), key=lambda x: x[1], reverse=True)
        rank_map = {cid: rank for rank, (cid, _) in enumerate(ranked)}
        query_verb_ranks[qid] = rank_map
    
    # Pre-compute TF-IDF cosine similarity (sparse)
    def tfidf_cosine(qtf, ctf):
        common = set(qtf.keys()) & set(ctf.keys()) & set(tfidf_idf.keys())
        if not common:
            return 0.0
        dot = sum(qtf[t] * tfidf_idf[t] * ctf[t] * tfidf_idf[t] for t in common)
        q_norm = math.sqrt(sum((qtf[t] * tfidf_idf[t]) ** 2 for t in qtf if t in tfidf_idf))
        c_norm = math.sqrt(sum((ctf[t] * tfidf_idf[t]) ** 2 for t in ctf if t in tfidf_idf))
        if q_norm < 1e-9 or c_norm < 1e-9:
            return 0.0
        return dot / (q_norm * c_norm)
    
    # ---- Extract features for target pairs ----
    features = []
    for pi, (qid, cid) in enumerate(target_pairs):
        if pi % 50000 == 0 and pi > 0:
            print(f"    Pair {pi}/{len(target_pairs)}...")
        
        feat = []
        
        # 1. BM25 score (raw)
        bm25_score = query_bm25_scores[qid].get(cid, 0.0)
        feat.append(bm25_score)
        
        # 2. BM25 rank (reciprocal)
        bm25_rank = query_bm25_ranks[qid].get(cid, N_cand)
        feat.append(1.0 / (bm25_rank + 1))
        
        # 3. Verbatim ngram score (raw)
        verb_score = query_verb_scores[qid].get(cid, 0.0)
        feat.append(verb_score)
        
        # 4. Verbatim rank (reciprocal)
        verb_rank = query_verb_ranks[qid].get(cid, N_cand)
        feat.append(1.0 / (verb_rank + 1))
        
        # 5. Citation overlap count
        cit_overlap = len(query_cits[qid] & cand_cits[cid])
        feat.append(cit_overlap)
        
        # 6. Bench name overlap (binary + count)
        bench_overlap = sum(1 for bn in cand_benches[cid] if bn in query_benches_text[qid])
        feat.append(bench_overlap)
        
        # 7. Section/Article overlap
        sec_overlap = len(query_sections[qid] & cand_sections[cid])
        feat.append(sec_overlap)
        
        # 8. Year proximity
        if query_years[qid] > 0 and cand_years[cid] > 0:
            year_diff = abs(query_years[qid] - cand_years[cid])
            feat.append(1.0 / (year_diff + 1))
        else:
            feat.append(0.0)
        
        # 9. Document length ratio
        q_len = query_word_lens[qid]
        c_len = cand_word_lens[cid]
        feat.append(min(q_len, c_len) / (max(q_len, c_len) + 1))
        
        # 10. TF-IDF cosine similarity
        cos_sim = tfidf_cosine(query_tfidf[qid], cand_tfidf[cid])
        feat.append(cos_sim)
        
        # 11. Log BM25 score
        feat.append(math.log1p(bm25_score))
        
        # 12. Log verbatim score
        feat.append(math.log1p(verb_score))
        
        # 13. Combined first-stage score
        feat.append(bm25_score + verb_score * 0.5)
        
        # 14. Has any citation overlap (binary)
        feat.append(1.0 if cit_overlap > 0 else 0.0)
        
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def main():
    print("=" * 70)
    print("LEARNING-TO-RANK (LTR) FOR PHASE 7 PCR")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_queries, train_candidates = pcr.load_split("train", data_dir=DATA_DIR)
    dev_queries, dev_candidates = pcr.load_split("dev", data_dir=DATA_DIR)
    test_queries, test_candidates = pcr.load_split("test", data_dir=DATA_DIR)
    
    train_q_ids = sorted(train_queries.keys())
    train_c_ids = sorted(train_candidates.keys())
    dev_q_ids = sorted(dev_queries.keys())
    dev_c_ids = sorted(dev_candidates.keys())
    test_q_ids = sorted(test_queries.keys())
    test_c_ids = sorted(test_candidates.keys())
    
    # ----------------------------------------------------------------
    # STEP 2: Create training pairs (positive + hard negative sampling)
    # ----------------------------------------------------------------
    print("\n[2/5] Creating training pairs...")
    
    # For training: use all gold pairs as positives
    # For negatives: top-ranked non-gold from BM25 (hard negatives)
    # First, compute BM25 scores on training data
    
    # Build BM25 for train candidates
    print("  Building BM25 for train candidates...")
    try:
        from advanced_pcr import PhraseBM25
    except ImportError:
        PhraseBM25 = None
    
    # We need to compute BM25 scores for training - let's do it inline
    train_pairs = []
    train_labels = []
    
    # Pre-compute features for ALL (train_query, train_candidate) pairs
    # This is too many (827 * 4320 = 3.57M pairs). Instead, sample smartly.
    
    # For each query: include all gold candidates + top-50 BM25 + 10 random
    HARD_NEG_K = 50
    RANDOM_NEG_K = 10
    
    # First we need BM25 scores
    target_pairs_train = []
    for qid in train_q_ids:
        gold = set(train_queries[qid].get("relevant_candidates", []))
        # Add all gold pairs
        for gcid in gold:
            if gcid in train_candidates:
                target_pairs_train.append((qid, gcid))
        # We'll add negatives after computing BM25 ranks
    
    print(f"  Gold pairs: {len(target_pairs_train)}")
    
    # Compute features for all candidates per query to get rankings
    # But that's 827 * 4320 = too many. Let's do top-K BM25 + random approach.
    
    # Build a quick BM25 index for train candidates
    print("  Quick BM25 scoring for negative sampling...")
    t0 = time.time()
    
    # Simplified BM25
    cand_tf_simple = {}
    cand_len_simple = {}
    df_simple = Counter()
    
    for cid in train_c_ids:
        words = PAT_WORD.findall(" ".join(train_candidates[cid]["text"]).lower())
        filtered = [w for w in words if w not in STOPWORDS]
        feats = list(filtered)
        for i in range(len(filtered) - 1):
            feats.append(filtered[i] + "__" + filtered[i+1])
        cand_tf_simple[cid] = Counter(feats)
        cand_len_simple[cid] = len(feats)
        for t in set(feats):
            df_simple[t] += 1
    
    avg_len = sum(cand_len_simple.values()) / len(train_c_ids)
    idf_simple = {t: math.log((len(train_c_ids) - df + 0.5) / (df + 0.5) + 1.0)
                  for t, df in df_simple.items() if df <= 250}
    
    bm25_idx = defaultdict(list)
    for cid, tf_map in cand_tf_simple.items():
        nf = 1.5 * (0.25 + 0.75 * (cand_len_simple[cid] / avg_len))
        for term, tf in tf_map.items():
            if term in idf_simple:
                w = (tf * 2.5) / (tf + nf) * idf_simple[term]
                if "__" in term:
                    w *= 2.5
                bm25_idx[term].append((cid, w))
    
    import random
    random.seed(42)
    
    # For each train query, get negatives
    neg_pairs_train = []
    for qi, qid in enumerate(train_q_ids):
        gold = set(train_queries[qid].get("relevant_candidates", []))
        if not gold:
            continue
        
        # BM25 score
        words = PAT_WORD.findall(" ".join(train_queries[qid]["text"]).lower())
        filtered = [w for w in words if w not in STOPWORDS]
        feats = list(filtered)
        for i in range(len(filtered) - 1):
            feats.append(filtered[i] + "__" + filtered[i+1])
        q_feats = Counter(feats)
        
        scores = defaultdict(float)
        for term in q_feats:
            if term in bm25_idx:
                for cid, w in bm25_idx[term]:
                    scores[cid] += w
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Hard negatives: top BM25 that aren't gold
        hard_negs = [cid for cid, _ in ranked if cid not in gold][:HARD_NEG_K]
        # Random negatives
        non_gold = [cid for cid in train_c_ids if cid not in gold and cid not in set(hard_negs)]
        rand_negs = random.sample(non_gold, min(RANDOM_NEG_K, len(non_gold)))
        
        for ncid in hard_negs + rand_negs:
            neg_pairs_train.append((qid, ncid))
    
    target_pairs_train.extend(neg_pairs_train)
    train_labels_arr = np.array(
        [1.0] * (len(target_pairs_train) - len(neg_pairs_train)) + 
        [0.0] * len(neg_pairs_train),
        dtype=np.float32
    )
    
    print(f"  Total training pairs: {len(target_pairs_train)} "
          f"({sum(train_labels_arr)} pos, {len(train_labels_arr) - sum(train_labels_arr)} neg)")
    print(f"  BM25 setup took {time.time()-t0:.1f}s")
    
    # ----------------------------------------------------------------
    # STEP 3: Extract features for training pairs
    # ----------------------------------------------------------------
    print("\n[3/5] Extracting features for training pairs...")
    X_train = extract_features_bulk(
        train_queries, train_candidates, train_q_ids, train_c_ids, target_pairs_train
    )
    y_train = train_labels_arr
    
    print(f"  Feature matrix: {X_train.shape}")
    
    # ----------------------------------------------------------------
    # STEP 4: Train classifiers
    # ----------------------------------------------------------------
    print("\n[4/5] Training classifiers...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Try multiple classifiers
    classifiers = {
        "GBM": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=10, random_state=42
        ),
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "LR": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
    }
    
    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        t0 = time.time()
        clf.fit(X_train_scaled, y_train)
        print(f"    ✓ Trained in {time.time()-t0:.1f}s")
        
        # Feature importance
        if hasattr(clf, 'feature_importances_'):
            feat_names = [
                "bm25_score", "bm25_recip_rank", "verb_score", "verb_recip_rank",
                "cit_overlap", "bench_overlap", "sec_overlap", "year_prox",
                "len_ratio", "tfidf_cosine", "log_bm25", "log_verb",
                "combined_first", "has_citation"
            ]
            imp = clf.feature_importances_
            sorted_feats = sorted(zip(feat_names, imp), key=lambda x: x[1], reverse=True)
            print(f"    Top features: {', '.join(f'{n}:{v:.3f}' for n, v in sorted_feats[:5])}")
    
    # ----------------------------------------------------------------
    # STEP 5: Evaluate on DEV and generate test submissions
    # ----------------------------------------------------------------
    print("\n[5/5] Evaluating on DEV and generating submissions...")
    
    best_dev_f1 = 0
    best_clf_name = None
    best_dev_preds = None
    
    for clf_name, clf in classifiers.items():
        print(f"\n  Evaluating {clf_name} on Dev...")
        
        # For dev: score top-100 BM25 candidates per query
        TOP_K = 100
        
        dev_preds = {}
        
        # Build features for dev pairs
        # First, identify which pairs to score
        dev_target_pairs = []
        dev_pair_query_map = defaultdict(list)  # qid -> list of indices
        
        # Quick BM25 for dev
        dev_bm25_scores = {}
        
        # Build dev BM25 index
        dev_cand_tf = {}
        dev_cand_len = {}
        dev_df = Counter()
        for cid in dev_c_ids:
            words = PAT_WORD.findall(" ".join(dev_candidates[cid]["text"]).lower())
            filtered = [w for w in words if w not in STOPWORDS]
            feats = list(filtered)
            for i in range(len(filtered) - 1):
                feats.append(filtered[i] + "__" + filtered[i+1])
            dev_cand_tf[cid] = Counter(feats)
            dev_cand_len[cid] = len(feats)
            for t in set(feats):
                dev_df[t] += 1
        
        dev_avg_len = sum(dev_cand_len.values()) / len(dev_c_ids)
        dev_idf = {t: math.log((len(dev_c_ids) - df + 0.5) / (df + 0.5) + 1.0)
                    for t, df in dev_df.items() if df <= 250}
        
        dev_bm25_idx = defaultdict(list)
        for cid, tf_map in dev_cand_tf.items():
            nf = 1.5 * (0.25 + 0.75 * (dev_cand_len[cid] / dev_avg_len))
            for term, tf in tf_map.items():
                if term in dev_idf:
                    w = (tf * 2.5) / (tf + nf) * dev_idf[term]
                    if "__" in term:
                        w *= 2.5
                    dev_bm25_idx[term].append((cid, w))
        
        for qid in dev_q_ids:
            words = PAT_WORD.findall(" ".join(dev_queries[qid]["text"]).lower())
            filtered = [w for w in words if w not in STOPWORDS]
            feats = list(filtered)
            for i in range(len(filtered) - 1):
                feats.append(filtered[i] + "__" + filtered[i+1])
            q_feats = Counter(feats)
            
            scores = defaultdict(float)
            for term in q_feats:
                if term in dev_bm25_idx:
                    for cid, w in dev_bm25_idx[term]:
                        scores[cid] += w
            
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_cands = [cid for cid, _ in ranked[:TOP_K]]
            # Also include all candidates with non-zero score
            # Ensure we have at least 7 candidates
            if len(top_cands) < 7:
                top_cands = dev_c_ids[:7]
            
            for cid in top_cands:
                idx = len(dev_target_pairs)
                dev_target_pairs.append((qid, cid))
                dev_pair_query_map[qid].append(idx)
        
        # Extract features
        X_dev = extract_features_bulk(
            dev_queries, dev_candidates, dev_q_ids, dev_c_ids, dev_target_pairs
        )
        X_dev_scaled = scaler.transform(X_dev)
        
        # Predict
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_dev_scaled)[:, 1]
        else:
            probs = clf.decision_function(X_dev_scaled)
        
        # Rank per query
        for qid in dev_q_ids:
            indices = dev_pair_query_map[qid]
            cid_scores = [(dev_target_pairs[i][1], probs[i]) for i in indices]
            cid_scores.sort(key=lambda x: x[1], reverse=True)
            dev_preds[qid] = [cid for cid, _ in cid_scores[:7]]
        
        # Evaluate
        res = pcr.evaluate_predictions(dev_queries, dev_preds)
        print(f"    {clf_name} Dev F1@7: {res['primary_metric_1']:.4f} "
              f"(P={res['secondary_metric_1']:.4f}, R={res['secondary_metric_2']:.4f})")
        print(f"    TP={res['tp']}, FP={res['fp']}, FN={res['fn']}")
        
        if res['primary_metric_1'] > best_dev_f1:
            best_dev_f1 = res['primary_metric_1']
            best_clf_name = clf_name
            best_dev_preds = dev_preds
    
    print(f"\n  ✓ Best classifier: {best_clf_name} with Dev F1@7: {best_dev_f1:.4f}")
    best_clf = classifiers[best_clf_name]
    
    # ----------------------------------------------------------------
    # Generate TEST predictions with best classifier
    # ----------------------------------------------------------------
    print("\n  Generating TEST predictions...")
    
    test_target_pairs = []
    test_pair_query_map = defaultdict(list)
    
    # Quick BM25 for test
    test_cand_tf = {}
    test_cand_len = {}
    test_df = Counter()
    for cid in test_c_ids:
        words = PAT_WORD.findall(" ".join(test_candidates[cid]["text"]).lower())
        filtered = [w for w in words if w not in STOPWORDS]
        feats = list(filtered)
        for i in range(len(filtered) - 1):
            feats.append(filtered[i] + "__" + filtered[i+1])
        test_cand_tf[cid] = Counter(feats)
        test_cand_len[cid] = len(feats)
        for t in set(feats):
            test_df[t] += 1
    
    test_avg_len = sum(test_cand_len.values()) / len(test_c_ids)
    test_idf_dict = {t: math.log((len(test_c_ids) - df + 0.5) / (df + 0.5) + 1.0)
                for t, df in test_df.items() if df <= 250}
    
    test_bm25_idx = defaultdict(list)
    for cid, tf_map in test_cand_tf.items():
        nf = 1.5 * (0.25 + 0.75 * (test_cand_len[cid] / test_avg_len))
        for term, tf in tf_map.items():
            if term in test_idf_dict:
                w = (tf * 2.5) / (tf + nf) * test_idf_dict[term]
                if "__" in term:
                    w *= 2.5
                test_bm25_idx[term].append((cid, w))
    
    TOP_K_TEST = 100
    for qid in test_q_ids:
        words = PAT_WORD.findall(" ".join(test_queries[qid]["text"]).lower())
        filtered = [w for w in words if w not in STOPWORDS]
        feats = list(filtered)
        for i in range(len(filtered) - 1):
            feats.append(filtered[i] + "__" + filtered[i+1])
        q_feats = Counter(feats)
        
        scores = defaultdict(float)
        for term in q_feats:
            if term in test_bm25_idx:
                for cid, w in test_bm25_idx[term]:
                    scores[cid] += w
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_cands = [cid for cid, _ in ranked[:TOP_K_TEST]]
        if len(top_cands) < 7:
            top_cands = test_c_ids[:7]
        
        for cid in top_cands:
            idx = len(test_target_pairs)
            test_target_pairs.append((qid, cid))
            test_pair_query_map[qid].append(idx)
    
    X_test = extract_features_bulk(
        test_queries, test_candidates, test_q_ids, test_c_ids, test_target_pairs
    )
    X_test_scaled = scaler.transform(X_test)
    
    if hasattr(best_clf, 'predict_proba'):
        test_probs = best_clf.predict_proba(X_test_scaled)[:, 1]
    else:
        test_probs = best_clf.decision_function(X_test_scaled)
    
    test_preds = {}
    for qid in test_q_ids:
        indices = test_pair_query_map[qid]
        cid_scores = [(test_target_pairs[i][1], test_probs[i]) for i in indices]
        cid_scores.sort(key=lambda x: x[1], reverse=True)
        test_preds[qid] = [cid for cid, _ in cid_scores[:7]]
    
    # Save and package
    import zipfile
    sub_dir = ROOT / "submissions"
    
    ltr_sub = sub_dir / f"P7_S09_ltr_{best_clf_name.lower()}"
    ltr_sub.mkdir(parents=True, exist_ok=True)
    with open(ltr_sub / "prediction.json", "w") as f:
        json.dump(test_preds, f, indent=2)
    with zipfile.ZipFile(ltr_sub / f"P7_S09_ltr_{best_clf_name.lower()}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(ltr_sub / "prediction.json", arcname="prediction.json")
    
    # Also save dev preds
    save_dir = RUNS_DIR / "ltr"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "dev_predictions.json", "w") as f:
        json.dump(best_dev_preds, f, indent=2)
    with open(save_dir / "test_predictions.json", "w") as f:
        json.dump(test_preds, f, indent=2)
    
    print("\n" + "=" * 70)
    print("LTR RESULTS:")
    print(f"  Best classifier: {best_clf_name}")
    print(f"  Dev F1@7: {best_dev_f1:.4f}")
    print(f"  Submission: {ltr_sub}")
    print("=" * 70)


if __name__ == "__main__":
    main()
