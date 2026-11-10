"""
Cross-Encoder Re-ranker for Phase 7 PCR.

Strategy:
1. Use Phrase-BM25 to retrieve top-K candidates per query (first-stage retrieval)
2. Train a cross-encoder on training data to re-rank candidates
3. The cross-encoder learns to classify (query_chunk, candidate_chunk) pairs as relevant/not

Key insight: Documents are very long (median ~20K chars). We use a chunked approach:
- Split each document into overlapping chunks of ~384 tokens
- Score each (query_chunk, candidate_chunk) pair with the cross-encoder
- Aggregate: max similarity across all chunk pairs

We use a pre-trained legal BERT model and fine-tune it as a binary classifier.
"""

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcr
from advanced_pcr import extract_phrases, extract_citations, clean_tokens, STOPWORDS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Phase_7_PCR"
RUNS_DIR = ROOT / "runs"

DEVICE = torch.device("cuda:3")  # Use GPU 3


# ======================================================================
# FIRST STAGE: Improved Phrase-BM25 with training-aware IDF
# ======================================================================
class PhraseBM25:
    """Phrase-BM25 retriever with bigram/trigram boosting."""
    
    def __init__(self, candidates, k1=1.5, b=0.75, max_df=250):
        self.c_ids = sorted(candidates.keys())
        self.N = len(self.c_ids)
        self.k1 = k1
        self.b = b
        
        print("  Building Phrase-BM25 index...")
        t0 = time.time()
        
        cand_tf = {}
        cand_lens = {}
        df_counts = Counter()
        
        PAT_WORD = re.compile(r'\b[a-z0-9_]{2,}\b')
        
        for cid in self.c_ids:
            text = " ".join(candidates[cid]["text"]).lower()
            words = PAT_WORD.findall(text)
            filtered = [w for w in words if w not in STOPWORDS]
            
            # unigrams + bigrams + trigrams
            feats = list(filtered)
            for i in range(len(filtered) - 1):
                feats.append(filtered[i] + "__" + filtered[i+1])
            for i in range(len(filtered) - 2):
                feats.append(filtered[i] + "__" + filtered[i+1] + "__" + filtered[i+2])
            
            cand_lens[cid] = len(feats)
            cand_tf[cid] = Counter(feats)
            for term in set(feats):
                df_counts[term] += 1
        
        self.avg_len = sum(cand_lens.values()) / self.N
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in df_counts.items() if 1 <= df <= max_df
        }
        
        # Build inverted index
        self.index = defaultdict(list)
        for cid, tf_map in cand_tf.items():
            norm_factor = k1 * (1.0 - b + b * (cand_lens[cid] / self.avg_len))
            for term, tf in tf_map.items():
                if term in self.idf:
                    w = (tf * (k1 + 1.0)) / (tf + norm_factor) * self.idf[term]
                    if "__" in term:
                        w *= (2.5 if term.count("__") == 1 else 5.0)
                    self.index[term].append((cid, w))
        
        print(f"    ✓ {len(self.index)} terms in {time.time()-t0:.1f}s")
    
    def score(self, query_text_list):
        """Score all candidates for a single query. Returns {cid: score}."""
        text = " ".join(query_text_list).lower()
        PAT_WORD = re.compile(r'\b[a-z0-9_]{2,}\b')
        words = PAT_WORD.findall(text)
        filtered = [w for w in words if w not in STOPWORDS]
        
        feats = list(filtered)
        for i in range(len(filtered) - 1):
            feats.append(filtered[i] + "__" + filtered[i+1])
        for i in range(len(filtered) - 2):
            feats.append(filtered[i] + "__" + filtered[i+1] + "__" + filtered[i+2])
        
        q_feats = Counter(feats)
        scores = defaultdict(float)
        for term in q_feats:
            if term in self.index:
                for cid, w in self.index[term]:
                    scores[cid] += w
        return scores
    
    def retrieve_top_k(self, query_text_list, k=50):
        """Retrieve top-k candidate IDs."""
        scores = self.score(query_text_list)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:k]]


# ======================================================================
# MULTI-SCALE VERBATIM + CITATION SCORER 
# ======================================================================
class VerbatimCitationScorer:
    """N-gram fingerprinting + citation + bench matching."""
    
    def __init__(self, candidates):
        self.c_ids = sorted(candidates.keys())
        self.N = len(self.c_ids)
        
        print("  Building Verbatim N-gram + Citation Index...")
        t0 = time.time()
        
        PAT_CITATION = re.compile(
            r'\b(?:\d{4}\s+)?(?:AIR|SCR|SCC|ITR|ILR|CRLJ|FLR|LLJ|COMP\s*CAS|IND\.\s*CAS\.)\s*(?:\(\d+\))?\s*\d+\b',
            re.IGNORECASE
        )
        PAT_BENCH = re.compile(r'BENCH:\s*([^\n]+)', re.IGNORECASE)
        
        self.ngram_index = defaultdict(list)
        self.ngram_df = Counter()
        ns = (5, 6, 7, 8)
        
        # Extract citations and bench names
        self.cand_cits = {}
        self.cand_benches = {}
        for cid in self.c_ids:
            text = " ".join(candidates[cid]["text"][:6])
            self.cand_cits[cid] = set(
                re.sub(r'\s+', ' ', m.group(0).lower())
                for m in PAT_CITATION.finditer(text)
            )
            head = " ".join(candidates[cid]["text"][:5])
            bench_names = set()
            for b_str in PAT_BENCH.findall(head):
                cl = b_str.replace("<NAME>", "").replace(",", " ").strip().lower()
                for w in cl.split():
                    if len(w) > 3 and w not in STOPWORDS:
                        bench_names.add(w)
            self.cand_benches[cid] = bench_names
        
        # Build n-gram index
        for cid in self.c_ids:
            text = " ".join(candidates[cid]["text"])
            tokens = clean_tokens(text)
            seen_cand = set()
            for n in ns:
                for i in range(len(tokens) - n + 1):
                    ng = " ".join(tokens[i:i+n])
                    if ng not in seen_cand:
                        seen_cand.add(ng)
                        self.ngram_index[ng].append(cid)
                        self.ngram_df[ng] += 1
        
        self.PAT_CITATION = PAT_CITATION
        print(f"    ✓ {len(self.ngram_index)} n-grams in {time.time()-t0:.1f}s")
    
    def score(self, query_text_list):
        """Score all candidates for a single query."""
        scores = defaultdict(float)
        q_text = " ".join(query_text_list)
        q_lower = q_text.lower()
        q_tokens = clean_tokens(q_text)
        
        # Citation matches
        q_cits = set(
            re.sub(r'\s+', ' ', m.group(0).lower())
            for m in self.PAT_CITATION.finditer(q_text)
        )
        for cid, ccits in self.cand_cits.items():
            overlap = q_cits & ccits
            if overlap:
                scores[cid] += len(overlap) * 50.0
        
        # Bench matches
        for cid, bnames in self.cand_benches.items():
            if any(bn in q_lower for bn in bnames):
                scores[cid] += 8.0
        
        # N-grams with IDF
        ns = (5, 6, 7, 8)
        seen_query = set()
        for n in ns:
            scale_wt = n ** 2.0
            for i in range(len(q_tokens) - n + 1):
                ng = " ".join(q_tokens[i:i+n])
                if ng in seen_query:
                    continue
                seen_query.add(ng)
                if ng in self.ngram_index:
                    df = self.ngram_df[ng]
                    if df <= 30:
                        idf_val = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                        val = scale_wt * idf_val
                        for cid in self.ngram_index[ng]:
                            scores[cid] += val
        
        return scores


# ======================================================================
# CROSS-ENCODER MODEL
# ======================================================================
class CrossEncoderModel(nn.Module):
    """Cross-encoder using a pre-trained transformer with classification head."""
    
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_output).squeeze(-1)
        return logits


class PairDataset(Dataset):
    """Dataset for cross-encoder training."""
    
    def __init__(self, pairs, tokenizer, max_len=512):
        self.pairs = pairs  # list of (query_chunk, cand_chunk, label)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        q_chunk, c_chunk, label = self.pairs[idx]
        encoding = self.tokenizer(
            q_chunk, c_chunk,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32)
        }


def get_text_chunk(text_list, start_char=0, max_chars=2000):
    """Extract a chunk of text from a document's text list."""
    full_text = " ".join(text_list)
    return full_text[start_char:start_char + max_chars]


def create_training_pairs(train_queries, train_candidates, bm25_retriever, 
                          num_hard_negs=3, num_random_negs=1, num_chunks_per_doc=3,
                          chunk_chars=1500):
    """
    Create training pairs for the cross-encoder.
    For each gold (query, candidate) pair, create positive pairs from document chunks.
    For hard negatives: use top BM25 candidates that are NOT gold.
    """
    pairs = []
    q_ids = sorted(train_queries.keys())
    c_ids = sorted(train_candidates.keys())
    
    print(f"  Creating training pairs for {len(q_ids)} queries...")
    
    for i, qid in enumerate(q_ids):
        if i % 100 == 0:
            print(f"    Processing query {i}/{len(q_ids)}...")
        
        qdata = train_queries[qid]
        gold_cids = set(qdata.get("relevant_candidates", []))
        if not gold_cids:
            continue
        
        q_text = " ".join(qdata["text"])
        q_len = len(q_text)
        
        # Get query chunks (beginning, middle, end)
        q_chunks = []
        if q_len <= chunk_chars:
            q_chunks = [q_text]
        else:
            # Take beginning (most important - contains case details)
            q_chunks.append(q_text[:chunk_chars])
            # Take a middle chunk
            mid = q_len // 2
            q_chunks.append(q_text[mid:mid+chunk_chars])
        
        # Get BM25 top candidates for hard negatives
        bm25_top = bm25_retriever.retrieve_top_k(qdata["text"], k=30)
        hard_negs = [cid for cid in bm25_top if cid not in gold_cids][:num_hard_negs * 2]
        
        # Positive pairs
        for gcid in gold_cids:
            if gcid not in train_candidates:
                continue
            c_text = " ".join(train_candidates[gcid]["text"])
            c_len = len(c_text)
            
            c_chunks = []
            if c_len <= chunk_chars:
                c_chunks = [c_text]
            else:
                c_chunks.append(c_text[:chunk_chars])
                mid = c_len // 2
                c_chunks.append(c_text[mid:mid+chunk_chars])
            
            # Create positive pairs from chunk combinations
            for qc in q_chunks[:2]:
                for cc in c_chunks[:2]:
                    pairs.append((qc, cc, 1.0))
        
        # Hard negative pairs  
        selected_negs = hard_negs[:num_hard_negs]
        # Also add random negatives
        random_negs = random.sample(
            [cid for cid in c_ids if cid not in gold_cids and cid not in set(hard_negs)],
            min(num_random_negs, len(c_ids) - len(gold_cids) - len(hard_negs))
        )
        selected_negs.extend(random_negs)
        
        for ncid in selected_negs:
            if ncid not in train_candidates:
                continue
            c_text = " ".join(train_candidates[ncid]["text"])
            c_chunks = [c_text[:chunk_chars]]
            
            for qc in q_chunks[:1]:  # fewer neg chunks to balance
                for cc in c_chunks:
                    pairs.append((qc, cc, 0.0))
    
    random.shuffle(pairs)
    print(f"  ✓ Created {len(pairs)} pairs ({sum(1 for _,_,l in pairs if l > 0)} pos, {sum(1 for _,_,l in pairs if l == 0)} neg)")
    return pairs


def train_cross_encoder(pairs, model, tokenizer, epochs=3, batch_size=32, lr=2e-5, max_len=512):
    """Train the cross-encoder on the created pairs."""
    dataset = PairDataset(pairs, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1
    )
    
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()
        
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (step + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Step {step+1}/{len(loader)}, "
                      f"Loss: {total_loss/(step+1):.4f}, Acc: {correct/total:.4f}")
        
        elapsed = time.time() - t0
        print(f"  ✓ Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, "
              f"Acc={correct/total:.4f}, Time={elapsed:.0f}s")
    
    return model


def rerank_with_cross_encoder(model, tokenizer, query_text_list, candidate_texts, 
                               candidate_ids, max_len=512, chunk_chars=1500, batch_size=64):
    """Re-rank candidates using the cross-encoder with chunked scoring."""
    model.eval()
    
    q_text = " ".join(query_text_list)
    # Use beginning of query (most informative)
    q_chunk = q_text[:chunk_chars]
    
    all_scores = {}
    
    # Process in batches
    pairs = []
    cids_order = []
    for cid in candidate_ids:
        c_text = " ".join(candidate_texts[cid]["text"])
        # Score with beginning of candidate
        c_chunk = c_text[:chunk_chars]
        pairs.append((q_chunk, c_chunk))
        cids_order.append(cid)
    
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        encodings = tokenizer(
            [p[0] for p in batch_pairs],
            [p[1] for p in batch_pairs],
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            input_ids = encodings["input_ids"].to(DEVICE)
            attention_mask = encodings["attention_mask"].to(DEVICE)
            token_type_ids = encodings.get("token_type_ids", torch.zeros_like(input_ids)).to(DEVICE)
            logits = model(input_ids, attention_mask, token_type_ids)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    
    for cid, score in zip(cids_order, scores):
        all_scores[cid] = score
    
    return all_scores


# ======================================================================
# SUPERVISED FEATURE-BASED RE-RANKER (sklearn)
# ======================================================================
def compute_features(qid, cid, phrase_scores, verbatim_scores, bge_scores, p1_scores,
                     q_ids_set, c_ids_set):
    """Compute feature vector for a (query, candidate) pair."""
    features = []
    
    # Phrase-BM25 score
    ps = phrase_scores.get(qid, {}).get(cid, 0.0)
    features.append(ps)
    
    # Verbatim score  
    vs = verbatim_scores.get(qid, {}).get(cid, 0.0)
    features.append(vs)
    
    # BGE-M3 dense score
    bs = bge_scores.get(qid, {}).get(cid, 0.0)
    features.append(bs)
    
    # Phase 1 dense score
    p1s = p1_scores.get(qid, {}).get(cid, 0.0)
    features.append(p1s)
    
    # Rank features (position in each scorer's ranking)
    # These are computed externally
    
    return features


# ======================================================================
# MAIN PIPELINE
# ======================================================================
def main():
    print("=" * 70)
    print("CROSS-ENCODER RE-RANKER FOR PHASE 7 PCR")
    print("=" * 70)
    
    # Load all data
    print("\n[1/7] Loading data...")
    train_queries, train_candidates = pcr.load_split("train", data_dir=DATA_DIR)
    dev_queries, dev_candidates = pcr.load_split("dev", data_dir=DATA_DIR)
    test_queries, test_candidates = pcr.load_split("test", data_dir=DATA_DIR)
    
    print(f"  Train: {len(train_queries)} queries, {len(train_candidates)} candidates")
    print(f"  Dev: {len(dev_queries)} queries, {len(dev_candidates)} candidates")
    print(f"  Test: {len(test_queries)} queries, {len(test_candidates)} candidates")
    
    # ----------------------------------------------------------------
    # STEP 2: Build first-stage retriever on train candidates
    # ----------------------------------------------------------------
    print("\n[2/7] Building Phrase-BM25 on TRAIN candidates...")
    train_bm25 = PhraseBM25(train_candidates)
    
    # ----------------------------------------------------------------
    # STEP 3: Create cross-encoder training pairs
    # ----------------------------------------------------------------
    print("\n[3/7] Creating cross-encoder training pairs...")
    random.seed(42)
    pairs = create_training_pairs(
        train_queries, train_candidates, train_bm25,
        num_hard_negs=5, num_random_negs=2, num_chunks_per_doc=2,
        chunk_chars=1200
    )
    
    # ----------------------------------------------------------------
    # STEP 4: Train cross-encoder
    # ----------------------------------------------------------------
    print("\n[4/7] Training cross-encoder...")
    model_name = "nlpaueb/legal-bert-base-uncased"
    print(f"  Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CrossEncoderModel(model_name=model_name).to(DEVICE)
    
    model = train_cross_encoder(
        pairs, model, tokenizer,
        epochs=3, batch_size=32, lr=2e-5, max_len=512
    )
    
    # Save trained model
    save_dir = RUNS_DIR / "cross_encoder"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    tokenizer.save_pretrained(save_dir / "tokenizer")
    print(f"  ✓ Model saved to {save_dir}")
    
    # ----------------------------------------------------------------
    # STEP 5: Build first-stage retriever for DEV and apply cross-encoder
    # ----------------------------------------------------------------
    print("\n[5/7] Re-ranking DEV candidates with cross-encoder...")
    dev_bm25 = PhraseBM25(dev_candidates)
    
    dev_q_ids = sorted(dev_queries.keys())
    dev_c_ids = sorted(dev_candidates.keys())
    
    # Also load precomputed scores for ensemble
    bge_dev = json.load(open(RUNS_DIR / "dense_bge" / "dev_scores.json"))
    p1_dev = json.load(open(RUNS_DIR / "dense_phase1" / "dev_scores.json"))
    
    # For each dev query, get top-50 from BM25, re-rank with cross-encoder
    TOP_K_FIRST_STAGE = 60
    
    all_preds = {}
    all_ce_scores = {}
    
    # Also build verbatim scorer for dev
    dev_verb_scorer = VerbatimCitationScorer(dev_candidates)
    
    for qi, qid in enumerate(dev_q_ids):
        if qi % 10 == 0:
            print(f"  Query {qi+1}/{len(dev_q_ids)}...")
        
        # First stage: Phrase-BM25 top candidates
        bm25_scores = dev_bm25.score(dev_queries[qid]["text"])
        
        # Also get verbatim scores
        verb_scores = dev_verb_scorer.score(dev_queries[qid]["text"])
        
        # Combine BM25 + Verbatim for first-stage
        combined_first = {}
        for cid in dev_c_ids:
            b = bm25_scores.get(cid, 0.0)
            v = verb_scores.get(cid, 0.0)
            # Normalize roughly
            combined_first[cid] = b + v * 0.5
        
        top_cands = sorted(combined_first.keys(), key=lambda c: combined_first[c], reverse=True)[:TOP_K_FIRST_STAGE]
        
        # Second stage: Cross-encoder re-ranking
        ce_scores = rerank_with_cross_encoder(
            model, tokenizer,
            dev_queries[qid]["text"],
            dev_candidates,
            top_cands,
            max_len=512,
            chunk_chars=1200,
            batch_size=64
        )
        
        # Final fusion: cross-encoder + BM25 + verbatim + dense
        final_scores = {}
        
        # Normalize BM25 scores among top candidates
        bm25_vals = [bm25_scores.get(c, 0.0) for c in top_cands]
        bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
        
        verb_vals = [verb_scores.get(c, 0.0) for c in top_cands]
        verb_min, verb_max = min(verb_vals), max(verb_vals)
        verb_range = verb_max - verb_min if verb_max > verb_min else 1.0
        
        ce_vals = [ce_scores.get(c, 0.0) for c in top_cands]
        ce_min, ce_max = min(ce_vals), max(ce_vals)
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
        
        for cid in top_cands:
            norm_bm25 = (bm25_scores.get(cid, 0.0) - bm25_min) / bm25_range
            norm_verb = (verb_scores.get(cid, 0.0) - verb_min) / verb_range
            norm_ce = (ce_scores.get(cid, 0.0) - ce_min) / ce_range
            norm_bge = bge_dev.get(qid, {}).get(cid, 0.0)
            norm_p1 = p1_dev.get(qid, {}).get(cid, 0.0)
            
            # Weights: cross-encoder gets high weight since it's supervised
            final_scores[cid] = (
                0.40 * norm_ce +
                0.30 * norm_bm25 +
                0.20 * norm_verb +
                0.07 * norm_bge +
                0.03 * norm_p1
            )
        
        # Also add any candidates NOT in top-K but that had high verbatim/citation scores
        # (in case BM25 missed some high-citation matches)
        for cid in dev_c_ids:
            if cid not in final_scores:
                final_scores[cid] = 0.0
        
        ranked = sorted(final_scores.keys(), key=lambda c: final_scores[c], reverse=True)[:7]
        all_preds[qid] = ranked
        all_ce_scores[qid] = final_scores
    
    # Evaluate on Dev
    res = pcr.evaluate_predictions(dev_queries, all_preds)
    print("\n" + "=" * 60)
    print("CROSS-ENCODER RE-RANKER DEV RESULTS:")
    print(f"  Micro-F1@7:        {res['primary_metric_1']:.4f}")
    print(f"  Micro-Precision@7: {res['secondary_metric_1']:.4f}")
    print(f"  Micro-Recall@7:    {res['secondary_metric_2']:.4f}")
    print(f"  True Positives:    {res['tp']} (FP: {res['fp']}, FN: {res['fn']})")
    print("=" * 60)
    
    # Save dev results
    with open(save_dir / "dev_predictions.json", "w") as f:
        json.dump(all_preds, f, indent=2)
    with open(save_dir / "dev_scores.json", "w") as f:
        json.dump(all_ce_scores, f)
    
    # ----------------------------------------------------------------
    # STEP 6: Also try PURE Phrase-BM25 + Verbatim (no cross-encoder) with 
    #          optimized weights learned on training set
    # ----------------------------------------------------------------
    print("\n[6/7] Learning optimal fusion weights on training set...")
    
    # Score all train queries with Phrase-BM25 + Verbatim
    train_verb_scorer = VerbatimCitationScorer(train_candidates)
    train_c_ids = sorted(train_candidates.keys())
    
    # Try different weight combos on train set
    best_f1 = 0
    best_weights = (0.5, 0.5)
    
    # Pre-compute all scores
    print("  Pre-computing train scores...")
    train_bm25_all = {}
    train_verb_all = {}
    for qi, qid in enumerate(sorted(train_queries.keys())):
        if qi % 100 == 0:
            print(f"    Query {qi+1}/{len(train_queries)}...")
        train_bm25_all[qid] = train_bm25.score(train_queries[qid]["text"])
        train_verb_all[qid] = train_verb_scorer.score(train_queries[qid]["text"])
    
    print("  Grid searching weights...")
    for w_bm25 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for w_verb in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds = {}
            for qid in train_queries:
                scores = {}
                for cid in train_c_ids:
                    scores[cid] = (
                        w_bm25 * train_bm25_all[qid].get(cid, 0.0) +
                        w_verb * train_verb_all[qid].get(cid, 0.0)
                    )
                ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:7]
                preds[qid] = ranked
            
            res = pcr.evaluate_predictions(train_queries, preds)
            if res['primary_metric_1'] > best_f1:
                best_f1 = res['primary_metric_1']
                best_weights = (w_bm25, w_verb)
                print(f"    New best: w_bm25={w_bm25}, w_verb={w_verb}, F1={best_f1:.4f}")
    
    print(f"  ✓ Best weights: BM25={best_weights[0]}, Verb={best_weights[1]}, Train F1={best_f1:.4f}")
    
    # Apply best weights to Dev
    print("\n  Applying optimal weights to Dev...")
    opt_preds = {}
    for qid in dev_q_ids:
        bm25_scores = dev_bm25.score(dev_queries[qid]["text"])
        verb_scores = dev_verb_scorer.score(dev_queries[qid]["text"])
        
        scores = {}
        for cid in dev_c_ids:
            scores[cid] = (
                best_weights[0] * bm25_scores.get(cid, 0.0) +
                best_weights[1] * verb_scores.get(cid, 0.0)
            )
        ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:7]
        opt_preds[qid] = ranked
    
    opt_res = pcr.evaluate_predictions(dev_queries, opt_preds)
    print(f"  Optimal weights Dev F1: {opt_res['primary_metric_1']:.4f}")
    
    # ----------------------------------------------------------------
    # STEP 7: Generate TEST predictions
    # ----------------------------------------------------------------
    print("\n[7/7] Generating TEST predictions...")
    test_bm25 = PhraseBM25(test_candidates)
    test_verb_scorer = VerbatimCitationScorer(test_candidates)
    test_q_ids = sorted(test_queries.keys())
    test_c_ids = sorted(test_candidates.keys())
    
    # Load precomputed test dense scores
    bge_test = json.load(open(RUNS_DIR / "dense_bge" / "test_scores.json"))
    p1_test = json.load(open(RUNS_DIR / "dense_phase1" / "test_scores.json"))
    
    # Method A: Cross-encoder re-ranker
    test_preds_ce = {}
    for qi, qid in enumerate(test_q_ids):
        if qi % 20 == 0:
            print(f"  [CE] Query {qi+1}/{len(test_q_ids)}...")
        
        bm25_scores = test_bm25.score(test_queries[qid]["text"])
        verb_scores = test_verb_scorer.score(test_queries[qid]["text"])
        
        combined_first = {}
        for cid in test_c_ids:
            combined_first[cid] = bm25_scores.get(cid, 0.0) + verb_scores.get(cid, 0.0) * 0.5
        
        top_cands = sorted(combined_first.keys(), key=lambda c: combined_first[c], reverse=True)[:TOP_K_FIRST_STAGE]
        
        ce_scores = rerank_with_cross_encoder(
            model, tokenizer,
            test_queries[qid]["text"],
            test_candidates,
            top_cands,
            max_len=512,
            chunk_chars=1200,
            batch_size=64
        )
        
        # Normalize and fuse
        bm25_vals = [bm25_scores.get(c, 0.0) for c in top_cands]
        bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
        
        verb_vals = [verb_scores.get(c, 0.0) for c in top_cands]
        verb_min, verb_max = min(verb_vals), max(verb_vals)
        verb_range = verb_max - verb_min if verb_max > verb_min else 1.0
        
        ce_vals = [ce_scores.get(c, 0.0) for c in top_cands]
        ce_min, ce_max = min(ce_vals), max(ce_vals)
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
        
        final_scores = {}
        for cid in top_cands:
            norm_bm25 = (bm25_scores.get(cid, 0.0) - bm25_min) / bm25_range
            norm_verb = (verb_scores.get(cid, 0.0) - verb_min) / verb_range
            norm_ce = (ce_scores.get(cid, 0.0) - ce_min) / ce_range
            norm_bge = bge_test.get(qid, {}).get(cid, 0.0)
            norm_p1 = p1_test.get(qid, {}).get(cid, 0.0)
            
            final_scores[cid] = (
                0.40 * norm_ce +
                0.30 * norm_bm25 +
                0.20 * norm_verb +
                0.07 * norm_bge +
                0.03 * norm_p1
            )
        
        for cid in test_c_ids:
            if cid not in final_scores:
                final_scores[cid] = 0.0
        
        ranked = sorted(final_scores.keys(), key=lambda c: final_scores[c], reverse=True)[:7]
        test_preds_ce[qid] = ranked
    
    # Method B: Optimized BM25+Verbatim (no cross-encoder)
    test_preds_opt = {}
    for qid in test_q_ids:
        bm25_scores = test_bm25.score(test_queries[qid]["text"])
        verb_scores = test_verb_scorer.score(test_queries[qid]["text"])
        
        scores = {}
        for cid in test_c_ids:
            scores[cid] = (
                best_weights[0] * bm25_scores.get(cid, 0.0) +
                best_weights[1] * verb_scores.get(cid, 0.0)
            )
        ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:7]
        test_preds_opt[qid] = ranked
    
    # Save both test predictions
    with open(save_dir / "test_predictions_ce.json", "w") as f:
        json.dump(test_preds_ce, f, indent=2)
    with open(save_dir / "test_predictions_opt.json", "w") as f:
        json.dump(test_preds_opt, f, indent=2)
    
    # Package submissions
    import zipfile
    sub_dir = ROOT / "submissions"
    
    # CE submission
    ce_sub = sub_dir / "P7_S07_cross_encoder_reranker"
    ce_sub.mkdir(parents=True, exist_ok=True)
    with open(ce_sub / "prediction.json", "w") as f:
        json.dump(test_preds_ce, f, indent=2)
    with zipfile.ZipFile(ce_sub / "P7_S07_cross_encoder_reranker.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(ce_sub / "prediction.json", arcname="prediction.json")
    
    # Optimized BM25+Verbatim submission
    opt_sub = sub_dir / "P7_S08_optimized_bm25_verbatim"
    opt_sub.mkdir(parents=True, exist_ok=True)
    with open(opt_sub / "prediction.json", "w") as f:
        json.dump(test_preds_opt, f, indent=2)
    with zipfile.ZipFile(opt_sub / "P7_S08_optimized_bm25_verbatim.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(opt_sub / "prediction.json", arcname="prediction.json")
    
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print(f"  CE Dev F1@7:       {res['primary_metric_1']:.4f}")
    print(f"  OptBM25 Dev F1@7:  {opt_res['primary_metric_1']:.4f}")
    print(f"\nSubmissions saved to: {sub_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
