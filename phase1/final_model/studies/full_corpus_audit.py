#!/usr/bin/env python3
"""Comprehensive Whole-Corpus Pre-training Data Audit.

Audits all 3.35+ million documents across the 4 official sources:
1. Global Corpus Scale & Storage Metrics
2. Document Length Distributions & Percentiles
3. Filtering Frontier (Yield vs Threshold Curves)
4. Quality & Noise Taxonomy (KrutiDev mojibake, symbol density, repetition)
5. Global PII Prevalence (Phones, Emails, Aadhaar, PAN)
6. Train-to-Validation Benchmark Contamination (13-gram matching)
7. Vocabulary Utilization & Dead-Token Analysis

Parallelized across CPU cores using SQLite rowid range chunking.
"""
from __future__ import annotations

import os
import sys
import re
import math
import time
import json
import sqlite3
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Any, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "artifacts" / "corpus"
CACHES_DIR = ROOT / "artifacts" / "caches"
RESULTS_DIR = ROOT / "results"

SOURCES = ["supreme_court", "acts", "tribunals", "high_courts"]

# Regex patterns
RE_AADHAAR = re.compile(r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b")
RE_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
RE_PHONE = re.compile(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b")
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
RE_KRUTIDEV = re.compile(r"[a-zA-Z]{0,2}[kK][a-zA-Z]{0,2}[;%@&][a-zA-Z]{1,4}|[fF]ookn|[iI]zfrrks\"k|vihy|cuke")
RE_ASCII_TABLE = re.compile(r"[-=_\.]{6,}|(?:\|[^|\n]+\|){2,}")

THRESHOLDS = [200, 400, 600, 800, 1000, 1500]


def process_chunk(args: Tuple[str, int, int]) -> Dict[str, Any]:
    """Process a chunk of documents from rowid start to end."""
    source, start_id, end_id = args
    db_path = CORPUS_DIR / source / "documents.sqlite"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-16384")
    
    cur = conn.execute(
        "SELECT rowid, split, text FROM docs WHERE rowid BETWEEN ? AND ?",
        (start_id, end_id)
    )
    
    char_lengths = []
    word_counts = []
    byte_counts = []
    
    # Counts
    total_docs = 0
    train_docs = 0
    val_docs = 0
    
    pii_phone = 0
    pii_email = 0
    pii_aadhaar = 0
    pii_pan = 0
    
    noise_krutidev = 0
    noise_table = 0
    noise_high_symbol = 0
    noise_high_digit = 0
    noise_word_len = 0
    noise_line_rep = 0
    
    yield_docs = {t: 0 for t in THRESHOLDS}
    yield_bytes = {t: 0 for t in THRESHOLDS}
    
    for rowid, split, text in cur:
        total_docs += 1
        if split == "train":
            train_docs += 1
        else:
            val_docs += 1
            
        c_len = len(text)
        b_len = len(text.encode("utf-8"))
        words = text.split()
        w_len = len(words)
        
        char_lengths.append(c_len)
        word_counts.append(w_len)
        byte_counts.append(b_len)
        
        # Yield analysis
        for t in THRESHOLDS:
            if c_len >= t:
                yield_docs[t] += 1
                yield_bytes[t] += b_len
                
        # PII checks
        if RE_PHONE.search(text): pii_phone += 1
        if RE_EMAIL.search(text): pii_email += 1
        if RE_AADHAAR.search(text): pii_aadhaar += 1
        if RE_PAN.search(text): pii_pan += 1
        
        # Noise checks
        if RE_KRUTIDEV.search(text): noise_krutidev += 1
        if RE_ASCII_TABLE.search(text): noise_table += 1
        
        if c_len > 0:
            sym_ratio = sum(1 for c in text if not (c.isalnum() or c.isspace())) / c_len
            digit_ratio = sum(1 for c in text if c.isdigit()) / c_len
            if sym_ratio > 0.20: noise_high_symbol += 1
            if digit_ratio > 0.20: noise_high_digit += 1
            
        if w_len > 0:
            mean_w = c_len / w_len
            if mean_w < 3.0 or mean_w > 12.0:
                noise_word_len += 1
                
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
        if len(lines) >= 5:
            rep_ratio = 1.0 - (len(set(lines)) / len(lines))
            if rep_ratio > 0.35:
                noise_line_rep += 1

    conn.close()
    
    return {
        "source": source,
        "total_docs": total_docs,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "char_lengths": char_lengths,
        "word_counts": word_counts,
        "byte_counts": byte_counts,
        "yield_docs": yield_docs,
        "yield_bytes": yield_bytes,
        "pii": {
            "phone": pii_phone,
            "email": pii_email,
            "aadhaar": pii_aadhaar,
            "pan": pii_pan,
        },
        "noise": {
            "krutidev": noise_krutidev,
            "table_ascii": noise_table,
            "high_symbol": noise_high_symbol,
            "high_digit": noise_high_digit,
            "bad_word_length": noise_word_len,
            "line_repetition": noise_line_rep,
        }
    }


def audit_contamination() -> Dict[str, Any]:
    """Audit 13-gram contamination between train and Supreme Court validation set."""
    print("[1/3] Auditing Train-to-Validation Contamination (13-gram overlap)...", flush=True)
    val_db = sqlite3.connect(f"file:{CORPUS_DIR / 'supreme_court' / 'documents.sqlite'}?mode=ro", uri=True)
    cur = val_db.execute("SELECT text FROM docs WHERE split='validation'")
    
    val_13grams = set()
    total_val_words = 0
    for (text,) in cur:
        words = text.lower().split()
        total_val_words += len(words)
        for i in range(len(words) - 13 + 1):
            shingle = " ".join(words[i:i+13])
            val_13grams.add(xxhash_64(shingle))
    val_db.close()
    
    print(f"  Extracted {len(val_13grams):,} unique 13-grams from {total_val_words:,} validation words.", flush=True)
    
    # Check sample of 5,000 training documents per source
    contamination_results = {}
    for s in SOURCES:
        db = sqlite3.connect(f"file:{CORPUS_DIR / s / 'documents.sqlite'}?mode=ro", uri=True)
        cur = db.execute("SELECT text FROM docs WHERE split='train' LIMIT 5000")
        docs_checked = 0
        contaminated_docs = 0
        matched_shingles = 0
        for (text,) in cur:
            docs_checked += 1
            words = text.lower().split()
            has_hit = False
            for i in range(len(words) - 13 + 1):
                shingle = " ".join(words[i:i+13])
                h = xxhash_64(shingle)
                if h in val_13grams:
                    matched_shingles += 1
                    has_hit = True
            if has_hit:
                contaminated_docs += 1
        db.close()
        contamination_results[s] = {
            "docs_checked": docs_checked,
            "contaminated_docs": contaminated_docs,
            "contamination_pct": (contaminated_docs / max(docs_checked, 1)) * 100,
            "matched_13grams": matched_shingles,
        }
    return contamination_results


def xxhash_64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def audit_vocab_utilization() -> Dict[str, Any]:
    """Audit vocabulary distribution & dead token count in 50k and 32k tokenizers."""
    print("[2/3] Auditing Vocabulary Utilization & Token Frequency...", flush=True)
    from tokenizers import Tokenizer
    tok50_path = CACHES_DIR / "bpe50k_mixA_600M" / "tokenizer.json"
    tok32_path = CACHES_DIR / "bpe32k_mixA_600M" / "tokenizer.json"
    
    res = {}
    for name, path in [("vocab_50k", tok50_path), ("vocab_32k", tok32_path)]:
        if not path.exists():
            continue
        tok = Tokenizer.from_file(str(path))
        vocab_size = tok.get_vocab_size()
        
        # Sample 1000 docs across sources and measure token frequencies
        counts = [0] * vocab_size
        total_tokens = 0
        for s in SOURCES:
            db = sqlite3.connect(f"file:{CORPUS_DIR / s / 'documents.sqlite'}?mode=ro", uri=True)
            cur = db.execute("SELECT text FROM docs LIMIT 250")
            for (text,) in cur:
                ids = tok.encode(text).ids
                total_tokens += len(ids)
                for tid in ids:
                    if tid < vocab_size:
                        counts[tid] += 1
            db.close()
            
        active_tokens = sum(1 for c in counts if c > 0)
        rare_tokens = sum(1 for c in counts if c <= 2)
        heavy_tokens = sum(1 for c in counts if c >= 50)
        
        res[name] = {
            "vocab_size": vocab_size,
            "active_tokens_sampled": active_tokens,
            "active_pct": (active_tokens / vocab_size) * 100,
            "rare_or_unused_pct": (rare_tokens / vocab_size) * 100,
            "heavy_tokens_pct": (heavy_tokens / vocab_size) * 100,
            "sample_tokens": total_tokens
        }
    return res


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"STARTING WHOLE-CORPUS PRE-TRAINING DATA AUDIT ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"Detected {cpu_count()} CPU cores. Partitioning SQLite ranges...")
    print("=" * 70, flush=True)
    
    # 1. Contamination Audit
    contamination = audit_contamination()
    
    # 2. Vocabulary Utilization Audit
    vocab_audit = audit_vocab_utilization()
    
    # 3. Whole Corpus Parallel Sweep
    print("[3/3] Running Whole-Corpus Scale, Distribution, Yield & PII Sweep...", flush=True)
    tasks = []
    chunk_size = 50_000
    
    for source in SOURCES:
        db_path = CORPUS_DIR / source / "documents.sqlite"
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        min_id, max_id, count = conn.execute("SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM docs").fetchone()
        conn.close()
        
        curr = min_id
        while curr <= max_id:
            tasks.append((source, curr, min(curr + chunk_size - 1, max_id)))
            curr += chunk_size
            
    print(f"Created {len(tasks)} parallel chunks across {SOURCES}.", flush=True)
    
    # Use 32 parallel workers
    n_workers = min(32, cpu_count())
    with Pool(processes=n_workers) as pool:
        chunk_results = pool.map(process_chunk, tasks)
        
    print(f"Chunk processing complete in {time.time() - t0:.1f}s! Aggregating metrics...", flush=True)
    
    # Aggregation per source
    source_stats = {}
    for source in SOURCES:
        s_chunks = [c for c in chunk_results if c["source"] == source]
        
        all_chars = []
        all_words = []
        all_bytes = []
        
        total_docs = sum(c["total_docs"] for c in s_chunks)
        train_docs = sum(c["train_docs"] for c in s_chunks)
        val_docs = sum(c["val_docs"] for c in s_chunks)
        
        yield_docs = {t: sum(c["yield_docs"][t] for c in s_chunks) for t in THRESHOLDS}
        yield_bytes = {t: sum(c["yield_bytes"][t] for c in s_chunks) for t in THRESHOLDS}
        
        pii = {
            k: sum(c["pii"][k] for c in s_chunks)
            for k in ["phone", "email", "aadhaar", "pan"]
        }
        noise = {
            k: sum(c["noise"][k] for c in s_chunks)
            for k in ["krutidev", "table_ascii", "high_symbol", "high_digit", "bad_word_length", "line_repetition"]
        }
        
        for c in s_chunks:
            all_chars.extend(c["char_lengths"])
            all_words.extend(c["word_counts"])
            all_bytes.extend(c["byte_counts"])
            
        arr_chars = np.array(all_chars, dtype=np.int64)
        arr_words = np.array(all_words, dtype=np.int64)
        arr_bytes = np.array(all_bytes, dtype=np.int64)
        
        total_raw_bytes = int(arr_bytes.sum())
        total_raw_words = int(arr_words.sum())
        total_raw_chars = int(arr_chars.sum())
        
        # Percentiles
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        char_percentiles = {f"p{p}": float(np.percentile(arr_chars, p)) for p in pcts}
        char_percentiles["min"] = int(arr_chars.min()) if len(arr_chars) else 0
        char_percentiles["max"] = int(arr_chars.max()) if len(arr_chars) else 0
        char_percentiles["mean"] = float(arr_chars.mean()) if len(arr_chars) else 0
        
        source_stats[source] = {
            "total_documents": total_docs,
            "train_documents": train_docs,
            "validation_documents": val_docs,
            "total_bytes_gb": total_raw_bytes / 1e9,
            "total_words_millions": total_raw_words / 1e6,
            "total_chars_millions": total_raw_chars / 1e6,
            "length_percentiles": char_percentiles,
            "yield_frontier": {
                f">={t}_chars": {
                    "retained_docs": yield_docs[t],
                    "retained_docs_pct": (yield_docs[t] / max(total_docs, 1)) * 100,
                    "retained_bytes_pct": (yield_bytes[t] / max(total_raw_bytes, 1)) * 100,
                }
                for t in THRESHOLDS
            },
            "pii_counts": pii,
            "pii_prevalence_pct": {k: (v / max(total_docs, 1)) * 100 for k, v in pii.items()},
            "noise_counts": noise,
            "noise_prevalence_pct": {k: (v / max(total_docs, 1)) * 100 for k, v in noise.items()},
        }
        
    audit_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_corpus_documents": sum(s["total_documents"] for s in source_stats.values()),
        "total_corpus_bytes_gb": sum(s["total_bytes_gb"] for s in source_stats.values()),
        "sources": source_stats,
        "contamination_audit": contamination,
        "vocabulary_utilization": vocab_audit,
        "execution_seconds": time.time() - t0,
    }
    
    out_json = RESULTS_DIR / "FULL_CORPUS_DATA_AUDIT.json"
    out_json.write_text(json.dumps(audit_report, indent=2))
    print(f"\n[DONE] Full audit saved to {out_json} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
