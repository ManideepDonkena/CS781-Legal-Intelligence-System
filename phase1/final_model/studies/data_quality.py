#!/usr/bin/env python3
"""Data Quality Framework for Legal LLM Pre-training.

Implements the multi-stage quality pipeline taught in lecture:
1. Rule-Based / Heuristic Quality Filtering (word lengths, char limits, symbol/digit ratios, repetition)
2. Legal Boilerplate & Header Stripping (formalizing raw case rosters to substantive prose)
3. PII Redaction / Masking (Phones, Emails, Aadhaar, PAN)
4. Fuzzy Deduplication (MinHash + Locality Sensitive Hashing - LSH)
"""
from __future__ import annotations

import re
import math
import hashlib
from typing import Iterator, Tuple, Dict, Any, List, Set
import numpy as np


def _hash64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


# ============================================================================
# 1. PII Redaction
# ============================================================================
_RE_AADHAAR = re.compile(r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b")
_RE_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_RE_PHONE = re.compile(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b")
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")

def redact_pii(text: str) -> str:
    """Mask sensitive PII while preserving legal context."""
    text = _RE_AADHAAR.sub("[AADHAAR]", text)
    text = _RE_PAN.sub("[PAN]", text)
    text = _RE_PHONE.sub("[PHONE]", text)
    text = _RE_EMAIL.sub("[EMAIL]", text)
    return text


# ============================================================================
# 2. Boilerplate & Legal Metadata Normalizer
# ============================================================================
_RE_COUNSEL_BLOCK = re.compile(r"(?:Counsel for (?:Appellant|Respondent|Petitioner|Opposite Party)\s*:-\s*[^.\n]+)+", re.IGNORECASE)
_RE_COURT_HEADER = re.compile(r"^Court No\.\s*-\s*\d+.*?(?:Hon'ble\s+[^,\n]+,\s*J\.?)", re.DOTALL | re.IGNORECASE)
_RE_MULTI_WS = re.compile(r"[ \t]+")
_RE_MULTI_NL = re.compile(r"\n{3,}")

def clean_legal_boilerplate(text: str) -> str:
    """Clean out non-semantic court headers and repetitious registry noise."""
    # Standardize whitespace
    text = _RE_MULTI_WS.sub(" ", text)
    
    # Strip excessively repetitious counsel blocks if they dominate
    text = _RE_COUNSEL_BLOCK.sub(" [COUNSEL_LIST] ", text)
    
    # Clean multiple newlines
    text = _RE_MULTI_NL.sub("\n\n", text)
    return text.strip()


# ============================================================================
# 3. Heuristic Quality Filter
# ============================================================================
class HeuristicFilter:
    def __init__(
        self,
        min_chars: int = 500,
        max_chars: int = 2_000_000,
        min_mean_word_length: float = 3.0,
        max_mean_word_length: float = 12.0,
        max_symbol_ratio: float = 0.25,
        max_digit_ratio: float = 0.20,
        max_line_repetition_ratio: float = 0.35,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_mean_word_len = min_mean_word_length
        self.max_mean_word_len = max_mean_word_length
        self.max_symbol_ratio = max_symbol_ratio
        self.max_digit_ratio = max_digit_ratio
        self.max_line_rep_ratio = max_line_repetition_ratio

    def evaluate(self, text: str) -> Tuple[bool, str]:
        """Returns (is_valid, reject_reason)."""
        n_chars = len(text)
        if n_chars < self.min_chars:
            return False, f"too_short ({n_chars} < {self.min_chars})"
        if n_chars > self.max_chars:
            return False, f"too_long ({n_chars} > {self.max_chars})"

        words = text.split()
        if not words:
            return False, "empty_words"

        mean_word_len = sum(len(w) for w in words) / len(words)
        if mean_word_len < self.min_mean_word_len:
            return False, f"mean_word_len_too_low ({mean_word_len:.2f})"
        if mean_word_len > self.max_mean_word_len:
            return False, f"mean_word_len_too_high ({mean_word_len:.2f})"

        # Non-alphanumeric & whitespace characters
        non_alpha = sum(1 for c in text if not (c.isalnum() or c.isspace()))
        symbol_ratio = non_alpha / n_chars
        if symbol_ratio > self.max_symbol_ratio:
            return False, f"symbol_ratio_too_high ({symbol_ratio:.2f})"

        digits = sum(1 for c in text if c.isdigit())
        digit_ratio = digits / n_chars
        if digit_ratio > self.max_digit_ratio:
            return False, f"digit_ratio_too_high ({digit_ratio:.2f})"

        # Check line-level repetition
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
        if len(lines) >= 5:
            unique_lines = set(lines)
            rep_ratio = 1.0 - (len(unique_lines) / len(lines))
            if rep_ratio > self.max_line_rep_ratio:
                return False, f"line_repetition_too_high ({rep_ratio:.2f})"

        return True, "pass"


# ============================================================================
# 4. Fuzzy Deduplication (Vectorized MinHash + LSH)
# ============================================================================
# Large Mersenne prime for universal hashing: 2^61 - 1
_MERSENNE_PRIME = np.uint64((1 << 61) - 1)

class MinHashLSH:
    """Ultra-fast, vectorized MinHash-LSH engine using universal hashing."""

    def __init__(
        self,
        num_perm: int = 64,
        num_bands: int = 8,
        k_shingle: int = 5,
        seed: int = 42
    ):
        assert num_perm % num_bands == 0, "num_perm must be divisible by num_bands"
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.k_shingle = k_shingle
        
        # Approximate Jaccard threshold: (1/b)^(1/r)
        self.threshold = (1.0 / num_bands) ** (1.0 / self.rows_per_band)
        
        rng = np.random.RandomState(seed)
        # Random coefficients a, b for h(x) = (a*x + b) % p
        # a must be non-zero and odd
        self.a = rng.randint(1, 2**32 - 1, size=(1, num_perm), dtype=np.uint64) | 1
        self.b = rng.randint(0, 2**32 - 1, size=(1, num_perm), dtype=np.uint64)
        
        # LSH buckets: band_idx -> bucket_hash -> list of doc_ids
        self.buckets: List[Dict[int, List[str]]] = [dict() for _ in range(num_bands)]
        self.seen_signatures: Dict[str, np.ndarray] = {}

    def _get_shingles(self, text: str) -> List[str]:
        words = text.lower().split()
        if len(words) < self.k_shingle:
            return [" ".join(words)]
        return [" ".join(words[i:i + self.k_shingle]) for i in range(len(words) - self.k_shingle + 1)]

    def compute_signature(self, text: str) -> np.ndarray:
        shingles = self._get_shingles(text)
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.uint64)
        
        # Hash each shingle once with _hash64
        hashes = np.fromiter(
            (_hash64(s.encode("utf-8")) for s in shingles),
            dtype=np.uint64,
            count=len(shingles)
        )[:, None]  # shape: (N, 1)

        # Vectorized universal hashing: (a * hashes + b) % p
        # Broadcasting produces shape (N, num_perm)
        permuted = (self.a * hashes + self.b) % _MERSENNE_PRIME
        # Signature is column-wise minimum
        return np.min(permuted, axis=0)

    def is_duplicate(self, doc_id: str, text: str) -> Tuple[bool, str | None, float]:
        """Check if document has an existing near-duplicate in index.
        Returns: (is_duplicate, matching_doc_id, estimated_jaccard)
        """
        sig = self.compute_signature(text)
        
        candidate_ids = set()
        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_bytes = sig[start:end].tobytes()
            b_hash = _hash64(band_bytes)
            
            bucket = self.buckets[b_idx]
            if b_hash in bucket:
                candidate_ids.update(bucket[b_hash])
                
        # Verify candidate signatures
        for cand_id in candidate_ids:
            cand_sig = self.seen_signatures[cand_id]
            # Fraction of matching minhash components = Jaccard estimate
            jaccard = float(np.mean(sig == cand_sig))
            if jaccard >= self.threshold:
                return True, cand_id, jaccard
                
        # If not duplicate, register document into LSH index
        self.seen_signatures[doc_id] = sig
        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_bytes = sig[start:end].tobytes()
            b_hash = _hash64(band_bytes)
            self.buckets[b_idx].setdefault(b_hash, []).append(doc_id)
            
        return False, None, 0.0
