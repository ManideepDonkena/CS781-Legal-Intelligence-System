"""Word-count features, logistic regression and nearest neighbours written from scratch.

No scikit-learn, SciPy or joblib: only the Python standard library, NumPy arrays and PyTorch tensors on the GPU.

Features (TF-IDF):
  * words: lowercase, tokens are runs of two or more word characters, then word n-grams of length 1..ngram;
  * letters: lowercase, every whitespace-separated word padded with one space on each side, then all letter
    sequences of length 3..ngram inside it (a word shorter than n gives itself once);
  * keep terms that occur in at least min_df documents and at most max_df of them, then the max_features terms
    with the largest total count;
  * weight = (1 + log(count)) * idf, idf = log((1 + n_docs) / (1 + df)) + 1, each document scaled to length 1.
Matrices are CSR arrays (indptr, indices, values) built here and turned into torch sparse tensors for the maths.

Logistic regression: one per output column, all trained together by full-batch gradient descent with the
gradient written out by hand, X^T (sigmoid(X W + b) - Y) / n + lam * W, and a hand-written Adam update.
lam = 1 / (C * n) gives the same optimum as the usual "C" formulation (except that the bias is not penalised).

Nearest neighbours: cosine similarity is a sparse matrix product of unit-length rows; the k most similar labelled
cases vote, each with weight similarity ** power.
"""
import math
import re
from collections import Counter
from multiprocessing import Pool

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Sparse")  # torch notes that its sparse tensors are in beta

WORD_RE = re.compile(r"(?u)\b\w\w+\b")
SPACE_RE = re.compile(r"\s\s+")
_TEXTS, _CFG, _VOCAB = None, None, None  # read by worker processes (inherited when the pool forks)


def _letter_grams(word, lo, hi, cache):
    grams = cache.get(word)
    if grams is None:
        w = " " + word + " "
        grams = []
        for n in range(lo, hi + 1):
            if len(w) <= n:  # the whole padded word, counted once, and no longer sequences
                grams.append(w)
                break
            grams.extend(w[i:i + n] for i in range(len(w) - n + 1))
        cache[word] = grams
    return grams


def terms(text, cfg, cache):
    """Counter of the terms of one document."""
    text = text.lower()
    if cfg["letters"]:
        out = Counter()
        for word, k in Counter(SPACE_RE.sub(" ", text).split()).items():
            for g in _letter_grams(word, 3, cfg["ngram"], cache):
                out[g] += k
        return out
    toks = WORD_RE.findall(text)
    grams = list(toks)
    for n in range(2, cfg["ngram"] + 1):
        grams.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return Counter(grams)


def _count_range(span):
    """Document frequency and total count of every term in a range of documents."""
    df, tf, cache = Counter(), Counter(), {}
    for i in range(*span):
        c = terms(_TEXTS[i], _CFG, cache)
        df.update(c.keys())
        tf.update(c)
    return df, tf


def _rows_range(span):
    """CSR pieces (row lengths, column ids, raw counts) of a range of documents for the fixed vocabulary."""
    lens, cols, vals, cache = [], [], [], {}
    for i in range(*span):
        c = terms(_TEXTS[i], _CFG, cache)
        pairs = sorted((_VOCAB[t], k) for t, k in c.items() if t in _VOCAB)
        lens.append(len(pairs))
        cols.extend(p[0] for p in pairs)
        vals.extend(p[1] for p in pairs)
    return np.array(lens, np.int64), np.array(cols, np.int64), np.array(vals, np.float32)


def _spans(n, parts):
    step = max(1, math.ceil(n / parts))
    return [(s, min(n, s + step)) for s in range(0, n, step)]


class Tfidf:
    def __init__(self, ngram=2, letters=False, min_df=3, max_df=0.9, max_features=400000, workers=32):
        self.cfg = {"ngram": ngram, "letters": letters}
        self.min_df, self.max_df, self.max_features, self.workers = min_df, max_df, max_features, workers

    def fit(self, texts):
        global _TEXTS, _CFG
        _TEXTS, _CFG = texts, self.cfg
        with Pool(self.workers) as pool:
            parts = pool.map(_count_range, _spans(len(texts), self.workers * 4))
        df, tf = Counter(), Counter()
        for d, t in parts:
            df.update(d)
            tf.update(t)
        n = len(texts)
        keep = [t for t, d in df.items() if self.min_df <= d <= self.max_df * n]
        keep.sort(key=lambda t: (-tf[t], t))
        keep = sorted(keep[:self.max_features])
        self.vocab = {t: j for j, t in enumerate(keep)}
        d = np.array([df[t] for t in keep], np.float64)
        self.idf = (np.log((1 + n) / (1 + d)) + 1).astype(np.float32)
        return self

    def transform(self, texts):
        """CSR arrays (indptr, indices, values) of the TF-IDF matrix, rows of unit length."""
        global _TEXTS, _CFG, _VOCAB
        _TEXTS, _CFG, _VOCAB = texts, self.cfg, self.vocab
        with Pool(self.workers) as pool:
            parts = pool.map(_rows_range, _spans(len(texts), self.workers * 4))
        lens = np.concatenate([p[0] for p in parts])
        cols = np.concatenate([p[1] for p in parts])
        vals = np.concatenate([p[2] for p in parts])
        vals = (1 + np.log(vals)) * self.idf[cols]
        indptr = np.concatenate([[0], np.cumsum(lens)])
        row = np.repeat(np.arange(len(lens)), lens)
        norm = np.sqrt(np.bincount(row, weights=vals.astype(np.float64) ** 2, minlength=len(lens)))
        vals = (vals / np.maximum(norm[row], 1e-12)).astype(np.float32)
        return indptr, cols, vals


def rows(csr, idx):
    """CSR arrays of the selected rows (in the given order)."""
    indptr, cols, vals = csr
    idx = np.asarray(idx, np.int64)
    lens = indptr[idx + 1] - indptr[idx]
    ptr = np.concatenate([[0], np.cumsum(lens)])
    take = np.repeat(indptr[idx] - ptr[:-1], lens) + np.arange(ptr[-1])
    return ptr, cols[take], vals[take]


def stack(*parts):
    """Rows of several CSR matrices one after another."""
    ptr, cols, vals, off = [np.zeros(1, np.int64)], [], [], 0
    for p, c, v in parts:
        ptr.append(p[1:] + off)
        off += p[-1]
        cols.append(c)
        vals.append(v)
    return np.concatenate(ptr), np.concatenate(cols), np.concatenate(vals)


def to_torch(csr, n_cols, device="cuda"):
    indptr, cols, vals = csr
    return torch.sparse_csr_tensor(torch.from_numpy(indptr), torch.from_numpy(cols), torch.from_numpy(vals),
                                   size=(len(indptr) - 1, n_cols), device=device)


def transpose(csr, n_cols):
    """CSR arrays of the transposed matrix (columns become rows)."""
    indptr, cols, vals = csr
    row = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    order = np.argsort(cols, kind="stable")
    counts = np.bincount(cols, minlength=n_cols)
    return np.concatenate([[0], np.cumsum(counts)]), row[order], vals[order]


def train_logreg(csr, n_cols, Y, C, steps=400, lr=0.2, device="cuda", log_every=0, weights=None):
    """One L2-penalised logistic regression per column of Y (n, outputs), trained jointly. Returns (W, b).
    weights (n,): optional weight of each case in the loss (scaled to average 1)."""
    n = len(csr[0]) - 1
    X = to_torch(csr, n_cols, device)
    XT = to_torch(transpose(csr, n_cols), n, device)
    Y = torch.as_tensor(np.asarray(Y, np.float32).reshape(n, -1), device=device)
    w = torch.ones(n, 1, device=device) if weights is None else \
        torch.as_tensor(np.asarray(weights, np.float32).reshape(n, 1) / np.mean(weights), device=device)
    lam = 1.0 / (C * n)
    W = torch.zeros(n_cols, Y.shape[1], device=device)
    b = torch.logit(Y.mean(0).clamp(1e-4, 1 - 1e-4))  # start from the base rate
    state = [[torch.zeros_like(W), torch.zeros_like(W)], [torch.zeros_like(b), torch.zeros_like(b)]]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        Z = X @ W + b
        R = w * (torch.sigmoid(Z) - Y) / n
        grads = [XT @ R + lam * W, R.sum(0)]
        step_lr = lr * 0.5 * (1 + math.cos(math.pi * (t - 1) / steps))
        for p, g, (m, v) in zip((W, b), grads, state):
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            p.sub_(step_lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps))
        if log_every and (t % log_every == 0 or t == steps):
            loss = torch.nn.functional.softplus(Z) - Y * Z  # binary cross-entropy from the logits
            obj = (w * loss).mean(0).sum() + 0.5 * lam * (W ** 2).sum()
            print(f"    step {t}: objective {obj.item():.5f}", flush=True)
    return W, b


def predict_logreg(csr, n_cols, W, b, device="cuda", block=20000):
    out = []
    for s in range(0, len(csr[0]) - 1, block):
        part = rows(csr, np.arange(s, min(len(csr[0]) - 1, s + block)))
        out.append(torch.sigmoid(to_torch(part, n_cols, device) @ W + b).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def knn_vote(q_csr, k_csr, n_cols, Y, k=30, power=2.0, device="cuda", block=1000):
    """Similarity-weighted share of each label among the k most similar rows of k_csr, for every row of q_csr."""
    K = to_torch(k_csr, n_cols, device)  # (labelled cases, features), sparse
    Y = torch.as_tensor(np.asarray(Y, np.float32), device=device)
    out = []
    for s in range(0, len(q_csr[0]) - 1, block):
        part = rows(q_csr, np.arange(s, min(len(q_csr[0]) - 1, s + block)))
        Q = to_torch(part, n_cols, device).to_dense()  # (block, features)
        sim = (K @ Q.T).T  # (block, labelled cases)
        top, idx = sim.topk(k, dim=1)
        w = top.clamp_min(0) ** power
        out.append((torch.einsum("bk,bkl->bl", w, Y[idx]) / w.sum(1, keepdim=True).clamp_min(1e-9)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def dev_folds(n, k, seed=0):
    """Fold number (0..k-1) of each case, from a fixed random order."""
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k
    return fold

