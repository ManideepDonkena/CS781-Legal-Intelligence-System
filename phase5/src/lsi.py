"""Shared code for Phase 5 (legal statute identification): data, scoring, thresholds, submission files.

Each case has 1-31 of 100 statute ids.  Scores follow the task: macro-F1 over the 100 statutes (primary), micro-F1
and subset accuracy (secondary).
"""
import json
import os
import re
import zipfile

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Phase_5_LSI")
N_LABELS = 100


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


MASKS = {"<ENTITY>": "X", "<SECTION>": "SECTION", "<ACT>": "ACT"}  # 5 tokens each -> 1 token each


def compact_masks(text):
    """Replace the placeholders by single-token stand-ins, so they take less of the model's window."""
    for k, v in MASKS.items():
        text = text.replace(k, v)
    return text


def load(split, data_dir=DATA):
    """Cases of 'train', 'dev' or 'test' as {id, text, y}; y is a 0/1 vector of length 100 (None for test)."""
    path = {"train": "Train/train.json", "dev": "Train/dev.json", "test": "Test/test.json"}[split]
    docs = []
    for x in json.load(open(f"{data_dir}/{path}")):
        y = None
        if "labels" in x and split != "test":
            y = np.zeros(N_LABELS, dtype=np.int8)
            y[sorted(set(x["labels"]))] = 1
        docs.append({"id": x["id"], "text": normalize(x["text"]), "y": y})
    return docs


def dev_folds(n, k, seed=0):
    """Fold number (0..k-1) of each dev case, for training on train + the other dev folds (cross-fitting)."""
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k
    return fold


def label_matrix(docs):
    return np.stack([d["y"] for d in docs])


def scores(gold, pred):
    """gold, pred: (docs, 100) 0/1.  Per-label F1 is 0 when a label has no gold and no predicted case."""
    gold, pred = np.asarray(gold).astype(bool), np.asarray(pred).astype(bool)
    tp = (gold & pred).sum(0)
    fp = (~gold & pred).sum(0)
    fn = (gold & ~pred).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    return {"macro_f1": float(f1.mean()), "micro_f1": float(micro),
            "subset_acc": float(np.mean((gold == pred).all(1))), "per_label": f1}


def report(name, s):
    return (f"{name}: macro-F1 {100 * s['macro_f1']:.2f} | micro-F1 {100 * s['micro_f1']:.2f} | "
            f"subset acc {100 * s['subset_acc']:.2f}")


def best_thresholds(gold, prob, grid=np.linspace(0.02, 0.9, 45)):
    """Per-label threshold that maximises that label's F1 (macro-F1 is the mean of per-label F1s).
    Besides the fixed grid, each label also tries the upper quantiles of its own scores, which matters when the
    scores are not calibrated probabilities (e.g. averaged standardised log-odds)."""
    gold = np.asarray(gold).astype(bool)
    th = np.zeros(gold.shape[1])
    for j in range(gold.shape[1]):
        best = -1
        for t in np.concatenate([grid, np.quantile(prob[:, j], np.linspace(0.5, 0.995, 45))]):
            p = prob[:, j] >= t
            tp = np.sum(p & gold[:, j])
            f = 2 * tp / max(p.sum() + gold[:, j].sum(), 1)
            if f > best:
                best, th[j] = f, t
    return th


def shrunk_thresholds(gold, prob, m):
    """Per-label thresholds pulled toward one shared threshold: label j keeps weight pos_j / (pos_j + m) on its own
    best threshold, where pos_j is its number of positive cases (rare labels' own thresholds are noisier)."""
    gold = np.asarray(gold).astype(bool)
    own = best_thresholds(gold, prob)
    if m == 0:
        return own
    grid = np.linspace(0.02, 0.9, 89)
    shared = grid[int(np.argmax([scores(gold, decide(prob, np.full(gold.shape[1], t)))["macro_f1"] for t in grid]))]
    w = gold.sum(0) / (gold.sum(0) + m)
    return w * own + (1 - w) * shared


MIX = np.linspace(0, 1, 6)  # share of family B tried for each statute


def fit_mix(gold, pa, pb, mix=MIX):
    """Per statute: the share of family B (pb) in the mix with family A (pa) and the threshold that together give
    that statute's best F1.  Returns (share per statute, threshold per statute)."""
    gold = np.asarray(gold).astype(bool)
    share, th = np.zeros(gold.shape[1]), np.zeros(gold.shape[1])
    for j in range(gold.shape[1]):
        best = -1
        for a in mix:
            p = a * pb[:, j] + (1 - a) * pa[:, j]
            for t in np.concatenate([np.linspace(0.02, 0.9, 45), np.quantile(p, np.linspace(0.5, 0.995, 45))]):
                pr = p >= t
                f = 2 * np.sum(pr & gold[:, j]) / max(pr.sum() + gold[:, j].sum(), 1)
                if f > best:
                    best, share[j], th[j] = f, a, t
    return share, th


def honest_mix_score(gold, pa, pb, seed=0, halves=3):
    """Like honest_threshold_score, but the per-statute mix is also chosen on the other half."""
    rng, out = np.random.default_rng(seed), []
    for _ in range(halves):
        perm = rng.permutation(len(gold))
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        for x, z in ((a, b), (b, a)):
            share, th = fit_mix(gold[x], pa[x], pb[x])
            out.append(scores(gold[z], decide(share * pb[z] + (1 - share) * pa[z], th))["macro_f1"])
    return float(np.mean(out))


def decide(prob, th):
    """Labels above their threshold; a case with none gets its single most likely label (every case has one)."""
    pred = prob >= th
    empty = ~pred.any(1)
    pred[empty, prob[empty].argmax(1)] = True
    return pred


def honest_threshold_score(gold, prob, seed=0, halves=5, shrink=0):
    """Thresholds tuned on one half of the cases and scored on the other half, both ways, averaged over a few
    random halvings: an estimate of the score after tuning that does not reuse the scored cases."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(halves):
        perm = rng.permutation(len(gold))
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        for x, z in ((a, b), (b, a)):
            th = shrunk_thresholds(gold[x], prob[x], shrink)
            out.append(scores(gold[z], decide(prob[z], th))["macro_f1"])
    return float(np.mean(out))


def write_submission(ids, pred, out_dir):
    """prediction.json as [{id, pred: [statute ids]}] plus a zip with that one file, named after the folder."""
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"id": i, "pred": [int(j) for j in np.flatnonzero(p)]} for i, p in zip(ids, pred)]
    with open(f"{out_dir}/prediction.json", "w") as f:
        json.dump(rows, f)
    name = os.path.basename(os.path.normpath(out_dir))
    with zipfile.ZipFile(f"{out_dir}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{out_dir}/prediction.json", "prediction.json")
    return f"{out_dir}/{name}.zip"
