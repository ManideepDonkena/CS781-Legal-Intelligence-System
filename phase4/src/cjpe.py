"""Shared code for Phase 4 (case judgement prediction, ILDC): data, scoring, submission files.

Train texts keep their capitals and have no line breaks, while dev and test are lowercased and wrapped with line
breaks, so every text is lowercased and its whitespace collapsed.  Train repeats 5,082 cases (single- and
multi-instance copies with identical text and label) and dev lists every case twice; both are de-duplicated by id.
The test file still carries a 'label' field although the task says labels were removed: it is dropped on load and
never used.
"""
import json
import os
import re
import zipfile

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Phase_4_CJPE")


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def load(split, data_dir=DATA):
    """Unique cases of 'train', 'dev' or 'test' as {id, text, y}; y is None for test."""
    path = {"train": "Train/train.json", "dev": "Train/dev.json", "test": "Test/test.json"}[split]
    docs, seen = [], set()
    for x in json.load(open(f"{data_dir}/{path}")):
        if x["id"] in seen:
            continue
        seen.add(x["id"])
        docs.append({"id": x["id"], "text": normalize(x["text"]),
                     "y": None if split == "test" else int(x["label"])})
    return docs


def scores(gold, pred):
    """Macro-F1 (primary) and macro precision / recall over the two classes."""
    gold, pred = np.asarray(gold), np.asarray(pred)
    p, r, f = [], [], []
    for c in (0, 1):
        tp = np.sum((pred == c) & (gold == c))
        pc = tp / max(np.sum(pred == c), 1)
        rc = tp / max(np.sum(gold == c), 1)
        p.append(pc)
        r.append(rc)
        f.append(2 * pc * rc / max(pc + rc, 1e-12))
    return {"macro_f1": float(np.mean(f)), "precision": float(np.mean(p)), "recall": float(np.mean(r)),
            "accuracy": float(np.mean(gold == pred))}


def best_threshold(gold, prob):
    """Threshold with the best macro-F1; candidates are the quantiles 5%..95% of prob (works for any calibration)."""
    grid = np.quantile(prob, np.linspace(0.05, 0.95, 181))
    f = [scores(gold, (prob >= t).astype(int))["macro_f1"] for t in grid]
    return float(grid[int(np.argmax(f))])


def honest(gold, prob, halves=20, seed=0):
    """Threshold chosen on one half of the cases and scored on the other, both ways, over random halvings."""
    rng, out = np.random.default_rng(seed), []
    for _ in range(halves):
        p = rng.permutation(len(gold))
        a, b = p[:len(p) // 2], p[len(p) // 2:]
        for x, z in ((a, b), (b, a)):
            out.append(scores(gold[z], (prob[z] >= best_threshold(gold[x], prob[x])).astype(int))["macro_f1"])
    return float(np.mean(out))


def report(name, s):
    return f"{name}: macro-F1 {100 * s['macro_f1']:.2f} | P {100 * s['precision']:.2f} R {100 * s['recall']:.2f} | acc {100 * s['accuracy']:.2f}"


def write_submission(ids, pred, out_dir):
    """prediction.json as [{id, pred}] plus a zip with that one file, named after the folder."""
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"id": i, "pred": int(p)} for i, p in zip(ids, pred)]
    with open(f"{out_dir}/prediction.json", "w") as f:
        json.dump(rows, f)
    name = os.path.basename(os.path.normpath(out_dir))
    with zipfile.ZipFile(f"{out_dir}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{out_dir}/prediction.json", "prediction.json")
    return f"{out_dir}/{name}.zip"
