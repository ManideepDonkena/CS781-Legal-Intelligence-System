"""Nearest-neighbour statutes: each case takes a similarity-weighted vote of the statutes of its --k most similar
labelled cases (cosine similarity of TF-IDF word 1-2 gram vectors).

Cross-fitted like baseline_tfidf.py --xfit: a dev case's neighbours come from train + the other dev folds; a test
case's neighbours come from train + all of dev.  Saves <out>/{dev,test}_prob.npy.
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

import lsi


def vote(Q, K, Y, k, power, block=2000):
    """For each row of Q: weighted share of each label among its k most similar rows of K (rows L2-normalised)."""
    out = np.zeros((Q.shape[0], Y.shape[1]), dtype=np.float32)
    for s in range(0, Q.shape[0], block):
        sim = (Q[s:s + block] @ K.T).toarray()
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        w = np.take_along_axis(sim, top, 1) ** power
        out[s:s + block] = np.einsum("bk,bkl->bl", w, Y[top]) / np.maximum(w.sum(1, keepdims=True), 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--power", type=float, default=2.0, help="similarity is raised to this power before voting")
    ap.add_argument("--xfit", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    tr, dv, te = lsi.load("train"), lsi.load("dev"), lsi.load("test")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=400000,
                          dtype=np.float32)
    vec.fit([d["text"] for d in tr + dv])
    Xtr, Xdv, Xte = (vec.transform([d["text"] for d in s]) for s in (tr, dv, te))
    Ytr, gold = lsi.label_matrix(tr).astype(np.float32), lsi.label_matrix(dv)
    fold = lsi.dev_folds(len(dv), args.xfit)
    pdv = np.zeros(gold.shape, dtype=np.float32)
    for f in range(args.xfit):
        keep = fold != f
        K = sp.vstack([Xtr, Xdv[keep]]).tocsr()
        pdv[fold == f] = vote(Xdv[fold == f], K, np.concatenate([Ytr, gold[keep]]), args.k, args.power)
    pte = vote(Xte, sp.vstack([Xtr, Xdv]).tocsr(), np.concatenate([Ytr, gold]), args.k, args.power)
    honest = lsi.honest_threshold_score(gold, pdv)
    print(f"kNN k={args.k} power={args.power}: dev macro-F1 with thresholds tuned on the other half: "
          f"{100 * honest:.2f} ({time.time() - t0:.0f}s)")
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/dev_prob.npy", pdv)
    np.save(f"{args.out}/test_prob.npy", pte)
    json.dump({"args": vars(args), "dev_honest_macro_f1": honest}, open(f"{args.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
