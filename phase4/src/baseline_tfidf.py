"""Word-count baseline: TF-IDF of word 1-2 grams + logistic regression, trained on train, scored on dev.

--last N keeps only the last N words of each case (the end of a judgment holds the court's reasoning).
Saves dev and test probabilities of class 1 to <out>/{dev,test}_prob.npy.
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import cjpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--last", type=int, default=0, help="keep only the last N words (0 = whole case)")
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--max_features", type=int, default=500000)
    args = ap.parse_args()
    t0 = time.time()
    tr, dv, te = cjpe.load("train"), cjpe.load("dev"), cjpe.load("test")
    cut = (lambda t: " ".join(t.split()[-args.last:])) if args.last else (lambda t: t)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
                          max_features=args.max_features, dtype=np.float32)
    Xtr = vec.fit_transform([cut(d["text"]) for d in tr])
    Xdv = vec.transform([cut(d["text"]) for d in dv])
    Xte = vec.transform([cut(d["text"]) for d in te])
    print(f"features {Xtr.shape} ({time.time() - t0:.0f}s)", flush=True)
    clf = LogisticRegression(C=args.C, solver="liblinear", max_iter=1000)
    clf.fit(Xtr, [d["y"] for d in tr])
    pdv, pte = clf.predict_proba(Xdv)[:, 1], clf.predict_proba(Xte)[:, 1]
    s = cjpe.scores([d["y"] for d in dv], (pdv >= 0.5).astype(int))
    print(cjpe.report(f"dev (last={args.last}, C={args.C})", s), f"({time.time() - t0:.0f}s)")
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/dev_prob.npy", pdv)
    np.save(f"{args.out}/test_prob.npy", pte)
    json.dump({"args": vars(args), "dev": s}, open(f"{args.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
