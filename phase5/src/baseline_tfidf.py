"""Word-count baseline: TF-IDF of word 1-2 grams + one logistic regression per statute, trained on train.

Per-statute thresholds are tuned on dev; the 'honest' dev score tunes them on one half of dev and scores the
other half.  Saves dev and test probabilities to <out>/{dev,test}_prob.npy.

--xfit K (cross-fitting) also trains on dev: each of K dev folds is predicted by a model trained on train + the
other folds, and the test set by a model trained on train + all of dev, so the dev probabilities stay honest.
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import lsi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--max_features", type=int, default=400000)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--balanced", action="store_true", help="weight each statute's rare positive cases up")
    ap.add_argument("--ngram", type=int, default=2, help="longest word sequence (or with --char, longest letters)")
    ap.add_argument("--char", action="store_true", help="letter 3..ngram sequences within words instead of words")
    ap.add_argument("--xfit", type=int, default=0, help="also train on dev with this many dev folds")
    args = ap.parse_args()
    t0 = time.time()
    tr, dv, te = lsi.load("train"), lsi.load("dev"), lsi.load("test")
    vec = TfidfVectorizer(ngram_range=(3 if args.char else 1, args.ngram), analyzer="char_wb" if args.char else "word",
                          min_df=3, max_df=0.9, sublinear_tf=True, max_features=args.max_features, dtype=np.float32)
    if args.xfit:
        vec.fit([d["text"] for d in tr + dv])
        Xtr = vec.transform([d["text"] for d in tr])
    else:
        Xtr = vec.fit_transform([d["text"] for d in tr])
    Xdv, Xte = vec.transform([d["text"] for d in dv]), vec.transform([d["text"] for d in te])
    for X in (Xtr, Xdv, Xte):
        X.sort_indices()  # parallel workers get read-only copies and cannot sort them themselves
    print(f"features {Xtr.shape} ({time.time() - t0:.0f}s)", flush=True)

    def model():
        return OneVsRestClassifier(LogisticRegression(C=args.C, solver="liblinear", max_iter=1000,
                                                      class_weight="balanced" if args.balanced else None),
                                   n_jobs=args.jobs)
    gold = lsi.label_matrix(dv)
    Ytr = lsi.label_matrix(tr)
    if args.xfit:
        fold = lsi.dev_folds(len(dv), args.xfit)
        pdv = np.zeros(gold.shape, dtype=np.float32)
        for k in range(args.xfit):
            fit = fold != k
            Xk = sp.vstack([Xtr, Xdv[fit]]).tocsr()
            Xk.sort_indices()
            clf = model().fit(Xk, np.concatenate([Ytr, gold[fit]]))
            pdv[fold == k] = clf.predict_proba(Xdv[fold == k])
            print(f"dev fold {k} done ({time.time() - t0:.0f}s)", flush=True)
        X_all = sp.vstack([Xtr, Xdv]).tocsr()
        X_all.sort_indices()
        pte = model().fit(X_all, np.concatenate([Ytr, gold])).predict_proba(Xte)
    else:
        clf = model().fit(Xtr, Ytr)
        pdv, pte = clf.predict_proba(Xdv), clf.predict_proba(Xte)
    s05 = lsi.scores(gold, lsi.decide(pdv, np.full(lsi.N_LABELS, 0.5)))
    th = lsi.best_thresholds(gold, pdv)
    s_tuned = lsi.scores(gold, lsi.decide(pdv, th))
    honest = lsi.honest_threshold_score(gold, pdv)
    print(lsi.report("dev, threshold 0.5", s05))
    print(lsi.report("dev, thresholds tuned on dev (optimistic)", s_tuned))
    print(f"dev macro-F1 with thresholds tuned on the other half: {100 * honest:.2f} ({time.time() - t0:.0f}s)")
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/dev_prob.npy", pdv.astype(np.float32))
    np.save(f"{args.out}/test_prob.npy", pte.astype(np.float32))
    json.dump({"args": vars(args), "dev_0.5": {k: v for k, v in s05.items() if k != "per_label"},
               "dev_tuned": {k: v for k, v in s_tuned.items() if k != "per_label"}, "dev_honest_macro_f1": honest},
              open(f"{args.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
