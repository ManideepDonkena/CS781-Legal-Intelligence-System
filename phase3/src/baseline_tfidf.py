"""Reference model: TF-IDF of the sentence and its neighbours + position in the document -> logistic regression.

Cross-validation pools the 90 labelled documents (train + dev) into k document folds, repeated with several seeds.
"""
import argparse
import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import rr

N_BINS = 10


def position_features(docs):
    rows = []
    for d in docs:
        n = len(d["sents"])
        for i in range(n):
            f = np.zeros(N_BINS + 6)
            f[min(int(N_BINS * i / n), N_BINS - 1)] = 1
            f[N_BINS] = i < 3
            f[N_BINS + 1] = n - i <= 3
            f[N_BINS + 2] = d["domain"] == "CL"
            f[N_BINS + 3] = i / n
            f[N_BINS + 4] = np.log1p(len(d["sents"][i].split())) / 5
            f[N_BINS + 5] = np.log1p(n) / 8
            rows.append(f)
    return sp.csr_matrix(np.array(rows))


def features(vec, docs, neighbour_weight):
    x = vec.transform([s for d in docs for s in d["sents"]])
    starts = np.cumsum([0] + [len(d["sents"]) for d in docs])
    prev_idx, next_idx = [], []
    for d, s in zip(docs, starts):
        n = len(d["sents"])
        prev_idx += [s + max(i - 1, 0) for i in range(n)]
        next_idx += [s + min(i + 1, n - 1) for i in range(n)]
    return sp.hstack([x, neighbour_weight * x[prev_idx], neighbour_weight * x[next_idx],
                      position_features(docs)]).tocsr()


def fit_predict(train, test, args):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=args.max_features)
    vec.fit([s for d in train for s in d["sents"]])
    y = np.concatenate([d["y"] for d in train])
    clf = LogisticRegression(C=args.C, class_weight="balanced" if args.balanced else None, max_iter=3000)
    clf.fit(features(vec, train, args.nw), y)
    p = clf.predict_proba(features(vec, test, args.nw))
    full = np.zeros((len(p), len(rr.LABELS)))
    full[:, clf.classes_] = p
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--nw", type=float, default=0.5, help="weight of the neighbouring sentences' features")
    ap.add_argument("--balanced", type=int, default=1)
    ap.add_argument("--max_features", type=int, default=200000)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    docs = rr.load_labelled()
    gold = np.concatenate([d["y"] for d in docs])
    for seed in map(int, args.seeds.split(",")):
        fold = rr.doc_folds(docs, args.k, seed)
        pred = [None] * len(docs)
        for f in range(args.k):
            tr = [d for d, g in zip(docs, fold) if g != f]
            te_idx = [i for i, g in enumerate(fold) if g == f]
            p = fit_predict(tr, [docs[i] for i in te_idx], args)
            starts = np.cumsum([0] + [len(docs[i]["sents"]) for i in te_idx])
            for j, i in enumerate(te_idx):
                pred[i] = p[starts[j]:starts[j + 1]].argmax(1)
        s = rr.scores(gold, np.concatenate(pred))
        print(rr.report(f"seed {seed}", s), flush=True)
        for dom in rr.DOMAINS:
            m = [d["domain"] == dom for d in docs]
            sd = rr.scores(np.concatenate([d["y"] for d, k in zip(docs, m) if k]),
                           np.concatenate([p for p, k in zip(pred, m) if k]))
            print(f"   {dom}: all13 {100 * sd['all13']:.2f} present {100 * sd['present']:.2f}")

    if args.out_dir:
        test = rr.load_test()
        p = fit_predict(docs, test, args)
        starts = np.cumsum([0] + [len(d["sents"]) for d in test])
        preds = {d["id"]: p[starts[j]:starts[j + 1]].argmax(1) for j, d in enumerate(test)}
        print("wrote", rr.write_submission(test, preds, args.out_dir))


if __name__ == "__main__":
    main()
