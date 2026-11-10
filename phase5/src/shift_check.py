"""Does the test set look more like dev or like train?  (Uses test TEXT only, never test answers.)

A logistic regression on TF-IDF words (our own code, ../../shared/scratch_text.py) learns to tell training cases
(0) from dev cases (1) on a random 80% of each, and is checked on the other 20%. It is then applied to the test
cases.  If test cases get "dev-like" scores as often as held-out dev cases do, the test set resembles dev, and
dev cases deserve more weight in training.  Also compares statute frequencies: train vs dev (gold) and the
current submission's test predictions.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "shared"))
import lsi  # noqa: E402
import scratch_text as st  # noqa: E402


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def main():
    tr, dv, te = lsi.load("train"), lsi.load("dev"), lsi.load("test")
    rng = np.random.default_rng(0)
    tf = st.Tfidf(ngram=1, max_features=100000).fit([d["text"] for d in tr + dv])
    Xtr, Xdv, Xte = (tf.transform([d["text"] for d in s]) for s in (tr, dv, te))
    V = len(tf.vocab)
    in_tr, in_dv = rng.random(len(tr)) < 0.8, rng.random(len(dv)) < 0.8
    fit = st.stack(st.rows(Xtr, np.flatnonzero(in_tr)), st.rows(Xdv, np.flatnonzero(in_dv)))
    y = np.concatenate([np.zeros(in_tr.sum()), np.ones(in_dv.sum())])
    weights = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (1 - y).sum()))  # balance the two groups
    W, b = st.train_logreg(fit, V, y[:, None], C=1.0, steps=300, weights=weights)
    p_tr = st.predict_logreg(st.rows(Xtr, np.flatnonzero(~in_tr)), V, W, b)[:, 0]
    p_dv = st.predict_logreg(st.rows(Xdv, np.flatnonzero(~in_dv)), V, W, b)[:, 0]
    p_te = st.predict_logreg(Xte, V, W, b)[:, 0]
    a = auc(np.concatenate([np.zeros(len(p_tr)), np.ones(len(p_dv))]), np.concatenate([p_tr, p_dv]))
    print(f"telling train from dev on held-out cases: AUC {a:.3f} (0.5 = impossible, 1.0 = always)")
    print(f"mean 'dev-like' score: held-out train {p_tr.mean():.3f} | held-out dev {p_dv.mean():.3f} | "
          f"test {p_te.mean():.3f}")
    print(f"share scored above 0.5: held-out train {np.mean(p_tr > .5):.3f} | held-out dev {np.mean(p_dv > .5):.3f} "
          f"| test {np.mean(p_te > .5):.3f}")
    f_tr, f_dv = lsi.label_matrix(tr).mean(0), lsi.label_matrix(dv).mean(0)
    sub = json.load(open(os.path.join(HERE, "..", "submissions", "P5_S02_scratch_xf_final", "prediction.json")))
    f_te = np.zeros(lsi.N_LABELS)
    for r in sub:
        f_te[r["pred"]] += 1
    f_te /= len(sub)
    corr = lambda u, v: float(np.corrcoef(np.log(u + 1e-4), np.log(v + 1e-4))[0, 1])  # noqa: E731
    print(f"statute frequency correlation (log scale): train vs dev {corr(f_tr, f_dv):.3f} | "
          f"test predictions vs dev {corr(f_te, f_dv):.3f} | test predictions vs train {corr(f_te, f_tr):.3f}")


if __name__ == "__main__":
    main()
