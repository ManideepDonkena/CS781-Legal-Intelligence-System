"""Second stage for Phase 5: one logistic regression per statute over the log-odds of ALL 100 statutes from every
given run (so it can weigh the runs and use statutes that tend to occur together), fitted on dev.

Scored honestly: 5-fold cross-validation within dev gives out-of-fold probabilities, and their thresholds are tuned
on one half of dev and scored on the other (lsi.honest_threshold_score).  For the test set the regressions are
refitted on all of dev and the thresholds tuned on the out-of-fold dev probabilities.
  python stack.py RUN[:TAG|@max] [...] [--C 0.1] [--out ../submissions/S0x_name]
"""
import argparse
import json

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import lsi
from combine import load_run


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def fit_predict(Xa, ya, Xb, C):
    if ya.min() == ya.max():
        return np.full(len(Xb), float(ya[0]))
    return LogisticRegression(C=C, max_iter=2000).fit(Xa, ya).predict_proba(Xb)[:, 1]


def stage(Xa, Ya, Xb, C, jobs):
    cols = Parallel(n_jobs=jobs)(delayed(fit_predict)(Xa, Ya[:, j], Xb, C) for j in range(Ya.shape[1]))
    return np.stack(cols, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dv, te = lsi.load("dev"), lsi.load("test")
    gold = lsi.label_matrix(dv)
    probs = [load_run(s) for s in args.runs]
    Xd = np.concatenate([logit(p) for p, _ in probs], 1)
    Xt = np.concatenate([logit(p) for _, p in probs], 1)
    mu, sd = Xd.mean(0), Xd.std(0) + 1e-6
    Xd, Xt = (Xd - mu) / sd, (Xt - mu) / sd
    oof = np.zeros(gold.shape, dtype=np.float32)
    for a, b in KFold(args.folds, shuffle=True, random_state=0).split(Xd):
        oof[b] = stage(Xd[a], gold[a], Xd[b], args.C, args.jobs)
    honest = lsi.honest_threshold_score(gold, oof)
    th = lsi.best_thresholds(gold, oof)
    s = lsi.scores(gold, lsi.decide(oof, th))
    print(f"stack of {' + '.join(args.runs)} (C={args.C}): out-of-fold " + lsi.report("dev", s)
          + f" | honest {100 * honest:.2f}")
    if args.out:
        pte = stage(Xd, gold, Xt, args.C, args.jobs)
        pred = lsi.decide(pte, th)
        z = lsi.write_submission([d["id"] for d in te], pred, args.out)
        json.dump({"runs": args.runs, "C": args.C, "thresholds": th.tolist(),
                   "dev_oof": {k: v for k, v in s.items() if k != "per_label"}, "dev_honest_macro_f1": honest,
                   "test_labels_per_case": float(pred.sum(1).mean())}, open(f"{args.out}/info.json", "w"), indent=1)
        print("wrote", z, f"(statutes per test case: {pred.sum(1).mean():.2f})")


if __name__ == "__main__":
    main()
