"""Combine chunk scores (chunk_predict.py) and other runs' probabilities with a small logistic regression fitted
on dev, scored by 10-fold cross-validation within dev (repeated over several shufflings).

Features per case: log-odds of chunk 0; mean, max and min log-odds of the earlier chunks (0 when there are none)
and their count; and the log-odds of each extra run given with --extra RUN[:TAG].
  python stack.py CHUNKDIR [CHUNKDIR ...] [--extra tfidf_all] [--out ../submissions/S0x_name]
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import cjpe
from combine import RUNS, load_run


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def chunk_features(c, earlier):
    rest = c[:, 1:1 + earlier] if earlier else c[:, :0]
    n = np.sum(~np.isnan(rest), 1)
    with np.errstate(all="ignore"):
        f = [c[:, 0]]
        if earlier:
            f += [np.where(n > 0, np.nanmean(rest, 1), 0), np.where(n > 0, np.nanmax(rest, 1), 0),
                  np.where(n > 0, np.nanmin(rest, 1), 0), n / max(earlier, 1)]
    return np.stack(f, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunkdirs", nargs="+")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--earlier", type=int, default=7, help="earlier chunks used (0 = last chunk only)")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dv, te = cjpe.load("dev"), cjpe.load("test")
    y = np.array([d["y"] for d in dv])
    Fd, Ft = [], []
    for cd in args.chunkdirs:
        d = f"{RUNS}/{cd}"
        Fd.append(chunk_features(np.load(f"{d}/dev_chunk_logit.npy"), args.earlier))
        Ft.append(chunk_features(np.load(f"{d}/test_chunk_logit.npy"), args.earlier))
    for s in args.extra:
        pd, pt = load_run(s)
        Fd.append(logit(pd)[:, None])
        Ft.append(logit(pt)[:, None])
    Xd, Xt = np.concatenate(Fd, 1), np.concatenate(Ft, 1)
    f1s = []
    for r in range(args.repeats):
        pred = np.zeros(len(y), dtype=int)
        for a, b in StratifiedKFold(10, shuffle=True, random_state=r).split(Xd, y):
            clf = LogisticRegression(C=args.C, max_iter=1000).fit(Xd[a], y[a])
            pred[b] = clf.predict(Xd[b])
        f1s.append(cjpe.scores(y, pred)["macro_f1"])
    print(f"{' + '.join(args.chunkdirs + args.extra)} | earlier={args.earlier} | dev 10-fold macro-F1 "
          f"{100 * np.mean(f1s):.2f} (± {100 * np.std(f1s):.2f} over shufflings)")
    clf = LogisticRegression(C=args.C, max_iter=1000).fit(Xd, y)
    print("weights", np.round(clf.coef_[0], 3), "bias", round(float(clf.intercept_[0]), 3))
    if args.out:
        pred = clf.predict(Xt)
        z = cjpe.write_submission([d["id"] for d in te], pred, args.out)
        json.dump({"chunkdirs": args.chunkdirs, "extra": args.extra, "earlier": args.earlier, "C": args.C,
                   "dev_10fold_macro_f1": float(np.mean(f1s)), "weights": clf.coef_[0].tolist(),
                   "test_share_pred_1": float(pred.mean())}, open(f"{args.out}/info.json", "w"), indent=1)
        print("wrote", z, f"(share predicted 1: {pred.mean():.3f})")


if __name__ == "__main__":
    main()
