"""Check (on dev, honestly) whether pulling per-statute thresholds toward one shared threshold helps.

  python threshold_shrink_check.py RUN[:TAG] [...] [--weights ...]
Prints the honest dev macro-F1 for a few shrink strengths m (0 = own thresholds, very large = one shared one).
"""
import argparse

import numpy as np

import lsi
from combine import load_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weights", default="")
    args = ap.parse_args()
    gold = lsi.label_matrix(lsi.load("dev"))
    w = np.array([float(x) for x in args.weights.split(",")]) if args.weights else np.ones(len(args.runs))
    w = w / w.sum()
    pdv = sum(wi * load_run(r)[0] for wi, r in zip(w, args.runs))
    for m in (0, 10, 30, 100, 1e9):
        print(f"shrink {m:g}: honest dev macro-F1 {100 * lsi.honest_threshold_score(gold, pdv, shrink=m):.2f}",
              flush=True)


if __name__ == "__main__":
    main()
