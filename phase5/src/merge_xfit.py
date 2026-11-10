"""Merge cross-fitted runs (ft_doc.py --dev_fold k, one run per dev fold) into one run folder.

Each dev case takes its probabilities from the run that did not train on it; test probabilities are the mean of
all fold runs.
  python merge_xfit.py OUT RUN_FOLD0 RUN_FOLD1 [...] --tag e3      (folders under ../runs)
"""
import argparse
import json
import os

import numpy as np

from combine import RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="one run per dev fold, in fold order")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    fold = np.load(f"{RUNS}/{args.runs[0]}/dev_fold.npy")
    dev = test = None
    for k, run in enumerate(args.runs):
        assert json.load(open(f"{RUNS}/{run}/log.json"))["args"]["dev_fold"] == k, run
        assert np.array_equal(np.load(f"{RUNS}/{run}/dev_fold.npy"), fold)
        d = np.load(f"{RUNS}/{run}/dev_prob_{args.tag}.npy")
        t = np.load(f"{RUNS}/{run}/test_prob_{args.tag}.npy")
        dev = np.zeros_like(d) if dev is None else dev
        dev[fold == k] = d[fold == k]
        test = t / len(args.runs) if test is None else test + t / len(args.runs)
    os.makedirs(f"{RUNS}/{args.out}", exist_ok=True)
    np.save(f"{RUNS}/{args.out}/dev_prob_{args.tag}.npy", dev)
    np.save(f"{RUNS}/{args.out}/test_prob_{args.tag}.npy", test)
    print("merged", args.runs, "->", args.out)


if __name__ == "__main__":
    main()
