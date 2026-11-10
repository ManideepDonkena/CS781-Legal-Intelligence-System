"""Average the class-1 probabilities of chosen runs, pick the decision threshold on dev, write a submission.

  python combine.py RUN[:TAG] [RUN[:TAG] ...] [--out ../submissions/S01_x]

RUN is a folder under ../runs; TAG picks {dev,test}_prob_TAG.npy (no tag: {dev,test}_prob.npy, the TF-IDF files).
The threshold maximising dev macro-F1 is used; the 'honest' score picks it on one half of dev and scores the other.
"""
import argparse
import json
import os

import numpy as np

import cjpe
from cjpe import best_threshold, honest

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def standardise(d, t):
    """Scale dev and test scores by the dev mean and spread (per column when 2-D)."""
    mu, sd = d.mean(0), d.std(0) + 1e-6
    return (d - mu) / sd, (t - mu) / sd


def load_run(spec):
    run, _, tag = spec.partition(":")
    sfx = f"_{tag}" if tag else ""
    return np.load(f"{RUNS}/{run}/dev_prob{sfx}.npy"), np.load(f"{RUNS}/{run}/test_prob{sfx}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--avg", choices=["prob", "z"], default="prob",
                    help="average probabilities, or log-odds standardised on dev")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dv, te = cjpe.load("dev"), cjpe.load("test")
    gold = np.array([d["y"] for d in dv])
    probs = [load_run(s) for s in args.runs]
    for s, (pd, _) in zip(args.runs, probs):
        print(f"{s:40s} dev @0.5 {100 * cjpe.scores(gold, (pd >= .5).astype(int))['macro_f1']:.2f}"
              f" | tuned-on-other-half {100 * honest(gold, pd):.2f}")
    if args.avg == "z":  # each run's log-odds standardised on dev, so every run has an equal say
        zs = [standardise(logit(pd), logit(pt)) for pd, pt in probs]
        pdv, pte = sigmoid(np.mean([a for a, _ in zs], 0)), sigmoid(np.mean([b for _, b in zs], 0))
    else:
        pdv = np.mean([p for p, _ in probs], 0)
        pte = np.mean([p for _, p in probs], 0)
    th = best_threshold(gold, pdv)
    s = cjpe.scores(gold, (pdv >= th).astype(int))
    h = honest(gold, pdv)
    print(f"average of {len(probs)}: threshold {th:.3f} | " + cjpe.report("dev (threshold tuned on dev)", s)
          + f" | honest {100 * h:.2f}")
    if args.out:
        pred = (pte >= th).astype(int)
        z = cjpe.write_submission([d["id"] for d in te], pred, args.out)
        json.dump({"runs": args.runs, "avg": args.avg, "threshold": th, "dev": s, "dev_honest_macro_f1": h,
                   "test_share_pred_1": float(pred.mean())}, open(f"{args.out}/info.json", "w"), indent=1)
        print("wrote", z, f"(share predicted 1: {pred.mean():.3f})")


if __name__ == "__main__":
    main()
