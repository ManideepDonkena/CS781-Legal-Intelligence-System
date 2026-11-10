"""Average the statute probabilities of chosen runs, tune one threshold per statute on dev, write a submission.

  python combine.py RUN[:TAG] [RUN[:TAG] ...] [--out ../submissions/S01_x]

RUN is a folder under ../runs; TAG picks {dev,test}_prob_TAG.npy (no tag: {dev,test}_prob.npy); RUN@max or
RUN@mean combines the per-chunk probabilities written by chunk_predict.py.
The 'honest' score tunes thresholds on one half of dev and scores the other half.
"""
import argparse
import json
import os

import numpy as np

import lsi

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def standardise(d, t):
    """Scale dev and test scores by the dev mean and spread of every column (statute)."""
    mu, sd = d.mean(0), d.std(0) + 1e-6
    return (d - mu) / sd, (t - mu) / sd


def load_run(spec):
    """RUN[:TAG] reads saved probabilities; RUN@max / RUN@mean reduces chunk_predict.py's per-chunk ones."""
    if "@" in spec:
        run, how = spec.split("@")
        f = {"max": np.nanmax, "mean": np.nanmean}[how]
        return tuple(f(np.load(f"{RUNS}/{run}/{s}_chunk_prob.npy"), axis=1) for s in ("dev", "test"))
    run, _, tag = spec.partition(":")
    sfx = f"_{tag}" if tag else ""
    return np.load(f"{RUNS}/{run}/dev_prob{sfx}.npy"), np.load(f"{RUNS}/{run}/test_prob{sfx}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weights", default="", help="comma list, one weight per run (default equal)")
    ap.add_argument("--avg", choices=["prob", "z"], default="prob",
                    help="average probabilities, or log-odds standardised per statute on dev")
    ap.add_argument("--mix_split", type=int, default=0,
                    help="per-statute mix: the first N runs are family A, the rest family B; each family is averaged "
                         "and every statute chooses its own share of B and its threshold on dev")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dv, te = lsi.load("dev"), lsi.load("test")
    gold = lsi.label_matrix(dv)
    probs = [load_run(s) for s in args.runs]
    for s, (pd, _) in zip(args.runs, probs):
        print(f"{s:40s} dev @0.5 {100 * lsi.scores(gold, lsi.decide(pd, np.full(100, .5)))['macro_f1']:.2f}"
              f" | tuned-on-other-half {100 * lsi.honest_threshold_score(gold, pd):.2f}")
    if args.mix_split:
        fa = [np.mean([p[k] for p in probs[:args.mix_split]], 0) for k in (0, 1)]
        fb = [np.mean([p[k] for p in probs[args.mix_split:]], 0) for k in (0, 1)]
        share, th = lsi.fit_mix(gold, fa[0], fb[0])
        pdv, pte = share * fb[0] + (1 - share) * fa[0], share * fb[1] + (1 - share) * fa[1]
        s = lsi.scores(gold, lsi.decide(pdv, th))
        h = lsi.honest_mix_score(gold, fa[0], fb[0])
        print(f"per-statute mix of {args.mix_split} + {len(probs) - args.mix_split} runs (mean share of B "
              f"{share.mean():.2f}): " + lsi.report("dev (tuned on dev)", s) + f" | honest {100 * h:.2f}")
        if args.out:
            pred = lsi.decide(pte, th)
            z = lsi.write_submission([d["id"] for d in te], pred, args.out)
            json.dump({"runs": args.runs, "mix_split": args.mix_split, "share_B": share.tolist(),
                       "thresholds": th.tolist(), "dev": {k: v for k, v in s.items() if k != "per_label"},
                       "dev_honest_macro_f1": h, "test_labels_per_case": float(pred.sum(1).mean())},
                      open(f"{args.out}/info.json", "w"), indent=1)
            print("wrote", z, f"(statutes per test case: {pred.sum(1).mean():.2f})")
        return
    w = np.array([float(x) for x in args.weights.split(",")]) if args.weights else np.ones(len(probs))
    w = w / w.sum()
    if args.avg == "z":  # each run's log-odds standardised per statute on dev, so every run has an equal say
        zs = [standardise(logit(pd), logit(pt)) for pd, pt in probs]
        pdv = sigmoid(sum(wi * a for wi, (a, _) in zip(w, zs)))
        pte = sigmoid(sum(wi * b for wi, (_, b) in zip(w, zs)))
    else:
        pdv = sum(wi * p for wi, (p, _) in zip(w, probs))
        pte = sum(wi * p for wi, (_, p) in zip(w, probs))
    th = lsi.best_thresholds(gold, pdv)
    s = lsi.scores(gold, lsi.decide(pdv, th))
    h = lsi.honest_threshold_score(gold, pdv)
    print(f"average of {len(probs)}: " + lsi.report("dev (thresholds tuned on dev)", s) + f" | honest {100 * h:.2f}")
    if args.out:
        pred = lsi.decide(pte, th)
        z = lsi.write_submission([d["id"] for d in te], pred, args.out)
        json.dump({"runs": args.runs, "avg": args.avg, "weights": w.tolist(), "thresholds": th.tolist(),
                   "dev": {k: v for k, v in s.items() if k != "per_label"}, "dev_honest_macro_f1": h,
                   "test_labels_per_case": float(pred.sum(1).mean())}, open(f"{args.out}/info.json", "w"), indent=1)
        print("wrote", z, f"(statutes per test case: {pred.sum(1).mean():.2f})")


if __name__ == "__main__":
    main()
