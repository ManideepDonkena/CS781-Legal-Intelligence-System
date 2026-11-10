"""Tabulate cross-validation results from runs/*/log.json (mean over the held-out folds)."""
import argparse
import collections
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", action="store_true", help="also print per-label F1 for each config")
    ap.add_argument("--key", default="", help="variant used for the per-label table (default: best)")
    args = ap.parse_args()

    groups = collections.defaultdict(dict)
    for path in sorted(glob.glob(f"{args.runs}/*/log.json")):
        name = os.path.basename(os.path.dirname(path))
        cfg, fold = name.rsplit("_", 1)
        with open(path) as f:
            log = json.load(f)
        done = [e for e in log["epochs"] if "macro_f1" in e]
        if done:
            groups[cfg][fold] = (log, done)

    rows = []
    for cfg, folds in groups.items():
        last = collections.defaultdict(list)
        for fold, (log, done) in folds.items():
            for k, v in done[-1]["macro_f1"].items():
                last[k].append(v)
        mean = {k: sum(v) / len(v) for k, v in last.items()}
        best = max(mean, key=mean.get)
        curve = [sum(folds[f][1][i]["macro_f1"]["raw"] for f in folds) / len(folds)
                 for i in range(min(len(d) for _, d in folds.values()))]
        rows.append((mean[best], cfg, mean, best, len(folds), done[-1]["epoch"], log["args"]["epochs"], curve))

    keys = sorted({k for r in rows for k in r[2]}, key=lambda k: (k != "raw", k))
    print(f"{'config':34} folds ep  " + "  ".join(f"{k:>8}" for k in keys) + "   raw by epoch")
    for _, cfg, mean, best, nf, ep, total, curve in sorted(rows, reverse=True):
        cells = "  ".join(f"{100 * mean[k]:8.2f}" if k in mean else " " * 8 for k in keys)
        print(f"{cfg:34} {nf:5d} {ep}/{total}  {cells}   " + " ".join(f"{100 * c:.0f}" for c in curve))

    if args.labels:
        for _, cfg, mean, best, *_ in sorted(rows, reverse=True):
            key = args.key or best
            per = collections.defaultdict(list)
            for fold, (log, done) in groups[cfg].items():
                for lab, v in done[-1]["scores"][key]["per_label"].items():
                    per[lab].append((v["p"], v["r"], v["f1"]))
            print(f"\n{cfg} [{key}]  P / R / F1 (mean of folds)")
            for lab, v in per.items():
                p, r, f = (100 * sum(x[i] for x in v) / len(v) for i in range(3))
                print(f"  {lab:10} {p:6.1f} {r:6.1f} {f:6.1f}")


if __name__ == "__main__":
    main()
