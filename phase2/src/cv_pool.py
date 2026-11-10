"""Cross-validated score of a pooled ensemble: pool the held-out spans of several configs, build the
document dictionary once, and score both folds together (the same path make_submission.py takes)."""
import argparse

import lner
from make_submission import pooled_spans, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--taus", default="0.2")
    ap.add_argument("--votes", default="1")
    ap.add_argument("--labels", action="store_true")
    args = ap.parse_args()
    labels = lner.load_labels(args.data)
    gold, texts, merged = {}, {}, {}
    for train, ev in (("fold1", "fold2"), ("fold2", "fold1")):
        t, g = lner.load_fold(args.data, ev)
        m, _ = pooled_spans([f"{args.runs}/{c}_{train}/eval_spans.json" for c in args.configs])
        texts.update(t), gold.update(g), merged.update(m)
    for mv in (int(x) for x in args.votes.split(",")):
        for tau in (float(x) for x in args.taus.split(",")):
            pred = {d: resolve(texts[d], merged[d], tau, mv) for d in texts}
            sc = lner.strict_scores(gold, pred, labels)
            print(f"n={len(args.configs)} votes>={mv} tau={tau:.2f}  F1={100 * sc['macro_f1']:.2f} "
                  f"P={100 * sc['macro_p']:.2f} R={100 * sc['macro_r']:.2f}")
            if args.labels:
                for l, v in sc["per_label"].items():
                    print(f"   {l:10} P={100 * v['p']:5.1f} R={100 * v['r']:5.1f} F1={100 * v['f1']:5.1f}")


if __name__ == "__main__":
    main()
