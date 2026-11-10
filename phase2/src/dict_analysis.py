"""How much of the gold dictionary do the held-out predictions reach?  Per label: share of gold
mention mass whose string was tagged at least once (any label / right label) by the pooled runs,
and per-label F1 as a function of the propagation threshold."""
import argparse
import collections

import lner
from make_submission import pooled_spans, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--votes", type=int, default=1)
    args = ap.parse_args()
    labels = lner.load_labels(args.data)
    gold, texts, merged = {}, {}, {}
    for train, ev in (("fold1", "fold2"), ("fold2", "fold1")):
        t, g = lner.load_fold(args.data, ev)
        m, _ = pooled_spans([f"{args.runs}/{c}_{train}/eval_spans.json" for c in args.configs])
        texts.update(t), gold.update(g), merged.update(m)
    mass = collections.Counter()
    hit_any, hit_lab, hit_sub = collections.Counter(), collections.Counter(), collections.Counter()
    for d, text in texts.items():
        tagged = collections.defaultdict(set)
        for (s, e, lab) in merged[d]:
            tagged[text[s:e]].add(lab)
        for s, e, lab in gold[d]:
            st = text[s:e]
            mass[lab] += 1
            hit_any[lab] += st in tagged
            hit_lab[lab] += lab in tagged.get(st, ())
            hit_sub[lab] += any((st in k or k in st) and len(k) > 2 for k in tagged)
    print("label        mentions  string tagged once: any-label  right-label  overlapping-string")
    for l in labels:
        print(f"  {l:10} {mass[l]:6d}   {100 * hit_any[l] / mass[l]:6.1f}  {100 * hit_lab[l] / mass[l]:6.1f}  {100 * hit_sub[l] / mass[l]:6.1f}")
    taus = [0.0, 0.05, 0.1, 0.2, 0.34, 0.5]
    print("\nper-label F1 by tau (votes>=%d):  " % args.votes + "  ".join(f"{t:5.2f}" for t in taus))
    table = {}
    for tau in taus:
        pred = {d: resolve(texts[d], merged[d], tau, args.votes) for d in texts}
        table[tau] = lner.strict_scores(gold, pred, labels)
    for l in labels:
        print(f"  {l:10} " + "  ".join(f"{100 * table[t]['per_label'][l]['f1']:5.1f}" for t in taus))
    print(f"  {'MACRO':10} " + "  ".join(f"{100 * table[t]['macro_f1']:5.1f}" for t in taus))


if __name__ == "__main__":
    main()
