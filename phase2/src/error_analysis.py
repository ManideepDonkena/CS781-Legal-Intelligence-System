"""Break held-out errors into boundary / label / miss / spurious, per label, for one config."""
import argparse
import collections
import json

import lner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tau", type=float, default=0.34)
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()
    labels = lner.load_labels(args.data)
    kinds = collections.defaultdict(collections.Counter)
    confusion = collections.Counter()
    spurious = collections.Counter()
    shown = collections.defaultdict(list)
    for train, ev in (("fold1", "fold2"), ("fold2", "fold1")):
        texts, gold = lner.load_fold(args.data, ev)
        with open(f"{args.runs}/{args.config}_{train}/eval_spans.json") as f:
            spans = json.load(f)
        for did, text in texts.items():
            pred = lner.propagate(text, [tuple(x) for x in spans[did]], tau=args.tau)
            pset = {(s, e): l for s, e, l in pred}
            gset = {(s, e): l for s, e, l in gold[did]}
            for (s, e), lab in gset.items():
                if pset.get((s, e)) == lab:
                    kinds[lab]["correct"] += 1
                elif (s, e) in pset:
                    kinds[lab]["wrong_label"] += 1
                    confusion[(lab, pset[(s, e)])] += 1
                else:
                    over = [(a, b, l) for (a, b), l in pset.items() if a < e and s < b]
                    if not over:
                        kinds[lab]["missed"] += 1
                        if len(shown[(lab, "missed")]) < args.examples:
                            shown[(lab, "missed")].append(text[max(0, s - 40):s] + "[[" + text[s:e] + "]]" + text[e:e + 30])
                    elif any(l == lab for _, _, l in over):
                        kinds[lab]["boundary"] += 1
                        if len(shown[(lab, "boundary")]) < args.examples:
                            a, b, _ = over[0]
                            shown[(lab, "boundary")].append(f"gold[{text[s:e]}]  pred[{text[a:b]}]")
                    else:
                        kinds[lab]["boundary+label"] += 1
            for (s, e), lab in pset.items():
                if not any(a < e and s < b for (a, b) in gset):
                    spurious[lab] += 1
                    if len(shown[(lab, "spurious")]) < args.examples:
                        shown[(lab, "spurious")].append(text[max(0, s - 40):s] + "[[" + text[s:e] + "]]" + text[e:e + 30])
    cols = ["correct", "wrong_label", "boundary", "boundary+label", "missed"]
    print(f"{'label':10} {'gold':>5} " + " ".join(f"{c:>14}" for c in cols) + f" {'spurious':>9}")
    tot = collections.Counter()
    for lab in labels:
        n = sum(kinds[lab].values())
        tot.update(kinds[lab])
        print(f"{lab:10} {n:5d} " + " ".join(f"{100 * kinds[lab][c] / max(1, n):13.1f}%" for c in cols) + f" {spurious[lab]:9d}")
    n = sum(tot.values())
    print(f"{'ALL':10} {n:5d} " + " ".join(f"{100 * tot[c] / n:13.1f}%" for c in cols) + f" {sum(spurious.values()):9d}")
    print("\nmost common label confusions (gold -> predicted, exact boundaries):")
    for (g, p), c in confusion.most_common(14):
        print(f"  {g:10} -> {p:10} {c}")
    for (lab, kind), ex in sorted(shown.items()):
        print(f"\n[{lab} / {kind}]")
        for x in ex:
            print("   ", x)


if __name__ == "__main__":
    main()
