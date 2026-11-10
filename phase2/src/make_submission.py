"""Build a validated Phase 2 submission from one or more span files written by train_ner.py.

With several span files (seeds / models) the mention-level predictions are pooled before the
document dictionary is built, so a string must be supported by the ensemble as a whole.
"""
import argparse
import json
import os
import zipfile

import lner


def pooled_spans(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    docs = {}
    for did in runs[0]:
        merged = {}
        for r in runs:
            for s, e, lab, sc in r[did]:
                conf, votes = merged.get((s, e, lab), (0.0, 0))
                merged[(s, e, lab)] = (conf + sc / len(runs), votes + 1)
        docs[did] = merged
    return docs, len(runs)


def resolve(text, merged, tau, min_votes):
    """Keep mentions predicted by at least `min_votes` runs, one label per exact span."""
    best = {}
    for (s, e, lab), (sc, votes) in merged.items():
        if votes < min_votes:
            continue
        if (s, e) not in best or sc > best[(s, e)][1]:
            best[(s, e)] = (lab, sc)
    spans = [(s, e, lab, sc) for (s, e), (lab, sc) in best.items()]
    spans.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    taken = bytearray(len(text))
    flat = []
    for s, e, lab, sc in spans:
        if not any(taken[s:e]):
            flat.append((s, e, lab, sc))
            taken[s:e] = b"\x01" * (e - s)
    return lner.propagate(text, lner.snap_to_words(text, flat), tau=tau)


def validate(pred, texts, labels):
    assert set(pred) == set(texts), "document ids differ from the test set"
    allowed = set(labels)
    for did, ents in pred.items():
        seen = set()
        for s, e, lab in ents:
            assert isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(texts[did]), (did, s, e)
            assert lab in allowed, lab
            assert (s, e, lab) not in seen, "duplicate span"
            seen.add((s, e, lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--spans", nargs="+", required=True)
    ap.add_argument("--fold", default="fold3")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--min_votes", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    labels = lner.load_labels(args.data)
    texts, gold = lner.load_fold(args.data, args.fold)
    merged, n = pooled_spans(args.spans)
    pred = {d: resolve(texts[d], merged[d], args.tau, args.min_votes) for d in texts}
    validate(pred, texts, labels)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {d: [[s, e, lab] for s, e, lab in ents] for d, ents in pred.items()}
    with open(f"{args.out_dir}/prediction.json", "w", encoding="utf-8") as f:
        json.dump(out, f)
    with zipfile.ZipFile(f"{args.out_dir}/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{args.out_dir}/prediction.json", "prediction.json")
    n_ent = sum(len(v) for v in pred.values())
    per = {l: sum(1 for v in pred.values() for x in v if x[2] == l) for l in labels}
    print(f"docs={len(pred)} entities={n_ent} per_doc={n_ent / len(pred):.1f}")
    print("per label:", per)
    if gold is not None:
        sc = lner.strict_scores(gold, pred, labels)
        print(f"strict macro-F1={sc['macro_f1']:.4f} P={sc['macro_p']:.4f} R={sc['macro_r']:.4f}")
    print(f"wrote {args.out_dir}/submission.zip")


if __name__ == "__main__":
    main()
