"""Score saved train_rr.py outputs on the held-out documents (one run, or the average of several runs/seeds) and
optionally write a submission from the matching test outputs.

--tau shifts every label's score by -tau * log(label frequency), which trades accuracy on common labels for
recall on rare ones (macro-F1 weighs all labels equally).
"""
import argparse
import json
import os

import numpy as np

import rr
from train_rr import decode


def log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def load(run, seed, n_docs, full_dir=None):
    h = np.load(f"{run}/heldout_seed{seed}.npz")
    t = np.load(f"{full_dir}/test_full_seed{seed}.npz" if full_dir else f"{run}/test_seed{seed}.npz")
    crf = f"{run}/crf_seed{seed}.json"
    crf = {k: np.array(v) for k, v in json.load(open(crf)).items()} if os.path.exists(crf) else None
    return [log_softmax(h[f"em_{i}"]) for i in range(n_docs)], {k: log_softmax(t[k]) for k in t.files}, crf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of run dirs; dir@3+4+5 overrides --seeds for that dir")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--tau", default="0", help="comma list; each value is scored")
    ap.add_argument("--no_crf", action="store_true", help="argmax even when CRF transitions exist")
    ap.add_argument("--suppress", default="PrecedentOverruled")
    ap.add_argument("--absorbing", action="store_true",
                    help="Dissent only as a suffix of the document (as in all training documents)")
    ap.add_argument("--full_suffix", default="", help="test outputs from <run><suffix>/test_full_seed*.npz (all-data models)")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    docs, test = rr.load_labelled(), rr.load_test()
    gold = np.concatenate([d["y"] for d in docs])
    prior = np.bincount(gold, minlength=len(rr.LABELS)) / len(gold)
    log_prior = np.log(np.maximum(prior, 1e-6))
    suppress = [rr.LABEL_ID[l] for l in args.suppress.split(",") if l]

    def dec(d, em):
        em = em - tau * log_prior
        if not args.absorbing:
            return decode(em, crf, suppress)
        return rr.absorbing_dissent(em, decode(em, crf, suppress + [rr.LABEL_ID["Dissent"]]))

    runs = [(r.split("@")[0], r.split("@")[1].split("+") if "@" in r else args.seeds.split(","))
            for r in args.runs.split(",")]
    parts = [load(r, int(s), len(docs), r + args.full_suffix if args.full_suffix else None) for r, seeds in runs for s in seeds]
    held = [np.mean([p[0][i] for p in parts], 0) for i in range(len(docs))]
    tst = {d["id"]: np.mean([p[1][d["id"]] for p in parts], 0) for d in test}
    crfs = [p[2] for p in parts]
    crf = None if args.no_crf or any(c is None for c in crfs) else {k: np.mean([c[k] for c in crfs], 0) for k in crfs[0]}

    for tau in map(float, args.tau.split(",")):
        pred = np.concatenate([dec(d, em) for d, em in zip(docs, held)])
        print(rr.report(f"tau {tau}", rr.scores(gold, pred)))
    if args.out_dir:
        preds = {d["id"]: dec(d, tst[d["id"]]) for d in test}
        print("wrote", rr.write_submission(test, preds, args.out_dir), "with tau", tau)


if __name__ == "__main__":
    main()
