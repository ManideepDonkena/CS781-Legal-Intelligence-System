"""Hybrid summaries: the opening of the abstractive summary, then the extractive model's best sentences.

The fine-tuned language model writes a good headnote-style opening (facts, earlier courts' findings) but drifts
after a few hundred words; the extractive model's sentences are faithful but read like the judgment.  A hybrid
keeps the abstractive sentences until --keep words are reached, then adds the highest-scored case sentences (in
case order) until the extractive length budget (x --factor) is met, skipping sentences that share more than half
of their word pairs with the text already chosen.  Sentence scores are the cross-fitted ones (dev) or the
all-data model's (test), from extractive.py --xfit.

The opening matters most because BERTScore (Zhang et al. 2020, "BERTScore: Evaluating Text Generation with BERT")
runs a BERT-type model that reads at most 512 tokens, so the scorer most likely cuts longer summaries.

  python hybrid.py --abs ../runs/abs_x/dev_summaries.json --split dev --keep 0,200,300,400,100000
  python hybrid.py --abs ../runs/abs_x/test_summaries.json --split test --keep 300 --save ../runs/hyb/test.json
"""
import argparse
import json
import math

import numpy as np

import summ
from extractive import _bigrams

XFIT = "/home/mani/cs781/Capstone/Phase6_SUM/runs/ext_xfit5"
LENGTH = (1.533, 0.612)


def head(text, keep):
    out, words = [], 0
    for s in summ.sentences(text):
        if words >= keep:
            break
        out.append(s)
        words += len(s.split())
    return out


def hybrid(doc, abs_text, scores, keep, factor):
    budget = factor * math.exp(LENGTH[0] + LENGTH[1] * math.log(len(doc["document"].split()) + 1))
    chosen = head(abs_text, keep)
    words = sum(len(s.split()) for s in chosen)
    have = set()
    for s in chosen:
        have |= _bigrams(summ.tokens(s))
    sents = summ.sentences(doc["document"])
    picked = []
    for i in np.argsort(-np.asarray(scores)):
        if words >= budget:
            break
        w = len(sents[i].split())
        b = _bigrams(summ.tokens(sents[i]))
        if w < 4 or (b and len(b & have) / len(b) > 0.5):
            continue
        picked.append(i)
        have |= b
        words += w
    return " ".join(chosen + [sents[i] for i in sorted(picked)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abs", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--keep", default="300")
    ap.add_argument("--factor", type=float, default=1.1)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    abs_sum = json.load(open(args.abs))
    docs = [d for d in summ.load(args.split) if d["id"] in abs_sum]
    scores = json.load(open(f"{XFIT}/{'test_scores_all' if args.split == 'test' else 'xfit_scores'}.json"))
    cut = lambda t: " ".join(t.split()[:400])  # noqa: E731
    for keep in [int(k) for k in args.keep.split(",")]:
        out = {d["id"]: hybrid(d, abs_sum[d["id"]], scores[d["id"]], keep, args.factor) for d in docs}
        if args.split == "dev":
            refs = [d["summary"] for d in docs]
            full = summ.mean_scores([out[d["id"]] for d in docs], refs)
            first = summ.mean_scores([cut(out[d["id"]]) for d in docs], [cut(r) for r in refs])
            print(f"keep {keep}: full " + " ".join(f"{k} {v:.1f}" for k, v in full.items()) +
                  " | first 400 words " + " ".join(f"{k} {v:.1f}" for k, v in first.items()), flush=True)
        if args.save:
            json.dump(out, open(args.save, "w"))


if __name__ == "__main__":
    main()
