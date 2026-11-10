"""Supervised extractive summaries: score every sentence of a case, keep the best ones up to a length budget.

Target of a sentence = share of its word pairs (bigrams) that also occur in the reference summary, a soft label in
[0, 1]; most headnote sentences reuse the judgment's wording, so this marks the sentences the summary drew from.
Features: TF-IDF of the sentence's words and word pairs (our own Tfidf), plus one-hot bins of its relative position
in the case, its index from the start and from the end, its length, and its cosine similarity to the whole case.
Model: our own logistic regression (soft labels are fine for the cross-entropy gradient).

Length budget: summary words = exp(a + b * log(case words)), fitted by least squares on the training cases, times a
factor chosen on dev.  Sentences are taken by score until the budget is reached and printed in case order.
Sentences under 4 words (headers, "Appeal allowed.") are never taken.

References: sentence labels derived from overlap with the reference summary, as in Nallapati, Zhai and Zhou 2017,
"SummaRuNNer"; extractive methods on Indian Supreme Court headnotes, Shukla et al. 2022, "Legal Case Document
Summarization: Extractive and Abstractive Methods and their Evaluation" (the IN-Abs dataset this data resembles).

  python extractive.py --out ../runs/ext_v1            # train on train, report dev, write dev/test summaries
  python extractive.py --out ../runs/ext_v1_all --all  # train on train + dev for the final test summaries
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/home/mani/cs781/Capstone/shared")
import summ  # noqa: E402
from scratch_text import Tfidf, predict_logreg, stack, train_logreg  # noqa: E402


def _bigrams(toks):
    return set(zip(toks, toks[1:]))


def _prep(doc):
    """Sentences, their lengths and (if there is a summary) their soft labels."""
    sents = summ.sentences(doc["document"])
    lab = []
    if "summary" in doc:
        ref = _bigrams(summ.tokens(doc["summary"]))
        for s in sents:
            b = _bigrams(summ.tokens(s))
            lab.append(len(b & ref) / len(b) if b else 0.0)
    return sents, lab


def prep_all(docs, workers=32):
    with Pool(workers) as pool:
        return pool.map(_prep, docs, chunksize=8)


N_POS, N_IDX, N_LEN, N_SIM = 20, 8, 10, 10


def dense_feats(sents, sent_csr_rows, start):
    """Column ids (offsets inside the extra block) of the one-hot extra features of every sentence of one case."""
    n = len(sents)
    indptr, cols, vals = sent_csr_rows
    # the case vector = sum of its sentence vectors; cosine of each sentence with it
    case = {}
    for i in range(n):
        for c, v in zip(cols[indptr[start + i]:indptr[start + i + 1]], vals[indptr[start + i]:indptr[start + i + 1]]):
            case[c] = case.get(c, 0.0) + v
    norm = math.sqrt(sum(v * v for v in case.values())) or 1.0
    out = []
    for i, s in enumerate(sents):
        a, b = indptr[start + i], indptr[start + i + 1]
        sim = sum(v * case.get(c, 0.0) for c, v in zip(cols[a:b], vals[a:b])) / norm
        w = len(s.split())
        f = [min(N_POS - 1, int(N_POS * i / n)),
             N_POS + min(i, N_IDX - 1),
             N_POS + N_IDX + min(n - 1 - i, N_IDX - 1),
             N_POS + 2 * N_IDX + min(N_LEN - 1, int(math.log2(w + 1))),
             N_POS + 2 * N_IDX + N_LEN + min(N_SIM - 1, int(N_SIM * max(0.0, sim)))]
        out.append(f)
    return out


N_EXTRA = N_POS + 2 * N_IDX + N_LEN + N_SIM


def features(tfidf, prepped):
    """CSR matrix (sentence rows) = TF-IDF block + one-hot extra block, and the case each row belongs to."""
    flat = [s for sents, _ in prepped for s in sents]
    owner = np.repeat(np.arange(len(prepped)), [len(s) for s, _ in prepped])
    X = tfidf.transform(flat)
    nv = len(tfidf.vocab)
    indptr, cols, vals = X
    ptr, cc, vv, pos = [0], [], [], 0
    for sents, _ in prepped:
        extra = dense_feats(sents, X, pos)
        for i, f in enumerate(extra):
            a, b = indptr[pos + i], indptr[pos + i + 1]
            cc.append(cols[a:b])
            cc.append(np.array([nv + x for x in f], np.int64))
            vv.append(vals[a:b])
            vv.append(np.full(len(f), 0.5, np.float32))
            ptr.append(ptr[-1] + (b - a) + len(f))
        pos += len(sents)
    return (np.array(ptr, np.int64), np.concatenate(cc).astype(cols.dtype), np.concatenate(vv)), owner, nv + N_EXTRA


def fit_length(docs):
    x = np.log([len(d["document"].split()) for d in docs])
    y = np.log([len(d["summary"].split()) for d in docs])
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def select(sents, score, budget, min_words=4):
    order = np.argsort(-np.asarray(score))
    keep, words = [], 0
    for i in order:
        w = len(sents[i].split())
        if w < min_words:
            continue
        keep.append(i)
        words += w
        if words >= budget:
            break
    return " ".join(sents[i] for i in sorted(keep))


def summarise(docs, prepped, probs, owner, length, factor):
    a, b = length
    out = []
    for k, (d, (sents, _)) in enumerate(zip(docs, prepped)):
        budget = factor * math.exp(a + b * math.log(len(d["document"].split()) + 1))
        out.append(select(sents, probs[owner == k], budget))
    return out


def fit_score(train_docs, p_train, score_sets, C):
    """Train on some cases, return the sentence scores of each list in score_sets (lists of prepped cases)."""
    tfidf = Tfidf(ngram=2, min_df=5, max_features=300000).fit([s for ss, _ in p_train for s in ss])
    Xtr, _, ncol = features(tfidf, p_train)
    Y = np.array([y for _, lab in p_train for y in lab], np.float32)
    W, b = train_logreg(Xtr, ncol, Y, C=C, steps=300)
    out = []
    for p in score_sets:
        X, owner, _ = features(tfidf, p)
        s = predict_logreg(X, ncol, W, b)[:, 0]
        out.append([s[owner == k].tolist() for k in range(len(p))])
    return out


def cross_fit(args):
    """Sentence scores of every training case (train + dev) from a model that did not see it (k folds), and of
    every test case from the model trained on all of them; saved as {id: [score per sentence]}."""
    from scratch_text import dev_folds
    docs, test = summ.load("all"), summ.load("test")
    prepped, p_test = prep_all(docs), prep_all(test)
    fold = dev_folds(len(docs), args.xfit)
    scores = {}
    for k in range(args.xfit):
        tr = [i for i in range(len(docs)) if fold[i] != k]
        ho = [i for i in range(len(docs)) if fold[i] == k]
        (s,) = fit_score([docs[i] for i in tr], [prepped[i] for i in tr], [[prepped[i] for i in ho]], args.C)
        scores.update({docs[i]["id"]: v for i, v in zip(ho, s)})
        print(f"fold {k} done", flush=True)
    (st,) = fit_score(docs, prepped, [p_test], args.C)
    json.dump(scores, open(f"{args.out}/xfit_scores.json", "w"))
    json.dump({d["id"]: v for d, v in zip(test, st)}, open(f"{args.out}/test_scores_all.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--xfit", type=int, default=0, help="only write cross-fitted sentence scores (k folds)")
    ap.add_argument("--all", action="store_true", help="train on train + dev (final test summaries)")
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--factors", default="0.6,0.7,0.8,0.9,1.0,1.1,1.2")
    ap.add_argument("--factor", type=float, default=0.0, help="with --all: the factor chosen on dev")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.xfit:
        return cross_fit(args)
    t0 = time.time()
    train = summ.load("all" if args.all else "train")
    dev, test = summ.load("dev"), summ.load("test")
    p_train, p_dev, p_test = prep_all(train), prep_all(dev), prep_all(test)
    print(f"sentences: train {sum(len(s) for s, _ in p_train)}, dev {sum(len(s) for s, _ in p_dev)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    tfidf = Tfidf(ngram=2, min_df=5, max_features=300000).fit([s for ss, _ in p_train for s in ss])
    Xtr, own_tr, ncol = features(tfidf, p_train)
    Xdv, own_dv, _ = features(tfidf, p_dev)
    Xte, own_te, _ = features(tfidf, p_test)
    Y = np.array([y for _, lab in p_train for y in lab], np.float32)
    print(f"features {ncol}, mean label {Y.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    W, b = train_logreg(Xtr, ncol, Y, C=args.C, steps=300, log_every=100)
    pdv, pte = predict_logreg(Xdv, ncol, W, b)[:, 0], predict_logreg(Xte, ncol, W, b)[:, 0]
    length = fit_length(train)
    ydv = np.array([y for _, lab in p_dev for y in lab])
    print(f"dev correlation score/label {np.corrcoef(pdv, ydv)[0, 1]:.3f}; length a={length[0]:.3f} "
          f"b={length[1]:.3f}", flush=True)
    refs = [d["summary"] for d in dev]
    res = {}
    lead = []
    for d, (sents, _) in zip(dev, p_dev):
        budget = math.exp(length[0] + length[1] * math.log(len(d["document"].split()) + 1))
        lead.append(select(sents, -np.arange(len(sents)), budget))
    res["lead_x1.0"] = summ.mean_scores(lead, refs)
    print("lead", res["lead_x1.0"], flush=True)
    best = None
    for f in [float(x) for x in args.factors.split(",")]:
        r = summ.mean_scores(summarise(dev, p_dev, pdv, own_dv, length, f), refs)
        res[f"model_x{f}"] = r
        print(f"factor {f}: {r}", flush=True)
        if best is None or r["r1"] + r["rl"] > best[1]:
            best = (f, r["r1"] + r["rl"])
    factor = args.factor or best[0]
    print("factor used", factor)
    dev_sum = summarise(dev, p_dev, pdv, own_dv, length, factor)
    test_sum = summarise(test, p_test, pte, own_te, length, factor)
    json.dump({d["id"]: s for d, s in zip(dev, dev_sum)}, open(f"{args.out}/dev_summaries.json", "w"))
    json.dump({d["id"]: s for d, s in zip(test, test_sum)}, open(f"{args.out}/test_summaries.json", "w"))
    np.save(f"{args.out}/dev_sentence_scores.npy", pdv)
    np.save(f"{args.out}/test_sentence_scores.npy", pte)
    json.dump({"args": vars(args), "length": length, "factor": factor, "dev": res,
               "seconds": time.time() - t0}, open(f"{args.out}/result.json", "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
