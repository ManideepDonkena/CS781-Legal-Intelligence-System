"""From-scratch word-count models for Phase 4 and Phase 5 (see scratch_text.py; no scikit-learn, SciPy or joblib).

  python wordcount_run.py --task cjpe --out tfidf_scratch                       # Phase 4, trained on train
  python wordcount_run.py --task lsi --out tfidf_scratch_xf5 --xfit 5           # Phase 5, cross-fitted on dev
  python wordcount_run.py --task lsi --out tfidf_c5_scratch_xf5 --letters --ngram 5 --max_features 1000000 --xfit 5
  python wordcount_run.py --task lsi --out knn30_scratch_xf5 --model knn --xfit 5

--model logreg: TF-IDF + one logistic regression per output; --model knn: similarity-weighted vote of the --k
nearest labelled cases.  With --xfit K, each of K dev folds is predicted by a model trained on train + the other
dev folds and the test set by a model trained on train + all of dev, so the dev probabilities stay honest.
Writes <phase>/runs/<out>/{dev,test}_prob.npy (Phase 4: probability of 'accepted'; Phase 5: 100 statute columns).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "Phase4_CJPE", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "Phase5_LSI", "src"))
import cjpe  # noqa: E402
import lsi  # noqa: E402
import scratch_text as st  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cjpe", "lsi"], required=True)
    ap.add_argument("--out", required=True, help="folder name under the phase's runs/")
    ap.add_argument("--model", choices=["logreg", "knn"], default="logreg")
    ap.add_argument("--letters", action="store_true")
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=0, help="0: 500k for cjpe, 400k for lsi")
    ap.add_argument("--C", type=float, default=0.0, help="0: 4 for cjpe, 8 for lsi (as in the library runs)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--xfit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dev_weight", type=float, default=1.0, help="with --xfit: loss weight of dev cases in training")
    ap.add_argument("--text_fix", action="store_true", help="cjpe: undo the company/number glitch first")
    args = ap.parse_args()
    t0 = time.time()
    mod = cjpe if args.task == "cjpe" else lsi
    runs = os.path.join(HERE, "..", "Phase4_CJPE" if args.task == "cjpe" else "Phase5_LSI", "runs")
    C = args.C or (4.0 if args.task == "cjpe" else 8.0)
    max_features = args.max_features or (500000 if args.task == "cjpe" else 400000)
    tr, dv, te = mod.load("train"), mod.load("dev"), mod.load("test")
    if args.text_fix and args.task == "cjpe":
        import text_fix
        table = text_fix.load_table()
        for d in tr + dv + te:
            d["text"] = text_fix.fix(d["text"], table)
    if args.task == "cjpe":
        Ytr, gold = np.array([[d["y"]] for d in tr]), np.array([[d["y"]] for d in dv])
    else:
        Ytr, gold = lsi.label_matrix(tr), lsi.label_matrix(dv)

    tf = st.Tfidf(ngram=args.ngram, letters=args.letters, max_features=max_features, workers=args.workers)
    tf.fit([d["text"] for d in (tr + dv if args.xfit else tr)])
    Xtr, Xdv, Xte = (tf.transform([d["text"] for d in s]) for s in (tr, dv, te))
    V = len(tf.vocab)
    print(f"features: {V} terms, {len(Xtr[1]) + len(Xdv[1]) + len(Xte[1])} non-zero values ({time.time() - t0:.0f}s)",
          flush=True)

    def fit_predict(train_csr, Y, targets, n_dev=0):
        if args.model == "knn":
            return [st.knn_vote(t, train_csr, V, Y, args.k, args.power) for t in targets]
        weights = None
        if n_dev and args.dev_weight != 1.0:  # the last n_dev training rows are dev cases
            weights = np.ones(len(Y))
            weights[-n_dev:] = args.dev_weight
        W, b = st.train_logreg(train_csr, V, Y, C, steps=args.steps, lr=args.lr, log_every=args.steps // 4,
                               weights=weights)
        return [st.predict_logreg(t, V, W, b) for t in targets]

    if args.xfit:
        fold = st.dev_folds(len(dv), args.xfit)
        pdv = np.zeros(gold.shape, np.float32)
        for k in range(args.xfit):
            inside, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            pdv[held] = fit_predict(st.stack(Xtr, st.rows(Xdv, inside)), np.concatenate([Ytr, gold[inside]]),
                                    [st.rows(Xdv, held)], n_dev=len(inside))[0]
            print(f"dev fold {k} done ({time.time() - t0:.0f}s)", flush=True)
        pte = fit_predict(st.stack(Xtr, Xdv), np.concatenate([Ytr, gold]), [Xte], n_dev=len(dv))[0]
    else:
        pdv, pte = fit_predict(Xtr, Ytr, [Xdv, Xte])

    os.makedirs(f"{runs}/{args.out}", exist_ok=True)
    if args.task == "cjpe":
        pdv, pte = pdv[:, 0], pte[:, 0]
        h = cjpe.honest(gold[:, 0], pdv)
    else:
        h = lsi.honest_threshold_score(gold, pdv)
    print(f"dev macro-F1 with the threshold(s) tuned on the other half: {100 * h:.2f} ({time.time() - t0:.0f}s)")
    np.save(f"{runs}/{args.out}/dev_prob.npy", pdv.astype(np.float32))
    np.save(f"{runs}/{args.out}/test_prob.npy", pte.astype(np.float32))
    json.dump({"args": vars(args), "C": C, "max_features": max_features, "terms": V, "dev_honest_macro_f1": h},
              open(f"{runs}/{args.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
