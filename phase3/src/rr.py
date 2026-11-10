"""Shared code for Phase 3 (rhetorical role prediction): data, folds, scoring, submission files."""
import json
import os
import random
import zipfile

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Phase_3_RR")
DOMAINS = ("CL", "IT")


def load_labels(data_dir=DATA):
    vocab = json.load(open(f"{data_dir}/Train/label_vocab.json"))
    return [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


LABELS = load_labels()
LABEL_ID = {l: i for i, l in enumerate(LABELS)}


def load_labelled(data_dir=DATA, splits=("train", "dev")):
    """Every labelled document as {id, domain, sents, y}; train and dev are pooled for cross-validation."""
    docs = []
    for split in splits:
        for dom in DOMAINS:
            for x in json.load(open(f"{data_dir}/Train/{dom}_{split}.json")):
                assert len(x["text"]) == len(x["labels"])
                docs.append({"id": x["id"], "domain": dom, "split": split, "sents": x["text"],
                             "y": np.array([LABEL_ID[l] for l in x["labels"]])})
    return docs


def load_test(data_dir=DATA):
    docs = []
    for dom in DOMAINS:
        for k, sents in json.load(open(f"{data_dir}/Test/{dom}_test.json")).items():
            docs.append({"id": k, "domain": dom, "sents": sents, "y": None})
    return docs


def doc_folds(docs, k, seed):
    """k document-level folds, each with the same share of CL and IT documents.  Returns a fold id per document."""
    rng = random.Random(seed)
    fold = [0] * len(docs)
    for dom in DOMAINS:
        idx = sorted((i for i, d in enumerate(docs) if d["domain"] == dom), key=lambda i: docs[i]["id"])
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % k
    return fold


def scores(gold, pred, n_labels=len(LABELS)):
    """Macro-F1 two ways, since the official scorer's handling of labels missing from the test set is unknown:
    'all13' averages over all 13 labels (a label absent from gold and predictions counts 0);
    'present' averages over labels that occur in gold or predictions (scikit-learn's default)."""
    gold, pred = np.asarray(gold), np.asarray(pred)
    f1 = np.zeros(n_labels)
    present = np.zeros(n_labels, dtype=bool)
    for c in range(n_labels):
        tp = np.sum((gold == c) & (pred == c))
        fp = np.sum((gold != c) & (pred == c))
        fn = np.sum((gold == c) & (pred != c))
        present[c] = tp + fp + fn > 0
        f1[c] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"all13": f1.mean(), "present": f1[present].mean(), "acc": float(np.mean(gold == pred)),
            "per_label": dict(zip(LABELS, f1.round(3).tolist()))}


def report(name, s):
    worst = sorted(s["per_label"].items(), key=lambda kv: kv[1])
    return (f"{name}: macroF1 all13 {100 * s['all13']:.2f} | present {100 * s['present']:.2f} | acc {100 * s['acc']:.1f}\n"
            "   " + ", ".join(f"{k} {v:.2f}" for k, v in worst))


def absorbing_dissent(lp, pred):
    """Dissent decoded as a suffix of the document.  In the training labels every sentence after a Dissent sentence
    is Dissent too (a separate opinion runs to the end), so the only choice is where it starts, or none at all:
    the start s maximises sum over i >= s of (log p(Dissent) - log p(best other label)); no Dissent if that sum is
    not positive.  lp: (sentences, labels) log-probabilities; pred: the labels decoded without Dissent."""
    d = LABEL_ID["Dissent"]
    pred = np.array(pred).copy()
    gain = lp[:, d] - lp[np.arange(len(pred)), pred]
    suffix = np.cumsum(gain[::-1])[::-1]
    s = int(suffix.argmax())
    if suffix[s] > 0:
        pred[s:] = d
    return pred


def write_submission(test_docs, preds, out_dir):
    """preds: {doc id: array of label ids}.  Checks the format rules, writes prediction.json and a zip."""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for d in test_docs:
        p = preds[d["id"]]
        assert len(p) == len(d["sents"]), d["id"]
        out[d["id"]] = [LABELS[int(i)] for i in p]
    assert set(out) == {d["id"] for d in test_docs}
    with open(f"{out_dir}/prediction.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    name = os.path.basename(os.path.normpath(out_dir))
    with zipfile.ZipFile(f"{out_dir}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{out_dir}/prediction.json", "prediction.json")
    return f"{out_dir}/{name}.zip"
