"""Assemble first-pass label probabilities for the second-pass (document-memory) tagger.

Cross-fitting: each training fold gets probabilities from the model trained on the *other* fold, so
the second pass never sees first-pass predictions made by a model that was trained on that document.
The test fold gets the average of both fold models.  Several configs can be averaged.

With --kfold the first pass is a set of k-fold runs (<config>_k0 ... each with heldout_probs.npz);
the test fold then takes the probabilities of the model trained on all documents (--final).
"""
import argparse
import glob
import json
import os

import numpy as np


def load(path):
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def average(dicts):
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in dicts[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kfold", action="store_true", help="configs are k-fold runs; needs --data and --final")
    ap.add_argument("--data", default="", help="dataset directory (to split the held-out documents by fold)")
    ap.add_argument("--final", nargs="*", default=[], help="full-data run of each config (for the test fold)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.kfold:
        assert len(args.final) == len(args.configs)
        per_config = []
        for c in args.configs:
            merged = {}
            for path in sorted(glob.glob(f"{args.runs}/{c}_k*/heldout_probs.npz")):
                merged.update(load(path))
            per_config.append(merged)
        heldout = average(per_config)
        fold3 = average([load(f"{args.runs}/{f}/fold3_probs.npz") for f in args.final])
        for fold in ("fold1", "fold2"):
            with open(f"{args.data}/Train/{fold}-text.json", encoding="utf-8") as f:
                ids = list(json.load(f))
            np.savez_compressed(f"{args.out}/{fold}_probs.npz", **{k: heldout[k].astype(np.float16) for k in ids})
            print(fold, len(ids), "docs")
        np.savez_compressed(f"{args.out}/fold3_probs.npz", **{k: v.astype(np.float16) for k, v in fold3.items()})
        print("fold3", len(fold3), "docs")
        return
    fold1 = average([load(f"{args.runs}/{c}_fold2/fold1_probs.npz") for c in args.configs])
    fold2 = average([load(f"{args.runs}/{c}_fold1/fold2_probs.npz") for c in args.configs])
    fold3 = average([load(f"{args.runs}/{c}_{f}/fold3_probs.npz") for c in args.configs for f in ("fold1", "fold2")])
    for name, d in (("fold1", fold1), ("fold2", fold2), ("fold3", fold3)):
        np.savez_compressed(f"{args.out}/{name}_probs.npz", **{k: v.astype(np.float16) for k, v in d.items()})
        print(name, len(d), "docs")


if __name__ == "__main__":
    main()
