"""Score every chunk of a case with a fine-tuned model (weights from ft_doc.py --save).

cjpe: chunks are counted from the end: chunk 0 is the last --max_tokens tokens (what the model was trained on),
      chunk 1 the --max_tokens before it, and so on.  Saves <out>/{dev,test}_chunk_logit.npy of shape
      (cases, chunks): log-odds of 'accepted', NaN where a case has no such chunk.
lsi:  chunks are counted from the start: chunk 0 is the first --max_tokens tokens.  Saves
      <out>/{dev,test}_chunk_prob.npy of shape (cases, chunks, 100), NaN where a case has no such chunk.
A piece shorter than --min_tokens (other than chunk 0) is skipped.
"""
import argparse
import os

import numpy as np
import torch

from ft_doc import DocClassifier, batches, load_model, pad  # also puts the shared and Phase 5 code on sys.path
import cjpe
import lsi


def chunk_ids(tok, docs, task, L, n_chunks, min_tokens):
    """Token chunks of every case and, per chunk, (case index, chunk number)."""
    if task == "cjpe":
        texts = [" ".join(d["text"].split()[-L * n_chunks:]) for d in docs]
    else:
        texts = [" ".join(d["text"].split()[:L * n_chunks]) for d in docs]
    seqs, owner = [], []
    for i, e in enumerate(tok.encode_batch(texts, add_special_tokens=False)):
        ids = e.ids[-L * n_chunks:] if task == "cjpe" else e.ids[:L * n_chunks]
        for k in range(n_chunks):
            if task == "cjpe":
                piece = ids[max(0, len(ids) - (k + 1) * L):len(ids) - k * L]
            else:
                piece = ids[k * L:(k + 1) * L]
            if len(piece) < (min_tokens if k else 1):
                break
            seqs.append(piece)
            owner.append((i, k))
    return seqs, owner


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--min_tokens", type=int, default=256)
    ap.add_argument("--batch_tokens", type=int, default=32768)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    task = a["task"]
    lm, tok, family = load_model(a["backbone"], a["repo"])
    n_out = 2 if task == "cjpe" else lsi.N_LABELS
    model = DocClassifier(lm, n_out, label_att=a.get("label_att", False))
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    os.makedirs(args.out, exist_ok=True)
    mod = cjpe if task == "cjpe" else lsi
    for split in ("dev", "test"):
        docs = mod.load(split)
        seqs, owner = chunk_ids(tok, docs, task, a["max_tokens"], args.chunks, args.min_tokens)
        shape = (len(docs), args.chunks) if task == "cjpe" else (len(docs), args.chunks, n_out)
        out = np.full(shape, np.nan, dtype=np.float32)
        lengths = [len(s) for s in seqs]
        order = sorted(range(len(seqs)), key=lambda j: lengths[j])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for b in batches(lengths, order, args.batch_tokens):
                lg = model(*pad(seqs, b)).float()
                v = (lg[:, 1] - lg[:, 0]) if task == "cjpe" else lg.sigmoid()
                for j, x in zip(b, v.cpu().numpy()):
                    out[owner[j]] = x
        name = "chunk_logit" if task == "cjpe" else "chunk_prob"
        np.save(f"{args.out}/{split}_{name}.npy", out)
        has = ~np.isnan(out if task == "cjpe" else out[..., 0])
        print(split, len(docs), "cases,", len(seqs), "chunks; cases with chunk k:",
              [int(has[:, k].sum()) for k in range(args.chunks)], flush=True)


if __name__ == "__main__":
    main()
