"""Sentence vectors from a frozen Phase 1 model.

Each sentence is read together with up to --context tokens of the sentences before it (the model is causal, so
it can only use text to the left).  For every layer we keep the mean of the sentence's token states and the
state of its last token.  Output: <out>/{mean,last}.npy of shape (sentences, layers + 1, d) in float16, plus
index.json giving each document's sentence range.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/mani/cs781/Capstone/Phase2_LNER/src")
import lner  # noqa: E402
import rr  # noqa: E402


def build_items(tok, docs, context, max_sent):
    """Per sentence: (left-context token ids, sentence token ids); plus each document's sentence range."""
    items, index = [], []
    for d in docs:
        start = len(items)
        enc = tok.encode_batch([(" " if i else "") + s for i, s in enumerate(d["sents"])], add_special_tokens=False)
        stream = []
        for e in enc:
            ids = e.ids[:max_sent] or [tok.token_to_id(" ") or 0]
            items.append((stream[-context:] if context else [], ids))
            stream += e.ids
        index.append({"id": d["id"], "domain": d["domain"], "start": start, "end": len(items)})
    return items, index


def batches(items, batch_tokens, order):
    """Consecutive groups from `order` whose padded size stays under batch_tokens."""
    pos = 0
    while pos < len(order):
        longest, batch = 0, []
        while pos < len(order):
            c, s = items[order[pos]]
            longest = max(longest, len(c) + len(s))
            if batch and longest * (len(batch) + 1) > batch_tokens:
                break
            batch.append(order[pos])
            pos += 1
        yield batch


def pad(items, batch, device="cuda"):
    """Right-padded ids, a mask over the sentence's own tokens, and each sentence's last position."""
    t = max(len(items[i][0]) + len(items[i][1]) for i in batch)
    ids = torch.zeros(len(batch), t, dtype=torch.long)
    m = torch.zeros(len(batch), t)
    last_pos = []
    for r, i in enumerate(batch):
        c, s = items[i]
        ids[r, :len(c) + len(s)] = torch.tensor(c + s)
        m[r, len(c):len(c) + len(s)] = 1
        last_pos.append(len(c) + len(s) - 1)
    return ids.to(device), m.to(device), torch.tensor(last_pos, device=device)


@torch.no_grad()
def extract(states_fn, items, n_layers, d_model, out, batch_tokens, token_layers=(), max_tok=96):
    """Write mean.npy / last.npy (and per-token states of token_layers) for every item.
    states_fn(ids) returns the list of per-layer states (b, t, d)."""
    os.makedirs(out, exist_ok=True)
    mean = np.lib.format.open_memmap(f"{out}/mean.npy", "w+", np.float16, (len(items), n_layers, d_model))
    last = np.lib.format.open_memmap(f"{out}/last.npy", "w+", np.float16, (len(items), n_layers, d_model))
    tl = list(token_layers)
    if tl:  # per-token states of the chosen layers, sentences stored one after another
        lens = np.array([min(len(s), max_tok) for _, s in items])
        tok_start = np.concatenate([[0], np.cumsum(lens)])
        np.save(f"{out}/tok_start.npy", tok_start)
        toks = np.lib.format.open_memmap(f"{out}/tokens.npy", "w+", np.float16, (int(tok_start[-1]), len(tl), d_model))
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    done = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in batches(items, batch_tokens, order):
            ids, m, last_pos = pad(items, batch)
            states = torch.stack(states_fn(ids), 1).float()  # (b, layers, t, d); right padding is harmless
            mv = (states * m[:, None, :, None]).sum(2) / m.sum(1)[:, None, None]
            lv = states[torch.arange(len(batch), device="cuda"), :, last_pos]
            if tl:
                sel = states[:, tl].transpose(1, 2).cpu().numpy().astype(np.float16)  # (b, t, layers, d)
                for r, i in enumerate(batch):
                    c = len(items[i][0])
                    toks[tok_start[i]:tok_start[i + 1]] = sel[r, c:c + lens[i]]
            mean[batch] = mv.cpu().numpy().astype(np.float16)
            last[batch] = lv.cpu().numpy().astype(np.float16)
            done += len(batch)
            print(f"\r{done}/{len(order)}", end="", flush=True)
    mean.flush()
    last.flush()
    if tl:
        toks.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--repo", default="/home/mani/cs781_phase1/Legal_LLM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=256, help="tokens of preceding text given as left context")
    ap.add_argument("--max_sent", type=int, default=256, help="sentence tokens kept (the rest is cut)")
    ap.add_argument("--batch_tokens", type=int, default=32768)
    ap.add_argument("--token_layers", default="", help="comma list of layers whose per-token states are also saved")
    ap.add_argument("--max_tok", type=int, default=96, help="sentence tokens saved per sentence with --token_layers")
    args = ap.parse_args()

    lm, tok, family = lner.load_backbone(args.backbone, args.repo)
    model = lner.Tagger(lm, family, n_tags=1).cuda().eval()
    items, index = build_items(tok, rr.load_labelled() + rr.load_test(), args.context, args.max_sent)
    tl = [int(x) for x in args.token_layers.split(",") if x]
    extract(lambda ids: model.states(ids, None), items, len(lm.layers) + 1, lm.token.weight.shape[1], args.out,
            args.batch_tokens, tl, args.max_tok)
    json.dump({"backbone": args.backbone, "context": args.context, "token_layers": tl, "docs": index},
              open(f"{args.out}/index.json", "w"))
    print("\ndone", len(items), "sentences")


if __name__ == "__main__":
    main()
