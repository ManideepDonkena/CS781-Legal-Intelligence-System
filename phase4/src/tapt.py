"""Task-adaptive pretraining: continue the Phase 1 next-token training of our language model on one task's text.

The model reads exactly the token windows the classifier will read later (ft_doc.token_ids, with the same
--text_fix), taken from the train and dev cases only; test text is never used.  2% of those cases are held out to
report the next-token loss before and after.  Writes <out>/best.pt in the Phase 1 checkpoint format and copies
tokenizer.json, so that `ft_doc.py --backbone <out>` fine-tunes from the adapted model.

  python tapt.py --task cjpe --out /workspace/models/cs781_phase45/tapt_cjpe --text_fix
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ft_doc import batches, load_model, pad, token_ids  # also puts the shared and Phase 5 code on sys.path
import cjpe
import lsi


def lm_loss(lm, ids, m):
    """Mean next-token cross-entropy over real (non-padding) positions."""
    logits = lm(ids)["logits"][:, :-1].float()
    tgt, keep = ids[:, 1:], m[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
    return (loss * keep.reshape(-1)).sum() / keep.sum()


@torch.no_grad()
def held_loss(lm, seqs, batch_tokens):
    lm.eval()
    lengths = [len(s) for s in seqs]
    tot, n = 0.0, 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches(lengths, sorted(range(len(seqs)), key=lambda i: lengths[i]), batch_tokens):
            ids, m, _ = pad(seqs, b)
            k = m[:, 1:].sum().item()
            tot, n = tot + lm_loss(lm, ids, m).item() * k, n + k
    lm.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cjpe", "lsi"], required=True)
    ap.add_argument("--backbone", default="/workspace/models/cs781_phase1_runs/ROPEFIX_295M")
    ap.add_argument("--repo", default="/home/mani/cs781_phase1/Legal_LLM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--tail_tokens", type=int, default=0)
    ap.add_argument("--text_fix", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch_tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many cases")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    t0 = time.time()
    mod = cjpe if args.task == "cjpe" else lsi
    docs = mod.load("train") + mod.load("dev")
    if args.limit:
        docs = random.sample(docs, args.limit)
    if args.text_fix:
        if args.task == "cjpe":
            import text_fix
            table = text_fix.load_table()
            for d in docs:
                d["text"] = text_fix.fix(d["text"], table)
        else:
            for d in docs:
                d["text"] = lsi.compact_masks(d["text"])
    lm, tok, family = load_model(args.backbone, args.repo)
    lm = lm.cuda()
    seqs = [s for s in token_ids(tok, docs, args.task, args.max_tokens, args.tail_tokens) if len(s) > 1]
    random.shuffle(seqs)
    n_held = max(200, len(seqs) // 50)
    held, train = seqs[:n_held], seqs[n_held:]
    print(f"{len(train)} training windows, {sum(len(s) for s in train) / 1e6:.1f}M tokens; "
          f"{n_held} held out ({time.time() - t0:.0f}s)", flush=True)
    before = held_loss(lm, held, args.batch_tokens)
    print(f"held-out next-token loss before: {before:.4f}", flush=True)

    opt = torch.optim.AdamW(lm.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    lengths = [len(s) for s in train]

    def epoch_batches():
        idx = sorted(range(len(train)), key=lambda i: lengths[i] + random.random())
        bs = list(batches(lengths, idx, args.batch_tokens))
        random.shuffle(bs)
        return bs

    n_steps = args.epochs * len(epoch_batches())
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / n_steps))))
    lm.train()
    step, tot, n = 0, 0.0, 0
    for ep in range(args.epochs):
        for b in epoch_batches():
            ids, m, _ = pad(train, b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lm_loss(lm, ids, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tot, n = tot + loss.item(), n + 1
            if step % 200 == 0:
                print(f"step {step}/{n_steps} train loss {tot / n:.4f} ({time.time() - t0:.0f}s, "
                      f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB)", flush=True)
                tot, n = 0.0, 0
    after = held_loss(lm, held, args.batch_tokens)
    print(f"held-out next-token loss after: {after:.4f} (before {before:.4f})", flush=True)
    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(f"{args.backbone}/best.pt", map_location="cpu", weights_only=False)
    torch.save({"model_config": ck["model_config"], "model": lm.state_dict(),
                "tapt": {"args": vars(args), "held_loss_before": before, "held_loss_after": after}},
               f"{args.out}/best.pt")
    shutil.copy(f"{args.backbone}/tokenizer.json", f"{args.out}/tokenizer.json")
    json.dump({"args": vars(args), "held_loss_before": before, "held_loss_after": after,
               "train_windows": len(train), "seconds": time.time() - t0}, open(f"{args.out}/tapt.json", "w"), indent=1)
    print("done", time.time() - t0, "s")


if __name__ == "__main__":
    main()
