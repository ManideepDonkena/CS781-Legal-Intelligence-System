"""Fine-tune the Phase 1 model as a sentence classifier (sentence + left context -> role) on the training documents of
one cross-validation fold, then write sentence vectors for every document in embed.py's format.

The vectors of fold f come from a model that never saw fold f's documents, so train_rr.py --fold_emb can score
fold f honestly.  With --fold -1 the model is trained on all labelled documents (for the test set only).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/home/mani/cs781/Capstone/Phase2_LNER/src")
import lner  # noqa: E402
import rr  # noqa: E402
from embed import batches, build_items, extract, pad  # noqa: E402

N_BINS = 20


class SentClassifier(nn.Module):
    def __init__(self, lm, family, heads=2, n_labels=len(rr.LABELS)):
        super().__init__()
        self.enc = lner.Tagger(lm, family, n_tags=1)
        d = lm.token.weight.shape[1]
        self.att = nn.Linear(d, heads)
        self.pos = nn.Embedding(N_BINS, 32)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(heads * d + d + 32, n_labels)

    def forward(self, ids, m, pos_bin):
        h = self.enc.lm.norm(self.enc.states(ids, None)[-1]).float()
        mean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        a = self.att(h).masked_fill(m[..., None] == 0, -1e4).softmax(1)
        pooled = torch.einsum("bth,btd->bhd", a, h).flatten(1)
        return self.out(self.drop(torch.cat([pooled, mean, self.pos(pos_bin)], -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="/workspace/models/cs781_phase1_runs/ROPEFIX_295M")
    ap.add_argument("--repo", default="/home/mani/cs781_phase1/Legal_LLM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, required=True, help="held-out fold, or -1 to train on everything")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--max_sent", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--batch_tokens", type=int, default=8192)
    ap.add_argument("--cw", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token_layers", default="")
    ap.add_argument("--lorma_rank", type=int, default=0, help="LoRMA study: adapter rank (0 = train every weight)")
    ap.add_argument("--lorma_alpha", type=float, default=32)
    ap.add_argument("--lorma_top", type=int, default=0, help="adapters only in the last k blocks (0 = all)")
    ap.add_argument("--no_extract", action="store_true", help="only train and score the classifier")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    lm, tok, family = lner.load_backbone(args.backbone, args.repo)
    os.makedirs(args.out, exist_ok=True)
    enc_params = None
    if args.lorma_rank:
        import adapters
        top = range(len(lm.layers) - args.lorma_top, len(lm.layers)) if args.lorma_top else None
        enc_params = adapters.add_adapters(lm, "lorma", args.lorma_rank, args.lorma_alpha, layers=top)
    model = SentClassifier(lm, family).cuda()
    if enc_params is None:
        enc_params = list(model.enc.parameters())
    print(f"trainable encoder parameters {sum(p.numel() for p in enc_params) / 1e6:.2f}M", flush=True)
    docs, test = rr.load_labelled(), rr.load_test()
    items, index = build_items(tok, docs + test, args.context, args.max_sent)
    fold = rr.doc_folds(docs, args.k, args.split_seed)
    labels, pos_bins, train_idx = [], [], []
    for j, (d, ix) in enumerate(zip(docs + test, index)):
        n = ix["end"] - ix["start"]
        pos_bins += list(np.minimum((N_BINS * np.arange(n) / n).astype(int), N_BINS - 1))
        labels += list(d["y"]) if d["y"] is not None else [-1] * n
        if j < len(docs) and fold[j] != args.fold:
            train_idx += range(ix["start"], ix["end"])
    labels, pos_bins = torch.tensor(labels), torch.tensor(pos_bins)
    counts = np.bincount(labels[train_idx].numpy(), minlength=len(rr.LABELS)).astype(float)
    cw = (counts.max() / np.maximum(counts, 1)) ** args.cw
    cw[counts == 0] = 0
    cw = torch.tensor(cw / cw[counts > 0].mean(), dtype=torch.float, device="cuda")

    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    # Length-sorted chunks keep padding low; the chunks themselves are shuffled every epoch.
    def epoch_batches():
        idx = sorted(train_idx, key=lambda i: len(items[i][0]) + len(items[i][1]) + random.random())
        bs = list(batches(items, args.batch_tokens, idx))
        random.shuffle(bs)
        return bs
    n_steps = args.epochs * len(epoch_batches())
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 100) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / n_steps))))
    t0, step = time.time(), 0
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for b in epoch_batches():
            ids, m, _ = pad(items, b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, m, pos_bins[b].cuda())
            loss = F.cross_entropy(logits.float(), labels[b].cuda(), weight=cw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tot, n = tot + loss.item(), n + 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{n_steps} loss {tot / n:.3f} ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    # Held-out accuracy of the sentence classifier alone (a quick check, not the final score).
    if args.fold >= 0:
        held = [i for j, ix in enumerate(index[:len(docs)]) if fold[j] == args.fold for i in range(ix["start"], ix["end"])]
        pred = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in batches(items, 32768, held):
                ids, m, _ = pad(items, b)
                pred += model(ids, m, pos_bins[b].cuda()).argmax(1).tolist()
        s = rr.scores(labels[held].numpy(), np.array(pred))
        print(rr.report(f"fold {args.fold} sentence classifier alone", s), flush=True)
        print(f"train minutes {(time.time() - t0) / 60:.1f}, peak GB {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}",
              flush=True)
        json.dump({"args": vars(args), "scores": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s.items()},
                   "minutes": (time.time() - t0) / 60, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30},
                  open(f"{args.out}/classifier_log.json", "w"), indent=1, default=float)
    if args.no_extract:
        return

    tl = [int(x) for x in args.token_layers.split(",") if x]
    extract(lambda ids: model.enc.states(ids, None), items, len(lm.layers) + 1, lm.token.weight.shape[1], args.out,
            32768, tl)
    json.dump({"backbone": args.backbone, "finetuned_fold": args.fold, "context": args.context,
               "token_layers": tl, "docs": index}, open(f"{args.out}/index.json", "w"))
    print("\ndone", time.time() - t0, "s")


if __name__ == "__main__":
    main()
