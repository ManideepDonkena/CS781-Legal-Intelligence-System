"""Fine-tune the Phase 1 model as a whole-case classifier.

  --task cjpe : Phase 4, accepted (1) or rejected (0); reads the LAST --max_tokens tokens of the case.
  --task lsi  : Phase 5, which of 100 statutes apply; reads the FIRST --max_tokens tokens (or head + tail with
                --tail_tokens).

The model reads the tokens causally, as in pre-training.  The final layer's states are pooled three ways
(attention pooling with a few heads, the mean, the last token) and a linear layer gives the outputs.
Trained on train only (with --dev_fold, also on the other dev folds); after every epoch the dev and test
probabilities are saved to <out>/{dev,test}_prob_e{epoch}.npy and the dev scores (held-out dev cases only) are
logged.  The test file's labels are never read.
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
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, "/home/mani/cs781/Capstone/shared")
sys.path.insert(0, "/home/mani/cs781/Capstone/Phase4_CJPE/src")
sys.path.insert(0, "/home/mani/cs781/Capstone/Phase5_LSI/src")
sys.path.insert(0, "/home/mani/cs781/Capstone/Phase2_LNER/src")
import cjpe  # noqa: E402
import lsi  # noqa: E402
from scratch_bpe import ScratchBPE  # noqa: E402


def load_model(run_dir, repo):
    """A Phase 1 checkpoint (our own architecture code in the Phase 1 repo) and our own encoder for its tokenizer."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from model_zoo import build_model
    ck = torch.load(f"{run_dir}/best.pt", map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    family = cfg.pop("family")
    cfg["gradient_checkpointing"] = False
    lm = build_model(family, **cfg)
    lm.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    return lm, ScratchBPE(f"{run_dir}/tokenizer.json"), family


class DocClassifier(nn.Module):
    def __init__(self, lm, n_out, heads=4, dropout=0.1, grad_ckpt=False, label_att=False):
        super().__init__()
        self.lm, self.grad_ckpt = lm, grad_ckpt
        d = lm.token.weight.shape[1]
        for m in lm.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout
        self.att = nn.Linear(d, heads)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(heads * d + 2 * d, n_out)
        # Optional: every output (statute) gets its own attention over the tokens and its own weights on the
        # pooled vector, added to the shared logits (starts at zero, so training begins from the shared model).
        self.label_att = nn.Linear(d, n_out) if label_att else None
        self.label_w = nn.Parameter(torch.zeros(n_out, d)) if label_att else None

    def encode(self, ids):
        """Final normalised states, computed exactly as ModernCausalLM.forward does."""
        lm = self.lm
        cos, sin = lm.rotary(ids, ids.shape[1], offset=0)
        x = lm.token(ids)
        for layer in lm.layers:
            if self.grad_ckpt and self.training:
                x = checkpoint(lambda x_, l=layer: l(x_, cos, sin, kv_cache=None)[0], x, use_reentrant=False)
            else:
                x = layer(x, cos, sin, kv_cache=None)[0]
        return lm.norm(x)

    def forward(self, ids, m, last_pos):
        h = self.encode(ids).float()
        mean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        a = self.att(h).masked_fill(m[..., None] == 0, -1e4).softmax(1)
        pooled = torch.einsum("bth,btd->bhd", a, h).flatten(1)
        last = h[torch.arange(len(ids), device=ids.device), last_pos]
        logits = self.out(self.drop(torch.cat([pooled, mean, last], -1)))
        if self.label_att is not None:
            la = self.label_att(h).masked_fill(m[..., None] == 0, -1e4).softmax(1)  # (b, t, outputs)
            logits = logits + torch.einsum("btl,btd,ld->bl", la, self.drop(h), self.label_w)
        return logits


def token_ids(tok, docs, task, max_tokens, tail_tokens):
    """Token ids per case: the end of the case for cjpe, the start (plus optionally its end) for lsi.
    Words are cut first (every word is at least one token) so that long cases are cheap to encode."""
    if task == "cjpe":
        texts = [" ".join(d["text"].split()[-max_tokens:]) for d in docs]
        return [e.ids[-max_tokens:] for e in tok.encode_batch(texts, add_special_tokens=False)]
    head = max_tokens - tail_tokens
    words = [d["text"].split() for d in docs]
    out = [e.ids[:head] for e in tok.encode_batch([" ".join(w[:head]) for w in words], add_special_tokens=False)]
    if tail_tokens:
        tails = tok.encode_batch([" " + " ".join(w[head:][-tail_tokens:]) if len(w) > head else "" for w in words],
                                 add_special_tokens=False)
        out = [h + t.ids[-tail_tokens:] for h, t in zip(out, tails)]
    return out


def batches(lengths, order, batch_tokens):
    """Consecutive groups from `order` whose padded size stays under batch_tokens."""
    pos = 0
    while pos < len(order):
        longest, batch = 0, []
        while pos < len(order):
            longest = max(longest, lengths[order[pos]])
            if batch and longest * (len(batch) + 1) > batch_tokens:
                break
            batch.append(order[pos])
            pos += 1
        yield batch


def pad(seqs, batch, device="cuda"):
    t = max(len(seqs[i]) for i in batch)
    ids = torch.zeros(len(batch), t, dtype=torch.long)
    m = torch.zeros(len(batch), t)
    for r, i in enumerate(batch):
        ids[r, :len(seqs[i])] = torch.tensor(seqs[i])
        m[r, :len(seqs[i])] = 1
    last = torch.tensor([len(seqs[i]) - 1 for i in batch])
    return ids.to(device), m.to(device), last.to(device)


@torch.no_grad()
def predict(model, seqs, task, batch_tokens):
    model.eval()
    lengths = [len(s) for s in seqs]
    order = sorted(range(len(seqs)), key=lambda i: lengths[i])
    out = [None] * len(seqs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches(lengths, order, batch_tokens):
            logits = model(*pad(seqs, b)).float()
            p = logits.softmax(-1)[:, 1] if task == "cjpe" else logits.sigmoid()
            for i, v in zip(b, p.cpu().numpy()):
                out[i] = v
    model.train()
    return np.stack(out).astype(np.float32)


def dev_report(task, gold, prob):
    if task == "cjpe":
        s05 = cjpe.scores(gold, (prob >= 0.5).astype(int))
        med = float(np.median(prob))
        s_med = cjpe.scores(gold, (prob >= med).astype(int))
        return {"f1_at_0.5": s05["macro_f1"], "f1_at_median": s_med["macro_f1"], "median": med,
                "text": cjpe.report("dev @0.5", s05) + " || " + cjpe.report(f"dev @median {med:.3f}", s_med)}
    s05 = lsi.scores(gold, lsi.decide(prob, np.full(lsi.N_LABELS, 0.5)))
    honest = lsi.honest_threshold_score(gold, prob, halves=2)
    return {"f1_at_0.5": s05["macro_f1"], "micro_at_0.5": s05["micro_f1"], "honest_tuned": honest,
            "text": lsi.report("dev @0.5", s05) + f" || dev macro-F1, thresholds tuned on other half: {100 * honest:.2f}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cjpe", "lsi"], required=True)
    ap.add_argument("--backbone", default="/workspace/models/cs781_phase1_runs/ROPEFIX_295M")
    ap.add_argument("--repo", default="/home/mani/cs781_phase1/Legal_LLM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--tail_tokens", type=int, default=0, help="lsi: tokens taken from the end of the case")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch_tokens", type=int, default=16384)
    ap.add_argument("--accum", type=int, default=1, help="batches per optimiser step")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--label_att", action="store_true", help="per-output attention pooling (for lsi)")
    ap.add_argument("--pos_weight", type=float, default=0.0,
                    help="lsi: weight positive cases of statute j by (negatives_j / positives_j) ** this")
    ap.add_argument("--evals_per_epoch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many cases per split")
    ap.add_argument("--save", default="", help="write the final model's weights (bf16) to this file")
    ap.add_argument("--dev_fold", type=int, default=-1,
                    help="cross-fitting: also train on the dev cases outside this fold; dev scores use only this fold")
    ap.add_argument("--dev_folds", type=int, default=2)
    ap.add_argument("--text_fix", action="store_true",
                    help="cjpe: undo the company/number glitch (text_fix.py); lsi: single-token placeholders")
    ap.add_argument("--lorma_rank", type=int, default=0, help="LoRMA study: adapter rank (0 = train every weight)")
    ap.add_argument("--lorma_alpha", type=float, default=32)
    ap.add_argument("--lorma_rs", action="store_true", help="rank-stabilised scale alpha / sqrt(rank)")
    ap.add_argument("--lorma_top", type=int, default=0, help="adapters only in the last k blocks (0 = all)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    mod = cjpe if args.task == "cjpe" else lsi
    tr, dv, te = mod.load("train"), mod.load("dev"), mod.load("test")
    if args.limit:
        tr, dv, te = tr[:args.limit], dv[:args.limit], te[:args.limit]
    if args.text_fix:
        if args.task == "cjpe":
            import text_fix
            table = text_fix.load_table()
            fix = lambda t: text_fix.fix(t, table)  # noqa: E731
        else:
            fix = lsi.compact_masks
        for d in tr + dv + te:
            d["text"] = fix(d["text"])
    held = np.ones(len(dv), dtype=bool)  # dev cases the model never trains on
    if args.dev_fold >= 0:
        fold = lsi.dev_folds(len(dv), args.dev_folds)
        held = fold == args.dev_fold
        tr = tr + [d for d, h in zip(dv, held) if not h]
        np.save(f"{args.out}/dev_fold.npy", fold)
    lm, tok, family = load_model(args.backbone, args.repo)
    lm_params = list(lm.parameters())
    if args.lorma_rank:
        import adapters
        top = range(len(lm.layers) - args.lorma_top, len(lm.layers)) if args.lorma_top else None
        lm_params = adapters.add_adapters(lm, "lorma", args.lorma_rank, args.lorma_alpha, layers=top,
                                          rs=args.lorma_rs)
    args.trainable_lm_params = sum(p.numel() for p in lm_params)
    n_out = 2 if args.task == "cjpe" else lsi.N_LABELS
    model = DocClassifier(lm, n_out, grad_ckpt=args.grad_ckpt, label_att=args.label_att).cuda()
    seq_tr, seq_dv, seq_te = (token_ids(tok, d, args.task, args.max_tokens, args.tail_tokens) for d in (tr, dv, te))
    print(f"tokenised {len(seq_tr)}/{len(seq_dv)}/{len(seq_te)} cases, mean train length "
          f"{np.mean([len(s) for s in seq_tr]):.0f} ({time.time() - t0:.0f}s)", flush=True)
    if args.task == "cjpe":
        y_tr = torch.tensor([d["y"] for d in tr])
        gold_dv = np.array([d["y"] for d in dv])
    else:
        y_tr = torch.tensor(lsi.label_matrix(tr), dtype=torch.float)
        gold_dv = lsi.label_matrix(dv)
        pos = y_tr.sum(0).clamp_min(1)
        pos_weight = ((len(y_tr) - pos) / pos).pow(args.pos_weight).cuda()

    head = [p for n, p in model.named_parameters() if not n.startswith("lm.")]
    opt = torch.optim.AdamW([{"params": lm_params, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    lengths = [len(s) for s in seq_tr]

    def epoch_batches():
        idx = sorted(range(len(seq_tr)), key=lambda i: lengths[i] + random.random())
        bs = list(batches(lengths, idx, args.batch_tokens))
        random.shuffle(bs)
        return bs

    per_epoch = len(epoch_batches())
    n_steps = args.epochs * per_epoch // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / n_steps))))
    eval_at = {round(per_epoch * (k + 1) / args.evals_per_epoch) for k in range(args.evals_per_epoch)}
    log, step = [], 0
    model.train()
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for bi, b in enumerate(epoch_batches(), 1):
            ids, m, last = pad(seq_tr, b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, m, last).float()
            y = y_tr[b].cuda()
            if args.task == "cjpe":
                loss = F.cross_entropy(logits, y)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            (loss / args.accum).backward()
            if bi % args.accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            tot, n = tot + loss.item(), n + 1
            if bi % 100 == 0:
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"ep {ep} batch {bi}/{per_epoch} loss {tot / n:.4f} ({time.time() - t0:.0f}s, {mem:.1f} GB)",
                      flush=True)
            if bi in eval_at:
                tag = f"e{ep + 1}" if bi == per_epoch else f"e{ep}_{bi}"
                pdv = predict(model, seq_dv, args.task, 2 * args.batch_tokens)
                pte = predict(model, seq_te, args.task, 2 * args.batch_tokens)
                np.save(f"{args.out}/dev_prob_{tag}.npy", pdv)
                np.save(f"{args.out}/test_prob_{tag}.npy", pte)
                r = dev_report(args.task, gold_dv[held], pdv[held])
                r.update(tag=tag, train_loss=tot / n, time=time.time() - t0,
                         peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
                log.append(r)
                print(f"[{tag}] {r['text']} ({time.time() - t0:.0f}s)", flush=True)
                json.dump({"args": vars(args), "log": log, "dev_ids": [d["id"] for d in dv],
                           "test_ids": [d["id"] for d in te]}, open(f"{args.out}/log.json", "w"), indent=1)
    if args.save:
        torch.save({"args": vars(args), "model": {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}},
                   args.save)
    print("done", time.time() - t0, "s")


if __name__ == "__main__":
    main()
