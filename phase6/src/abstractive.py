"""Abstractive summaries from our own Phase 1 language model (ROPEFIX_295M), fine-tuned to continue a case with
its summary.

Input of one example: the case text, shortened to --in_tokens tokens by keeping its highest-scored sentences (the
extractive model's cross-fitted scores, so a training case is never shortened by a model that saw its summary) in
their original order, then the separator "\n\nSUMMARY:\n", then the reference summary (cut to --out_tokens) and the
end-of-text token.  The next-token loss is taken on the summary tokens only.  The model was pre-trained on 2,048
tokens; its YaRN rotary scaling (extend_context) stretches it to --context tokens.

Generation: our own batched greedy decoding with the key/value cache; prompts are padded on the left and an
attention mask hides the padding.  A token that would repeat a 4-token sequence already generated is banned
(no-repeat n-gram), and the end token is banned until --min_frac of the predicted summary length is written.

References: YaRN context scaling, Peng et al. 2023, "YaRN: Efficient Context Window Extension of Large Language
Models"; no-repeat n-gram (trigram) blocking, Paulus, Xiong and Socher 2018, "A Deep Reinforced Model for
Abstractive Summarization"; selecting salient sentences before rewriting them, Chen and Bansal 2018, "Fast
Abstractive Summarization with Reinforce-Selected Sentence Rewriting".

  python abstractive.py train --out /workspace/models/cs781_phase6/abs_v1 --gpu 4
  python abstractive.py generate --ckpt /workspace/models/cs781_phase6/abs_v1/e2.pt --split dev --out ../runs/abs_v1
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

sys.path.insert(0, "/home/mani/cs781/Capstone/Phase4_CJPE/src")
from ft_doc import batches, load_model  # noqa: E402  (also puts the shared code on sys.path)
import summ  # noqa: E402

BACKBONE = "/workspace/models/cs781_phase1_runs/ROPEFIX_295M"
REPO = "/home/mani/cs781_phase1/Legal_LLM"
XFIT = "/home/mani/cs781/Capstone/Phase6_SUM/runs/ext_xfit5"
SEP = "\n\nSUMMARY:\n"
LENGTH = (1.533, 0.612)  # log summary words = a + b log case words (extractive.fit_length on train)


def shorten(doc, scores, max_words):
    """The case text cut to about max_words words: its best-scored sentences, in case order."""
    sents = summ.sentences(doc["document"])
    if sum(len(s.split()) for s in sents) <= max_words or scores is None:
        return " ".join(" ".join(sents).split()[:max_words])
    keep, words = [], 0
    for i in np.argsort(-np.asarray(scores)):
        w = len(sents[i].split())
        if words + w > max_words:
            continue
        keep.append(i)
        words += w
    return " ".join(sents[i] for i in sorted(keep))


def build(tok, docs, scores, in_tokens, out_tokens, with_summary=True):
    """(prompt ids, summary ids) per case.  Words are cut first at 0.75 x in_tokens (legal text averages ~1.25
    tokens per word), then the tokens are cut exactly."""
    texts = [shorten(d, scores.get(d["id"]), int(in_tokens * 0.75)) for d in docs]
    sep = tok.encode(SEP)
    prompts = [e.ids[:in_tokens] + sep for e in tok.encode_batch(texts)]
    if not with_summary:
        return prompts, None
    eot = tok.token_to_id("<|endoftext|>")
    sums = []
    for e in tok.encode_batch([d["summary"].strip() for d in docs]):
        sums.append(e.ids[:out_tokens] if len(e.ids) >= out_tokens else e.ids + [eot])
    return prompts, sums


def forward_loss(lm, ids, tgt_mask, grad_ckpt):
    """Next-token loss on the positions marked in tgt_mask (the summary tokens)."""
    cos, sin = lm.rotary(ids, ids.shape[1], offset=0)
    x = lm.token(ids)
    for layer in lm.layers:
        if grad_ckpt:
            x = checkpoint(lambda x_, l=layer: l(x_, cos, sin, kv_cache=None)[0], x, use_reentrant=False)
        else:
            x = layer(x, cos, sin, kv_cache=None)[0]
    sel = tgt_mask[:, 1:].bool()
    h = lm.norm(x[:, :-1][sel])
    return F.cross_entropy(lm.lm_head(h).float(), ids[:, 1:][sel])


def pad_pairs(prompts, sums, batch):
    t = max(len(prompts[i]) + len(sums[i]) for i in batch)
    ids = torch.zeros(len(batch), t, dtype=torch.long)
    m = torch.zeros(len(batch), t)
    for r, i in enumerate(batch):
        p, s = prompts[i], sums[i]
        ids[r, :len(p) + len(s)] = torch.tensor(p + s)
        m[r, len(p):len(p) + len(s)] = 1
    return ids.cuda(), m.cuda()


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    docs = summ.load("all" if args.all else "train")
    dev = summ.load("dev")[:args.n_dev_loss]
    scores = json.load(open(f"{XFIT}/xfit_scores.json"))
    lm, tok, _ = load_model(args.backbone, REPO)
    lm.extend_context(args.context)
    if args.lorma_rank:  # LoRMA study: multiplicative low-rank adapters, only on the chosen linear layers
        sys.path.insert(0, "/home/mani/cs781/Capstone/Phase2_LNER/src")
        import adapters
        targets = adapters.FFN if args.lorma_ffn_only else adapters.TARGETS
        adapters.add_adapters(lm, "lorma", args.lorma_rank, args.lorma_alpha, targets=targets)
    lm = lm.cuda()
    print(f"trainable parameters {sum(p.numel() for p in lm.parameters() if p.requires_grad) / 1e6:.2f}M", flush=True)
    P, S = build(tok, docs, scores, args.in_tokens, args.out_tokens)
    Pd, Sd = build(tok, dev, scores, args.in_tokens, args.out_tokens)
    lengths = [len(p) + len(s) for p, s in zip(P, S)]
    print(f"{len(P)} examples, {sum(lengths) / 1e6:.1f}M tokens, {sum(len(s) for s in S) / 1e6:.1f}M summary "
          f"tokens ({time.time() - t0:.0f}s)", flush=True)

    @torch.no_grad()
    def dev_loss():
        lm.eval()
        tot, n = 0.0, 0
        dl = [len(p) + len(s) for p, s in zip(Pd, Sd)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for b in batches(dl, sorted(range(len(Pd)), key=lambda i: dl[i]), args.batch_tokens):
                ids, m = pad_pairs(Pd, Sd, b)
                k = m[:, 1:].sum().item()
                tot, n = tot + forward_loss(lm, ids, m, False).item() * k, n + k
        lm.train()
        return tot / n

    log = {"args": vars(args), "dev_loss": [dev_loss()]}
    print(f"dev summary loss before: {log['dev_loss'][0]:.4f}", flush=True)
    decay = [p for n, p in lm.named_parameters() if p.dim() >= 2 and p.requires_grad]
    other = [p for n, p in lm.named_parameters() if p.dim() < 2 and p.requires_grad]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def epoch_batches():
        idx = sorted(range(len(P)), key=lambda i: lengths[i] + random.random() * 64)
        bs = list(batches(lengths, idx, args.batch_tokens))
        random.shuffle(bs)
        return bs

    n_steps = args.epochs * len(epoch_batches())
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / n_steps))))
    lm.train()
    step, tot, n = 0, 0.0, 0
    for ep in range(1, args.epochs + 1):
        for b in epoch_batches():
            ids, m = pad_pairs(P, S, b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forward_loss(lm, ids, m, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tot, n = tot + loss.item(), n + 1
            if step % 50 == 0:
                print(f"step {step}/{n_steps} train loss {tot / n:.4f} ({time.time() - t0:.0f}s, "
                      f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB)", flush=True)
                tot, n = 0.0, 0
        log["dev_loss"].append(dev_loss())
        print(f"epoch {ep}: dev summary loss {log['dev_loss'][-1]:.4f}", flush=True)
        ck = torch.load(f"{args.backbone}/best.pt", map_location="cpu", weights_only=False)
        state = lm.state_dict()
        if args.lorma_rank:  # fold the adapters into the weights: the saved model is a plain model again
            import copy
            state = adapters.merge_adapters(copy.deepcopy(lm)).state_dict()
        torch.save({"model_config": ck["model_config"], "model": state, "context": args.context,
                    "in_tokens": args.in_tokens, "out_tokens": args.out_tokens}, f"{args.out}/e{ep}.pt")
        json.dump(log, open(f"{args.out}/train_log.json", "w"), indent=1)
    print("done", time.time() - t0)


# ------------------------------------------------------------------------------------------------ generation
def step_forward(lm, ids, offset, caches, key_ok):
    """One forward pass over new tokens `ids` (b, t) given the caches; key_ok (b, T) marks real (non-pad) keys
    among all T = offset + t positions.  Returns the last position's logits and the new caches."""
    b, t = ids.shape
    cos, sin = lm.rotary(ids, t, offset=offset)
    T = offset + t
    q_pos = torch.arange(offset, T, device=ids.device)[:, None]
    k_pos = torch.arange(T, device=ids.device)[None, :]
    mask = (k_pos <= q_pos)[None, None] & key_ok[:, None, None, :]
    mask = mask | (k_pos == q_pos)[None, None]  # padding positions see themselves (avoids empty rows)
    x = lm.token(ids)
    new = []
    for i, layer in enumerate(lm.layers):
        a, kv = layer.attn(layer.attn_norm(x), cos, sin, kv_cache=caches[i] if caches else None, attn_mask=mask)
        x = x + a
        x = x + layer.ffn(layer.ffn_norm(x))
        new.append(kv)
    return lm.lm_head(lm.norm(x[:, -1])).float(), new


@torch.no_grad()
def generate(lm, prompts, max_new, min_new, eot, no_repeat=4, rep_penalty=1.0):
    """Greedy decoding for a batch of prompts (lists of ids); max_new/min_new are per-prompt lists."""
    b, L = len(prompts), max(len(p) for p in prompts)
    ids = torch.zeros(b, L, dtype=torch.long, device="cuda")
    key_ok = torch.zeros(b, L, dtype=torch.bool, device="cuda")
    for r, p in enumerate(prompts):
        ids[r, L - len(p):] = torch.tensor(p, device="cuda")
        key_ok[r, L - len(p):] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, caches = step_forward(lm, ids, 0, None, key_ok)
    out = [[] for _ in range(b)]
    seen = [dict() for _ in range(b)]
    done = [False] * b
    pos = L
    for t in range(max(max_new)):
        logits[:, eot] = torch.where(torch.tensor([t < m for m in min_new], device="cuda"),
                                     torch.full_like(logits[:, eot], -1e9), logits[:, eot])
        if rep_penalty != 1.0:
            for r in range(b):
                if out[r]:
                    u = torch.tensor(sorted(set(out[r][-200:])), device="cuda")
                    v = logits[r, u]
                    logits[r, u] = torch.where(v > 0, v / rep_penalty, v * rep_penalty)
        if no_repeat:
            for r in range(b):
                key = tuple(out[r][-(no_repeat - 1):])
                if len(key) == no_repeat - 1 and key in seen[r]:
                    logits[r, list(seen[r][key])] = -1e9
        nxt = logits.argmax(-1).tolist()
        for r in range(b):
            if done[r]:
                nxt[r] = eot
                continue
            out[r].append(nxt[r])
            if no_repeat and len(out[r]) >= no_repeat:
                key = tuple(out[r][-no_repeat:-1])
                seen[r].setdefault(key, set()).add(nxt[r])
            if nxt[r] == eot or len(out[r]) >= max_new[r]:
                done[r] = True
        if all(done):
            break
        key_ok = torch.cat([key_ok, torch.ones(b, 1, dtype=torch.bool, device="cuda")], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, caches = step_forward(lm, torch.tensor(nxt, device="cuda")[:, None], pos, caches, key_ok)
        pos += 1
    return [[x for x in o if x != eot] for o in out]


def run_generate(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    lm, tok, _ = load_model(BACKBONE, REPO)
    lm.extend_context(ck["context"])
    lm.load_state_dict(ck["model"])
    lm = lm.cuda().eval()
    docs = summ.load(args.split)
    if args.limit:
        docs = docs[:args.limit]
    scores = json.load(open(f"{XFIT}/xfit_scores.json")) if args.split != "test" else \
        json.load(open(f"{XFIT}/test_scores_all.json"))
    P, _ = build(tok, docs, scores, ck["in_tokens"], ck["out_tokens"], with_summary=False)
    eot = tok.token_to_id("<|endoftext|>")
    # predicted summary length in tokens (1.3 tokens per word), used for the minimum and maximum length
    pred = [1.3 * math.exp(LENGTH[0] + LENGTH[1] * math.log(len(d["document"].split()) + 1)) for d in docs]
    max_new = [min(ck["out_tokens"], ck["context"] - len(p), int(args.max_frac * x)) for p, x in zip(P, pred)]
    min_new = [min(m - 1, int(args.min_frac * x)) for m, x in zip(max_new, pred)]
    order = sorted(range(len(docs)), key=lambda i: -len(P[i]))
    outs = [None] * len(docs)
    for s in range(0, len(order), args.batch):
        b = order[s:s + args.batch]
        res = generate(lm, [P[i] for i in b], [max_new[i] for i in b], [min_new[i] for i in b], eot,
                       args.no_repeat, args.rep_penalty)
        for i, o in zip(b, res):
            outs[i] = " ".join(tok.decode(o).split())
        print(f"{s + len(b)}/{len(docs)} ({time.time() - t0:.0f}s)", flush=True)
    summaries = {d["id"]: o for d, o in zip(docs, outs)}
    tag = f"{args.split}_summaries{args.tag}.json"
    json.dump(summaries, open(f"{args.out}/{tag}", "w"))
    if args.split == "dev":
        r = summ.mean_scores([summaries[d["id"]] for d in docs], [d["summary"] for d in docs])
        words = np.mean([len(o.split()) for o in outs])
        print("dev", r, f"mean words {words:.0f}", flush=True)
        json.dump({"args": vars(args), "dev": r, "mean_words": words}, open(f"{args.out}/dev_result{args.tag}.json", "w"),
                  indent=1)
    print("done", time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--backbone", default=BACKBONE)
    t.add_argument("--all", action="store_true", help="train on train + dev")
    t.add_argument("--context", type=int, default=4096)
    t.add_argument("--in_tokens", type=int, default=2560)
    t.add_argument("--out_tokens", type=int, default=1500)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--batch_tokens", type=int, default=16384)
    t.add_argument("--n_dev_loss", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lorma_rank", type=int, default=0, help="LoRMA study: adapter rank (0 = train every weight)")
    t.add_argument("--lorma_alpha", type=float, default=32)
    t.add_argument("--lorma_ffn_only", action="store_true", help="adapters only on the feed-forward layers")
    g = sub.add_parser("generate")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--split", default="dev")
    g.add_argument("--out", required=True)
    g.add_argument("--tag", default="")
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--batch", type=int, default=16)
    g.add_argument("--min_frac", type=float, default=0.7)
    g.add_argument("--max_frac", type=float, default=1.8)
    g.add_argument("--no_repeat", type=int, default=4)
    g.add_argument("--rep_penalty", type=float, default=1.0)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else run_generate(args)


if __name__ == "__main__":
    main()
