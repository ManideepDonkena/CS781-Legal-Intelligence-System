"""Phase 7 LoRMA study (offline, dev set only, nothing uploaded): turn the Phase 1 model into a retriever.

Idea: the frozen Phase 1 model was trained to predict the next word, not to say "these two cases are related", so
its mean-pooled states are a weak search vector.  I teach it that with LoRMA adapters (the backbone stays frozen)
and a contrastive loss on the training queries:

  * each training query is paired with one case it cites (5,308 pairs from 827 queries)
  * negatives: the other cases in the batch, plus one hard negative per query, a case that BM25 ranks high for
    that query but that it does not cite (so the model must learn more than word overlap)
  * a case in the batch that the query does cite is never used as a negative (masked out)
  * loss: cross-entropy over cosine similarities / temperature (InfoNCE, van den Oord et al. 2018; hard negatives
    as in DPR, Karpukhin et al. 2020)

Evaluation on dev (118 queries, 1,023 candidates; the query itself is never returned):
  frozen model vs LoRMA-trained model alone, and each fused with the Phrase-BM25 scores of advanced_pcr.py
  using a FIXED weight (0.3 dense + 0.7 BM25, both min-max scaled per query), set before looking at dev.

  CUDA_VISIBLE_DEVICES=1 python lorma_retriever.py --out ../runs/lorma
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/mani/cs781/Capstone/Phase4_CJPE/src")
sys.path.insert(0, "/home/mani/cs781/Capstone/Phase2_LNER/src")
import adapters  # noqa: E402
import pcr  # noqa: E402
from ft_doc import load_model  # noqa: E402


def doc_text(d):
    return " ".join(d["text"]) if isinstance(d["text"], list) else d["text"]


def bm25_top(queries, cands, k=30, k1=1.2, b=0.75):
    """Plain word BM25 on the first 1,500 words, only used to find hard negatives for training."""
    def words(t):
        return [w for w in pcr.tokenize([t], include_sections=False)][:1500]
    c_ids = list(cands)
    c_tf = [Counter(words(doc_text(cands[c]))) for c in c_ids]
    lens = np.array([sum(t.values()) for t in c_tf], dtype=float)
    avg = lens.mean()
    df = Counter(w for t in c_tf for w in t)
    n = len(c_ids)
    index = defaultdict(list)
    for j, t in enumerate(c_tf):
        for w, f in t.items():
            index[w].append((j, f))
    out = {}
    for qid, q in queries.items():
        s = np.zeros(n)
        for w in set(words(doc_text(q))):
            if w not in index:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            for j, f in index[w]:
                s[j] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[j] / avg))
        out[qid] = [c_ids[j] for j in np.argsort(-s)[:k + 1] if c_ids[j] != qid][:k]
    return out


def encode(lm, seqs, device="cuda"):
    """Mean of the final normalised states over the real tokens, then L2-normalised."""
    width = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), width, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), width, device=device)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, device=device)
        mask[i, :len(s)] = 1
    cos, sin = lm.rotary(ids, width, offset=0)
    x = lm.token(ids)
    for layer in lm.layers:
        x = layer(x, cos, sin, kv_cache=None)[0]
    h = lm.norm(x).float()
    return F.normalize((h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True), dim=-1)


@torch.no_grad()
def embed_all(lm, seqs, batch=32):
    lm.eval()
    out = []
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), batch):
        b = order[s:s + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append((b, encode(lm, [seqs[i] for i in b]).cpu()))
    emb = torch.zeros(len(seqs), out[0][1].shape[1])
    for b, e in out:
        emb[b] = e
    return emb


def minmax(v):
    return (v - v.min()) / max(v.max() - v.min(), 1e-9)


def evaluate(lm, tok_seq, dq, dc, phrase, w_dense=0.3):
    q_ids, c_ids = list(dq), list(dc)
    qe = embed_all(lm, [tok_seq[("q", q)] for q in q_ids])
    ce = embed_all(lm, [tok_seq[("c", c)] for c in c_ids])
    sim = (qe @ ce.T).numpy()
    res = {}
    for name, w in (("dense", 1.0), ("bm25", 0.0), ("fused", w_dense)):
        preds = {}
        for i, q in enumerate(q_ids):
            d = minmax(sim[i])
            p = minmax(np.array([phrase[q].get(c, 0.0) for c in c_ids]))
            s = w * d + (1 - w) * p
            s[[j for j, c in enumerate(c_ids) if c == q]] = -1e9  # never return the query itself
            preds[q] = [c_ids[j] for j in np.argsort(-s)[:pcr.K]]
        r = pcr.evaluate_predictions(dq, preds)
        res[name] = {"f1": r["primary_metric_1"], "p": r["secondary_metric_1"], "r": r["secondary_metric_2"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="/workspace/models/cs781_phase1_runs/ROPEFIX_295M")
    ap.add_argument("--repo", default="/home/mani/cs781_phase1/Legal_LLM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=8, help="query-positive pairs per batch")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="write the trained LoRMA adapter weights (A, B of every layer) here")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    tq, tc = pcr.load_split("train")
    dq, dc = pcr.load_split("dev")
    phrase = json.load(open(HERE.parent / "runs" / "advanced_pcr" / "dev_phrase_scores.json"))
    lm, tok, _ = load_model(args.backbone, args.repo)
    keys = [("q", k) for k in tq] + [("c", k) for k in tc] + [("q", k) for k in dq] + [("c", k) for k in dc]
    src = {"q": {**tq, **dq}, "c": {**tc, **dc}}
    texts = [" ".join(doc_text(src[t][k]).split()[:args.max_tokens]) for t, k in keys]
    tok_seq = {key: (e.ids[:args.max_tokens] or [0]) for key, e in
               zip(keys, tok.encode_batch(texts, add_special_tokens=False))}
    print(f"tokenised {len(keys)} documents ({time.time() - t0:.0f}s)", flush=True)
    hard = bm25_top(tq, tc)
    print(f"BM25 hard negatives ready ({time.time() - t0:.0f}s)", flush=True)

    lm = lm.cuda()
    log = {"args": vars(args), "frozen": evaluate(lm, tok_seq, dq, dc, phrase)}
    print("frozen", json.dumps(log["frozen"]), flush=True)

    lm = lm.cpu()
    params = adapters.add_adapters(lm, "lorma", args.rank, args.alpha)
    lm = lm.cuda()
    log["trainable_params"] = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    pairs = [(q, c) for q, v in tq.items() for c in v["relevant_candidates"] if c in tc]
    gold = {q: set(v["relevant_candidates"]) for q, v in tq.items()}
    n_steps = args.epochs * (len(pairs) // args.pairs)
    warm = max(1, n_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / n_steps))))
    torch.cuda.reset_peak_memory_stats()
    step, t1 = 0, time.time()
    for ep in range(args.epochs):
        random.shuffle(pairs)
        lm.train()
        tot = 0.0
        for s in range(0, len(pairs) - args.pairs + 1, args.pairs):
            b = pairs[s:s + args.pairs]
            qs = [q for q, _ in b]
            docs = [c for _, c in b]
            for q in qs:  # one BM25 hard negative per query (not cited by any query in the batch)
                pool = [c for c in hard[q] if all(c not in gold[x] for x in qs) and c not in docs]
                docs.append(random.choice(pool[:10]) if pool else random.choice(list(tc)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qe = encode(lm, [tok_seq[("q", q)] for q in qs])
                de = encode(lm, [tok_seq[("c", c)] for c in docs])
            logits = qe @ de.T / args.temp
            false_neg = torch.tensor([[c in gold[q] and j != i for j, c in enumerate(docs)] for i, q in enumerate(qs)],
                                     device="cuda")
            loss = F.cross_entropy(logits.masked_fill(false_neg, -1e4), torch.arange(len(qs), device="cuda"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            tot += loss.item()
            if step % 50 == 0:
                print(f"step {step}/{n_steps} loss {tot / 50:.3f} ({time.time() - t0:.0f}s, "
                      f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB)", flush=True)
                tot = 0.0
    log["train_minutes"] = (time.time() - t1) / 60
    log["peak_train_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    log["lorma"] = evaluate(lm, tok_seq, dq, dc, phrase)
    print("lorma", json.dumps(log["lorma"]), flush=True)
    json.dump(log, open(f"{args.out}/retriever_log.json", "w"), indent=1)
    if args.save:  # only the trained adapter matrices; load the Phase 1 model, add_adapters(...), then load these
        torch.save({"args": vars(args), "adapters": {n: p.detach().cpu() for n, p in lm.named_parameters()
                                                     if p.requires_grad}}, args.save)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
