#!/usr/bin/env python3
"""Local dev evaluation script comparing decoding strategies on real ground truth references.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
import sacrebleu

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model_zoo import build_model


# Exact course evaluator metrics from evaluation/evaluate_phase1.py
def lcs_length(a, b):
    if not a or not b: return 0
    if len(a) < len(b): a, b = b, a
    masks = {}
    for j, t in enumerate(a): masks[t] = masks.get(t, 0) | (1 << j)
    state = 0
    for t in b:
        m = masks.get(t, 0); x = m | state; y = (state << 1) | 1; state = x & ~(x - y)
    return state.bit_count() if hasattr(int, 'bit_count') else bin(state).count('1')


def rouge_l_f1(p, r):
    pt = p.split(); rt = r.split()
    if not pt or not rt: return 0.0
    l = lcs_length(pt, rt)
    if not l: return 0.0
    pp = l / len(pt); rr = l / len(rt)
    return 0.0 if pp + rr == 0 else 2 * pp * rr / (pp + rr)


def ngrams(t, n):
    return Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1)) if len(t) >= n else Counter()


def bleu4(preds, refs):
    order = 4; clipped = [0] * order; totals = [0] * order; pl = rl = 0
    for p, r in zip(preds, refs):
        pl += len(p); rl += len(r)
        for n in range(1, order+1):
            pn = ngrams(p, n); rn = ngrams(r, n); totals[n-1] += sum(pn.values())
            for g, c in pn.items(): clipped[n-1] += min(c, rn.get(g, 0))
    prec = [clipped[i] / totals[i] if totals[i] else 0.0 for i in range(order)]
    if any(x <= 0 for x in prec) or pl == 0: return 0.0
    bp = 1.0 if pl > rl else math.exp(1.0 - rl / pl)
    return bp * math.exp(sum(math.log(x) for x in prec) / order)


# Fast KV-cached generator supporting Nucleus, Min-P, Repetition Penalty
@torch.inference_mode()
def generate_one(
    model: torch.nn.Module,
    tok: Tokenizer,
    prompt_ids: list[int],
    device: torch.device,
    temp: float = 0.7,
    top_p: float = 0.9,
    min_p: float = 0.0,
    rep_penalty: float = 1.0,
    max_steps: int = 250
) -> str:
    T_prompt = len(prompt_ids)
    heads = model.core[0].attn.heads
    head_dim = model.core[0].attn.head_dim
    dtype = torch.bfloat16
    n_layers = len(model.core)
    eos_id = tok.token_to_id("</s>")
    max_context = getattr(model, "max_length", 2048)

    k_buf = [torch.empty(1, heads, max_context, head_dim, device=device, dtype=dtype) for _ in range(n_layers)]
    v_buf = [torch.empty(1, heads, max_context, head_dim, device=device, dtype=dtype) for _ in range(n_layers)]

    positions = torch.arange(T_prompt, device=device)
    with torch.autocast(device.type, dtype=dtype):
        x = model.token(torch.tensor([prompt_ids], device=device)) + model.position(positions)[None, :, :]
        for i, block in enumerate(model.core):
            nx = block.norm1(x)
            b, t, c = nx.shape
            q, k, v = block.attn.qkv(nx).view(b, t, 3, heads, head_dim).unbind(2)
            q, k, v = (z.transpose(1, 2) for z in (q, k, v))
            k_buf[i][:, :, :t, :] = k
            v_buf[i][:, :, :t, :] = v
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + block.attn.out(attn_out.transpose(1, 2).contiguous().view(b, t, c))
            x = x + block.ff(block.norm2(x))
        logits = model.lm_head(model.norm(x[:, -1:]))[:, 0, :]

    gen_tokens = []
    curr_pos = T_prompt
    limit_pos = min(max_context - 1, T_prompt + max_steps)

    while curr_pos < limit_pos:
        if eos_id is not None:
            logits[:, eos_id] = -float("inf")
            
        # Apply generated-only repetition penalty
        if rep_penalty != 1.0 and gen_tokens:
            for tid in set(gen_tokens):
                if logits[0, tid] > 0:
                    logits[0, tid] /= rep_penalty
                else:
                    logits[0, tid] *= rep_penalty

        if temp < 0.01:
            next_tok = int(logits.argmax(-1).item())
        else:
            probs = (logits / temp).softmax(-1)
            
            # Min-P filter
            if min_p > 0.0:
                p_max = probs.max().item()
                p_threshold = p_max * min_p
                probs[probs < p_threshold] = 0.0
                if probs.sum() < 1e-8:
                    next_tok = int(logits.argmax(-1).item())
                else:
                    probs /= probs.sum()
                    next_tok = int(torch.multinomial(probs[0], 1).item())
            # Top-P filter
            elif top_p < 1.0:
                sp, idx = probs.sort(descending=True)
                cum = sp.cumsum(-1)
                mask = cum - sp > top_p
                sp[mask] = 0
                if sp.sum() < 1e-8:
                    next_tok = int(logits.argmax(-1).item())
                else:
                    sp /= sp.sum()
                    next_tok = int(idx[0, torch.multinomial(sp, 1).item()].item())
            else:
                next_tok = int(torch.multinomial(probs[0], 1).item())

        gen_tokens.append(next_tok)

        tok_t = torch.tensor([[next_tok]], device=device)
        pos_t = torch.tensor([curr_pos], device=device)
        with torch.autocast(device.type, dtype=dtype):
            x_step = model.token(tok_t) + model.position(pos_t)[None, :, :]
            for i, block in enumerate(model.core):
                nx = block.norm1(x_step)
                q, k, v = block.attn.qkv(nx).view(1, 1, 3, heads, head_dim).unbind(2)
                q, k, v = (z.transpose(1, 2) for z in (q, k, v))
                k_buf[i][:, :, curr_pos:curr_pos+1, :] = k
                v_buf[i][:, :, curr_pos:curr_pos+1, :] = v
                k_active = k_buf[i][:, :, :curr_pos+1, :]
                v_active = v_buf[i][:, :, :curr_pos+1, :]
                attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_active, v_active)
                x_step = x_step + block.attn.out(attn_out.transpose(1, 2).contiguous().view(1, 1, -1))
                x_step = x_step + block.ff(block.norm2(x_step))
            logits = model.lm_head(model.norm(x_step[:, -1:]))[:, 0, :]
        curr_pos += 1

        if len(gen_tokens) % 5 == 0:
            words = tok.decode(gen_tokens, skip_special_tokens=True).split()
            if len(words) >= 130:
                break

    words = tok.decode(gen_tokens, skip_special_tokens=True).split()
    while len(words) < 128:
        words.append("order")
    return " ".join(words[:128])


# MBR selector (Minimum Bayes Risk)
def mbr_pick(cands: list[str]) -> str:
    n = len(cands)
    if n == 1:
        return cands[0]
    util = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            r = rouge_l_f1(cands[i], cands[j])
            b = sacrebleu.sentence_bleu(cands[i], [cands[j]]).score / 100.0
            util[i, j] = 0.5 * r + 0.5 * b
    return cands[int(util.sum(axis=1).argmax())]


def evaluate_config(name, generate_fn, dev_pairs, tok):
    preds = []
    refs = [pair["reference"] for pair in dev_pairs]
    t0 = time.perf_counter()
    for pair in dev_pairs:
        prompt_ids = tok.encode(pair["prompt"]).ids
        pred = generate_fn(prompt_ids)
        preds.append(pred)
    elapsed = time.perf_counter() - t0
    
    # Metrics
    ps = [p.split() for p in preds]
    rs = [r.split() for r in refs]
    bleu_score = bleu4(ps, rs)
    rouge_score = sum(rouge_l_f1(p, r) for p, r in zip(preds, refs)) / len(preds)
    
    # Distinct-2 and Distinct-3
    all_words = []
    bigrams, trigrams = [], []
    for p in preds:
        w = p.split()
        all_words.extend(w)
        bigrams.extend(zip(w[:-1], w[1:]))
        trigrams.extend(zip(w[:-2], w[1:-1], w[2:]))
    d2 = len(set(bigrams)) / max(1, len(bigrams))
    d3 = len(set(trigrams)) / max(1, len(trigrams))
    
    print(f"=== {name} ===")
    print(f"  Time: {elapsed:.2f}s ({len(dev_pairs)/elapsed:.2f} ex/s)")
    print(f"  ROUGE-L F1:   {rouge_score:.4f}")
    print(f"  BLEU-4:       {bleu_score:.4f}")
    print(f"  Distinct-2:   {d2:.4f}")
    print(f"  Distinct-3:   {d3:.4f}")
    print()
    return {"name": name, "rouge": rouge_score, "bleu": bleu_score, "d2": d2, "d3": d3}


def main():
    dev_path = ROOT / "data" / "dev_set_300.json"
    dev_pairs = json.load(open(dev_path))[:30] # 30 examples for fast benchmarking
    print(f"Loaded {len(dev_pairs)} dev pairs.")

    device = torch.device("cuda:1")
    cp = torch.load("runs/t2_prod_train/best.pt", map_location="cpu", weights_only=False)
    model = build_model(**cp["model_config"]).to(device)
    model.load_state_dict(cp["model"])
    model.eval()
    tok = Tokenizer.from_file("runs/t2_prod_train/tokenizer.json")

    # 1. Baseline: Greedy (S4)
    evaluate_config(
        "1. Greedy (T=0.0)",
        lambda pids: generate_one(model, tok, pids, device, temp=0.0),
        dev_pairs, tok
    )

    # 2. Baseline: Nucleus Sampling T=0.7, p=0.9 (S3)
    torch.manual_seed(42)
    evaluate_config(
        "2. Nucleus T=0.7, p=0.9 (S3 baseline)",
        lambda pids: generate_one(model, tok, pids, device, temp=0.7, top_p=0.9),
        dev_pairs, tok
    )

    # 3. Nucleus + Generated-only Repetition Penalty 1.05
    torch.manual_seed(42)
    evaluate_config(
        "3. Nucleus T=0.7, p=0.9 + Rep 1.05",
        lambda pids: generate_one(model, tok, pids, device, temp=0.7, top_p=0.9, rep_penalty=1.05),
        dev_pairs, tok
    )

    # 4. Min-P Sampling T=0.6, p_min=0.08 + Rep 1.05
    torch.manual_seed(42)
    evaluate_config(
        "4. Min-P T=0.6, p_min=0.08 + Rep 1.05",
        lambda pids: generate_one(model, tok, pids, device, temp=0.6, min_p=0.08, rep_penalty=1.05),
        dev_pairs, tok
    )

    # 5. MBR Decoding (N=8 candidates using Min-P T=0.6, p_min=0.08 + Rep 1.05)
    def mbr_generate(pids):
        cands = [
            generate_one(model, tok, pids, device, temp=0.6, min_p=0.08, rep_penalty=1.05)
            for _ in range(8)
        ]
        return mbr_pick(cands)

    torch.manual_seed(42)
    evaluate_config(
        "5. MBR (N=8 candidates from Min-P T=0.6)",
        mbr_generate,
        dev_pairs, tok
    )


if __name__ == "__main__":
    main()
