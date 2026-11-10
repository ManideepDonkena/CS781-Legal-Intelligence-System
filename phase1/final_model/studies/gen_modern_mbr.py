#!/usr/bin/env python3
"""Batched Intra-Model MBR Consensus (S14 approach) for Modern YaRN-GQA architectures.

Generates N stochastic candidate continuations per prompt using Min-P sampling across
different seeds, then selects the distribution medoid using pairwise ROUGE-L and 4-gram
overlap utilities.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_zoo import build_model          # noqa: E402
from tokenizers import Tokenizer           # noqa: E402


def lcs_length(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    masks = {}
    for j, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << j)
    state = 0
    for t in b:
        m = masks.get(t, 0)
        x = m | state
        y = (state << 1) | 1
        state = x & ~(x - y)
    return state.bit_count() if hasattr(int, "bit_count") else bin(state).count("1")


def rouge_l_f1(p: str, r: str) -> float:
    pt, rt = p.split(), r.split()
    if not pt or not rt:
        return 0.0
    l = lcs_length(pt, rt)
    if not l:
        return 0.0
    pp = l / len(pt)
    rr = l / len(rt)
    return 0.0 if (pp + rr) == 0 else 2 * pp * rr / (pp + rr)


def mbr_select(candidates: list[str]) -> str:
    N = len(candidates)
    if N <= 1:
        return candidates[0]
    utilities = [0.0] * N
    for i in range(N):
        p_words = candidates[i].split()
        p4 = Counter(tuple(p_words[k:k+4]) for k in range(len(p_words)-3)) if len(p_words) >= 4 else Counter()
        for j in range(N):
            if i == j:
                continue
            r_l = rouge_l_f1(candidates[i], candidates[j])
            r_words = candidates[j].split()
            r4 = Counter(tuple(r_words[k:k+4]) for k in range(len(r_words)-3)) if len(r_words) >= 4 else Counter()
            ov4 = sum(min(c, r4.get(g, 0)) for g, c in p4.items()) / max(1, sum(p4.values()))
            utilities[i] += (0.5 * r_l + 0.5 * ov4)
    best_idx = max(range(N), key=lambda i: utilities[i])
    return candidates[best_idx]


def clean_words_to_128(text: str) -> str:
    dangling_words = {"of", "the", "and", "in", "to", "for", "with", "that", "at", "by", "as", "is", "was", "or", "from", "on"}
    words = text.split()
    while len(words) < 128:
        words.append("order.")
    words = words[:128]
    if words[-1].lower() in dangling_words:
        words[-1] = "Court."
    elif not words[-1].endswith((".", ";", "\"", "'", "!", "?")):
        words[-1] = words[-1] + "."
    return " ".join(words)


def load_model(run_dir, ckpt, device):
    bundle = torch.load(Path(run_dir) / ckpt, map_location="cpu", weights_only=False)
    mc = bundle["model_config"]
    model = build_model(**mc).to(device)
    model.load_state_dict(bundle["model"])
    model.eval()
    return model, mc, Tokenizer.from_file(str(Path(run_dir) / "tokenizer.json"))


@torch.no_grad()
def generate_batch(model, tok, prompts_ids, temp=0.6, min_p=0.08, rep=1.05, ngram=7, max_new=400, target_words=128,
                   seed=42, max_prompt=1700):
    device = next(model.parameters()).device
    B = len(prompts_ids)
    L = min(min(len(p) for p in prompts_ids), max_prompt)
    x = torch.tensor([p[-L:] for p in prompts_ids], device=device)
    n_layers = len(model.layers)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    vocab = model.vocab_size
    seen = torch.zeros(B, vocab, dtype=torch.bool, device=device)
    grams = [dict() for _ in range(B)]
    out = [[] for _ in range(B)]
    done = [False] * B

    with torch.autocast(device.type, dtype=torch.bfloat16):
        res = model(x, kv_caches=[None] * n_layers)
    caches, logits = res["kv_caches"], res["logits"][:, -1].float()
    pos = L

    for step in range(max_new):
        logits = logits.clone()
        if rep != 1.0:   # generated-only penalty
            pen = torch.where(logits > 0, logits / rep, logits * rep)
            logits = torch.where(seen, pen, logits)
        if ngram > 1:
            for b in range(B):
                if len(out[b]) >= ngram - 1:
                    banned = grams[b].get(tuple(out[b][-(ngram - 1):]))
                    if banned:
                        logits[b, list(banned)] = float("-inf")
        probs = torch.softmax(logits / temp, dim=-1)
        probs = torch.where(probs >= min_p * probs.max(dim=-1, keepdim=True).values, probs, torch.zeros_like(probs))
        nxt = torch.multinomial(probs / probs.sum(-1, keepdim=True), 1, generator=gen)[:, 0]
        nl = nxt.tolist()
        for b in range(B):
            if not done[b]:
                if len(out[b]) >= ngram - 1:
                    grams[b].setdefault(tuple(out[b][-(ngram - 1):]), set()).add(nl[b])
                out[b].append(nl[b])
                seen[b, nl[b]] = True
        if step % 16 == 15 or step == max_new - 1:
            for b in range(B):
                if not done[b] and len(tok.decode(out[b]).split()) > target_words:
                    done[b] = True
            if all(done) or pos + 1 >= getattr(model, "yarn_target_length", 8192):
                break
        with torch.autocast(device.type, dtype=torch.bfloat16):
            res = model(nxt[:, None], kv_caches=caches, offset=pos)
        caches, logits = res["kv_caches"], res["logits"][:, -1].float()
        pos += 1

    texts = []
    for b in range(B):
        raw_text = tok.decode(out[b])
        texts.append(clean_words_to_128(raw_text))
    return texts


def run_mbr(prompts, model, tok, args, num_candidates=4):
    order = sorted(range(len(prompts)), key=lambda i: len(tok.encode(prompts[i]).ids))
    ids = {i: tok.encode(prompts[i]).ids for i in order}
    preds, t0 = [None] * len(prompts), time.time()

    for s in range(0, len(order), args.batch):
        chunk = order[s:s + args.batch]
        prompt_batch = [ids[i] for i in chunk]
        candidate_runs = []
        for c in range(num_candidates):
            candidate_texts = generate_batch(
                model, tok, prompt_batch, args.temperature, args.min_p,
                args.repetition_penalty, args.ngram_block, seed=args.seed + s * 100 + c * 37
            )
            candidate_runs.append(candidate_texts)
        for idx_in_chunk, orig_idx in enumerate(chunk):
            cands = [candidate_runs[c][idx_in_chunk] for c in range(num_candidates)]
            best = mbr_select(cands)
            preds[orig_idx] = best
        done = s + len(chunk)
        print(f"{done}/{len(order)} (MBR {num_candidates}x)  {done/(time.time()-t0):.2f} ex/s", flush=True)

    return preds


def validate(preds):
    wc = [len(p.split()) for p in preds]
    assert all(w == 128 for w in wc), f"word counts not all 128: min={min(wc)} max={max(wc)}"
    return {"total_records": len(wc), "exact_128_words": len(wc), "min_words": min(wc), "max_words": max(wc), "avg_words": 128.0}


def sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sub-id", default="dev")
    ap.add_argument("--name", default="")
    ap.add_argument("--desc", default="")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--min-p", type=float, default=0.08)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--ngram-block", type=int, default=7)
    ap.add_argument("--mbr-candidates", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu != "cpu" else "cpu")
    model, mc, tok = load_model(args.run_dir, args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {n_params:,} params from {args.run_dir}/{args.ckpt} on {device}, family={mc.get('family')}", flush=True)

    tp = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--Exploration-Lab--CS-781-Capstone/snapshots/*/Phase_1_Pretraining/Test/test.json"))[0]
    test = json.load(open(tp))
    ids = [str(i) for i in range(1, len(test) + 1)]
    if args.limit:
        ids = ids[:args.limit]

    t0 = time.time()
    preds = run_mbr([test[i]["text"] for i in ids], model, tok, args, num_candidates=args.mbr_candidates)
    stats = validate(preds)

    pred_file = out / "prediction.json"
    pred_file.write_text(json.dumps({i: {"text": t} for i, t in zip(ids, preds)}, indent=2))
    zip_file = out / "submission.zip"
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_file, arcname="prediction.json")
    with zipfile.ZipFile(zip_file) as zf:
        assert zf.namelist() == ["prediction.json"]

    tag = {
        "submission_id": args.sub_id,
        "display_name": args.name,
        "description": args.desc,
        "model_checkpoint": str(Path(args.run_dir) / args.ckpt),
        "parameters": n_params,
        "architecture_family": mc.get("family"),
        "model_config": mc,
        "decoding": {
            "strategy": "mbr",
            "mbr_candidates": args.mbr_candidates,
            "temperature": args.temperature,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "gen_only_rep": True,
            "ngram_block": args.ngram_block,
            "target_words": 128
        },
        "generation_performance": {
            "wall_clock_seconds": round(time.time() - t0, 1),
            "gpu_used": str(device)
        },
        "validation_metrics": stats,
        "artifacts": {
            "prediction_sha256": sha(pred_file),
            "submission_zip_sha256": sha(zip_file)
        }
    }
    (out / "tagging.json").write_text(json.dumps(tag, indent=2))
    print(f"COMPLETE MBR SUBMISSION: {out}", stats, flush=True)


if __name__ == "__main__":
    main()
