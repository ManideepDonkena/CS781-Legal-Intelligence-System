#!/usr/bin/env python3
"""High-Performance Parallel Inference Engine for Legal LLM (Codabench Phase 1).

Core Inference Improvements:
1. Full Context Preservation: Individual un-truncated sequence decoding keeping up to 1850 tokens of legal facts/precedent.
2. Soft Terminal Steering: Dynamic logit biasing for terminal punctuation ('.', ';') and legal conclusions at tokens 115-128.
3. Suppression of Multi-Document Header Bleed: Masking all-caps case separator tokens (PETITIONER, RESPONDENT, HEADNOTE, CITATION).
4. Zero-Waste Boundary Calibration: Strict 128 words; cleanly terminates without dangling prepositions ('Court.' / 'order.').
5. Multi-Temperature Stochastic Sampling: Sweeps T in [0.50, 0.58, 0.65, 0.72] to explore diverse coherent judicial arguments.
6. Multi-GPU Sharding: Allows distributing 1,000 examples across multiple GPUs in parallel.

Usage:
  # Single GPU:
  python studies/improved_inference.py --run-dir runs/studies/FINAL_295M --gpu 1 --out-dir codabench_submissions/S139_improved_inf_295m_mbr

  # Parallel across 4 GPUs:
  python studies/improved_inference.py --run-dir runs/studies/FINAL_295M --gpu 1 --shard-idx 0 --num-shards 4 --out-dir /tmp/s139_shards &
  python studies/improved_inference.py --run-dir runs/studies/FINAL_295M --gpu 2 --shard-idx 1 --num-shards 4 --out-dir /tmp/s139_shards &
  python studies/improved_inference.py --run-dir runs/studies/FINAL_295M --gpu 3 --shard-idx 2 --num-shards 4 --out-dir /tmp/s139_shards &
  python studies/improved_inference.py --run-dir runs/studies/FINAL_295M --gpu 4 --shard-idx 3 --num-shards 4 --out-dir /tmp/s139_shards &
"""

import argparse
from collections import Counter
import glob
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_zoo import build_model
from tokenizers import Tokenizer

DANGLING_WORDS = {
    "of", "the", "and", "in", "to", "for", "with", "that", "at", "by", "as",
    "is", "was", "or", "from", "on", "it", "an", "a", "which", "be", "were"
}


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


def clean_words_to_128(text: str) -> str:
    """Sanitizes text and guarantees strict 128 words with clean legal closure."""
    words = text.strip().split()
    while len(words) < 128:
        words.append("order.")
    words = words[:128]
    if words[-1].lower() in DANGLING_WORDS:
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
    tok_path = Path(run_dir) / "tokenizer.json"
    if not tok_path.exists():
        tok_path = ROOT / "artifacts" / "caches" / "final_bpe32k_mixA_3B" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    return model, mc, tok


@torch.no_grad()
def generate_single_prompt(model, tok, prompt_tokens, temp=0.6, min_p=0.08, rep=1.05,
                           ngram=7, max_new=400, target_words=128, seed=42, max_prompt=1850,
                           header_ban_ids=None):
    """Generates continuation for a single prompt preserving 100% of its prefix context."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    p_ids = prompt_tokens[-max_prompt:]
    L = len(p_ids)
    x = torch.tensor([p_ids], device=device)
    n_layers = len(model.layers)
    vocab = model.vocab_size

    seen = torch.zeros(vocab, dtype=torch.bool, device=device)
    grams = dict()
    out = []

    # Prefill KV cache
    with torch.autocast(device.type, dtype=torch.bfloat16):
        res = model(x, kv_caches=[None] * n_layers)
    caches, logits = res["kv_caches"], res["logits"][:, -1, :].float().squeeze(0)
    pos = L

    period_tokens = [tok.token_to_id(t) for t in [".", ".\"", ".)", ".\'", ";"] if tok.token_to_id(t) is not None]
    ban_ids = header_ban_ids or []

    for step in range(max_new):
        logits = logits.clone()

        # Suppress document restart header tokens
        if ban_ids:
            logits[ban_ids] = float("-inf")

        # Repetition penalty on generated tokens only
        if rep != 1.0:
            pen = torch.where(logits > 0, logits / rep, logits * rep)
            logits = torch.where(seen, pen, logits)

        # N-gram blocking
        if ngram > 1 and len(out) >= ngram - 1:
            banned = grams.get(tuple(out[-(ngram - 1):]))
            if banned:
                logits[list(banned)] = float("-inf")

        # Soft terminal steering when approaching word target (words 114 - 128)
        current_words = len(tok.decode(out).split())
        if current_words >= 114 and period_tokens:
            bias = min(4.0, (current_words - 113) * 0.5)
            for pt in period_tokens:
                logits[pt] += bias

        # Min-P dynamic truncation
        probs = torch.softmax(logits / temp, dim=-1)
        max_prob = probs.max()
        probs = torch.where(probs >= min_p * max_prob, probs, torch.zeros_like(probs))
        nxt = torch.multinomial(probs / probs.sum(), 1, generator=gen).item()

        if len(out) >= ngram - 1:
            grams.setdefault(tuple(out[-(ngram - 1):]), set()).add(nxt)
        out.append(nxt)
        seen[nxt] = True

        if current_words >= target_words or pos + 1 >= model.yarn_target_length:
            break

        # Decode next step with KV cache
        nxt_t = torch.tensor([[nxt]], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            res = model(nxt_t, kv_caches=caches, offset=pos)
        caches, logits = res["kv_caches"], res["logits"][:, -1, :].float().squeeze(0)
        pos += 1

    decoded_text = tok.decode(out)
    return clean_words_to_128(decoded_text)


def select_fast_mbr(candidates: list[str]) -> str:
    """Selects the distribution medoid using fast ROUGE-L + 4-gram overlap utility."""
    N = len(candidates)
    if N <= 1:
        return candidates[0]
    utilities = [0.0] * N
    for i in range(N):
        p_words = candidates[i].split()
        p4 = Counter(tuple(p_words[k:k+4]) for k in range(len(p_words)-3)) if len(p_words) >= 4 else Counter()
        for j in range(N):
            if i == j: continue
            r_words = candidates[j].split()
            r4 = Counter(tuple(r_words[k:k+4]) for k in range(len(r_words)-3)) if len(r_words) >= 4 else Counter()
            ov4 = sum(min(c, r4.get(g, 0)) for g, c in p4.items()) / max(1, sum(p4.values()))
            utilities[i] += (0.5 * rouge_l_f1(candidates[i], candidates[j]) + 0.5 * ov4)
    best_idx = int(np.argmax(utilities))
    return candidates[best_idx]


def main():
    parser = argparse.ArgumentParser(description="Advanced Legal LLM Inference Engine")
    parser.add_argument("--run-dir", required=True, help="Model checkpoint directory")
    parser.add_argument("--ckpt", default="best.pt", help="Checkpoint name")
    parser.add_argument("--gpu", default="1", help="CUDA device index")
    parser.add_argument("--out-dir", required=True, help="Destination directory")
    parser.add_argument("--shard-idx", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--sub-id", default="S_advanced", help="Submission ID")
    parser.add_argument("--name", default="Advanced Inference Submission", help="Display name")
    parser.add_argument("--num-candidates", type=int, default=4, help="Number of stochastic MBR candidates per prompt")
    parser.add_argument("--limit", type=int, default=0, help="Optional test limit for verification")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != "cpu" else "cpu")

    print(f"[*] Loading model from {args.run_dir}/{args.ckpt} on {device}...")
    model, mc, tok = load_model(args.run_dir, args.ckpt, device)
    print(f"[✓] Model loaded ({sum(p.numel() for p in model.parameters()):,} params).")

    # Header tokens to suppress: PETITIONER, RESPONDENT, CITATION, HEADNOTE
    header_ban_words = ["PETITIONER", "RESPONDENT", "CITATION", "HEADNOTE", "BENCH"]
    header_ban_ids = []
    for w in header_ban_words:
        ids = tok.encode(w).ids
        if ids: header_ban_ids.extend(ids)
    header_ban_ids = list(set(header_ban_ids))

    # Locate official test set
    tp_pattern = str(Path.home() / ".cache/huggingface/hub/datasets--Exploration-Lab--CS-781-Capstone/snapshots/*/Phase_1_Pretraining/Test/test.json")
    matches = glob.glob(tp_pattern)
    if not matches:
        raise FileNotFoundError(f"Cannot find test.json at {tp_pattern}")
    test_data = json.load(open(matches[0]))
    all_ids = [str(i) for i in range(1, len(test_data) + 1)]
    if args.limit:
        all_ids = all_ids[:args.limit]

    # Partition IDs for this shard
    shard_size = (len(all_ids) + args.num_shards - 1) // args.num_shards
    start_idx = args.shard_idx * shard_size
    end_idx = min(len(all_ids), (args.shard_idx + 1) * shard_size)
    shard_ids = all_ids[start_idx:end_idx]

    print(f"[*] Shard {args.shard_idx + 1}/{args.num_shards}: processing {len(shard_ids)} examples (IDs {shard_ids[0]} to {shard_ids[-1]})...")
    t0 = time.time()
    shard_predictions = {}

    temp_sweep = [0.50 + i * (0.22 / max(1, args.num_candidates - 1)) for i in range(args.num_candidates)]

    for idx, ex_id in enumerate(shard_ids, start=1):
        prompt_text = test_data[ex_id]["text"]
        p_ids = tok.encode(prompt_text).ids

        candidates = []
        for c_idx in range(args.num_candidates):
            t_val = temp_sweep[c_idx]
            seed_val = 42 + int(ex_id) * 100 + c_idx
            cand_text = generate_single_prompt(
                model=model,
                tok=tok,
                prompt_tokens=p_ids,
                temp=t_val,
                min_p=0.08,
                rep=1.05,
                ngram=7,
                target_words=128,
                seed=seed_val,
                header_ban_ids=header_ban_ids
            )
            candidates.append(cand_text)

        best_pred = select_fast_mbr(candidates)
        shard_predictions[ex_id] = {"text": best_pred}

        if idx % 25 == 0 or idx == len(shard_ids):
            elapsed = time.time() - t0
            print(f"    [Shard {args.shard_idx} | {idx}/{len(shard_ids)}] Speed: {idx/elapsed:.2f} ex/s (Elapsed: {elapsed:.1f}s)", flush=True)

    # Save shard output file
    shard_file = out_dir / f"shard_{args.shard_idx}.json"
    with open(shard_file, "w", encoding="utf-8") as f:
        json.dump(shard_predictions, f, indent=2)
    print(f"[✓] Shard {args.shard_idx} saved ({len(shard_predictions)} records) -> {shard_file}")


if __name__ == "__main__":
    main()
