#!/usr/bin/env python3
"""Batched KV-cache generation for the *modern* (yarn_gqa) models + Codabench submission packaging.

`run_generate_submissions.py` is hard-wired to the legacy `model.core[...]` Transformer, so the modern
architecture needs its own path. Decoding defaults reproduce the best config found in the earlier
decoding ablation (S10): Min-P (T=0.6, min_p=0.08), generated-only repetition penalty 1.05, 7-gram blocking.

Prompts of similar token length are batched; inside a batch every prompt is left-truncated to the shortest
one (a handful of the *oldest* prompt tokens are dropped; the end of the prompt is untouched).

Test submission:
  python studies/gen_modern.py --run-dir runs/studies/FINAL --sub-id S101_final --name "..." --out-dir codabench_submissions/S101_final
Dev scoring on the frozen held-out pairs of a cache (official scorer):
  python studies/gen_modern.py --run-dir runs/studies/X --dev-cache bpe50k_mixA_600M --out-dir /tmp/dev_X
"""
import argparse, glob, hashlib, json, subprocess, sys, time, zipfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_zoo import build_model          # noqa: E402
from tokenizers import Tokenizer           # noqa: E402


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
    gen = torch.Generator(device=device); gen.manual_seed(seed)
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
        if rep != 1.0:   # generated-only penalty (prompt tokens are never penalised)
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
            if all(done) or pos + 1 >= model.yarn_target_length:
                break
        with torch.autocast(device.type, dtype=torch.bfloat16):
            res = model(nxt[:, None], kv_caches=caches, offset=pos)
        caches, logits = res["kv_caches"], res["logits"][:, -1].float()
        pos += 1
    texts = []
    for b in range(B):
        words = tok.decode(out[b]).split()
        if len(words) < target_words:                     # extremely rare; keep format valid
            words += ["the"] * (target_words - len(words))
        texts.append(" ".join(words[:target_words]))
    return texts


def run(prompts, model, tok, args):
    order = sorted(range(len(prompts)), key=lambda i: len(tok.encode(prompts[i]).ids))
    ids = {i: tok.encode(prompts[i]).ids for i in order}
    preds, t0 = [None] * len(prompts), time.time()
    for s in range(0, len(order), args.batch):
        chunk = order[s:s + args.batch]
        texts = generate_batch(model, tok, [ids[i] for i in chunk], args.temperature, args.min_p, args.repetition_penalty,
                               args.ngram_block, seed=args.seed + s)
        for i, t in zip(chunk, texts): preds[i] = t
        done = s + len(chunk)
        print(f"{done}/{len(order)}  {done/(time.time()-t0):.2f} ex/s", flush=True)
    return preds


def validate(preds):
    wc = [len(p.split()) for p in preds]
    assert all(w == 128 for w in wc), f"word counts not all 128: min={min(wc)} max={max(wc)}"
    return {"total_records": len(wc), "exact_128_words": len(wc), "min_words": min(wc), "max_words": max(wc)}


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--gpu", default="0"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sub-id", default="dev"); ap.add_argument("--name", default=""); ap.add_argument("--desc", default="")
    ap.add_argument("--dev-cache", help="score held-out pairs of this cache instead of the 1000 test prompts")
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--min-p", type=float, default=0.08)
    ap.add_argument("--repetition-penalty", type=float, default=1.05); ap.add_argument("--ngram-block", type=int, default=7)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu != "cpu" else "cpu")
    model, mc, tok = load_model(args.run_dir, args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {n_params:,} params, family={mc.get('family')}", flush=True)

    if args.dev_cache:
        c = ROOT / "artifacts" / "caches" / args.dev_cache
        prompts_d = json.loads((c / "held_out_prompts.json").read_text())
        keys = list(prompts_d)
        preds = run([prompts_d[k]["text"] for k in keys], model, tok, args)
        validate(preds)
        pred = {k: {"text": t} for k, t in zip(keys, preds)}
        (out / "prediction.json").write_text(json.dumps(pred, indent=1))
        r = subprocess.run([sys.executable, str(ROOT / "evaluation" / "evaluate_phase1.py"), "--truth",
                            str(c / "held_out_truth.json"), "--prediction", str(out / "prediction.json")],
                           capture_output=True, text=True)
        (out / "dev_scores.txt").write_text(r.stdout + r.stderr)
        print(r.stdout[-600:], r.stderr[-300:]); return

    tp = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--Exploration-Lab--CS-781-Capstone/snapshots/*/Phase_1_Pretraining/Test/test.json"))[0]
    test = json.load(open(tp))
    ids = [str(i) for i in range(1, len(test) + 1)]
    if args.limit: ids = ids[:args.limit]
    t0 = time.time()
    preds = run([test[i]["text"] for i in ids], model, tok, args)
    stats = validate(preds)
    pred_file = out / "prediction.json"
    pred_file.write_text(json.dumps({i: {"text": t} for i, t in zip(ids, preds)}, indent=2))
    zip_file = out / "submission.zip"
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_file, arcname="prediction.json")
    with zipfile.ZipFile(zip_file) as zf:
        assert zf.namelist() == ["prediction.json"]
    tag = {"submission_id": args.sub_id, "display_name": args.name, "description": args.desc,
           "model_checkpoint": str(Path(args.run_dir) / args.ckpt), "parameters": n_params,
           "architecture_family": mc.get("family"), "model_config": mc,
           "decoding": {"strategy": "min_p", "temperature": args.temperature, "min_p": args.min_p,
                        "repetition_penalty": args.repetition_penalty, "gen_only_rep": True,
                        "ngram_block": args.ngram_block, "target_words": 128,
                        "batching": f"sorted by length, batch={args.batch}, left-truncate to shortest prompt in batch"},
           "generation_performance": {"wall_clock_seconds": round(time.time() - t0, 1)},
           "validation_metrics": stats,
           "artifacts": {"prediction_sha256": sha(pred_file), "submission_zip_sha256": sha(zip_file)}}
    (out / "tagging.json").write_text(json.dumps(tag, indent=2))
    print("COMPLETE", stats, flush=True)


if __name__ == "__main__":
    main()
