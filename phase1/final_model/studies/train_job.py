#!/usr/bin/env python3
"""One study training run: build a preset from CLI flags, train via experiments/runner.py, then
post-evaluate the best checkpoint on the per-source validation files and append a result row.

Example:
  python studies/train_job.py --id G3_base_s1 --study G3 --cache bpe50k_mixA_600M --family yarn_gqa \
      --d-model 512 --layers 11 --heads 8 --kv-heads 2 --tokens 600000000 --lr 2e-3 --seed 1
"""
import argparse, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
CACHES = ROOT / "artifacts" / "caches"
RESULTS = ROOT / "results" / "results.jsonl"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--study", default="", help="study tag, e.g. H3, G3, A2")
    p.add_argument("--note", default="")
    p.add_argument("--cache", required=True)
    p.add_argument("--family", default="yarn_gqa")
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--heads", type=int, required=True)
    p.add_argument("--kv-heads", type=int, default=0, help="0 = heads//4 (GQA 4:1); =heads gives MHA")
    p.add_argument("--ffn-hidden", type=int, default=0)
    p.add_argument("--untie", action="store_true")
    p.add_argument("--init-fix", action="store_true", help="legacy `transformer` family only")
    p.add_argument("--extra", default="{}", help="json merged into model_extra")
    p.add_argument("--tokens", type=int, required=True, help="training token budget (may exceed cache => epochs)")
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--ga", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--evals", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--legacy-optim", action="store_true", help="decay all params, no LR floor (old behaviour)")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--keep-best", action="store_true", help="keep best.pt (needed for finalists / submissions)")
    p.add_argument("--mem-gb", type=float, default=0)
    p.add_argument("--out-root", default=str(ROOT / "runs" / "studies"))
    p.add_argument("--init-from", default="",
                   help="best.pt of a finished run with the same tokenizer and model shape: start from its weights "
                        "(fresh optimizer and schedule) instead of random initialisation, e.g. Wikipedia -> legal")
    return p.parse_args()


def main():
    a = parse()
    cache = CACHES / a.cache
    manifest = json.loads((cache / "manifest.json").read_text())
    settings = dict(manifest["identity"]["settings"])
    out = Path(a.out_root) / a.id
    metrics_file = out / "metrics.json"
    if metrics_file.exists() and json.loads(metrics_file.read_text()).get("complete") and (out / "post_eval.json").exists():
        print("already complete:", a.id); return

    per_update = a.bs * a.ga * a.context
    steps = max(10, round(a.tokens / per_update))
    extra = json.loads(a.extra)
    if a.family in ("yarn_gqa", "modern_transformer"):
        extra.setdefault("n_kv_heads", a.kv_heads or max(1, a.heads // 4))
        extra.setdefault("tie_embeddings", not a.untie)
        if a.ffn_hidden: extra.setdefault("ffn_hidden", a.ffn_hidden)
    elif a.init_fix:
        extra["init_fix"] = True
    cfg = dict(settings)
    cfg.update(family=a.family, d_model=a.d_model, layers=a.layers, heads=a.heads, context=a.context,
               dropout=0.0, gradient_checkpointing=a.grad_ckpt, batch_size=a.bs, grad_accum=a.ga,
               learning_rate=a.lr, weight_decay=a.wd, warmup_steps=max(20, int(a.warmup_frac * steps)),
               eval_every=max(50, steps // a.evals), max_steps=steps, shared_groups=2, local_window=256,
               model_extra=extra, decay_split=not a.legacy_optim, compile=not a.no_compile,
               min_lr_ratio=0.0 if a.legacy_optim else a.min_lr_ratio, require_competition_size=True)
    if a.init_from:
        cfg["init_from"] = str(Path(a.init_from).resolve())  # part of the run identity
    preset = {"id": a.id, "title": a.id, "question": a.note or a.study, "hypothesis": a.note,
              "protocol": {"study": a.study, "cache": a.cache}, "settings": cfg}
    pdir = ROOT / "studies" / "presets"; pdir.mkdir(exist_ok=True)
    (pdir / f"{a.id}.json").write_text(json.dumps(preset, indent=1))

    sys.argv = ["run", "--prepared-dir", str(cache), "--profile", "final", "--output-dir", str(out),
                "--seed", str(a.seed), "--precision", "bf16", "--cpu-threads", "8"]
    if a.mem_gb: sys.argv += ["--max-gpu-memory-gb", str(a.mem_gb)]
    import runner
    if a.init_from:
        # engine.py is fingerprinted for resume, so the initial weights are injected here instead of
        # there: the model the engine builds comes back already loaded.  On resume the engine then
        # overwrites it from last.pt as usual.
        import torch, engine
        build = engine.build_model

        def build_from_checkpoint(**kwargs):
            model = build(**kwargs)
            state = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)["model"]
            model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
            print(f"Initial weights loaded from {cfg['init_from']}", flush=True)
            return model

        engine.build_model = build_from_checkpoint
    runner.main(pdir / f"{a.id}.json")

    # ---------- post evaluation of the best checkpoint on every validation file ----------
    import torch
    from engine import Blocks, evaluate
    from model_zoo import build_model
    bundle = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(**bundle["model_config"]).cuda()
    model.load_state_dict(bundle["model"]); model.eval()
    dtype = manifest["tokens"].get("dtype", "<u4")
    post = {}
    sets = {"validation": manifest["tokens"]["validation"]}
    sets.update({f"validation_{s}": v for s, v in manifest["tokens"]["validation_by_source"].items()})
    for name, info in sets.items():
        blocks = Blocks(cache / f"{name}.bin", a.context, dtype)
        loss = evaluate(model, blocks, torch.device("cuda"), torch.bfloat16, True)
        bpt = info["raw_bytes"] / info["tokens"]
        post[name] = {"loss": loss, "bpb": loss / math.log(2) / bpt, "blocks": blocks.count}
    (out / "post_eval.json").write_text(json.dumps(post, indent=1))

    m = json.loads(metrics_file.read_text())
    row = {"id": a.id, "study": a.study, "note": a.note, "cache": a.cache, "family": a.family,
           "params_M": m["parameters"] / 1e6, "d_model": a.d_model, "layers": a.layers, "heads": a.heads,
           "kv_heads": extra.get("n_kv_heads"), "tied": not a.untie, "ffn_hidden": a.ffn_hidden or None,
           "vocab": manifest["tokenizer"]["actual_vocab_size"], "tokens": m["input_tokens_seen"],
           "epochs": m["equivalent_cache_passes"], "lr": a.lr, "seed": a.seed, "complete": m["complete"],
           "val_loss": m["best_validation_loss"], "val_bpb": m["best_validation_bpb"],
           "bpb_by_source": {k: v["bpb"] for k, v in post.items()},
           "tok_per_s": m["input_tokens_per_compute_second"], "gpu_hours": m["training_compute_gpu_hours"],
           "peak_gb": m["peak_gpu_allocated_bytes"] / 2**30, "initial_loss": m["initial_validation_loss"],
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("RESULT", json.dumps(row), flush=True)
    # disk hygiene (caches + checkpoints are large and the disk is nearly full)
    if m["complete"]:
        (out / "last.pt").unlink(missing_ok=True)
        if not a.keep_best:
            (out / "best.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
