"""CLI used by every experiment's run.py; --help and --dry-run need no ML packages."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULTS = dict(seed=42, sources=["supreme_court"], validation_fraction=0.05,
    max_document_bytes=16_000_000, max_invalid_fraction=0.05,
    tokenizer="bpe", vocab_size=32000, tokenizer_sample_bytes=20_000_000,
    train_tokens=200_000_000, val_tokens=5_000_000, validation_pairs=20,
    family="transformer", d_model=640, layers=8, heads=10, context=2048,
    shared_groups=2, local_window=256, dropout=0.0, gradient_checkpointing=True,
    batch_size=1, grad_accum=8, learning_rate=3e-4, weight_decay=0.1,
    warmup_steps=500, eval_every=1000, max_steps=0)
DEFAULTS.update(max_documents_per_source=0, max_source_text_bytes=0,
                require_competition_size=False, tokenizer_sample_strategy="natural")


def parser():
    p = argparse.ArgumentParser(description="Train a legal LM and tokenizer from scratch on the supplied training data")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--data-root", help="directory containing train_*.json or the official Phase_1_Pretraining tree")
    source.add_argument("--download", action="store_true", help="download selected official training files; uses HF_TOKEN if needed")
    source.add_argument("--prepared-dir", help="reuse a completed, possibly read-only prepared cache; verify its hashes and settings")
    p.add_argument("--revision", default="main", help="HF dataset revision; pin a commit for reproducible downloads")
    p.add_argument("--profile", choices=["example", "screen", "research", "final"], default="example")
    p.add_argument("--cache-dir", default=str(ROOT / "artifacts" / "prepared"))
    p.add_argument("--output-dir", help="defaults to Legal_LLM/runs/EXPERIMENT/PROFILE")
    p.add_argument("--sources", nargs="+", choices=["acts", "high_courts", "supreme_court", "tribunals"])
    p.add_argument("--tokenizer", choices=["bpe", "legal_bpe", "digit_bpe", "ayn_bpe"],
                   help="override the preset tokenizer; all variants train from scratch")
    p.add_argument("--tokenizer-sample-strategy", choices=["natural", "balanced"])
    for field in ["seed", "vocab_size", "tokenizer_sample_bytes", "train_tokens", "val_tokens", "context", "d_model",
                  "layers", "heads", "batch_size", "grad_accum", "max_steps", "eval_every", "warmup_steps",
                  "local_window", "shared_groups", "max_document_bytes", "validation_pairs",
                  "max_documents_per_source", "max_source_text_bytes"]:
        p.add_argument("--" + field.replace("_", "-"), type=int)
    for field in ["learning_rate", "weight_decay", "dropout", "validation_fraction", "max_invalid_fraction"]:
        p.add_argument("--" + field.replace("_", "-"), type=float)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N; select GPUs with CUDA_VISIBLE_DEVICES")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--stop-after", type=int, help="save and stop at this update without changing the planned LR schedule")
    p.add_argument("--max-run-minutes", type=float, default=0, help="soft invocation wall budget including preparation; 0 disables")
    p.add_argument("--save-margin-seconds", type=float, default=120, help="stop updates this far before the wall budget to save/evaluate")
    p.add_argument("--max-gpu-memory-gb", type=float, default=0, help="PyTorch allocator ceiling in GiB; 0 uses the device default")
    p.add_argument("--require-competition-size", action="store_true", default=None,
                   help="enforce the working 50M–300M parameter range also outside final mode")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print resolved settings without imports, downloads, or training")
    p.add_argument("--predict", action="store_true", help="after training, generate and validate 128-word continuations")
    p.add_argument("--predict-only", action="store_true", help="resume existing best checkpoint and generate; no updates")
    p.add_argument("--test-file", help="local official 512-word prompt JSON; otherwise use frozen held-out prompts")
    p.add_argument("--temperature", type=float, default=0.1, help="0 means greedy decoding")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-generation-tokens", type=int, default=768)
    p.add_argument("--generation-seed", type=int, default=42)
    return p


def configuration(preset, args):
    cfg = dict(DEFAULTS)
    cfg.update(preset["settings"])
    cfg["experiment_id"] = preset["id"]
    cfg["stage"] = args.profile
    if args.profile == "example":
        cfg.update(d_model=128, layers=2, heads=4, shared_groups=1, train_tokens=200_000,
                   val_tokens=20_000, tokenizer_sample_bytes=1_000_000, batch_size=1,
                   grad_accum=1, max_steps=20, warmup_steps=2, eval_every=10)
        # Explicit context experiments retain their treatment in example mode.
        cfg["context"] = preset["settings"].get("context", 256)
        cfg["local_window"] = min(64, cfg["context"])
    elif args.profile == "screen":
        cfg.update(train_tokens=2_097_152, val_tokens=32_768, warmup_steps=8, eval_every=32)
        cfg["context"] = preset["settings"].get("context", 512)
        cfg["grad_accum"] = max(1, 16_384 // cfg["context"])
    elif args.profile == "research":
        cfg.update(train_tokens=10_000_000, val_tokens=500_000, warmup_steps=100, eval_every=100)
    for key in DEFAULTS:
        if hasattr(args, key) and getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.no_gradient_checkpointing:
        cfg["gradient_checkpointing"] = False
    for key in ["vocab_size", "tokenizer_sample_bytes", "train_tokens", "val_tokens", "context", "d_model",
                "layers", "heads", "batch_size", "grad_accum", "eval_every", "local_window", "shared_groups", "max_document_bytes"]:
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["context"] < 2 or cfg["vocab_size"] < 257 or cfg["d_model"] % cfg["heads"]:
        raise ValueError("Require context>=2, vocabulary>=257, and d_model divisible by heads")
    if cfg["max_steps"] < 0 or cfg["warmup_steps"] < 0 or cfg["validation_pairs"] < 0:
        raise ValueError("max_steps, warmup_steps, validation_pairs must be nonnegative")
    if not 0 < cfg["validation_fraction"] < 1 or not 0 <= cfg["max_invalid_fraction"] <= 1:
        raise ValueError("Invalid split or removal fraction")
    if not 0 <= cfg["dropout"] < 1 or cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0:
        raise ValueError("Invalid optimizer/dropout configuration")
    if not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    if args.cpu_threads < 1 or (args.stop_after is not None and args.stop_after < 1):
        raise ValueError("cpu-threads and stop-after must be positive")
    if not 0 < args.top_p <= 1 or args.temperature < 0 or args.max_generation_tokens < 1:
        raise ValueError("Invalid decoding settings")
    if args.prepare_only and (args.predict or args.predict_only):
        raise ValueError("prepare-only cannot be combined with prediction")
    if cfg["max_documents_per_source"] < 0 or cfg["max_source_text_bytes"] < 0:
        raise ValueError("Source selection limits must be nonnegative")
    if args.max_run_minutes < 0 or args.save_margin_seconds < 0 or args.max_gpu_memory_gb < 0:
        raise ValueError("Time budget and save margin must be nonnegative")
    if args.max_run_minutes and args.save_margin_seconds >= args.max_run_minutes * 60:
        raise ValueError("Save margin must be smaller than the invocation budget")
    return cfg


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
    def flush(self):
        self.console.flush()
        self.log.flush()


def main(preset_file):
    args = parser().parse_args()
    args.invocation_started = time.monotonic()
    preset = json.loads(Path(preset_file).read_text())
    cfg = configuration(preset, args)
    output = Path(args.output_dir or ROOT / "runs" / preset["id"] / args.profile).expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"experiment": preset, "resolved": cfg, "output": str(output)}, indent=2))
        return
    if not args.data_root and not args.download and not args.prepared_dir:
        raise ValueError("Provide --data-root, --download, or --prepared-dir. See this experiment's README.")
    # Keep dependency loading after --help/--dry-run.
    from data import prepare, write_json, digest_file, fingerprint
    code_files = [Path(__file__), ROOT / "model_zoo.py", Path(__file__).with_name("engine.py"), Path(__file__).with_name("data.py"), Path(__file__).with_name("tokenizer_io.py")]
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "tokenizers", "ijson", "matplotlib")}
    versions["python"] = sys.version
    if cfg["tokenizer"] == "ayn_bpe":
        versions["sentencepiece"] = importlib.metadata.version("sentencepiece")
    args.code_hash = fingerprint({"files": {p.name: digest_file(p) for p in code_files}, "environment": versions})
    cache, manifest = prepare(args, cfg)
    print(f"Prepared cache: {cache}", flush=True)
    if args.prepare_only:
        if args.output_dir:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "preparation_manifest.json", manifest)
            (output / "PREPARATION.md").write_text(
                f"# {preset['id']}: preparation and tokenizer diagnostics\n\n"
                f"Hypothesis: {preset['hypothesis']}\n\nCache: `{cache}`\n\n"
                "No LM training or model-quality evaluation was performed.\n\n```json\n" +
                json.dumps({"protocol": preset.get("protocol", {}), "data": manifest}, indent=2) + "\n```\n")
        return
    if args.max_run_minutes and time.monotonic() - args.invocation_started >= args.max_run_minutes * 60 - args.save_margin_seconds:
        print("Preparation used this invocation's budget. Cache is complete; rerun to start training.", flush=True)
        return
    from engine import fit, predict
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock, (output / "training.log").open("a", buffering=1) as log:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is using this output directory") from None
        original = sys.stdout
        sys.stdout = Tee(original, log)
        try:
            model, tokenizer, device = fit(cfg, args, cache, manifest, output)
            write_json(output / "data_manifest.json", manifest)
            write_json(output / "experiment_protocol.json", preset)
            versions["code_sha256"] = args.code_hash
            write_json(output / "environment.json", versions)
            if args.predict or args.predict_only:
                predict(model, tokenizer, device, cache, output, args)
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            sys.stdout = original


if __name__ == "__main__":
    raise SystemExit("Use an experiment's run.py; see experiments/README.md")
