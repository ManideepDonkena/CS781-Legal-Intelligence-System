"""Common training, complete checkpoint recovery, and strict continuation export."""
from __future__ import annotations

import json
import math
import random
import resource
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model_zoo import build_model
from data import write_json, fingerprint, digest_file


def atomic_save(path, value):
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


class Blocks:
    """Memory-mapped nonoverlapping contexts. Loss shifts targets exactly once."""
    def __init__(self, path, context, dtype="<u4"):
        self.tokens = np.memmap(path, mode="r", dtype=dtype)
        self.context = context
        self.count = len(self.tokens) // context
        if not self.count:
            raise ValueError(f"{path} contains no full {context}-token block")

    def get(self, indices, device):
        batch = np.stack([self.tokens[i*self.context:(i+1)*self.context] for i in indices])
        return torch.as_tensor(batch.astype(np.int64), device=device)


def block_indices(cursor, count, size, seed):
    """Constant-memory affine permutation; every block visited once per epoch.

    This is less thoroughly shuffled than randperm but reproducible without a
    corpus-sized index array. The absolute cursor is the complete sampler state.
    """
    indices = []
    for position in range(cursor, cursor + size):
        epoch, offset = divmod(position, count)
        rng = random.Random(seed + epoch)
        stride = rng.randrange(1, count + 1)
        while math.gcd(stride, count) != 1:
            stride = stride % count + 1
        origin = rng.randrange(count)
        indices.append((origin + offset * stride) % count)
    return indices


def amp_settings(device, precision):
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "fp32"
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("Use fp32 on CPU")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support bf16; use auto or fp32")
    return precision, torch.bfloat16 if precision == "bf16" else torch.float16


def evaluate(model, data, device, dtype, mixed):
    model.eval()
    loss_sum = 0.0
    batch = 8  # every block has identical length, so the mean of batch means is exact up to the tail
    seen = 0
    with torch.inference_mode():
        for start in range(0, data.count, batch):
            idx = list(range(start, min(start + batch, data.count)))
            ids = data.get(idx, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed):
                loss = model(ids, labels=ids)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            loss_sum += float(loss) * len(idx)
            seen += len(idx)
    model.train()
    return loss_sum / seen


def save_plot(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([r["step"] for r in history], [r["train_loss"] for r in history], label="train (update mean)")
    ax.plot([r["step"] for r in history], [r["validation_loss"] for r in history], label="validation")
    ax.set(xlabel="Optimizer updates", ylabel="Next-token cross entropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_validation_loss.png", dpi=150)
    plt.close(fig)


def fit(cfg, args, cache, manifest, output):
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This runner uses one device. Do not launch with torchrun/DDP.")
    torch.set_num_threads(args.cpu_threads)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.max_gpu_memory_gb:
        total = torch.cuda.get_device_properties(device).total_memory
        # Reserve at least 10% for CUDA context / non-PyTorch allocations.
        torch.cuda.set_per_process_memory_fraction(min(0.9, args.max_gpu_memory_gb * 2**30 / total), device)
    precision, dtype = amp_settings(device, args.precision)
    model_config = {"vocab_size": manifest["tokenizer"]["actual_vocab_size"], "family": cfg["family"],
                    "d_model": cfg["d_model"], "n_layers": cfg["layers"], "n_heads": cfg["heads"],
                    "max_length": cfg["context"], "local_window": cfg["local_window"],
                    "shared_groups": cfg["shared_groups"], "dropout": cfg["dropout"],
                    "gradient_checkpointing": cfg["gradient_checkpointing"]}
    model_config.update(cfg.get("model_extra", {}))
    model = build_model(**model_config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if (args.profile == "final" or cfg.get("require_competition_size")) and not 50_000_000 <= parameters <= 300_000_000:
        raise ValueError(f"Final run has {parameters:,} parameters; required working range is 50M–300M")
    print(f"{cfg['experiment_id']}: {parameters:,} unique parameters, {device}, {precision}", flush=True)
    token_dtype = manifest["tokens"].get("dtype", "<u4")
    train, val = (Blocks(cache / "train.bin", cfg["context"], token_dtype),
                  Blocks(cache / "validation.bin", cfg["context"], token_dtype))
    # bytes of raw text per validation token -> bits-per-byte is comparable across tokenizers
    bytes_per_token = manifest["tokens"]["validation"]["raw_bytes"] / manifest["tokens"]["validation"]["tokens"]
    batch_blocks = cfg["batch_size"] * cfg["grad_accum"]
    planned = cfg["max_steps"] or max(1, train.count // batch_blocks)
    identity = fingerprint({"config": cfg, "cache": manifest["cache_id"], "planned_steps": planned,
                            "precision": precision, "device_type": device.type, "code": args.code_hash})
    if args.predict_only:
        bundle = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        if bundle["identity"] != identity:
            raise ValueError("Inference configuration/data/code differs from the saved run; reuse the same training options")
        if digest_file(output / "tokenizer.json") != bundle["tokenizer_sha256"]:
            raise ValueError("Saved tokenizer does not match the trained checkpoint")
        model.load_state_dict(bundle["model"])
        model.eval()
        # Inference must not rewrite training metrics or allocate optimizer state.
        return model, Tokenizer.from_file(str(output / "tokenizer.json")), device
    if cfg.get("decay_split", False):
        # Decay only weight matrices; norms/biases/embeddings are excluded (standard practice).
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and not n.startswith(("token.", "position.", "lm_head."))]
        no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and not n.startswith(("token.", "position.", "lm_head.")))]
        groups = [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    else:
        groups = model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95),
                                  fused=device.type == "cuda")
    warmup = min(cfg["warmup_steps"], max(0, planned - 1))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, planned - warmup)
        floor = cfg.get("min_lr_ratio", 0.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
    last = output / "last.pt"
    step = cursor = skipped_updates = 0
    history, best_loss, best_step, compute_seconds = [], float("inf"), 0, 0.0
    initial_validation_loss = None
    if last.exists():
        # Only load locally produced trusted checkpoints; torch pickle is not
        # an interchange format for untrusted downloads.
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Run configuration/data/code changed. Choose a new --output-dir; refusing incompatible resume.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        step, cursor = saved["step"], saved["cursor"]
        history, best_loss, best_step = saved["history"], saved["best_loss"], saved["best_step"]
        compute_seconds, skipped_updates = saved["compute_seconds"], saved["skipped_updates"]
        initial_validation_loss = saved["initial_validation_loss"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
        print(f"Resuming update {step}, block cursor {cursor}", flush=True)
    elif args.predict_only:
        raise FileNotFoundError("--predict-only requires an existing run with last.pt and best.pt")
    elif (output / "run_config.json").exists():
        old = json.loads((output / "run_config.json").read_text())
        if old["identity"] != identity:
            raise ValueError("Output directory belongs to another run. Choose a fresh --output-dir")
    shutil.copyfile(cache / "tokenizer.json", output / "tokenizer.json")
    write_json(output / "run_config.json", {"identity": identity, "config": cfg, "model_config": model_config,
        "cache": str(cache), "cache_id": manifest["cache_id"], "parameters": parameters,
        "embedding_parameters": model.token.weight.numel(), "planned_steps": planned, "precision": precision,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "allocator_limit_GiB_requested": args.max_gpu_memory_gb,
        "train_blocks": train.count, "validation_blocks": val.count,
        "discarded_train_tokens": len(train.tokens) % cfg["context"],
        "discarded_validation_tokens": len(val.tokens) % cfg["context"],
        "target_tokens_per_update": batch_blocks * (cfg["context"] - 1),
        "packing": "EOS between documents; attention may cross document boundaries; one loss shift",
        "sampler": "deterministic affine block permutation per epoch; absolute cursor persisted"})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    # torch.compile only wraps the training forward (1.5-1.7x faster); evaluation, checkpoints and
    # state_dict keys stay on the eager module so saved weights are unaffected.
    fwd = torch.compile(model) if cfg.get("compile", False) and device.type == "cuda" else model
    started = time.perf_counter()
    if initial_validation_loss is None:
        initial_validation_loss = evaluate(model, val, device, dtype, precision != "fp32")
    deadline = (args.invocation_started + args.max_run_minutes * 60 - args.save_margin_seconds
                if args.max_run_minutes else None)
    budget_stop = False
    stop = min(planned, args.stop_after) if args.stop_after else planned
    while step < stop and not args.predict_only:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_start = time.perf_counter()
        cursor_before = cursor
        retry_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if device.type == "cuda" else []) if scaler.is_enabled() else None
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(cfg["grad_accum"]):
            ids = train.get(block_indices(cursor, train.count, cfg["batch_size"], cfg["seed"]), device)
            cursor += cfg["batch_size"]
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
                loss = fwd(ids, labels=ids)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at update {step+1}")
            train_loss += float(loss.detach()) / cfg["grad_accum"]
            scaler.scale(loss / cfg["grad_accum"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale = scaler.get_scale()
        if not torch.isfinite(grad_norm) and not scaler.is_enabled():
            raise FloatingPointError("Nonfinite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_seconds += time.perf_counter() - compute_start
        if scaler.get_scale() < old_scale:
            skipped_updates += 1
            # Retry the SAME data/dropout draw after AMP reduces its scale.
            # Overflow must not give a family extra unseen training documents
            # or silently consume a different token budget.
            cursor = cursor_before
            if retry_rng is not None:
                torch.set_rng_state(retry_rng[0])
                if device.type == "cuda":
                    torch.cuda.set_rng_state_all(retry_rng[1])
            if skipped_updates > 20:
                raise FloatingPointError("Repeated FP16 overflows; try bf16 or fp32 in a fresh run")
            continue
        scheduler.step()
        step += 1
        budget_stop = deadline is not None and time.monotonic() >= deadline
        if step == 1 or step % cfg["eval_every"] == 0 or step == stop or budget_stop:
            validation = evaluate(model, val, device, dtype, precision != "fp32")
            row = {"step": step, "train_loss": train_loss, "validation_loss": validation,
                   "validation_bpb": validation / math.log(2) / bytes_per_token,
                   "input_tokens_seen": cursor * cfg["context"], "learning_rate": scheduler.get_last_lr()[0],
                   "training_compute_seconds": compute_seconds}
            history.append(row)
            if validation < best_loss:
                best_loss, best_step = validation, step
                atomic_save(output / "best.pt", {"model": model.state_dict(), "model_config": model_config,
                    "step": step, "validation_loss": validation, "identity": identity,
                    "tokenizer_sha256": manifest["tokenizer"]["sha256"]})
            print(f"step {step}/{planned} train={train_loss:.4f} validation={validation:.4f}", flush=True)
            atomic_save(last, {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "cursor": cursor,
                "history": history, "best_loss": best_loss, "best_step": best_step, "compute_seconds": compute_seconds,
                "initial_validation_loss": initial_validation_loss,
                "skipped_updates": skipped_updates, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
            write_json(output / "history.json", history)
            budget_stop = deadline is not None and time.monotonic() >= deadline
        if budget_stop:
            print("Wall budget reached; saved a resumable checkpoint. No schedule or token target was changed.", flush=True)
            break
    if not history:
        raise ValueError("No training checkpoint available")
    metrics = {"experiment_id": cfg["experiment_id"], "stage": args.profile, "parameters": parameters,
               "steps": step, "planned_steps": planned, "complete": step >= planned,
               "stop_reason": "complete" if step >= planned else "wall_budget" if budget_stop else "stop_after",
               "best_step": best_step, "best_validation_loss": best_loss,
               "initial_validation_loss": initial_validation_loss,
               "best_validation_perplexity": math.exp(best_loss) if best_loss < 700 else None,
               "best_validation_bpb": best_loss / math.log(2) / bytes_per_token,
               "validation_bytes_per_token": bytes_per_token,
               "input_tokens_seen": cursor * cfg["context"], "target_tokens_seen": cursor * (cfg["context"]-1),
               "processed_input_tokens_including_fp16_retries": (cursor + skipped_updates * batch_blocks) * cfg["context"],
               "equivalent_cache_passes": cursor / train.count, "training_compute_seconds": compute_seconds,
               "input_tokens_per_compute_second": cursor * cfg["context"] / compute_seconds if compute_seconds else None,
               "this_invocation_seconds": time.perf_counter() - started,
               "training_compute_gpu_hours": compute_seconds / 3600 if device.type == "cuda" else 0,
               "remaining_train_compute_seconds_estimate": (planned - step) * compute_seconds / max(1, step),
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
               "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
               "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "skipped_fp16_updates": skipped_updates, "history": history}
    write_json(output / "metrics.json", metrics)
    save_plot(history, output)
    # Export best weights with config + tokenizer, suitable for explicit loading.
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    del best
    model.eval()
    return model, Tokenizer.from_file(str(output / "tokenizer.json")), device


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_prompts(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if isinstance(data, dict):
        pairs = data.items()
    elif isinstance(data, list):
        pairs = [(str(r.get("id", i)) if isinstance(r, dict) else str(i), r) for i, r in enumerate(data)]
    else:
        raise ValueError("Test JSON must be an ID mapping or list")
    result = {}
    for key, row in pairs:
        key = str(key)
        text = row.get("text") if isinstance(row, dict) else row
        if not isinstance(text, str) or not text.strip() or key in result:
            raise ValueError(f"Invalid prompt or duplicate ID {key}")
        if len(text.split()) != 512:
            raise ValueError(f"Prompt {key} has {len(text.split())} words; expected 512. No silent truncation.")
        result[key] = text
    if not result:
        raise ValueError("No eligible 512-word prompts; provide --test-file or more held-out data")
    return result


@torch.inference_mode()
def generate(model, tokenizer, prompts, device, args):
    predictions = {}
    args.generated_token_count = 0
    precision, dtype = amp_settings(device, getattr(args, "precision", "auto"))
    eos = tokenizer.token_to_id("<|endoftext|>")
    torch.manual_seed(args.generation_seed)
    for key, prompt in prompts.items():
        ids = tokenizer.encode(prompt).ids
        if len(ids) >= model.max_length:
            raise ValueError(f"Prompt {key} needs {len(ids)} tokens; model context is {model.max_length}. Train a longer context.")
        generated = []
        for _ in range(args.max_generation_tokens):
            if (getattr(args, "max_run_minutes", 0) and time.monotonic() - args.invocation_started
                    >= args.max_run_minutes * 60 - args.save_margin_seconds):
                raise TimeoutError("Inference wall budget reached; no incomplete submission was written")
            # Never slide away any portion of the official prompt.
            if len(ids) + len(generated) > model.max_length:
                break
            x = torch.tensor([ids + generated], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
                logits = model(x)["logits"][0, -1].float()
            # Declared decoding policy: no EOS until enough complete words.
            logits[eos] = -float("inf")
            if args.temperature == 0:
                token = int(logits.argmax())
            else:
                probs = (logits / args.temperature).softmax(-1)
                probs, indices = probs.sort(descending=True)
                probs[probs.cumsum(-1) - probs > args.top_p] = 0
                token = int(indices[torch.multinomial(probs, 1)])
            generated.append(token)
            args.generated_token_count += 1
            continuation = tokenizer.decode(generated, skip_special_tokens=False)
            # A 129th word (or trailing whitespace) establishes the 128th is
            # complete; otherwise the next token may still extend that word.
            words = continuation.split()
            if len(words) >= 129 or (len(words) >= 128 and continuation[-1:].isspace()):
                predictions[key] = {"text": " ".join(words[:128])}
                print(f"generated {len(predictions)}/{len(prompts)}", flush=True)
                break
        if key not in predictions:
            raise ValueError(f"{key}: failed to reach 128 complete words within token/context limits. No padded submission produced.")
    return predictions


def predict(model, tokenizer, device, cache, output, args):
    path = Path(args.test_file) if args.test_file else cache / "held_out_prompts.json"
    prompts = read_prompts(path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    predictions = generate(model, tokenizer, prompts, device, args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    words = sum(len(value["text"].split()) for value in predictions.values())
    if set(prompts) != set(predictions) or any(len(v["text"].split()) != 128 for v in predictions.values()):
        raise ValueError("Prediction validation failed")
    write_json(output / "prediction.json", predictions)
    archive = output / "submission.zip"
    temp = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output / "prediction.json", "prediction.json")
    temp.replace(archive)
    write_json(output / "generation_config.json", {"prompt_file": str(path.resolve()), "examples": len(prompts),
        "prompt_sha256": digest_file(path), "prediction_sha256": digest_file(output / "prediction.json"),
        "temperature": args.temperature, "top_p": args.top_p, "seed": args.generation_seed,
        "max_new_tokens": args.max_generation_tokens, "eos_policy": "suppressed until 128 complete words",
        "prompt_policy": "all 512 words preserved; fail if context insufficient", "checkpoint": "best.pt"})
    write_json(output / "generation_metrics.json", {"examples": len(prompts), "output_words": words,
        "sampled_tokens_including_boundary_lookahead": args.generated_token_count,
        "sampled_tokens_per_second": args.generated_token_count / duration if duration else None,
        "wall_seconds": duration, "words_per_second": words / duration if duration else None,
        "examples_per_second": len(prompts) / duration if duration else None,
        "active_gpu_hours": duration / 3600 if device.type == "cuda" else 0,
        "implementation": "single-example full-prefix recomputation; no KV cache"})
    print(f"Validated {archive} (local artifact only)", flush=True)
