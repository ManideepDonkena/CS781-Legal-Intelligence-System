"""Train and compare the four Phase-1 causal language model families.

Example:
  python Legal_LLM/train_compare.py --dataset-dir task1_data --tokenizer gpt2 \
      --output-dir runs/p1 --families transformer gru linear hybrid --max-steps 1000

The input datasets are the HuggingFace datasets written by dataset.pipeline.
This script does not download data or train a tokenizer. Keep tokenizer, data,
sequence length, token budget, and optimizer settings fixed for architecture
comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk

from model_zoo import FAMILIES, build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True, help="folder containing train_dataset/ and validation_dataset/")
    p.add_argument("--tokenizer", required=True, help="local tokenizer directory or HuggingFace tokenizer ID")
    p.add_argument("--output-dir", default="runs/p1")
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=640)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=10)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--local-window", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--skip-generation", action="store_true", help="skip the optional fixed-prompt sample")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batchify(rows, device):
    ids = torch.as_tensor(rows["input_ids"], dtype=torch.long, device=device)
    return ids


@torch.no_grad()
def evaluate(model, loader, device, max_batches=100):
    model.eval()
    losses, count = 0.0, 0
    for batch in loader:
        ids = batchify(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
            out = model(ids, labels=ids)
        if out["loss"] is not None and torch.isfinite(out["loss"]):
            losses += float(out["loss"].item())
            count += 1
        if count >= max_batches:
            break
    model.train()
    return losses / max(1, count)


@torch.no_grad()
def generate_sample(model, tokenizer, device, context, target_words=128):
    prompt = "In the present case, the Court has considered the submissions of the parties."
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = torch.tensor([ids], dtype=torch.long, device=device)
    # Stop once the decoded continuation reaches the requested word count.
    for _ in range(384):
        input_ids = generated[:, -context:]
        logits = model(input_ids)["logits"][:, -1, :].float() / 0.8
        probs = logits.softmax(-1)
        sorted_probs, sorted_ids = probs.sort(descending=True)
        remove = sorted_probs.cumsum(-1) - sorted_probs > 0.9
        sorted_probs[remove] = 0
        sorted_probs /= sorted_probs.sum(-1, keepdim=True)
        next_id = sorted_ids.gather(-1, torch.multinomial(sorted_probs, 1))
        generated = torch.cat((generated, next_id), dim=1)
        continuation = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
        if len(continuation.split()) >= target_words or (tokenizer.eos_token_id is not None and int(next_id.item()) == tokenizer.eos_token_id):
            break
    continuation = " ".join(continuation.split()[:target_words])
    return {"prompt": prompt, "continuation": continuation, "words": len(continuation.split())}


def fit_family(family, train_data, val_data, tokenizer, vocab_size, args, device, output_dir):
    seed_everything(args.seed)
    model = build_model(family, vocab_size=vocab_size, d_model=args.d_model,
                        n_layers=args.layers, n_heads=args.heads,
                        max_length=args.context, local_window=args.local_window).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{family}: {params / 1e6:.1f}M parameters; device={device}")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Train and validation datasets must each contain at least one batch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    use_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    history, best_val, best_step = [], float("inf"), 0
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step, micro, started = 0, 0, time.time()
    iterator = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        ids = batchify(batch, device)
        amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            loss = model(ids, labels=ids)["loss"] / args.grad_accum
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss for {family} at step {step}")
        scaler.scale(loss).backward()
        micro += 1
        if micro % args.grad_accum:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step == 1 or step % args.eval_every == 0 or step == args.max_steps:
            val_loss = evaluate(model, val_loader, device)
            row = {"step": step, "train_loss": float(loss.item() * args.grad_accum), "validation_loss": val_loss}
            history.append(row)
            print(f"{family} step {step:5d} | train {row['train_loss']:.4f} | val {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.2f}")
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                torch.save({"model": model.state_dict(), "family": family,
                            "model_config": {"vocab_size": vocab_size, "d_model": args.d_model,
                                             "n_layers": args.layers, "n_heads": args.heads,
                                             "max_length": args.context, "local_window": args.local_window},
                            "step": step, "validation_loss": val_loss}, output_dir / "best.pt")
    duration = time.time() - started
    result = {"family": family, "parameters": params, "best_validation_loss": best_val,
              "best_validation_perplexity": math.exp(min(best_val, 20)), "best_step": best_step,
              "steps": step, "training_seconds": duration, "history": history}
    if not args.skip_generation:
        best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        model.eval()
        result["sample"] = generate_sample(model, tokenizer, device, args.context)
        (output_dir / "sample.json").write_text(json.dumps(result["sample"], indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    data_dir = Path(args.dataset_dir)
    train_data = load_from_disk(str(data_dir / "train_dataset"))
    val_data = load_from_disk(str(data_dir / "validation_dataset"))
    for ds_name, dataset in (("train", train_data), ("validation", val_data)):
        if "input_ids" not in dataset.column_names:
            raise ValueError(f"{ds_name} dataset has no input_ids column")
        if len(dataset) and len(dataset[0]["input_ids"]) != args.context:
            raise ValueError(f"{ds_name} blocks do not match --context={args.context}")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for family in args.families:
        result = fit_family(family, train_data, val_data, tokenizer, len(tokenizer), args, device, root / family)
        results.append({k: v for k, v in result.items() if k != "history"})
        (root / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nCOMPARISON (lower validation loss is better)")
    for row in sorted(results, key=lambda item: item["best_validation_loss"]):
        print(f"{row['family']:12} val_loss={row['best_validation_loss']:.4f}  ppl={row['best_validation_perplexity']:.2f}  params={row['parameters']/1e6:.1f}M  best_step={row['best_step']}")


if __name__ == "__main__":
    main()
