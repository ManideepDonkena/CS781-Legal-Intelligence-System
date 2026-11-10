"""Random-initialized training, resumable Accelerate state, 1 GPU or FSDP."""
import argparse
import math
import os
from pathlib import Path
import shutil
import time
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
from src.common import *
from src.prepare import verify
from src.blocks import Blocks
from src.model_factory import configuration, build, check_parameters, CLASSES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=["A_llama", "B_pythia", "C_falcon", "D_moe"], required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gate-report")
    p.add_argument("--smoke", action="store_true", help="tiny development model only")
    p.add_argument("--gate-pilot", action="store_true", help="bounded real-size Gate 0 run (at most 3 updates)")
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--precision", choices=["no", "bf16", "fp16"], default="bf16")
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--tokens", type=int, default=500_000_000)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--full-pass", action="store_true", help="one pass over every complete training block")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-blocks-per-source", type=int, default=64, help="0 evaluates all validation blocks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=2)
    return p


@torch.no_grad()
def evaluate(model, accelerator, prepared, manifest, context, limit):
    model.eval()
    results = {}
    # Identical iteration counts on all ranks are essential for FSDP forward
    # collectives. Validation is replicated; reduction does not multiply counts.
    for source in manifest["streams"]["validation"]:
        data = Blocks(prepared, manifest, "validation", context, source=source)
        total = count = 0
        for i in range(min(len(data), limit) if limit else len(data)):
            x = data[i].unsqueeze(0).to(accelerator.device)
            with accelerator.autocast():
                logits = model(input_ids=x).logits
                nll = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                                        x[:, 1:].reshape(-1), reduction="sum")
            total += float(nll)
            count += context - 1
        sums = accelerator.reduce(torch.tensor([total, count], dtype=torch.float64, device=accelerator.device), reduction="mean")
        results[source] = {"nll": sums[0].item(), "targets": int(sums[1].item()), "loss": (sums[0] / sums[1]).item()}
    total = sum(v["nll"] for v in results.values())
    count = sum(v["targets"] for v in results.values())
    model.train()
    return {"loss": total / count, "per_source": results}


def main():
    a = parser().parse_args()
    if a.gate_pilot and not 1 <= a.max_steps <= 3:
        raise ValueError("Gate pilot is limited to 1–3 updates")
    if min(a.context, a.batch, a.grad_accum, a.tokens, a.eval_every, a.cpu_threads) <= 0 or a.context < 2 or a.max_steps < 0 or a.eval_blocks_per_source < 0 or a.warmup < 0 or a.lr <= 0:
        raise ValueError("Invalid training budget/settings")
    torch.set_num_threads(a.cpu_threads)
    if not a.smoke and not torch.cuda.is_available():
        raise ValueError("Production training and real-size Gate 0 pilots require CUDA")
    set_seed(a.seed)
    prepared, output = Path(a.prepared).resolve(), Path(a.output).resolve()
    manifest = verify(prepared)
    tok = load_tokenizer(prepared / "tokenizer.json")
    cfg = configuration(a.arch, tok.get_vocab_size(), tok.token_to_id("<|endoftext|>"), a.context, a.smoke)
    parameters = check_parameters(cfg, a.smoke)
    plugin = None
    if a.fsdp:
        if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 2:
            raise ValueError("FSDP requires a multi-process CUDA launch")
        plugin = FullyShardedDataParallelPlugin(sharding_strategy="FULL_SHARD", use_orig_params=True,
            sync_module_states=True, auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=[CLASSES[cfg.model_type]], state_dict_type="SHARDED_STATE_DICT")
    accelerator = Accelerator(mixed_precision=a.precision, gradient_accumulation_steps=a.grad_accum,
                              fsdp_plugin=plugin, step_scheduler_with_optimizer=False)
    if not a.smoke and not a.gate_pilot:
        if not a.gate_report:
            raise ValueError("Production training requires --gate-report from Gate 0")
        gate = read(a.gate_report)
        if not gate.get("passed") or gate["code"] != code_hash() or gate["prepared_sha256"] != digest_file(prepared / "manifest.json"):
            raise ValueError("Gate report missing, failed, stale or belongs to another preparation")
        if gate["hardware_profile"] != "production":
            raise ValueError("A tiny-model Gate 0 report cannot authorize production training")
        if gate["world_size"] != accelerator.num_processes or gate["precision"] != a.precision or gate["context"] != a.context:
            raise ValueError("Gate must validate this world size, precision and context")
    model = build(cfg)
    # HF checkpointing is compatible with DDP; FSDP activation checkpointing is
    # selected through the plugin to avoid wrapping the same layer twice.
    if a.fsdp:
        plugin.activation_checkpointing = True
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    data = Blocks(prepared, manifest, "train", a.context, a.seed)
    global_blocks = a.batch * a.grad_accum * accelerator.num_processes
    steps = a.max_steps or (math.ceil(data.count / global_blocks) if a.full_pass else max(1, a.tokens // (a.context * global_blocks)))
    data.length = steps * global_blocks
    loader = DataLoader(data, batch_size=a.batch, shuffle=False, num_workers=0, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    warmup = min(a.warmup, max(0, steps - 1))
    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    accelerator.register_for_checkpointing(scheduler)
    settings = vars(a).copy()
    for key in ("output", "prepared", "gate_report", "stop_after"):
        settings.pop(key)
    identity = fingerprint({"settings": settings, "world": accelerator.num_processes,
                            "data": digest_file(prepared / "manifest.json"), "code": code_hash(), "steps": steps})
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run.json"
    if config_path.exists() and read(config_path)["identity"] != identity:
        raise ValueError("Resume settings/code/data changed; use a new run directory")
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    history, best_loss, step = [], float("inf"), 0
    compute_seconds = 0.0
    source_tokens = {source: 0 for source in data.names}
    latest = output / "latest.json"
    if latest.exists():
        state = read(latest)
        checkpoint = output / state["checkpoint"]
        if not (checkpoint / "COMPLETE.json").exists():
            raise ValueError("Incomplete recovery checkpoint")
        accelerator.load_state(str(checkpoint))
        step, history, best_loss = state["step"], state["history"], state["best_loss"]
        compute_seconds = state.get("compute_seconds", 0.0)
        source_tokens = state.get("source_tokens", source_tokens)
    if accelerator.is_main_process:
        import importlib.metadata
        write_json(output / "environment.json", {"python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "numpy", "tokenizers", "sentencepiece")},
            "cuda_runtime": torch.version.cuda, "device": str(accelerator.device),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "world_size": accelerator.num_processes, "code_sha256": code_hash()})
        if not (output / "code_snapshot.zip").exists():
            from src.evidence import snapshot
            snapshot(output)
        write_json(config_path, {"identity": identity, "settings": settings, "parameters": parameters,
            "planned_steps": steps, "world_size": accelerator.num_processes, "global_input_tokens_per_step": global_blocks * a.context,
            "prepared_sha256": digest_file(prepared / "manifest.json"), "tail_tokens": data.tails,
            "unique_training_blocks": data.count, "planned_cache_passes": data.length / data.count})
    accelerator.wait_for_everyone()
    started = time.monotonic()
    model.train()
    active_loader = accelerator.skip_first_batches(loader, step * a.grad_accum)
    optimizer.zero_grad()
    for x in active_loader:
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize()
        compute_start = time.monotonic()
        with accelerator.accumulate(model):
            result = model(input_ids=x, labels=x)
            loss = result.loss
            finite = accelerator.reduce(torch.tensor(int(torch.isfinite(loss)), device=accelerator.device), reduction="sum")
            if int(finite) != accelerator.num_processes:
                raise FloatingPointError("Nonfinite loss on at least one rank")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if accelerator.sync_gradients and accelerator.optimizer_step_was_skipped:
                raise FloatingPointError("FP16 overflow; rerun from last saved state with a stable precision/budget")
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad()
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize()
        compute_seconds += time.monotonic() - compute_start
        if not accelerator.sync_gradients:
            continue
        step += 1
        for index in range((step - 1) * global_blocks, step * global_blocks):
            source, _ = data.location(index)
            source_tokens[data.names[source]] += a.context
        stop = step == steps or (a.stop_after and step >= a.stop_after)
        if step % a.eval_every == 0 or stop:
            validation = evaluate(model, accelerator, prepared, manifest, a.context, a.eval_blocks_per_source)
            history.append({"step": step, "last_microbatch_train_loss": float(loss.detach()), "validation": validation})
            improved = validation["loss"] < best_loss
            if improved:
                best_loss = validation["loss"]
                weights = accelerator.get_state_dict(model)
                if accelerator.is_main_process:
                    export = output / "best"
                    export.mkdir(exist_ok=True)
                    torch.save(weights, export / "model.pt.tmp")
                    (export / "model.pt.tmp").replace(export / "model.pt")
                    cfg.save_pretrained(export)
                    shutil.copyfile(prepared / "tokenizer.json", export / "tokenizer.json")
                    write_json(export / "provenance.json", {"identity": identity, "step": step, "tokenizer_sha256": digest_file(export / "tokenizer.json")})
                del weights
            checkpoint = output / f"checkpoint_{step:09d}"
            accelerator.save_state(str(checkpoint))
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                state = {"step": step, "history": history, "best_loss": best_loss, "checkpoint": checkpoint.name,
                         "compute_seconds": compute_seconds, "source_tokens": source_tokens}
                write_json(checkpoint / "COMPLETE.json", {"identity": identity, "step": step})
                write_json(latest, state)
                write_json(output / "metrics.json", {**state, "parameters": parameters, "complete": step == steps,
                    "input_tokens": step * global_blocks * a.context,
                    "target_tokens": step * global_blocks * (a.context - 1),
                    "input_tokens_per_compute_second": step * global_blocks * a.context / compute_seconds,
                    "invocation_wall_seconds": time.monotonic() - started,
                    "peak_gpu_allocated_bytes_rank0": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
                for old in sorted(output.glob("checkpoint_*"))[:-2]:
                    shutil.rmtree(old)
                print(f"{a.arch} step {step}/{steps} validation={validation['loss']:.5f}", flush=True)
            accelerator.wait_for_everyone()
        if stop:
            break
    accelerator.end_training()

if __name__ == "__main__":
    main()
