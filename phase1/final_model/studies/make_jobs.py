#!/usr/bin/env python3
"""Generate studies/jobs.json: stage 1 (caches + LR sweep) and the planner job that appends wave 2.

Waves
  1. caches (CPU) + LR sweep at ~57M params / 600M tokens (GPU)
  2. (planner plan_wave2.py, after LR sweep) seed noise G3, architecture fixes, vocab/depth (H2/A2),
     tying (H3), GQA (A3), width-depth, epochs (C4), source mixture (C5)
  3. (planner plan_final.py, after wave 2) build the 3B-token cache with the winning vocab/mixture,
     train the final <=300M model(s), generate Codabench submissions, write the report
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = "/home/mani/miniconda3/envs/phase_1_capstone/bin/python"

MIX_A = "supreme_court=0.35,high_courts=0.25,tribunals=0.25,acts=0.15"   # domain-weighted, every source < 1 epoch @600M
MIX_SC = "supreme_court=1.0"                                              # target-domain only (~2.3 epochs @600M)
MIX_NAT = "supreme_court=0.07,high_courts=0.55,tribunals=0.34,acts=0.04"  # natural supply proportions
TOK600 = 600_000_000


def cache_job(name, vocab, mix, tokens=TOK600, tokenizer_from=None, tok="bpe", prio=0):
    cmd = f"{PY} -u studies/build_cache.py --name {name} --tokenizer {tok} --vocab {vocab} --train-tokens {tokens} --mix {mix}"
    if tokenizer_from: cmd += f" --tokenizer-from {tokenizer_from}"
    return {"id": f"cache_{name}", "kind": "cpu", "cmd": cmd, "deps": [], "priority": prio}


def train_job(jid, study, cache, d, L, h, kv=0, tokens=TOK600, lr=2e-3, seed=1, mem=12, deps=(), extra="", prio=10, note=""):
    cmd = (f"{PY} -u studies/train_job.py --id {jid} --study {study} --cache {cache} --d-model {d} --layers {L} "
           f"--heads {h} --tokens {tokens} --lr {lr} --seed {seed} {extra}")
    if kv: cmd += f" --kv-heads {kv}"
    if note: cmd += f" --note '{note}'"
    return {"id": jid, "kind": "gpu", "cmd": cmd, "deps": list(deps) + [f"cache_{cache}"], "gpu_mem_gb": mem, "priority": prio}


def main():
    jobs = []
    # caches needed in wave 1/2
    jobs.append(cache_job("bpe50k_mixA_600M", 50000, MIX_A, prio=0))
    for v in (16000, 32000, 64000):
        jobs.append(cache_job(f"bpe{v//1000}k_mixA_600M", v, MIX_A, prio=1))
    jobs.append(cache_job("bpe50k_mixSC_600M", 50000, MIX_SC, tokenizer_from="bpe50k_mixA_600M", prio=2))
    jobs[-1]["deps"] = ["cache_bpe50k_mixA_600M"]
    jobs.append(cache_job("bpe50k_mixNAT_600M", 50000, MIX_NAT, tokenizer_from="bpe50k_mixA_600M", prio=2))
    jobs[-1]["deps"] = ["cache_bpe50k_mixA_600M"]
    # LR sweep (56.6M modern model, d512 L11 h8 kv2)
    lr_ids = []
    for lr in (1e-3, 2e-3, 4e-3):
        jid = f"LR_modern57M_lr{lr:g}"
        jobs.append(train_job(jid, "LR", "bpe50k_mixA_600M", 512, 11, 8, kv=2, lr=lr, prio=5, mem=10))
        lr_ids.append(jid)
    jobs.append({"id": "PLAN_W2", "kind": "cpu", "cmd": f"{PY} -u studies/plan_wave2.py", "deps": lr_ids, "priority": 6})
    (ROOT / "studies" / "jobs.json").write_text(json.dumps(jobs, indent=1))
    print(f"wrote {len(jobs)} jobs")


if __name__ == "__main__":
    main()
