#!/usr/bin/env python3
"""Wave-2 planner: runs after the LR sweep, picks the best LR and appends the study jobs to jobs.json.

Studies (all ~57M params unless stated, 600M tokens = ~1 epoch, scored on frozen Supreme Court validation BPB):
  G3   seed noise: 3 seeds of the base model                         -> epsilon for every decision rule
  ARCH legacy Transformer vs +init fix vs +init+optimizer fix vs modern (base)
  H3   weight tying (57M and 113M)
  A3   KV heads 8(MHA)/4/2(base)/1(MQA) and MQA with reinvested layer
  H2   width/depth allocation at equal params; vocabulary 16k/32k/50k/64k with depth re-balanced to equal params
  C4   1/2/4 epochs;  C5 source mixture (domain-weighted / SC-only / natural)
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_jobs import train_job, ROOT, PY  # noqa: E402


def best_lr():
    rows = [json.loads(l) for l in (ROOT / "results" / "results.jsonl").read_text().splitlines()]
    lr = [r for r in rows if r["study"] == "LR" and r["complete"]]
    best = min(lr, key=lambda r: r["val_bpb"])
    print("LR sweep:", {r["lr"]: round(r["val_bpb"], 4) for r in lr}, "-> best", best["lr"])
    return best["lr"]


def main():
    lr = best_lr()
    C = "bpe50k_mixA_600M"
    J = []
    J += [train_job(f"G3_base_s{s}", "G3", C, 512, 11, 8, kv=2, lr=lr, seed=s, prio=10) for s in (1, 2, 3)]
    # architecture fixes (legacy 'transformer' family: LayerNorm, GELU, learned positions; d512 L10 ~ 58M)
    leg = dict(lr=lr, mem=22, prio=11)
    J.append(train_job("ARCH_legacy", "ARCH", C, 512, 10, 8, extra="--legacy-optim --no-compile", **leg) | {})
    J[-1]["cmd"] = J[-1]["cmd"].replace("--tokens", "--family transformer --tokens", 1)
    J.append(train_job("ARCH_legacy_initfix", "ARCH", C, 512, 10, 8, extra="--legacy-optim --init-fix --no-compile", **leg))
    J[-1]["cmd"] = J[-1]["cmd"].replace("--tokens", "--family transformer --tokens", 1)
    J.append(train_job("ARCH_legacy_initfix_optim", "ARCH", C, 512, 10, 8, extra="--init-fix --no-compile", **leg))
    J[-1]["cmd"] = J[-1]["cmd"].replace("--tokens", "--family transformer --tokens", 1)
    # H3 tying
    J.append(train_job("H3_untied_57M", "H3", C, 512, 11, 8, kv=2, lr=lr, extra="--untie", prio=12))
    J.append(train_job("H3_tied_113M", "H3", C, 768, 12, 12, kv=3, lr=lr * 0.7, mem=16, prio=12))
    J.append(train_job("H3_untied_113M", "H3", C, 768, 12, 12, kv=3, lr=lr * 0.7, mem=18, extra="--untie", prio=12))
    # A3 kv heads
    for kv in (8, 4, 1):
        J.append(train_job(f"A3_kv{kv}", "A3", C, 512, 11, 8, kv=kv, lr=lr, prio=12))
    J.append(train_job("A3_kv1_L12", "A3", C, 512, 12, 8, kv=1, lr=lr, prio=12))
    # H2 depth/width
    J.append(train_job("H2_deep_d384_L24", "H2", C, 384, 24, 6, kv=1, lr=lr, prio=12))
    J.append(train_job("H2_wide_d768_L3", "H2", C, 768, 3, 12, kv=3, lr=lr, prio=12))
    # vocabulary (depth re-balanced to ~57M)
    for v, L in ((16, 17), (32, 14), (64, 8)):
        J.append(train_job(f"VOC_{v}k_L{L}", "VOC", f"bpe{v}k_mixA_600M", 512, L, 8, kv=2, lr=lr, prio=12))
    # C4 epochs (cache holds 600M tokens, so 1.2B/2.4B tokens = 2/4 passes)
    J.append(train_job("C4_ep2", "C4", C, 512, 11, 8, kv=2, lr=lr, tokens=1_200_000_000, prio=13))
    J.append(train_job("C4_ep4", "C4", C, 512, 11, 8, kv=2, lr=lr, tokens=2_400_000_000, prio=14))
    # C5 mixtures (same tokenizer)
    J.append(train_job("C5_mixSC", "C5", "bpe50k_mixSC_600M", 512, 11, 8, kv=2, lr=lr, prio=12))
    J.append(train_job("C5_mixNAT", "C5", "bpe50k_mixNAT_600M", 512, 11, 8, kv=2, lr=lr, prio=12))
    ids = [j["id"] for j in J]
    J.append({"id": "PLAN_FINAL", "kind": "cpu", "cmd": f"{PY} -u studies/plan_final.py", "deps": ids, "priority": 15})
    jobs_f = ROOT / "studies" / "jobs.json"
    jobs = json.loads(jobs_f.read_text())
    have = {j["id"] for j in jobs}
    jobs += [j for j in J if j["id"] not in have]
    jobs_f.write_text(json.dumps(jobs, indent=1))
    print(f"appended {sum(j['id'] not in have for j in J)} wave-2 jobs (lr={lr})")


if __name__ == "__main__":
    main()
