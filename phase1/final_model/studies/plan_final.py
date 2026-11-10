#!/usr/bin/env python3
"""Final planner: runs after wave 2. Applies the predeclared decision rules and appends the final jobs.

Decision rule (per choice): take the lowest-BPB candidate, but keep the default unless the winner beats it
by more than 2*eps (eps = seed std from G3). The defaults are cheaper/simpler, so ties go to them.
  vocabulary  default 50k   candidates: VOC_16k/32k/64k + G3_base_s1 (=50k)
  KV heads    default 2/8   candidates: A3_kv8/4/1 + G3_base_s1
  mixture     default A     candidates: G3_base_s1 (A), C5_mixNAT   (SC-only is reported but capped, see below)
Final models (scratch, <=300M, official data only):
  FINAL_295M  d1024, layers from the parameter cap, 3B tokens (~10 tokens/param)
  FINAL_112M  d768 L12, 3B tokens (~27 tokens/param) - the faster, safer candidate
Each is followed by a dev score (official scorer on frozen held-out pairs) and a Codabench submission.
"""
import json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_jobs import cache_job, train_job, ROOT, PY, MIX_A, MIX_NAT  # noqa: E402

R = {}
for l in (ROOT / "results" / "results.jsonl").read_text().splitlines():
    r = json.loads(l); R[r["id"]] = r
bpb = lambda i: R[i]["val_bpb"]
seeds = [bpb(f"G3_base_s{s}") for s in (1, 2, 3) if f"G3_base_s{s}" in R]
eps = st.stdev(seeds) if len(seeds) >= 2 else 0.01
print(f"eps={eps:.4f} from {len(seeds)} seeds; base mean={st.mean(seeds):.4f}")


def pick(cands, default):
    cands = {k: v for k, v in cands.items() if k in R}
    best = min(cands, key=bpb)
    chosen = default if bpb(default) - bpb(best) <= 2 * eps else best
    print(f"  {sorted(((round(bpb(k),4), k) for k in cands))} -> {chosen}")
    return chosen


base = "G3_base_s1"
vocab_run = pick({base: 50, "VOC_16k_L17": 16, "VOC_32k_L14": 32, "VOC_64k_L8": 64}, base)
vocab = {base: 50000, "VOC_16k_L17": 16000, "VOC_32k_L14": 32000, "VOC_64k_L8": 64000}[vocab_run]
kv_run = pick({base: 2, "A3_kv8": 8, "A3_kv4": 4, "A3_kv1": 1}, base)
kv8 = {base: 2, "A3_kv8": 8, "A3_kv4": 4, "A3_kv1": 1}[kv_run]       # kv heads when heads=8
mix_run = pick({base: "A", "C5_mixNAT": "NAT"}, base)
mix_name, mix = ("mixA", MIX_A) if mix_run == base else ("mixNAT", MIX_NAT)
lrs = [R[i] for i in R if R[i]["study"] == "LR"]
lr = min(lrs, key=lambda r: r["val_bpb"])["lr"]
print(f"chosen: vocab={vocab} kv_ratio=8/{kv8} mix={mix_name} lr57M={lr}")

# parameter-cap aware depth for the 295M model (d=1024, 16 heads)
d, h = 1024, 16
kv16 = kv8 * 2
hidden = ((int(2 * 4 * d / 3) + 63) // 64) * 64
per_layer = d * d * 2 + 2 * d * (d // h) * kv16 + 3 * d * hidden + 2 * d + 2 * (d // h)
L295 = int((298e6 - vocab * d) // per_layer)
print(f"FINAL_295M: layers={L295} (~{(vocab*d + L295*per_layer)/1e6:.1f}M params)")

tok_src = f"bpe{vocab//1000}k_mixA_600M"
cname = f"final_bpe{vocab//1000}k_{mix_name}_3B"
jobs_f = ROOT / "studies" / "jobs.json"
jobs = json.loads(jobs_f.read_text()); have = {j["id"] for j in jobs}
J = [cache_job(cname, vocab, mix, tokens=3_000_000_000, tokenizer_from=tok_src, prio=20)]
J[0]["deps"] = []
J.append(train_job("FINAL_295M", "FINAL", cname, d, L295, h, kv=kv16, tokens=3_000_000_000, lr=lr * 0.5, mem=34,
                   extra="--bs 8 --ga 4 --keep-best --evals 30", prio=21))
J.append(train_job("FINAL_112M", "FINAL", cname, 768, 12, 12, kv=max(1, 12 * kv8 // 8), tokens=3_000_000_000, lr=lr * 0.7,
                   mem=18, extra="--keep-best --evals 30", prio=21))
for run, sid, name in (("FINAL_295M", "S101_modern295M_minp", "Phase 1 - Modern YaRN-GQA 295M (SwiGLU, QK-norm, RoPE) Min-P"),
                       ("FINAL_112M", "S102_modern113M_minp", "Phase 1 - Modern YaRN-GQA 113M (SwiGLU, QK-norm, RoPE) Min-P")):
    J.append({"id": f"DEV_{run}", "kind": "gpu", "gpu_mem_gb": 14, "priority": 22, "deps": [run],
              "cmd": f"{PY} -u studies/gen_modern.py --run-dir runs/studies/{run} --gpu 0 --dev-cache {cname} --out-dir results/dev_{run}"})
    J.append({"id": f"GEN_{run}", "kind": "gpu", "gpu_mem_gb": 14, "priority": 22, "deps": [run],
              "cmd": (f"{PY} -u studies/gen_modern.py --run-dir runs/studies/{run} --gpu 0 --sub-id {sid} "
                      f"--name '{name}' --desc 'Scratch-trained on official data only; 3B tokens; decoding from S10 ablation' "
                      f"--out-dir codabench_submissions/{sid}")})
J.append({"id": "REPORT", "kind": "cpu", "cmd": f"{PY} -u studies/report.py", "priority": 30,
          "deps": ["DEV_FINAL_295M", "DEV_FINAL_112M", "GEN_FINAL_295M", "GEN_FINAL_112M"]})
jobs += [j for j in J if j["id"] not in have]
jobs_f.write_text(json.dumps(jobs, indent=1))
print("appended final jobs")
