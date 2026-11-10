"""Average the Phase 2 adaptation study (run_adapt_study.sh) over the 5 held-out splits.

  python summarize_adapt.py            -> table on screen + ../runs/adapt/summary.json
"""
import glob
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "adapt")
NAMES = ["frozen", "top6", "full", "lora", "lorma", "student", "kd"]
out = {}
for name in NAMES:
    logs = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/{name}_k*/log.json"))]
    if not logs:
        continue
    raw = np.array([x["macro_f1"]["raw"] for x in logs]) * 100
    prop = np.array([x["macro_f1"]["prop0.34"] for x in logs]) * 100
    out[name] = {
        "splits": len(logs),
        "backbone_params_M": logs[0]["backbone_params"] / 1e6,
        "trainable_params_M": logs[0]["trainable_params"] / 1e6,
        "f1_raw_mean": raw.mean(), "f1_raw_sd": raw.std(ddof=1) if len(raw) > 1 else 0.0,
        "f1_prop_mean": prop.mean(), "f1_prop_sd": prop.std(ddof=1) if len(prop) > 1 else 0.0,
        "f1_prop_per_split": prop.round(1).tolist(),
        "peak_train_gb": float(np.mean([x["peak_train_gb"] for x in logs])),
        "train_min": float(np.mean([x["train_minutes"] for x in logs])),
        "infer_ktok_per_s": float(np.mean([x["eval_tokens"] / x["infer_seconds"] / 1e3 for x in logs])),
    }
    o = out[name]
    print(f"{name:8s} n={o['splits']} backbone {o['backbone_params_M']:6.1f}M trainable {o['trainable_params_M']:7.2f}M "
          f"raw {o['f1_raw_mean']:.1f}±{o['f1_raw_sd']:.1f}  copied {o['f1_prop_mean']:.1f}±{o['f1_prop_sd']:.1f} "
          f"{o['f1_prop_per_split']}  mem {o['peak_train_gb']:.1f}GB  train {o['train_min']:.1f}min  "
          f"infer {o['infer_ktok_per_s']:.1f}k tok/s")
json.dump(out, open(f"{R}/summary.json", "w"), indent=1)
