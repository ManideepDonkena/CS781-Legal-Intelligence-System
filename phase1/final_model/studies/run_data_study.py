#!/usr/bin/env python3
"""Run Data Quality comparative experiment on GPU 1.

Trains the best modern architecture (56.6M params, 11x512, kv=2, lr=0.004, seed=1)
on the newly cleaned cache `bpe50k_mixA_clean_600M` and evaluates on the identical
frozen validation benchmark to measure the impact of data quality.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/mani/miniconda3/envs/phase_1_capstone/bin/python"
CACHE_DIR = ROOT / "artifacts" / "caches" / "bpe50k_mixA_clean_600M"

def wait_for_cache():
    print(f"[{time.strftime('%H:%M:%S')}] Waiting for cache {CACHE_DIR.name}...", flush=True)
    manifest = CACHE_DIR / "manifest.json"
    while not manifest.exists():
        time.sleep(10)
    print(f"[{time.strftime('%H:%M:%S')}] Cache ready! Starting training on GPU 1...", flush=True)

def train_clean_model():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    cmd = [
        PYTHON, "-u", str(ROOT / "studies" / "train_job.py"),
        "--id", "DATA_clean_modern57M",
        "--study", "DATA",
        "--cache", "bpe50k_mixA_clean_600M",
        "--family", "yarn_gqa",
        "--d-model", "512",
        "--layers", "11",
        "--heads", "8",
        "--kv-heads", "2",
        "--tokens", "600000000",
        "--lr", "0.004",
        "--seed", "1",
    ]
    
    print(f"Executing: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] Training finished! Updating study report...", flush=True)
    subprocess.run([PYTHON, str(ROOT / "studies" / "report.py")], check=True)
    print(f"[{time.strftime('%H:%M:%S')}] Study report updated!", flush=True)

if __name__ == "__main__":
    wait_for_cache()
    train_clean_model()
