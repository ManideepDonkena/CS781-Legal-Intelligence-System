#!/usr/bin/env python3
"""Show what a model writes after a few short prompts (general knowledge, questions, legal text).

A quick, informal look at whether a model has general language ability; it is not a score.

Example:
  python studies/probe_prompts.py --gpu 4 --runs /workspace/models/cs781_phase1_runs/WIKI_56M runs/studies/FINAL_295M
"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "studies"))

PROMPTS = [
    "The capital of France is",
    "Water is made of",
    "The Second World War ended in the year",
    "A dog is an animal that",
    "Question: What is the capital of India? Answer:",
    "Article 213 of the Constitution of India deals with",
    "What is Article 21 of the Constitution of India? Article 21",
    "The appellant was convicted under Section 302 of the Indian Penal Code and",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories holding best.pt and tokenizer.json")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--words", type=int, default=35)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--prompt", action="append", help="extra prompt (repeatable)")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    from gen_modern import load_model, generate_batch

    prompts = PROMPTS + (a.prompt or [])
    for run in a.runs:
        model, cfg, tok = load_model(run, "best.pt", torch.device("cuda"))
        print(f"\n=== {run}  (attention fix: {bool(cfg.get('rope_float'))})")
        for p in prompts:
            ids = tok.encode(p).ids
            text = generate_batch(model, tok, [ids], temp=a.temperature, max_new=3 * a.words, target_words=a.words)[0]
            print(f"  [{p}] -> {text}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
