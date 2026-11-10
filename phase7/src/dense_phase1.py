"""Dense Semantic Retrieval using our own pre-trained Phase 1 295M Legal Model (SC_2EPOCHS_DEEP).
Completely from-scratch architecture and pre-trained weights from Phase 1.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/mani/cs781_phase1/Legal_LLM")
sys.path.insert(0, "/home/mani/cs781/Capstone/shared")

import pcr  # noqa: E402
from model_zoo import build_model  # noqa: E402
from scratch_bpe import ScratchBPE  # noqa: E402

CKPT_DIR = "/home/mani/cs781_phase1/Legal_LLM/runs/studies/SC_2EPOCHS_DEEP"


def load_phase1_model(ckpt_dir: str, device: str):
    """Load our Phase 1 checkpoint and custom BPE tokenizer."""
    print(f"[*] Loading Phase 1 model from {ckpt_dir} on {device}...")
    ck = torch.load(f"{ckpt_dir}/best.pt", map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    family = cfg.pop("family")
    cfg["gradient_checkpointing"] = False
    lm = build_model(family, **cfg).to(device)
    lm.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    lm.eval()

    bpe = ScratchBPE(f"{ckpt_dir}/tokenizer.json")
    return lm, bpe


def encode_split_phase1(lm, bpe, texts: list[str], device: str, max_tokens: int = 512, batch_size: int = 16) -> torch.Tensor:
    """Encode texts using our Phase 1 model's causal transformer hidden states."""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_tokens = []
        max_b_len = 0
        for t in batch_texts:
            toks = bpe.encode(t)[:max_tokens]
            if not toks:
                toks = [0]
            batch_tokens.append(toks)
            max_b_len = max(max_b_len, len(toks))

        # Pad with zeros
        padded = []
        for toks in batch_tokens:
            pad_len = max_b_len - len(toks)
            padded.append(toks + [0] * pad_len)

        inp = torch.tensor(padded, dtype=torch.long, device=device)
        with torch.no_grad():
            out = lm(inp, return_hidden_states=True)
            # Masked mean pooling over non-padded tokens
            hidden = out["hidden_states"]  # (B, L, D)
            mask = (inp != 0).unsqueeze(-1).float()
            sum_hidden = (hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1.0)
            mean_hidden = sum_hidden / lengths
            norm_e = torch.nn.functional.normalize(mean_hidden, p=2, dim=-1)
            embs.append(norm_e.cpu())

    return torch.cat(embs, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 SC 295M Dense Retriever")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Split to run on")
    parser.add_argument("--gpu", type=int, default=3, help="CUDA GPU device index")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per case document")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--out-dir", type=str, default=str(HERE.parent / "runs" / "dense_phase1"), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print(f"[*] Loading {args.split} split data...")
    queries, candidates = pcr.load_split(args.split)
    q_ids = sorted(list(queries.keys()))
    c_ids = sorted(list(candidates.keys()))
    print(f"[*] Loaded {len(q_ids)} queries and {len(c_ids)} candidates.")

    lm, bpe = load_phase1_model(CKPT_DIR, device)

    q_texts = [" ".join(queries[qid]["text"]) for qid in q_ids]
    c_texts = [" ".join(candidates[cid]["text"]) for cid in c_ids]

    print(f"[*] Encoding {len(q_texts)} queries with Phase 1 295M LM...")
    t0 = time.time()
    q_embs = encode_split_phase1(lm, bpe, q_texts, device, max_tokens=args.max_tokens, batch_size=args.batch_size)
    print(f"[✓] Queries encoded in {time.time() - t0:.2f}s.")

    print(f"[*] Encoding {len(c_texts)} candidates with Phase 1 295M LM...")
    t1 = time.time()
    c_embs = encode_split_phase1(lm, bpe, c_texts, device, max_tokens=args.max_tokens, batch_size=args.batch_size)
    print(f"[✓] Candidates encoded in {time.time() - t1:.2f}s.")

    print(f"[*] Computing cosine similarity matrix...")
    sim_matrix = torch.matmul(q_embs, c_embs.T).numpy()

    preds = {}
    full_scores = {}
    for q_idx, qid in enumerate(q_ids):
        scores = {cid: float(sim_matrix[q_idx, c_idx]) for c_idx, cid in enumerate(c_ids)}
        full_scores[qid] = scores
        top7 = sorted(c_ids, key=lambda c: scores[c], reverse=True)[:pcr.K]
        preds[qid] = top7

    scores_path = out_dir / f"{args.split}_scores.json"
    preds_path = out_dir / f"{args.split}_predictions.json"
    embs_path = out_dir / f"{args.split}_embeddings.pt"
    torch.save({"q_ids": q_ids, "c_ids": c_ids, "q_embs": q_embs, "c_embs": c_embs}, embs_path)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(full_scores, f)
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

    print(f"[✓] Saved embeddings to {embs_path}")
    print(f"[✓] Saved full scores to {scores_path}")
    print(f"[✓] Saved predictions to {preds_path}")

    if args.split == "dev":
        res = pcr.evaluate_predictions(queries, preds)
        print("\n==========================================")
        print(f"Dense Phase 1 295M Model Dev Results:")
        print(f"  Micro-F1@7:        {res['primary_metric_1']:.6f}")
        print(f"  Micro-Precision@7: {res['secondary_metric_1']:.6f}")
        print(f"  Micro-Recall@7:    {res['secondary_metric_2']:.6f}")
        print(f"  TP={res['tp']} FP={res['fp']} FN={res['fn']}")
        print("==========================================\n")


if __name__ == "__main__":
    main()
