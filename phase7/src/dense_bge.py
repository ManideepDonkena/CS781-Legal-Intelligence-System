"""Dense Semantic Retrieval using BAAI/bge-m3.
Encodes long legal texts into dense normalized embeddings and computes cosine similarity matrices.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pcr  # noqa: E402


def encode_split(model, tokenizer, texts: list[str], device: str, max_len: int = 1024, batch_size: int = 16) -> torch.Tensor:
    """Encode a list of legal documents into L2-normalized mean-pooled embeddings."""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, max_length=max_len, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over token embeddings
            e = outputs.last_hidden_state.mean(dim=1)
            e = torch.nn.functional.normalize(e, p=2, dim=1)
            embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Dense BGE-M3 Retriever")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Split to run on")
    parser.add_argument("--gpu", type=int, default=5, help="CUDA GPU device index")
    parser.add_argument("--max-len", type=int, default=1024, help="Max sequence length for transformer")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--out-dir", type=str, default=str(HERE.parent / "runs" / "dense_bge"), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print(f"[*] Loading {args.split} split data...")
    queries, candidates = pcr.load_split(args.split)
    q_ids = sorted(list(queries.keys()))
    c_ids = sorted(list(candidates.keys()))
    print(f"[*] Loaded {len(q_ids)} queries and {len(c_ids)} candidates.")

    print(f"[*] Loading BAAI/bge-m3 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3").to(device)
    model.eval()

    q_texts = [" ".join(queries[qid]["text"]) for qid in q_ids]
    c_texts = [" ".join(candidates[cid]["text"]) for cid in c_ids]

    print(f"[*] Encoding {len(q_texts)} queries (max_len={args.max_len})...")
    t0 = time.time()
    q_embs = encode_split(model, tokenizer, q_texts, device, max_len=args.max_len, batch_size=args.batch_size)
    print(f"[✓] Queries encoded in {time.time() - t0:.2f}s.")

    print(f"[*] Encoding {len(c_texts)} candidates (max_len={args.max_len})...")
    t1 = time.time()
    c_embs = encode_split(model, tokenizer, c_texts, device, max_len=args.max_len, batch_size=args.batch_size)
    print(f"[✓] Candidates encoded in {time.time() - t1:.2f}s.")

    print(f"[*] Computing dense cosine similarity matrix...")
    sim_matrix = torch.matmul(q_embs, c_embs.T).numpy()

    # Build predictions and scores dict
    preds = {}
    full_scores = {}
    for q_idx, qid in enumerate(q_ids):
        scores = {cid: float(sim_matrix[q_idx, c_idx]) for c_idx, cid in enumerate(c_ids)}
        full_scores[qid] = scores
        top7 = sorted(c_ids, key=lambda c: scores[c], reverse=True)[:pcr.K]
        preds[qid] = top7

    # Save scores and predictions
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
        print(f"Dense BGE-M3 Dev Results:")
        print(f"  Micro-F1@7:        {res['primary_metric_1']:.6f}")
        print(f"  Micro-Precision@7: {res['secondary_metric_1']:.6f}")
        print(f"  Micro-Recall@7:    {res['secondary_metric_2']:.6f}")
        print(f"  TP={res['tp']} FP={res['fp']} FN={res['fn']}")
        print("==========================================\n")


if __name__ == "__main__":
    main()
