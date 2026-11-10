"""Phase 7: Prior Case Retrieval (PCR) core utilities.
Loads data, computes Micro-F1@7 / Precision@7 / Recall@7, and validates submission formats.
"""
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Phase_7_PCR"
K = 7

# Common legal and English stopwords for clean tokenization
STOPWORDS = {
    'the', 'of', 'and', 'in', 'to', 'a', 'is', 'that', 'for', 'it', 'as', 'was', 'with', 'be', 'by', 'on',
    'not', 'he', 'i', 'this', 'are', 'or', 'his', 'from', 'at', 'which', 'but', 'they', 'an', 'had', 'have',
    'were', 'been', 'one', 'their', 'has', 'would', 'what', 'will', 'there', 'if', 'can', 'all', 'her',
    'who', 'out', 'up', 'so', 'said', 'them', 'she', 'no', 'when', 'time', 'my', 'then', 'some', 'we', 'any',
    'these', 'under', 'such', 'may', 'other', 'after', 'more', 'also', 'into', 'only', 'its', 'first',
    'name', 'org', 'petitioner', 'respondent', 'appellant', 'court', 'judgment', 'appeal', 'advocate',
    'counsel', 'learned', 'bench', 'state', 'dated', 'delhi', 'bombay', 'madras', 'calcutta', 'high',
    'supreme', 'india', 'order', 'petition', 'case', 'no', 'vs', 'ors', 'anr', 'justice'
}

PAT_SEC = re.compile(r'\b(?:section|sec|s\.)\s*(\d+[a-z]?)\b', re.IGNORECASE)
PAT_ART = re.compile(r'\b(?:article|art\.)\s*(\d+[a-z]?)\b', re.IGNORECASE)


def load_split(split: str, data_dir: Path = DATA_DIR):
    """Load query documents and candidate documents for a split ('train', 'dev', 'test')."""
    split_cap = split.capitalize()
    q_file = data_dir / split_cap / f"{split.lower()}.json"
    c_file = data_dir / split_cap / f"{split.lower()}_candidates.json"

    with open(q_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(c_file, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    return queries, candidates


def extract_sections(text: str) -> set[str]:
    """Extract statutory sections and constitutional articles."""
    secs = {'sec_' + s.lower() for s in PAT_SEC.findall(text)}
    arts = {'art_' + a.lower() for a in PAT_ART.findall(text)}
    return secs | arts


def tokenize(text_list: list[str], include_sections: bool = True) -> list[str]:
    """Tokenize text into lowercase alpha-numeric tokens and legal section tags."""
    text = " ".join(text_list).lower()
    tokens = re.findall(r'\b[a-z0-9_]{2,}\b', text)
    filtered = [t for t in tokens if t not in STOPWORDS]
    if include_sections:
        secs = ['sec_' + s.lower() for s in PAT_SEC.findall(text)]
        arts = ['art_' + a.lower() for a in PAT_ART.findall(text)]
        return filtered + secs * 2 + arts * 2
    return filtered


def evaluate_predictions(gold_queries: dict, predictions: dict) -> dict[str, float]:
    """Evaluate predictions against gold relevant candidates using Micro-F1@7."""
    tp = fp = fn = 0
    for qid, qdata in gold_queries.items():
        gold = set(qdata.get("relevant_candidates", []))
        pred = predictions.get(qid, [])[:K]
        p_set = set(pred)
        tp += len(gold & p_set)
        fp += len(p_set - gold)
        fn += len(gold - p_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "primary_metric_1": f1,       # Micro-F1@7
        "secondary_metric_1": precision, # Micro-Precision@7
        "secondary_metric_2": recall,    # Micro-Recall@7
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def validate_predictions(predictions: dict, candidate_ids: set[str], expected_qids: set[str]) -> list[str]:
    """Strict validation matching the official Codabench / evaluate_phase7.py criteria."""
    errors = []
    if not isinstance(predictions, dict):
        return ["prediction.json must be a JSON object mapping query IDs to lists of candidate IDs."]

    pred_ids = set(predictions.keys())
    missing = expected_qids - pred_ids
    extra = pred_ids - expected_qids
    if missing:
        errors.append(f"Missing {len(missing)} query IDs: {sorted(list(missing))[:5]}...")
    if extra:
        errors.append(f"Unexpected extra {len(extra)} query IDs: {sorted(list(extra))[:5]}...")

    for qid in sorted(expected_qids & pred_ids):
        ranked = predictions[qid]
        if not isinstance(ranked, list):
            errors.append(f"Query '{qid}': prediction must be a list.")
            continue
        if len(ranked) != K:
            errors.append(f"Query '{qid}': expected exactly {K} candidates, got {len(ranked)}.")
        seen = set()
        for i, cid in enumerate(ranked):
            if not isinstance(cid, str) or not cid.strip():
                errors.append(f"Query '{qid}': candidate index {i} is not a valid non-empty string.")
            elif cid in seen:
                errors.append(f"Query '{qid}': duplicate candidate ID '{cid}'.")
            seen.add(cid)
            if cid not in candidate_ids:
                errors.append(f"Query '{qid}': candidate ID '{cid}' is outside the valid candidate pool.")

    return errors


def package_submission(predictions: dict, sub_dir: Path, sub_name: str) -> Path:
    """Save prediction.json and package clean submission.zip at root."""
    sub_dir.mkdir(parents=True, exist_ok=True)
    pred_path = sub_dir / "prediction.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    zip_path = sub_dir / f"{sub_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")

    return zip_path
