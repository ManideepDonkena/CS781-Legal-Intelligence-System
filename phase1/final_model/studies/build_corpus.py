#!/usr/bin/env python3
"""Ingest one official source into a per-source SQLite document store (dedup + hash split).

One process per source so the four sources ingest in parallel. Reuses experiments/data.py
`ingest`, so the content-hash split (seed 42, 5% validation) is identical to every earlier run.
Usage: python studies/build_corpus.py acts|high_courts|supreme_court|tribunals
"""
import sys, json, glob, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from data import ingest, write_json

SNAP = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--Exploration-Lab--CS-781-Capstone/snapshots/*/Phase_1_Pretraining/Train"))[0]
CFG = dict(seed=42, validation_fraction=0.05, max_document_bytes=16_000_000, max_invalid_fraction=0.05,
           max_documents_per_source=0, max_source_text_bytes=0)

if __name__ == "__main__":
    source = sys.argv[1]
    out = ROOT / "artifacts" / "corpus" / source
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    db = ingest({source: Path(SNAP) / f"train_{source}.json"}, out, CFG)
    audit = {s: json.loads(a) for s, a in db.execute("SELECT source,audit FROM progress")}
    counts = dict(db.execute("SELECT split,COUNT(*) FROM docs GROUP BY split"))
    write_json(out / "corpus_manifest.json", {"source": source, "audit": audit, "split_documents": counts,
                                              "cfg": CFG, "seconds": time.time() - t})
    print("DONE", source, counts, f"{time.time()-t:.0f}s", flush=True)
