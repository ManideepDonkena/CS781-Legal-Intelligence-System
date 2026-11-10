#!/usr/bin/env python3
"""Build a token cache from text outside the official corpus (e.g. Wikipedia parquet files).

The text is encoded with the tokenizer of an existing cache, so a model trained on this cache can
continue on that one (`train_job.py --init-from`).  `validation.bin` holds held-out documents of the
new text; the per-source legal validation files and the held-out legal prompts of the tokenizer's
cache are reused, so every run is still scored on the same legal text.

The cache is written under --out-root (the root disk is nearly full) and linked into
artifacts/caches/<name>.  Needs pyarrow: run with the `cs781` conda environment.

Example:
  python studies/build_text_cache.py --name bpe50k_wiki_600M --tokenizer-from bpe50k_mixA_600M \
      --parquet '/workspace/models/cs781_data/wikipedia_en/*.parquet' --train-tokens 600000000
"""
import argparse, glob, json, shutil, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from data import write_json, digest_file, fingerprint, PREPARATION_KEYS  # noqa: E402

CACHES = ROOT / "artifacts" / "caches"
EOS = "<|endoftext|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--parquet", required=True, help="glob of parquet files, read in sorted order")
    ap.add_argument("--source", default="wikipedia_en", help="label recorded in the manifest")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--id-column", default="id")
    ap.add_argument("--tokenizer-from", required=True, help="existing cache whose tokenizer and legal validation are reused")
    ap.add_argument("--train-tokens", type=int, required=True)
    ap.add_argument("--val-tokens", type=int, default=4_000_000)
    ap.add_argument("--min-chars", type=int, default=500, help="skip stub documents shorter than this")
    ap.add_argument("--batch-docs", type=int, default=512)
    ap.add_argument("--out-root", default="/workspace/models/cs781_data/caches")
    a = ap.parse_args()

    base = CACHES / a.tokenizer_from
    base_manifest = json.loads((base / "manifest.json").read_text())
    out = Path(a.out_root) / a.name
    link = CACHES / a.name
    if (out / "manifest.json").exists():
        print("already built:", out); return
    out.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(out, target_is_directory=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    shutil.copyfile(base / "tokenizer.json", out / "tokenizer.json")
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    eos = tok.token_to_id(EOS)
    dtype = base_manifest["tokens"].get("dtype", "<u4")
    assert dtype == "<u2" and tok.get_vocab_size() <= 65535, "expects a uint16 cache"

    # One pass over the files: every 20th document (by id) goes to validation until it is full.
    quota = {"train": a.train_tokens, "validation": a.val_tokens}
    stats = {k: {"tokens": 0, "raw_bytes": 0, "documents": 0} for k in quota}
    handles = {k: (out / f"{k}.bin").open("wb") for k in quota}
    skipped = 0

    def flush(split, texts):
        for text, enc in zip(texts, tok.encode_batch(texts)):
            room = quota[split] - stats[split]["tokens"]
            if room <= 0:
                return
            ids = (enc.ids + [eos])[:room]
            np.asarray(ids, dtype=dtype).tofile(handles[split])
            stats[split]["tokens"] += len(ids)
            # bytes are counted in proportion to the tokens kept, so bits-per-byte stays exact
            stats[split]["raw_bytes"] += round(len(text.encode()) * len(ids) / (len(enc.ids) + 1))
            stats[split]["documents"] += 1

    files = sorted(glob.glob(a.parquet))
    assert files, f"no files match {a.parquet}"
    for path in files:
        if all(stats[k]["tokens"] >= quota[k] for k in quota):
            break
        for batch in pq.ParquetFile(path).iter_batches(batch_size=a.batch_docs, columns=[a.id_column, a.text_column]):
            pending = {"train": [], "validation": []}
            for doc_id, text in zip(batch.column(a.id_column).to_pylist(), batch.column(a.text_column).to_pylist()):
                text = (text or "").strip()
                if len(text) < a.min_chars:
                    skipped += 1
                    continue
                pending["validation" if int(doc_id) % 20 == 0 else "train"].append(text)
            for split, texts in pending.items():
                if texts and stats[split]["tokens"] < quota[split]:
                    flush(split, texts)
            if stats["train"]["tokens"] >= quota["train"] and stats["validation"]["tokens"] >= quota["validation"]:
                break
        log(f"{Path(path).name}: train {stats['train']['tokens'] / 1e6:,.0f}M / {a.train_tokens / 1e6:,.0f}M tokens, "
            f"validation {stats['validation']['tokens'] / 1e6:.1f}M, {stats['train']['tokens'] / (time.time() - t0) / 1e6:.2f}M tok/s")
    for fh in handles.values():
        fh.close()
    assert stats["train"]["tokens"] == a.train_tokens, f"only {stats['train']['tokens']:,} train tokens available; add files"
    assert stats["validation"]["tokens"] == a.val_tokens, f"only {stats['validation']['tokens']:,} validation tokens available"

    tokens = {"dtype": dtype, "mixture_epochs": {a.source: 0}}
    for split in quota:
        tokens[split] = {**stats[split], "source_tokens": {a.source: stats[split]["tokens"]},
                         "sha256": digest_file(out / f"{split}.bin")}
    # legal validation files and held-out prompts of the tokenizer's cache, unchanged
    tokens["validation_by_source"] = base_manifest["tokens"]["validation_by_source"]
    for s in tokens["validation_by_source"]:
        shutil.copyfile(base / f"validation_{s}.bin", out / f"validation_{s}.bin")
    for name in ("held_out_prompts", "held_out_truth"):
        shutil.copyfile(base / f"{name}.json", out / f"{name}.json")
        tokens[f"{name}_sha256"] = digest_file(out / f"{name}.json")
    tokens["held_out_pairs"] = base_manifest["tokens"]["held_out_pairs"]

    settings = dict(base_manifest["identity"]["settings"])
    settings.update(sources=[a.source], train_tokens=a.train_tokens, val_tokens=a.val_tokens)
    assert set(settings) == set(PREPARATION_KEYS)
    identity = {"settings": settings, "mixture": {a.source: 1.0}, "name": a.name, "builder": digest_file(__file__),
                "tokenizer_from": a.tokenizer_from, "files": [Path(f).name for f in files]}
    tok_stats = dict(base_manifest["tokenizer"], reused_from=a.tokenizer_from)
    manifest = {"cache_id": fingerprint(identity), "identity": identity, "tokenizer": tok_stats, "tokens": tokens,
                "mixture": {a.source: 1.0}, "build_seconds": time.time() - t0, "skipped_short_documents": skipped,
                "policy": f"{a.source}: documents under {a.min_chars} characters skipped; every 20th document id held "
                          "out for validation; EOS packed; legal per-source validation reused from the tokenizer's cache"}
    write_json(out / "manifest.json", manifest)
    log(f"DONE {out} in {time.time() - t0:.0f}s; {stats['train']['documents']:,} train documents, "
        f"{stats['train']['raw_bytes'] / stats['train']['tokens']:.2f} bytes per token")


if __name__ == "__main__":
    main()
