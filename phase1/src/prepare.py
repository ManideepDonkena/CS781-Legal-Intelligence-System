"""Streaming ingestion; frozen global split; train-only tokenizers; per-source token streams."""
import argparse
import fcntl
import sqlite3
import os
from collections import Counter
from pathlib import Path
import numpy as np
from .common import *
from data import SOURCES, resolve_sources, ingest, train_tokenizer, tokenizer_diagnostics
from runner import DEFAULTS


def identity_differences(previous, current, prefix=""):
    """Describe cache-incompatible fields without changing the saved identity."""
    if isinstance(previous, dict) and isinstance(current, dict):
        changes = []
        for key in sorted(previous.keys() | current.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in previous:
                changes.append(f"{name}: added")
            elif key not in current:
                changes.append(f"{name}: removed")
            else:
                changes.extend(identity_differences(previous[key], current[key], name))
        return changes
    if previous == current:
        return []
    return [f"{prefix}: saved={str(previous)[:100]!r}, current={str(current)[:100]!r}"]


def require_same_identity(previous, current, label):
    changes = identity_differences(previous, current)
    if changes:
        raise ValueError(label + "\nChanged fields:\n  " + "\n  ".join(changes)
                         + "\nKeep the raw files. Restore the saved settings to resume, or use fresh --output"
                         " and --document-store directories (a fresh --work-dir for run_all.py)."
                         " Do not delete or edit the identity markers to bypass this check.")


def prepare(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS, sources=list(SOURCES), tokenizer=args.tokenizer, vocab_size=args.vocab_size,
               tokenizer_sample_bytes=args.sample_bytes, tokenizer_sample_strategy="balanced",
               seed=args.seed, validation_fraction=args.val_fraction,
               max_documents_per_source=args.max_docs_per_source)
    data_root = args.data_root
    if not data_root:
        if output.parent.name != "prepared" or output.parent.parent.name != "data":
            raise ValueError("Automatic data discovery requires --output WORK_DIR/data/prepared/VARIANT;"
                             " use that layout or provide the optional --data-root override")
        from .download import download
        raw = output.parent.parent / "raw"
        download(raw, args.revision)
        data_root = str(raw)
    paths = resolve_sources(argparse.Namespace(data_root=data_root, download=False,
                                              revision=args.revision), cfg["sources"])
    import importlib.metadata
    packages = {x: importlib.metadata.version(x) for x in ("tokenizers", "sentencepiece", "numpy", "ijson")}
    identity = {"config": cfg, "validation_pairs_per_source": args.validation_pairs_per_source,
                "source_hashes": {s: digest_file(p) for s, p in paths.items()},
                "code": fingerprint({str(p.relative_to(ROOT)): digest_file(p) for p in
                    (Path(__file__), ROOT / "core/data.py", ROOT / "core/tokenizer_io.py", ROOT / "core/runner.py")}),
                "packages": packages}
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "manifest.json"
        if marker.exists():
            previous = read(marker)
            require_same_identity(previous["identity"], identity, "Preparation settings/data/code changed")
            verify(output)
            return output
        partial = output / "preparation_identity.json"
        if partial.exists():
            require_same_identity(read(partial), identity, "Interrupted preparation belongs to other inputs")
        write_json(partial, identity)
        if args.document_store:
            store = Path(args.document_store).resolve()
            store.mkdir(parents=True, exist_ok=True)
            store_identity = {"sources": identity["source_hashes"], "seed": args.seed,
                              "val_fraction": args.val_fraction, "max_docs_per_source": args.max_docs_per_source,
                              "code": identity["code"]}
            with (store / ".lock").open("w") as store_lock:
                fcntl.flock(store_lock, fcntl.LOCK_EX)
                store_marker = store / "identity.json"
                if store_marker.exists():
                    require_same_identity(read(store_marker), store_identity, "Document-store inputs changed")
                if not store_marker.exists():
                    db = ingest(paths, store, cfg)
                    db.close()
                    write_json(store_marker, store_identity)
            link = output / "documents.sqlite"
            if not link.exists():
                link.symlink_to(os.path.relpath(store / "documents.sqlite", output))
            db = sqlite3.connect(link)
        else:
            db = ingest(paths, output, cfg)
        try:
            tok, stats = train_tokenizer(db, output, cfg)
            stats["diagnostics"] = tokenizer_diagnostics(db, tok, max_documents=100)
            # A bounded training-only lexicon for descriptive corpus overlap,
            # never used as a tokenizer vocabulary or a legal-validity score.
            import re
            lexicon, lexicon_bytes = Counter(), 0
            for (text,) in db.execute("SELECT text FROM docs WHERE split='train' ORDER BY digest"):
                lexicon.update(set(re.findall(r"[^\W\d_]{4,}", text.lower(), re.UNICODE)))
                lexicon_bytes += len(text.encode())
                if lexicon_bytes >= 5_000_000:
                    break
            write_json(output / "corpus_lexicon.json", dict(lexicon.most_common(5000)))
            streams = {}
            for split in ("train", "validation"):
                streams[split] = {}
                for source in SOURCES:
                    dest = output / f"{split}_{source}.bin"
                    count = docs = 0
                    with dest.open("wb") as f:
                        for _, text in db.execute("SELECT digest,text FROM docs WHERE split=? AND source=? ORDER BY digest", (split, source)):
                            ids = tok.encode(text).ids + [tok.token_to_id("<|endoftext|>")]
                            np.asarray(ids, dtype="<u4").tofile(f)
                            count += len(ids)
                            docs += 1
                    if not count:
                        raise ValueError(f"No {split} data for {source}; increase the source sample")
                    streams[split][source] = dict(file=dest.name, tokens=count, documents=docs, sha256=digest_file(dest))
            prompts, truth = {}, {}
            for source in SOURCES:
                selected = 0
                for key, text in db.execute("SELECT digest,text FROM docs WHERE split='validation' AND source=? ORDER BY digest", (source,)):
                    words = text.split()
                    if len(words) >= 640:
                        prompts[key] = {"text": " ".join(words[:512]), "source": source}
                        truth[key] = {"text": " ".join(words[512:640]), "source": source}
                        selected += 1
                        if selected >= args.validation_pairs_per_source:
                            break
            if not prompts:
                raise ValueError("No eligible held-out 512+128-word examples")
            write_json(output / "held_out_prompts.json", prompts)
            write_json(output / "held_out_truth.json", truth)
            manifest = {"identity": identity, "tokenizer": stats, "streams": streams,
                "audit": {s: json.loads(a) for s, a in db.execute("SELECT source,audit FROM progress")},
                "split_documents": dict(db.execute("SELECT split,COUNT(*) FROM docs GROUP BY split")),
                "held_out_prompts_sha256": digest_file(output / "held_out_prompts.json"),
                "held_out_truth_sha256": digest_file(output / "held_out_truth.json"),
                "policy": "NFKC-equivalent duplicates grouped globally before hash split; original text retained; tokenizer uses training only; per-source EOS streams; source tails dropped only at block loading",
                "scope": "full supplied files" if not args.max_docs_per_source else "bounded source prefixes"}
            write_json(marker, manifest)
        finally:
            db.close()
    return output


def verify(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    expected = {"tokenizer.json": manifest["tokenizer"]["sha256"],
                "held_out_prompts.json": manifest["held_out_prompts_sha256"],
                "held_out_truth.json": manifest["held_out_truth_sha256"]}
    for sources in manifest["streams"].values():
        expected.update({v["file"]: v["sha256"] for v in sources.values()})
    for name, sha in expected.items():
        if digest_file(directory / name) != sha:
            raise ValueError(f"Corrupt preparation artifact: {name}")
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--data-root", help="optional manual data override; default is WORK_DIR/data/raw inferred from --output")
    source.add_argument("--download", action="store_true", help="reuse/download into inferred WORK_DIR/data/raw (also the default)")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", required=True)
    p.add_argument("--document-store", help="shared on-disk document database across tokenizer variants")
    p.add_argument("--tokenizer", choices=KINDS, default="ayn_bpe")
    p.add_argument("--vocab-size", type=int, choices=[32000, 64000], default=64000)
    p.add_argument("--sample-bytes", type=int, default=100_000_000)
    p.add_argument("--max-docs-per-source", type=int, default=0, help="0 scans all documents")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--validation-pairs-per-source", type=int, default=20)
    return p


def main():
    args = parser().parse_args()
    if args.vocab_size < 300 or args.sample_bytes < 10000 or not 0 < args.val_fraction < 1 or args.max_docs_per_source < 0 or args.validation_pairs_per_source < 1:
        raise ValueError("Invalid vocabulary, sample, split or selection budget")
    print(prepare(args))

if __name__ == "__main__":
    main()
