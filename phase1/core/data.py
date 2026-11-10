"""Disk-backed, training-only preparation shared by the single-run experiments.

RAM scales with one parsed document, the bounded tokenizer sample, and BPE's
vocabulary statistics, not the whole corpus. SQLite keeps texts and provenance
on disk. Source JSON arrays are parsed incrementally with ijson.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sqlite3
import re
import unicodedata
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
from tokenizer_io import KINDS, SentencePieceBPE, load_tokenizer, normalized_text, train_sentencepiece

SOURCES = {s: f"Phase_1_Pretraining/Train/train_{s}.json" for s in
           ("acts", "high_courts", "supreme_court", "tribunals")}
REPO = "Exploration-Lab/CS-781-Capstone"


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def records(path):
    """Accept the starter's array-of-records schema or explicitly named JSONL."""
    if Path(path).suffix == ".jsonl":
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        import ijson
        with Path(path).open("rb") as f:
            first = f.read(1)
            while first and first.isspace():
                first = f.read(1)
            if first != b"[":
                raise ValueError(f"Expected a JSON array in {path}; inspect this source's actual schema")
            f.seek(0)
            yield from ijson.items(f, "item")


def resolve_sources(args, sources):
    paths = {}
    for source in sources:
        if args.download:
            from huggingface_hub import hf_hub_download
            # HF_TOKEN is read by the library. Never serialize credentials.
            path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=SOURCES[source],
                                   revision=args.revision)
        else:
            if not args.data_root:
                raise ValueError("Pass --data-root with the supplied corpus, or --download")
            root = Path(args.data_root).expanduser().resolve()
            candidates = [root / SOURCES[source], root / Path(SOURCES[source]).name,
                          root / (Path(SOURCES[source]).stem + ".jsonl")]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Missing {source}: expected {candidates[0]} or {candidates[1]}")
        paths[source] = Path(path).resolve()
    return paths


def ingest(paths, directory, cfg):
    db = sqlite3.connect(directory / "documents.sqlite")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("CREATE TABLE IF NOT EXISTS docs (digest TEXT PRIMARY KEY, split TEXT, source TEXT, text TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS provenance (source TEXT, row_id INTEGER, digest TEXT, PRIMARY KEY(source,row_id))")
    db.execute("CREATE TABLE IF NOT EXISTS progress (source TEXT PRIMARY KEY, row_id INTEGER, audit TEXT)")
    for source, path in paths.items():
        prev = db.execute("SELECT row_id,audit FROM progress WHERE source=?", (source,)).fetchone()
        last, audit = prev if prev else (-1, json.dumps(dict(records=0, valid=0, unique=0, duplicate=0,
                                                           missing=0, nonstring=0, empty=0, malformed=0)))
        audit = json.loads(audit)
        audit.setdefault("selected_text_bytes", 0)
        complete_source = True
        for i, row in enumerate(records(path)):
            if i <= last:
                continue  # recover committed work; JSON must be rescanned to this record
            if ((cfg.get("max_documents_per_source", 0) and audit["records"] >= cfg["max_documents_per_source"])
                    or (cfg.get("max_source_text_bytes", 0) and audit["selected_text_bytes"] >= cfg["max_source_text_bytes"])):
                complete_source = False
                break
            audit["records"] += 1
            if not isinstance(row, dict):
                audit["malformed"] += 1
            elif "text" not in row:
                audit["missing"] += 1
            elif not isinstance(row["text"], str):
                audit["nonstring"] += 1
            elif not row["text"].strip():
                audit["empty"] += 1
            else:
                text = row["text"].strip()
                # Reject, never silently truncate, an oversized legal record.
                if len(text.encode("utf-8")) > cfg["max_document_bytes"]:
                    raise ValueError(f"{source} record {i} exceeds max_document_bytes; inspect and explicitly raise the limit")
                audit["selected_text_bytes"] += len(text.encode("utf-8"))
                # Canonically equivalent Unicode documents share a split for
                # every variant, including NFC-based AYN. Retain original text.
                key = hashlib.sha256(unicodedata.normalize("NFKC", text).encode()).hexdigest()
                split_hash = hashlib.sha256(f"{cfg['seed']}:{key}".encode()).hexdigest()
                split = "validation" if int(split_hash[:16], 16) / 2**64 < cfg["validation_fraction"] else "train"
                inserted = db.execute("INSERT OR IGNORE INTO docs VALUES (?,?,?,?)", (key, split, source, text)).rowcount
                db.execute("INSERT INTO provenance VALUES (?,?,?)", (source, i, key))
                audit["valid"] += 1
                audit["unique"] += inserted
                audit["duplicate"] += 1 - inserted
            db.execute("INSERT OR REPLACE INTO progress VALUES (?,?,?)", (source, i, json.dumps(audit)))
            if (i + 1) % 1000 == 0:
                db.commit()
                print(f"prepare {source}: {i+1:,} records", flush=True)
        audit["complete_source_scan"] = complete_source
        db.execute("UPDATE progress SET audit=? WHERE source=?", (json.dumps(audit), source))
        db.commit()
        if not audit["records"] or (audit["records"] - audit["valid"]) / audit["records"] > cfg["max_invalid_fraction"]:
            raise ValueError(f"Unexpected removal rate in {source}: {audit}")
    db.execute("CREATE INDEX IF NOT EXISTS split_order ON docs(split,digest)")
    db.commit()
    return db


def texts(db, split):
    # Sorting hashes provides a deterministic, disk-backed document order.
    yield from db.execute("SELECT digest,source,text FROM docs WHERE split=? ORDER BY digest", (split,))


def train_tokenizer(db, directory, cfg):
    if cfg["tokenizer"] not in KINDS:
        raise ValueError(f"Unknown tokenizer: {cfg['tokenizer']}")
    from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    byte = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    if cfg["tokenizer"] == "legal_bpe":
        # Protect matched spans from the following generic word splitting.
        # Remaining text uses GPT-2-style boundaries. No normalization/replacement.
        legal = r"(?:Section|Sections|Article|Articles)\s+\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))*|\(\d{4}\)\s+\d+\s+SCC\s+\d+"
        ordinary = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(legal + "|" + ordinary), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    elif cfg["tokenizer"] == "digit_bpe":
        # AYN-inspired ablation: isolate each digit before ordinary byte BPE.
        # ByteLevel's alphabet still permits exact recovery of unseen UTF-8.
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r"\p{N}"), behavior="isolated"),
            byte,
        ])
    else:
        tok.pre_tokenizer = byte
    tok.decoder = decoders.ByteLevel()
    if cfg["tokenizer"].startswith("kl3m_"):
        from tokenizers import normalizers
        tok.normalizer = normalizers.NFKC()
    sample_ids, sample_bytes, sample_documents = directory / "tokenizer_sample.jsonl", 0, 0
    source_bytes = {}
    # Read SQLite on this thread, then give tokenizers an in-memory bounded
    # sample. Its Rust iterator may request Python items from a different
    # worker thread, so a SQLite-backed generator is not thread-safe here.
    sample_texts = []
    def sample_rows():
        if cfg.get("tokenizer_sample_strategy", "natural") == "natural":
            yield from texts(db, "train")
            return
        sources = [s for (s,) in db.execute("SELECT DISTINCT source FROM docs WHERE split='train' ORDER BY source")]
        # Equal soft byte quota per represented source, exceeded by at most
        # one complete document per source. No corpus-sized Python index.
        quota = max(1, cfg["tokenizer_sample_bytes"] // max(1, len(sources)))
        for source in sources:
            used = 0
            for row in db.execute("SELECT digest,source,text FROM docs WHERE split='train' AND source=? ORDER BY digest", (source,)):
                if used >= quota:
                    break
                used += len(row[2].encode())
                yield row
    with sample_ids.open("w", encoding="utf-8") as f:
        for key, source, text in sample_rows():
            if cfg.get("tokenizer_sample_strategy", "natural") == "natural" and sample_bytes >= cfg["tokenizer_sample_bytes"]:
                break
            # Whole documents: upper bound is sample budget + one document.
            sample_bytes += len(text.encode())
            source_bytes[source] = source_bytes.get(source, 0) + len(text.encode())
            sample_documents += 1
            sample_texts.append(text)
            f.write(json.dumps({"digest": key, "source": source}) + "\n")
    if not sample_documents:
        raise ValueError("No training documents; provide a larger corpus subset")
    if cfg["tokenizer"] == "ayn_bpe":
        tok = train_sentencepiece(sample_texts, cfg["vocab_size"])
    else:
        trainer_options = {}
        if cfg["tokenizer"] in ("kl3m_char3", "kl3m_char4"):
            trainer_options["max_token_length"] = int(cfg["tokenizer"][-1])
        tok.train_from_iterator(iter(sample_texts), trainer=trainers.BpeTrainer(
            vocab_size=cfg["vocab_size"], min_frequency=2, special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False, **trainer_options))
    for text in ["Section 482(1), Article 21 — न्यायालय\nA v. B (2024) 2 SCC 19.",
                 "Unknown987654321\t<|endoftext|>  \n"]:
        if tok.decode(tok.encode(text).ids, skip_special_tokens=False) != normalized_text(tok, text):
            raise ValueError("Tokenizer does not round-trip exact text")
    tok.save(str(directory / "tokenizer.json"))
    reloaded = load_tokenizer(directory / "tokenizer.json")
    if reloaded.encode(text).ids != tok.encode(text).ids:
        raise ValueError("Tokenizer reload changed token IDs")
    return tok, {"sample_documents": sample_documents, "sample_bytes": sample_bytes,
                 "sample_source_bytes": source_bytes,
                 "sample_strategy": cfg.get("tokenizer_sample_strategy", "natural"),
                 "normalization": "NFC; whitespace preserved" if cfg["tokenizer"] == "ayn_bpe" else "NFKC" if cfg["tokenizer"].startswith("kl3m_") else "none",
                 "offset_reference": "NFC text" if cfg["tokenizer"] == "ayn_bpe" else "original text",
                 "actual_vocab_size": tok.get_vocab_size(), "requested_vocab_size": cfg["vocab_size"],
                 "sha256": digest_file(directory / "tokenizer.json"), "variant": cfg["tokenizer"]}


def tokenizer_diagnostics(db, tok, max_documents=20, max_bytes=2_000_000):
    """Same raw held-out documents across tokenizer variants; no LM scoring."""
    pattern = re.compile(r"(?:Section|Sections|Article|Articles)\s+\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))*|\(\d{4}\)\s+\d+\s+SCC\s+\d+")
    keys, byte_count, words, count, forms, pieces = [], 0, 0, 0, 0, 0
    raw_exact = True
    for key, _, text in texts(db, "validation"):
        if (max_documents and len(keys) >= max_documents) or (max_bytes and byte_count >= max_bytes):
            break
        encoded = tok.encode(text)
        decoded = tok.decode(encoded.ids, skip_special_tokens=False)
        reference = normalized_text(tok, text)
        raw_exact &= decoded == text
        if decoded != reference:
            raise ValueError("Tokenizer failed a held-out text round trip")
        starts, ends = [a for a, _ in encoded.offsets], [b for _, b in encoded.offsets]
        for match in pattern.finditer(reference if isinstance(tok, SentencePieceBPE) else text):
            pieces += bisect_left(starts, match.end()) - bisect_right(ends, match.start())
            forms += 1
        keys.append(key)
        byte_count += len(text.encode())
        words += len(text.split())
        count += len(encoded.ids)
    return {"held_out_document_ids": keys, "sample_sha256": fingerprint(keys), "raw_bytes": byte_count,
            "words": words, "tokens": count, "tokens_per_byte": count / byte_count if byte_count else None,
            "tokens_per_word": count / words if words else None, "recognized_legal_forms": forms,
            "mean_tokens_per_legal_form": pieces / forms if forms else None, "all_round_trips_exact": raw_exact,
            "all_normalized_round_trips_exact": True,
            "scope": f"first {max_documents or 'all'} hash-ordered held-out documents; "
                     f"{max_bytes or 'no'} soft raw-byte cap"}


def cache_tokens(db, tok, directory, cfg):
    result = {}
    eos = tok.token_to_id("<|endoftext|>")
    for split, limit in [("train", cfg["train_tokens"]), ("validation", cfg["val_tokens"])]:
        count, used, raw_bytes = 0, 0, 0
        per_source = {}
        with (directory / f"{split}.bin").open("wb") as f, (directory / f"{split}_offsets.jsonl").open("w") as offsets:
            for key, source, text in texts(db, split):
                if count >= limit:
                    break
                encoded = tok.encode(text)
                offset_text = normalized_text(tok, text) if cfg["tokenizer"] == "ayn_bpe" else text
                ids = encoded.ids + [eos]
                taken = min(len(ids), limit - count)
                # The terminal capped document is partial; record its exact offset.
                end_char = len(offset_text) if taken >= len(encoded.ids) else encoded.offsets[taken-1][1]
                offsets.write(json.dumps({"digest": key, "source": source, "start": count, "tokens": taken,
                                          "complete": taken == len(ids), "end_char": end_char,
                                          "offset_reference": "NFC text" if cfg["tokenizer"] == "ayn_bpe" else "original text"}) + "\n")
                np.asarray(ids[:taken], dtype="<u4").tofile(f)
                count += taken
                used += 1
                raw_bytes += len(offset_text[:end_char].encode())
                per_source[source] = per_source.get(source, 0) + taken
        if count < cfg["context"]:
            raise ValueError(f"{split} has {count} tokens, fewer than one context of {cfg['context']}; use more data")
        result[split] = {"tokens": count, "documents": used, "raw_bytes": raw_bytes,
                         "byte_reference": "NFC text" if cfg["tokenizer"] == "ayn_bpe" else "original text",
                         "source_tokens": per_source, "sha256": digest_file(directory / f"{split}.bin")}
    # Frozen development prompts/references only from held-out documents.
    prompts, truth = {}, {}
    for key, source, text in texts(db, "validation"):
        words = text.split()
        if len(words) >= 640 and len(prompts) < cfg["validation_pairs"]:
            prompts[key] = {"text": " ".join(words[:512]), "source": source}
            truth[key] = {"text": " ".join(words[512:640])}
    write_json(directory / "held_out_prompts.json", prompts)
    write_json(directory / "held_out_truth.json", truth)
    result["held_out_pairs"] = len(prompts)
    result["held_out_prompts_sha256"] = digest_file(directory / "held_out_prompts.json")
    result["held_out_truth_sha256"] = digest_file(directory / "held_out_truth.json")
    return result


PREPARATION_KEYS = ("sources", "seed", "validation_fraction", "max_document_bytes", "max_invalid_fraction",
                    "tokenizer", "vocab_size", "tokenizer_sample_bytes", "train_tokens", "val_tokens", "validation_pairs",
                    "max_documents_per_source", "max_source_text_bytes", "tokenizer_sample_strategy")


def validate_cache(directory, cfg):
    result = json.loads((directory / "manifest.json").read_text())
    if result["cache_id"] != fingerprint(result["identity"]):
        raise ValueError("Cache manifest fingerprint mismatch")
    expected = {key: cfg[key] for key in PREPARATION_KEYS}
    if result["identity"]["settings"] != expected:
        different = [k for k in expected if result["identity"]["settings"].get(k) != expected[k]]
        raise ValueError(f"Prepared cache settings differ: {different}. Reuse its exact preparation settings.")
    for name in ("train", "validation"):
        if digest_file(directory / f"{name}.bin") != result["tokens"][name]["sha256"]:
            raise ValueError(f"Corrupted {name} cache; restore or rebuild {directory}")
    if digest_file(directory / "tokenizer.json") != result["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer cache checksum mismatch")
    for name in ("held_out_prompts", "held_out_truth"):
        if digest_file(directory / f"{name}.json") != result["tokens"][f"{name}_sha256"]:
            raise ValueError(f"{name} checksum mismatch; frozen evaluation data changed")
    return result


def prepare(args, cfg):
    if getattr(args, "prepared_dir", None):
        directory = Path(args.prepared_dir).expanduser().resolve()
        return directory, validate_cache(directory, cfg)
    paths = resolve_sources(args, cfg["sources"])
    source_info = {s: {"path": str(p), "sha256": digest_file(p), "bytes": p.stat().st_size} for s, p in paths.items()}
    identity = {"settings": {k: cfg[k] for k in PREPARATION_KEYS}, "sources": source_info,
                "implementation": digest_file(__file__),
                "tokenizer_implementation": digest_file(Path(__file__).with_name("tokenizer_io.py")),
                "unicode_version": unicodedata.unidata_version,
                "packages": {name: importlib.metadata.version(name) for name in ("tokenizers", "ijson", "numpy")}}
    if cfg["tokenizer"] == "ayn_bpe":
        identity["packages"]["sentencepiece"] = importlib.metadata.version("sentencepiece")
    cache_id = fingerprint(identity)
    directory = Path(args.cache_dir).expanduser().resolve() / cache_id
    directory.mkdir(parents=True, exist_ok=True)
    # One process per cache: OS lock is released if an interrupted process dies.
    import fcntl
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = directory / "manifest.json"
        if marker.exists():
            return directory, validate_cache(directory, cfg)
        write_json(directory / "preparation_config.json", identity)
        db = ingest(paths, directory, cfg)
        try:
            tok, tokenizer_stats = train_tokenizer(db, directory, cfg)
            tokenizer_stats["diagnostics"] = tokenizer_diagnostics(db, tok)
            token_stats = cache_tokens(db, tok, directory, cfg)
            audit = {s: json.loads(a) for s, a in db.execute("SELECT source,audit FROM progress")}
            split_counts = dict(db.execute("SELECT split,COUNT(*) FROM docs GROUP BY split"))
            result = {"cache_id": cache_id, "identity": identity, "audit": audit, "split_documents": split_counts,
                      "tokenizer": tokenizer_stats, "tokens": token_stats,
                      "policy": "NFKC-equivalent text grouped for dedup retaining original text; hash split; EOS packed across documents; no zero padding"}
            write_json(marker, result)
        finally:
            db.close()
    return directory, result
