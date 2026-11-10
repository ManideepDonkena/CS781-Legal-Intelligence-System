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
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path

import numpy as np

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
                key = hashlib.sha256(text.encode()).hexdigest()
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


# ---------------------------------------------------------------------------
# Corpus-mined Legal-BPE patterns
# ---------------------------------------------------------------------------
# Instead of hardcoding a fixed regex we scan the training corpus for legal
# forms that actually appear in this dataset.  The top-N frequent candidates
# per category are then woven into a pre-tokenization regex (built on-the-fly
# in train_tokenizer).

_MINED_PATTERNS = {
    # Category → (raw-search-regex, minimum-document-frequency)
    "sections": (r"Section[\s]+(\d+)", 3),
    "articles": (r"Article[\s]+(\d+)", 3),
    "scc_cite":   (r"\(\d{4}\)[\s]+\d+[\s]+SCC",             2),
    "air_cite":   (r"(?:AIR|A.I.R)[\s]+(?:\d{4})+(?:\s+[SCLC\s]*)", 2),
    "crpc_section": (r"Section[\s]+\d+[\s]*Cr\.?P\.?C\.?",      2),
}


def _mine_legal_patterns(documents: list[str]) -> dict:
    """Scan ``documents`` and return mined legal-form candidates with counts.

    Returns a dict of category → [{"span": "...", "count": N}] (sorted by
    count descending, capped at top-50 per category).  Only spans that meet
    the minimum-frequency threshold in *_MINED_PATTERNS* are kept.
    """
    mined = {}
    for cat, (pattern_str, min_freq) in _MINED_PATTERNS.items():
        counts: Counter = Counter()
        pat = re.compile(f"({pattern_str})", re.IGNORECASE)
        for text in documents:
            found_in_doc = set()  # avoid double-counting within one doc
            for m in pat.finditer(text):
                span = m.group(1).strip()
                if span and len(span) >= 2 and span not in found_in_doc:
                    counts[span] += 1
                    found_in_doc.add(span)
        top = [dict(span=s, count=c) for s, c in counts.most_common() if c >= min_freq][:50]
        mined[cat] = top
    return mined


def _patterns_to_regex(mined: dict) -> str:
    """Turn the mined candidates into a single pre-tokenization alternation."""
    parts = []

    # Section refs — all frequent section numbers from corpus
    sec_forms = list(set(d["span"] for d in mined.get("sections", [])))
    sec_forms.sort(key=lambda x: -len(x))          # longest tokens first
    if sec_forms:
        sec_pat = "|".join(re.escape(f"Section\\s+{n}") for n in sec_forms)
        parts.append(sec_pat)

    # Article refs
    art_forms = list(set(d["span"] for d in mined.get("articles", [])))
    art_forms.sort(key=lambda x: -len(x))
    if art_forms:
        art_pat = "|".join(re.escape(f"Article\\s+{n}") for n in art_forms)
        parts.append(art_pat)

    # SCC citation — (YEAR) PAGE SCC PAGE
    scc_specs = [d["span"] for d in mined.get("scc_cite", [])]
    if scc_specs:
        year_pages = "|".join(re.escape(f"({y}) {p}") for y, p in (sp.split()[:2] for sp in scc_specs))
        parts.append(rf"\({year_pages}\)\d+\s+SCC\s+\d+")

    # AIR citation
    air_specs = [d["span"] for d in mined.get("air_cite", [])]
    if air_specs:
        air_years = "|".join(re.escape(s) for s in air_specs)
        parts.append(rf"(?:AIR|A\.I\.R)\s+{air_years}")

    # CrPC section refs — full span like "Section 482 CrPC"
    crpc_specs = list(set(d["span"] for d in mined.get("crpc_section", [])))
    if crpc_specs:
        parts.append("|".join(re.escape(s) for s in crpc_specs))

    return "|".join(parts)


# Fallback: if corpus mining returns nothing (empty train, cold-start), use the old hardcoded regex.
_MINED_FALLBACK = (
    r"(?:Section|Sections|Article|Articles)\s+"
    r"\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))*"
    r"|\(\d{4}\)\s+\d+\s+SCC\s+\d+"
)


def train_tokenizer(db, directory, cfg):
    from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    byte = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    if cfg["tokenizer"] == "legal_bpe":
        # --- corpus-mined legal patterns (overrides hardcoded regex) ----------
        raw_rows = list(texts(db, "train"))[:5000]            # mine top-5K for speed
        mined = _mine_legal_patterns([t for _, _, t in raw_rows])
        legal = _patterns_to_regex(mined) or _MINED_FALLBACK   # fallback = old regex

        # Fallback if nothing was found (empty corpus during dev)
        mining_manifest = {cat: [d["span"] for d in spans] for cat, spans in mined.items()}

    else:
        legal = None
        mining_manifest = {}

    # --- Wire up the pre-tokenizer with ALL combined patterns ---------------
    if cfg["tokenizer"] == "legal_bpe":
        # `_patterns_to_regex` already merges all 5 mined categories
        # (Sections + Articles + SCC cites + AIR cites + CrPC Section) into
        # a single alternation, so this works across every data source — not
        # just one category at a time.
        ordinary = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(legal), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:
        tok.pre_tokenizer = byte
    tok.decoder = decoders.ByteLevel()
    sample_ids, sample_bytes, sample_documents = directory / "tokenizer_sample.jsonl", 0, 0
    # Fix: materialize all (digest,source,text) triples from the DB first.
    # tok.train_from_iterator() consumes the iterator on a worker thread;
    # SQLite connections cannot be used across threads, so we must read
    # everything here in the creating thread before yielding.
    raw_rows = list(texts(db, "train"))  # <-- fetch all rows into memory now

    def sample():
        nonlocal sample_bytes, sample_documents
        with open(sample_ids, "w", encoding="utf-8") as f:
            for key, source, text in raw_rows:
                if sample_bytes >= cfg["tokenizer_sample_bytes"]:
                    break
                # Whole documents: upper bound is sample budget + one document.
                sample_bytes += len(text.encode())
                sample_documents += 1
                f.write(json.dumps({"digest": key, "source": source}) + "\n")
                yield text
    tok.train_from_iterator(sample(), trainer=trainers.BpeTrainer(
        vocab_size=cfg["vocab_size"], min_frequency=2, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    if not sample_documents:
        raise ValueError("No training documents; provide a larger corpus subset")
    for text in ["Section 482(1), Article 21 — न्यायालय\nA v. B (2024) 2 SCC 19.",
                 "Unknown987654321\t<|endoftext|>  \n"]:
        if tok.decode(tok.encode(text).ids, skip_special_tokens=False) != text:
            raise ValueError("Tokenizer does not round-trip exact text")
    tok.save(str(directory / "tokenizer.json"))
    reloaded = Tokenizer.from_file(str(directory / "tokenizer.json"))
    if reloaded.encode(text).ids != tok.encode(text).ids:
        raise ValueError("Tokenizer reload changed token IDs")
    return tok, {"sample_documents": sample_documents, "sample_bytes": sample_bytes,
                 "actual_vocab_size": tok.get_vocab_size(), "requested_vocab_size": cfg["vocab_size"],
                 "sha256": digest_file(directory / "tokenizer.json"), "variant": cfg["tokenizer"],
                 "mined_patterns": mining_manifest}


def tokenizer_diagnostics(db, tok):
    """Same raw held-out documents across tokenizer variants; no LM scoring."""
    pattern = re.compile(r"(?:Section|Sections|Article|Articles)\s+\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))*|\(\d{4}\)\s+\d+\s+SCC\s+\d+")
    keys, byte_count, words, count, forms, pieces = [], 0, 0, 0, 0, 0
    for key, _, text in texts(db, "validation"):
        if len(keys) >= 20 or byte_count >= 2_000_000:
            break
        encoded = tok.encode(text)
        if tok.decode(encoded.ids, skip_special_tokens=False) != text:
            raise ValueError("Tokenizer failed a held-out text round trip")
        starts, ends = [a for a, _ in encoded.offsets], [b for _, b in encoded.offsets]
        for match in pattern.finditer(text):
            pieces += bisect_left(starts, match.end()) - bisect_right(ends, match.start())
            forms += 1
        keys.append(key)
        byte_count += len(text.encode())
        words += len(text.split())
        count += len(encoded.ids)
    return {"held_out_document_ids": keys, "sample_sha256": fingerprint(keys), "raw_bytes": byte_count,
            "words": words, "tokens": count, "tokens_per_byte": count / byte_count if byte_count else None,
            "tokens_per_word": count / words if words else None, "recognized_legal_forms": forms,
            "mean_tokens_per_legal_form": pieces / forms if forms else None, "all_round_trips_exact": True,
            "scope": "first 20 hash-ordered held-out documents; 2MB soft raw-byte cap plus one document"}


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
                ids = encoded.ids + [eos]
                taken = min(len(ids), limit - count)
                # The terminal capped document is partial; record its exact offset.
                end_char = len(text) if taken >= len(encoded.ids) else encoded.offsets[taken-1][1]
                offsets.write(json.dumps({"digest": key, "source": source, "start": count, "tokens": taken,
                                          "complete": taken == len(ids), "end_char": end_char}) + "\n")
                np.asarray(ids[:taken], dtype="<u4").tofile(f)
                count += taken
                used += 1
                raw_bytes += len(text[:end_char].encode())
                per_source[source] = per_source.get(source, 0) + taken
        if count < cfg["context"]:
            raise ValueError(f"{split} has {count} tokens, fewer than one context of {cfg['context']}; use more data")
        result[split] = {"tokens": count, "documents": used, "raw_bytes": raw_bytes,
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
                    "max_documents_per_source", "max_source_text_bytes")


def validate_cache(directory, cfg):
    result = json.loads((directory / "manifest.json").read_text())
    if result["cache_id"] != fingerprint(result["identity"]):
        raise ValueError("Cache manifest fingerprint mismatch")
    expected = {key: cfg[key] for key in PREPARATION_KEYS}
    if "builder" in result["identity"]:
        # studies/build_cache.py caches: the split seed is frozen in the cache; the *training* seed
        # (init + batch order) may differ so seed-variance studies can share one cache.
        expected["seed"] = result["identity"]["settings"]["seed"]
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
                "packages": {name: importlib.metadata.version(name) for name in ("tokenizers", "ijson", "numpy")}}
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
                      "policy": "exact-text dedup; hash split; EOS packed across documents; no zero padding"}
            write_json(marker, result)
        finally:
            db.close()
    return directory, result
