#!/usr/bin/env python3
"""Build a token cache (train.bin / validation.bin / tokenizer.json / manifest.json).

Compatible with `experiments/runner.py --prepared-dir`, but unlike `data.prepare` it can
  * mix the four official sources with explicit token shares (and repeat a source for >1 epoch),
  * train the tokenizer on a sample drawn with the same mixture,
  * store ids as uint16 (vocab <= 65535) to halve disk,
  * emit a *target-domain* validation set (Supreme Court) plus per-source validation files, so
    every study is scored on the same raw text and bits-per-byte is comparable across tokenizers.

Requires `studies/build_corpus.py` to have ingested the sources first.

Example:
  python studies/build_cache.py --name bpe50k_mixA_600M --tokenizer bpe --vocab 50000 \
      --train-tokens 600000000 --mix supreme_court=0.3,high_courts=0.3,tribunals=0.3,acts=0.1
"""
import argparse, json, math, os, random, sqlite3, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from data import train_tokenizer, write_json, digest_file, fingerprint, PREPARATION_KEYS  # noqa: E402

SOURCES = ["supreme_court", "high_courts", "tribunals", "acts"]
CORPUS = ROOT / "artifacts" / "corpus"
CACHES = ROOT / "artifacts" / "caches"
EOS = "<|endoftext|>"


def open_db(source):
    db = sqlite3.connect(f"file:{CORPUS / source / 'documents.sqlite'}?mode=ro", uri=True)
    db.execute("PRAGMA cache_size=-65536")
    return db


def doc_stream(source, split, cycle, clean_fn=None):
    """Texts of one source in deterministic digest order; optionally repeated forever."""
    epoch = 0
    while True:
        db = open_db(source)
        cur = db.execute("SELECT digest,text FROM docs WHERE split=? ORDER BY digest", (split,))
        n = 0
        for key, text in cur:
            if clean_fn is not None and split == "train":
                text = clean_fn(key, text)
                if text is None:
                    continue
            n += 1
            yield key, text, epoch
        db.close()
        if not cycle or n == 0:
            return
        epoch += 1


def encode_to_file(tok, eos, texts, fh, dtype, limit):
    """Encode a batch of texts; write at most `limit` tokens; return (tokens_written, bytes_consumed)."""
    written, raw = 0, 0
    for text, enc in zip(texts, tok.encode_batch(texts)):
        ids = enc.ids + [eos]
        take = min(len(ids), limit - written)
        if take < len(ids):  # truncated final doc: count only the covered raw text
            end_char = enc.offsets[take - 1][1] if take > 0 else 0
            raw += len(text[:end_char].encode())
        else:
            raw += len(text.encode())
        np.asarray(ids[:take], dtype=dtype).tofile(fh)
        written += take
        if written >= limit:
            break
    return written, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", choices=["bpe", "legal_bpe"], default="bpe")
    ap.add_argument("--vocab", type=int, default=50000)
    ap.add_argument("--train-tokens", type=int, required=True)
    ap.add_argument("--val-tokens", type=int, default=4_000_000)
    ap.add_argument("--val-per-source-tokens", type=int, default=1_000_000)
    ap.add_argument("--mix", required=True, help="source=share,... (shares are normalised)")
    ap.add_argument("--tokenizer-sample-bytes", type=int, default=60_000_000)
    ap.add_argument("--tokenizer-from", help="reuse tokenizer.json from an existing cache dir name (controlled comparisons)")
    ap.add_argument("--clean-data", action="store_true", help="apply heuristic filtering, boilerplate cleaning, PII scrubbing and MinHash LSH dedup")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-docs", type=int, default=256)
    a = ap.parse_args()

    mix = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
    assert set(mix) <= set(SOURCES), mix
    total_w = sum(mix.values())
    share = {k: v / total_w for k, v in mix.items()}
    out = CACHES / a.name
    if (out / "manifest.json").exists():
        print("already built:", out); return
    out.mkdir(parents=True, exist_ok=True)
    dtype = "<u2" if a.vocab <= 65535 else "<u4"
    t0 = time.time()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    # ---------------- data quality cleaner ----------------
    clean_fn = None
    if a.clean_data:
        from studies.data_quality import HeuristicFilter, MinHashLSH, clean_legal_boilerplate, redact_pii
        hfilter = HeuristicFilter(min_chars=500)
        lsh = MinHashLSH(num_perm=64, num_bands=8, k_shingle=5, seed=a.seed)

        def _clean(key, text):
            cleaned = clean_legal_boilerplate(text)
            valid, _ = hfilter.evaluate(cleaned)
            if not valid:
                return None
            redacted = redact_pii(cleaned)
            is_dup, _, _ = lsh.is_duplicate(key, redacted)
            if is_dup:
                return None
            return redacted

        clean_fn = _clean
        log("Data quality pipeline enabled: Heuristic filter (min_chars=500), boilerplate cleaner, PII redaction, MinHash-LSH dedup")

    # ---------------- tokenizer ----------------
    from tokenizers import Tokenizer
    if a.tokenizer_from:
        src = CACHES / a.tokenizer_from / "tokenizer.json"
        (out / "tokenizer.json").write_bytes(src.read_bytes())
        tok = Tokenizer.from_file(str(out / "tokenizer.json"))
        tok_stats = {"sample_documents": 0, "sample_bytes": 0, "actual_vocab_size": tok.get_vocab_size(),
                     "requested_vocab_size": a.vocab, "sha256": digest_file(out / "tokenizer.json"),
                     "variant": a.tokenizer, "reused_from": a.tokenizer_from}
    else:
        sample_path = out / "tokenizer_sample.sqlite"
        sample_path.unlink(missing_ok=True)
        sample_db = sqlite3.connect(sample_path)
        sample_db.execute("CREATE TABLE IF NOT EXISTS docs (digest TEXT PRIMARY KEY, split TEXT, source TEXT, text TEXT)")
        for s, w in share.items():
            quota, got = a.tokenizer_sample_bytes * w, 0
            for key, text, _ in doc_stream(s, "train", cycle=False, clean_fn=clean_fn):
                if got >= quota: break
                sample_db.execute("INSERT OR IGNORE INTO docs VALUES (?,?,?,?)", (key, "train", s, text))
                got += len(text.encode())
        sample_db.commit()
        cfg = dict(tokenizer=a.tokenizer, vocab_size=a.vocab, tokenizer_sample_bytes=a.tokenizer_sample_bytes)
        tok, tok_stats = train_tokenizer(sample_db, out, cfg)
        sample_db.close()
        sample_path.unlink()

    eos = tok.token_to_id(EOS)
    log(f"tokenizer ready: vocab={tok.get_vocab_size()} eos={eos}")

    # ---------------- train tokens (weighted document interleave) ----------------
    rng = random.Random(a.seed)
    quota = {s: int(a.train_tokens * w) for s, w in share.items()}
    quota[max(share, key=share.get)] += a.train_tokens - sum(quota.values())
    streams = {s: doc_stream(s, "train", cycle=True, clean_fn=clean_fn) for s in share}
    written = {s: 0 for s in share}
    raw_bytes = 0
    epochs_seen = {s: 0 for s in share}
    done = 0
    with (out / "train.bin").open("wb") as fh:
        while done < a.train_tokens:
            open_srcs = [s for s in share if written[s] < quota[s]]
            # pick source proportional to remaining quota -> shares are met exactly and interleaved
            weights = [quota[s] - written[s] for s in open_srcs]
            s = rng.choices(open_srcs, weights)[0]
            texts = []
            for _ in range(a.batch_docs):
                key, text, ep = next(streams[s])
                epochs_seen[s] = max(epochs_seen[s], ep)
                texts.append(text)
            n, rb = encode_to_file(tok, eos, texts, fh, dtype, quota[s] - written[s])
            written[s] += n; done += n; raw_bytes += rb
            if (done // a.batch_docs) % 50 == 0:
                log(f"train {done/1e6:,.0f}M / {a.train_tokens/1e6:,.0f}M tokens  {done/(time.time()-t0)/1e6:.2f}M tok/s")
    log(f"train done: {done:,} tokens, epochs per source {epochs_seen}")

    # ---------------- validation (target domain = supreme_court) + per-source ----------------
    def build_val(sources, limit, path):
        got, rb, per = 0, 0, {}
        with path.open("wb") as fh:
            for s in sources:
                n_s = 0
                for key, text, _ in doc_stream(s, "validation", cycle=False):
                    if n_s >= limit: break
                    n, r = encode_to_file(tok, eos, [text], fh, dtype, limit - n_s)
                    n_s += n; rb += r
                per[s] = n_s; got += n_s
        return {"tokens": got, "raw_bytes": rb, "source_tokens": per, "sha256": digest_file(path)}

    result_tokens = {"train": {"tokens": done, "raw_bytes": raw_bytes, "source_tokens": written,
                               "documents": None, "sha256": digest_file(out / "train.bin")},
                     "dtype": dtype, "mixture_epochs": epochs_seen}
    result_tokens["validation"] = build_val(["supreme_court"], a.val_tokens, out / "validation.bin")
    result_tokens["validation_by_source"] = {}
    for s in SOURCES:
        result_tokens["validation_by_source"][s] = build_val([s], a.val_per_source_tokens, out / f"validation_{s}.bin")
    log(f"validation: {result_tokens['validation']}")

    # frozen held-out prompts/references from Supreme Court validation documents
    prompts, truth = {}, {}
    for key, text, _ in doc_stream("supreme_court", "validation", cycle=False):
        words = text.split()
        if len(words) >= 640 and len(prompts) < 20:
            prompts[key] = {"text": " ".join(words[:512]), "source": "supreme_court"}
            truth[key] = {"text": " ".join(words[512:640])}
    write_json(out / "held_out_prompts.json", prompts); write_json(out / "held_out_truth.json", truth)
    result_tokens["held_out_pairs"] = len(prompts)
    result_tokens["held_out_prompts_sha256"] = digest_file(out / "held_out_prompts.json")
    result_tokens["held_out_truth_sha256"] = digest_file(out / "held_out_truth.json")

    settings = dict(sources=sorted(share), seed=a.seed, validation_fraction=0.05, max_document_bytes=16_000_000,
                    max_invalid_fraction=0.05, tokenizer=a.tokenizer, vocab_size=a.vocab,
                    tokenizer_sample_bytes=a.tokenizer_sample_bytes, train_tokens=a.train_tokens,
                    val_tokens=a.val_tokens, validation_pairs=20, max_documents_per_source=0, max_source_text_bytes=0)
    assert set(settings) == set(PREPARATION_KEYS)
    identity = {"settings": settings, "mixture": share, "name": a.name, "builder": digest_file(__file__)}
    manifest = {"cache_id": fingerprint(identity), "identity": identity, "tokenizer": tok_stats,
                "tokens": result_tokens, "mixture": share, "build_seconds": time.time() - t0,
                "policy": "exact-text dedup per source; content-hash split seed 42; EOS packed; validation = Supreme Court"}
    write_json(out / "manifest.json", manifest)
    log(f"DONE {out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
