import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
import sqlite3
import numpy as np
import torch
from src.common import *
from src.prepare import verify
from src.model_factory import configuration, check_parameters

ARCHES = ["A_llama", "B_pythia", "C_falcon", "D_moe"]

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--hardware-profile", choices=["tiny", "production"], default="production")
    p.add_argument("--precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--world-size", type=int, default=1)
    return p.parse_args()

def run_check(name):
    a = args()
    prepared, output = Path(a.prepared), Path(a.output)
    manifest = read(prepared / "manifest.json")
    tok = load_tokenizer(prepared / "tokenizer.json")
    if name == "unit_tests":
        import os
        env = os.environ.copy()
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], check=True, env=env)
    elif name == "sources":
        verify(prepared)
        if set(manifest["audit"]) != {"acts", "high_courts", "supreme_court", "tribunals"}:
            raise ValueError("All four sources are required")
        for source, audit in manifest["audit"].items():
            if audit["valid"] < 1 or manifest["tokenizer"]["sample_source_bytes"].get(source, 0) < 1:
                raise ValueError(f"Source missing from data or tokenizer sample: {source}")
            print(source, audit, flush=True)
    elif name == "tokenizer":
        for text in ["  Section 21A — न्यायालय\n", "é e\u0301 1234567890\t", "AIR 2020 SC 1234", "<|endoftext|> ▁ 🙂", "Ａrticle ①"]:
            ids = tok.encode(text).ids
            if not ids or max(ids) >= tok.get_vocab_size() or tok.decode(ids, skip_special_tokens=False) != normalized_text(tok, text):
                raise ValueError(f"Tokenizer round trip/range failed for {text!r}")
        if tok.token_to_id("<|endoftext|>") is None:
            raise ValueError("Missing EOS")
        print("Actual vocabulary", tok.get_vocab_size(), manifest["tokenizer"]["normalization"])
    elif name == "split":
        db = sqlite3.connect(prepared / "documents.sqlite")
        seed = manifest["identity"]["config"]["seed"]
        frac = manifest["identity"]["config"]["validation_fraction"]
        for digest, split in db.execute("SELECT digest,split FROM docs"):
            value = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:16], 16) / 2**64
            assert split == ("validation" if value < frac else "train")
        for line in (prepared / "tokenizer_sample.jsonl").open():
            digest = json.loads(line)["digest"]
            assert db.execute("SELECT split FROM docs WHERE digest=?", (digest,)).fetchone()[0] == "train"
        for digest in read(prepared / "held_out_prompts.json"):
            assert db.execute("SELECT split FROM docs WHERE digest=?", (digest,)).fetchone()[0] == "validation"
        assert not db.execute("SELECT count(*) FROM provenance p LEFT JOIN docs d USING(digest) WHERE d.digest IS NULL").fetchone()[0]
        db.close()
    elif name == "packing":
        db = sqlite3.connect(prepared / "documents.sqlite")
        for split, sources in manifest["streams"].items():
            for source, info in sources.items():
                data = np.memmap(prepared / info["file"], dtype="<u4", mode="r")
                if len(data) < a.context:
                    raise ValueError(f"No complete block: {split}/{source}")
                expected = []
                for (text,) in db.execute("SELECT text FROM docs WHERE split=? AND source=? ORDER BY digest LIMIT 2", (split, source)):
                    expected += tok.encode(text).ids + [tok.token_to_id("<|endoftext|>")]
                assert np.array_equal(data[:len(expected)], expected)
                # Scan min/max in bounded chunks across the complete stream.
                for start in range(0, len(data), 1_000_000):
                    assert int(data[start:start+1_000_000].max()) < tok.get_vocab_size()
                print(split, source, "blocks", len(data)//a.context, "dropped tail", len(data)%a.context)
        db.close()
    elif name == "params":
        rows = {arch: check_parameters(configuration(arch, tok.get_vocab_size(), tok.token_to_id("<|endoftext|>"), a.context)) for arch in ARCHES}
        write_json(output / "parameters.json", rows)
        print(rows)
    elif name in ("train_1gpu", "train_6gpu"):
        world = 1 if name == "train_1gpu" else a.world_size
        if name == "train_6gpu" and world == 1:
            raise ValueError("Multi-GPU check requires the requested world size >1")
        for arch in ARCHES:
            cmd = [sys.executable, str(ROOT / "train.py"), "--arch", arch, "--prepared", str(prepared),
                "--output", str(output / name / arch), "--max-steps", "3", "--context", str(a.context),
                "--batch", "1", "--grad-accum", "1", "--eval-every", "3", "--eval-blocks-per-source", "1",
                "--precision", a.precision, "--warmup", "1"]
            cmd += ["--smoke"] if a.hardware_profile == "tiny" else ["--gate-pilot"]
            if world > 1:
                cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(world), *cmd[1:], "--fsdp"]
            subprocess.run([*cmd, "--stop-after", "1"], check=True)
            subprocess.run(cmd, check=True)
            metrics = read(output / name / arch / "metrics.json")
            assert metrics["complete"] and metrics["step"] == 3
    elif name == "generate":
        from src.generate import read_prompts
        prompts = read_prompts(prepared / "held_out_prompts.json")
        # One frozen real held-out prompt per source; never fabricated/padded.
        selected, seen = {}, set()
        for key, row in prompts.items():
            if row["source"] not in seen:
                selected[key] = row
                seen.add(row["source"])
        write_json(output / "prompts.json", selected)
        for arch in ARCHES:
            subprocess.run([sys.executable, "-m", "src.generate", "--model", str(output / "train_1gpu" / arch / "best"),
                "--prompts", str(output / "prompts.json"), "--output", str(output / "generation" / arch),
                "--device", "cuda" if torch.cuda.is_available() else "cpu", "--temperature", "1.0", "--max-tokens", "1536"], check=True)
    else:
        raise ValueError(name)
    print(f"PASS: {name}", flush=True)
