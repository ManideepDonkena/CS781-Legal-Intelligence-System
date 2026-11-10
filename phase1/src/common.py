from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from data import digest_file, write_json, fingerprint
from tokenizer_io import load_tokenizer, normalized_text, KINDS


def read(path):
    return json.loads(Path(path).read_text())


def code_hash():
    return fingerprint({str(p.relative_to(ROOT)): digest_file(p)
                        for folder in ("src", "core", "stages", "configs")
                        for p in sorted((ROOT / folder).rglob("*"))
                        if p.is_file() and p.suffix in (".py", ".json", ".yaml")}
                       | {p.name: digest_file(p) for p in ROOT.glob("*.py")})
