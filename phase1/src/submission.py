"""Independent local submission validator; never uploads anything."""
import argparse
import json
import zipfile
from pathlib import Path
from .generate import read_prompts

def validate(prompts_path, archive_path):
    prompts = read_prompts(prompts_path)
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["prediction.json"]:
            raise ValueError("ZIP must contain exactly prediction.json at its root")
        values = json.loads(archive.read("prediction.json"), object_pairs_hook=unique)
    if not isinstance(values, dict) or set(values) != set(prompts):
        raise ValueError("Submission IDs differ from prompt IDs")
    for key, row in values.items():
        if not isinstance(row, dict) or set(row) != {"text"} or not isinstance(row["text"], str) or len(row["text"].split()) != 128:
            raise ValueError(f"Submission schema/128-word check failed: {key}")
    return {"examples": len(values), "exactly_128_words": True, "ids_match": True, "zip_layout_valid": True}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", required=True)
    p.add_argument("--zip", required=True)
    a = p.parse_args()
    print(json.dumps(validate(a.prompts, a.zip), indent=2))
