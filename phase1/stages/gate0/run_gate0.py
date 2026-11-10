"""Fail-fast gates; record CPU/tiny versus real-size GPU evidence explicitly."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from src.environment import require_environment

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--hardware-profile", choices=["tiny", "production"], default="production")
    p.add_argument("--precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--world-size", type=int, default=1)
    a = p.parse_args()
    if a.world_size not in (1, 2, 3, 4, 5, 6, 8):
        raise ValueError("Unsupported GPU process count")
    require_environment(Path(a.output) / "environment.json")
    from src.common import ROOT, code_hash, digest_file, write_json
    output = Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = {"passed": False, "code": code_hash(), "prepared_sha256": digest_file(Path(a.prepared) / "manifest.json"),
             "world_size": a.world_size, "precision": a.precision, "context": a.context,
             "hardware_profile": a.hardware_profile, "checks": {}}
    write_json(output / "gate0.json", state)
    checks = ["unit_tests", "sources", "tokenizer", "split", "packing", "params", "train_1gpu"]
    if a.world_size > 1:
        checks.append("train_6gpu")
    else:
        state["checks"]["train_6gpu"] = {"status": "not_requested", "reason": "single-device execution only"}
    checks.append("generate")
    for check in checks:
        command = [sys.executable, "-m", f"stages.gate0.checks.check_{check}",
                   "--prepared", str(Path(a.prepared).resolve()), "--output", str(output),
                   "--context", str(a.context), "--hardware-profile", a.hardware_profile,
                   "--precision", a.precision, "--world-size", str(a.world_size)]
        start = time.monotonic()
        print(f"Gate 0: {check}; log {output / (check + '.log')}", flush=True)
        with (output / f"{check}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state["checks"][check] = {"status": "passed" if result.returncode == 0 else "failed",
                                  "seconds": time.monotonic() - start, "returncode": result.returncode}
        write_json(output / "gate0.json", state)
        if result.returncode:
            print((output / f"{check}.log").read_text()[-6000:], file=sys.stderr)
            raise SystemExit(result.returncode)
    state["passed"] = True
    write_json(output / "gate0.json", state)
    print(f"GATE 0 PASSED ({a.hardware_profile}, {a.world_size} process(es))")

if __name__ == "__main__":
    main()
