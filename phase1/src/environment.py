"""Check dependencies using the active interpreter, without downloading data/models."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys

PACKAGES = {
    "torch": "torch", "transformers": "transformers", "accelerate": "accelerate",
    "tokenizers": "tokenizers", "sentencepiece": "sentencepiece",
    "protobuf": "google.protobuf", "numpy": "numpy", "ijson": "ijson",
    "huggingface_hub": "huggingface_hub", "safetensors": "safetensors",
    "matplotlib": "matplotlib", "reportlab": "reportlab", "pytest": "pytest",
    "packaging": "packaging",
}


def inspect_environment(packages=None):
    report = {"python": sys.version, "executable": sys.executable,
              "packages": {}, "errors": []}
    if sys.version_info < (3, 10):
        report["errors"].append("Python 3.10 or newer is required")
    for distribution, module in (PACKAGES if packages is None else packages).items():
        try:
            version = importlib.metadata.version(distribution)
            importlib.import_module(module)
            report["packages"][distribution] = version
        except Exception as exc:
            report["errors"].append(f"{distribution} ({module}): {type(exc).__name__}: {exc}")
    # Import the bindings actually used, including protobuf (a separate package).
    if "sentencepiece" in report["packages"] and "protobuf" in report["packages"]:
        try:
            importlib.import_module("sentencepiece.sentencepiece_pb2")
        except Exception as exc:
            report["errors"].append(f"SentencePiece protobuf bindings: {exc}")
    if "transformers" in report["packages"]:
        try:
            module = importlib.import_module("transformers")
            for name in ("LlamaForCausalLM", "GPTNeoXForCausalLM", "FalconForCausalLM", "MixtralForCausalLM"):
                getattr(module, name)
        except Exception as exc:
            report["errors"].append(f"Transformers architecture imports: {type(exc).__name__}: {exc}")
    if "matplotlib" in report["packages"]:
        try:
            plotting = importlib.import_module("src.plotting")
            probe = getattr(plotting, "probe", None)
            if not callable(probe):
                error = (f"Plotting source mismatch: {getattr(plotting, '__file__', 'src/plotting.py')} "
                         "does not provide callable probe(). Restore src/plotting.py from the same release "
                         "as src/environment.py, then run python -m src.plotting. "
                         "Reinstalling pip packages cannot add a missing project function.")
                report.setdefault("source_errors", []).append(error)
                report["errors"].append(error)
            else:
                report["plotting"] = probe()
        except Exception as exc:
            report["errors"].append(f"Plot rendering: {type(exc).__name__}: {exc}")
    report["passed"] = not report["errors"]
    return report


def require_environment(output=None, device=None, world_size=1, precision="no"):
    report = inspect_environment()
    if "torch" in report["packages"]:
        import torch
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["visible_gpus"] = torch.cuda.device_count()
        if device == "cuda":
            if not report["cuda_available"] or report["visible_gpus"] < world_size:
                report["errors"].append(f"Require {world_size} visible CUDA GPU(s); found {report['visible_gpus']}")
            elif precision == "bf16":
                for index in range(world_size):
                    with torch.cuda.device(index):
                        if not torch.cuda.is_bf16_supported():
                            report["errors"].append(f"Visible GPU {index} does not support BF16; select --precision fp16")
    report["passed"] = not report["errors"]
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        if report.get("source_errors"):
            raise SystemExit("Environment check failed BEFORE data preparation/training:\n  "
                             + "\n  ".join(report["errors"])
                             + f"\nActive Python: {sys.executable}"
                             + "\nRepair the project source files from one complete release."
                             + "\nRun python verify_installation.py to list missing/changed release files."
                             + "\nKeep experiments/ and saved checkpoint identity markers.")
        command = shlex.join([sys.executable, "setup_environment.py"])
        raise SystemExit("Environment check failed BEFORE data preparation/training:\n  "
                         + "\n  ".join(report["errors"])
                         + f"\nActive Python: {sys.executable}\nInstall the complete stack from the project root:\n  {command}"
                         + "\nFor only the reported missing pytest: python -m pip install 'pytest>=7,<10'"
                         + "\nCUDA errors require a compatible CUDA PyTorch build/assigned GPU IDs."
                         + "\nNo tests are skipped and no packages are auto-installed by run_all.py.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--precision", choices=["no", "bf16", "fp16"], default="no")
    args = parser.parse_args()
    report = require_environment(args.output, args.device, args.world_size, args.precision)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
