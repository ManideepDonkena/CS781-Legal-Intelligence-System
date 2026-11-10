"""Reuse verified local official data before downloading; keep raw files in each run."""
import argparse
import errno
import fcntl
import os
from pathlib import Path
import shutil
from .common import *
from data import REPO, SOURCES


def matches_revision(saved, revision):
    # main means the locally pinned snapshot once downloaded, as before.
    if not isinstance(saved, dict):
        raise ValueError("Invalid download manifest")
    return revision == "main" or revision in (saved.get("revision"), saved.get("requested_revision"))


def verify_download(directory, revision="main"):
    directory = Path(directory).resolve()
    saved = read(directory / "download_manifest.json")
    if saved.get("repo") != REPO or set(saved.get("files", {})) != set(SOURCES):
        raise ValueError("Download manifest must identify the official repo and all four sources")
    if not isinstance(saved.get("revision"), str) or not saved["revision"]:
        raise ValueError("Download manifest has no pinned revision")
    if not matches_revision(saved, revision):
        raise ValueError("Different dataset revision requested; use a fresh experiment folder")
    for source, filename in SOURCES.items():
        row = saved["files"][source]
        if row.get("file") != filename:
            raise ValueError(f"Unexpected source filename in manifest: {source}")
        path = directory / filename
        if (not path.is_file() or path.stat().st_size != row.get("bytes")
                or digest_file(path) != row.get("sha256")):
            raise ValueError(f"Missing/corrupt downloaded file: {path}")
    return saved


def find_local_download(directory, revision="main"):
    """Search shallow, known locations, never arbitrary directories or sample files."""
    directory = Path(directory).resolve()
    candidates = {ROOT / "data/raw"}
    parents = {ROOT / "experiments"}
    # Standard layout: PARENT/EXPERIMENT/data/raw. Reuse sibling experiments
    # even when their parent is outside the extracted project directory.
    if directory.name == "raw" and directory.parent.name == "data":
        parents.add(directory.parent.parent.parent)
    for parent in sorted(parents):
        candidates.update(marker.parent for marker in parent.glob("*/data/raw/download_manifest.json"))
    for candidate in sorted({path.resolve() for path in candidates}):
        if candidate == directory or not (candidate / "download_manifest.json").is_file():
            continue
        try:
            saved = read(candidate / "download_manifest.json")
            if not matches_revision(saved, revision):
                continue
            saved = verify_download(candidate, revision)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Skipping unusable local download {candidate}: {error}", flush=True)
            continue
        return candidate, saved
    return None


def reuse_download(source, directory, saved):
    methods = {}
    for name, row in saved["files"].items():
        original, target = source / row["file"], directory / row["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or digest_file(target) != row["sha256"]:
                raise ValueError(f"Conflicting existing raw file: {target}; use a fresh work directory")
            methods[name] = "already_present"
            continue
        try:
            os.link(original, target)
            methods[name] = "hardlink"
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK):
                raise
            shutil.copyfile(original, target)
            if digest_file(target) != row["sha256"]:
                raise ValueError(f"Copy verification failed: {target}")
            methods[name] = "copy"
    saved = dict(saved, reused_from=str(source), reuse_methods=methods)
    write_json(directory / "download_manifest.json", saved)
    print(f"Reused verified official data from {source}; revision={saved['revision']}; methods={methods}. No network download.", flush=True)


def download(directory, revision="main"):
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "download_manifest.json").exists():
            saved = verify_download(directory, revision)
            print(f"Reusing verified official source files in {directory}; revision={saved['revision']}. No network download.", flush=True)
            return
        found = find_local_download(directory, revision)
        if found:
            reuse_download(found[0], directory, found[1])
            return
        print("No compatible verified local download found; fetching official files from Hugging Face.", flush=True)
        _download_remote(directory, revision)


def _download_remote(directory, revision):
    from huggingface_hub import HfApi, hf_hub_download
    commit = HfApi().dataset_info(REPO, revision=revision).sha
    files = {}
    for source, filename in SOURCES.items():
        path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset", filename=filename,
                                  revision=commit, local_dir=directory))
        files[source] = {"file": str(path.relative_to(directory)), "sha256": digest_file(path), "bytes": path.stat().st_size}
        print(f"Downloaded and hashed {source}: {path.stat().st_size:,} bytes", flush=True)
    write_json(directory / "download_manifest.json", {"repo": REPO, "requested_revision": revision, "revision": commit, "files": files})

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--revision", default="main")
    a = p.parse_args()
    download(a.output, a.revision)
