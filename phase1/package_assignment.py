"""Create Hello IITK code/ + report/ ZIP from reviewed material; separate from Codabench."""
import argparse
from pathlib import Path
import re
import zipfile
from src.common import *

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="chosen Phase-1 run directory")
    p.add_argument("--report-pdf", required=True, help="reviewed final report including completed course LLM questionnaire")
    p.add_argument("--firstname", required=True)
    p.add_argument("--lastname", required=True)
    p.add_argument("--rollno", required=True)
    p.add_argument("--github-url", required=True)
    p.add_argument("--llm-declaration", required=True, help="completed official declaration/template file")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--additional-phase", action="append", default=[], help="phase2=/path/to/code; repeat for completed phases")
    a = p.parse_args()
    for field in (a.firstname, a.lastname, a.rollno):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", field):
            raise ValueError("Identity fields may contain letters, digits, underscore and hyphen")
    run, pdf, declaration = Path(a.run), Path(a.report_pdf), Path(a.llm_declaration)
    if not pdf.is_file() or pdf.read_bytes()[:5] != b"%PDF-" or not declaration.is_file():
        raise ValueError("Provide the reviewed PDF and completed declaration")
    if not a.github_url.startswith("https://github.com/"):
        raise ValueError("Provide the project GitHub URL")
    if not read(run / "metrics.json").get("complete") or read(run / "run.json")["settings"]["smoke"]:
        raise ValueError("Assignment packaging requires a completed production run")
    snapshot = run / "code_snapshot.zip"
    if digest_file(snapshot) != read(run / "code_snapshot_manifest.json")["zip_sha256"]:
        raise ValueError("Code snapshot checksum mismatch")
    stem = f"CS781-CP-{a.firstname}-{a.lastname}-{a.rollno}"
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{stem}-report_code.zip"
    with zipfile.ZipFile(destination.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
        with zipfile.ZipFile(snapshot) as code:
            for name in code.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Unsafe code snapshot member")
                archive.writestr("code/phase1/" + name, code.read(name))
        for name in ("run.json", "metrics.json", "scores.json", "code_snapshot_manifest.json"):
            if (run / name).exists():
                archive.write(run / name, "code/phase1/selected_run/" + name)
        for name in ("config.json", "tokenizer.json", "provenance.json"):
            archive.write(run / "best" / name, "code/phase1/selected_run/" + name)
        for item in a.additional_phase:
            phase, folder = item.split("=", 1)
            if not re.fullmatch(r"phase[2-9][0-9]*", phase) or not Path(folder).is_dir():
                raise ValueError("Additional phase must be phaseN=/existing/directory")
            for file in Path(folder).rglob("*"):
                if file.is_file() and not file.is_symlink():
                    archive.write(file, f"code/{phase}/{file.relative_to(folder)}")
        archive.write(pdf, f"report/{stem}-report.pdf")
        archive.write(declaration, "report/LLM_declaration" + declaration.suffix)
        archive.writestr("code/phase1/SUBMISSION_METADATA.json", json.dumps({"github": a.github_url,
            "selected_run": str(run.resolve()), "weights_included": False, "scope": "Phase 1 plus explicitly supplied completed phases"}, indent=2))
    destination.with_suffix(".tmp").replace(destination)
    with zipfile.ZipFile(destination) as archive:
        assert {name.split("/")[0] for name in archive.namelist()} == {"code", "report"}
        assert f"report/{stem}-report.pdf" in archive.namelist()
        assert archive.testzip() is None
    print(destination)

if __name__ == "__main__":
    main()
