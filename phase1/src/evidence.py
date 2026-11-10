"""Generate a factual draft report and preserve reproducibility evidence per run."""
import argparse
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
from .common import *

def code_files():
    for directory in (ROOT / "src", ROOT / "core", ROOT / "configs", ROOT / "stages", ROOT / "tests", ROOT / "evaluation"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".yaml", ".txt", ".md"):
                yield path
    for path in sorted(ROOT.iterdir()):
        if path.is_file() and path.suffix in (".py", ".txt", ".md"):
            yield path

def snapshot(output):
    output = Path(output)
    with zipfile.ZipFile(output / "code_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code_files():
            archive.write(path, str(path.relative_to(ROOT)))
    write_json(output / "code_snapshot_manifest.json", {"code_sha256": code_hash(),
        "zip_sha256": digest_file(output / "code_snapshot.zip"),
        "files": {str(p.relative_to(ROOT)): digest_file(p) for p in code_files()}})

def export(run, prepared):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from .plotting import pyplot
    plt = pyplot()
    run, prepared = Path(run), Path(prepared)
    cfg, metrics, manifest = read(run / "run.json"), read(run / "metrics.json"), read(prepared / "manifest.json")
    scores = read(run / "scores.json") if (run / "scores.json").exists() else {}
    evidence = run / "evidence"
    evidence.mkdir(exist_ok=True)
    history = metrics["history"]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.plot([h["step"] for h in history], [h["validation"]["loss"] for h in history], marker="o")
    ax.set(xlabel="Optimizer update", ylabel="Validation NLL (nats/token)", title="Recorded validation loss")
    fig.tight_layout()
    fig.savefig(evidence / "loss_curve.png", dpi=160)
    plt.close(fig)
    sections = [
        ("Abstract", f"This is an automatically generated draft for {cfg['settings']['arch']}. "
         f"It records {cfg['parameters']:,} unique parameters and {metrics['input_tokens']:,} input tokens. "
         "Human review and the official LLM declaration are required before final assignment submission."),
        ("Introduction / Motivation", "The task is Phase-1 legal continuation: retain a 512-word prompt and generate exactly 128 words. "
         "The primary quality metric is the supplied BERTScore F1. Word count and ZIP structure are correctness gates."),
        ("Methods", f"Weights were randomly initialized. Tokenizer: {manifest['identity']['config']['tokenizer']}, "
         f"actual vocabulary {manifest['tokenizer']['actual_vocab_size']:,}. Data scope: {manifest['scope']}. "
         "All four sources feed a disk-backed, globally deduplicated hash split. The tokenizer uses training documents only. "
         "EOS-separated streams are packed separately by source. Model and optimizer settings are preserved in run.json."),
        ("Experiments", f"Seed {cfg['settings']['seed']}; context {cfg['settings']['context']}; "
         f"updates {metrics['step']}/{cfg['planned_steps']}; training complete: {metrics['complete']}. "
         f"Best validation NLL: {metrics['best_loss']:.6f}; perplexity: {math.exp(min(700, metrics['best_loss'])):.4f}. "
         f"Official scores: {json.dumps(scores) if scores else 'not run'}. "
         "The comparison report lists other architectures/tokenizers; this run alone does not establish superiority."),
        ("Analysis", "Per-source validation losses and token counts are recorded in metrics.json. "
         "Per-token perplexity must not be used to rank different tokenizers. Corpus vocabulary overlap and citation-like "
         "span diagnostics are descriptive and do not establish legal correctness. Failures and return codes are retained "
         "in the experiment folder's logs, gate reports and pipeline.json."),
        ("Conclusions", "Choose the final candidate using common held-out official BERTScore, inspect secondary metrics "
         "and failure diagnostics, and validate predictions on official test IDs. Further work includes repeated seeds, "
         "expert review, and additional compute only when measured costs justify it. No legal validity claim is made."),
        ("Required author additions", "Before Hello IITK submission, add your identity, GitHub repository link, the completed "
         "provided LLM questionnaire, comparison interpretation, and the specific failed approaches and observed causes. "
         "This draft is not a completed course report."),
    ]
    (evidence / "report.md").write_text("# Phase 1 run report - DRAFT\n\n" + "\n\n".join(f"## {title}\n\n{text}" for title, text in sections))
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    story = [Paragraph("Phase 1 run report - DRAFT", styles["Title"]), Spacer(1, 12)]
    for title, text in sections:
        story.extend([Paragraph(title, styles["Heading2"]), Paragraph(escape(text), styles["BodyText"]), Spacer(1, 8)])
    story += [Image(str(evidence / "loss_curve.png"), width=460, height=224)]
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 24, "Generated from recorded run artifacts | Draft for review")
        canvas.drawRightString(A4[0]-42, 24, str(doc.page))
    SimpleDocTemplate(str(evidence / "report.pdf"), pagesize=A4, leftMargin=42, rightMargin=42,
                      topMargin=38, bottomMargin=42).build(story, onFirstPage=footer, onLaterPages=footer)
    write_json(evidence / "artifact_manifest.json", {"run_config_sha256": digest_file(run / "run.json"),
        "metrics_sha256": digest_file(run / "metrics.json"), "data_manifest_sha256": digest_file(prepared / "manifest.json"),
        "report_status": "draft; author review and course declaration required"})

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--prepared", required=True)
    a = p.parse_args()
    export(a.run, a.prepared)
