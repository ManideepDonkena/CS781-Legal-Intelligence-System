"""One comparison entry point: official primary metric, plots, per-source losses and diagnostics."""
import argparse
import math
import re
from collections import Counter
from pathlib import Path
from src.common import *

def diagnostics(predictions, truth, lexicon):
    generated = " ".join(row["text"] for row in predictions.values())
    words = generated.lower().split()
    alphabetic = re.findall(r"[^\W\d_]{4,}", generated.lower(), re.UNICODE)
    citations = re.compile(r"\b(?:Sections?|Articles?)\s+\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))*|\(\d{4}\)\s+\d+\s+SCC\s+\d+", re.I)
    proposed, matches = 0, 0
    for key, row in predictions.items():
        forms = citations.findall(row["text"])
        reference = set(citations.findall(truth[key]["text"])) if key in truth else set()
        proposed += len(forms)
        matches += sum(x in reference for x in forms)
    trigrams = list(zip(words, words[1:], words[2:]))
    return {"corpus_vocabulary_hit_rate": sum(w in lexicon for w in alphabetic)/len(alphabetic) if alphabetic else 0,
            "citation_like_spans": proposed, "exact_reference_citation_overlap": matches/proposed if proposed else None,
            "repeated_trigram_fraction": 1-len(set(trigrams))/len(trigrams) if trigrams else 0,
            "distinct_word_fraction": len(set(words))/len(words) if words else 0,
            "interpretation": "Style/overlap diagnostics only. Not legal validity, factuality, citation correctness or a legalness score."}


def compare(work):
    from src.plotting import pyplot
    plt = pyplot()
    work = Path(work).resolve()
    report_dir = work / "analysis"
    report_dir.mkdir(exist_ok=True)
    preparations = sorted(p.parent for p in (work / "data/prepared").glob("*/manifest.json"))
    if preparations:
        from tokenizer_study import compare as compare_tokenizers
        compare_tokenizers(preparations, report_dir)
    rows = []
    for runfile in sorted((work / "runs").glob("*/*/seed_*/run.json")):
        run = runfile.parent
        settings = read(runfile)
        metrics_file = run / "metrics.json"
        if not metrics_file.exists():
            continue
        metrics = read(metrics_file)
        variant = run.parents[1].name
        prepared = work / "data/prepared" / variant
        scores = read(run / "scores.json") if (run / "scores.json").exists() else {}
        row = {"run": str(run.relative_to(work)), "variant": variant, "arch": run.parent.name,
               "seed": settings["settings"]["seed"], "parameters": settings["parameters"],
               "complete": metrics["complete"], "input_tokens": metrics["input_tokens"],
               "source_tokens": metrics.get("source_tokens", {}),
               "input_tokens_per_compute_second": metrics.get("input_tokens_per_compute_second"),
               "peak_gpu_allocated_bytes_rank0": metrics.get("peak_gpu_allocated_bytes_rank0"),
               "loss": metrics["best_loss"], "perplexity": math.exp(metrics["best_loss"]) if metrics["best_loss"] < 700 else None,
               "official_scores": scores, "per_source": min(metrics["history"], key=lambda h: h["validation"]["loss"])["validation"]["per_source"],
               "history": metrics["history"]}
        prediction = run / "validation/prediction.json"
        if prediction.exists():
            row["diagnostics"] = diagnostics(read(prediction), read(prepared / "held_out_truth.json"), read(prepared / "corpus_lexicon.json"))
        rows.append(row)
    candidates = [r for r in rows if r["complete"] and "bertscore_f1" in r["official_scores"]]
    truth_hashes = {r["official_scores"]["truth_sha256"] for r in candidates}
    if len(truth_hashes) > 1:
        raise ValueError("Official scores use different references; cannot select one winner")
    winner = max(candidates, key=lambda r: r["official_scores"]["bertscore_f1"]) if candidates else None
    result = {"primary_metric": "bertscore_f1", "selected_run": winner["run"] if winner else None, "runs": rows,
              "selection_scope": "highest held-out BERTScore among complete scored runs; not a significance claim",
              "limitations": "PPL is only comparable within identical tokenizer/evaluation settings. Diagnostics do not measure legal correctness."}
    write_json(report_dir / "comparison.json", result)
    lines = ["# Experiment comparison", "", "Primary: supplied BERTScore F1. Secondary: supplied ROUGE-L F1 and BLEU-4.",
             "Exactly 128 generated words is a correctness gate, not a quality metric.", "",
             "| Run | Params | Tokens | Val loss | PPL* | BERTScore F1 | ROUGE-L F1 | BLEU-4 | Complete |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        scores = r["official_scores"]
        lines.append(f"| {r['run']} | {r['parameters']:,} | {r['input_tokens']:,} | {r['loss']:.4f} | {r['perplexity']:.3f} | "
                     f"{scores.get('bertscore_f1','not run')} | {scores.get('rouge_l_f1','not run')} | {scores.get('bleu4','not run')} | {r['complete']} |")
    lines += ["", "*Do not rank different tokenizers by per-token perplexity. Equal token budgets also mean different raw-text exposure.",
              "", f"Selected run: **{result['selected_run'] or 'none: official scores required'}**.",
              "Selection is provisional until all requested runs succeed and sample outputs are reviewed.", "",
              "## Per-source validation", "", "| Run | Source | Loss | Perplexity | Targets |", "| --- | --- | ---: | ---: | ---: |"]
    for r in rows:
        for source, values in r["per_source"].items():
            lines.append(f"| {r['run']} | {source} | {values['loss']:.4f} | {math.exp(min(700,values['loss'])):.3f} | {values['targets']:,} |")
    lines += ["", "## Style and overlap diagnostics", "",
              "Corpus vocabulary overlap, repetition and reference citation overlap are in comparison.json. They do not establish legal validity or factual correctness.",
              "Human review should check invented authorities, citation accuracy, contradictions and whether the continuation follows the prompt. Do not use hidden test references to select settings.",
              "", "## Verification", "", "Gate logs and machine-readable reports are under gates/. Stage commands, return codes and failures are in pipeline.json and logs/. GPU feasibility is established only by successful server gates.", ""]
    (report_dir / "REPORT.md").write_text("\n".join(lines))
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for r in rows:
            label = f"{r['variant']}/{r['arch']}/{r['seed']}"
            axes[0].plot([h["step"] for h in r["history"]], [h["validation"]["loss"] for h in r["history"]], marker=".", label=label)
        axes[0].set(xlabel="Optimizer update", ylabel="Validation loss (nats/token)", title="Interpret within each tokenizer")
        axes[0].legend(fontsize=6)
        if candidates:
            axes[1].barh([f"{r['variant']}/{r['arch']}/{r['seed']}" for r in candidates], [r["official_scores"]["bertscore_f1"] for r in candidates])
            axes[1].set(xlabel="Official BERTScore F1", title="Held-out continuation quality")
        else:
            axes[1].text(.5,.5,"Official scores not run",ha="center",va="center")
        fig.tight_layout()
        fig.savefig(report_dir / "comparison.png", dpi=160)
        plt.close(fig)
    print(report_dir / "REPORT.md")
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", required=True)
    compare(p.parse_args().work_dir)
