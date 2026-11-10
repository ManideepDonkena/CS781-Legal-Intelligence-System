#!/usr/bin/env python3
"""Build results/STUDY_REPORT.md from results/results.jsonl (+ queue state).

Each study has a predeclared decision rule expressed in units of eps = seed std of validation BPB
(measured by the G3 study). A difference is called real only if it exceeds 2*eps. Run any time.
"""
import json, statistics as st, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def rows():
    f = RES / "results.jsonl"
    out = {}
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line); out[r["id"]] = r          # latest row per id wins
    return list(out.values())


def fmt(x, n=4): return "-" if x is None else f"{x:.{n}f}"


def table(rs, cols):
    head = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(c[1](r)) for c in cols) + " |" for r in rs) + "\n"


COLS = [("run", lambda r: r["id"]), ("params M", lambda r: f"{r['params_M']:.1f}"), ("vocab", lambda r: r["vocab"]),
        ("L×d", lambda r: f"{r['layers']}×{r['d_model']}"), ("kv", lambda r: r["kv_heads"]), ("tied", lambda r: r["tied"]),
        ("tokens M", lambda r: f"{r['tokens']/1e6:.0f}"), ("epochs", lambda r: f"{r['epochs']:.2f}"),
        ("val BPB (SC)", lambda r: fmt(r["val_bpb"])), ("tok/s k", lambda r: f"{(r['tok_per_s'] or 0)/1e3:.0f}"),
        ("GPU-h", lambda r: f"{r['gpu_hours']:.2f}")]


def eps_estimate(rs):
    g = [r["val_bpb"] for r in rs if r["study"] == "G3"]
    return (st.stdev(g), len(g)) if len(g) >= 2 else (None, len(g))


def verdict(delta, eps, k=2.0):
    if eps is None: return "undetermined (no seed-noise estimate yet)"
    return ("**real** (|Δ| > %.0f·ε)" if abs(delta) > k * eps else "within noise (|Δ| ≤ %.0f·ε)") % k


def main():
    rs = rows()
    by = defaultdict(list)
    for r in rs: by[r["study"]].append(r)
    eps, n = eps_estimate(rs)
    L = [f"# Legal LLM — study report (auto-generated {time.strftime('%Y-%m-%d %H:%M')})\n\n",
         f"{len(rs)} completed runs. All BPB values are bits-per-byte on the same frozen Supreme Court validation "
         f"text (content-hash split, seed 42), comparable across tokenizers.\n\n"]
    L.append(f"**Seed noise ε (G3, n={n}):** " + (f"{eps:.4f} BPB (std over seeds)\n\n" if eps else "not yet measured\n\n"))
    qf = RES / "queue_state.json"
    queue = json.loads(qf.read_text()) if qf.exists() else {}
    cnt = defaultdict(int)
    for s in queue.values(): cnt[s["status"]] += 1
    L.append("Queue: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())) + "\n")
    for study in sorted(by):
        L.append(f"\n## {study}\n\n")
        grp = sorted(by[study], key=lambda r: r["val_bpb"])
        L.append(table(sorted(by[study], key=lambda r: r["id"]), COLS))
        if len(grp) >= 2 and study != "G3":
            best, second = grp[0], grp[1]
            L.append(f"\nBest: `{best['id']}` ({best['val_bpb']:.4f}); runner-up `{second['id']}` "
                     f"(Δ={second['val_bpb']-best['val_bpb']:+.4f}) → {verdict(second['val_bpb']-best['val_bpb'], eps)}\n")
        srcs = ["validation_supreme_court", "validation_high_courts", "validation_tribunals", "validation_acts"]
        L.append("\nPer-source BPB (best checkpoint):\n\n| run | " + " | ".join(s.replace("validation_", "") for s in srcs)
                 + " |\n|---|" + "---|" * len(srcs) + "\n")
        for r in grp:
            L.append(f"| {r['id']} | " + " | ".join(fmt(r["bpb_by_source"].get(s)) for s in srcs) + " |\n")
    (RES / "STUDY_REPORT.md").write_text("".join(L))
    print("wrote", RES / "STUDY_REPORT.md")


if __name__ == "__main__":
    main()
