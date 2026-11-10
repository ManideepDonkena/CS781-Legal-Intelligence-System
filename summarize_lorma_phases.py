"""Collect the 9 Nov LoRMA study of Phases 3-7 (offline, held-out/dev data only; nothing was uploaded) into one table.

Each phase's LoRMA run is compared with the full fine-tuning run that had the same data, model and settings, read
from that run's own log.

  python summarize_lorma_phases.py      -> table on screen + lorma_phases_summary.json
"""
import json
import re
import statistics as st
from pathlib import Path

C = Path("/home/mani/cs781/Capstone")
out = {}

# Phase 3: sentence classifier alone, 5 held-out folds; full fine-tuning = runs/ft_c256_k*.log
lor, full, mins, gb = [], [], [], []
for k in range(5):
    f = C / f"Phase3_RR/runs/lorma/top8_k{k}/classifier_log.json"
    if not f.exists():
        continue
    j = json.load(open(f))
    lor.append(100 * j["scores"]["all13"])
    mins.append(j["minutes"])
    gb.append(j["peak_gb"])
    txt = open(C / f"Phase3_RR/runs/ft_c256_k{k}.log").read()
    full.append(float(re.search(r"macroF1 all13 ([\d.]+)", txt).group(1)))
if lor:
    full_min = []
    for k in range(len(lor)):
        steps = re.findall(r"step \d+/\d+ loss [\d.]+ \((\d+)s\)", open(C / f"Phase3_RR/runs/ft_c256_k{k}.log").read())
        full_min.append(int(steps[-1]) / 60)
    out["phase3"] = {"metric": "sentence macro-F1 (13 roles), 5 held-out folds", "lorma_per_fold": lor,
                     "full_per_fold": full, "lorma_mean": st.mean(lor), "full_mean": st.mean(full),
                     "lorma_train_min": st.mean(mins), "full_train_min_approx": st.mean(full_min),
                     "lorma_peak_gb": st.mean(gb)}


def ft_log(path):
    j = json.load(open(path))
    return {r["tag"]: r for r in j["log"]}, j["args"]


# Phase 4: dev macro-F1 at threshold 0.5 and at the dev median; full fine-tuning = runs/ft_tapt_fix_e2 (2 epochs)
p = C / "Phase4_CJPE/runs/lorma/tapt_lorma_e1/log.json"
if p.exists():
    lg, a = ft_log(p)
    fl, _ = ft_log(C / "Phase4_CJPE/runs/ft_tapt_fix_e2/log.json")
    out["phase4"] = {"metric": "dev macro-F1 (@0.5 / @median)", "trainable_lm_params": a.get("trainable_lm_params"),
                     "lorma": {t: r["text"] for t, r in lg.items()},
                     "lorma_peak_gb": max(r.get("peak_gb", 0) for r in lg.values()),
                     "lorma_minutes": max(r["time"] for r in lg.values()) / 60,
                     "full": {t: r["text"] for t, r in fl.items()}}

# Phase 5: dev macro-F1 with thresholds tuned on the other half; full fine-tuning = runs/ft_tapt_compact_e3
p = C / "Phase5_LSI/runs/lorma/rs64_e1/log.json"
if p.exists():
    lg, a = ft_log(p)
    fl, _ = ft_log(C / "Phase5_LSI/runs/ft_tapt_compact_e3/log.json")
    out["phase5"] = {"metric": "dev macro-F1 (@0.5 | thresholds tuned on the other half)",
                     "trainable_lm_params": a.get("trainable_lm_params"),
                     "lorma": {t: r["text"] for t, r in lg.items()},
                     "lorma_peak_gb": max(r.get("peak_gb", 0) for r in lg.values()),
                     "lorma_minutes": max(r["time"] for r in lg.values()) / 60,
                     "full": {t: r["text"] for t, r in fl.items()}}

# Phase 6: ROUGE on the first 100 dev cases, written summary only; full fine-tuning = abs_1ep_lr5e5 (1 epoch)
p = C / "Phase6_SUM/runs/lorma/ffn_r16/dev_result_n100.json"
if p.exists():
    lor6 = json.load(open(p))
    full6 = json.load(open(C / "Phase6_SUM/runs/abs_1ep_lr5e5/dev_result_n100.json"))
    tl = json.load(open("/workspace/models/cs781_lorma/p6_ffn_r16/train_log.json"))
    tf = json.load(open("/workspace/models/cs781_phase6/abs_1ep_lr5e5/train_log.json"))
    out["phase6"] = {"metric": "ROUGE-1/2/L, 100 dev cases (abstractive part alone)",
                     "lorma": lor6["dev"], "lorma_words": lor6["mean_words"], "lorma_dev_loss": tl["dev_loss"],
                     "full": full6["dev"], "full_words": full6["mean_words"], "full_dev_loss": tf["dev_loss"]}

# Phase 7: dev Micro-F1@7, frozen vs LoRMA-trained retriever, alone and fused with Phrase-BM25
p = C / "Phase7_PCR/runs/lorma/retriever_log.json"
if p.exists():
    out["phase7"] = json.load(open(p))

json.dump(out, open(C / "lorma_phases_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
