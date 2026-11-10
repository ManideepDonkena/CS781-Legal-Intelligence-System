"""Write one Kaggle notebook per phase (phase1_kaggle.ipynb ... phase7_kaggle.ipynb).

Every notebook: (1) settings, (2) setup - get the code (attached Kaggle dataset or GitHub), download the Phase 1
weights from the GitHub release models-v1, and rebuild the folder layout the code was written for (the code uses
fixed paths such as /home/mani/cs781/Capstone/Phase4_CJPE/src and /workspace/models/...; Kaggle runs as root, so
the same paths are created there), (3) the phase's main recipe (as in its README), (4) the 9 Nov LoRMA run of that
phase, (5) the scores.  QUICK = True runs a short smoke test; QUICK = False runs the full recipe (hours on Kaggle).

  python make_notebooks.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/ManideepDonkena/CS781-Legal-Intelligence-System"
REL = REPO_URL + "/releases/download/models-v1/"

PHASES = {
    1: ("Phase1_LM", "Phase_1_Pretraining", "Legal language model (ROPEFIX_295M)"),
    2: ("Phase2_LNER", "Phase_2_LNER", "Legal named entities (L-NER)"),
    3: ("Phase3_RR", "Phase_3_RR", "Rhetorical roles"),
    4: ("Phase4_CJPE", "Phase_4_CJPE", "Court judgment prediction (CJPE)"),
    5: ("Phase5_LSI", "Phase_5_LSI", "Statute identification (LSI)"),
    6: ("Phase6_SUM", "hf/Phase_6_Summ", "Summarization"),
    7: ("Phase7_PCR", "Phase_7_PCR", "Prior case retrieval (PCR)"),
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


SETUP = r'''
# ---- setup: code, weights, folder layout ----------------------------------------------------------------------
import glob, os, shutil, subprocess, sys, urllib.request

def sh(cmd):
    """Run a shell command, print its output as it comes, stop on error."""
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash")
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")

# 1. the code: an attached Kaggle dataset that holds phase1..phase7 (the unzipped archive's code/ folder),
#    else a clone of the GitHub repository (the repository must be public for this)
found = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/phase1/final_model", recursive=True)]
if found:
    REPO = os.path.dirname(found[0])
else:
    REPO = "/kaggle/working/repo"
    if not os.path.isdir(REPO):
        sh(f"git clone --depth 1 {REPO_URL} {REPO}")
print("code:", REPO)
sh(f"pip install -q -r {REPO}/requirements.txt || true")

# 2. the layout the code expects
CAP = "/home/mani/cs781/Capstone"
LLM = "/home/mani/cs781_phase1/Legal_LLM"
os.makedirs(os.path.dirname(LLM), exist_ok=True)
if not os.path.exists(LLM):
    shutil.copytree(f"{REPO}/phase1/final_model", LLM)                 # model code (modern_arch.py, rope.py, ...)
NAMES = {2: "Phase2_LNER", 3: "Phase3_RR", 4: "Phase4_CJPE", 5: "Phase5_LSI", 6: "Phase6_SUM", 7: "Phase7_PCR"}
for p, name in NAMES.items():                                          # every phase's src (they import each other)
    os.makedirs(f"{CAP}/{name}/runs", exist_ok=True)
    if not os.path.exists(f"{CAP}/{name}/src"):
        shutil.copytree(f"{REPO}/phase{p}/src", f"{CAP}/{name}/src")
if not os.path.exists(f"{CAP}/shared"):
    shutil.copytree(f"{REPO}/shared", f"{CAP}/shared")

# 3. this phase's competition data (attach it as a Kaggle dataset; DATA_INPUT is its folder)
if PHASE > 1:
    target = f"{CAP}/{NAMES[PHASE]}/data/{DATA_SUBDIR}"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not os.path.exists(target):
        os.symlink(DATA_INPUT, target)
    print("data:", target, "->", os.listdir(target)[:10])

# 4. the Phase 1 weights from the GitHub release models-v1 (about 1.2 GB)
W = "/workspace/models/cs781_phase1_runs/ROPEFIX_295M"
os.makedirs(W, exist_ok=True)
for name, dest in [("phase1_ROPEFIX_295M_weights.pt", "best.pt"), ("phase1_tokenizer.json", "tokenizer.json"),
                   ("phase1_run_config.json", "run_config.json")]:
    if not os.path.exists(f"{W}/{dest}"):
        print("downloading", name, flush=True)
        urllib.request.urlretrieve(REL + name, f"{W}/{dest}")
sys.path[:0] = [LLM, f"{CAP}/shared"]
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      "| bf16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
'''

CHECK_MODEL = r'''
# ---- the Phase 1 model loads and uses my Assignment 3 code ------------------------------------------------------
sys.path.insert(0, f"{CAP}/Phase4_CJPE/src")
from ft_doc import load_model
import modern_arch, rope, attention
lm, tok, family = load_model(W, LLM)
print(type(lm).__name__, "| attention:", type(lm.layers[0].attn).__mro__[1].__module__ + ".CausalSelfAttention",
      "| RoPE:", rope.build_rope_cache.__module__ + ".build_rope_cache")
print(f"{sum(p.numel() for p in lm.parameters()):,} parameters")
'''


def notebook(p):
    folder, data_sub, title = PHASES[p]
    cells = [md(f"""
# CS781 capstone - Phase {p}: {title}
Donkena Manideep (261040623).  Runs the code of `phase{p}/` from my submission on Kaggle.

**Before running:** turn on a GPU (T4 x2 or better; the code uses bfloat16), turn on Internet (for the weights and
`pip`), and attach as Kaggle datasets: (a) the code (the `code/` folder of my archive) unless the GitHub repository is
public, (b) the Phase {p} competition data{' (not needed: only the trained model is used here)' if p == 1 else ''}.
Set `DATA_INPUT` below to the folder of (b).  `QUICK = True` runs short smoke tests; `False` runs the full recipe
(the full runs took 15-95 minutes each on an RTX PRO 5000 and take several times longer on a T4).
"""),
             code(f'''
PHASE = {p}
QUICK = True                                    # False = the full recipe
DATA_INPUT = "/kaggle/input/cs781-phase{p}-data/{data_sub.split('/')[-1]}"   # <- the attached competition data
DATA_SUBDIR = "{data_sub}"
REPO_URL = "{REPO_URL}"
REL = "{REL}"
'''),
             code(SETUP)]
    if p == 1:
        cells += [md("## The model and the code it uses\nThe final model's code is `phase1/final_model/` "
                     "(`modern_arch.py` uses `rope.py` and `attention.py` from my Assignment 3 unchanged)."),
                  code(CHECK_MODEL),
                  code(r'''
# ---- loss on a few sentences and a short continuation (greedy) --------------------------------------------------
lm = lm.cuda().eval()
text = "The appellant filed a writ petition under Article 226 of the Constitution challenging the order of the"
ids = torch.tensor([tok.encode(text)]).cuda()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    print("loss per token:", lm(ids, labels=ids)["loss"].item())
    for _ in range(40):
        nxt = lm(ids)["logits"][0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], 1)
print(tok.decode(ids[0].tolist()))
'''),
                  md("## Training (needs the Phase 1 data cache; 38.7 GPU-hours for the full model)\n"
                     "The exact command that trained ROPEFIX_295M is below; with `QUICK` it trains a small model "
                     "for 20M tokens only to show the program runs. See `phase1/final_model/README.md`."),
                  code(r'''
cmd = (f"cd {LLM} && python studies/train_job.py --id ROPEFIX_295M --study ROPEFIX --cache final_bpe32k_mixA_3B "
       "--d-model 1024 --layers 23 --heads 16 --tokens 3000000000 --bs 8 --ga 4 --lr 1e-3 --wd 0.1 --seed 42 "
       "--extra '{\"rope_float\": true}' --keep-best")
if QUICK:
    cmd = cmd.replace("--d-model 1024 --layers 23", "--d-model 256 --layers 4").replace("3000000000", "20000000")
print(cmd)   # needs the token cache built by studies/build_cache.py from the Phase 1 data; run it once that exists:
# sh(cmd)
''')]
        return cells
    s = f"{{CAP}}/{folder}/src"
    if p == 2:
        cells += [md("## Main recipe (README of phase 2)"),
                  code(rf'''
q = "--epochs 1 --max_steps 20" if QUICK else ""
sh(f"cd {s} && python train_ner.py --backbone {{W}} --kfold 5:0 --out_dir ../runs/nb_k0 {{q}}")
'''),
                  md("## Fine-tuning methods, LoRMA and distillation (9 Nov study, 5 held-out splits)\n"
                     "`adapters.py` uses my Assignment 4 LoRMA functions (`q4_peft_lorma.py`, unchanged)."),
                  code(rf'''
q = "--epochs 1 --max_steps 20" if QUICK else ""
for method in ["frozen", "full", "lora", "lorma"]:
    sh(f"cd {s} && python train_adapt.py --backbone {{W}} --method {{method}} --kfold 5:0 --out_dir ../runs/adapt/{{method}}_k0 {{q}}")
sh(f"cd {s} && python summarize_adapt.py")
''')]
    elif p == 3:
        cells += [md("## Main recipe (frozen model + sentence model)"),
                  code(rf'''
sh(f"cd {s} && python embed.py --backbone {{W}} --out ../runs/emb")
sh(f"cd {s} && python train_rr.py --emb ../runs/emb --out_dir ../runs/rr_nb" + (" --epochs 2" if QUICK else ""))
'''),
                  md("## LoRMA run: adapters only in the top 8 of 23 blocks (5 held-out folds)"),
                  code(rf'''
folds = [0] if QUICK else [0, 1, 2, 3, 4]
for k in folds:
    sh(f"cd {s} && python finetune.py --backbone {{W}} --fold {{k}} --context 256 --max_sent 256 --lr 5e-4 "
       f"--lorma_rank 16 --lorma_alpha 32 --lorma_top 8 --no_extract --out ../runs/lorma/top8_k{{k}}"
       + (" --epochs 1" if QUICK else ""))
''')]
    elif p == 4:
        cells += [md("## Main recipe: text repair, TAPT, fine-tuning on the last 2,048 tokens"),
                  code(rf'''
q = " --limit 200 --epochs 1" if QUICK else " --epochs 2 --evals_per_epoch 2"
T = "/workspace/models/cs781_phase45/tapt_cjpe_fix"
sh(f"cd {s} && python text_fix.py")
sh(f"cd {s} && python tapt.py --task cjpe --out {{T}} --text_fix" + (" --limit 200" if QUICK else ""))
sh(f"cd {s} && python ft_doc.py --task cjpe --backbone {{T}} --text_fix --out ../runs/ft_tapt_fix_e2" + q)
'''),
                  md("## LoRMA run: LoRMA on top of the TAPT model (1 epoch, adapter lr 5e-5)"),
                  code(rf'''
sh(f"cd {s} && python ft_doc.py --task cjpe --backbone {{T}} --text_fix --epochs 1 --evals_per_epoch 2 --lr 5e-5 "
   f"--lorma_rank 16 --lorma_alpha 32 --out ../runs/lorma/tapt_lorma_e1" + (" --limit 200" if QUICK else ""))
''')]
    elif p == 5:
        s4 = "{CAP}/Phase4_CJPE/src"
        cells += [md("## Main recipe: own TF-IDF + fine-tuned model, cross-fitted on dev"),
                  code(rf'''
q = " --limit 200 --epochs 1" if QUICK else " --epochs 3"
T = "/workspace/models/cs781_phase45/tapt_lsi_compact"
sh(f"cd {{CAP}}/shared && python wordcount_run.py --task lsi --out tfidf_scratch_xf5 --xfit 5")
sh(f"cd {s4} && python tapt.py --task lsi --out {{T}} --text_fix --tail_tokens 512" + (" --limit 200" if QUICK else ""))
for f in [0, 1]:
    sh(f"cd {s4} && python ft_doc.py --task lsi --backbone {{T}} --text_fix --tail_tokens 512 --dev_fold {{f}} "
       f"--out {{CAP}}/Phase5_LSI/runs/xf_tapt_compact_e3_f{{f}}" + q)
sh(f"cd {s} && python merge_xfit.py xf_tapt_compact_e3 xf_tapt_compact_e3_f0 xf_tapt_compact_e3_f1 --tag e3")
'''),
                  md("## LoRMA run: rank 64 with rank-stabilised scale alpha/sqrt(r) (1 epoch, adapter lr 2e-4)"),
                  code(rf'''
sh(f"cd {s4} && python ft_doc.py --task lsi --backbone {{T}} --text_fix --tail_tokens 512 --epochs 1 --lr 2e-4 "
   f"--lorma_rank 64 --lorma_alpha 16 --lorma_rs --out {{CAP}}/Phase5_LSI/runs/lorma/rs64_e1" + (" --limit 200" if QUICK else ""))
''')]
    elif p == 6:
        cells += [md("## Main recipe: extractive model, abstractive opening, hybrid summary"),
                  code(rf'''
M = "/workspace/models/cs781_phase6"
sh(f"cd {s} && python extractive.py --out ../runs/ext_xfit5 --xfit 5")
sh(f"cd {s} && python abstractive.py train --out {{M}}/abs_1ep_lr5e5 --lr 5e-5 --epochs 1")
sh(f"cd {s} && python abstractive.py generate --ckpt {{M}}/abs_1ep_lr5e5/e1.pt --split dev --out ../runs/abs_1ep_lr5e5 "
   f"--tag _n100 --limit " + ("10" if QUICK else "100"))
'''),
                  md("## LoRMA run: feed-forward layers only, folded into the weights before generating"),
                  code(rf'''
sh(f"cd {s} && python abstractive.py train --out {{M}}/lorma_ffn_r16 --epochs 1 --lr 5e-4 --lorma_rank 16 "
   f"--lorma_alpha 32 --lorma_ffn_only")
sh(f"cd {s} && python abstractive.py generate --ckpt {{M}}/lorma_ffn_r16/e1.pt --split dev --out ../runs/lorma/ffn_r16 "
   f"--tag _n100 --limit " + ("10" if QUICK else "100"))
''')]
    elif p == 7:
        cells += [md("## Main recipe: Phrase-BM25, citation matching, fusion (own code)"),
                  code(rf'''
sh(f"cd {s} && python advanced_pcr.py dev")
'''),
                  md("## LoRMA run: my own model as a retriever (contrastive training with BM25 hard negatives)\n"
                     "Needs the train split in the data folder and `runs/advanced_pcr/dev_phrase_scores.json` "
                     "(written by the cell above)."),
                  code(rf'''
sh(f"cd {s} && python lorma_retriever.py --backbone {{W}} --out ../runs/lorma" + (" --pairs 4" if QUICK else ""))
''')]
    cells += [md("## Scores"),
              code(rf'''
for f in sorted(glob.glob(f"{{CAP}}/{folder}/runs/**/*.json", recursive=True))[-20:]:
    print(f)
''')]
    if p == 2:
        cells[-1] = code(rf'''
import json
for f in sorted(glob.glob(f"{{CAP}}/{folder}/runs/**/log.json", recursive=True)):
    j = json.load(open(f)); print(f.split("/runs/")[1], j.get("macro_f1"))
''')
    return cells


for p in PHASES:
    nb = {"cells": [code(CHECK_MODEL)] if False else notebook(p),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
          "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(HERE, f"phase{p}_kaggle.ipynb")
    json.dump(nb, open(path, "w"), indent=1)
    print("wrote", path)
