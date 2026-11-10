"""Phase 2 adaptation study (offline, held-out data only; nothing here was uploaded to Codabench).

Every run starts from a Phase 1 model and trains the same tagger (BiLSTM + CRF on top of the language model) on 80%
of the labelled documents; the other 20% (--kfold 5:i) are scored with the official strict macro-F1.

  --method frozen   only the tagger head is trained
  --method full     every weight is trained
  --method top      the last --top_k layers + head are trained (partial / layer-wise fine-tuning)
  --method lora     LoRA adapters on all attention and feed-forward matrices + head
  --method lorma    LoRMA adapters (multiplicative low-rank) on the same matrices + head
  --teacher DIR     knowledge distillation: also match the per-token tag distribution of a saved, fine-tuned
                    teacher tagger (loss = (1 - a) * CRF loss + a * T^2 * KL)

Besides the score it records trainable parameters, peak GPU memory, training minutes and inference speed.
"""
import argparse
import json
import math
import os
import random
import time

import torch

import adapters
import lner
from train_ner import build_docs, evaluate, make_batch, predict, window_item, windows


def make_tagger(args, lm, family, tags):
    return lner.Tagger(lm, family, len(tags), "causal", "bilstm", args.dropout, args.lstm_hidden, False, False,
                       args.method == "frozen", tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/mani/cs781/Capstone/Phase2_LNER/data/Phase_2_LNER")
    ap.add_argument("--repo", default="/home/mani/cs781_phase1/Legal_LLM")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--method", choices=["frozen", "full", "top", "lora", "lorma"], required=True)
    ap.add_argument("--kfold", default="5:0")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--eval_stride", type=int, default=512)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5, help="learning rate of the trained backbone weights/adapters")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lstm_hidden", type=int, default=384)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--teacher", default="", help="run directory of a saved teacher (model.pt + args)")
    ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--kd_temp", type=float, default=2.0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    args = ap.parse_args()
    args.crf, args.repeat, args.attention, args.head = True, False, "causal", "bilstm"  # what predict() reads

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    labels = lner.load_labels(args.data)
    tags = lner.bio_tags(labels)
    tag2id = {t: i for i, t in enumerate(tags)}

    lm, tok, family = lner.load_backbone(args.backbone, args.repo)
    backbone_params = sum(p.numel() for p in lm.parameters())
    docs = []
    for fold in ("fold1", "fold2"):
        texts, gold = lner.load_fold(args.data, fold)
        docs += build_docs(tok, texts, gold, tag2id)
    k, i = (int(x) for x in args.kfold.split(":"))
    ids = sorted(d["id"] for d in docs)
    random.Random(0).shuffle(ids)  # same split as train_ner.py --kfold
    held = set(ids[i::k])
    eval_docs = [d for d in docs if d["id"] in held]
    eval_gold = {d["id"]: d["gold"] for d in eval_docs}
    train_docs = [d for d in docs if d["id"] not in held]

    if args.method in ("lora", "lorma"):
        lm_params = adapters.add_adapters(lm, args.method, args.rank, args.alpha)
    elif args.method == "top":
        lm_params = adapters.unfreeze_top(lm, args.top_k)
    elif args.method == "full":
        lm_params = list(lm.parameters())
    else:
        lm_params = []
        for p in lm.parameters():
            p.requires_grad_(False)
    model = make_tagger(args, lm, family, tags).to(device)
    head = model.head_parameters()
    groups = [{"params": head, "lr": args.head_lr}]
    if lm_params:
        groups.append({"params": lm_params, "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in groups]
    trainable = sum(p.numel() for g in groups for p in g["params"])

    teacher = None
    if args.teacher:
        ck = torch.load(f"{args.teacher}/model.pt", map_location="cpu", weights_only=False)
        ta = argparse.Namespace(**ck["args"])
        tlm, _, tfam = lner.load_backbone(ta.backbone, args.repo)
        if ta.method in ("lora", "lorma"):
            adapters.add_adapters(tlm, ta.method, ta.rank, ta.alpha)
        teacher = make_tagger(ta, tlm, tfam, tags)
        teacher.load_state_dict(ck["state"])
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    per_epoch = sum(len(windows(len(d["ids"]), args.width, args.stride)) for d in train_docs) // args.batch
    total = max(1, per_epoch * args.epochs)
    warm = max(1, int(0.1 * total))
    log = {"args": vars(args), "backbone_params": backbone_params, "trainable_params": trainable,
           "train_docs": len(train_docs), "eval_docs": len(eval_docs), "epochs": []}
    print(json.dumps({k: log[k] for k in ("backbone_params", "trainable_params", "train_docs", "eval_docs")}),
          flush=True)
    torch.cuda.reset_peak_memory_stats()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        if args.method == "frozen":
            model.lm.eval()
        items = []
        for d in train_docs:
            for s in windows(len(d["ids"]), args.width, args.stride, shift=random.randrange(args.stride)):
                items.append(window_item(d, s, args.width))
        random.shuffle(items)
        running, seen = 0.0, 0
        for j in range(0, len(items) - args.batch + 1, args.batch):
            x, keep, y, feats, mem = make_batch(items[j:j + args.batch], args.width, False, device)
            scale = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * scale
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, keep, feats, mem)
            loss = model.crf.nll(logits.float(), y, keep)
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_logits = teacher(x, keep, feats, mem)
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * adapters.distill_loss(
                    logits, t_logits, keep, args.kd_temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            running += loss.item()
            seen += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        rec = {"epoch": epoch + 1, "loss": running / max(1, seen), "minutes": (time.time() - t0) / 60}
        log["epochs"].append(rec)
        print(json.dumps(rec), flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    log["train_minutes"] = (time.time() - t0) / 60
    log["peak_train_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30

    t1 = time.time()
    spans, _ = predict(model, eval_docs, args, tags, device)
    log["infer_seconds"] = time.time() - t1
    log["eval_tokens"] = sum(len(d["ids"]) for d in eval_docs)
    scores = evaluate(spans, eval_docs, eval_gold, labels, [0.34])
    log["macro_f1"] = {k: round(v["macro_f1"], 4) for k, v in scores.items()}
    log["scores"] = scores
    print(json.dumps({k: log[k] for k in ("macro_f1", "train_minutes", "peak_train_gb", "infer_seconds")}),
          flush=True)
    with open(f"{args.out_dir}/log.json", "w") as f:
        json.dump(log, f, indent=1, default=float)
    if args.save:
        torch.save({"state": model.state_dict(), "args": vars(args), "tags": tags}, f"{args.out_dir}/model.pt")


if __name__ == "__main__":
    main()
