"""Fine-tune a Phase 1 language model as an L-NER tagger and evaluate with the strict macro-F1.

Cross-validation:   --train fold1 --eval fold2      (and the reverse)
K-fold held-out:    --train fold1,fold2 --kfold 5:0 (... 5:4); every document is predicted by a model
                    trained on the other 80%, which is what the second stage (stage2.py) trains on
Test predictions:   --train fold1,fold2 --predict fold3 --out_dir runs/x
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import lner


def build_docs(tok, texts, gold, tag2id, memory=None):
    """`memory`: {doc id: first-pass label probabilities per token} for the second-pass tagger."""
    docs = []
    for did, text in texts.items():
        ids, offs = lner.tokenize(tok, text)
        y = lner.spans_to_tags(offs, gold[did], tag2id) if gold is not None else None
        doc = {"id": did, "text": text, "ids": ids, "offs": offs, "y": y,
               "feat": lner.token_features(text, offs), "gold": gold[did] if gold is not None else None}
        if memory is not None:
            assert len(memory[did]) == len(ids), "first-pass probabilities do not match tokenization"
            doc["mem"] = lner.document_memory(text, offs, memory[did]) if ids else torch.zeros(0, 1)
        docs.append(doc)
    return docs


def augmented(tok, doc, pool, tag2id, p_swap):
    text, spans = lner.swap_mentions(doc["text"], doc["gold"], pool, random, p_swap)
    ids, offs = lner.tokenize(tok, text)
    return {"id": doc["id"], "text": text, "ids": ids, "offs": offs, "y": lner.spans_to_tags(offs, spans, tag2id),
            "feat": lner.token_features(text, offs)}


def load_memory(memory_dir, fold):
    if not memory_dir:
        return None
    with np.load(f"{memory_dir}/{fold}_probs.npz") as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def windows(n, width, stride, shift=0):
    """Window start positions covering [0, n).  The document head and tail always get a window
    (the head carries the parties, counsel and bench); `shift` only moves the interior ones."""
    if n <= width:
        return [0]
    return sorted({0, n - width, *range(shift or stride, n - width, stride)})


def window_item(doc, s, width, labelled=True):
    return (doc["ids"][s:s + width], doc["y"][s:s + width] if labelled else None, doc["feat"][s:s + width],
            doc["mem"][s:s + width] if "mem" in doc else None)


def make_batch(items, width, repeat, device):
    """items: (ids, tags or None, features).  With `repeat`, the window is fed twice and only the
    second copy, which has already read the whole window, is labelled."""
    total = width * (2 if repeat else 1)
    ids = torch.zeros(len(items), total, dtype=torch.long)
    keep = torch.zeros(len(items), total, dtype=torch.bool)
    y = torch.full((len(items), total), -100, dtype=torch.long)
    feats = torch.zeros(len(items), total, len(lner.FEATURE_SIZES), dtype=torch.long)
    mem = None if items[0][3] is None else torch.zeros(len(items), total, items[0][3].shape[1])
    for r, (x, t, f, m) in enumerate(items):
        n = len(x)
        seq = x + x if repeat else x
        ids[r, :len(seq)] = torch.tensor(seq)
        feats[r, :len(seq)] = torch.tensor(f + f if repeat else f)
        keep[r, :len(seq)] = True
        if mem is not None:
            mem[r, :len(seq)] = torch.cat([m, m]) if repeat else m
        if t is not None:
            y[r, (n if repeat else 0):(2 * n if repeat else n)] = torch.tensor(t)
    return ids.to(device), keep.to(device), y.to(device), feats.to(device), None if mem is None else mem.to(device)


def type_probs(probs):
    """Collapse BIO tag probabilities to label probabilities: [O, label_1, ..., label_n]."""
    return torch.cat([probs[:, :1], probs[:, 1::2] + probs[:, 2::2]], dim=1)


@torch.no_grad()
def predict(model, docs, args, id2tag, device):
    """Sliding-window inference; overlapping windows are averaged with weights that favour the
    positions holding the most context for the chosen attention pattern.  Returns character spans
    and per-token label probabilities."""
    model.eval()
    out, token_probs = {}, {}
    w = args.width
    pos = torch.arange(w, dtype=torch.float)
    if args.repeat:
        weight = torch.ones(w)
    elif args.attention == "bidir" or args.head == "bilstm":
        weight = torch.minimum(pos + 1, w - pos)
    else:
        weight = pos + 1
    for doc in docs:
        n = len(doc["ids"])
        if n == 0:  # the test fold contains an empty document
            out[doc["id"]] = []
            token_probs[doc["id"]] = np.zeros((0, 1 + (len(id2tag) - 1) // 2), dtype=np.float16)
            continue
        starts = windows(n, w, args.eval_stride)
        probs = torch.zeros(n, len(id2tag))
        norm = torch.zeros(n)
        for i in range(0, len(starts), args.eval_batch):
            chunk = starts[i:i + args.eval_batch]
            items = [window_item(doc, s, w, labelled=False) for s in chunk]
            ids, keep, _, feats, mem = make_batch(items, w, args.repeat, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, keep, feats, mem)
            # A CRF needs averaged emission scores; otherwise average probabilities.
            p = logits.float().cpu() if args.crf else logits.float().softmax(-1).cpu()
            for r, s in enumerate(chunk):
                m = len(items[r][0])
                seg = p[r, m:2 * m] if args.repeat else p[r, :m]
                probs[s:s + m] += seg * weight[:m, None]
                norm[s:s + m] += weight[:m]
        probs /= norm[:, None]
        if args.crf:
            tags = model.crf.viterbi(probs)
            probs = probs.softmax(-1)
            conf = probs[torch.arange(n), torch.tensor(tags)].tolist()
        else:
            conf, tags = (x.tolist() for x in probs.max(-1))
        out[doc["id"]] = lner.tags_to_spans(doc["text"], doc["offs"], tags, conf, id2tag)
        token_probs[doc["id"]] = type_probs(probs).numpy().astype(np.float16)
    return out, token_probs


def evaluate(spans, docs, gold, labels, taus):
    res = {"raw": lner.strict_scores(gold, {d: [(s, e, l) for s, e, l, _ in v] for d, v in spans.items()}, labels)}
    text = {d["id"]: d["text"] for d in docs}
    for tau in taus:
        pred = {d: lner.propagate(text[d], lner.snap_to_words(text[d], v), tau=tau) for d, v in spans.items()}
        res[f"prop{tau}"] = lner.strict_scores(gold, pred, labels)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", default="/home/mani/cs781_phase1/Legal_LLM")
    ap.add_argument("--backbone", required=True, help="Phase 1 run directory containing best.pt")
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default="")
    ap.add_argument("--predict", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--attention", choices=["causal", "bidir"], default="causal")
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--head", choices=["linear", "bilstm"], default="linear")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--eval_stride", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freeze", action="store_true", help="train the head only")
    ap.add_argument("--layer_mix", action="store_true", help="learned mixture over all backbone layers")
    ap.add_argument("--feats", action="store_true",
                    help="add token shape, document-position and cause-title side features")
    ap.add_argument("--feat_set", choices=["all", "side"], default="all",
                    help="with --feats: all token features, or only the cause-title side")
    ap.add_argument("--lstm_hidden", type=int, default=384)
    ap.add_argument("--crf", action="store_true", help="BIO-constrained CRF output layer")
    ap.add_argument("--memory_dir", default="",
                    help="directory of {fold}_probs.npz from first-pass taggers (out-of-fold for training folds)")
    ap.add_argument("--aug", type=float, default=0.0,
                    help="per epoch, probability that a training document is replaced by a mention-swapped copy")
    ap.add_argument("--aug_swap", type=float, default=0.7, help="share of a document's entity strings swapped")
    ap.add_argument("--kfold", default="", help="K:i - hold out split i of K of the --train documents for evaluation")
    ap.add_argument("--save_probs", action="store_true", help="write per-token label probabilities")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--taus", default="0.2,0.34")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--save", action="store_true", help="write the trained tagger to out_dir/model.pt")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke-test cap")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    labels = lner.load_labels(args.data)
    tags = lner.bio_tags(labels)
    tag2id = {t: i for i, t in enumerate(tags)}
    taus = [float(x) for x in args.taus.split(",")]

    lm, tok, family = lner.load_backbone(args.backbone, args.repo)
    train_docs = []
    for fold in args.train.split(","):
        texts, gold = lner.load_fold(args.data, fold)
        train_docs += build_docs(tok, texts, gold, tag2id, load_memory(args.memory_dir, fold))
    eval_docs = eval_gold = None
    if args.kfold:
        assert not args.eval
        k, i = (int(x) for x in args.kfold.split(":"))
        ids = sorted(d["id"] for d in train_docs)
        random.Random(0).shuffle(ids)  # the split does not depend on --seed
        held = set(ids[i::k])
        eval_docs = [d for d in train_docs if d["id"] in held]
        eval_gold = {d["id"]: d["gold"] for d in eval_docs}
        train_docs = [d for d in train_docs if d["id"] not in held]
        args.eval = "heldout"
    elif args.eval:
        texts, eval_gold = lner.load_fold(args.data, args.eval)
        eval_docs = build_docs(tok, texts, None, tag2id, load_memory(args.memory_dir, args.eval))

    pool = lner.mention_pool(train_docs) if args.aug else None
    assert not (args.aug and args.memory_dir), "augmentation changes the text the memory was computed on"
    memory_dim = train_docs[0]["mem"].shape[1] if args.memory_dir else 0
    model = lner.Tagger(lm, family, len(tags), args.attention, args.head, args.dropout, args.lstm_hidden,
                        args.layer_mix, args.feat_set if args.feats else False, args.freeze, tags if args.crf else None, memory_dim).to(device)
    if args.freeze:
        for p in model.lm.parameters():
            p.requires_grad_(False)
    groups = [{"params": model.head_parameters(), "lr": args.head_lr}]
    if not args.freeze:
        groups.append({"params": list(model.lm.parameters()), "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in groups]
    per_epoch = sum(len(windows(len(d["ids"]), args.width, args.stride)) for d in train_docs) // args.batch
    total = max(1, per_epoch * args.epochs)
    warm = max(1, int(0.1 * total))

    log = {"args": vars(args), "family": family, "train_docs": len(train_docs),
           "train_tokens": sum(len(d["ids"]) for d in train_docs), "epochs": []}
    print(json.dumps({k: log[k] for k in ("family", "train_docs", "train_tokens")}), flush=True)
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        if args.freeze:
            model.lm.eval()
        items = []
        for d in train_docs:
            if args.aug and random.random() < args.aug:
                d = augmented(tok, d, pool, tag2id, args.aug_swap)
            for s in windows(len(d["ids"]), args.width, args.stride, shift=random.randrange(args.stride)):
                items.append(window_item(d, s, args.width))
        random.shuffle(items)
        running, seen = 0.0, 0
        for i in range(0, len(items) - args.batch + 1, args.batch):
            ids, keep, y, feats, mem = make_batch(items[i:i + args.batch], args.width, args.repeat, device)
            scale = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * scale
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, keep, feats, mem)
            if args.crf:
                loss = model.crf.nll(logits.float(), y, keep)
            else:
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            seen += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        rec = {"epoch": epoch + 1, "loss": running / max(1, seen), "minutes": (time.time() - t0) / 60}
        if eval_docs is not None and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            spans, probs = predict(model, eval_docs, args, tags, device)
            scores = evaluate(spans, eval_docs, eval_gold, labels, taus)
            rec["macro_f1"] = {k: round(v["macro_f1"], 4) for k, v in scores.items()}
            rec["scores"] = scores
        log["epochs"].append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "scores"}), flush=True)
        with open(f"{args.out_dir}/log.json", "w") as f:
            json.dump(log, f, indent=1)
        if args.max_steps and step >= args.max_steps:
            break

    if args.save:
        state = {k: v for k, v in model.state_dict().items() if not (args.freeze and k.startswith("lm."))}
        torch.save({"state": state, "args": vars(args), "tags": tags}, f"{args.out_dir}/model.pt")
    if eval_docs is not None:
        with open(f"{args.out_dir}/eval_spans.json", "w") as f:
            json.dump(spans, f)
        if args.save_probs:
            np.savez_compressed(f"{args.out_dir}/{args.eval}_probs.npz", **probs)
    for fold in filter(None, args.predict.split(",")):
        texts, _ = lner.load_fold(args.data, fold)
        docs = build_docs(tok, texts, None, tag2id, load_memory(args.memory_dir, fold))
        spans, probs = predict(model, docs, args, tags, device)
        with open(f"{args.out_dir}/{fold}_spans.json", "w") as f:
            json.dump(spans, f)
        if args.save_probs:
            np.savez_compressed(f"{args.out_dir}/{fold}_probs.npz", **probs)


if __name__ == "__main__":
    main()
