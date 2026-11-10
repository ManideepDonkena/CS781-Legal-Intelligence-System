"""Sentence-sequence model over frozen Phase 1 sentence vectors (from embed.py).

Each sentence vector (a learned mix of the backbone's layers) plus its position in the document goes through a
two-layer BiLSTM that reads the document's sentences in order, then a linear layer (and optionally a CRF) gives
one of the 13 roles per sentence.

Evaluation: k document folds over the 90 labelled documents; for every fold a model is trained on the rest for a
fixed number of epochs and scored on the held-out documents.  The k fold models also predict the test documents,
and their outputs are averaged.  Saves <out_dir>/heldout_seed{s}.npz and test_seed{s}.npz with emission scores.
"""
import argparse
import collections
import glob
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import rr

N_BINS = 20


def log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))
WORD_RE = re.compile(r"[a-z]+|\d+")


def build_bow(all_docs, size, min_count=3):
    """Every sentence as a set of word and word-pair ids (digits mapped to 0); id 0 = no known term.
    The vocabulary is the `size` most frequent terms over all sentences (no labels used)."""
    terms = []
    for d in all_docs:
        for s in d["sents"]:
            w = [re.sub(r"\d", "0", x) for x in WORD_RE.findall(s.lower())]
            terms.append(set(w) | {a + " " + b for a, b in zip(w, w[1:])})
    cnt = collections.Counter(t for ts in terms for t in ts)
    vocab = {t: i + 1 for i, (t, c) in enumerate(cnt.most_common(size)) if c >= min_count}
    return [torch.tensor(sorted(vocab[t] for t in ts if t in vocab) or [0]) for ts in terms], len(vocab) + 1


class CRF(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n, n))  # trans[i, j]: label i followed by label j
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def nll(self, em, y, mask):
        """em (b, t, n), y (b, t), mask (b, t) with every row a prefix of True values."""
        y = y.clamp_min(0)
        score = self.start[y[:, 0]] + em[:, 0].gather(1, y[:, :1]).squeeze(1)
        alpha = self.start + em[:, 0]
        for i in range(1, em.shape[1]):
            m = mask[:, i]
            step = self.trans[y[:, i - 1], y[:, i]] + em[:, i].gather(1, y[:, i:i + 1]).squeeze(1)
            score = score + m * step
            nxt = torch.logsumexp(alpha[:, :, None] + self.trans[None] + em[:, i, None, :], 1)
            alpha = torch.where(m[:, None], nxt, alpha)
        last = y.gather(1, (mask.sum(1) - 1)[:, None]).squeeze(1)
        logz = torch.logsumexp(alpha + self.end, 1)
        return (logz - score - self.end[last]).sum() / mask.sum()

    def params(self):
        return {k: v.detach().cpu().numpy() for k, v in (("trans", self.trans), ("start", self.start), ("end", self.end))}


def viterbi(em, trans, start, end):
    """em (t, n) numpy -> best label sequence."""
    t, n = em.shape
    score = start + em[0]
    back = np.zeros((t, n), dtype=np.int64)
    for i in range(1, t):
        s = score[:, None] + trans
        back[i] = s.argmax(0)
        score = s.max(0) + em[i]
    best = [int((score + end).argmax())]
    for i in range(t - 1, 0, -1):
        best.append(int(back[i, best[-1]]))
    return np.array(best[::-1])


class SeqModel(nn.Module):
    def __init__(self, n_layers, d, pools, hidden, dropout, crf, bow_vocab=0, bow_dim=128, tok_layers=0, heads=2,
                 sent_tf=0, n_labels=len(rr.LABELS)):
        super().__init__()
        self.mix = nn.ParameterList(nn.Parameter(torch.zeros(n_layers)) for _ in pools)
        self.pos = nn.Embedding(N_BINS, 32)
        self.dom = nn.Embedding(2, 16)
        self.bow = nn.EmbeddingBag(bow_vocab, bow_dim, mode="mean", padding_idx=0) if bow_vocab else None
        # Attention pooling over the sentence's token states (a learned mix of the saved layers): each head
        # scores every token and takes the score-weighted average, so cue words can dominate the sentence vector.
        self.tok_mix = nn.Parameter(torch.zeros(tok_layers)) if tok_layers else None
        self.att = nn.Linear(d, heads) if tok_layers else None
        self.proj = nn.Linear(d * len(pools) + 32 + 16 + 3 + (bow_dim if bow_vocab else 0) + (heads * d if tok_layers else 0),
                              hidden)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        # Optional transformer layers over the BiLSTM outputs: every sentence attends to the whole window/document.
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(2 * hidden, 8, 4 * hidden, dropout, batch_first=True,
                                                                   norm_first=True), sent_tf) if sent_tf else None
        self.out = nn.Linear(2 * hidden, n_labels)
        # Side task (Malik et al. 2022): does the role change between this sentence and the previous one?
        self.shift = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.crf = CRF(n_labels) if crf else None

    def forward(self, feats, pos_bin, dom, extra, bow=None, tk=None, valid=None):
        """feats: list (one per pooling) of (b, t, layers, d) tensors; bow: (flat ids, offsets, rows, cols)."""
        vs = []
        for w, x in zip(self.mix, feats):
            x = F.layer_norm(x.float(), x.shape[-1:])
            vs.append((w.softmax(0)[:, None] * x).sum(-2))
        if self.bow is not None:
            flat, offsets, rows, cols = bow
            v = torch.zeros(*pos_bin.shape, self.bow.embedding_dim, device=pos_bin.device)
            v[rows, cols] = self.bow(flat, offsets)
            vs.append(v)
        if self.att is not None:
            g, mask, rows, cols = tk  # g: (sentences, tokens, layers, d), already layer-normed, float16
            x = torch.einsum("ntld,l->ntd", g, self.tok_mix.softmax(0).to(g.dtype)).float()
            a = self.att(x).masked_fill(~mask[..., None], -1e4).softmax(1)  # (sentences, tokens, heads)
            pooled = torch.einsum("nth,ntd->nhd", a, x).flatten(1)
            v = torch.zeros(*pos_bin.shape, pooled.shape[1], device=pos_bin.device)
            v[rows, cols] = pooled
            vs.append(v)
        h = torch.cat(vs + [self.pos(pos_bin), self.dom(dom)[:, None].expand(-1, pos_bin.shape[1], -1), extra], -1)
        h = self.drop(torch.relu(self.proj(self.drop(h))))
        h, _ = self.lstm(h)
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=~valid)
        self.last_h = h
        return self.out(self.drop(h))

    def shift_logits(self):
        h = self.drop(self.last_h)
        return self.shift(torch.cat([h[:, 1:], h[:, :-1], h[:, 1:] * h[:, :-1]], -1)).squeeze(-1)


def doc_inputs(n, domain):
    i = np.arange(n)
    pos_bin = np.minimum((N_BINS * i / n).astype(int), N_BINS - 1)
    extra = np.stack([i / n, (i < 3).astype(float), (n - i <= 3).astype(float)], 1)
    return pos_bin, int(domain == "IT"), extra


def make_batch(items, feats, device, bow_ids=None, tokens=None):
    """items: list of (doc, start, end) windows."""
    t = max(e - s for _, s, e in items)
    b = len(items)
    xs = [torch.zeros(b, t, *f.shape[1:], dtype=f.dtype, device=device) for f in feats]
    pos = torch.zeros(b, t, dtype=torch.long, device=device)
    extra = torch.zeros(b, t, 3, device=device)
    y = torch.full((b, t), -1, dtype=torch.long, device=device)
    valid = torch.zeros(b, t, dtype=torch.bool, device=device)
    dom = torch.zeros(b, dtype=torch.long, device=device)
    for r, (d, s, e) in enumerate(items):
        for x, f in zip(xs, feats):
            x[r, :e - s] = f[d["row"] + s:d["row"] + e]
        p, dm, ex = d["inputs"]
        pos[r, :e - s] = torch.as_tensor(p[s:e], device=device)
        extra[r, :e - s] = torch.as_tensor(ex[s:e], dtype=torch.float, device=device)
        dom[r] = dm
        valid[r, :e - s] = True
        if d["y"] is not None:
            y[r, :e - s] = torch.as_tensor(d["y"][s:e], device=device)
    bow = None
    if bow_ids is not None:
        bags = [bow_ids[d["row"] + j] for d, s, e in items for j in range(s, e)]
        rows = torch.tensor([r for r, (_, s, e) in enumerate(items) for _ in range(s, e)], device=device)
        cols = torch.tensor([j for _, s, e in items for j in range(e - s)], device=device)
        offsets = torch.tensor([0] + [len(b) for b in bags[:-1]], device=device).cumsum(0)
        bow = (torch.cat(bags).to(device), offsets, rows, cols)
    tk = None
    if tokens is not None:
        tok, tok_start = tokens
        sent = torch.tensor([d["row"] + j for d, s, e in items for j in range(s, e)], device=device)
        start, lens = tok_start[sent], tok_start[sent + 1] - tok_start[sent]
        ar = torch.arange(int(lens.max()), device=device)
        mask = ar[None] < lens[:, None]
        g = tok[(start[:, None] + ar[None]).clamp_max(tok.shape[0] - 1)]
        rows = torch.tensor([r for r, (_, s, e) in enumerate(items) for _ in range(s, e)], device=device)
        cols = torch.tensor([j for _, s, e in items for j in range(e - s)], device=device)
        tk = (g, mask, rows, cols)
    return xs, pos, dom, extra, y, bow, tk, valid


def class_weights(train, power, device):
    y = np.concatenate([d["y"] for d in train])
    c = np.bincount(y[y >= 0], minlength=len(rr.LABELS)).astype(float)
    w = (c.max() / np.maximum(c, 1)) ** power
    w[c == 0] = 0
    return torch.tensor(w / w[c > 0].mean(), dtype=torch.float, device=device)


def train_one(train, feats, args, seed, device, bow_ids, bow_vocab, tokens):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_layers, d = feats[0].shape[1:]
    model = SeqModel(n_layers, d, feats, args.hidden, args.dropout, args.crf, bow_vocab, args.bow_dim,
                     tokens[0].shape[1] if tokens is not None else 0, args.heads, args.sent_tf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    cw = class_weights(train, args.cw, device)
    for ep in range(args.epochs):
        model.train()
        items = []
        for d in train:
            n = len(d["sents"])
            for _ in range(max(1, int(np.ceil(n / args.window)))):
                s = int(rng.integers(0, max(1, n - args.window + 1)))
                items.append((d, s, min(n, s + args.window)))
        rng.shuffle(items)
        for b in range(0, len(items), args.batch):
            xs, pos, dom, extra, y, bow, tk, valid = make_batch(items[b:b + args.batch], feats, device, bow_ids, tokens)
            em = model(xs, pos, dom, extra, bow, tk, valid)
            mask = y >= 0
            if model.crf is not None:
                loss = model.crf.nll(em, y, mask)
                if args.cw > 0:
                    loss = loss + F.cross_entropy(em[mask], y[mask], weight=cw)
            else:
                loss = F.cross_entropy(em[mask], y[mask], weight=cw)
            if args.shift > 0:
                m2 = mask[:, 1:] & mask[:, :-1]
                target = (y[:, 1:] != y[:, :-1]).float()
                loss = (1 - args.shift) * loss + args.shift * F.binary_cross_entropy_with_logits(
                    model.shift_logits()[m2], target[m2])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def emissions(model, docs, feats, device, bow_ids, tokens):
    model.eval()
    out = []
    for d in docs:
        xs, pos, dom, extra, _, bow, tk, valid = make_batch([(d, 0, len(d["sents"]))], feats, device, bow_ids, tokens)
        out.append(model(xs, pos, dom, extra, bow, tk, valid)[0].float().cpu().numpy())
    return out


def decode(em, crf_params, suppress):
    em = em.copy()
    em[:, suppress] = -1e4
    if crf_params is None:
        return em.argmax(1)
    return viterbi(em, crf_params["trans"], crf_params["start"], crf_params["end"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pools", default="mean", help="comma list of mean,last")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cw", type=float, default=0.0, help="class weight = (max count / count) ** cw")
    ap.add_argument("--crf", action="store_true")
    ap.add_argument("--bow", type=int, default=0, help="vocabulary size of the word / word-pair feature (0 = off)")
    ap.add_argument("--bow_dim", type=int, default=128)
    ap.add_argument("--tokens", default="", help="embed.py output with per-token states: adds attention pooling")
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--sent_tf", type=int, default=0, help="transformer layers over the sentence sequence")
    ap.add_argument("--pseudo", default="", help="comma list of run dirs whose test outputs give pseudo-labels")
    ap.add_argument("--pseudo_thr", type=float, default=0.0)
    ap.add_argument("--pseudo_honest", action="store_true",
                    help="CV fold f uses pseudo-labels from the source runs' fold-f models (same seed and split)")
    ap.add_argument("--full_only", action="store_true", help="skip CV; train on all labelled documents for the test set")
    ap.add_argument("--split_seed", type=int, default=0, help="document split used with per-fold vectors")
    ap.add_argument("--shift", type=float, default=0.0, help="weight of the label-shift side task (0 = off)")
    ap.add_argument("--suppress", default="PrecedentOverruled", help="labels never predicted")
    args = ap.parse_args()
    args.pools = args.pools.split(",")
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.out_dir}/args.json", "w"), indent=1)

    # --emb: comma list of embed.py / finetune.py output dirs; a dir containing "{f}" is per fold (vectors from an
    # encoder fine-tuned without fold f), which fixes the document split to --split_seed for every seed.
    sources = args.emb.split(",")
    per_fold = any("{f}" in x for x in sources + [args.tokens])
    names = [f"{os.path.basename(x)}:{p}" for x in sources for p in args.pools]

    def load_feats(f):
        return [torch.from_numpy(np.load(f"{x.format(f=f)}/{p}.npy")).to(device) for x in sources for p in args.pools]

    def load_tokens(f):
        if not args.tokens:
            return None
        # Each saved layer is layer-normed and the layers are averaged once here (the learned layer weights of the
        # sentence vectors came out nearly uniform), which keeps GPU memory at a quarter.
        path = args.tokens.format(f=f)
        raw = np.load(f"{path}/tokens.npy", mmap_mode="r")
        tok = torch.empty(raw.shape[0], 1, raw.shape[2], dtype=torch.float16, device=device)
        for i in range(0, raw.shape[0], 65536):
            x = torch.from_numpy(np.ascontiguousarray(raw[i:i + 65536])).to(device).float()
            tok[i:i + 65536, 0] = F.layer_norm(x, x.shape[-1:]).mean(1).half()
        return tok, torch.from_numpy(np.load(f"{path}/tok_start.npy")).to(device)

    index = json.load(open(f"{sources[0].format(f=0)}/index.json"))["docs"]
    docs, test = rr.load_labelled(), rr.load_test()
    for d, ix in zip(docs + test, index):
        assert d["id"] == ix["id"] and ix["end"] - ix["start"] == len(d["sents"])
        d["row"] = ix["start"]
        d["inputs"] = doc_inputs(len(d["sents"]), d["domain"])
    suppress = [rr.LABEL_ID[l] for l in args.suppress.split(",") if l]
    gold = np.concatenate([d["y"] for d in docs])
    bow_ids, bow_vocab = build_bow(docs + test, args.bow) if args.bow else (None, 0)
    feats, tokens = (None, None) if per_fold else (load_feats(0), load_tokens(0))
    # Self-training: test documents labelled by earlier runs (averaged test outputs) join every training set;
    # sentences whose predicted probability is below --pseudo_thr are left unlabelled (-1).
    def honest_pseudo(seed, f):
        parts = [np.load(f"{r}/test_fold_seed{seed}.npz") for r in args.pseudo.split(",")]
        out = []
        for d in test:
            em = np.mean([log_softmax(p[f"{f}/{d['id']}"]) for p in parts], 0)
            y = em.argmax(1)
            y[np.exp(em.max(1)) < args.pseudo_thr] = -1
            out.append(dict(d, y=y))
        return out

    pseudo = []
    if args.pseudo and not args.pseudo_honest:
        parts = [np.load(f) for r in args.pseudo.split(",") for f in sorted(glob.glob(f"{r}/test_seed*.npz"))]
        for d in test:
            em = np.mean([log_softmax(p[d["id"]]) for p in parts], 0)
            y = em.argmax(1)
            y[np.exp(em.max(1)) < args.pseudo_thr] = -1
            pseudo.append(dict(d, y=y))
        n = sum(len(p["y"]) for p in pseudo)
        print(f"pseudo-labelled test sentences: {sum((p['y'] >= 0).sum() for p in pseudo)}/{n}", flush=True)

    log = open(f"{args.out_dir}/log.txt", "a")
    if args.full_only:
        # Final models: same settings, trained on all labelled documents; only their test outputs are saved.
        for seed in map(int, args.seeds.split(",")):
            model = train_one(docs, feats, args, seed * 100 + 99, device, bow_ids, bow_vocab, tokens)
            np.savez_compressed(f"{args.out_dir}/test_full_seed{seed}.npz",
                                **{d["id"]: em for d, em in zip(test, emissions(model, test, feats, device, bow_ids, tokens))})
            print("full model seed", seed, "done", flush=True)
        return
    for seed in map(int, args.seeds.split(",")):
        t0 = time.time()
        fold = rr.doc_folds(docs, args.k, args.split_seed if per_fold else seed)
        held = [None] * len(docs)
        test_em = [np.zeros((len(d["sents"]), len(rr.LABELS))) for d in test]
        test_fold = {}
        crfs, mixes = [], []
        for f in range(args.k):
            tr = [d for d, g in zip(docs, fold) if g != f]
            tr += honest_pseudo(seed, f) if args.pseudo_honest else pseudo
            te_idx = [i for i, g in enumerate(fold) if g == f]
            if per_fold:
                feats = tokens = None
                torch.cuda.empty_cache()
                feats, tokens = load_feats(f), load_tokens(f)
            model = train_one(tr, feats, args, seed * 100 + f, device, bow_ids, bow_vocab, tokens)
            mixes.append([w.detach().softmax(0).cpu().numpy() for w in model.mix])
            cp = model.crf.params() if model.crf is not None else None
            crfs.append(cp)
            for i, em in zip(te_idx, emissions(model, [docs[i] for i in te_idx], feats, device, bow_ids, tokens)):
                held[i] = (em, cp)
            for j, em in enumerate(emissions(model, test, feats, device, bow_ids, tokens)):
                test_em[j] += em / args.k
                test_fold[f"{f}/{test[j]['id']}"] = em
        pred = np.concatenate([decode(em, cp, suppress) for em, cp in held])
        s = rr.scores(gold, pred)
        msg = rr.report(f"seed {seed}", s)
        for dom in rr.DOMAINS:
            m = np.concatenate([[d["domain"] == dom] * len(d["sents"]) for d in docs])
            sd = rr.scores(gold[m], pred[m])
            msg += f"\n   {dom}: all13 {100 * sd['all13']:.2f} present {100 * sd['present']:.2f}"
        for p, w in zip(names, np.mean(mixes, 0)):
            msg += f"\n   layer mix {p}: " + " ".join(f"{100 * x:.0f}" for x in w)
        msg += f"\n   ({time.time() - t0:.0f}s)"
        print(msg, flush=True)
        print(msg, file=log, flush=True)
        np.savez_compressed(f"{args.out_dir}/heldout_seed{seed}.npz", fold=np.array(fold),
                            **{f"em_{i}": em for i, (em, _) in enumerate(held)})
        crf_mean = None
        if args.crf:
            crf_mean = {k: np.mean([c[k] for c in crfs], 0) for k in crfs[0]}
            json.dump({k: v.tolist() for k, v in crf_mean.items()}, open(f"{args.out_dir}/crf_seed{seed}.json", "w"))
        np.savez_compressed(f"{args.out_dir}/test_seed{seed}.npz", **{d["id"]: em for d, em in zip(test, test_em)})
        np.savez_compressed(f"{args.out_dir}/test_fold_seed{seed}.npz", **test_fold)
        json.dump({"all13": s["all13"], "present": s["present"], "acc": s["acc"], "per_label": s["per_label"]},
                  open(f"{args.out_dir}/scores_seed{seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
