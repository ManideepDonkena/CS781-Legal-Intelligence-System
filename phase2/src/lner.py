"""Phase 2 L-NER library: data alignment, Phase 1 backbone wrappers, decoding, post-processing, metric.

The gold annotation of this dataset is a per-document dictionary {entity string -> label} applied to
every case-sensitive substring occurrence (longest string first).  `propagate` reproduces that process
on model output; profile_data.py prints the evidence.
"""
import collections
import json
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PIECE_RE = re.compile(r"\s*(?:\w+|[^\w\s])|\s+")


# ----------------------------------------------------------------------------- data

def load_fold(data_dir, fold):
    sub = "Test" if fold == "fold3" else "Train"
    with open(f"{data_dir}/{sub}/{fold}-text.json", encoding="utf-8") as f:
        text = json.load(f)
    gold = None
    if sub == "Train":
        with open(f"{data_dir}/{sub}/{fold}-gold.json", encoding="utf-8") as f:
            gold = {d: [(e[0], e[1], e[2]) for e in ents] for d, ents in json.load(f).items()}
    return text, gold


def load_labels(data_dir):
    with open(f"{data_dir}/Train/labels.txt", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def tokenize(tok, text):
    """BPE-encode `text` with every punctuation character isolated, so that token boundaries exist
    wherever an annotated span can end.  Returns (ids, char offsets)."""
    pieces = [(m.start(), m.group()) for m in PIECE_RE.finditer(text)]
    ids, offs = [], []
    for (start, _), enc in zip(pieces, tok.encode_batch([p for _, p in pieces], add_special_tokens=False)):
        for i, (a, b) in zip(enc.ids, enc.offsets):
            ids.append(i)
            offs.append((start + a, start + b))
    return ids, offs


TOKEN_RE = re.compile(r"[a-z]{3,}")
VS_STRONG = re.compile(r"\b(?:versus|vs\.?|v/s\.?)(?=\W)", re.I)
VS_WEAK = re.compile(r"\sv\.\s")  # lower case only: "SUJATA V. MANOHAR" is a judge's initial
HEAD = 4000  # characters searched for the cause title


def find_versus(text):
    """The "versus" that separates the two parties in the cause title (a regex match or None)."""
    head = text[:HEAD]
    return VS_STRONG.search(head) or VS_WEAK.search(head)


# Cause title: "<appellant> ...Appellant(s) Versus <respondent> ...Respondent(s)" or
# "PETITIONER: <appellant> Vs. RESPONDENT: <respondent> DATE OF JUDGMENT".
ZONE_START = re.compile(r"\d{4}\s*[\])]?|\d\s*/\s*\d{2,4}|JURISDICTION|"
                        r"(?:PETITIONER|APPELLANT|APPLICANT|PLAINTIFF)S?\s*[:\-]+|[\])]\s", re.I)
ZONE_END = re.compile(r"RESPONDENT|DEFENDANT|OPPOSITE PART|DATE OF JUDGMENT|J\s?U\s?D\s?G\s?M\s?E\s?N\s?T|\bORDER\b|"
                      r"-{3,}|[.\u2026]{2,}|\u2026|\bMr\.|\bShri\b|\bBENCH\b|\bCORAM\b|\bFor the\b|\bPresent\b", re.I)
ZONE_LEAD = re.compile(r"\s*(?:RESPONDENTS?\s*[:\-]*\s*)?(?:\d+\s*[\].)]\s*)?", re.I)
ROLE_TAIL = re.compile(r"[\s.\u2026:\-]*(?:APPELLANT|PETITIONER|APPLICANT|PLAINTIFF|COMPLAINANT)S?\s*(?:\(\s*S\s*\))?"
                       r"[\s.\u2026:\-]*$", re.I)
NO_VS = re.compile(r"PETITIONER\s*:(.{3,200}?)RESPONDENT\s*:(.{3,200}?)(?:DATE OF JUDGMENT|BENCH|CITATION|$)", re.S)
ZONE_STOP = {"the", "and", "ors", "anr", "others", "another", "for", "through", "its"}


def party_zones(text):
    """Text of the appellant side and of the respondent side of the cause title ('' when not found)."""
    m = find_versus(text)
    if not m:
        m2 = NO_VS.search(text[:1500])
        return (m2.group(1), m2.group(2)) if m2 else ("", "")
    before = ROLE_TAIL.sub("", text[max(0, m.start() - 200):m.start()])  # drop "...APPELLANT(S)" first
    starts = [x.end() for x in ZONE_START.finditer(before)]
    app = before[starts[-1]:] if starts else before[-120:]
    after = text[m.end():m.end() + 260]
    lead = ZONE_LEAD.match(after).end()
    end = ZONE_END.search(after, lead + 2)
    return app, after[lead:end.start() if end else lead + 120]


def zone_tokens(s):
    return frozenset(w for w in TOKEN_RE.findall(s.lower()) if w not in ZONE_STOP)


N_SHAPES, N_REL, N_LOG, N_ZONE = 9, 20, 16, 4
FEATURE_SIZES = (N_SHAPES, N_REL, N_LOG, N_LOG, N_ZONE)


def token_features(text, offs):
    """Per-token [shape, relative position, log distance from start, log distance from end, side].
    `side` says whether the token's word occurs on the appellant side of the cause title (1), on the
    respondent side (2), on both (3) or on neither (0).  A window of the text does not contain the
    cause title, so this is how the tagger can tell that a name in the body is a party."""
    n = len(offs)
    zone_app, zone_resp = (zone_tokens(z) for z in party_zones(text))
    side_of_char = {}
    if zone_app or zone_resp:
        for m in re.finditer(r"[A-Za-z]{3,}", text):
            w = m.group().lower()
            side = (w in zone_app) + 2 * (w in zone_resp)
            if side:
                for c in range(m.start(), m.end()):
                    side_of_char[c] = side
    feats = []
    for i, (a, b) in enumerate(offs):
        w = text[a:b]
        if not w.strip():
            shape = 0
        elif w.isalpha():
            shape = 1 if w.islower() else 4 if len(w) == 1 else 3 if w.isupper() else 2 if w[0].isupper() and w[1:].islower() else 6
        elif w.isdigit():
            shape = 5
        elif w.isalnum():
            shape = 6
        elif not any(c.isalnum() for c in w):
            shape = 7
        else:
            shape = 8
        side = next((side_of_char[c] for c in range(a, b) if c in side_of_char), 0)
        feats.append((shape, min(N_REL - 1, N_REL * i // n), min(N_LOG - 1, (i + 1).bit_length()),
                      min(N_LOG - 1, (n - i).bit_length()), side))
    return feats


WORD_RE = re.compile(r"\w+")


def document_memory(text, offs, type_probs):
    """Second-pass features: for every token, the mean and max of a first-pass tagger's label
    probabilities over all tokens of the same (case-folded) word anywhere in the document, plus a
    log-count.  Lets a mention inherit evidence from other mentions of the same name, e.g. the
    party line in the header."""
    word_of_char = {}
    for m in WORD_RE.finditer(text):
        key = m.group().lower()
        for c in range(m.start(), m.end()):
            word_of_char[c] = key
    keys = [word_of_char.get(a) if b > a else None for a, b in offs]
    groups = collections.defaultdict(list)
    for i, k in enumerate(keys):
        if k is not None:
            groups[k].append(i)
    probs = torch.as_tensor(type_probs, dtype=torch.float)
    out = torch.zeros(len(offs), 2 * probs.shape[1] + 1)
    out[:, 0] = 1.0  # tokens without a word key look like "always O"
    for k, idx in groups.items():
        p = probs[idx]
        out[idx] = torch.cat([p.mean(0), p.max(0).values, torch.tensor([min(len(idx), 64) ** 0.5 / 8])])
    return out


def mention_pool(docs):
    """Label -> entity strings seen in the training documents (one entry per document and string)."""
    pool = collections.defaultdict(list)
    for d in docs:
        for st, lab in sorted({(d["text"][s:e], lab) for s, e, lab in d["gold"]}):
            pool[lab].append(st)
    return pool


def swap_mentions(text, spans, pool, rng, p_swap):
    """Augmentation: replace a document's entity strings by other training strings of the same label.
    A string is replaced consistently at all of its mentions, which keeps the annotation a document
    dictionary.  Returns the new text and its spans."""
    mapping, out, new_spans, pos, cur = {}, [], [], 0, 0
    for s, e, lab in sorted(spans):
        if s < pos:
            continue
        key = (text[s:e], lab)
        if key not in mapping:
            mapping[key] = rng.choice(pool[lab]) if rng.random() < p_swap else text[s:e]
        rep = mapping[key]
        out.append(text[pos:s])
        cur += s - pos
        out.append(rep)
        new_spans.append((cur, cur + len(rep), lab))
        cur += len(rep)
        pos = e
    out.append(text[pos:])
    return "".join(out), new_spans


def bio_tags(labels):
    tags = ["O"]
    for l in labels:
        tags += [f"B-{l}", f"I-{l}"]
    return tags


def spans_to_tags(offs, spans, tag2id):
    y = [0] * len(offs)
    starts = [a for a, _ in offs]
    import bisect
    for s, e, lab in spans:
        i = bisect.bisect_left(starts, s)
        while i > 0 and offs[i - 1][1] > s:
            i -= 1
        first = True
        while i < len(offs) and offs[i][0] < e:
            a, b = offs[i]
            if b > a and b > s:
                y[i] = tag2id[("B-" if first else "I-") + lab]
                first = False
            i += 1
    return y


def tags_to_spans(text, offs, tag_ids, conf, id2tag):
    """Greedy BIO decoding to character spans (whitespace-trimmed) with a mean-confidence score."""
    out, cur = [], None

    def close():
        nonlocal cur
        if cur is not None:
            lab, i0, i1, sc = cur
            a, b = offs[i0][0], offs[i1][1]
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            if b > a:
                out.append((a, b, lab, sc / (i1 - i0 + 1)))
        cur = None

    for i, t in enumerate(tag_ids):
        tag = id2tag[t]
        if tag == "O" or offs[i][1] <= offs[i][0]:
            if tag == "O":
                close()
            continue
        kind, lab = tag[0], tag[2:]
        if kind == "I" and cur is not None and cur[0] == lab:
            cur = (lab, cur[1], i, cur[3] + conf[i])
        else:
            close()
            cur = (lab, i, i, conf[i])
    close()
    return out


# ----------------------------------------------------------------------------- post-processing

def snap_bounds(text, s, e, keep_suffixes=("s", "al")):
    """Extend [s, e) to the edges of the letter (or digit) run it starts or ends in.  A span that
    stops just before a plural/adjectival suffix is left alone: the gold dictionary is applied by
    substring match, so `High Court` inside `High Courts` is a legitimate annotation."""
    def same(a, b):
        return (a.isalpha() and b.isalpha()) or (a.isdigit() and b.isdigit())

    while s > 0 and same(text[s - 1], text[s]):
        s -= 1
    j = e
    while j < len(text) and same(text[j - 1], text[j]):
        j += 1
    return s, e if text[e:j] in keep_suffixes else j


def snap_to_words(text, spans, keep_suffixes=("s", "al")):
    """Snap every span to word boundaries (the tagger works on sub-word tokens and sometimes labels
    only part of a word); fragments of one word that now coincide are merged."""
    merged = {}
    for s, e, lab, sc in spans:
        s, e = snap_bounds(text, s, e, keep_suffixes)
        if (s, e) not in merged or sc > merged[(s, e)][1]:
            merged[(s, e)] = (lab, sc)
    return [(s, e, lab, sc) for (s, e), (lab, sc) in sorted(merged.items())]


def _occurrences(text, s):
    i = text.find(s)
    while i >= 0:
        yield i
        i = text.find(s, i + 1)


def propagate(text, spans, tau=0.5, min_len=2):
    """Turn mention-level predictions into a document dictionary and re-apply it everywhere.

    A string is kept if the tagger marked at least `tau` of its free occurrences (those not inside a
    longer predicted span); its label is the confidence-weighted vote.  Kept strings are then placed
    longest-first on every case-sensitive substring occurrence, as in the gold annotation."""
    votes = collections.defaultdict(lambda: collections.defaultdict(float))
    exact = collections.defaultdict(set)
    cover = [0] * (len(text) + 1)
    for s, e, lab, sc in spans:
        votes[text[s:e]][lab] += sc
        exact[text[s:e]].add(s)
        for k in range(s, e):
            if e - s > cover[k]:
                cover[k] = e - s
    kept = []
    for st, labs in votes.items():
        if len(st) < min_len:
            continue
        n_exact = n_free = 0
        occ = list(_occurrences(text, st))
        for a in occ:
            if a in exact[st]:
                n_exact += 1
            elif min(cover[a:a + len(st)]) <= len(st):
                n_free += 1
        if n_exact / max(1, n_exact + n_free) >= tau:
            kept.append((st, max(labs, key=labs.get), occ))
    return apply_dictionary(text, [(st, lab) for st, lab, _ in kept])


def apply_dictionary(text, entries):
    """Place {string: label} entries on every case-sensitive occurrence, longest string first."""
    taken = bytearray(len(text))
    out = []
    for st, lab in sorted(entries, key=lambda x: -len(x[0])):
        n = len(st)
        for a in _occurrences(text, st):
            if not any(taken[a:a + n]):
                out.append((a, a + n, lab))
                taken[a:a + n] = b"\x01" * n
    return sorted(out)


# ----------------------------------------------------------------------------- metric

def strict_scores(gold, pred, labels):
    """Mirror of evaluation/evaluate_phase2.py: exact (start, end, label) match, macro over labels."""
    stats = {l: [0, 0, 0] for l in labels}
    for did in gold:
        g = {(s, e, l) for s, e, l in gold[did]}
        p = {(x[0], x[1], x[2]) for x in pred.get(did, [])}
        for l in labels:
            gs = {x for x in g if x[2] == l}
            ps = {x for x in p if x[2] == l}
            stats[l][0] += len(gs & ps)
            stats[l][1] += len(ps - gs)
            stats[l][2] += len(gs - ps)
    per = {}
    for l in labels:
        tp, fp, fn = stats[l]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[l] = {"p": pr, "r": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0}
    n = len(labels)
    return {"macro_f1": sum(v["f1"] for v in per.values()) / n,
            "macro_p": sum(v["p"] for v in per.values()) / n,
            "macro_r": sum(v["r"] for v in per.values()) / n,
            "per_label": per}


# ----------------------------------------------------------------------------- model

def load_backbone(run_dir, repo):
    """Load a Phase 1 checkpoint (either architecture family) and its tokenizer."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from model_zoo import build_model
    from tokenizers import Tokenizer
    ck = torch.load(f"{run_dir}/best.pt", map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    family = cfg.pop("family")
    cfg["gradient_checkpointing"] = False
    lm = build_model(family, **cfg)
    lm.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    return lm, Tokenizer.from_file(f"{run_dir}/tokenizer.json"), family


class CRF(nn.Module):
    """Linear-chain CRF with BIO constraints (an I- tag may only follow B-/I- of the same label)."""

    def __init__(self, tags):
        super().__init__()
        n = len(tags)
        self.trans = nn.Parameter(torch.zeros(n, n))  # trans[i, j]: score of tag i followed by tag j
        illegal = torch.zeros(n, n, dtype=torch.bool)
        for i, a in enumerate(tags):
            for j, b in enumerate(tags):
                illegal[i, j] = b.startswith("I-") and (a == "O" or a[2:] != b[2:])
        self.register_buffer("illegal", illegal)
        self.register_buffer("bad_start", torch.tensor([t.startswith("I-") for t in tags]))

    def scores(self):
        return self.trans.masked_fill(self.illegal, -1e4)

    def nll(self, em, y, mask):
        """Mean per-token negative log-likelihood.  Windows may begin inside an entity, so the first
        tag is unconstrained here; the start constraint is applied when decoding whole documents."""
        tr = self.scores()
        yy = y.clamp_min(0)
        gold = em[:, 0].gather(1, yy[:, :1]).squeeze(1)
        alpha = em[:, 0]
        for t in range(1, em.shape[1]):
            m = mask[:, t]
            gold = gold + (tr[yy[:, t - 1], yy[:, t]] + em[:, t].gather(1, yy[:, t:t + 1]).squeeze(1)) * m
            nxt = torch.logsumexp(alpha.unsqueeze(2) + tr, dim=1) + em[:, t]
            alpha = torch.where(m.unsqueeze(1), nxt, alpha)
        return (torch.logsumexp(alpha, dim=1) - gold).sum() / mask.sum()

    @torch.no_grad()
    def viterbi(self, em):
        """Best tag sequence for one document; em is (T, K) on any device."""
        tr = self.scores().float().cpu().numpy()
        em = em.float().cpu().numpy()
        cols = np.arange(em.shape[1])
        score = em[0] + np.where(self.bad_start.cpu().numpy(), -1e4, 0.0)
        back = np.zeros(em.shape, dtype=np.int32)
        for t in range(1, em.shape[0]):
            cand = score[:, None] + tr
            back[t] = cand.argmax(0)
            score = cand[back[t], cols] + em[t]
        path = [int(score.argmax())]
        for t in range(em.shape[0] - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]


class Tagger(nn.Module):
    """Token classifier over a Phase 1 language model.

    attention: 'causal' keeps the pre-training mask; 'bidir' lets every token see the whole window.
    head: 'linear' or 'bilstm' (adds right context independently of the backbone's attention).
    """

    def __init__(self, lm, family, n_tags, attention="causal", head="linear", dropout=0.1, lstm_hidden=384,
                 layer_mix=False, feats=False, frozen=False, tags=None, memory_dim=0):
        super().__init__()
        self.lm, self.family, self.attention = lm, family, attention
        self.modern = family in ("modern_transformer", "yarn_gqa")
        self.frozen = frozen
        d = lm.token.weight.shape[1]
        for m in lm.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout
        self.drop = nn.Dropout(dropout)
        n_layers = len(lm.layers) if self.modern else lm.n_layers
        self.mix = nn.Parameter(torch.zeros(n_layers + 1)) if layer_mix else None
        self.feat_emb = None
        if feats:  # True / "all": every token feature; "side": only the cause-title side
            self.feat_idx = [len(FEATURE_SIZES) - 1] if feats == "side" else list(range(len(FEATURE_SIZES)))
            self.feat_emb = nn.ModuleList(nn.Embedding(FEATURE_SIZES[i], 16) for i in self.feat_idx)
            d += 16 * len(self.feat_emb)
        # Document memory: per-token summary of a first-pass tagger's predictions for the same word
        # elsewhere in the document (see train_ner.py --memory).
        self.memory_proj = nn.Linear(memory_dim, 64) if memory_dim else None
        d += 64 if memory_dim else 0
        self.crf = CRF(tags) if tags is not None else None
        self.lstm = None
        if head == "bilstm":
            self.lstm = nn.LSTM(d, lstm_hidden, num_layers=2, batch_first=True, bidirectional=True,
                                dropout=dropout)
            d = 2 * lstm_hidden
        self.out = nn.Linear(d, n_tags)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("lm.")]

    def states(self, ids, keep):
        """Residual-stream output of the embedding and of every layer."""
        full = keep[:, None, None, :] if self.attention == "bidir" else None
        lm = self.lm
        b, t = ids.shape
        states = []
        if self.modern:
            # Called exactly as ModernCausalLM.forward does, so positions match pre-training.
            cos, sin = lm.rotary(ids, t, offset=0)
            x = lm.token(ids)
            states.append(x)
            for layer in lm.layers:
                h, _ = layer.attn(layer.attn_norm(x), cos, sin, kv_cache=None, attn_mask=full)
                x = x + layer.drop(h)
                x = x + layer.drop(layer.ffn(layer.ffn_norm(x)))
                states.append(x)
            return states
        x = lm.token(ids) + lm.position(torch.arange(t, device=ids.device))[None]
        states.append(x)
        for i in range(lm.n_layers):
            blk = lm.core[i % len(lm.core)]
            a = blk.attn
            q, k, v = a.qkv(blk.norm1(x)).view(b, t, 3, a.heads, a.head_dim).unbind(2)
            q, k, v = (z.transpose(1, 2) for z in (q, k, v))
            if full is None:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=full)
            x = x + blk.drop(a.out(y.transpose(1, 2).contiguous().view(b, t, -1)))
            x = x + blk.drop(blk.ff(blk.norm2(x)))
            states.append(x)
        return states

    def pool(self, states):
        """Final hidden state, or (with layer_mix) a learned mixture of all layers."""
        if self.mix is None:
            return self.lm.norm(states[-1]).float()
        w = self.mix.softmax(0)
        return sum(wi * F.layer_norm(h.float(), h.shape[-1:]) for wi, h in zip(w, states))

    def forward(self, ids, keep, feats=None, memory=None):
        if self.frozen:
            with torch.no_grad():
                states = self.states(ids, keep)
        else:
            states = self.states(ids, keep)
        h = self.pool(states)
        if self.feat_emb is not None:
            h = torch.cat([h] + [emb(feats[..., i]) for i, emb in zip(self.feat_idx, self.feat_emb)], dim=-1)
        h = self.drop(h)
        if self.memory_proj is not None:
            h = torch.cat([h, torch.tanh(self.memory_proj(memory))], dim=-1)
        if self.lstm is not None:
            h, _ = self.lstm(h)
            h = self.drop(h)
        return self.out(h)
