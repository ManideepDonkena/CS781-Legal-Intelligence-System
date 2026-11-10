"""Phase 6 data, sentence splitting and ROUGE, written from scratch (standard library + NumPy only).

Data: data/hf/Phase_6_Summ/Train/train.json (7,030 cases with reference summaries) and Test/test.json (100 cases,
no summaries).  There is no official dev file, so a fixed random 400 training cases (seed 0) are held out as "dev";
`load("train")` returns the rest, `load("all")` everything (for the final models).  Cases with an empty case text
or an empty summary are dropped from training.

ROUGE (Lin, 2004) as in the usual `rouge_score` package: lowercase, non-alphanumerics become spaces, tokens are
the remaining runs; ROUGE-n F1 from clipped n-gram counts; ROUGE-L F1 from the longest common subsequence over the
whole text (not split into sentences).  No stemming, so the absolute numbers differ a little from a scorer that
stems; we only use them to compare our own variants.  The LCS uses the bit-parallel method of Allison and Dix
(1986) / Hyyro (2004) on Python integers, which is exact and fast enough for 1,000-word texts.
"""
import json
import random
import re
from collections import Counter

ROOT = "/home/mani/cs781/Capstone/Phase6_SUM/data/hf/Phase_6_Summ"
N_DEV = 400


def load(split):
    if split == "test":
        return json.load(open(f"{ROOT}/Test/test.json"))
    docs = [d for d in json.load(open(f"{ROOT}/Train/train.json"))
            if len(d["document"].split()) >= 50 and d["summary"].strip()]
    ids = sorted(d["id"] for d in docs)
    dev = set(random.Random(0).sample(ids, N_DEV))
    if split == "all":
        return docs
    return [d for d in docs if (d["id"] in dev) == (split == "dev")]


# ----------------------------------------------------------------------------------------------- sentences
# A full stop ends a sentence unless the word before it is a known abbreviation, a single letter (initials such
# as "A.K."), or a short number (list items "1.", page numbers), and the next word must start like a sentence.
ABBREV = {"no", "nos", "v", "vs", "s", "ss", "sec", "secs", "art", "arts", "mr", "mrs", "ms", "dr", "rs", "j", "jj",
          "cj", "co", "ltd", "pvt", "corpn", "govt", "dt", "etc", "viz", "i.e", "e.g", "cl", "para", "paras", "r",
          "rr", "o", "ors", "anr", "st", "sh", "smt", "hon", "ld", "sr", "jr", "u", "p", "m", "cr", "crl", "c", "w",
          "ch", "vol", "pp", "ibid", "supra", "id", "ex", "approx", "dy", "addl", "asst", "inc", "bros", "nd",
          "spl", "misc", "wp", "slp", "air", "scr", "scc", "lr", "ilr", "all", "bom", "cal", "mad", "pat", "ker",
          "guj", "raj", "del", "ori", "mp", "ap", "hp", "jk", "pb", "hry", "mys", "assam", "cri", "lj", "ca", "rev"}
_CAND = re.compile(r"[.?!][\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def sentences(text):
    """Split a case or summary into sentences (list of strings, whitespace collapsed)."""
    text = " ".join(text.split())
    out, start = [], 0
    for m in _CAND.finditer(text):
        end = m.start() + 1
        before = text[start:end - 1].rsplit(" ", 1)[-1].lower().strip("(\"'")
        if text[m.start()] == "." and (before in ABBREV or len(before) <= 1 or "." in before
                                       or (before.isdigit() and len(before) <= 3)):
            continue
        piece = text[start:m.end()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


# ----------------------------------------------------------------------------------------------- ROUGE
_TOK = re.compile(r"[a-z0-9]+")


def tokens(text):
    return _TOK.findall(text.lower())


def _f1(overlap, n_pred, n_ref):
    if overlap == 0:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_n(pred_toks, ref_toks, n):
    p = Counter(tuple(pred_toks[i:i + n]) for i in range(len(pred_toks) - n + 1))
    r = Counter(tuple(ref_toks[i:i + n]) for i in range(len(ref_toks) - n + 1))
    return _f1(sum((p & r).values()), sum(p.values()), sum(r.values()))


def lcs(a, b):
    """Length of the longest common subsequence of two token lists (bit-parallel, exact)."""
    if not a or not b:
        return 0
    masks = {}
    for i, t in enumerate(b):
        masks[t] = masks.get(t, 0) | (1 << i)
    full = (1 << len(b)) - 1
    s = 0
    for t in a:
        x = masks.get(t, 0) | s
        s = x & ((x - ((s << 1) | 1)) ^ x) & full
    return bin(s).count("1")


def rouge_l(pred_toks, ref_toks):
    return _f1(lcs(pred_toks, ref_toks), len(pred_toks), len(ref_toks))


def scores(pred, ref):
    """ROUGE-1, ROUGE-2 and ROUGE-L F1 (x100) for one prediction/reference pair of strings."""
    p, r = tokens(pred), tokens(ref)
    return {"r1": 100 * rouge_n(p, r, 1), "r2": 100 * rouge_n(p, r, 2), "rl": 100 * rouge_l(p, r)}


def _score_pair(pr):
    return scores(*pr)


def mean_scores(preds, refs, workers=32):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        rows = pool.map(_score_pair, list(zip(preds, refs)), chunksize=4)
    return {k: sum(r[k] for r in rows) / len(rows) for k in ("r1", "r2", "rl")}


if __name__ == "__main__":
    # self-check of the bit-parallel LCS against the textbook dynamic programme
    rng = random.Random(1)
    for _ in range(300):
        a = [rng.randrange(6) for _ in range(rng.randrange(0, 40))]
        b = [rng.randrange(6) for _ in range(rng.randrange(0, 40))]
        t = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                t[i + 1][j + 1] = t[i][j] + 1 if a[i] == b[j] else max(t[i][j + 1], t[i + 1][j])
        assert lcs(a, b) == t[-1][-1]
    print("lcs ok")
    tr, dv, te = load("train"), load("dev"), load("test")
    print(len(tr), len(dv), len(te))
    s = sentences(dv[0]["document"])
    print(len(s), "sentences; first five:")
    for x in s[:5]:
        print("  |", x[:160])
