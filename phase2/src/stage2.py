"""Second stage of the L-NER pipeline: a classifier over document-dictionary entries.

The gold annotation is a per-document {string -> label} dictionary applied to every occurrence, so
the unit that matters is the string, not the mention.  Candidates are the strings the first-stage
taggers marked at least once (held-out predictions for the labelled folds).  Each candidate gets
features that pool evidence over the whole document: how often and how consistently the taggers
marked it, where it first appears, what words surround its occurrences, and how the same string was
labelled in other training documents.  A gradient-boosted classifier then keeps or drops the string
and picks its label; the kept entries are placed longest-first, as in the gold annotation.

A config names a set of cross-validation runs (R/<config>_fold1, _fold2 or R/<config>_k0 ... _k4).
Cross-validate:  stage2.py --data D --runs R --configs a b c
Test:            stage2.py --data D --runs R --configs a b c --test_spans x.json y.json --out_dir S
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import re
import zipfile

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import lner
from lner import TOKEN_RE, find_versus, party_zones, zone_tokens
from make_submission import validate

CUES = ["appellant", "petitioner", "respondent", "applicant", "complainant", "defendant", "plaintiff", "accused",
        "versus", "vs", "v.", "for the", "counsel", "advocate", "adv.", "mr.", "ms.", "mrs.", "shri", "sri", "smt",
        "dr.", "justice", "hon'ble", "coram", "bench", "before", "j.", "jj.", "pw", "p.w", "dw", "witness",
        "deposed", "examined", "dated", "on ", "section", "article", "under", "act", "code", "rules", "in ", "case",
        "judgment", "reported", "air", "scc", "no.", "appeal", "petition", "writ", "court", "state", "union",
        "government", "through", "ors", "another", "(", ")", ",", ":", "of ", "and ", "the "]
WINDOW, MAX_OCC = 50, 40
# Cues that fix a party's or witness's role; a name inherits them from related strings (see `relatives`).
ROLE_CUES = ["appellant", "petitioner", "respondent", "applicant", "complainant", "defendant", "plaintiff",
             "accused", "versus", "vs", "pw", "p.w", "witness", "counsel", "advocate", "for the"]
# Sharp local patterns, matched in the 24 characters before / after an occurrence (or inside the string).
ADJ_PRE = {"wit": r"\b[PDC]\.?\s?W\.?\s?-?\s?\d+\W{0,4}$", "accused": r"(?:\bA-?\d+|accused)\W{0,4}$",
           "justice": r"(?:Justice|Hon'ble)\W{0,3}(?:(?:Mr|Ms|Mrs|Dr)\.?\W{0,2})?$",
           "honorific": r"\b(?:Mr|Ms|Mrs|Shri|Sri|Smt|Dr|Sh)\.?\s*$", "in_case": r"\b(?:in|In|see|See|of|and)\s+$",
           "role_app": r"(?:appellant|petitioner|applicant)s?\W{0,3}(?:No\.?\s*\d+\W{0,3})?$",
           "role_resp": r"(?:respondent|defendant)s?\W{0,3}(?:No\.?\s*\d+\W{0,3})?$"}
ADJ_POST = {"wit": r"^\W{0,4}[PDC]\.?\s?W\.?\s?-?\s?\d+", "accused": r"^\W{0,4}(?:A-?\d+\b|accused)",
            "judge": r"^\s*,?\s*(?:JJ?\.|CJI|C\.J\.)", "advocate": r"^\W{0,3}(?:Sr\.|Senior|Adv|Advocate|learned|Ld\.)",
            "citation": r"^\W{0,4}(?:\(\d{4}\)|\[\d{4}\]|\d{4}\s*\(|AIR\b|reported)",
            "role_app": r"^\W{0,6}(?:appellant|petitioner|applicant)", "role_resp": r"^\W{0,6}(?:respondent|defendant)"}
ADJ_PRE = {k: re.compile(v, re.I if k in ("accused", "justice", "role_app", "role_resp") else 0)
           for k, v in ADJ_PRE.items()}
ADJ_POST = {k: re.compile(v, re.I if k in ("accused", "role_app", "role_resp", "advocate") else 0) for k, v in ADJ_POST.items()}
INSIDE = {"versus": re.compile(r"\sv(?:s|ersus)?\.?\s", re.I), "number": re.compile(r"No\.?\s*\d|\d\s*(?:of|/)\s*\d{4}", re.I),
          "year": re.compile(r"\b(?:1[89]|20)\d{2}\b"), "court": re.compile(r"court|tribunal|bench", re.I),
          "body": re.compile(r"commission|board|authority|government|department|officer|police|corporation", re.I),
          "law": re.compile(r"\b(?:act|code|rules|constitution|regulations|ordinance)\b", re.I)}
ADJ = 24
# Which side a counsel appears for: "A and B, for the appellant.  C for respondent No. 1." (cue after
# the names) or "Petitioner Through: Mr. A ... Respondent Through: Mr. B" (marker before them).
SIDE_AFTER = {"app": re.compile(r"for\s+(?:the\s+)?(?:appellant|petitioner|applicant|complainant|plaintiff)", re.I),
              "resp": re.compile(r"for\s+(?:the\s+)?(?:respondent|defendant|opposite)", re.I)}
SIDE_BEFORE = {"app": re.compile(r"(?:appellant|petitioner|applicant|complainant|plaintiff)s?(?:\s*\(s\))?"
                                 r"\s*(?:through|by|:)|for\s+(?:the\s+)?(?:appellant|petitioner|applicant)", re.I),
               "resp": re.compile(r"(?:respondent|defendant)s?(?:\s*\(s\))?\s*(?:through|by|:)|"
                                  r"for\s+(?:the\s+)?(?:respondent|defendant)", re.I)}
SIDE_SPAN = 400


def side_distances(text, start, end):
    """Distance (0..1, 1 = none within SIDE_SPAN characters) to the next side cue after [start, end)
    and to the last side marker before it: [next app, next resp, previous app, previous resp]."""
    after, before = text[end:end + SIDE_SPAN], text[max(0, start - SIDE_SPAN):start]
    out = []
    for pat in SIDE_AFTER.values():
        m = pat.search(after)
        out.append(m.start() / SIDE_SPAN if m else 1.0)
    for pat in SIDE_BEFORE.values():
        last = None
        for last in pat.finditer(before):
            pass
        out.append((len(before) - last.end()) / SIDE_SPAN if last else 1.0)
    return out
STOP = {"the", "and", "ors", "anr", "others", "another", "state", "union", "india", "court", "high", "supreme",
        "ltd", "limited", "pvt", "private", "shri", "smt", "sri", "kumar", "singh", "devi", "lal", "ram",
        "government", "act", "section", "appeal", "civil", "criminal", "case", "company", "corporation", "bank",
        "board", "officer", "commissioner", "for", "with", "under"}


def load_heldout(runs_dir, config, texts):
    """Held-out spans of one config for every labelled document, merged over its cross-validation
    runs: <config>_fold1 / _fold2 (two-fold) or <config>_k0 ... _k4 (k-fold)."""
    merged = {}
    for path in sorted(glob.glob(f"{runs_dir}/{config}_*/eval_spans.json")):
        if re.search(r"_(?:fold[12]|k\d+)$", os.path.dirname(path)[len(runs_dir) + 1 + len(config):]):
            with open(path) as f:
                part = json.load(f)
            assert not set(part) & set(merged), f"{config}: a document is held out twice"
            merged.update(part)
    assert set(merged) == set(texts), f"{config}: held-out predictions cover {len(merged)} of {len(texts)} documents"
    return merged


def load_runs(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def candidates(text, runs, did, labels, gaz=None, gaz_min=0):
    """Pool the runs' mentions of one document into per-string statistics.
    Returns {string: stats}; stats["votes"][label] counts (mention, run) pairs.  With `gaz_min`,
    strings that at least that many other training dictionaries contain are added as untagged
    candidates when they occur in the text."""
    n_runs = len(runs)
    mentions = collections.defaultdict(lambda: collections.defaultdict(list))  # (s, e) -> label -> [(run, conf)]
    for r, run in enumerate(runs):
        seen = set()
        for s, e, lab, sc in run[did]:
            s, e = lner.snap_bounds(text, s, e)
            if (s, e, lab) not in seen:  # fragments of one word collapse to the same span
                seen.add((s, e, lab))
                mentions[(s, e)][lab].append((r, sc))
    cover = np.zeros(len(text) + 1, dtype=np.int32)
    for s, e in mentions:
        np.maximum(cover[s:e], e - s, out=cover[s:e])
    cands = {}
    for (s, e), by_lab in mentions.items():
        st = text[s:e]
        if len(st) < 2:
            continue
        c = cands.setdefault(st, {"starts": set(), "votes": collections.Counter(), "conf": collections.Counter(),
                                  "runs": collections.defaultdict(set), "peak": collections.Counter()})
        c["starts"].add(s)
        for lab, hits in by_lab.items():
            c["votes"][lab] += len(hits)
            c["conf"][lab] += sum(sc for _, sc in hits)
            c["runs"][lab].update(r for r, _ in hits)
            c["peak"][lab] = max(c["peak"][lab], len(hits))
    if gaz_min:
        for st, counts in gaz.items():
            if st not in cands and len(st) >= 3 and sum(counts.values()) >= gaz_min and st in text:
                cands[st] = {"starts": set(), "votes": collections.Counter(), "conf": collections.Counter(),
                             "runs": collections.defaultdict(set), "peak": collections.Counter()}
    for st, c in cands.items():
        occ = list(lner._occurrences(text, st))
        c["occ"] = occ
        c["free"] = sum(1 for a in occ if a not in c["starts"] and cover[a:a + len(st)].min() <= len(st))
        c["n_runs"] = n_runs
    return cands


def feature_names(labels):
    names = ["log_occ", "len", "words", "tag_rate", "tau_stat", "vote_rate", "runs_any", "top_share", "second_share",
             "first_pos", "first_tagged_pos", "head_share", "is_upper", "is_title", "has_digit", "digit_frac",
             "starts_cap", "ends_punct", "starts_lower", "inside_longer", "contains_other", "free_share",
             "has_vs", "before_vs", "vs_dist", "near_before_vs", "near_after_vs", "gaz_docs", "gaz_any"]
    for l in labels:
        names += [f"vote_{l}", f"peak_{l}", f"conf_{l}", f"runs_{l}", f"gaz_{l}"]
    names += [f"pre:{c}" for c in CUES] + [f"post:{c}" for c in CUES]
    names += [f"adjpre:{k}" for k in ADJ_PRE] + [f"adjpre_any:{k}" for k in ADJ_PRE]
    names += [f"adjpost:{k}" for k in ADJ_POST] + [f"adjpost_any:{k}" for k in ADJ_POST]
    names += [f"inside:{k}" for k in INSIDE]
    names += [f"side_{w}:{k}" for w in ("first", "min") for k in ("next_app", "next_resp", "prev_app", "prev_resp")]
    names += ["side_next", "side_prev"]
    names += ["zone_app_cov", "zone_resp_cov", "zone_app_rev", "zone_resp_rev", "zone_app_exact", "zone_resp_exact"]
    names += [f"anypre:{c}" for c in CUES] + [f"anypost:{c}" for c in CUES]
    names += ["n_relatives"] + [f"rel:{e}" for e in evidence_names(labels)]
    return names


def evidence_names(labels):
    return ([f"peak_{l}" for l in labels] + [f"anypre:{c}" for c in ROLE_CUES] + [f"anypost:{c}" for c in ROLE_CUES]
            + ["near_before_vs", "near_after_vs", "in_head"])


def features(text, low, st, c, labels, others, gaz, zones):
    """`gaz`: {string: {label: number of other training documents whose dictionary has it}}.
    Returns the feature list and the string's own role evidence, which `relatives` hands on to
    related strings of the same document."""
    occ, n_occ, n_runs, n = c["occ"], len(c["occ"]), c["n_runs"], max(1, len(text))
    votes = c["votes"]
    total = sum(votes.values())
    den = max(1, total)
    ranked = sorted(votes.values(), reverse=True) + [0, 0]
    all_runs = set().union(*c["runs"].values())
    vs = find_versus(text)
    vs_pos = vs.start() if vs else None
    first = occ[0]
    g = gaz.get(st, {})
    f = [math.log1p(n_occ), len(st), len(st.split()), len(c["starts"]) / n_occ,
         len(c["starts"]) / max(1, len(c["starts"]) + c["free"]), total / (n_runs * n_occ), len(all_runs) / n_runs,
         ranked[0] / den, ranked[1] / den, first / n, min(c["starts"], default=n) / n,
         sum(1 for a in occ if a < 0.1 * n) / n_occ, st.isupper(), st.istitle(), any(ch.isdigit() for ch in st),
         sum(ch.isdigit() for ch in st) / len(st), st[0].isupper(), not st[-1].isalnum(), st[0].islower(),
         sum(1 for o in others if len(o) > len(st) and st in o), sum(1 for o in others if len(o) < len(st) and o in st),
         c["free"] / n_occ,
         vs_pos is not None, vs_pos is not None and first < vs_pos,
         0.0 if vs_pos is None else max(-3.0, min(3.0, (first - vs_pos) / 300)),
         vs_pos is not None and any(0 < vs_pos - a <= 300 for a in occ),
         vs_pos is not None and any(0 <= a - vs_pos <= 300 for a in occ),
         math.log1p(sum(g.values())), bool(g)]
    for l in labels:
        v = votes.get(l, 0)
        f += [v / (n_runs * n_occ), c["peak"].get(l, 0) / n_runs, c["conf"].get(l, 0.0) / v if v else 0.0,
              len(c["runs"].get(l, ())) / n_runs, math.log1p(g.get(l, 0))]
    step = max(1, n_occ // MAX_OCC)
    sample = occ[::step][:MAX_OCC]
    pre = [low[max(0, a - WINDOW):a] for a in sample]
    post = [low[a + len(st):a + len(st) + WINDOW] for a in sample]
    f += [sum(cue in w for w in pre) / len(sample) for cue in CUES]
    f += [sum(cue in w for w in post) / len(sample) for cue in CUES]
    adj_pre = [text[max(0, a - ADJ):a] for a in occ]
    adj_post = [text[a + len(st):a + len(st) + ADJ] for a in occ]
    for pats, wins in ((ADJ_PRE, adj_pre), (ADJ_POST, adj_post)):
        hits = [sum(1 for w in wins if pat.search(w)) for pat in pats.values()]
        f += [h / n_occ for h in hits] + [h > 0 for h in hits]
    f += [bool(pat.search(st)) for pat in INSIDE.values()]
    sides = [side_distances(text, a, a + len(st)) for a in sample]
    f += sides[0] + [min(x[i] for x in sides) for i in range(4)]
    f += [(sides[0][1] > sides[0][0]) - (sides[0][0] > sides[0][1]), (sides[0][3] > sides[0][2]) - (sides[0][2] > sides[0][3])]
    # Share of the string's words found on each side of the cause title, and the reverse coverage.
    toks = zone_tokens(st)
    for zone_low, zone_tok in zones:
        hit = len(toks & zone_tok)
        f += [hit / len(toks) if toks else 0.0]
    for zone_low, zone_tok in zones:
        f += [len(toks & zone_tok) / len(zone_tok) if zone_tok else 0.0]
    f += [bool(zone_low) and st.lower() in zone_low for zone_low, _ in zones]
    # The shares above dilute a single decisive context ("... Merlin, the appellant"), so also record
    # whether any occurrence has the cue; these scan every occurrence, not the sample.
    all_pre = [low[max(0, a - WINDOW):a] for a in occ]
    all_post = [low[a + len(st):a + len(st) + WINDOW] for a in occ]
    f += [any(cue in w for w in all_pre) for cue in CUES]
    f += [any(cue in w for w in all_post) for cue in CUES]
    evidence = ([c["peak"].get(l, 0) / n_runs for l in labels]
                + [any(cue in w for w in all_pre) for cue in ROLE_CUES]
                + [any(cue in w for w in all_post) for cue in ROLE_CUES]
                + [vs_pos is not None and any(0 < vs_pos - a <= 300 for a in occ),
                   vs_pos is not None and any(0 <= a - vs_pos <= 300 for a in occ), first < 1500])
    return f, evidence


def relatives(strings, evidence):
    """For each string, the strongest evidence among other strings of the document that share a
    distinctive word with it (case-folded), scaled by word overlap.  `Surendra` in the body thereby
    inherits what the taggers and the cause title say about `SURENDRA KUMAR ... Appellant`."""
    toks = [frozenset(w for w in TOKEN_RE.findall(st.lower()) if w not in STOP) for st in strings]
    by_tok = collections.defaultdict(list)
    for i, t in enumerate(toks):
        for w in t:
            by_tok[w].append(i)
    ev = np.asarray(evidence, dtype=np.float32)
    out = np.zeros((len(strings), 1 + ev.shape[1]), dtype=np.float32)
    for i, t in enumerate(toks):
        rel = {j for w in t for j in by_tok[w] if j != i}
        out[i, 0] = len(rel)
        for j in rel:
            jac = len(t & toks[j]) / len(t | toks[j])
            np.maximum(out[i, 1:], jac * ev[j], out=out[i, 1:])
    return out


def gazetteer(gold_dicts, exclude=None):
    gaz = collections.defaultdict(collections.Counter)
    for did, entries in gold_dicts.items():
        if did != exclude:
            for st, lab in entries:
                gaz[st][lab] += 1
    return gaz


def build(texts, runs, labels, gold_dicts, leave_one_out, gaz_min=0):
    """Feature matrix over all candidates of `texts`.  Returns X, rows [(doc, string)]."""
    X, rows = [], []
    full = gazetteer(gold_dicts)
    for did, text in texts.items():
        if not text:
            continue
        gaz = gazetteer(gold_dicts, exclude=did) if leave_one_out else full
        cands = candidates(text, runs, did, labels, gaz, gaz_min)
        low = text.lower()
        others = list(cands)
        zones = [(z.lower(), zone_tokens(z)) for z in party_zones(text)]
        base, evidence = zip(*(features(text, low, st, c, labels, others, gaz, zones) for st, c in cands.items()))
        X.append(np.concatenate([np.asarray(base, dtype=np.float32), relatives(others, evidence)], axis=1))
        rows += [(did, st) for st in others]
    return np.concatenate(X), rows


def decode(texts, rows, proba, classes, thr):
    """Keep a string when its best label is at least `thr` probable; place entries longest-first."""
    entries = collections.defaultdict(list)
    lab_idx = [i for i, c in enumerate(classes) if c != "NONE"]
    for (did, st), p in zip(rows, proba):
        best = max(lab_idx, key=lambda i: p[i])
        if p[best] >= thr:
            entries[did].append((st, classes[best]))
    return {did: lner.apply_dictionary(text, entries[did]) if text else [] for did, text in texts.items()}


def make_model(seed, args):
    return HistGradientBoostingClassifier(max_iter=args.iters, learning_rate=0.05, max_leaf_nodes=args.leaves,
                                          min_samples_leaf=args.min_leaf, l2_regularization=1.0,
                                          max_features=args.max_features, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--configs", nargs="+", required=True, help="cross-validation configs (held-out spans)")
    ap.add_argument("--test_spans", nargs="*", default=[], help="fold3 span files; switches to test mode")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--thrs", default="0.3,0.4,0.5,0.6")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=0, help="document shuffle for the cross-validation split")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--weight", choices=["none", "log", "sqrt"], default="log")
    ap.add_argument("--gaz_min", type=int, default=0,
                    help="add strings found in at least this many other training dictionaries as candidates")
    ap.add_argument("--drop", default="anypre:,anypost:", help="comma-separated feature-name prefixes to leave out")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--min_leaf", type=int, default=10)
    ap.add_argument("--max_features", type=float, default=1.0)
    ap.add_argument("--dump", default="", help="write the out-of-fold entry probabilities to this JSON file")
    ap.add_argument("--importance", action="store_true")
    args = ap.parse_args()

    labels = lner.load_labels(args.data)
    classes = ["NONE"] + labels
    texts, gold = {}, {}
    for fold in ("fold1", "fold2"):
        t, g = lner.load_fold(args.data, fold)
        texts.update(t), gold.update(g)
    gold_dicts = {did: {(texts[did][s:e], l) for s, e, l in gold[did]} for did in texts}
    X, rows = build(texts, [load_heldout(args.runs, c, texts) for c in args.configs], labels, gold_dicts,
                    leave_one_out=True, gaz_min=args.gaz_min)
    names = feature_names(labels)
    assert len(names) == X.shape[1], (len(names), X.shape)
    keep = [i for i, n in enumerate(names) if not any(n.startswith(p) for p in filter(None, args.drop.split(",")))]
    names, X = [names[i] for i in keep], X[:, keep]
    label_of = [dict(gold_dicts[did]).get(st, "NONE") for did, st in rows]
    y = np.array([classes.index(l) for l in label_of])
    occ = np.array([texts[did].count(st) for did, st in rows], dtype=np.float32)
    w = {"none": np.ones_like(occ), "log": 1 + np.log(occ), "sqrt": np.sqrt(occ)}[args.weight]
    groups = np.array([did for did, _ in rows])
    reach = sum(1 for d in texts for st, _ in gold_dicts[d] if (d, st) in set(rows))
    print(f"candidates={len(rows)} positives={int((y > 0).sum())} features={X.shape[1]} "
          f"gold strings reached={reach}/{sum(len(v) for v in gold_dicts.values())}", flush=True)

    if not args.test_spans:
        proba = np.zeros((len(rows), len(classes)))
        docs = sorted(set(groups))
        random.Random(args.cv_seed).shuffle(docs)
        part = {d: i % args.splits for i, d in enumerate(docs)}
        fold_of = np.array([part[g] for g in groups])
        for k in range(args.splits):
            tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
            for seed in range(args.seeds):
                m = make_model(seed, args).fit(X[tr], y[tr], sample_weight=w[tr])
                proba[np.ix_(te, m.classes_)] += m.predict_proba(X[te]) / args.seeds
        if args.dump:
            with open(args.dump, "w") as f:
                json.dump({"rows": rows, "classes": classes, "proba": proba.round(4).tolist(), "y": y.tolist()}, f)
        best = None
        for thr in (float(x) for x in args.thrs.split(",")):
            sc = lner.strict_scores(gold, decode(texts, rows, proba, classes, thr), labels)
            print(f"thr={thr:.2f}  F1={100 * sc['macro_f1']:.2f} P={100 * sc['macro_p']:.2f} R={100 * sc['macro_r']:.2f}",
                  flush=True)
            if best is None or sc["macro_f1"] > best[1]["macro_f1"]:
                best = (thr, sc)
        print(f"per label at thr={best[0]}")
        for l, v in best[1]["per_label"].items():
            print(f"   {l:10} P={100 * v['p']:5.1f} R={100 * v['r']:5.1f} F1={100 * v['f1']:5.1f}")
        if args.importance:
            from sklearn.inspection import permutation_importance
            m = make_model(0, args).fit(X, y, sample_weight=w)
            imp = permutation_importance(m, X, y, n_repeats=2, random_state=0, n_jobs=8).importances_mean
            for i in np.argsort(-imp)[:30]:
                print(f"   {names[i]:22} {imp[i]:.4f}")
        return

    # A slot written a.json+b.json holds alternative runs for the same role (e.g. the five k-fold
    # models of one variant); each combination of one run per slot is scored and the entry
    # probabilities are averaged, so the number of runs per document matches the cross-validation
    # setting the classifier was trained in.
    test_texts, _ = lner.load_fold(args.data, "fold3")
    models = [make_model(seed, args).fit(X, y, sample_weight=w) for seed in range(args.seeds)]
    slots = [slot.split("+") for slot in args.test_spans]
    # Alternatives are paired up (first of every slot, second of every slot, ...); shorter slots repeat.
    combos = [tuple(x[j % len(x)] for x in slots) for j in range(max(len(x) for x in slots))]
    assert all(len(c) == len(args.configs) for c in combos), "one test slot per cross-validation config"
    acc = collections.defaultdict(lambda: np.zeros(len(classes)))
    for combo in combos:
        Xt, rows_c = build(test_texts, load_runs(combo), labels, gold_dicts, leave_one_out=False, gaz_min=args.gaz_min)
        Xt = Xt[:, keep]
        for m in models:
            p = np.zeros((len(rows_c), len(classes)))
            p[:, m.classes_] = m.predict_proba(Xt)
            for row, pi in zip(rows_c, p):
                acc[row] += pi / (len(models) * len(combos))
    rows_t = list(acc)
    proba = np.stack([acc[r] for r in rows_t])
    pred = decode(test_texts, rows_t, proba, classes, args.thr)
    validate(pred, test_texts, labels)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/prediction.json", "w", encoding="utf-8") as f:
        json.dump({d: [[s, e, lab] for s, e, lab in ents] for d, ents in pred.items()}, f)
    with zipfile.ZipFile(f"{args.out_dir}/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{args.out_dir}/prediction.json", "prediction.json")
    n_ent = sum(len(v) for v in pred.values())
    print(f"test candidates={len(rows_t)} docs={len(pred)} entities={n_ent} per_doc={n_ent / len(pred):.1f}")
    print("per label:", {l: sum(1 for v in pred.values() for x in v if x[2] == l) for l in labels})
    print(f"wrote {args.out_dir}/submission.zip")


if __name__ == "__main__":
    main()
