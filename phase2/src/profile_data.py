"""Profile the Phase 2 L-NER data: sizes, label balance, span shape, annotation consistency."""
import argparse
import collections
import json
import re
import statistics as st


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to Phase_2_LNER directory")
    args = ap.parse_args()
    D = args.data

    labels = [l.strip() for l in open(f"{D}/Train/labels.txt", encoding="utf-8") if l.strip()]
    folds = {
        "fold1": (load(f"{D}/Train/fold1-text.json"), load(f"{D}/Train/fold1-gold.json")),
        "fold2": (load(f"{D}/Train/fold2-text.json"), load(f"{D}/Train/fold2-gold.json")),
    }
    test = load(f"{D}/Test/fold3-text.json")

    print("## Sizes")
    for name, (txt, gold) in folds.items():
        chars = [len(t) for t in txt.values()]
        words = [len(t.split()) for t in txt.values()]
        n_ent = sum(len(v) for v in gold.values())
        print(f"{name}: docs={len(txt)} ids_match={set(txt) == set(gold)} entities={n_ent} "
              f"chars med={int(st.median(chars))} max={max(chars)} | words med={int(st.median(words))} "
              f"p90={pct(words, .9)} max={max(words)} total={sum(words)}")
    chars = [len(t) for t in test.values()]
    words = [len(t.split()) for t in test.values()]
    print(f"fold3(test): docs={len(test)} chars med={int(st.median(chars))} max={max(chars)} | "
          f"words med={int(st.median(words))} p90={pct(words, .9)} max={max(words)} total={sum(words)}")
    overlap = (set(folds["fold1"][0]) | set(folds["fold2"][0])) & set(test)
    print(f"train/test id overlap: {len(overlap)}")

    print("\n## Offset integrity and span shape")
    bad_text = bad_range = 0
    lead_ws = trail_ws = 0
    overlaps = nested = dup = 0
    starts_mid_word = ends_mid_word = 0
    newline_in_text = sum("\n" in t for f in folds.values() for t in f[0].values())
    for name, (txt, gold) in folds.items():
        for did, ents in gold.items():
            t = txt[did]
            seen = set()
            prev_end = -1
            for e in sorted(ents, key=lambda e: (e[0], -e[1])):
                s, en, lab = e[0], e[1], e[2]
                if not (0 <= s < en <= len(t)):
                    bad_range += 1
                    continue
                if len(e) > 3 and t[s:en] != e[3]:
                    bad_text += 1
                if t[s].isspace():
                    lead_ws += 1
                if t[en - 1].isspace():
                    trail_ws += 1
                if s > 0 and t[s - 1].isalnum() and t[s].isalnum():
                    starts_mid_word += 1
                if en < len(t) and t[en - 1].isalnum() and t[en].isalnum():
                    ends_mid_word += 1
                if (s, en, lab) in seen:
                    dup += 1
                seen.add((s, en, lab))
                if s < prev_end:
                    overlaps += 1
                    if en <= prev_end:
                        nested += 1
                prev_end = max(prev_end, en)
    print(f"bad_range={bad_range} text_mismatch={bad_text} lead_ws={lead_ws} trail_ws={trail_ws} "
          f"start_mid_word={starts_mid_word} end_mid_word={ends_mid_word}")
    print(f"overlapping={overlaps} nested={nested} duplicate={dup} docs_with_newlines={newline_in_text}")

    print("\n## Label distribution (count per fold, docs containing, span length in chars/words)")
    print(f"{'label':10} {'f1':>5} {'f2':>5} {'tot':>5} {'%':>6} {'docs':>5} {'uniq':>5} "
          f"{'w_med':>5} {'w_p90':>5} {'w_max':>5} {'c_med':>5}")
    total = sum(len(v) for f in folds.values() for v in f[1].values())
    for lab in labels:
        cnt = {}
        docs = 0
        wl, cl, uniq = [], [], set()
        for name, (txt, gold) in folds.items():
            c = 0
            for did, ents in gold.items():
                es = [e for e in ents if e[2] == lab]
                c += len(es)
                docs += bool(es)
                for e in es:
                    s = txt[did][e[0]:e[1]]
                    wl.append(len(s.split()))
                    cl.append(len(s))
                    uniq.add(s.lower())
            cnt[name] = c
        tot = sum(cnt.values())
        print(f"{lab:10} {cnt['fold1']:5d} {cnt['fold2']:5d} {tot:5d} {100 * tot / total:6.1f} {docs:5d} "
              f"{len(uniq):5d} {int(st.median(wl)) if wl else 0:5d} {pct(wl, .9) if wl else 0:5d} "
              f"{max(wl) if wl else 0:5d} {int(st.median(cl)) if cl else 0:5d}")

    print("\n## Where entities sit in the document (relative position, deciles of count)")
    for lab in labels:
        pos = []
        for name, (txt, gold) in folds.items():
            for did, ents in gold.items():
                L = len(txt[did])
                pos += [e[0] / L for e in ents if e[2] == lab]
        hist = [0] * 10
        for p in pos:
            hist[min(9, int(p * 10))] += 1
        n = max(1, len(pos))
        print(f"{lab:10} " + " ".join(f"{100 * h / n:4.0f}" for h in hist))

    print("\n## Surface-string label ambiguity (same string, different labels, across corpus)")
    by_str = collections.defaultdict(collections.Counter)
    for name, (txt, gold) in folds.items():
        for did, ents in gold.items():
            for e in ents:
                by_str[txt[did][e[0]:e[1]].lower()][e[2]] += 1
    amb = {s: c for s, c in by_str.items() if len(c) > 1}
    amb_mentions = sum(sum(c.values()) for c in amb.values())
    print(f"ambiguous strings={len(amb)} of {len(by_str)} | mentions affected={amb_mentions} of {total}")
    pair = collections.Counter()
    for c in amb.values():
        pair[" / ".join(sorted(c))] += sum(c.values())
    for k, v in pair.most_common(12):
        print(f"  {v:5d}  {k}")

    print("\n## Within-document consistency: is a tagged string tagged at every occurrence?")
    tagged_occ = untagged_occ = 0
    per_label_miss = collections.Counter()
    per_label_all = collections.Counter()
    for name, (txt, gold) in folds.items():
        for did, ents in gold.items():
            t = txt[did]
            spans = {(e[0], e[1]) for e in ents}
            strs = {}
            for e in ents:
                strs.setdefault(t[e[0]:e[1]], e[2])
            for s, lab in strs.items():
                if len(s) < 4:
                    continue
                for m in re.finditer(r"(?<!\w)" + re.escape(s) + r"(?!\w)", t):
                    per_label_all[lab] += 1
                    if (m.start(), m.end()) in spans:
                        tagged_occ += 1
                    else:
                        covered = any(a <= m.start() and m.end() <= b for a, b in spans)
                        if not covered:
                            untagged_occ += 1
                            per_label_miss[lab] += 1
    print(f"exact-string occurrences tagged={tagged_occ} untagged(and not inside another span)={untagged_occ}")
    for lab in labels:
        if per_label_all[lab]:
            print(f"  {lab:10} untagged {per_label_miss[lab]:5d} / {per_label_all[lab]:5d} "
                  f"({100 * per_label_miss[lab] / per_label_all[lab]:.0f}%)")

    print("\n## Train->other-fold string coverage (how much is memorisable)")
    for a, b in (("fold1", "fold2"), ("fold2", "fold1")):
        ta, ga = folds[a]
        tb, gb = folds[b]
        known = collections.defaultdict(set)
        for did, ents in ga.items():
            for e in ents:
                known[e[2]].add(ta[did][e[0]:e[1]].lower())
        line = []
        for lab in labels:
            ms = [tb[did][e[0]:e[1]].lower() for did, ents in gb.items() for e in ents if e[2] == lab]
            if ms:
                line.append(f"{lab}={100 * sum(m in known[lab] for m in ms) / len(ms):.0f}%")
        print(f"  seen in {a}, mentions of {b} covered: " + " ".join(line))

    print("\n## Most common strings per label")
    for lab in labels:
        c = collections.Counter()
        for name, (txt, gold) in folds.items():
            for did, ents in gold.items():
                for e in ents:
                    if e[2] == lab:
                        c[txt[did][e[0]:e[1]]] += 1
        print(f"{lab:10} " + " | ".join(f"{s[:40]} ({n})" for s, n in c.most_common(7)))


if __name__ == "__main__":
    main()
