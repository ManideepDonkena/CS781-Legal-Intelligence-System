"""Undo the ILDC "co." / "no." glitch in Phase 4 texts.

The ILDC preprocessing replaced "co" + any character with "company" and "no" + any character with "number"
(e.g. court -> companyrt, commissioner -> companymissioner, notice -> numberice, november -> numberember). Our Phase 1
model learned from clean Supreme Court text where these spellings never occur.

For every word that starts with "company" or "number" followed by more letters, the candidates are
"co" + x + rest (or "no" + x + rest) for every letter x; the word is replaced by the candidate that is most frequent
in a clean sample of the Phase 1 Supreme Court text, if that candidate occurs at least MIN_COUNT times and at least
RATIO times as often as the damaged word itself. If no letter fits but the rest is itself a common word, the
replaced character was a space or hyphen ("numberdoubt" -> "no doubt", "companyaccused" -> "co-accused").
Lone "company" / "number" are left alone (their original word
cannot be recovered from the word itself).  The table is built from train + dev words only.

  python text_fix.py            # builds ../data/glitch_fix.json and prints examples
"""
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "..", "data", "glitch_fix.json")
CLEAN = ("/workspace/models/cs781_old/p1-legal-lm_experiments/p1_full_pass/data/raw/Phase_1_Pretraining/Train/"
         "train_supreme_court.json")
PREFIXES = {"company": "co", "number": "no"}
WORD_RE = re.compile(r"\b(company|number)([a-z]+)\b")
MIN_COUNT, RATIO = 5, 20


def build(texts, clean_chars=300_000_000):
    damaged = Counter(m.group(0) for t in texts for m in WORD_RE.finditer(t))
    with open(CLEAN, encoding="utf-8", errors="ignore") as f:
        clean = Counter(re.findall(r"[a-z]+", f.read(clean_chars).lower()))
    table = {}
    for word in damaged:
        head, rest = WORD_RE.match(word).groups()
        cands = [PREFIXES[head] + x + rest for x in "abcdefghijklmnopqrstuvwxyz"]
        best = max(cands, key=lambda c: clean[c])
        if clean[best] >= MIN_COUNT and clean[best] >= RATIO * max(clean[word], 1):
            table[word] = best
        elif clean[rest] >= 100 * MIN_COUNT and clean[rest] >= RATIO * max(clean[word], 1):
            # the replaced character was a space or hyphen: "no doubt" -> "numberdoubt", "co-accused" -> "companyaccused"
            table[word] = ("no " if head == "number" else "co-") + rest
    return table, damaged


def load_table():
    return json.load(open(TABLE))


def fix(text, table):
    return WORD_RE.sub(lambda m: table.get(m.group(0), m.group(0)), text)


if __name__ == "__main__":
    import cjpe
    docs = cjpe.load("train") + cjpe.load("dev")
    table, damaged = build([d["text"] for d in docs])
    json.dump(table, open(TABLE, "w"), indent=0, sort_keys=True)
    total = sum(damaged.values())
    fixed = sum(k for w, k in damaged.items() if w in table)
    print(f"damaged word types {len(damaged)}, occurrences {total}; fixed types {len(table)}, "
          f"occurrences {fixed} ({100 * fixed / total:.1f}%)")
    for w, k in damaged.most_common(40):
        print(f"  {w:22s} {k:7d} -> {table.get(w, '(kept)')}")
    kept = [(w, k) for w, k in damaged.most_common() if w not in table][:15]
    print("most frequent kept:", kept)
