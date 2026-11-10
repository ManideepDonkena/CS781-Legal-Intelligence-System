"""Byte-level BPE encoder written from scratch for the Phase 1 tokenizer file (tokenizer.json).

It reproduces the encoding of the Phase 1 tokenizer without the Hugging Face `tokenizers` library:
  1. the special token <|endoftext|> is matched first and becomes its own id;
  2. the rest is split with the GPT-2 pattern  's|'t|'re|'ve|'m|'ll|'d| ?L+| ?N+| ?[^sLN]+|s+(?!S)|s+  where L is
     every Unicode letter, N every Unicode number and s every Unicode white-space character (the classes are built
     from the standard-library `unicodedata` tables, because Python's `re` has no \\p{L});
  3. each piece's UTF-8 bytes are mapped to printable characters (the GPT-2 byte table);
  4. the merges are applied in rank order (the lowest-ranked adjacent pair is merged everywhere, repeatedly);
  5. the resulting symbols are looked up in the vocabulary.
Pieces repeat a lot, so their ids are cached.  Use check() to compare with the library on real text.
"""
import json
import re
import sys
import unicodedata
from multiprocessing import Pool

# Unicode White_Space property (the characters a regex \s means in the original pattern).
WHITE = "\t\n\x0b\x0c\r \x85\xa0            " \
        "    　"


def _class(pred):
    """A regex character class (ranges) of every code point whose Unicode category satisfies pred."""
    ranges, start = [], None
    for cp in range(sys.maxunicode + 1):
        ok = pred(unicodedata.category(chr(cp)))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, sys.maxunicode))
    esc = lambda c: re.escape(chr(c))  # noqa: E731
    return "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in ranges)


def _bytes_to_unicode():
    """GPT-2's reversible map from the 256 byte values to printable characters."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = list(keep), 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(keep, chars)}


class ScratchBPE:
    SPECIAL = "<|endoftext|>"

    def __init__(self, path):
        t = json.load(open(path, encoding="utf-8"))
        m = t["model"]
        assert m["type"] == "BPE" and not m.get("dropout") and not m.get("ignore_merges")
        self.vocab = m["vocab"]
        merges = [tuple(x) if isinstance(x, list) else tuple(x.split(" ")) for x in m["merges"]]
        self.rank = {pair: i for i, pair in enumerate(merges)}
        self.special_id = {a["content"]: a["id"] for a in t.get("added_tokens", [])}
        L = _class(lambda c: c[0] == "L")
        N = _class(lambda c: c[0] == "N")
        S = "".join(re.escape(c) for c in WHITE)
        self.split_re = re.compile(
            rf"'s|'t|'re|'ve|'m|'ll|'d| ?[{L}]+| ?[{N}]+| ?[^{S}{L}{N}]+|[{S}]+(?![^{S}])|[{S}]+")
        self.byte_char = _bytes_to_unicode()
        self.cache = {}

    def _bpe(self, piece):
        ids = self.cache.get(piece)
        if ids is not None:
            return ids
        word = [self.byte_char[b] for b in piece.encode("utf-8")]
        while len(word) > 1:
            best, best_rank = None, None
            for pair in zip(word, word[1:]):
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            merged, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == best[0] and word[i + 1] == best[1]:
                    merged.append(best[0] + best[1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        ids = [self.vocab[s] for s in word]
        self.cache[piece] = ids
        return ids

    def encode(self, text):
        out = []
        for k, part in enumerate(text.split(self.SPECIAL)):
            if k:
                out.append(self.special_id[self.SPECIAL])
            for piece in self.split_re.findall(part):
                out.extend(self._bpe(piece))
        return out

    def token_to_id(self, token):
        return self.special_id.get(token, self.vocab.get(token))

    def decode(self, ids):
        """Text of a list of ids: token strings back to bytes through the inverse byte map, then UTF-8."""
        if not hasattr(self, "id_token"):
            self.id_token = {i: t for t, i in self.vocab.items()}
            self.id_token.update({i: t for t, i in self.special_id.items()})
            self.char_byte = {c: b for b, c in self.byte_char.items()}
        text = "".join(self.id_token.get(i, "") for i in ids)
        return bytes(self.char_byte[c] for c in text if c in self.char_byte).decode("utf-8", errors="replace")

    def encode_batch(self, texts, add_special_tokens=False, workers=32):
        """Same call shape as the library: returns objects with an .ids list."""
        global _ENCODER
        _ENCODER = self
        if len(texts) < 64 or workers <= 1:
            return [_Ids(self.encode(t)) for t in texts]
        with Pool(workers) as pool:
            return [_Ids(x) for x in pool.map(_encode_one, texts, chunksize=max(1, len(texts) // (workers * 8)))]


class _Ids:
    __slots__ = ("ids",)

    def __init__(self, ids):
        self.ids = ids


_ENCODER = None


def _encode_one(text):
    return _ENCODER.encode(text)


def check(path, texts):
    """Number of texts whose ids differ from the library's (0 means identical)."""
    from tokenizers import Tokenizer  # only to verify; not used for encoding
    ref = Tokenizer.from_file(path)
    mine = ScratchBPE(path).encode_batch(texts)
    lib = ref.encode_batch(texts, add_special_tokens=False)
    bad = [i for i, (a, b) in enumerate(zip(mine, lib)) if a.ids != b.ids]
    return len(bad), bad[:5]
