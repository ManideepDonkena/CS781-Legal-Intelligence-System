"""Portable tokenizer loading for byte BPE and the AYN-inspired SentencePiece BPE.

AYN uses NFC, digit splitting and byte fallback. We explicitly preserve whitespace
and use no dummy prefix. Offsets for this variant refer to NFC text, not raw text.
The JSON bundle embeds the native model, so copied model bundles are self-contained.
"""
from __future__ import annotations

import base64
from array import array
from bisect import bisect_left
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unicodedata

KINDS = ["bpe", "legal_bpe", "digit_bpe", "ayn_bpe", "kl3m_bpe", "kl3m_char3", "kl3m_char4"]
EOS = "<|endoftext|>"


def normalized_text(tokenizer, text):
    if isinstance(tokenizer, SentencePieceBPE):
        return unicodedata.normalize("NFC", text)
    return tokenizer.normalizer.normalize_str(text) if tokenizer.normalizer else text


class SentencePieceBPE:
    def __init__(self, model):
        import sentencepiece as spm
        from sentencepiece import sentencepiece_pb2
        self.model = model
        self.processor = spm.SentencePieceProcessor(model_proto=model)
        self._text_proto = sentencepiece_pb2.SentencePieceText
        self.special_ids = [self.processor.unk_id(), self.processor.eos_id()]

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)
        ids, offsets, pieces = [], [], []
        start = 0
        # SentencePiece reserves U+2581 as its space marker. Encode literal
        # occurrences through byte fallback so they remain lossless as well.
        for segment in text.split("▁"):
            if start:
                for byte in "▁".encode():
                    piece = f"<0x{byte:02X}>"
                    ids.append(self.processor.piece_to_id(piece))
                    offsets.append((start - 1, start))
                    pieces.append(piece)
            # Serialized protobuf is supported by both SentencePiece 0.2.1
            # and 0.2.2. Unlike the removed immutable wrapper, its offsets
            # refer to UTF-8 bytes; convert them back to NFC character offsets.
            encoded = self._text_proto()
            encoded.ParseFromString(self.processor.encode(segment, out_type="serialized_proto"))
            boundaries = None
            if not segment.isascii():
                boundaries = array("I", [0])
                for char in segment:
                    boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
            for p in encoded.pieces:
                begin, end = p.begin, p.end
                if boundaries is not None:
                    begin, end = bisect_left(boundaries, begin), bisect_left(boundaries, end)
                    if (begin >= len(boundaries) or end >= len(boundaries)
                            or boundaries[begin] != p.begin or boundaries[end] != p.end):
                        raise ValueError("SentencePiece returned an offset inside a UTF-8 character")
                ids.append(p.id)
                offsets.append((start + begin, start + end))
                pieces.append(p.piece)
            start += len(segment) + 1
        return SimpleNamespace(ids=ids, offsets=offsets, tokens=pieces)

    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            ids = [int(i) for i in ids if i not in self.special_ids]
        return self.processor.decode([int(i) for i in ids])

    def token_to_id(self, token):
        index = self.processor.piece_to_id(token)
        return index if self.processor.id_to_piece(index) == token else None

    def get_vocab_size(self):
        return self.processor.get_piece_size()

    def save(self, path):
        Path(path).write_text(json.dumps({
            "backend": "sentencepiece_bpe_nfc_v1",
            "normalization": "NFC; whitespace preserved; offsets in NFC text",
            "model_base64": base64.b64encode(self.model).decode("ascii"),
            "unicode_version": unicodedata.unidata_version,
        }, indent=2) + "\n")


def load_tokenizer(path):
    data = json.loads(Path(path).read_text())
    if data.get("backend") == "sentencepiece_bpe_nfc_v1":
        if data["unicode_version"] != unicodedata.unidata_version:
            raise ValueError("NFC Unicode version changed; use the tokenizer's training Python environment")
        return SentencePieceBPE(base64.b64decode(data["model_base64"], validate=True))
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def train_sentencepiece(sample_texts, vocab_size):
    import sentencepiece as spm
    def sentences():
        for text in sample_texts:
            # Bounded chunks prevent SentencePiece's default line-length limit
            # from silently discarding long judgments. No validation input here.
            text = unicodedata.normalize("NFC", text)
            for start in range(0, len(text), 4000):
                yield text[start:start + 4000]
    buffer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences(), model_writer=buffer,
        model_type="bpe", vocab_size=vocab_size, hard_vocab_limit=False,
        character_coverage=0.9995, byte_fallback=True, split_digits=True,
        normalization_rule_name="identity", remove_extra_whitespaces=False,
        add_dummy_prefix=False, split_by_whitespace=True,
        unk_id=0, bos_id=-1, eos_id=1, pad_id=-1, eos_piece=EOS,
        num_threads=1, shuffle_input_sentence=False, input_sentence_size=0,
        max_sentence_length=16001, minloglevel=1,
    )
    return SentencePieceBPE(buffer.getvalue())
