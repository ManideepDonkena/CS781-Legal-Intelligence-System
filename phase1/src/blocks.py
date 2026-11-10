"""Memory-mapped blocks. Buffers never cross sources; one tail per source."""
from bisect import bisect_right
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class Blocks(Dataset):
    def __init__(self, directory, manifest, split, context, seed=42, length=None, source=None):
        self.context, self.seed = context, seed
        self.arrays, self.ends, self.names = [], [], []
        self.tails = {}
        for name, info in manifest["streams"][split].items():
            if source and name != source:
                continue
            array = np.memmap(Path(directory) / info["file"], dtype="<u4", mode="r")
            count = len(array) // context
            self.tails[name] = len(array) % context
            if count:
                self.arrays.append(array)
                self.names.append(name)
                self.ends.append(count + (self.ends[-1] if self.ends else 0))
        self.count = self.ends[-1] if self.ends else 0
        if not self.count:
            raise ValueError(f"No complete blocks for {split}/{source}; provide more data or shorter context")
        self.length = length or self.count
        # Four-round Feistel permutation, cycle-walked into [0, count).
        # Unlike a small affine stride, this scatters early batches without
        # allocating a corpus-sized permutation array. Not a security primitive.
        self.half_bits = max(1, (max(1, self.count - 1).bit_length() + 1) // 2)
        self.mask = (1 << self.half_bits) - 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        source, start = self.location(index)
        return torch.from_numpy(np.array(self.arrays[source][start:start + self.context], dtype=np.int64))

    def location(self, index):
        epoch, pos = divmod(index, self.count)
        while True:
            left, right = pos >> self.half_bits, pos & self.mask
            for round_id in range(4):
                digest = hashlib.blake2b(f"{self.seed}:{epoch}:{round_id}:{right}".encode(), digest_size=8).digest()
                left, right = right, left ^ (int.from_bytes(digest, "big") & self.mask)
            pos = (left << self.half_bits) | right
            if pos < self.count:
                break
        source = bisect_right(self.ends, pos)
        start = (pos - (self.ends[source - 1] if source else 0)) * self.context
        return source, start
