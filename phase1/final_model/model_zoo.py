"""Small, comparable causal language models for the Legal LLM P1 study.

All models consume [batch, sequence] token IDs and return logits plus an
optional next-token loss. The families include:
standard attention, recurrent GRU, causal linear attention, and a local/global
hybrid Transformer, plus a Transformer with shared layer groups.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def lm_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))


class CausalAttention(nn.Module):
    def __init__(self, width, heads, window=None):
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.heads, self.head_dim, self.window = heads, width // heads, window
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        if self.window is None:
            # PyTorch can dispatch this to Flash Attention on supported GPUs.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            pos = torch.arange(t, device=x.device)
            allowed = (pos[None, :] <= pos[:, None]) & (pos[None, :] >= pos[:, None] - self.window + 1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[None, None])
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))


class LinearCausalAttention(nn.Module):
    """Positive-feature causal attention using prefix sums (linear in length)."""
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2).float() for z in (q, k, v))
        q, k = F.elu(q) + 1, F.elu(k) + 1
        # Checkpoint bounded chunks instead of retaining [B,H,T,D,D] for
        # the entire context. This is a reference implementation, not a
        # specialized linear-attention kernel; benchmark actual memory/time.
        state = k.new_zeros(b, self.heads, self.head_dim, self.head_dim)
        total = k.new_zeros(b, self.heads, self.head_dim)
        chunks = []
        def scan(qc, kc, vc, previous, previous_total):
            with torch.autocast(device_type=qc.device.type, enabled=False):
                kv = torch.einsum("bhtd,bhte->bhtde", kc, vc).cumsum(2) + previous[:, :, None]
                ks = kc.cumsum(2) + previous_total[:, :, None]
                num = torch.einsum("bhtd,bhtde->bhte", qc, kv)
                den = (qc * ks).sum(-1, keepdim=True).clamp_min(1e-6)
                return num / den, kv[:, :, -1], ks[:, :, -1]
        for start in range(0, t, 32):
            args = (q[:, :, start:start+32], k[:, :, start:start+32],
                    v[:, :, start:start+32], state, total)
            part, state, total = checkpoint(scan, *args, use_reentrant=False) if self.training else scan(*args)
            chunks.append(part)
        y = torch.cat(chunks, dim=2).to(x.dtype)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, attention="full", window=256, dropout=0.0):
        super().__init__()
        if attention == "linear":
            self.attn = LinearCausalAttention(width, heads)
        else:
            self.attn = CausalAttention(width, heads, window if attention == "local" else None)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.ff(self.norm2(x)))


class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=640, n_layers=8, n_heads=10, max_length=2048,
                 family="transformer", local_window=256, dropout=0.0,
                 shared_groups=2, gradient_checkpointing=False, init_fix=False):
        super().__init__()
        self.family, self.max_length = family, max_length
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.token = nn.Embedding(vocab_size, d_model)
        if family == "gru":
            self.position = nn.Embedding(max_length, d_model)
            self.core = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
            self.norm = nn.LayerNorm(d_model)
        else:
            self.position = nn.Embedding(max_length, d_model)
            blocks = []
            if family == "shared" and not 1 <= shared_groups <= n_layers:
                raise ValueError("shared_groups must be between 1 and n_layers")
            for layer in range(shared_groups if family == "shared" else n_layers):
                kind = ("full" if family in ("transformer", "shared") else "linear" if family == "linear"
                        else "local" if family == "local" else ("local" if layer % 2 == 0 else "full"))
                blocks.append(TransformerBlock(d_model, n_heads, kind, local_window, dropout))
            self.core = nn.ModuleList(blocks)
            self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight
        if init_fix:
            # The legacy init leaves tied nn.Embedding at N(0,1): initial loss ~450 instead of
            # ln(V)~10.8 and the first ~1000 updates are spent undoing it. GPT-2-style init:
            # N(0,0.02) everywhere, residual output projections scaled by 1/sqrt(2L).
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.zeros_(m.bias)
            if family != "gru":
                std = 0.02 / math.sqrt(2 * n_layers)
                for block in self.core:
                    nn.init.normal_(block.attn.out.weight, mean=0.0, std=std)
                    nn.init.normal_(block.ff[2].weight, mean=0.0, std=std)

    def forward(self, input_ids, labels=None):
        _, length = input_ids.shape
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length={self.max_length}")
        positions = torch.arange(length, device=input_ids.device)
        x = self.token(input_ids) + self.position(positions)[None, :, :]
        if self.family == "gru":
            x, _ = self.core(x)
        else:
            for layer in range(self.n_layers):
                block = self.core[layer % len(self.core)]
                x = checkpoint(block, x, use_reentrant=False) if self.gradient_checkpointing and self.training else block(x)
        logits = self.lm_head(self.norm(x))
        return {"logits": logits, "loss": lm_loss(logits, labels) if labels is not None else None}


FAMILIES = ("transformer", "gru", "linear", "hybrid", "shared", "local", "modern_transformer", "yarn_gqa")


def build_model(family, **kwargs):
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}; got {family!r}")
    if family in ("modern_transformer", "yarn_gqa"):
        from modern_arch import ModernCausalLM
        kw = dict(kwargs)
        if "layers" in kw and "n_layers" not in kw:
            kw["n_layers"] = kw.pop("layers")
        if "heads" in kw and "n_heads" not in kw:
            kw["n_heads"] = kw.pop("heads")
        if "context" in kw and "max_length" not in kw:
            kw["max_length"] = kw.pop("context")
        kw.pop("family", None)
        return ModernCausalLM(family=family, **kw)
    return CausalLM(family=family, **kwargs)
