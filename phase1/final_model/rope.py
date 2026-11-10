# From my Assignment 3 (Q3.1, rope.py), unchanged: the same RoPE code is used by the capstone model (modern_arch.py).
# TODO 3.1: complete the RoPE cache / rotate-half / apply-RoPE implementation in this cell.
"""Rotary Positional Embeddings, implemented from scratch."""
import torch


def build_rope_cache(max_seq_len: int, d_head: int, base: float = 10000.0):
    """
    Returns (cos, sin) tensors of shape (max_seq_len, d_head) ready to be applied to
    query/key tensors of head dimension d_head. d_head must be even. This scaffold uses
    the split-half rotate_half convention: channel i is paired with channel i+d_head/2.
    """
    assert d_head % 2 == 0, "RoPE requires an even head dimension"
    # TODO: compute inverse frequencies: 1 / base^(2i/d_head) for i in [0, d_head/2)
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))   # (d_head/2,)
    # TODO: outer product of positions [0..max_seq_len) with inverse frequencies -> (T, d_head/2)
    positions = torch.arange(max_seq_len, dtype=torch.float32)        # float, never integer
    freqs = torch.outer(positions, inv_freq)
    # TODO: construct angles = torch.cat([freqs, freqs], dim=-1) so the shape
    #       becomes (T, d_head) and matches the split-half rotate_half convention.
    #       Do NOT use repeat_interleave here; it produces a different channel layout.
    #       Then return angles.cos(), angles.sin().
    angles = torch.cat([freqs, freqs], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and return (-x2, x1) concatenated -- the 90-degree
    rotation step used by the "rotate_half" formulation of RoPE."""
    # TODO
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, n_heads, T, d_head) -- a query or key tensor.
    cos, sin: (T, d_head), sliced from build_rope_cache's output for the current T.
    Returns the rotated tensor, same shape as x.
    Formula: x_rotated = x * cos + rotate_half(x) * sin
    """
    # TODO: broadcast cos/sin to (1, 1, T, d_head) and apply the formula above
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin
