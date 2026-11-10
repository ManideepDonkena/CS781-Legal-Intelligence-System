"""Modern Transformer Architecture for Legal LLM.

Core Architecture:
- Decoder-only Transformer
- Pre-RMSNorm
- SwiGLU activations (hidden_dim = 8/3 * d_model)
- QK-RMSNorm (per-head query and key RMSNorm)
- Tied Embeddings (W_E = W_U^T)
- Multi-Head Latent Attention (MLA) or GQA (1:4)
- Rotary Position Embeddings with YaRN context scaling (2K -> 8K)
- Inference: Contrastive search (alpha=0.55, k=4) or scoped presence penalties (repeat_last_n=64)
- Speculative Decoding draft model support (50M parameter configuration)
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple, List, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# RoPE: the same code as my Assignment 3 (rope.py: build_rope_cache, rotate_half, apply_rope).
from rope import build_rope_cache, rotate_half, apply_rope  # noqa: E402
# Attention: subclass of my Assignment 3 CausalSelfAttention (attention.py).
from attention import CausalSelfAttention as A3CausalSelfAttention  # noqa: E402

# Naming follows my Assignment 3 model (attention.py / model_arch.py): CausalSelfAttention, FeedForward, Block,
# DecoderLM, with d_head, n_rep and attention_scale inside attention.  The features differ (GQA 16:4, QK-RMSNorm,
# Pre-RMSNorm, SwiGLU feed-forward, YaRN), and the stored weight names (token, layers, attn_norm, ffn_norm, norm,
# lm_head) are kept so that the trained checkpoints load unchanged.  The old class names remain as aliases.


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def build_yarn_rope_cache(max_seq_len: int, d_head: int, base: float, scale: float, original_max_len: int,
                          beta_fast: float = 32.0, beta_slow: float = 1.0):
    """build_rope_cache (Assignment 3) with YaRN-interpolated frequencies (Peng et al., 2023), used only when the
    context is stretched beyond the training length (scale > 1).  Same steps: inverse frequencies, outer product
    with float positions, cat([freqs, freqs]) for the split-half rotate_half layout, then cos and sin."""
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    wavelength = 2.0 * math.pi / inv_freq
    r = original_max_len / wavelength
    # gamma = 0 (long wavelengths) -> fully interpolated (freq / scale); gamma = 1 (short) -> unchanged
    gamma = torch.clamp((r - beta_slow) / (beta_fast - beta_slow), 0.0, 1.0)
    inv_freq = (1.0 - gamma) * (inv_freq / scale) + gamma * inv_freq
    positions = torch.arange(max_seq_len, dtype=torch.float32)        # float, never integer
    freqs = torch.outer(positions, inv_freq)
    angles = torch.cat([freqs, freqs], dim=-1)
    return angles.cos(), angles.sin()


class YaRNRotaryEmbedding(nn.Module):
    """The (cos, sin) tables for every position, built once like the rope_cos / rope_sin buffers of Assignment 3's
    CausalSelfAttention, and sliced per call (offset = position of the first token, for the key/value cache).
    scale == 1 (training and Phases 2-5, 7): exactly build_rope_cache.  scale > 1 (Phase 6, 4,096 tokens): YaRN
    frequencies and the attention temperature mscale = 0.1 ln(scale) + 1."""
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base_context_length: int = 2048,
        base: float = 10000.0,
        scale: float = 4.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        float_output: bool = False,
    ):
        super().__init__()
        self.dim, self.base, self.scale = dim, base, scale
        self.max_position_embeddings = max_position_embeddings
        self.base_context_length = base_context_length
        self.float_output = float_output
        self.mscale = float(0.1 * math.log(scale) + 1.0) if scale > 1.0 else 1.0
        if scale <= 1.0:
            cos, sin = build_rope_cache(max_position_embeddings, dim, base)
        else:
            cos, sin = build_yarn_rope_cache(max_position_embeddings, dim, base, scale, base_context_length,
                                             beta_fast, beta_slow)
        # kept in float32 on the model's device (not a state-dict entry, so checkpoints are unaffected)
        self.rope_cos, self.rope_sin = cos * self.mscale, sin * self.mscale

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.float_output:
            if self.rope_cos.device != x.device:
                self.rope_cos, self.rope_sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)
            return self.rope_cos[offset:offset + seq_len], self.rope_sin[offset:offset + seq_len]
        # Legacy behaviour (the rotary bug), kept so the pre-fix checkpoints behave exactly as trained: callers pass
        # the token ids as `x`, so the cast truncates cos/sin to whole numbers (almost all zeros) and attention
        # becomes uniform.  Truncation flips on the last bit (0.99999994 -> 0, 1.0 -> 1), so this path recomputes
        # the table on the input's device exactly as the pre-fix code did.  The final models never take this path.
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        if self.scale > 1.0:
            wavelength = 2.0 * math.pi / inv_freq
            gamma = torch.clamp((self.base_context_length / wavelength - 1.0) / (32.0 - 1.0), 0.0, 1.0)
            inv_freq = (1.0 - gamma) * (inv_freq / self.scale) + gamma * inv_freq
        inv_freq = inv_freq.to(x.device)
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=inv_freq.dtype)
        emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim=-1)
        return (emb.cos() * self.mscale).to(x.dtype), (emb.sin() * self.mscale).to(x.dtype)


class FeedForward(nn.Module):
    """Assignment 3's FeedForward with a SwiGLU instead of GELU: (SiLU(x W_gate) * (x W_up)) W_down."""
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            # 8/3 * d_model rounded to nearest multiple of 64
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class CausalSelfAttention(A3CausalSelfAttention):
    """Subclass of my Assignment 3 CausalSelfAttention (attention.py): its __init__ builds q_proj, k_proj, v_proj,
    out_proj, n_rep, d_head, attention_scale and the causal mask exactly as in the assignment.  Added features:
    QK-RMSNorm (each head's query and key RMS-normalised before RoPE), cos/sin passed in from the model (so YaRN
    and the cache offset work), and a key/value cache for generation.
    (use_mla: an optional latent-attention variant that the final model does not use.)"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        use_mla: bool = False,
        mla_latent_dim: Optional[int] = None,
        scale_factor: float = 1.0,
        attention_scale: Optional[float] = None,
        use_sdpa: bool = True,
        max_seq_len: int = 2048,
    ):
        n_kv = (n_heads if use_mla else (max(1, n_heads // 4) if n_kv_heads is None else n_kv_heads))
        # Assignment 3 __init__: projections, n_rep, d_head, attention_scale, causal mask.  use_rope=False because
        # the model passes in its own cos/sin (built with the same build_rope_cache).
        super().__init__(d_model, n_heads, n_kv, max_seq_len, dropout=0.0, use_rope=False,
                         attention_scale=attention_scale)
        self.d_model = d_model
        self.head_dim = self.d_head                      # old name
        self.use_mla = use_mla
        self.scale_factor = scale_factor
        # True: PyTorch's fused kernel for the same softmax(q k^T * scale + causal mask) v (needed for memory at
        # 2,048 tokens); False: the explicit Assignment 3 computation.  Both give the same result.
        self.use_sdpa = use_sdpa
        if self.use_mla:
            self.mla_latent_dim = mla_latent_dim or (d_model // 2)
            del self.k_proj, self.v_proj
            self.kv_down = nn.Linear(d_model, self.mla_latent_dim, bias=False)
            self.k_up = nn.Linear(self.mla_latent_dim, d_model, bias=False)
            self.v_up = nn.Linear(self.mla_latent_dim, d_model, bias=False)

        # QK-RMSNorm per head
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        if self.use_mla:
            q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            kv_latent = self.kv_down(x)
            k = self.k_up(kv_latent).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            v = self.v_up(kv_latent).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        else:
            q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # QK-RMSNorm, then RoPE on q and k (NOT v), as in Assignment 3
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # key/value cache for generation: earlier positions' k, v are kept and extended
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v) if kv_cache is not None or not self.training else None

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        S = k.shape[2]                                   # keys = cached + new positions
        if self.use_sdpa:
            is_causal = (T > 1) and (T == S) and (attn_mask is None)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.attention_scale,
                                                 is_causal=is_causal)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool \
                    else scores + attn_mask
            elif T == S and T <= self.causal_mask.shape[0]:
                scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))     # as in Assignment 3
            elif T > 1:
                # with a cache, query i (absolute position S - T + i) may look at keys 0 .. S - T + i
                causal = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
                scores = scores.masked_fill(~causal, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(weights, v)               # (B, n_heads, T, d_head)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        return self.out_proj(out), new_kv_cache


class Block(nn.Module):
    """Assignment 3's Block with norm_first=True (x + attn(norm(x)), then x + ffn(norm(x))), using RMSNorm
    (attn_norm, ffn_norm) instead of LayerNorm (ln1, ln2), and the SwiGLU FeedForward."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        use_mla: bool = False,
        dropout: float = 0.0,
        scale_factor: float = 1.0,
        ffn_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_mla=use_mla,
            scale_factor=scale_factor,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, hidden_dim=ffn_hidden, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-RMSNorm for Attention
        attn_out, new_kv = self.attn(self.attn_norm(x), cos, sin, kv_cache=kv_cache)
        x = x + self.drop(attn_out)
        # Pre-RMSNorm for SwiGLU FFN
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x, new_kv


class GradientScaler(torch.autograd.Function):
    """Scales gradients flowing through embeddings during backpropagation."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return x
    return GradientScaler.apply(x, scale)


class DecoderLM(nn.Module):
    """Assignment 3's DecoderLM (token embedding -> blocks -> final norm -> tied head) with RoPE inside every
    attention layer (no position table at the input), Pre-RMSNorm, SwiGLU, QK-RMSNorm, GQA 16:4 and YaRN.
    Stored weight names: token (tok_emb), layers (blocks), norm (ln_f), lm_head (head)."""
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        n_layers: int = 20,
        n_heads: int = 12,
        n_kv_heads: Optional[int] = None,
        max_length: int = 2048,
        yarn_target_length: int = 8192,
        yarn_train: bool = False,
        ffn_hidden: Optional[int] = None,
        residual_init: bool = True,
        use_mla: bool = False,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        tie_embeddings: bool = True,
        input_gradient_scale: float = 1.0,
        family: str = "yarn_gqa",
        rope_float: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.family = family
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or max(1, n_heads // 4)
        self.max_length = max_length
        self.yarn_target_length = yarn_target_length
        self.gradient_checkpointing = gradient_checkpointing
        self.tie_embeddings = tie_embeddings
        self.input_gradient_scale = input_gradient_scale
        self.rope_float = rope_float

        # Token embedding
        self.token = nn.Embedding(vocab_size, d_model)

        # YaRN RoPE (2K -> 8K context scaling)
        # Standard practice: pre-train with plain RoPE (scale 1) and apply YaRN only when
        # extending context (see extend_context). yarn_train=True restores the legacy
        # behaviour of interpolating frequencies already during pre-training.
        self.head_dim = d_model // n_heads
        self.yarn_train = yarn_train
        yarn_scale = max(1.0, yarn_target_length / max_length) if yarn_train else 1.0
        self.rotary = YaRNRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=yarn_target_length,
            base_context_length=max_length,
            scale=yarn_scale,
            float_output=rope_float,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            Block(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=self.n_kv_heads,
                use_mla=use_mla,
                dropout=dropout,
                scale_factor=1.0,
                ffn_hidden=ffn_hidden,
            )
            for _ in range(n_layers)
        ])

        # Final Pre-RMSNorm before unembedding
        self.norm = RMSNorm(d_model)

        # Embeddings: Tied (W_E = W_U^T) or Untied (independent W_U)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token.weight
        else:
            self.lm_head.weight = nn.Parameter(torch.empty(vocab_size, d_model))
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Weight initialization
        self.apply(self._init_weights)
        if residual_init:
            # GPT-2/Llama style: residual-stream output projections start small so the
            # variance of the residual stream does not grow with depth.
            std = 0.02 / math.sqrt(2 * n_layers)
            for layer in self.layers:
                for lin in (layer.attn.out_proj, layer.ffn.w_down):
                    torch.nn.init.normal_(lin.weight, mean=0.0, std=std)

    @torch.no_grad()
    def extend_context(self, target_length: int) -> None:
        """Switch to YaRN-scaled RoPE for inference/continued training beyond max_length."""
        device = self.token.weight.device
        scale = max(1.0, target_length / self.max_length)
        self.yarn_target_length = target_length
        self.rotary = YaRNRotaryEmbedding(dim=self.head_dim, max_position_embeddings=target_length,
                                          base_context_length=self.max_length, scale=scale,
                                          float_output=self.rope_float).to(device)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        offset: int = 0,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        b, length = input_ids.shape
        if offset + length > self.yarn_target_length:
            raise ValueError(f"Sequence length {offset + length} exceeds maximum context {self.yarn_target_length}")

        cos, sin = self.rotary(input_ids, length, offset=offset)
        x = self.token(input_ids)
        if self.input_gradient_scale != 1.0:
            x = scale_gradient(x, self.input_gradient_scale)

        new_kv_caches = [] if kv_caches is not None else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing during training
                x, new_c = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, layer_cache, use_reentrant=False
                )
            else:
                x, new_c = layer(x, cos, sin, kv_cache=layer_cache)
            if new_kv_caches is not None:
                new_kv_caches.append(new_c)

        final_hidden = self.norm(x)
        logits = self.lm_head(final_hidden)

        loss = lm_loss(logits, labels) if labels is not None else None
        res = {"logits": logits, "loss": loss}
        if new_kv_caches is not None:
            res["kv_caches"] = new_kv_caches
        if return_hidden_states:
            res["hidden_states"] = final_hidden
        return res


# Old names, so existing code and scripts keep working.
ModernAttention = CausalSelfAttention
ModernTransformerBlock = Block
SwiGLU = FeedForward
ModernCausalLM = DecoderLM


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Old helper: apply_rope (Assignment 3) to q and k."""
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin)
