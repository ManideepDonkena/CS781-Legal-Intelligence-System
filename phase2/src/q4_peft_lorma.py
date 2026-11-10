"""From my Assignment 4 (Q4, q4_peft.py, section "4.4 LoRMA"), copied unchanged with its TODO comments.
The capstone adapters (adapters.py: LoRMALinear) build their factors with make_lorma_factors and compute the
adapted layer with lorma_reordered from this file; lorma_dense is the literal paper equation used to check it.
(Lines 193-248 of q4_peft.py; LoRAConv1D / LoRMAConv1D themselves wrap GPT-2's Conv1D, whose weight is stored
[in, out], so they cannot wrap the nn.Linear layers of my model directly: adapters.py has the nn.Linear versions.)
"""
import math

import torch

# --- 4.4 LoRMA ---
LORMA_VARIANT = {"name": "LoRMA+ (additive rank inflation)",
                 "equation": "W' = (alpha/r * B A + I_{d_out}) W,  B in R^{d_out x r}, A in R^{r x d_out}; "
                             "h = W x + (alpha/r) B (A (W x))",
                 "paper_version": "Bihany, Patel & Modi, arXiv:2506.07621v1, Eq. 8 (LoRMA+); init B = 0, A random",
                 "scaling": 2.0}                 # alpha / r with alpha = 2r, the same scale as the LoRA runs (16 / 8)


def _lorma_scaling(variant_config):
    cfg = variant_config or LORMA_VARIANT
    if "alpha" in cfg and "rank" in cfg:
        return cfg["alpha"] / cfg["rank"]
    return cfg.get("scaling", LORMA_VARIANT["scaling"])


def make_lorma_factors(d_in, d_out, rank, init=False, generator=None, dtype=torch.float64, swap_ab=False):
    """Factor tensors for YOUR variant acting on a weight W of shape [d_out, d_in] (nn.Linear
    orientation). init=True -> the paper's base-preserving initialisation (output == x @ W.T);
    init=False -> random non-trivial factors. Returns a tuple accepted by lorma_dense/lorma_reordered.
    swap_ab (4.4 "Variants", only meaningful with init=True; default False = the paper's initialisation):
    exchange which factor is zero-initialised (A = 0, B random) while keeping the output == x @ W.T."""
    # TODO 9: Construct factors for the selected variant and initialization mode.
    # LoRMA+ multiplies W from the output side, so both factors live in the d_out space: (A, B) with
    # A [r, d_out] and B [d_out, r]; d_in only fixes the shape of W itself.
    if not init:
        A = torch.randn(rank, d_out, generator=generator, dtype=dtype) / math.sqrt(d_out)
        B = torch.randn(d_out, rank, generator=generator, dtype=dtype) / math.sqrt(rank)
        return A, B
    bound_a, bound_b = 1.0 / math.sqrt(d_out), 1.0 / math.sqrt(rank)          # Kaiming-uniform bounds (fan_in)
    if not swap_ab:          # paper initialisation: B = 0, A random -> B A = 0 -> W' = W
        A = (torch.rand(rank, d_out, generator=generator, dtype=dtype) * 2 - 1) * bound_a
        B = torch.zeros(d_out, rank, dtype=dtype)
    else:                    # swapped: A = 0, B random -> still B A = 0 -> W' = W
        A = torch.zeros(rank, d_out, dtype=dtype)
        B = (torch.rand(d_out, rank, generator=generator, dtype=dtype) * 2 - 1) * bound_b
    return A, B


def lorma_dense(x, frozen_weight, factors, variant_config=None):
    """Literal paper equation: materialise the adapted weight, return x @ W'.T."""
    # TODO 10: Build the adapted weight from the selected paper equation.
    A, B = factors
    s = _lorma_scaling(variant_config)
    eye = torch.eye(frozen_weight.shape[0], dtype=frozen_weight.dtype, device=frozen_weight.device)
    adapted = (s * (B @ A) + eye) @ frozen_weight                 # [d_out, d_out] @ [d_out, d_in]
    return x @ adapted.T


def lorma_reordered(x, frozen_weight, factors, variant_config=None):
    """Same function without materialising d x d products."""
    # TODO 11: Apply the same transformation without materializing square weight products.
    A, B = factors
    s = _lorma_scaling(variant_config)
    y = x @ frozen_weight.T                                       # W x            [N, d_out]
    return y + s * ((y @ A.T) @ B.T)                              # + s B (A (W x)), only [N, r] intermediates

