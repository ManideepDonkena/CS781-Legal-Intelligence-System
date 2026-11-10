"""Parameter-efficient ways to adapt the Phase 1 model (written from the papers, no PEFT library).

LoRMA uses my Assignment 4 Q4 functions directly: q4_peft_lorma.py is section 4.4 of q4_peft.py copied unchanged
(LORMA_VARIANT, make_lorma_factors, lorma_dense, lorma_reordered, with the TODOs).  LoRALinear / LoRMALinear are
the assignment's LoRAConv1D / LoRMAConv1D (same names lora_A, lora_B, lorma_A, lorma_B, swap_ab, merged copy)
for nn.Linear layers (weight stored [out, in]); the Conv1D classes cannot wrap them directly ([in, out]).

LoRA  (Hu et al., 2021):            W' = W + s B A            A: [r, d_in] random, B: [d_out, r] zeros
LoRMA (Bihany, Patel, Modi, 2025):  W' = (s B A + I_{d_out}) W  A: [r, d_out] random, B: [d_out, r] zeros
    "LoRMA+" (additive rank inflation, the paper's Eq. 8): the low-rank matrix multiplies the frozen weight from
    the output side, and the identity keeps W' full rank.  Computed in the reordered way (lorma_reordered):
    y = W x, then y + s B (A y), so no d_out x d_out matrix is ever built.
Both start exactly at the frozen model because B starts at zero.  s = alpha / r (alpha = 2r gives s = 2, as in
the assignment), or alpha / sqrt(r) with rs=True (rank-stabilised scaling, Kalajdzievski 2023).
As in the assignment, LoRMA acts on W x without the bias (all linear layers of my model have no bias anyway).

Additions for the capstone runs (not in the assignment):
  dropout  on the adapter input (0.05 in every capstone run)
  layers   only wrap the linear layers inside these transformer blocks (e.g. the top 8 in Phase 3): the frozen
           lower blocks then need no backward pass at all
  targets  which linear layers to wrap (all 7 per block, or only the feed-forward ones in Phase 6)
  merge_adapters folds the trained factors into the weights (the assignment's merged_copy), so the adapted model
           is an ordinary model with no extra cost when it generates text
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "w_gate", "w_up", "w_down")
FFN = ("w_gate", "w_up", "w_down")


def _scaling(rank, alpha, rs=False):
    return alpha / (math.sqrt(rank) if rs else rank)


# --- LoRA (assignment 4.3) ---
class LoRALinear(nn.Module):
    """LoRA around a frozen nn.Linear: base(x) + s * (x A^T) B^T.  lora_A [r, in] Kaiming-uniform, lora_B [out, r]
    zeros (swap_ab=True: A = 0, B random), so the update is exactly zero at step 0."""

    def __init__(self, base, rank=16, alpha=32, dropout=0.0, rs=False, swap_ab=False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.out_features, self.in_features = self.base.weight.shape           # nn.Linear stores [out, in]
        self.rank, self.alpha, self.scaling = rank, alpha, _scaling(rank, alpha, rs)
        A, B = torch.empty(rank, self.in_features), torch.empty(self.out_features, rank)
        if swap_ab:
            nn.init.zeros_(A); nn.init.kaiming_uniform_(B, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5)); nn.init.zeros_(B)
        self.lora_A, self.lora_B = nn.Parameter(A), nn.Parameter(B)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)                                                      # frozen W x (+ b)
        A, B = self.lora_A.to(x.dtype), self.lora_B.to(x.dtype)
        return out + self.scaling * ((self.drop(x) @ A.T) @ B.T)                # x A^T B^T = (B A x)^T

    @torch.no_grad()
    def merged_weight(self):
        return self.base.weight.float() + self.scaling * (self.lora_B.float() @ self.lora_A.float())


# --- LoRMA (assignment 4.4): the assignment's own functions, unchanged, from q4_peft_lorma.py ---
from q4_peft_lorma import LORMA_VARIANT, _lorma_scaling, lorma_dense, lorma_reordered, make_lorma_factors  # noqa


class LoRMALinear(nn.Module):
    """Frozen nn.Linear with the trainable multiplicative adaptation (the assignment's LoRMAConv1D, Linear
    orientation): h = W x + b + s B (A (W x)); the bias is not transformed."""

    def __init__(self, base, rank=16, alpha=32, dropout=0.0, rs=False, swap_ab=False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.out_features, self.in_features = self.base.weight.shape           # nn.Linear stores [out, in]
        self.rank, self.alpha, self.scaling = rank, alpha, _scaling(rank, alpha, rs)
        gen = torch.Generator().manual_seed(torch.randint(0, 2 ** 31 - 1, (1,)).item())   # follows the global seed
        A, B = make_lorma_factors(self.in_features, self.out_features, rank, init=True, generator=gen,
                                  dtype=torch.float32, swap_ab=swap_ab)
        self.lorma_A, self.lorma_B = nn.Parameter(A), nn.Parameter(B)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        bias = 0 if self.base.bias is None else self.base.bias
        if not self.training or self.drop.p == 0:
            # exactly the assignment's lorma_reordered: y = W x, then y + s B (A y)
            return lorma_reordered(x, self.base.weight, (self.lorma_A, self.lorma_B), {"scaling": self.scaling}) + bias
        # while training with adapter dropout (0.05 in the capstone runs): the same line, dropout on A's input
        wx = F.linear(x, self.base.weight)                                      # W x without the bias
        A, B = self.lorma_A.to(wx.dtype), self.lorma_B.to(wx.dtype)
        return wx + bias + self.scaling * ((self.drop(wx) @ A.T) @ B.T)

    @torch.no_grad()
    def merged_weight(self):
        """W' = (s B A + I) W = W + s B (A W): the update applied once (the assignment's merged_copy)."""
        W = self.base.weight.float()
        return W + self.scaling * self.lorma_B.float() @ (self.lorma_A.float() @ W)


# --- attach, merge ---
def add_adapters(lm, kind, r=16, alpha=32, dropout=0.05, targets=TARGETS, layers=None, rs=False):
    """Freeze every backbone weight and wrap the target linear layers (only inside lm.layers[i] for i in `layers`
    when it is given).  Returns the new trainable parameters."""
    for p in lm.parameters():
        p.requires_grad_(False)
    cls = {"lora": LoRALinear, "lorma": LoRMALinear}[kind]
    roots = [lm] if layers is None else [lm.layers[i] for i in layers]
    new = []
    for module in [m for root in roots for m in root.modules()]:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                wrapped = cls(child, r, alpha, dropout, rs)
                setattr(module, name, wrapped)
                new += [p for p in wrapped.parameters() if p.requires_grad]
    return new


@torch.no_grad()
def merge_adapters(lm):
    """Replace every adapter by its plain linear layer carrying the merged weight (bias unchanged)."""
    for module in list(lm.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, (LoRALinear, LoRMALinear)):
                child.base.weight.copy_(child.merged_weight())
                setattr(module, name, child.base)
    return lm


def unfreeze_top(lm, k):
    """Partial fine-tuning: only the last k transformer layers and the final norm are trained."""
    for p in lm.parameters():
        p.requires_grad_(False)
    keep = list(lm.layers[-k:]) + [lm.norm]
    params = [p for m in keep for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    return params


def distill_loss(student, teacher, keep, temperature):
    """Mean over real tokens of T^2 * KL(teacher || student) on the per-token tag distributions (Hinton et al.)."""
    t = (teacher.float() / temperature).log_softmax(-1)
    s = (student.float() / temperature).log_softmax(-1)
    kl = (t.exp() * (t - s)).sum(-1)
    return temperature ** 2 * (kl * keep).sum() / keep.sum()
