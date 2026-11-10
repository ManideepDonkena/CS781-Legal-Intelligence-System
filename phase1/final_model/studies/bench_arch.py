#!/usr/bin/env python3
"""Throughput / memory / parameter-count benchmark of candidate configs (random tokens)."""
import sys, time, json, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from model_zoo import build_model

def bench(name, bs=8, ctx=2048, steps=6, compile_=False, **kw):
    m = build_model(**kw).cuda()
    n = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, kw["vocab_size"], (bs, ctx)).cuda()
    fwd = torch.compile(m) if compile_ else m
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 2):
        if i == 2: torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fwd(x, labels=x)["loss"]
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dt = (time.time() - t) / steps
    r = dict(name=name, params_M=round(n / 1e6, 1), tok_s=int(bs * ctx / dt), peak_GB=round(torch.cuda.max_memory_allocated() / 2**30, 1), compile=compile_)
    print(json.dumps(r), flush=True); return r

if __name__ == "__main__":
    base = dict(family="yarn_gqa", vocab_size=50000, max_length=2048)
    cfgs = [("legacy_181M", dict(family="transformer", vocab_size=50000, d_model=768, n_layers=20, n_heads=12, max_length=2048, gradient_checkpointing=False)),
            ("modern_55M", dict(base, d_model=512, n_layers=11, n_heads=8)),
            ("modern_125M", dict(base, d_model=768, n_layers=12, n_heads=12)),
            ("modern_285M", dict(base, d_model=1024, n_layers=22, n_heads=16))]
    for comp in (False, True):
        for name, kw in cfgs:
            try: bench(name, compile_=comp, **kw)
            except Exception as e: print(name, comp, "FAILED", type(e).__name__, str(e)[:120], flush=True)
            torch.cuda.empty_cache()
