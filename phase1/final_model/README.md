# Phase 1 final model code (ROPEFIX_295M)

This folder is the exact code that trained the Phase 1 models used in Phases 2-7 (`ROPEFIX_112M`, `ROPEFIX_112M_WIDE`,
`ROPEFIX_295M`), copied with its original layout so the imports work.

- `modern_arch.py` - the `yarn_gqa` model: 23 layers, width 1024, 16 query heads sharing 4 key/value heads, QK-RMSNorm,
  RMSNorm, SwiGLU, rotary positions with YaRN scaling, tied embeddings; 287,557,504 parameters for ROPEFIX_295M.
  The rotary-table fix (cos/sin kept in floating point instead of the token-id integer type) is in this file.
- `rope.py`, `attention.py` - my Assignment 3 files, unchanged (TODOs kept). `modern_arch.py` builds its RoPE tables
  with `build_rope_cache`, rotates q and k with `apply_rope`, and its `CausalSelfAttention` is a subclass of the
  assignment's `CausalSelfAttention` (same q/k/v/out projections, `n_rep`, `d_head`, `attention_scale`). Class names
  follow the assignment (`CausalSelfAttention`, `FeedForward`, `Block`, `DecoderLM`; old names kept as aliases); the
  stored weight names are unchanged so the checkpoints load. Checked on 9 Nov: same loss before and after
  (2.929502 on 8 dev cases x 2,048 tokens).
- `model_zoo.py` - `build_model(family, **config)` for every architecture family used in the studies.
- `studies/train_job.py` - training job; `studies/presets/*.json` hold every study setting
  (`ROPEFIX_295M.json` is the final model). `studies/run_queue.py` runs a queue of jobs.
- `studies/gen_modern.py`, `gen_modern_mbr.py` - text generation / MBR decoding for the Phase 1 uploads.
- `experiments/data.py` - corpus loading and BPE tokenizer training (Hugging Face `tokenizers` BpeTrainer, trained
  only on course text; no ready-made tokenizer); `experiments/engine.py` - training loop; `runner.py` - run bookkeeping.
- `ROPEFIX_295M_run_config.json` - the settings of the final run (lr 1e-3, warm-up 915, weight decay 0.1, 45,776 steps).

Train the final model (expects the course data caches under `artifacts/caches`, built by `studies/build_cache.py`):

    python studies/train_job.py --id ROPEFIX_295M --study ROPEFIX --cache final_bpe32k_mixA_3B \
        --d-model 1024 --layers 23 --heads 16 --tokens 3000000000 --bs 8 --ga 4 --lr 1e-3 --wd 0.1 \
        --seed 42 --extra '{"rope_float": true}' --keep-best

(4 key/value heads and tied embeddings are the defaults; the full resolved settings are in
`studies/presets/ROPEFIX_295M.json`.)

The older files one level up (`../train.py`, `../src/model_factory.py`, `../train_compare.py`) are early experiments
built on Hugging Face model classes (random initialisation, no pretrained weights); they were not used for the final model.
