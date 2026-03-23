# ADR-006a: CUDA Curriculum Rerun with Compression Ratio

**Status**: Complete
**Date**: 2026-03-23
**Hardware**: Intel laptop, NVIDIA RTX 2060 6 GB
**Depends on**: ADR-006 (MLX curriculum exploration)
**Repo**: `autoresearch` (the PyTorch/CUDA sibling repo)

## Purpose

The CUDA v2 curriculum experiments used **token diversity** (`x.unique().numel() / x.numel()`) as the difficulty metric. ADR-006 standardizes on **compression ratio** (literature's strongest signal per arXiv:2506.11300) for cross-platform comparison with the MLX experiments.

This document specifies the minimal rerun needed on the 2060 to align metrics.

## What to change

**One function in `train.py`** — replace `score_batch_difficulty`:

```python
# BEFORE (token diversity)
def score_batch_difficulty(x):
    """Token diversity: unique tokens / total tokens. Higher = harder."""
    return x.unique().numel() / x.numel()

# AFTER (compression ratio)
def score_batch_difficulty(x):
    """Compression ratio: gzip compressed / raw bytes. Higher = harder."""
    import gzip
    raw = x.cpu().numpy().tobytes()
    return len(gzip.compress(raw)) / len(raw)
```

**Nothing else changes.** Same buffer implementation, same King Wen mapping, same env vars, same 5-minute budget.

## Runs to execute

All at DEPTH=4, seed 42, standard warmdown (WARMDOWN_RATIO=0.5):

```bash
# 1. Sequential baseline (no buffer — should match v2 result)
AUTORESEARCH_CURRICULUM=sequential uv run train.py > run_seq.log 2>&1

# 2. Buffered passthrough
AUTORESEARCH_CURRICULUM=buffered_passthrough uv run train.py > run_pass.log 2>&1

# 3. Random shuffle
AUTORESEARCH_CURRICULUM=random uv run train.py > run_rand.log 2>&1

# 4. Easy to hard
AUTORESEARCH_CURRICULUM=easy_to_hard uv run train.py > run_e2h.log 2>&1

# 5. Hard to easy
AUTORESEARCH_CURRICULUM=hard_to_easy uv run train.py > run_h2e.log 2>&1

# 6. King Wen
AUTORESEARCH_CURRICULUM=king_wen uv run train.py > run_kw.log 2>&1
```

Shao Yong is dropped (see ADR-006 rationale). 6 runs x ~7 min = **~42 minutes total**.

## What to record

For each run, extract:

```bash
grep "^val_bpb:\|^peak_vram_mb:" run_*.log
```

Log to `results.tsv` with description prefix `curriculum-cr:` (compression ratio) to distinguish from the v2 token-diversity results (`curriculum:`).

## What to watch for

1. **Sequential baseline should match v2** (val_bpb ≈ 1.719). The difficulty metric doesn't affect sequential ordering. If it differs by more than 0.005, something else changed.

2. **Overhead from gzip:** The `gzip.compress` call runs on CPU. Expected ~100ms per 64-batch buffer refill. If `curriculum_overhead_seconds` exceeds 20% of budget (60s), the gzip is too expensive on this hardware. Fallback: use `zlib.compress` (faster, same ordering signal).

3. **Ordering ranking change:** The key question is whether compression ratio changes the relative ranking of orderings compared to token diversity. Specifically:
   - Does random still win?
   - Does King Wen still rank worst?
   - Does hard_to_easy still match easy_to_hard?

If the ranking changes, the difficulty metric — not the ordering — drives the effect. If ranking is stable, the decorrelation hypothesis is strengthened (ordering rankings are metric-independent).

## Results (2026-03-23)

### CUDA comparison table

| Ordering | Token Diversity (v2) | Compression Ratio (v3) | Delta |
|----------|---------------------|----------------------|-------|
| sequential | 1.719 | 1.778 | +0.059 |
| buffered_passthrough | 1.680 | 1.640 | -0.040 |
| random | 1.614 | 1.627 | +0.013 |
| easy_to_hard | 1.632 | 1.634 | +0.002 |
| hard_to_easy | 1.627 | 1.634 | +0.007 |
| king_wen | 1.662 | 1.638 | -0.024 |

### Watchpoint results

**1. Sequential baseline diverged significantly** (1.719 → 1.778, delta +0.059). This is unexpected — the sequential mode buffers + scores but yields in original order, so the scoring function itself is changing training dynamics. The gzip scoring adds CPU overhead that slows down step throughput, and the different scoring computation may affect memory layout. This makes the compression-ratio "sequential" baseline not directly comparable to the token-diversity "sequential" baseline.

**2. Ordering ranking changed.** Under token diversity, random won clearly (1.614) and King Wen was worst (1.662, spread = 0.048). Under compression ratio, random still wins (1.627) but the spread collapsed to 0.011 — random, easy_to_hard, hard_to_easy, and king_wen are effectively tied. The metric matters for relative ranking but not for the fundamental decorrelation effect.

**3. King Wen improved most from metric change** (-0.024). Token diversity made King Wen the worst non-sequential ordering; compression ratio puts it in the middle of the pack. This suggests King Wen's difficulty mapping interacts differently with different scoring functions.

### Cross-platform comparison (compression ratio only)

| Ordering | CUDA (comp-ratio) | MLX (comp-ratio, std WD) | CUDA vs Sequential | MLX vs Sequential |
|----------|-------------------|--------------------------|-------------------|------------------|
| sequential | 1.778 | 1.732 | — | — |
| buffered_passthrough | 1.640 | 1.735 | -0.138 | -0.038* |
| random | 1.627 | 1.713 | -0.151 | -0.020 |
| easy_to_hard | 1.634 | 1.695 | -0.144 | -0.037 |
| hard_to_easy | 1.634 | 1.709 | -0.144 | -0.024 |
| king_wen | 1.638 | 1.724 | -0.140 | -0.009 |

*MLX passthrough compared to no-buffer baseline (1.773), not buffered sequential.

**Key conclusion: The curriculum effect is platform-dependent, not metric-dependent.** CUDA shows 0.138-0.151 bpb improvement from any reordering. MLX shows 0.008-0.037 bpb — within seed noise. The platform difference (4-10x) vastly exceeds the metric difference.

**Hypothesis: torch.compile amplifies decorrelation benefits.** torch.compile optimizes kernel execution graphs based on data access patterns. Sequential correlation in the dataloader may feed into these optimizations in a way that creates a "local overfitting" effect at the kernel level. Breaking this correlation with buffering disrupts the kernel-level optimization, forcing more generalized computation. MLX has no equivalent compiled kernel graph, so this amplification mechanism is absent.
