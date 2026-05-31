# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Apple Silicon (MLX) port of Karpathy's autoresearch. An autonomous agent repeatedly edits `train.py`, runs a fixed 5-minute training experiment, checks `val_bpb`, and keeps or reverts the change. No PyTorch or CUDA required.

## Commands

```bash
uv sync                          # Install dependencies
uv run prepare.py                # One-time: download data shards + train tokenizer
uv run prepare.py --num-shards 8 # Download fewer shards (for testing)
uv run train.py                  # Run training (5-min budget)
uv run train.py > run.log 2>&1   # Run with output capture (preferred)
grep "^val_bpb:\|^peak_vram_mb:" run.log  # Parse results
```

Data and tokenizer are cached in `~/.cache/autoresearch/`.

## Architecture

Two files, strict boundary:

- **`train.py`** — the ONLY mutable file. Contains model (GPT with sliding window attention, value embeddings, RoPE), custom AdamW optimizer (per-parameter LR groups), curriculum dataloader, and training loop. Hyperparameters are module-level constants in the block following the `Hyperparameters` comment (currently ~line 362; line numbers drift as the file is edited, so locate by name, not number). Runs as a top-level script (no `main()` — intentional for the autoresearch edit-and-run workflow).
- **`prepare.py`** — READ-ONLY. Data download (ClimbMix-400B parquet shards), BPE tokenizer (rustbpe/tiktoken), best-fit-packing dataloader, and `evaluate_bpb()` evaluation. Constants: `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`, `EVAL_TOKENS=3*524288`, `VOCAB_SIZE=8192`.
- **`king_wen_schedules.py`** — standalone helper (King Wen surprise-based LR schedules). NOT imported by `train.py`; the surprise values are inlined into `train.py`. Optional reference only.

### Environment-variable overrides (ADR-006 sweep harness)

`train.py` reads several `AUTORESEARCH_*` env vars at module load so a single committed file can be swept across configs without editing it. This is how the `run_*.log` matrix was generated:

- `AUTORESEARCH_DEPTH`, `AUTORESEARCH_DEVICE_BATCH_SIZE`, `AUTORESEARCH_WARMDOWN_RATIO`, `AUTORESEARCH_FINAL_LR_FRAC`, `AUTORESEARCH_SEED`
- `AUTORESEARCH_CURRICULUM` — batch reordering: `none` (default), `random`, `sequential`, `easy_to_hard`, `hard_to_easy`, `king_wen`, `passthrough_buffered`
- `AUTORESEARCH_DIFFICULTY_METRIC` — e.g. `compression_ratio` (default), used to score batches for curriculum ordering

Example: `AUTORESEARCH_DEPTH=6 AUTORESEARCH_CURRICULUM=king_wen uv run train.py > run_d6_king_wen.log 2>&1`

## Key Metric

**val_bpb** (validation bits per byte) — lower is better. Vocab-size-independent, architecture-independent. See ADR-001 in `docs/adr/`. Baseline on this hardware: check `results.tsv` for the current best.

## Experiment Protocol

Defined in `program.md`. The loop:

1. Edit `train.py` with an experimental idea
2. `git add autoresearch-mlx/train.py && git commit -m "experiment: <description>"`
3. `uv run train.py > run.log 2>&1`
4. Parse results from run.log
5. Log to `results.tsv` (tab-separated: commit, val_bpb, memory_gb, status, description — no commas in descriptions)
6. If improved: `git add autoresearch-mlx/results.tsv && git commit --amend --no-edit`
7. If worse: record discard hash, then `git reset --hard <previous kept commit>`

Branches: `autoresearch/<tag>` (e.g., `autoresearch/mar23`). Never use `git add -A` — this project may live inside a larger repo; always use explicit paths.

## Constraints

- Only edit `train.py`. Never modify `prepare.py`.
- No new dependencies beyond `pyproject.toml`.
- ~7 min per experiment (5 min training + compile/eval). Kill runs exceeding 15 min.
- Memory is a soft constraint. MLX uses unified memory; some increase is acceptable for meaningful val_bpb gains.
- Simplicity criterion: small improvement + ugly complexity = not worth it. Improvement from deleting code = always keep.

## Research Context

`docs/adr/` contains architectural decision records documenting completed experiments:
- ADR-001: Why `val_bpb` is the metric
- ADR-002: King Wen LR modulation hurts training (confirmed harmful)
- ADR-003: Curriculum ordering blocked by torch.compile buffer bug on PyTorch (MLX may not have this issue)
- ADR-004: Seed sensitivity is ~0.04 bpb at DEPTH=4 (30-seed sweep)
- ADR-005: Junzi hypothesis status and next steps
- ADR-006 / 006a: Full curriculum exploration design for this MacBook Pro (the active plan) + CUDA rerun
- ADR-007: 3-state prototype postmortem
- ADR-008: King Wen mathematical analysis

When concluding an experiment that changes the project's direction or confirms/rejects a hypothesis, add or update an ADR rather than only logging to `results.tsv`.
