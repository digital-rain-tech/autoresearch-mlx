# ADR-006: MLX Curriculum Exploration — Full Experimental Design

**Status**: Planned
**Date**: 2026-03-23
**Hardware**: MacBook Pro, 96 GB unified memory, Apple Silicon (MLX)
**Depends on**: ADR-001 (val_bpb metric), ADR-002 (King Wen LR — not supported), ADR-003 (curriculum ordering — buffer bug on PyTorch), ADR-004 (seed sensitivity), ADR-005 (next steps)

## Motivation

ADR-002 showed King Wen LR modulation hurts training. ADR-003 reframed King Wen as a curriculum ordering strategy, but implementation v1 on PyTorch/CUDA was blocked by a torch.compile tensor-cloning bug. A v2 fix is in progress on the Intel/NVIDIA RTX 2060 machine.

This ADR designs a new, broader experiment on the 96 GB MacBook Pro using the MLX port. MLX does not use torch.compile, so the buffer bug likely does not apply. The 96 GB unified memory also enables model scales (DEPTH=8, 12) impossible on the 6 GB RTX 2060.

Concurrent literature review (arXiv:2511.18903, arXiv:2506.11300, arXiv:2508.15475) revealed critical findings:
1. Standard LR decay suppresses curriculum benefits by up to 44x
2. Compression ratio, lexical diversity, and readability are the strongest difficulty signals
3. Curriculum-as-warmup yields lasting gains of up to 3.5%
4. Model-centric difficulty scoring outperforms human heuristics

This experiment incorporates these findings into a three-phase design that tests **ordering × LR regime × model scale** interactions.

Note: arXiv:2508.15475 (influence-driven curriculum) informed the decision to include loss-based scoring as a Phase 3 difficulty metric comparison. Influence scoring itself is too expensive for a 5-minute budget (requires a surrogate model), but loss-based scoring is a cheap approximation of the same idea: let the model define difficulty rather than a static heuristic.

## Core Research Question

Does data ordering interact with LR regime and model scale to produce meaningful val_bpb improvements under a fixed 5-minute training budget on Apple Silicon?

---

## Phase 0: Instrumentation (prerequisite)

`train.py` hardcodes all hyperparameters as module-level constants (DEPTH, DEVICE_BATCH_SIZE, WARMDOWN_RATIO, FINAL_LR_FRAC) and the seed (`mx.random.seed(42)` at line 392). There is no env var or CLI support. Before any experiment runs, add env var overrides for:

| Env var | Controls | Default |
|---------|----------|---------|
| `AUTORESEARCH_DEPTH` | DEPTH | 4 |
| `AUTORESEARCH_DEVICE_BATCH_SIZE` | DEVICE_BATCH_SIZE | 16 |
| `AUTORESEARCH_WARMDOWN_RATIO` | WARMDOWN_RATIO | 0.5 |
| `AUTORESEARCH_FINAL_LR_FRAC` | FINAL_LR_FRAC | 0.0 |
| `AUTORESEARCH_SEED` | mx.random.seed() | 42 |
| `AUTORESEARCH_CURRICULUM` | Curriculum ordering mode | `none` (no buffering) |

Implementation: simple `os.environ.get()` with type conversion, placed immediately after the hardcoded defaults. When `AUTORESEARCH_CURRICULUM=none`, the training loop uses `make_dataloader` directly (no buffer wrapper). Any other value activates the curriculum buffer.

Note on constant LR: when `WARMDOWN_RATIO=0.0`, the `get_lr_multiplier` function returns 1.0 for all progress values because the condition `progress < 1.0 - WARMDOWN_RATIO` (i.e., `progress < 1.0`) catches everything before the warmdown branch. `FINAL_LR_FRAC` is therefore a no-op under constant LR. This is correct behavior but should not cause confusion — the env var exists for completeness and for intermediate warmdown ratios.

**Phase 0 also includes:**
- Update README.md to document the new env vars and their defaults
- Add structured per-run logging: after each run, append a JSONL record to `experiment_log.jsonl` containing all fields from the "Recorded per run" table below, plus the resolved config (all env var values). This supplements `results.tsv` (which remains the human-readable summary) with a machine-readable artifact for later analysis and reproduction.

---

## Evaluation Protocol (locked before any runs)

### Fixed across all runs

- **Budget:** Fixed wall-clock training time, `TIME_BUDGET = 300s`. Curriculum overhead (difficulty scoring, buffer sorting, refill) counts against this budget. Different orderings may get different step counts — curriculum that wastes steps on overhead is penalized.
- **Seed:** Seed 42 for all first-pass comparisons. Multi-seed (42, 7, 123) only in Phase 3 validation.
- **Evaluation:** `evaluate_bpb()` from `prepare.py`, unchanged. Same validation shard, same `EVAL_TOKENS = 3 * 524288` (~1.5M tokens). Note: ADR-001 references `EVAL_TOKENS = 40 * 524288` (~21M tokens) — the MLX port reduced this, presumably for speed. This means eval-level noise is higher than the original. The Phase 1 noise floor measurement accounts for this implicitly (it measures total variance including eval noise). If the noise floor is uncomfortably large, consider increasing EVAL_TOKENS as a one-time change to prepare.py (requires explicit decision since prepare.py is nominally read-only).
- **Metric:** val_bpb (ADR-001). Lower is better.

### Pre-registered success criteria

- **Meaningful improvement threshold:** val_bpb delta strictly greater than the measured Phase 1 noise floor for that depth/config family. At DEPTH=4, the provisional threshold is 0.04 bpb (from ADR-004, 30-seed range). This threshold is replaced by the actual measured 3-seed range at each depth after Phase 1.
- **Provisional winner (Phase 2):** Any ordering that beats sequential baseline by more than the noise floor under the same depth and LR regime.
- **Confirmed winner (Phase 3):** Must beat matched sequential baseline (same depth, LR regime, seed set, budget) in the 3-seed mean AND in at least 2 of 3 individual seeds.
- **King Wen validation:** Must beat ALL controls (sequential, random, easy_to_hard, hard_to_easy) under the same depth/LR to support the anti-habituation hypothesis.

### Validity guardrails

- **Minimum training threshold:** If total_tokens_M < 70% of sequential baseline at same depth/LR, the run is marked `undertrained` and excluded from winner selection.
- **Overhead alarm:** If curriculum_overhead_seconds > 20% of budget_seconds_consumed, flag for investigation (buffer size may need tuning). Note: the Phase 1d depth selection criterion uses a tighter 15% threshold — that is a go/no-go gate for entering Phase 2, while 20% is a per-run alarm during Phase 2 execution.

### Recorded per run

| Field | Description |
|-------|-------------|
| val_bpb | evaluate_bpb() output |
| depth | Model depth |
| ordering | Curriculum ordering condition |
| lr_regime | `standard` or `constant` |
| seed | Random seed |
| num_steps | Optimizer step count |
| total_tokens_M | steps × TOTAL_BATCH_SIZE / 1e6 |
| budget_seconds_consumed | Total wall clock spent in training loop |
| optimizer_step_seconds | Sum of forward/backward/update dt |
| curriculum_overhead_seconds | Time in difficulty scoring + sorting + buffer refill |
| eval_seconds | Evaluation wall clock |
| startup_excluded_seconds | Data loading + compilation time |
| tokens_per_sec_training | total_tokens / optimizer_step_seconds |
| tokens_per_sec_budget | total_tokens / budget_seconds_consumed |
| peak_vram_mb | MLX peak memory |
| mean_buffer_score_std | Mean standard deviation of difficulty scores within each buffer refill. Diagnostic: measures whether the difficulty metric actually discriminates between batches. If std is near zero, the metric is not providing useful signal for reordering. |

---

## Phase 1: Foundation (~90 min)

### 1a. Establish baselines at three depths

Run unmodified `train.py` at DEPTH=4, 8, and 12 with seed 42.

| Depth | model_dim | n_head | Approx params | DEVICE_BATCH_SIZE |
|-------|-----------|--------|---------------|-------------------|
| 4 | 256 | 2 | ~3M | 16 (start) |
| 8 | 512 | 4 | ~20M | 16 (start) |
| 12 | 768 | 6 | ~50M | 16 (start) |

DEVICE_BATCH_SIZE=16 is a hypothesis, not a default expectation. If step time becomes disproportionately worse at larger depths, test DEVICE_BATCH_SIZE=8 as well. Choose the batch size that gives the best val_bpb under fixed wall-clock, not merely the largest batch that fits in memory.

Records: val_bpb, num_steps, total_tokens_M, peak_vram_mb, dt per step, tokens/sec.

### 1b. Implement and validate curriculum buffer

Add to `train.py`:
- A `curriculum_dataloader` generator wrapping `make_dataloader`
- Buffers 64 micro-batches as MLX arrays
- Scores each batch by compression ratio (gzip token bytes, ratio = compressed_size / original_size; lower ratio = more redundant = "easier")
- Reorders according to `AUTORESEARCH_CURRICULUM` env var
- Yields batches one at a time to the training loop
- Records curriculum_overhead_seconds separately from optimizer step time

Buffer size 64 is a starting point. If refill cost is high, test 32 and 128 as engineering variants — buffer size is a systems parameter, not part of the curriculum hypothesis.

**Critical validation — two controls:**

1. `CURRICULUM=passthrough_buffered` — buffering enabled, no scoring, no sorting, original order. Isolates buffer mechanics.
2. `CURRICULUM=sequential` — buffering + scoring but yield in original order. Isolates scoring overhead.

Both must match unmodified baseline:
- val_bpb within ±0.005 (target tolerance, not automatic pass — inspect loss traces and token counts for anomalies regardless)
- num_steps within ±2
- No sawtooth loss pattern (the ADR-003 v1 bug symptom: loss drops within buffer, spikes at refill boundaries)

If either control fails, debug before proceeding. The buffer implementation is the foundation of everything that follows.

### 1c. Measure noise floor at each depth

Run 3 seeds (42, 7, 123) at each depth with unmodified baseline.

Records: val_bpb range and standard deviation per depth. These replace the provisional 0.04 threshold with measured thresholds per config family.

### 1d. Select larger depth for Phase 2

Choose the larger depth (8 or 12) that satisfies:
- Best baseline val_bpb among larger depths
- Preserves ≥80% of DEPTH=4 effective tokens/sec
- Curriculum overhead (from 1b sequential validation) ≤15% of budget
- DEVICE_BATCH_SIZE determined empirically per 1a

If neither DEPTH=8 nor 12 meets these criteria, Phase 2 runs DEPTH=4 only.

### Phase 1 run count

- 3 baseline runs (3 depths × seed 42) — these double as the seed-42 noise floor runs in 1c
- 2-3 batch size experiments (if DEPTH=8 or 12 needs tuning)
- 2 buffer validation runs (passthrough + sequential, at DEPTH=4)
- 6 additional noise floor runs (3 depths × 2 remaining seeds: 7, 123)
- **Total: ~13-14 unique runs, ~95 min**

### Phase 1 deliverables

- Baseline val_bpb, steps, tokens/sec at DEPTH=4, 8, 12
- Confirmed working curriculum buffer on MLX
- Noise floor (3-seed std and range) at each depth
- Per-depth validity guardrail (70% of baseline steps/tokens)
- Selected larger depth for Phase 2 (or decision to run DEPTH=4 only)

---

## Phase 2: Curriculum × LR Co-Design (~2.5 hours)

### Independent variables

**Orderings (5 conditions):**

| Ordering | What it tests |
|----------|---------------|
| sequential | Baseline — buffer passthrough, no scoring |
| random | "Any reordering helps?" — shuffle buffer randomly |
| easy_to_hard | Classical curriculum — sort ascending by difficulty |
| hard_to_easy | Null/bad control — sort descending by difficulty |
| king_wen | Anti-habituation hypothesis — map King Wen surprise values to difficulty buckets |

Shao Yong was included in ADR-003 but is dropped here. Rationale: ADR-002 showed Shao Yong performed identically to random perturbation (~0.02 bpb, within seed noise per ADR-004). As a highly autocorrelated sawtooth, it is structurally similar to easy_to_hard (monotonic progression). Including it would add 4 runs to Phase 2 without testing a distinct hypothesis. If results from Phase 2 are ambiguous, Shao Yong can be added as a Phase 3 diagnostic.

**King Wen mapping algorithm (concrete specification):**

The King Wen sequence assigns each of the 64 hexagrams a position. To map this to curriculum ordering:

1. Compute the "surprise value" for each position: the absolute difference between consecutive hexagram numbers in the King Wen sequence, normalized to [0, 1]. Position 1 uses the difference from position 64 (wrap-around).
2. Sort the 64 buffered micro-batches by their compression-ratio difficulty score (ascending = easiest first).
3. Rank the 64 King Wen positions by their surprise values (ascending = lowest surprise first).
4. Map by rank: the batch at difficulty rank k is placed at the King Wen position with surprise rank k. High-surprise positions get harder batches; low-surprise positions get easier batches.

This is a rank-order mapping: the King Wen sequence determines the *presentation order* of difficulty-ranked batches, not a linear interpolation of difficulty scores. Two implementers following this algorithm will produce the same ordering given the same input batches.

The King Wen sequence itself (the 64 hexagram numbers in traditional order) must be hardcoded as a constant array. Source: the standard King Wen ordering as documented in the [King Wen AGI Framework](https://github.com/augchan42/king-wen-agi-framework).

**Compression ratio computation (concrete specification):**

For each micro-batch (shape: [DEVICE_BATCH_SIZE, MAX_SEQ_LEN], dtype int32):

1. Convert the MLX int32 array to a Python bytes object: `batch_bytes = np.array(batch).tobytes()`
2. Compress with gzip: `compressed = gzip.compress(batch_bytes)`
3. Ratio: `difficulty = len(compressed) / len(batch_bytes)`

This measures redundancy in token-ID space, not linguistic complexity. Lower ratio = more redundant patterns in token IDs = "easier" (the model can exploit repetitive token patterns more readily). This is fast (~ms per batch) and requires no tokenizer decode step.

Alternative: decode token IDs back to text and gzip the text bytes. This would measure linguistic redundancy more directly but adds `tokenizer.decode()` overhead per batch. Deferred to Phase 3e as the "linguistic compression ratio" variant if compression ratio in token-ID space proves ineffective.

**LR regimes (2 conditions):**

| Regime | WARMDOWN_RATIO | FINAL_LR_FRAC | Rationale |
|--------|---------------|---------------|-----------|
| standard | 0.5 | 0.0 | Current default, matches Intel ADR experiments |
| constant | 0.0 | 1.0 | Literature recommendation for curriculum (arXiv:2511.18903) |

Model averaging (CMA) deferred to Phase 3. Test the simpler version first.

**Known risk:** The literature (arXiv:2511.18903) recommends constant LR *with* model averaging (CMA), not constant LR alone. Without CMA, the model may oscillate around the minimum rather than converging, which could mask real curriculum effects under constant LR. If constant LR performs poorly for ALL orderings (including sequential), this is the likely cause — in that case, add CMA before concluding that constant LR doesn't help curriculum.

**Model sizes (2 conditions):**
- DEPTH=4 (Intel-comparable anchor)
- Larger depth selected from Phase 1

`TOTAL_BATCH_SIZE` is held constant at 2^16 = 65536 tokens across all depths. This means `grad_accum_steps` varies with `DEVICE_BATCH_SIZE` (e.g., 2 at batch 16, 4 at batch 8). This is a deliberate choice: we want to isolate the effect of model capacity (depth) from batch size effects. If a depth comparison appears confounded by batch dynamics, this can be revisited, but the default is to hold total batch size constant.

**Difficulty metric:** Compression ratio only. Literature's strongest performer. If curriculum shows no effect with the best-known metric, testing weaker metrics is not justified. Metric comparison deferred to Phase 3.

### Run matrix

5 orderings × 2 LR regimes × 2 depths = **20 runs**, all seed 42.

At ~7 min per run for DEPTH=4 ≈ **2.5 hours**. Note: larger depths will have longer compilation and possibly slower eval. DEPTH=12 runs may take 10-12 min each. Adjust time estimate after Phase 1 baselines establish actual per-run durations at each depth.

### Winner selection

A condition is a **provisional winner** if:
1. val_bpb is strictly lower than sequential baseline at same depth/LR by more than the Phase 1 noise floor
2. The run is not marked `undertrained`
3. Winner status is provisional until compared against all other controls at the same depth/LR

For King Wen specifically: must beat ALL of random, easy_to_hard, and hard_to_easy at the same depth/LR.

### Analysis priorities (in order)

1. **LR interaction:** Does constant LR amplify curriculum benefits relative to standard warmdown? Compare ordering effect sizes across LR regimes.
2. **Scale interaction:** Does the curriculum effect grow or shrink with depth?
3. **King Wen vs easy_to_hard:** If both beat baseline, which one and by how much? If easy_to_hard beats King Wen, anti-habituation may be a liability.
4. **hard_to_easy diagnostic:** Expected to perform poorly. If it is competitive with easy_to_hard or King Wen, inspect the difficulty metric and learning dynamics before drawing conclusions — this is a diagnostic expectation, not a required outcome.
5. **Constant LR standalone:** Did constant LR help sequential baseline regardless of curriculum? If yes, that is an independently valuable finding.

---

## Phase 3: Deep Dives on Winners (~1-2.5 hours, conditional)

Phase 3 runs only on conditions that met the provisional winner threshold in Phase 2. If nothing wins, skip to 3f.

### 3a. Multi-seed validation (required for any winner)

Run 3 seeds (42, 7, 123) on every provisional winner AND its matched sequential baseline (same depth, LR regime, budget).

A winner is **confirmed** only if:
- Mean val_bpb across 3 seeds is strictly better than mean matched sequential baseline across the same 3 seeds
- The improvement exceeds the Phase 1 noise floor for that depth
- At least **2 of 3 individual seeds** beat matched sequential (prevents one outlier pulling the mean)

Runs: 3 per winner + 3 matched baselines (may overlap with Phase 1 noise floor runs). ~6-9 runs.

### 3b. King Wen as hypothesis class (if King Wen is a confirmed winner)

Test which structural property drives the effect. All at winning depth/LR, seed 42.

| Variant | Construction rule | What it isolates |
|---------|-------------------|-----------------|
| Canonical King Wen | The full traditional 64-element sequence mapped to difficulty buckets | Full hypothesis |
| Blockwise shuffled | Divide the 64 positions into 8 contiguous blocks of 8 in canonical order; shuffle the 8 block positions randomly; preserve within-block order | Global structure matters, local order doesn't |
| Locally perturbed | For each consecutive pair (1,2), (3,4), ..., (63,64), swap with probability 0.5 using a fixed random seed (seed=0) | Exact adjacency vs approximate adjacency |
| Reverse King Wen | Reverse the full 64-element sequence (position 1→64, 2→63, ...) | Directional flow matters |

4 runs, ~30 min. Perturbation rules are locked now to prevent post-hoc tuning.

Note: Phase 3b is locked to buffer_size=64, matching the King Wen sequence length. If buffer size was changed during Phase 1 engineering (e.g., to 32 or 128), Phase 3b must use 64 regardless, with the King Wen mapping padded or truncated appropriately — or buffer size must be restored to 64 for these runs.

### 3c. Curriculum-as-warmup (if easy_to_hard or King Wen confirmed)

Test whether curriculum throughout training is necessary, or whether early exposure is sufficient. Warmup percentage defined by **elapsed training budget time** (wall clock), not steps.

| Variant | Description |
|---------|-------------|
| Warmup 25% | First 75s of budget uses curriculum ordering, remaining 225s sequential |
| Warmup 50% | First 150s curriculum, remaining 150s sequential |
| Full | Curriculum throughout (the Phase 2 winner, for comparison) |

3 runs at winning depth/LR, seed 42. ~21 min.

**Implementation note:** The curriculum_dataloader needs access to the training loop's elapsed budget time to switch from curriculum to sequential ordering mid-run. Implementation approach: pass a shared `total_training_time` reference (or closure) into the curriculum generator, so it can check elapsed time at each yield and stop reordering after the warmup fraction is consumed.

### 3d. Interleaved curriculum (if easy_to_hard is a **confirmed** winner after 3a)

Gated tightly: only run if easy_to_hard survives multi-seed validation. Interleaved = repeated easy-to-hard cycles within each buffer.

Implementation: divide each 64-batch buffer into 4 mini-cycles of 16 batches, each internally sorted easy-to-hard by compression ratio.

2 runs (interleaved + matched easy_to_hard for comparison), seed 42. ~14 min.

### 3e. Difficulty metric comparison (if any curriculum confirmed)

Test whether the difficulty signal matters. All at winning depth/LR/ordering, seed 42.

| Metric | Compute cost | Description |
|--------|-------------|-------------|
| Compression ratio | Cheap (~ms per buffer) | gzip ratio, Phase 2 default |
| Token diversity | Free | unique_tokens / total_tokens, ADR-003's original |
| Loss-based | One forward pass per buffer refill | Model's own mean loss per batch |

3 runs, ~21 min.

Loss-based is the most accurate but most expensive. Frame this as a diagnostic comparison — loss-based becomes the new default only if it yields a clearly larger gain net of overhead. Otherwise compression ratio is the practical winner.

### 3f. If nothing wins in Phase 2

1. **Check constant LR standalone:** Did constant LR help sequential baseline? If yes, the LR finding is independently valuable.
2. **Check hard_to_easy:** If it's not clearly worse, the difficulty metric may not be discriminating. Run one experiment with loss-based scoring to test whether the metric is the problem.
3. **Document the null result:** "Curriculum ordering at DEPTH 4-N, 5-min budget, compression-ratio difficulty, under both standard warmdown and constant LR, shows no effect on val_bpb" is a real contribution that constrains the hypothesis space.

### Phase 3 run estimates

| Scenario | Runs | Time |
|----------|------|------|
| Nothing wins in Phase 2 | 2-3 diagnostic | ~20 min |
| 1 winner, not King Wen | ~9 runs (3a + 3c + 3e) | ~65 min |
| 1 winner, is King Wen | ~13 runs (3a + 3b + 3c + 3e) | ~95 min |
| Multiple winners | ~15-20 runs | ~2-2.5 hours |

### Phase 3 deliverables

- Confirmed winners with multi-seed validation, or confirmed null result
- If King Wen won: which structural property (exact order, block structure, variance profile, directionality) drives it
- If curriculum won: whether warmup-only is sufficient
- Best difficulty metric for the winning ordering
- Decision: proceed to overnight autonomous exploration (Approach B) with best config, or document findings

---

## Post-Phase 3: Transition to Autonomous Exploration

If Phase 3 produces a confirmed winner:
1. Wire the winning configuration (ordering, LR regime, difficulty metric, depth) into `train.py` as the new default
2. Run the autoresearch autonomous loop (program.md) overnight
3. The agent explores freely from this stronger starting point — architecture changes, hyperparameter tuning, further curriculum refinements
4. ~70 experiments over 8 hours of sleep

If Phase 3 produces a null result:
1. Run the autonomous loop from unmodified baseline at the best depth
2. The agent may independently discover curriculum-related improvements or entirely different optimizations
3. The null result is documented and constrains future work

---

## Relationship to Intel/NVIDIA RTX 2060 Experiments

- ADR-003 v2 fix is running concurrently on the Intel machine. Results will provide a cross-platform comparison point at DEPTH=4.
- If both machines produce curriculum results at DEPTH=4, compare: if effects differ, investigate whether the difference is hardware-specific (torch.compile vs MLX) or configuration-specific.
- The Intel machine cannot test DEPTH=8 or 12. Scale-dependent findings from the MacBook Pro are unique to this experiment.
- The Intel machine is useful for tiny CUDA confirmation runs on the 1-2 best configurations from this experiment.

---

## Total Experiment Budget

| Phase | Runs | Time |
|-------|------|------|
| Phase 1: Foundation | ~14 | ~95 min |
| Phase 2: Curriculum × LR | 20 | ~140-200 min (depth-dependent) |
| Phase 3: Deep dives (max) | ~20 | ~150 min |
| **Total (max)** | **~54** | **~6.5-7.5 hours** |

Minimum total (null result): ~36 runs, ~4-5 hours.

---

## References

- ADR-001: val_bpb metric
- ADR-002: King Wen LR schedule — not supported
- ADR-003: Curriculum ordering design, v1 buffer bug
- ADR-004: Seed sensitivity — negligible at DEPTH=4
- ADR-005: Junzi hypothesis status and next steps
- arXiv:2511.18903 — How LR Decay Wastes Your Best Data in Curriculum-Based Pretraining
- arXiv:2506.11300 — Beyond Random Sampling: Curriculum Learning for LM Pretraining
- arXiv:2508.15475 — Influence-driven Curriculum Learning for Pre-training on Limited Data
- arXiv:2601.07239 — Stochastic CHAOS
- Chan (2026) — King Wen AGI Framework, Junzi Alignment hypothesis
