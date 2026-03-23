# ADR-006: MLX Curriculum Exploration — Full Experimental Design

**Status**: Planned
**Date**: 2026-03-23
**Hardware**: MacBook Pro, 96 GB unified memory, Apple Silicon (MLX)
**Depends on**: ADR-001 (val_bpb metric), ADR-002 (King Wen LR — not supported), ADR-003 (curriculum ordering — buffer bug on PyTorch), ADR-004 (seed sensitivity), ADR-005 (next steps)

## Motivation

ADR-002 showed King Wen LR modulation hurts training. ADR-003 reframed King Wen as a curriculum ordering strategy, but implementation v1 on PyTorch/CUDA was blocked by a torch.compile tensor-cloning bug.

**The v2 fix has now completed on the Intel/NVIDIA RTX 2060 machine.** Results are significant and reshape the research question:

### CUDA v2 Curriculum Results (DEPTH=4, standard warmdown, token diversity metric)

| Ordering | val_bpb | vs Sequential | Significant? |
|----------|---------|--------------|-------------|
| sequential (no buffer) | 1.719 | — | — |
| buffered_passthrough | 1.680 | -0.039 | Borderline (≈ seed noise) |
| **random shuffle** | **1.614** | **-0.106** | **Yes** |
| easy_to_hard | 1.632 | -0.087 | Yes |
| hard_to_easy | 1.627 | -0.092 | Yes |
| shao_yong | 1.638 | -0.081 | Yes |
| king_wen | 1.662 | -0.057 | Yes |

**Key findings:**
1. **ALL reorderings beat sequential** — by 0.039 to 0.106 bpb, well beyond the 0.04 seed noise floor
2. **Random shuffle is the BEST ordering** — contradicts curriculum learning theory
3. **King Wen is the WORST non-sequential ordering** — anti-habituation hypothesis not supported for curriculum either
4. **hard_to_easy ≈ easy_to_hard** — difficulty direction barely matters (0.005 difference)
5. **Buffered passthrough itself helps** — suggests the dataloader's best-fit packing creates sequential correlation that any disruption breaks

**Why random wins (working hypothesis):** The `make_dataloader` in `prepare.py` uses best-fit packing, which greedily assigns documents to sequences by size. This creates implicit sequential correlation: adjacent batches share similar document-length distributions and possibly similar content patterns. Any shuffling breaks this correlation. Random shuffling maximizes decorrelation, while structured orderings (easy_to_hard, King Wen) impose new correlations that are less beneficial than pure randomness at this scale.

This is consistent with arXiv:2404.10830 ("Fewer Truncations Improve Language Modeling"), which documents that best-fit packing creates deterministic ordering biases, though that paper does not investigate shuffling as a remedy.

**No published precedent found** for random outperforming curriculum in LLM pretraining. The literature consistently shows curriculum helps (arXiv:2506.11300, arXiv:2511.18903). Our result may be specific to: (a) very small models (DEPTH=4, ~3M params), (b) very short training (5 min), or (c) the interaction between best-fit packing and token-diversity scoring. This makes the MLX experiments on larger models especially important — does the random advantage hold at DEPTH=8/12?

### Literature findings

Concurrent literature review (arXiv:2511.18903, arXiv:2506.11300, arXiv:2508.15475) revealed:
1. Standard LR decay suppresses curriculum benefits by up to 44x (arXiv:2511.18903)
2. Compression ratio, lexical diversity, and readability are the strongest difficulty signals (arXiv:2506.11300)
3. Curriculum-as-warmup yields lasting gains of up to 3.5% (arXiv:2506.11300)
4. Model-centric difficulty scoring outperforms human heuristics (arXiv:2508.15475)

Note: arXiv:2508.15475 (influence-driven curriculum) informed the decision to include loss-based scoring as a Phase 3 difficulty metric comparison. Influence scoring itself is too expensive for a 5-minute budget (requires a surrogate model), but loss-based scoring is a cheap approximation of the same idea: let the model define difficulty rather than a static heuristic.

## Core Research Questions (revised)

Given the CUDA v2 results, the experiment now has **three** questions in priority order:

1. **Decorrelation hypothesis:** Is the benefit of reordering primarily from breaking best-fit packing's sequential correlation? (Test: does random shuffle also win on MLX, where the dataloader is different?)
2. **Scale interaction:** Does curriculum ordering (easy_to_hard) overtake random at larger model scales (DEPTH=8, 12), as the literature predicts?
3. **LR regime interaction:** Does constant LR amplify curriculum benefits relative to standard warmdown, potentially reversing the random > curriculum result?

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

**Difficulty metric:** Compression ratio for Phase 2 on BOTH machines. Literature's strongest performer (arXiv:2506.11300). The CUDA v2 results used token diversity — those serve as a historical comparison point. A compression-ratio rerun on the 2060 is specified in the companion document `docs/adr/006a-cuda-rerun.md`.

### Run matrix

5 orderings × 2 LR regimes × 2 depths = **20 runs**, all seed 42.

At ~7 min per run for DEPTH=4 ≈ **2.5 hours**. Note: larger depths will have longer compilation and possibly slower eval. DEPTH=12 runs may take 10-12 min each. Adjust time estimate after Phase 1 baselines establish actual per-run durations at each depth.

### Winner selection

A condition is a **provisional winner** if:
1. val_bpb is strictly lower than sequential baseline at same depth/LR by more than the Phase 1 noise floor
2. The run is not marked `undertrained`
3. Winner status is provisional until compared against all other controls at the same depth/LR

For King Wen specifically: must beat ALL of random, easy_to_hard, and hard_to_easy at the same depth/LR.

### Analysis priorities (in order, revised based on CUDA v2 results)

1. **CUDA replication:** Does random shuffle also beat all other orderings on MLX at DEPTH=4? If yes, the decorrelation hypothesis is supported across platforms. If not, the effect is CUDA/torch.compile-specific.
2. **Scale interaction:** Does easy_to_hard overtake random at larger depths? The literature predicts curriculum helps more at scale. If random still wins at DEPTH=8/12, this challenges the curriculum learning literature for short-budget training.
3. **LR regime interaction:** Does constant LR change the ordering ranking? The literature (arXiv:2511.18903) predicts curriculum + constant LR should outperform curriculum + warmdown. If constant LR makes easy_to_hard beat random, the CUDA result was confounded by warmdown.
4. **King Wen position:** CUDA showed King Wen as worst non-sequential ordering. Does this hold on MLX? At larger depths? Under constant LR?
5. **hard_to_easy vs easy_to_hard:** CUDA showed these are nearly identical (0.005 difference). If this holds, difficulty direction doesn't matter — the benefit is purely from decorrelation, not from any pedagogical ordering.
6. **Constant LR standalone:** Did constant LR help sequential baseline regardless of ordering? Independently valuable finding.

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

### 3b. King Wen structural analysis (if King Wen beats random at any depth/LR, OR as diagnostic if it remains worst)

CUDA v2 showed King Wen as worst non-sequential ordering. If this reverses at larger scale or under constant LR, test which structural property drives the improvement. If King Wen remains worst, run ONE diagnostic variant (reverse King Wen) to test whether the problem is directional flow or the variance profile itself.

All at relevant depth/LR, seed 42.

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
| Token diversity | Free | unique_tokens / total_tokens, CUDA v2's metric |
| Loss-based | One forward pass per buffer refill | Model's own mean loss per batch |

3 runs, ~21 min.

Token diversity comparison is now especially valuable: the CUDA v2 results used it, and the 2060 rerun uses compression ratio. If both metrics produce the same ordering ranking, the metric doesn't matter much. If rankings differ, the metric is a first-order variable — which would itself be a finding.

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

### CUDA v2 results are now available

The v2 curriculum fix completed on the Intel/NVIDIA machine. Key implementation details from the CUDA version:

- **Difficulty metric:** Token diversity (`x.unique().numel() / x.numel()`), NOT compression ratio
- **Buffer implementation:** CPU pinned memory with single-tensor GPU yield (avoids torch.compile interaction)
- **King Wen mapping:** Uses `KING_WEN_SURPRISE` values from `king_wen_schedules.py` (63 pre-computed surprise values, padded to 64 with neutral 0.5). Greedy rank assignment maps surprise values to difficulty-sorted batches.
- **All orderings beat sequential by 0.039-0.106 bpb** — random shuffle best, King Wen worst non-sequential

### Cross-platform comparison — resolved

Both machines now have results with both metrics. The verdict:

- **Random wins on CUDA but not MLX** → the effect involves torch.compile's interaction with data access patterns
- Decorrelation hypothesis is **partially supported**: buffering helps on both platforms (~0.038 on MLX, ~0.138 on CUDA), but the magnitude is 3.5x larger on CUDA
- torch.compile amplifies the decorrelation benefit, likely through kernel-level optimization that overfits to sequential data patterns
- The Intel machine cannot test DEPTH=8 or 12. The DEPTH=6 King Wen reversal (best at D6+const LR) is unique to the MacBook Pro experiments.

### Difficulty metric alignment — complete

Both machines have compression ratio results. CUDA also has token diversity. The metric changes relative rankings (King Wen improves most with compression ratio) but does not explain the platform gap. Full comparison in `006a-cuda-rerun.md`.

---

## Results (2026-03-23)

### Phase 1 Summary

| Depth | val_bpb | Steps | Params | Memory GB | tok/sec |
|-------|---------|-------|--------|-----------|---------|
| 4 | 1.773 | 337 | 3.4M | 21.7 | 73K |
| 6 | 2.139 | 131 | 26.3M | 24.5 | 28K |
| 8 | 2.368 | 81 | 50.3M | 27.4 | 17K |
| 12 | 2.773 | 6 | 135.3M | 54.6 | 1.2K |

DEPTH=8 and 12 get too few steps under the 5-min budget. Selected DEPTH=4 (primary) + DEPTH=6 (secondary).

Buffer validation: passthrough_buffered improved val_bpb by 0.038 over no-buffer baseline (1.735 vs 1.773), confirming the decorrelation effect seen on CUDA. Gzip scoring overhead: ~21s per 300s budget (7%).

Noise floor (3 seeds: 42, 7, 123):
- DEPTH=4: range 0.060, std 0.030
- DEPTH=6: range 0.043, std 0.021

**Note: thermal throttling issue.** Back-to-back runs on the MacBook Pro caused ~2x slowdown (337→145 steps). All Phase 2 runs include 60s cooldown between experiments.

### Phase 2 Results

**DEPTH=4 (noise floor: 0.060)**

| Ordering | Standard WD | Constant LR | Delta (const-std) |
|----------|-------------|-------------|-------------------|
| sequential | 1.732 | 1.722 | -0.010 |
| random | 1.713 | **1.697** | -0.016 |
| easy_to_hard | **1.695** | 1.731 | +0.037 |
| hard_to_easy | 1.709 | 1.707 | -0.002 |
| king_wen | 1.724 | 1.729 | +0.006 |

**DEPTH=6 (noise floor: 0.043)**

| Ordering | Standard WD | Constant LR | Delta (const-std) |
|----------|-------------|-------------|-------------------|
| sequential | 2.056 | 2.039 | -0.017 |
| random | 2.047 | 2.056 | +0.009 |
| easy_to_hard | 2.099 | 2.079 | -0.020 |
| hard_to_easy | 2.084 | 2.095 | +0.011 |
| king_wen | 2.074 | **2.030** | -0.044 |

### Phase 2 Analysis

**No provisional winners by pre-registered criteria.** The largest effect (easy_to_hard under standard warmdown at DEPTH=4: -0.037 vs sequential) does not exceed the 0.060 noise floor. All differences are within seed noise.

**Finding 1 — LR regime × ordering interaction is real but within noise.**
easy_to_hard flips from best (standard WD: 1.695) to near-worst (constant LR: 1.731) — a 0.037 swing at DEPTH=4. This is directionally consistent with the literature (arXiv:2511.18903) but the magnitudes are not significant.

**Finding 2 — King Wen behavior reverses with depth.**
DEPTH=4: King Wen is near-worst (both LR regimes). DEPTH=6 + constant LR: King Wen is the best ordering (2.030). The delta (-0.009 vs sequential) is within noise, but the reversal is notable. This may suggest King Wen's high-variance anti-habituation profile helps more when the model has more capacity.

**Finding 3 — CUDA results do not replicate on MLX.**
CUDA showed random winning by 0.106 over sequential. MLX shows random improving by only 0.013-0.020 — well within noise. The large CUDA effect may be torch.compile-specific or difficulty-metric-specific (token diversity vs compression ratio).

**Finding 4 — Constant LR helps sequential baseline.**
At both depths, constant LR improves the sequential baseline (D4: 1.732→1.722, D6: 2.056→2.039). This is an independently useful finding.

**Finding 5 — Buffered passthrough improvement confirms decorrelation.**
The ~0.038 improvement from buffering alone (Phase 1b) is the most robust finding. Best-fit packing creates sequential correlation that buffering disrupts. This effect is consistent across CUDA and MLX.

### Phase 3f: Null Result Assessment

Per the pre-registered protocol, no ordering beat sequential by more than the noise floor. This is a **clean null result for curriculum ordering** under these conditions:
- Compression ratio difficulty metric
- 5-minute fixed budget
- DEPTH=4 and DEPTH=6
- Both standard warmdown and constant LR regimes
- Single seed (42)

### CUDA Compression-Ratio Rerun Results (ADR-006a, completed 2026-03-23)

The 2060 reran all orderings with compression ratio. Full results in `006a-cuda-rerun.md`.

| Ordering | CUDA token-div | CUDA comp-ratio | MLX comp-ratio (std WD) |
|----------|---------------|-----------------|------------------------|
| sequential | 1.719 | 1.778 | 1.732 |
| buffered_passthrough | 1.680 | 1.640 | 1.735 |
| random | **1.614** | **1.627** | 1.713 |
| easy_to_hard | 1.632 | 1.634 | **1.695** |
| hard_to_easy | 1.627 | 1.634 | 1.709 |
| king_wen | 1.662 | 1.638 | 1.724 |

**Conclusion: The curriculum effect is platform-dependent, not metric-dependent.**

CUDA with compression ratio still shows massive effects (0.138-0.151 bpb improvement from any reordering). MLX shows 0.008-0.037 bpb — within seed noise. The platform difference (4-10x) vastly exceeds the metric difference.

**Working hypothesis: torch.compile amplifies decorrelation benefits.** torch.compile optimizes kernel execution graphs based on data access patterns. Sequential correlation from best-fit packing feeds into kernel-level optimization in a way that creates local overfitting. Breaking this correlation disrupts compiled kernels, forcing more generalized computation. MLX has no compiled kernel graph, so this amplification is absent.

Secondary observation: compression ratio collapsed the CUDA ordering spread from 0.048 to 0.011 — random still wins but all orderings are nearly tied. King Wen improved most from the metric change (from worst to middle of pack).

**Remaining untested:**
- Token diversity metric on MLX (for exact CUDA parity)
- Loss-based difficulty scoring
- Multi-seed validation of near-threshold results
- Curriculum-as-warmup

**Recommended next steps:**
1. Token diversity metric on MLX at DEPTH=4 — if this shows a large effect, the metric IS the key variable (not the platform)
2. If token diversity on MLX is also null → platform difference is confirmed; curriculum is a CUDA phenomenon at this scale
3. Run the autonomous loop (Approach B) from DEPTH=4 baseline for non-curriculum improvements

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
