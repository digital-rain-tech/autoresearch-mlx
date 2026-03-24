# ADR-007: Three-State Prototype Postmortem — Why the Triangle Killed Han

**Status**: Complete (negative result)
**Date**: 2026-03-24
**Hardware**: MacBook Pro, 96 GB, Apple Silicon (MLX repo, CPU-only for game)

## Context

ADR-007 (autoresearch) proposed a 3-state Warring States prototype (Qin, Han, Chu) on OpenSpiel as a Pareto-optimal stepping stone before building the full 7-state game. The reasoning was sound on paper: 3 states = simplest non-trivial multi-agent topology, enough strategic depth for King Wen's 64 hexagrams, buildable in ~2 weeks vs months.

We built it and ran a 100-game smoke test per condition. The results killed the approach in under 4 seconds.

## Results

| Condition | Han Survival | Han Win | Han Return |
|-----------|-------------|---------|------------|
| random_all | 9% | 7% | -0.856 |
| scripted_baseline | 7% | 6% | -0.875 |
| kw_hash | 4% | 4% | -0.920 |
| kw_trigram | 5% | 4% | -0.922 |
| kw_sequential | 5% | 4% | -0.913 |
| scrambled_kw | 4% | 4% | -0.920 |
| random_prior | 7% | 7% | -0.867 |

Han dies in ~93-96% of games regardless of agent strategy. King Wen priors actually perform *worse* than random (4-5% survival vs 7-9%).

## Why the Triangle Topology Fails

The failure is **structural, not parametric**. Tuning combat math or starting resources won't fix it. The 3-state triangle removes every mechanism that historically kept Han alive for 223 years (453–230 BC).

### 1. No Buffer Value

Han survived historically as a buffer between Qin and the eastern states. The "lips and teeth" (唇亡齒寒) doctrine meant that destroying Han would expose Qin's flank to Zhao, Wei, and Chu — states that preferred Han alive as a shield.

In a triangle, **Qin and Chu are already adjacent.** Han provides no buffer. Neither neighbor has a strategic reason to keep Han alive.

### 2. No Third-Party Leverage

The Zhanguoce records Han's kings swinging between Su Qin's anti-Qin coalition and Zhang Yi's pro-Qin alliance, extracting concessions from both sides. This only works with **multiple potential partners**.

In a triangle, Han has exactly 2 neighbors. If one attacks, Han can ally with the other — but that other already shares a border with the attacker and faces its own survival pressure. There's no distant state (Qi, Yan) that Han can recruit as a disinterested ally.

### 3. No Combinatorial Diplomacy

With 7 states, Han can:
- Join a 3-state coalition against Qin
- Mediate between Chu and Qi
- Back different internal factions to maintain parallel relationships
- Play sick during diplomatic missions to signal remaining options (as Zhang Cui did in the Zhanguoce)

With 3 states, Han's diplomacy is binary: ally left or ally right. The action space is too small for sophisticated strategy — and too small for 64 hexagrams to map meaningfully.

### 4. Gang-Up Dynamics

In a 3-player simultaneous game, the optimal strategy for the two stronger players is often to eliminate the weakest first, then fight each other. Game theory calls this the "kingmaker problem" — but here Han isn't even a kingmaker. It's just prey.

With 7 states, ganging up on Han means coordinating 2+ states, which creates counter-coalition opportunities for the remaining states. The coordination cost protects the weak.

### 5. King Wen's Anti-Habituation Is Irrelevant

King Wen's hypothesized value is creating **unpredictable** opponents that can't be exploited. But in a triangle where Han dies in 5 rounds, there aren't enough rounds for predictability to matter. You can't exploit a corpse, and you can't be unpredictable when your only viable move is "survive this turn."

## Lesson for Future Game Design

The 3-state prototype was supposed to de-risk the full game. Instead, it proved that **the historical dynamics ARE the hypothesis.** You can't test whether the I-Ching helps Han survive a complex multi-agent environment by removing the complexity that makes survival possible.

This echoes the autoresearch finding that Kuhn Poker (~12 information sets) was too simple for 64 hexagrams (ADR-007, Workstream B revision). The pattern is consistent: **King Wen's structure needs a rich enough substrate to matter.** The minimum viable substrate for Warring States is the 7-state topology.

## Analogy to Curriculum Experiments

| Curriculum (ADR-006) | Game (this ADR) |
|---------------------|-----------------|
| DEPTH=4 model too small for curriculum to help | 3-state game too simple for diplomacy to matter |
| Random shuffle ≈ any ordering (no capacity for curriculum effects) | Random agent ≈ King Wen agent (no room for strategic effects) |
| Buffering itself was the only real effect | The topology itself determines outcomes |
| Needed DEPTH=8+ to test properly | Need 7 states to test properly |

## Decision

Abandon the 3-state prototype. Build the 7-state game directly using:
- The full topology from warringstates-day ADR-002 (Qin, Han, Wei, Zhao, Qi, Chu, Yan)
- Asymmetric starting conditions from the historical state profiles
- The alliance/betrayal mechanics that actually allow Han's survival strategies
- Historical adjacency map (not a regular graph — Yan is isolated, Qin has natural barriers)

The 3-state prototype took ~2 hours to build and ~4 seconds to falsify. That's good Pareto efficiency — we failed fast and learned something real about the minimum viable complexity for the hypothesis.

## What's Salvageable

- OpenSpiel Python game API patterns (registration, simultaneous moves, `_apply_actions`)
- King Wen → action mappings (hash, trigram, sequential) are topology-independent
- Tournament runner with paired seed sweep and drift detection
- The combat/alliance resolution mechanics (need rebalancing for 7 states)
- The scripted agent archetypes map directly to historical state personalities
