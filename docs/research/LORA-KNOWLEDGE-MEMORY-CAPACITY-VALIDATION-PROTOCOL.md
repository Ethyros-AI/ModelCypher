# LoRA Knowledge-Memory Capacity Validation Protocol

**Status:** Predictions pre-registered; experiments pending  
**Promotion state:** `[EXPLORATORY]`  
**Date:** 2026-03-05  
**Primary external target:** Back et al. (2026), arXiv:2603.01097, "Understanding LoRA as Knowledge Memory: An Empirical Analysis"

Use with:
- `docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`
- `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`
- `docs/research/geometric_capacity_paper_experiment_matrix.md`
- `docs/research/lora_spectral_scale_bound.md`
- `src/modelcypher/core/domain/training/geometric_lora.py`
- `src/modelcypher/core/domain/geometry/null_space_accessibility.py`
- `src/modelcypher/core/domain/geometry/channel_projector.py`

---

## 1. Purpose

This protocol treats Back et al. as **empirical phenomenology**, not as
mechanistic confirmation.

The paper reports:
- rank-scaled capacity with finite saturation,
- low-rank parameter-efficiency peaks,
- routing failure in multi-LoRA systems,
- merge interference as the number of merged adapters increases,
- hybrid LoRA + external-context gains on long, multi-hop tasks.

ModelCypher already contains candidate geometric mechanisms for these effects.
This document converts those candidate mechanisms into **pre-registered
predictions with falsifiers**.

Core question:

```text
Do the paper's observed LoRA memory phenomena collapse onto geometry_state once
rank, scale, parameterization, routing, and merge interference are measured
explicitly?
```

This is the primary Area 3 capacity-validation experiment.
Areas 1-2 provide instrumentation and reproduction infrastructure.
Area 4 is downstream system architecture if the predictions survive.

---

## 2. Required Claim Form

All claims in this protocol use the repository-wide contract:

```text
observable = f(geometry_state, architecture_state, scale_state, precision_state, measurement_operator)
```

Registered state variables for this protocol:

- `geometry_state`
  - `tail_dims_i`
  - `sigma_k_i`
  - `spectral_gap_i`
  - `utilized_tail_dims_i = min(r_cap_i, tail_dims_i)`
  - `null_rank_i`
  - `condition_number_i`
  - `collision_score`
  - `grassmann_geodesic_distance`
  - `overlap_fraction`
- `architecture_state`
  - base model family
  - target module set
  - adapter parameterization
  - router type
  - merge operator
  - document partition scheme
- `scale_state`
  - global rank cap `r_cap`
  - trainable parameter count
  - applied per-layer scale ratios
  - number of merged adapters `N`
  - training-token budget
  - context budget
- `precision_state`
  - dtype
  - quantization operator state
  - attention kernel implementation
- `measurement_operator`
  - benchmark score
  - saturation-point estimator
  - parameter-efficiency estimator
  - router recall estimator
  - merge-loss estimator
  - subspace-overlap estimator

Primary scope:
- full-precision or reduced full-precision (`bf16`/`fp16`) replication first
- no promotion beyond `[EXPLORATORY]` without an explicit follow-up precision pass
  under quantization, because `precision_state` is part of the claim form

---

## 3. Mechanism Hypotheses

### H1. Saturation is governed by utilized tail capacity, not rank alone

Mechanism:

```text
T_sat = g(sum_i utilized_tail_dims_i, architecture_state, precision_state, measurement_operator)
```

Where:

```text
utilized_tail_dims_i = min(r_cap_i, tail_dims_i)
tail_dims_i = full_rank_i - floor(shannon_effective_rank_i)
```

Interpretation:
- rank is not the cause,
- rank is a control on how much structural tail capacity can be utilized,
- saturation occurs when useful tail capacity is exhausted under the chosen
  architecture and precision state

### H2. Apparent efficiency curves are partly scale-safety curves

Mechanism:

```text
failure_or_coherence_loss = h(max_i scale_ratio_i, architecture_state, precision_state)
scale_ratio_i = ||Delta_i||_2 / sigma_k_i
```

Interpretation:
- if standard LoRA runs exceed the spectral safety bound, measured "capacity"
  is contaminated by scale-induced degradation rather than pure storage limit

### H3. Merge interference is a subspace-collision phenomenon

Mechanism:

```text
merge_loss = q(collision_score, N, merge_operator, router_state, precision_state)
```

Registered collision operator:

```text
collision_score =
mean_{a<b} overlap_fraction(row(P_null Delta_a), row(P_null Delta_b))
```

with supporting diagnostics:
- pairwise Grassmann geodesic distance,
- pairwise principal-angle spectra,
- overlap with target available basis

Interpretation:
- adapters interfere when their deltas occupy overlapping accessible directions
- this is distinct from router failure

### H4. Norm-bounded Cayley parameterization changes capacity-per-parameter

Mechanism:

```text
eta_mem = T_sat / n_trainable_params
eta_mem = k(parameterization, geometry_state, architecture_state, precision_state)
```

Interpretation:
- if standard LoRA wastes effective rank through zero-product initialization and
  unbounded growth, NB-LoRA should shift the usable capacity frontier at matched
  parameter budget

### H5. Multi-LoRA has two regimes, not one

Mechanism:

```text
Perf(N) = CoverageGain(router_state, N) - InterferenceLoss(collision_score, N)
```

Two registered regimes:

1. **Oracle / pure merge regime**
   - routing error removed
   - `CoverageGain = 0`
   - prediction reduces to monotone non-improvement with increasing `N`

2. **Practical routing / recall regime**
   - routing miss rate non-zero
   - `CoverageGain` may exceed `InterferenceLoss` for small `N`
   - `Top-k` can outperform `Top-1` when recall gains dominate

This is the registered explanation for why:
- merge-only studies can peak at `N = 1`,
- while long-document multi-hop tasks can still show `Top-3 > Top-1`

---

## 4. Registered Observables

Primary observables:

- `T_sat(tau)`
  - largest tokenized knowledge load whose score remains above the paper-matched
    threshold `tau`
- `eta_mem`
  - `T_sat / n_trainable_params`
- `scale_ratio_i`
  - per-layer spectral safety ratio
- `base_preservation`
  - paper-task score on unedited/base capability probes after adapter attach or merge
- `router_recall@k`
  - whether the relevant adapter set is contained in top-`k`
- `merge_loss`
  - score drop from oracle best single adapter to merged adapter set
- `collision_score`
  - mean overlap in projected adapter row spaces
- `grassmann_geodesic_distance`
  - pairwise subspace distance for projected adapter deltas

Supporting observables:

- `tail_dims_i`
- `sigma_k_i`
- `spectral_gap_i`
- `null_rank_i`
- `condition_number_i`
- `behavioral_preserved_fraction`
- `behavioral_cosine_similarity`
- `principal_angle_mean`
- `principal_angle_max`

---

## 5. Commensurability Rules

Cross-run comparison is valid only if all of the following hold:

1. Same benchmark split and same scoring operator within each comparison.
2. Same base-model family within each primary comparison.
3. Same target-module family unless the target set is itself the manipulated axis.
4. Same tokenization regime for the compared runs.
5. Same context budget for routing and hybrid comparisons.
6. Same precision state within each primary comparison.
7. Same parameter-budget accounting rule for all efficiency comparisons.

Cross-family conclusions are exploratory until the same sign survives separately
within each registered family.

---

## 6. Experimental Arms

To separate mechanisms, we do not compare "paper style" against a single
"everything geometric" bundle only. We register layered intervention arms.

### B0. Paper-style baseline

- fixed global rank sweep
- standard LoRA parameterization
- paper-matched `alpha` schedule where possible
- paper-matched router and merge operators

Purpose:
- reproduce the external phenomenology on our run surface

### G1. Spectral-scale arm

- same as B0
- enforce per-layer geometric scale bounds

Purpose:
- isolate whether safety alone shifts apparent capacity and efficiency

### G2. Tail-capacity arm

- same as G1
- per-layer rank uses:

```text
r_i = min(r_cap, tail_dims_i)
```

Purpose:
- test whether utilized tail capacity predicts saturation better than raw global rank

### G3. NB-LoRA arm

- same as G2
- replace standard LoRA parameterization with norm-bounded Cayley NB-LoRA

Purpose:
- test whether parameterization changes capacity-per-parameter at matched safe budget

### M0. Oracle merge arm

- correct adapters supplied directly
- no routing uncertainty
- vary `N`

Purpose:
- isolate pure merge interference

### M1. Practical routing arm

- embedding-based routing or equivalent paper-style router
- evaluate `Top-1`, `Top-3`, and variable `N`

Purpose:
- separate recall benefit from merge interference

---

## 7. Area Structure

### Area 1. Reproduction Infrastructure

Goal:
- rebuild the paper's phenomenology under commensurable settings

Required datasets:
- PhoneBook / CounterFact class tasks for capacity
- PaperQA class tasks for synthetic-format, routing, and merge analysis
- NarrativeQA / QuALITY class tasks for long-document multi-hop and hybrids

Required model families:
- primary: Qwen-family and Llama-family models matching the paper when available
- fallback local models may be used for smoke tests only
- fallback runs do not count for claim promotion

### Area 2. Supporting Measurements

Before any benchmark run, emit:
- per-layer geometry table
  - `tail_dims_i`, `sigma_k_i`, `spectral_gap_i`, `null_rank_i`, condition number
- per-adapter safety table
  - `scale_ratio_i`, delta row rank, behavioral preservation metrics
- per-pair adapter collision table
  - overlap fraction, Grassmann distance, principal angles
- router table
  - top-`k` recall, miss type, relevant-chunk coverage

These measurements are infrastructure, not conclusions.

### Area 3. Primary Capacity Validation

This is the main experiment family.

Tracks:

1. **Capacity sweep**
   - reproduce Q1-Q3 style rank/load curves under B0, G1, G2, G3

2. **Efficiency sweep**
   - compare `eta_mem` across the same arms

3. **Oracle merge interference**
   - reproduce Q10-Q11 style merge-only degradation under M0

4. **Routing + recall tradeoff**
   - reproduce Q9 and the `Top-1` vs `Top-3` split under M1

5. **Long-document hybrid secondary pass**
   - reproduce Q12-Q14 only after Tracks 1-4 are instrumented

### Area 4. Downstream Architecture

Only if Area 3 survives:

- add pre-merge collision diagnostics to adapter composition
- replace fixed `Top-k` routing with recall/collision-aware `N` selection
- treat LoRA memory service as a geometry-measured persistent memory system
  rather than a generic adapter store

---

## 8. Pre-Registered Predictions and Falsifiers

### P-LKM-1. Utilized tail capacity predicts saturation points

**Claim state:** `[EXPLORATORY]`

Prediction:
- within a fixed family and fixed precision state, median `T_sat` is
  non-decreasing in total utilized tail capacity
- once `r_i >= tail_dims_i` for all targeted layers, further raw rank increase
  does not create a new geometric cause for additional capacity

Directional expectation:

```text
Spearman(T_sat, sum_i utilized_tail_dims_i) > 0
```

Falsifier:
- within any registered family, `Spearman(T_sat, sum_i utilized_tail_dims_i) <= 0`
  under safe-scale runs, or
- saturation keeps improving after all targeted layers have reached their
  tail-capacity ceiling

### P-LKM-2. Spectral scale bounds shift apparent efficiency curves

**Claim state:** `[EXPLORATORY]`

Prediction:
- unsafe paper-style runs show degraded coherence and/or base preservation when
  `max_i scale_ratio_i > 1`
- enforcing the spectral bound shifts the usable efficiency curve relative to B0

Directional expectation:

```text
Corr(max_i scale_ratio_i, coherence_loss) > 0
```

Falsifier:
- coherence loss is not positively associated with bound violation, or
- G1 produces no measurable shift in the efficiency frontier relative to B0

### P-LKM-3. Merge interference is predicted by subspace collision

**Claim state:** `[EXPLORATORY]`

Prediction:
- in oracle merge runs, larger `collision_score` implies larger `merge_loss`
- equivalently, smaller Grassmann distance implies larger merge degradation

Directional expectation:

```text
Corr(collision_score, merge_loss) > 0
Corr(grassmann_geodesic_distance, merge_loss) < 0
```

Falsifier:
- merge loss does not track collision metrics in sign, or
- oracle `N > 1` systematically improves performance without a corresponding
  reduction in collision

### P-LKM-4. NB-LoRA changes capacity-per-parameter

**Claim state:** `[EXPLORATORY]`

Prediction:
- at matched safe scale and matched parameter budget, G3 improves either
  `T_sat`, `eta_mem`, or both relative to G2

Directional expectation:

```text
median(eta_mem_G3 - eta_mem_G2) > 0
```

Falsifier:
- `median(eta_mem_G3 - eta_mem_G2) <= 0` in every registered family

### P-LKM-5. Two-regime `N` law

**Claim state:** `[EXPLORATORY]`

Prediction:
- under M0 (oracle merge), performance is non-improving with larger `N`
- under M1 (practical routing), `Top-3` can exceed `Top-1` when router recall
  improves enough to offset collision cost

Directional expectation:

```text
Perf_M0(N + 1) - Perf_M0(N) <= 0
Perf_M1(3) - Perf_M1(1) may be > 0
```

Falsifier:
- M0 improves with larger `N`, or
- M1 never shows a positive `Top-3 - Top-1` regime despite measurable
  `router_recall@3 > router_recall@1`

---

## 9. Experiment Design

### 9.1 Capacity Sweep

Goal:
- reproduce the paper's rank/load saturation curves under B0, G1, G2, G3

Manipulated axes:
- global rank cap `r_cap`
- knowledge load
- arm type

Held fixed within each comparison:
- base family
- benchmark split
- target module set
- precision state
- token budget

Primary outputs:
- `capacity_curve.json`
- `saturation_points.json`
- `geometry_table.json`
- `falsifier_outcome.json`

### 9.2 Parameter-Efficiency Sweep

Goal:
- determine whether low-rank efficiency peaks survive after scale safety and
  parameterization are controlled

Primary outputs:
- `efficiency_curve.json`
- `trainable_parameter_table.json`
- `scale_ratio_table.json`
- `decision.json`

### 9.3 Oracle Merge Interference

Goal:
- measure pure merge loss with routing removed

Manipulated axes:
- `N`
- merge operator
- arm type for source adapters

Primary outputs:
- `merge_loss_curve.json`
- `collision_metrics.json`
- `principal_angle_table.json`

### 9.4 Practical Routing

Goal:
- decompose performance into router recall and merge interference

Manipulated axes:
- router type
- `N`
- merge operator

Primary outputs:
- `router_metrics.json`
- `topk_performance.json`
- `coverage_vs_interference.json`

### 9.5 Long-Document Secondary Pass

Goal:
- test whether the two-regime law explains the paper's long-document results

Condition for running:
- Areas 1-2 complete
- P-LKM-3 and P-LKM-5 have measurable signal on shorter, controlled tasks

Primary outputs:
- `longdoc_hybrid_results.json`
- `top1_top3_decomposition.json`

---

## 10. Measurement Operators To Reuse

Existing code already provides most of the needed operators:

- `src/modelcypher/core/domain/training/geometric_lora.py`
  - `tail_dims`
  - `sigma_k`
  - structural rank terms
- `docs/research/lora_spectral_scale_bound.md`
  - spectral safety ratio logic
- `src/modelcypher/core/domain/geometry/null_space_accessibility.py`
  - projected behavior
  - principal-angle and Grassmann diagnostics
- `src/modelcypher/core/domain/geometry/channel_projector.py`
  - shared null-space projection surface for multi-channel composition

Important honesty constraint:
- current adapter merge code does **not** yet implement the registered
  collision-gating mechanism
- this protocol therefore includes both:
  - measurements that already exist,
  - measurements that must be wired into the experiment harness

---

## 11. Artifact Bundle

Each run directory must contain:

- `config.json`
- `geometry_table.json`
- `scale_ratio_table.json`
- `capacity_curve.json`
- `efficiency_curve.json`
- `router_metrics.json` when routing is involved
- `collision_metrics.json` when merge is involved
- `raw_scores.jsonl`
- `falsifier_outcome.json`
- `decision.json`

Recommended root:

```text
results/lora_memory_capacity_validation/<run_id>/
```

---

## 12. Decision Discipline

A claim from this protocol is not promotable unless:

1. the registered observable is emitted,
2. the registered directional prediction is evaluated directly,
3. no registered falsifier is triggered,
4. the sign survives within each registered model family,
5. the precision follow-up is completed before any claim is promoted beyond
   full-precision exploratory status

If results split across families:
- classify as `[MECHANISM_UNDERSPECIFIED]` unless the divergence was
  pre-registered in the architecture or scale term

If the measurement operator degenerates:
- classify as `[MEASUREMENT_INVALID]`

Do not write "partially confirmed."

---

## 13. Immediate Next Step

Build Area 1-2 first.

Minimum implementation sequence:

1. Reproduce B0 on one paper-matched capacity task.
2. Emit geometry and scale tables for the same run.
3. Add G1 and test P-LKM-2.
4. Add G2 and test P-LKM-1.
5. Add oracle merge collision logging and test P-LKM-3.
6. Add G3 and test P-LKM-4.
7. Only then run long-document `Top-1` vs `Top-3` and test P-LKM-5.

This keeps Area 3 primary and prevents architecture work from outrunning
measurement.
