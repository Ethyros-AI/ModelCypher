# Geometric Conjectures and Falsification Protocol

**Updated:** 2026-02-16

---

## Purpose

This document defines how ModelCypher evaluates major geometric claims without
guessing, folklore, or narrative drift.

Core rule:
- **Mechanism claims must be stated as equations.**
- **Equations must map to measurable observables.**
- **Observables must have pre-registered pass/fail criteria.**

No claim graduates based on intuition, single runs, or "looks promising."

---

## Mechanism Contract

For fixed parameters and fixed input, transformer inference is deterministic:

```text
h_L = F_theta(prefix)
logits = W_out h_L + b
```

Softmax and cross-entropy are readout/accounting tools:

```text
loss = log(sum_j exp(logit_j)) - logit_correct
```

They do not replace the geometric mechanism.

---

## Claim Status Taxonomy

See [EVIDENCE-TAXONOMY.md](../EVIDENCE-TAXONOMY.md) for the full 5-label system used project-wide.

| Status | Meaning |
|--------|---------|
| `[PROVEN]` | Theorem-level result with assumptions stated and checked in scope |
| `[VALIDATED]` | Null-hypothesis tested, reproduced across multiple settings/models (formerly `SUPPORTED`) |
| `[EMPIRICAL]` | Measured and reproducible, but not falsification-tested |
| `[CONJECTURAL]` | Theoretically motivated hypothesis with insufficient evidence (formerly `OPEN`) |
| `[DISPROVEN]` | Tested and rejected; pre-registered rejection condition met (formerly `FALSIFIED`) |

All broad claims must explicitly carry one of these labels.

---

## Conjecture Register

### C1: Lifted Geometry Reduces Constraint Intersections

**Status:** `[CONJECTURAL]`

**Statement:** For a problem family represented as constraint manifolds
`{C_i}` in ambient dimension `d`, a learned lift `Phi_d` increases
transversality margin and reduces harmful manifold intersections relative to
lower-dimensional embeddings.

**Primary observables:**
- Pairwise transversality margin:
  `tau_ij = min ||J_i(x)^T J_j(y)||_F` on matched support points
- Intersection incidence estimate on sampled neighborhoods
- Geodesic detour ratio between feasible points

**Pre-registered pass condition:**
- Across problem sizes `n` and seeds, lifted setting shows strictly better
  transversality statistics with confidence intervals excluding zero effect.

**Falsification condition:**
- No measurable transversality improvement versus control across the registered
  size sweep.

---

### C2: Lifted Geometry Improves Optimization Scaling

**Status:** `[CONJECTURAL]`

**Statement:** If C1 holds for a task family, optimization cost scaling
`T(n, d)` improves with lift dimension `d` in a reproducible way.

**Primary observables:**
- Runtime to target quality: `T(n, d)`
- Iterations to target quality: `K(n, d)`
- Fitted complexity exponent from `log T` vs `log n`

**Pre-registered pass condition:**
- Difference in fitted exponent between lifted and control settings is
  negative with confidence interval excluding zero.

**Falsification condition:**
- Exponent does not improve under lift, or quality target cannot be reached
  without increased compute scaling.

**Important scope note:**
- This is not a blanket "P=NP" claim. It is a task-family scaling claim under
  explicit model class, lift construction, and quality targets.

---

### C3: Geometric Support Beats Softmax Confidence for Abstention

**Status:** `[CONJECTURAL]`

**Statement:** Distance-to-support metrics provide better OOD/hallucination
detection than softmax confidence for the same model.

**Primary observables:**
- Geometric support score:
  manifold distance, local density, and trajectory consistency
- Softmax confidence baselines:
  max probability, entropy, margin
- Selective risk curves and AUROC/AUPRC for error detection

**Pre-registered pass condition:**
- Geometric score dominates softmax baselines on selective risk and AUROC
  across all registered domains.

**Falsification condition:**
- Softmax baseline matches or exceeds geometric support score on the primary
  metrics in the majority of registered domains.

---

### C4: Stop Certificate Outperforms Val-Loss Plateau

**Status:** `[CONJECTURAL]`

**Statement:** A geometric stop certificate predicts "no further meaningful
improvement" better than validation-loss plateau heuristics.

Certificate components:
- Stationarity term: `||P^(1/2) grad L_train||`
- Max directional validation gain bound
- Validation uncertainty bound
- Trajectory-class worst-case check

**Primary observables:**
- Post-stop delta on held-out benchmark suite
- Overtraining incidence after stop signal
- Compute spent after true performance saturation

**Pre-registered pass condition:**
- Certificate stops closer to true saturation with lower overtraining incidence
  and lower wasted compute than val-loss-only stopping.

**Falsification condition:**
- Certificate does not improve saturation timing or compute efficiency relative
  to val-loss-only stopping.

---

## Standard Experimental Protocol

### 1. Pre-registration

Before running:
- task family and data generators
- model families and scales
- lift constructions and controls
- seeds
- primary/secondary metrics
- pass/fail rules for each conjecture

No metric substitution after results are visible.

### 2. Controlled Axes

Change one axis per run:
- geometry lift
- optimizer/controller
- stop rule
- abstention policy

Everything else stays fixed.

### 3. Reporting Artifacts

Each run must emit:
- raw per-seed measurements
- fitted scaling parameters with confidence intervals
- mechanism traces per epoch/step
- pass/fail decision per conjecture with cited criterion

Required files:
- `config.json`
- `full_results.json`
- `analysis.json`
- `decision.json` (explicit conjecture verdicts)

### 4. Decision Rule

A conjecture can move from `[CONJECTURAL]` to `[VALIDATED]` only when:
- all pre-registered primary metrics pass
- result reproduces across registered families/scales
- confidence intervals support sign and magnitude claims

A conjecture moves to `[DISPROVEN]` immediately when its rejection condition is
met. No silent re-interpretation.

---

## Immediate Next Matrix

### M1: C3 (Abstention) first

Compare on same checkpoints:
- softmax confidence abstention
- geometric support abstention

Output:
- selective risk
- AUROC/AUPRC
- failure slices by trajectory class

### M2: C4 (Stopping) second

Compare:
- val-loss plateau stopping
- geometric stop certificate

Output:
- post-stop benchmark delta vs compute
- overtraining incidence

### M3: C1/C2 (Lift + Scaling) third

Task families with size sweeps:
- SAT variant
- TSP variant
- structured symbolic reasoning family

Output:
- transversality/intersection metrics
- scaling exponent differences

---

## Anti-Patterns (Disallowed)

- Claiming universal tractability from a single family
- Using only averaged metrics when class-level failures are visible
- Moving thresholds after observing outcomes
- Replacing failed conjectures with narrative reinterpretation

---

## Summary

ModelCypher treats broad claims as engineering hypotheses on geometric
mechanisms. Every claim must survive the same pipeline:

`Equation -> Observable -> Protocol -> Verdict`

No guesses. No vibes. No status without falsification pressure.
