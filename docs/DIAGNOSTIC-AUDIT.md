# Diagnostic Audit: What Drives What?

A metric that doesn't change behavior is noise. This document maps every diagnostic to the decision it informs.

## Decision Categories

| Category | Purpose | Example Decision |
|----------|---------|------------------|
| **GATE** | Stop/proceed binary | "Abort merge if CKA < 0.9" |
| **SELECT** | Choose between options | "Use geodesic if curvature > 0.1" |
| **WEIGHT** | Scale a parameter | "Set transfer_strength = density_weight" |
| **DIAGNOSE** | Explain failure (post-hoc) | "Merge failed because X" |
| **ORPHAN** | Currently unused | Just informational |

---

## Metrics That SHOULD Drive Decisions (But Don't Yet)

### 1. Gram Condition Number
- **Computed in**: `gram_aligner.py` → `AlignmentResult.gram_condition_number`
- **Currently**: Logged, not acted upon
- **SHOULD**:
  - **GATE**: If κ > 1e5, require `--full-atlas` or abort
  - **SELECT**: If κ > 1e3, use regularized pseudoinverse

### 2. Intrinsic Dimension per Layer
- **Computed in**: `model_profile.py` → `LayerProfile`
- **Currently**: Reported in profile
- **SHOULD**:
  - **SELECT**: Identify bottleneck layer for transfer focus
  - **GATE**: If ID < 2 everywhere, model may be degenerate

### 3. Sparse Region Indices
- **Computed in**: `sparse_region_locator.py` → `AnalysisResult.sparse_layers`
- **Currently**: Logged
- **SHOULD**:
  - **WEIGHT**: Reduce transfer_strength in sparse_layers
  - **SELECT**: Skip layers in skip_layers from transfer

### 4. Anisotropy Score
- **Computed in**: `manifold_curvature.py` → `ManifoldCurvatureProfile.anisotropy_score`
- **Currently**: Not used
- **SHOULD**:
  - **SELECT**: If anisotropy > threshold, use geodesic distance (not Euclidean)

### 5. Effective Rank vs Hidden Dim
- **Computed in**: `effective_rank.py`
- **Currently**: Reported
- **SHOULD**:
  - **WEIGHT**: null_space_capacity = hidden_dim - effective_rank
  - **GATE**: If null_space_capacity < transfer_size, warn about overwrite

### 6. Coherence Variance Across Prompts
- **Computed in**: `trajectory_coherence.py`
- **Currently**: Not computed as variance
- **SHOULD**:
  - **DIAGNOSE**: High variance → prompt-dependent instability

---

## Metrics That DO Drive Decisions (Keep These)

### Alignment Phase
| Metric | Decision | Location |
|--------|----------|----------|
| `achieved_cka` | GATE: must be ≥ 1.0 - sqrt(eps) | `gram_aligner.py` |
| `misaligned_anchors` | DIAGNOSE: which probes broke | `alignment_diagnostic.py` |

### Transfer Phase
| Metric | Decision | Location |
|--------|----------|----------|
| `density_weights` | WEIGHT: per-direction transfer strength | `transplant.py` |
| `preserved_fraction` | DIAGNOSE: was transfer effective? | `transplant.py` |
| `behavioral_norm` | GATE: should → 0 after projection | `transplant.py` |

### Validation Phase
| Metric | Decision | Location |
|--------|----------|----------|
| `is_degenerate` | GATE: fail if true | `trajectory_coherence.py` |
| `repetition_score` | GATE: fail if > threshold | `trajectory_coherence.py` |

---

## Metrics That Are Pure ORPHANS (Consider Removing or Wiring In)

| Metric | File | Status |
|--------|------|--------|
| `entanglement_entropy` | `entanglement_spectrum.py` | NEW - no decision yet |
| `effective_rank_renyi` | `entanglement_spectrum.py` | NEW - no decision yet |
| `spectral_entropy` | `effective_rank.py` | Logged only |
| `ollivier_ricci` | `ollivier_ricci.py` | Optional, unused |
| `participation_ratio` | various | Logged only |
| `wasserstein_distance` | `sliced_wasserstein.py` | Diagnostic only |

---

## The Two Layer Matchers Problem

We have TWO layer matchers because we haven't decided:

| Question | Constraint | Matcher |
|----------|------------|---------|
| Should layers match in order? | Monotonic | `CrossArchitectureLayerMatcher` (DP) |
| Should layers match optimally? | Unconstrained | `HungarianLayerMatcher` |

**What metric SHOULD decide this?**

Candidates:
1. Architecture similarity (same family → monotonic, different → unconstrained)
2. Layer count ratio (similar → monotonic, different → unconstrained)
3. Per-layer CKA variance (low variance → monotonic works, high → need Hungarian)

**Proposed rule**:
```
IF source.layers == target.layers AND same_architecture_family:
    USE monotonic DP
ELSE:
    USE Hungarian
```

This is a SELECT decision that should be driven by architecture metadata, not a runtime metric.

---

## Action Items

### High Priority (Wire in now)
1. **Gram condition number → GATE** for alignment stage
2. **Sparse layers → WEIGHT** for transfer stage
3. **Effective rank → null capacity GATE**

### Medium Priority (Add decision logic)
4. **Intrinsic dimension → bottleneck layer SELECT**
5. **Anisotropy → geodesic vs Euclidean SELECT**

### Low Priority (Consider removal)
6. Entanglement spectrum - tested, doesn't improve layer matching
7. Ollivier-Ricci - expensive, unclear benefit
8. Spectral entropy - interesting but no decision point

---

## Principle

**Every metric should complete this sentence:**

> "If [METRIC] is [CONDITION], then [ACTION]."

If you can't complete the sentence, the metric is noise.
