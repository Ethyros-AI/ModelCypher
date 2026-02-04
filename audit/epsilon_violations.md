# Audit: Hardcoded Epsilon Values

> **Audit Date:** 2026-01-30
> **Purpose:** Catalog all hardcoded epsilon values that should use dtype-derived functions from `epsilon_utils.py`.

---

## Summary

The codebase has `epsilon_utils.py` with proper dtype-derived functions:
- `division_epsilon(backend, array)` - sqrt(eps) for safe division
- `regularization_epsilon(backend, array)` - eps^0.75 for regularization
- `safe_log_epsilon(backend, array)` - tiny value for log(x + eps)
- `machine_epsilon(backend, array)` - raw machine epsilon

However, many files still use hardcoded `1e-10` or `1e-8` instead.

---

## Violations by Priority

### P0 - Critical (Affects Numerical Correctness)

#### geodesic_deviation.py

| Line | Code | Should Be |
|------|------|-----------|
| 682 | `if var_curv < 1e-10` | `division_epsilon(backend, array)` |
| 688 | `if var_rate < 1e-10` | `division_epsilon(backend, array)` |
| 703 | `eps = 1e-10` | `division_epsilon(backend, array)` |

**Context:** Used in correlation computation - affects curvature-deviation correlation accuracy.

#### manifold_boundary.py

| Line | Code | Should Be |
|------|------|-----------|
| 194 | `mx.maximum(y_perturbed, 1e-10)` | `safe_log_epsilon()` |
| 477 | `b.log(b.array(r + 1e-10))` | `safe_log_epsilon()` |
| 781 | `b.maximum(u_norm, b.array([1e-10]))` | `division_epsilon()` |
| 789 | `b.maximum(v_norm, b.array([1e-10]))` | `division_epsilon()` |

**Context:** Used in power iteration for spectral norm - affects boundary estimation.

#### self_align.py

| Line | Code | Should Be |
|------|------|-----------|
| 182 | `if final_norm > 1e-10` | `division_epsilon(backend, array)` |
| 189 | `if norms[i - 1] > 1e-10` | `division_epsilon(backend, array)` |
| 202 | `mx.log(probs + 1e-10)` | `safe_log_epsilon()` |

**Context:** Used in comp/phi computation - directly affects the core metric.

---

### P1 - Medium (Affects Derived Metrics)

#### self_consistency/ files

| File | Line | Code | Should Be |
|------|------|------|-----------|
| `invariant_mapping.py` | 210 | `S[j] > 1e-10` | `svd_rank_threshold()` |
| `invariant_mapping.py` | 229 | `S[1] > 1e-10` | `svd_rank_threshold()` |
| `invariant_mapping.py` | 233 | `S_sum > 1e-10` | `division_epsilon()` |
| `invariant_mapping.py` | 235 | `np.log(S_norm + 1e-10)` | `safe_log_epsilon()` |
| `thinking_loop.py` | 184 | `S[j] > 1e-10` | `svd_rank_threshold()` |
| `thinking_loop.py` | 201 | `S_sum < 1e-10` | `division_epsilon()` |
| `thinking_loop.py` | 205 | `np.log(S_norm + 1e-10)` | `safe_log_epsilon()` |
| `geometric_learning.py` | 164 | `S[j] > 1e-10` | `svd_rank_threshold()` |
| `geometric_learning.py` | 205 | `S[1] > 1e-10` | `svd_rank_threshold()` |
| `geometric_learning.py` | 237 | `norm > 1e-10` | `division_epsilon()` |
| `consistency_measure.py` | 102 | `a_norm_val < 1e-10` | `division_epsilon()` |
| `contrastive_geometric_learning.py` | 240 | `S[j] > 1e-10` | `svd_rank_threshold()` |
| `contrastive_geometric_learning.py` | 305 | `norm > 1e-10` | `division_epsilon()` |
| `contrastive_geometric_learning.py` | 411 | `np.linalg.norm(direction) > 1e-10` | `division_epsilon()` |
| `iterative_geometric_learning.py` | 181 | `np.linalg.norm(v2) + 1e-10` | `division_epsilon()` |
| `iterative_geometric_learning.py` | 230, 258 | `S[j] > 1e-10` | `svd_rank_threshold()` |
| `surgical_geometric_alignment.py` | 178, 195 | `S[j] > 1e-10` | `svd_rank_threshold()` |
| `progressive_learning.py` | 289 | `S[j] > 1e-10` | `svd_rank_threshold()` |
| `progressive_learning.py` | 306 | `S_sum > 1e-10` | `division_epsilon()` |
| `progressive_learning.py` | 308 | `np.log(S_norm + 1e-10)` | `safe_log_epsilon()` |

---

### P2 - Low (Non-Core Functionality)

#### adapters/geometry/lora_geometry_diagnostic.py

| Line | Code | Should Be |
|------|------|-----------|
| 229 | `delta_norm < 1e-10` | `division_epsilon()` |
| 328 | `np.linalg.norm(sv_before_arr) + 1e-10` | `division_epsilon()` |
| 339 | `frobenius_before + 1e-10` | `division_epsilon()` |
| 502 | `diff_norm < 1e-8` | `division_epsilon()` |
| 585 | `before_norm + 1e-10` | `division_epsilon()` |

#### adapters/geometry/hash_analyzer.py

| Line | Code | Should Be |
|------|------|-----------|
| 411 | `std_i > 1e-10 and std_j > 1e-10` | `division_epsilon()` |
| 538 | `S_arr[1:] + 1e-10` | `division_epsilon()` |

#### experiments/ files

| File | Line | Code |
|------|------|------|
| `cross_model_universality.py` | 135, 154 | `eps = 1e-8` |
| `geometric_guardrails.py` | 336 | `max(precision + recall, 1e-8)` |
| `cross_model_alignment.py` | 354 | `max(target_native_accuracy, 1e-8)` |
| `jailbreak_detection.py` | 200 | `max(precision + recall, 1e-8)` |

#### Other files

| File | Line | Code |
|------|------|------|
| `cli/commands/entropy.py` | 414 | `mx.log(p + 1e-10)` |
| `fundamental_constants.py` | 251, 439 | `abs(denominator) < 1e-10` |
| `trajectory_complexity.py` | 310 | `eps = 1e-10` |
| `lora_memory_store.py` | 1093, 1101 | `max(total_original, 1e-10)` |
| `transplant_weight_processor.py` | 1955 | `max(min_scale, 1e-10)` |
| `behavioral_analyzer.py` | 1027 | `variance > 0 else 1e-10` |

---

## Proper Usage Examples

### Already Correct: relative_representation.py:214

```python
# Use dtype-derived epsilon, not arbitrary 1e-10
```

This file was already updated to use proper epsilon utilities.

---

## Statistics

| Priority | Count | Description |
|----------|-------|-------------|
| P0 - Critical | 9 | Core geometry/alignment code |
| P1 - Medium | 19 | Self-consistency modules |
| P2 - Low | 17 | Adapters, experiments, utilities |
| **Total** | **45** | Hardcoded epsilon violations |

---

## Remediation Plan

### Phase 1: Critical Files (This Audit)

Fix these files:
1. `geodesic_deviation.py` (3 violations)
2. `manifold_boundary.py` (4 violations)
3. `self_align.py` (3 violations)

### Phase 2: Self-Consistency Modules (Future)

Fix `src/modelcypher/core/use_cases/self_consistency/*.py` (19 violations)

### Phase 3: Adapters and Experiments (Future)

Fix remaining files (17 violations)

---

## Import Pattern for Fixes

```python
from modelcypher.core.domain.geometry._primitives.epsilon_utils import (
    division_epsilon,
    safe_log_epsilon,
    svd_rank_threshold,
)
```

For platform-specific code without backend:

```python
# Use sqrt(machine_epsilon) directly
eps = float(machine_epsilon("float32")) ** 0.5
```
