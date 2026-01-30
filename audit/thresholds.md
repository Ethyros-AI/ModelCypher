# Audit: Hardcoded Numeric Thresholds

> **Audit Date:** 2026-01-30
> **Purpose:** Catalog all arbitrary numeric thresholds that impose discrete decisions on continuous metrics.

---

## Summary

This audit identifies places where arbitrary numeric thresholds create binary decisions from continuous values. These thresholds are often:
- Not dtype-derived (should use `sqrt(eps)` or similar)
- Not data-derived (should come from empirical distribution)
- Not documented with rationale

---

## Category 1: Pass/Fail Thresholds

### validate_phi_proxy.py:325

```python
CORRELATION_THRESHOLD = 0.7
passed = correlation >= CORRELATION_THRESHOLD
```

| Property | Value |
|----------|-------|
| **File** | `scripts/validate_phi_proxy.py` |
| **Line** | 325-326 |
| **Threshold** | 0.7 |
| **Used For** | Pass/fail determination for phi proxy validation |
| **Issue** | Arbitrary cutoff - why 0.7 and not 0.6 or 0.8? |
| **Remediation** | Report continuous correlation value; remove binary pass/fail |
| **Priority** | P1 - Medium |

---

## Category 2: Uncertainty/Confidence Thresholds

### local_inference.py:944-945

```python
entropy_threshold: float = 0.7,
eigenscore_threshold: float = 0.6,
```

| Property | Value |
|----------|-------|
| **File** | `src/modelcypher/adapters/local_inference.py` |
| **Lines** | 944-945 |
| **Thresholds** | 0.7 (entropy), 0.6 (eigenscore) |
| **Used For** | Inference decision gates |
| **Issue** | Default values not data-derived; appear arbitrary |
| **Remediation** | Document rationale or derive from data; expose as user parameters |
| **Priority** | P2 - Low (already user-configurable) |

---

## Category 3: Statistical Thresholds

### cross_model_alignment.py:211, 328

```python
threshold_idx = int(0.95 * len(sorted_harmless))
```

| Property | Value |
|----------|-------|
| **File** | `src/modelcypher/experiments/cross_model_alignment.py` |
| **Lines** | 211, 328 |
| **Threshold** | 0.95 (95th percentile) |
| **Used For** | Threshold selection for classification |
| **Issue** | Standard statistical cutoff, but rationale not documented |
| **Remediation** | Add comment explaining why 95% (standard practice for outlier detection) |
| **Priority** | P2 - Low |

---

## Category 4: Algorithm Parameters

### compression/ranking_optimizer.py:169

```python
beta = 0.9  # Momentum coefficient
```

| Property | Value |
|----------|-------|
| **File** | `src/modelcypher/core/domain/compression/ranking_optimizer.py` |
| **Line** | 169 |
| **Value** | 0.9 |
| **Used For** | Momentum coefficient in optimization |
| **Issue** | Standard value from literature, comment exists but no citation |
| **Remediation** | Add citation (this is standard SGD momentum) |
| **Priority** | P3 - Informational |

### composable_compressor.py:432

```python
min_compressibility: float = 0.5,
```

| Property | Value |
|----------|-------|
| **File** | `src/modelcypher/core/domain/compression/composable_compressor.py` |
| **Line** | 432 |
| **Value** | 0.5 |
| **Used For** | Minimum compressibility threshold |
| **Issue** | Exposed as parameter (good), but default needs rationale |
| **Remediation** | Document rationale for 0.5 default |
| **Priority** | P3 - Informational |

---

## Category 5: comp/phi Thresholds (CRITICAL)

### WORK-SUMMARY.md and related code

| comp/phi | Label | Used In |
|----------|-------|---------|
| ~1.0 | "Correct reasoning" | Training targets |
| >1.4 | "Conceptual confusion" | Decision gates |
| <0.8 | "Confident hallucination" | Decision gates |

| Property | Value |
|----------|-------|
| **Files** | `docs/LFM2-350M-WORK-SUMMARY.md`, training code |
| **Thresholds** | 0.8, 1.0, 1.4 |
| **Used For** | Categorizing reasoning quality |
| **Issue** | These are empirically observed but treated as universal constants |
| **Remediation** | Research first - measure across diverse inputs before assuming universal |
| **Priority** | P0 - Critical |

---

## Remediation Summary

| Priority | Count | Action |
|----------|-------|--------|
| **P0 - Critical** | 1 | Research comp/phi distribution before training toward 1.0 |
| **P1 - Medium** | 1 | Remove pass/fail from validate_phi_proxy.py |
| **P2 - Low** | 2 | Add documentation for rationale |
| **P3 - Informational** | 2 | Add citations/comments |

---

## Next Steps

1. Update `validate_phi_proxy.py` to report continuous values (Task #4)
2. Update documentation to remove universal claims (Task #5)
3. Create measurement script for phi distribution research (Task #6)
