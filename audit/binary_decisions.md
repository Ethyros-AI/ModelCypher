# Audit: Binary Decisions on Continuous Metrics

> **Audit Date:** 2026-01-30
> **Purpose:** Catalog places where continuous metrics are converted to binary decisions, losing information.

---

## Summary

The codebase frequently converts rich continuous information into binary pass/fail or correct/incorrect labels. This loses valuable gradient information and imposes discrete thinking on what is fundamentally a continuous system.

---

## Category 1: Pass/Fail from Correlation

### ~~validate_phi_proxy.py:325-326~~ (RESOLVED - script removed)

~~This script has been removed from the codebase. Validation is now done via unit tests in `tests/test_differentiable_expansion.py` which report continuous correlation values without pass/fail thresholds.~~

---

## Category 2: Correct/Incorrect from String Matching

### self_reflection.py (training evaluation)

```python
# Exact substring matching for correctness
correct = expected in response
```

| Property | Value |
|----------|-------|
| **File** | `src/modelcypher/core/domain/training/self_reflection.py` |
| **Context** | Training evaluation |
| **Input** | Model response text, expected answer |
| **Output** | Boolean correct/incorrect |
| **Information Lost** | Semantic similarity, partial matches, reasoning quality |
| **Remediation** | Use embedding similarity or LLM judge for evaluation |
| **Priority** | P1 |

**Better approach:**
```python
# Instead of substring matching:
similarity = embedding_similarity(response, expected)
# Returns continuous [0, 1] measure
```

---

## Category 3: Decision Gates from Z-Scores

### Inference decision gates (conceptual)

```python
# Threshold-based gating
if entropy_zscore > threshold:
    trigger_reflection()
```

| Property | Value |
|----------|-------|
| **Context** | Inference-time decision making |
| **Input** | Continuous z-score |
| **Output** | Binary trigger/no-trigger |
| **Information Lost** | Z-score magnitude, probability of being wrong |
| **Remediation** | Use probabilistic soft gating |
| **Priority** | P2 |

**Better approach:**
```python
# Soft gating with probability
reflection_probability = sigmoid(entropy_zscore - threshold)
should_reflect = random() < reflection_probability
```

---

## Category 4: CKA Equality Checks

### Multiple test files and validation code

```python
assert result.cka == 1.0
```

| Property | Value |
|----------|-------|
| **Files** | Multiple test files, validation code |
| **Input** | CKA value (continuous [0, 1]) |
| **Output** | Pass/fail assertion |
| **Information Lost** | Numerical precision, floating-point tolerance |
| **Remediation** | Use `pytest.approx(1.0, rel=sqrt(eps))` |
| **Priority** | P0 - Critical for test stability |

**Found in:**
- `docs/architecture/multi_channel_merge.md:112` - "Should be 1.0" (documentation)
- Various test assertions

---

## Category 5: comp/phi State Classification

### WORK-SUMMARY.md and training code

```python
# State table in documentation
| State                   | comp/phi | Action          |
|-------------------------|----------|-----------------|
| Correct reasoning       | ~1.0     | Proceed         |
| Conceptual confusion    | >1.4     | Admit uncertainty |
| Confident hallucination | <0.8     | Verify with CoT |
```

| Property | Value |
|----------|-------|
| **Files** | `docs/LFM2-350M-WORK-SUMMARY.md`, training code |
| **Input** | Continuous comp/phi ratio |
| **Output** | Discrete state categories |
| **Information Lost** | Position within category, boundary uncertainty |
| **Remediation** | Report as continuous metric with soft category suggestions |
| **Priority** | P0 - Critical (this is the core insight of the audit) |

**Better approach:**
```python
# Instead of:
state = "confusion" if comp_phi > 1.4 else "correct"

# Report:
{
    "comp_phi": 1.35,
    "distance_from_nominal": 0.35,
    "category_suggestion": "elevated - may indicate complex processing or confusion",
    "confidence": 0.7  # based on empirical distribution
}
```

---

## Category 6: Benchmark Status Icons

### benchmark.py (conceptual)

```python
status_icon = "" if accuracy >= 0.8 else "?" if accuracy >= 0.5 else ""
```

| Property | Value |
|----------|-------|
| **Context** | Benchmark reporting |
| **Input** | Continuous accuracy [0, 1] |
| **Output** | Discrete status icon |
| **Information Lost** | Exact accuracy value, trend information |
| **Remediation** | Show continuous value alongside icon |
| **Priority** | P3 - Low (UI preference) |

---

## Philosophical Note

> "The entire system is a curved irrational system... God the unknowable built a perfectly logical yet irrational system."

The word "irrational" in mathematics means "cannot be expressed as a ratio of integers" - yet these irrational numbers (phi, pi, e) describe the most fundamental patterns. Our code should reflect this:

- **Binary decisions impose integer thinking** on a fundamentally continuous system
- **Thresholds create artificial cliffs** where gradients should flow
- **Categories lose information** that might be useful downstream

---

## Statistics

| Priority | Count | Description |
|----------|-------|-------------|
| P0 - Critical | 2 | CKA assertions, comp/phi categories |
| P1 - Medium | 1 | ~~Pass/fail from correlation~~ ✅, correctness from string |
| P2 - Low | 1 | Decision gates from z-scores |
| P3 - Informational | 1 | Benchmark icons |
| **Total** | **5** | Binary decision patterns (1 resolved) |

---

## Remediation Summary

### Immediate Actions

1. **Fix CKA test assertions** to use `pytest.approx()` with dtype-derived tolerance
2. ~~**Update validate_phi_proxy.py** to report continuous values without pass/fail~~ ✅ RESOLVED (script removed, tests use continuous values)
3. **Update documentation** to present expansion_ratio as continuous metric, not state categories

### Future Improvements

4. Replace substring matching with embedding similarity in training evaluation
5. Implement soft gating for inference decisions
6. Add continuous values alongside any remaining categorical displays
