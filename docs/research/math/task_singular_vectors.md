# Task Singular Vectors (TSV)

> Low-rank task vector decomposition (research reference).

---

## Status in ModelCypher

TSV is **not implemented** in ModelCypher. There is no
`task_singular_vectors.py` module in the codebase at this time. This document
is retained as a research note and should not be treated as an available
feature.

---

## Core Idea (External Literature)

Given a task delta matrix $\Delta W$ from fine-tuning, TSV proposes decomposing
it via SVD and keeping only the dominant singular directions. The intuition is
that task deltas are low-rank and can be compressed while preserving behavior.

---

## If TSV Is Added Later

Any TSV implementation in ModelCypher should follow existing principles:

1. **Backend-only math**: Use `geodesic_svd` and backend ops (no NumPy).
2. **Data-derived rank**: Choose rank from singular value gaps or
   condition thresholds, not fixed heuristics.
3. **No blending**: TSV should not replace null-space addition in merging.
4. **Raw metrics**: Report measurements only, no qualitative labels.

---

## Related Modules

- [`src/modelcypher/core/domain/geometry/geometric_lora.py`](../../../../src/modelcypher/core/domain/geometry/geometric_lora.py) - SVD-based low-rank factors
- [`src/modelcypher/core/domain/geometry/safety_polytope.py`](../../../../src/modelcypher/core/domain/geometry/safety_polytope.py) - TSV_PRUNE diagnostic label
- [spectral_analysis.md](spectral_analysis.md) - Spectral ratios and conditioning
