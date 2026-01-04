# TIES-Merging (Trim, Elect Sign, Merge)

> Parameter-space merge baseline (research reference).

---

## Status in ModelCypher

TIES-Merging is **not implemented** in ModelCypher. The merge pipeline uses
geometric alignment and null-space addition, not parameter-space averaging.
This document is retained as a reference to external literature.

---

## Core Idea (External Literature)

Given task vectors $\tau_t = \theta_t - \theta_{pre}$, TIES proposes:

1. **Trim**: keep only the largest-magnitude parameters per task.
2. **Elect sign**: choose a consensus sign per parameter.
3. **Merge**: average magnitudes that agree with the elected sign.

This is a parameter-space heuristic and does not preserve geometric structure
on curved manifolds.

---

## If TIES Is Added Later

Any implementation should respect ModelCypher constraints:

1. **Backend-only math**: Use the Backend protocol (no NumPy).
2. **Data-derived thresholds**: Avoid fixed density heuristics.
3. **No blending in production merges**: TIES may be used only for comparative
   research, not as a merge step.
4. **Raw measurements**: Report metrics only; no qualitative labels.

---

## Related Modules

- [procrustes_analysis.md](procrustes_analysis.md) - Alignment before comparison
- [spectral_analysis.md](spectral_analysis.md) - Scale/conditioning diagnostics
- [`src/modelcypher/core/domain/geometry/permutation_aligner.py`](../../../../src/modelcypher/core/domain/geometry/permutation_aligner.py) - Alignment for permutation symmetries
