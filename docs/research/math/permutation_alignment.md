# Permutation Alignment (Git Re-Basin)

> Aligning networks by resolving permutation symmetries before comparing or merging weights.

---

## Why This Matters for Model Merging

Many neural networks have **permutation symmetries**: you can permute hidden units (or other internal ordering choices) and obtain a functionally equivalent model. Naively averaging or directly comparing weights across two models can be misleading if their internal coordinates differ by a permutation.

**In ModelCypher**: permutation alignment is part of the historical merging literature (and a useful mental model), but ModelCypher’s current merge pipeline emphasizes representation/Gram alignment and null-space addition rather than “blend after re-basin”.

---

## Core Idea

Given two parameter sets that are functionally similar but internally permuted, find a permutation that best aligns them:

- Choose an objective (e.g., minimize weight difference after permuting, maximize correlation, match activations).
- Solve an assignment problem (often via Hungarian algorithm / optimal matching).
- Apply the permutation to reorder units so parameters become comparable in the same coordinate system.

---

## Relationship to Other Alignment Methods

- **Procrustes alignment**: solves a continuous orthogonal alignment (rotation/reflection) between representations.
- **Permutation alignment**: solves a discrete alignment (reindexing) over units.
- **Gram/CKA alignment**: compares representations in a way that is invariant to many coordinate choices (and can be used as an alignment diagnostic).

These methods address different failure modes; they are often complementary.

---

## Code Pointers (ModelCypher)

- Merge pipeline notes: `src/modelcypher/core/use_cases/merge/pipeline.py`
- CKA implementation: `src/modelcypher/core/domain/geometry/cka.py`

---

## References

- [Ainsworth et al. (2023)](../../references/arxiv/Ainsworth_2023_Git_ReBasin.pdf). *Git Re-Basin: Merging Models modulo Permutation Symmetries*. [arXiv:2209.04836](https://arxiv.org/abs/2209.04836)

