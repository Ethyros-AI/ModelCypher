# The Manifold Swapping Hypothesis

> **Status**: Theoretical / Highly Experimental
> **Goal**: Explore compositional models by swapping sub-manifolds (speculative).

## The Hypothesis

If neural networks are composed of discrete functional manifolds (syntax, retrieval, reasoning),
then it should be possible to swap those components between models when the correct geometric
alignment is applied at the interface.

## The Proposal: "Ship of Theseus" Models

Instead of training a monolithic model, one could imagine composing a system from specialized
components, for example:
1. **Syntax-specialized early layers** (roughly 0-N): tokenization and grammar.
2. **Retrieval/knowledge middle layers** (roughly N-M): factual association.
3. **Planning/tool-use late layers** (roughly M-L): multi-step execution.

This is not an established capability. Layer semantics are not modular parts in a clean
engineering sense; the proposal is included as a falsifiable research hypothesis.

## The Glue: Alignment Transforms (Current Code vs Hypothesis)

In the current merge pipeline, "stitching" refers to linear transforms derived during the
probe stage to align hidden and attention dimensions across architectures. These transforms
are applied in the transplant stage (see `src/modelcypher/core/use_cases/merge/stages/transplant.py`)
and are computed from feature transforms, not learned adapters. There is no implementation
of a learned stitcher or layer-swap pipeline yet.

If manifold swapping becomes a concrete research track, alignment transforms from the probe
stage (and local Procrustes rotations in `src/modelcypher/core/domain/geometry/manifold_stitcher.py`)
are the most likely starting point for interface alignment.

## Risks and Falsification

Potential failure modes include isolated concept islands or insufficient alignment samples.
If alignment fails to reach CKA = 1.0 at a swap interface, debug the alignment pipeline before
concluding incompatibility.
