# The Manifold Swapping Hypothesis

> **Status**: Theoretical / Highly Experimental
> **Goal**: To explore compositional models by stitching optimized sub-manifolds (highly speculative).

## The Hypothesis

If neural networks are composed of discrete "Functional Manifolds" (e.g., Syntax Manifold, Fact Retrieval Manifold, Reasoning Manifold), then it should be possible to **swap** these components between models, provided we apply the correct geometric "glue" (linear stitching layers).

## The Proposal: "Ship of Theseus" Models

Instead of training a monolithic model, one could imagine composing a system from specialized components, e.g.:
1.  **Syntax-specialized early layers** (roughly 0–N): strong token/grammar handling.
2.  **Retrieval/knowledge middle layers** (roughly N–M): strong factual association.
3.  **Planning/tool-use late layers** (roughly M–L): strong multi-step task execution.

This is not an established capability. Layer semantics are not modular “parts” in a clean engineering sense; the proposal is included here as an explicit (and falsifiable) research hypothesis.

## The Glue: Stitching Layers

We cannot just stack them. The activation spaces are misaligned.
We train a **Lasso Stitcher** ($\lambda$-stitch) at each interface:

$$ h_{out} = \text{Stitch}(h_{in}) = W_{stitch} \cdot h_{in} + b_{stitch} $$

$W_{stitch}$ is initialized via Procrustes Alignment on Semantic Primes, then fine-tuned.

## Risks: "The Islet of Lance"

Experimental results suggest that some concepts (like "Lance") are "Islets"—disconnected from the main manifold. Swapping these regions often leads to catastrophic semantic collapse (gibberish).

**Falsification**: If Procrustes alignment fails to recover >80% CKA at the swap interface, the Manifold Swapping Hypothesis is invalid for that model pair.
