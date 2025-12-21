# The Manifold Swapping Hypothesis

> **Status**: Theoretical / Highly Experimental
> **Goal**: To build "Frankenstein" models by stitching optimized sub-manifolds.

## The Hypothesis

If neural networks are composed of discrete "Functional Manifolds" (e.g., Syntax Manifold, Fact Retrieval Manifold, Reasoning Manifold), then it should be possible to **swap** these components between models, provided we apply the correct geometric "glue" (linear stitching layers).

## The Proposal: "Ship of Theseus" Models

Instead of training a monolithic 100B parameter model, we envision:
1.  **Llama-3 Syntax** (Layers 0-10): Best-in-class parsing.
2.  **Grok Fact Retrieval** (Layers 10-20): Real-time knowledge access.
3.  **Claude Reasoning** (Layers 20-30): Superior chain-of-thought.

## The Glue: Stitching Layers

We cannot just stack them. The activation spaces are misaligned.
We train a **Lasso Stitcher** ($\lambda$-stitch) at each interface:

$$ h_{out} = \text{Stitch}(h_{in}) = W_{stitch} \cdot h_{in} + b_{stitch} $$

$W_{stitch}$ is initialized via Procrustes Alignment on Semantic Primes, then fine-tuned.

## Risks: "The Islet of Lance"

Experimental results suggest that some concepts (like "Lance") are "Islets"—disconnected from the main manifold. Swapping these regions often leads to catastrophic semantic collapse (gibberish).

**Falsification**: If Procrustes alignment fails to recover >80% CKA at the swap interface, the Manifold Swapping Hypothesis is invalid for that model pair.
