# Geometric Adapter Transfer (Cross-LoRA)

> **Status**: Future Work / Experimental
> **Related**: `semantic_primes.md`

## The Dream: Write Once, Run Anywhere

Currently, if you train a "coding adapter" for Llama-3, it is useless for Qwen-2.5. You must burn GPU hours to retrain it.

**Geometric Adapter Transfer** attempts to solve this by projecting the *learned difference* ($\Delta W$) from the Source Manifold to the Target Manifold using invariant anchors.

## The Theory: Operational Invariants

We assume that "Coding" looks like a specific geometric deformation of the "Logic" subspace, regardless of the underlying model.

$$ \Delta W_{target} \approx P^T \cdot \Delta W_{source} \cdot P $$

Where $P$ is the orthogonal Procrustes rotation matrix derived from Semantic Primes.

## The Algorithm

1.  **Extract Anchors**: Compute the "Skeleton" of both Source and Target models using semantic prime anchors (see `mc geometry primes …`).
2.  **Align**: Compute the Rotation $R$ that maps $Source \to Target$.
3.  **Project**: Apply $R$ to the LoRA matrices $A$ and $B$.
4.  **Smooth**: Fine-tune the projected adapter on a small "calibration set" (orders of magnitude cheaper than full training).

**Repo note:** In ModelCypher, “semantic primes” are an anchor inventory stored in `src/modelcypher/data/semantic_primes.json` and surfaced via `mc geometry primes …`.
`ProbeCorpus` is a separate concept: a standardized set of prompts for activation probing (see `src/modelcypher/core/domain/geometry/probe_corpus.py`).

## Rotation Field Roughness

We measure the "Roughness" of the rotation field across layers.
-   **Smooth (< 0.2)**: Easy transfer. Models are "cognitively similar".
-   **Rough (> 1.0)**: "Tearing" risk. The conceptual mapping is non-linear. Transfer may fail.

## References

-   **Cross-LoRA (2025)**: Data-Free LoRA Transfer Framework.
-   **Trans-LoRA (2024)**: Towards data-free Transferable Parameter Efficient Finetuning.
