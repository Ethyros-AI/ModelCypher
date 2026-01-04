# Geometric Adapter Transfer (Cross-LoRA)

> **Status**: Future Work / Experimental
> **Related**: `semantic_primes.md`

## The Dream: Write Once, Run Anywhere

Currently, if you train a "coding adapter" for Llama-3, it is typically not directly usable on Qwen-2.5 without retraining or additional conversion work.

**Geometric Adapter Transfer** is a hypothesis-driven approach: project a learned low-rank update ($\Delta W$) from a source model into a target model using *candidate* anchors and diagnostics, then validate with downstream evals.

## The Theory: Operational Invariants

**Hypothesis**: Some fine-tuned behaviors (e.g., "coding") correspond to approximately transferable low-rank structure that is detectable via anchor-induced relational geometry, even across model families. This can fail due to tokenizer mismatch, non-bijective layer roles, and genuinely new features introduced during fine-tuning.

$$ \Delta W_{target} \approx P^T \cdot \Delta W_{source} \cdot P $$

Where $P$ is the orthogonal Procrustes rotation matrix derived from Semantic Primes.

## The Algorithm

1.  **Extract Anchors**: Compute the "Skeleton" of both Source and Target models using semantic prime anchors (see `mc geometry primes …`).
2.  **Align**: Compute the Rotation $R$ that maps $Source \to Target$.
3.  **Project**: Apply $R$ to the LoRA matrices $A$ and $B$.
4.  **Smooth**: Fine-tune the projected adapter on a small "calibration set" (orders of magnitude cheaper than full training).

**Repo note:** In ModelCypher, “semantic primes” are an anchor inventory stored in `src/modelcypher/data/semantic_primes.json` and surfaced via `mc geometry primes …`.
`KnowledgeProbeCorpus` is a separate concept: a standardized set of prompts for knowledge probing (see `src/modelcypher/core/domain/merging/knowledge_transfer_validator.py`).

## Rotation Field Roughness

Roughness measures how non-uniformly the rotation field varies across layers.
-   **What it measures**: The variance in Procrustes rotation matrices between adjacent layers.
-   **Lower values**: More consistent rotation across layers (uniform transformation).
-   **Higher values**: Rotation varies significantly between layers (non-uniform mapping).

Interpret roughness relative to baseline observations for similar model pairs.

## References

-   **Transferring Linear Features Across Language Models With Model Stitching. arXiv:2506.06609. (2025).**
