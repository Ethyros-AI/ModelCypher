# Semantic Primes: The Skeleton of Meaning

> **Status**: Core Theory
> **Implementation**: `src/modelcypher/core/domain/geometry/probe_corpus.py`

## The Problem: How do we compare alien minds?

Llama-3 and Qwen-2.5 are "alien minds" relative to each other. They have different architectures, different training data, and different tokenizers. Comparing their weights directly is impossible. Comparing their outputs is subjective.

We need a **Rosetta Stone**.

## The Solution: Semantic Primes (NSM)

We use the **Natural Semantic Metalanguage (NSM)**, a set of ~65 universal concepts found in all human languages (e.g., "I", "YOU", "GOOD", "BAD", "DO", "HAPPEN").

**Hypothesis**: If these concepts are universal to human cognition, they must also be invariant anchors in the "knowledge manifold" of any sufficiently advanced LLM.

### The "Skeleton" of the Manifold

By probing a model with these 65 primes (and their translations), we can triangulate the geometry of its latent space.

1.  **Probe**: Feed "I" into Model A and Model B.
2.  **Measure**: Capture the activation vector.
3.  **Correlate**: If the "I" vector in Model A relates to the "YOU" vector in Model A in the distinct way that "I" relates to "YOU" in Model B, then the **geometry is preserved**.

## Multilingual Anchors

We don't just use English. We use "Multilingual Primes" to average out tokenizer bias.
-   Anchor "I" = average(Vector("I"), Vector("Je"), Vector("Ich"), Vector("Yo"))

This creates a robust, language-agnostic centroid for the concept of "Self".

## Empirical Results

Our experiments (`docs/research/prime_geometry/`) show:
-   **High CKA (>0.9)**: Between English and Multilingual prime skeletons.
-   **Universal Structure**: The "shape" of the relationship between Primes (e.g., "GOOD" vs "BAD") is conserved across Llama, Mistral, and Qwen, even before alignment.

## Usage in ModelCypher

The `ProbeCorpus` class defines these standard anchors.

```python
# From src/modelcypher/core/domain/geometry/probe_corpus.py
class ProbeCorpus(Enum):
    SEMANTIC_PRIMES = "semantic_primes"  # The 65 NSM primes
    COMPUTATIONAL_GATES = "computational_gates"  # Logic primitives (IF, THEN, ELSE)
```

We use these anchors to compute the **Intersection Map** between models.
