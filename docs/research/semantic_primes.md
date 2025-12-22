# Semantic Primes: The Skeleton of Meaning

> **Status**: Core Theory
> **Inventory**: `src/modelcypher/data/semantic_primes.json` (+ `semantic_prime_multilingual.json`, `semantic_prime_frames.json`)
> **Core Types**: `src/modelcypher/core/domain/agents/semantic_primes.py`
> **CLI**: `mc geometry primes` (list/probe/compare)

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

For experimental framing and measurement targets (CKA, null controls, falsification criteria), see:
- [Paper I Draft: The Manifold Hypothesis of Agency](../../papers/paper-1-manifold-hypothesis-of-agency.md)
- [Scientific Method: Falsification Experiments](falsification_experiments.md)

## Usage in ModelCypher

Semantic primes are treated as an **anchor inventory** (a small, standardized probe set).
In ModelCypher, the canonical inventories live in `src/modelcypher/data/`.

```python
# Inventory types (see src/modelcypher/core/domain/agents/semantic_primes.py)
# - SemanticPrimeInventory.english2014()
# - SemanticPrimeSignature
```

### CLI workflow

```bash
# List the prime inventory
mc geometry primes list

# Probe a local model directory for prime signals (lightweight proxy)
mc geometry primes probe /path/to/model

# Compare two local model directories
mc geometry primes compare --model-a /path/to/model-a --model-b /path/to/model-b
```

Notes:
- `mc geometry primes …` currently uses a lightweight, embedding-based proxy for “prime activation”. Deeper activation probing is tracked as ongoing work (see `../PARITY.md`).
- `ProbeCorpus` is a separate concept: a standardized **prompt corpus** for activation probing (see `src/modelcypher/core/domain/geometry/probe_corpus.py`).
