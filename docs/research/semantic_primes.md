# Semantic Primes: The Skeleton of Meaning

> **Status**: Core Theory
> **Inventory**: `src/modelcypher/data/semantic_primes.json` (+ `semantic_prime_multilingual.json`, `semantic_prime_frames.json`)
> **Core Types**: `src/modelcypher/core/domain/agents/semantic_primes.py`
> **CLI**: `mc geometry primes` (list/probe/compare)

## The Problem: Cross-Model Alignment Without Shared Coordinates

Comparing representations between disjoint model families (e.g., Llama-3 vs Qwen-2.5) is challenging because their weight matrices W and activation spaces A reside in different bases and dimensions. Direct comparison (e.g., $||W_A - W_B||$) is undefined.

To measure relational structure, we require a **shared anchor inventory**—a set of concepts $C = \{c_1, ..., c_k\}$ assumed to define a stable subspace across models.

## The Solution: Anchor-Based Probing

We utilize the **Natural Semantic Metalanguage (NSM)** inventory: ~65 proposed semantic primes (e.g., "I", "YOU", "BS", "GOOD") which serve as cross-linguistically stable anchors.

**Hypothesis**: If these primes induce a stable relational structure, the Gram matrices $G = XX^T$ of their embeddings should exhibit high Centered Kernel Alignment (CKA) between models, significantly exceeding that of frequency-matched controls.

### Methodology

1.  **Probe**: Extract embedding vectors $v_i$ for each prime $c_i$ in both models.
2.  **Multilingual Averaging**: Compute $v_i = \frac{1}{|L|} \sum_{l \in L} v_{i,l}$ across languages to reduce tokenizer bias.
3.  **Gram Matrix Comparison**: Compute CKA($G_A, G_B$) to measure structural similarity invariant to rotation.

## Empirical Validation

See [Paper 1: The Manifold Hypothesis of Agency](../../papers/paper-1-manifold-hypothesis-of-agency.md) for full results, including:
-   CKA scores vs null distributions
-   Falsification criteria (failed convergence > 10% drift)
-   Control baselines (frequency-matched random words)
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
- `mc geometry primes …` currently uses a lightweight, embedding-based proxy for "prime activation".
- `ProbeCorpus` is a separate concept: a standardized **prompt corpus** for activation probing (see `src/modelcypher/core/domain/geometry/probe_corpus.py`).
