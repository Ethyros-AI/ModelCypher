# Semantic Primes: The Skeleton of Meaning

> **Status**: Core Theory
> **Inventory**: `src/modelcypher/data/semantic_primes.json` (+ `semantic_prime_multilingual.json`, `semantic_prime_frames.json`)
> **Core Types**: `src/modelcypher/core/domain/agents/semantic_primes.py`
> **CLI**: `mc geometry primes` (list/probe/compare)

## The Problem: How do we compare disjoint model families?

Llama-3 and Qwen-2.5 are disjoint model families: different architectures, different training mixes, and different tokenizers. Comparing weights directly is usually meaningless; output-only evaluation is necessary but often insufficient for explaining *why* models differ.

We need a shared anchor inventory (analogy: a **calibration standard**) that lets us probe *relational structure* in a consistent way.

## The Solution: Semantic Primes (NSM)

We use the **Natural Semantic Metalanguage (NSM)** inventory: ~65 proposed semantic primes intended to be cross-linguistically basic meanings (e.g., "I", "YOU", "GOOD", "BAD", "DO", "HAPPEN").

**Hypothesis**: If these concepts are sufficiently cross-linguistically stable, they may serve as useful **candidate anchors** for probing LLM representations. We treat “invariance” as a measurable, falsifiable claim, not an assumption.

### The "Skeleton" of the Manifold

By probing a model with these ~65 primes (and their translations), we can characterize probe-induced relational structure without assuming a shared coordinate system.

1.  **Probe**: Feed "I" into Model A and Model B.
2.  **Measure**: Capture the activation vector.
3.  **Correlate**: If the *relational structure* among primes is similar across models (e.g., via Gram correlation/CKA vs controls), that suggests the anchors induce stable structure under this probe protocol.

## Multilingual Anchors

We don't just use English. We use "Multilingual Primes" to average out tokenizer bias.
-   Anchor "I" = average(Vector("I"), Vector("Je"), Vector("Ich"), Vector("Yo"))

This creates a multilingual **proxy** vector that can reduce tokenizer-specific artifacts. It is not a definitive “representation of self,” and it may fail depending on tokenization and context.

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
