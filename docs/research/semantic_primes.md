# Semantic Primes: Anchor Inventory for Cross-Model Comparison

> **Status**: Core inventory and CLI support
> **Data**: `src/modelcypher/data/semantic_primes.json`,
>           `semantic_prime_multilingual.json`, `semantic_prime_frames.json`
> **Code**: `src/modelcypher/core/domain/agents/semantic_primes.py`,
>           `src/modelcypher/core/domain/agents/unified_atlas.py`
> **CLI**: `mc geometry primes` (list / probe-model / compare)

---

## Overview

Semantic primes (NSM) are treated as a small, standardized anchor set for
cross-model comparison. ModelCypher uses the English 2014 inventory to probe
embedding-space structure and compute CKA-based coherence metrics.

---

## Inventory Sources

- `semantic_primes.json` provides the English 2014 prime list and categories.
- `semantic_prime_multilingual.json` and `semantic_prime_frames.json` are
  available for future multilingual/frames analysis.
- `semantic_primes.py` contains the English 2014 inventory used by the CLI.

---

## CLI Workflow

```bash
# List the prime inventory
mc geometry primes list

# Probe a local model directory (writes optional JSON)
mc geometry primes probe-model /path/to/model --output-file primes.json

# Compare two activation JSON files
mc geometry primes compare model_a_primes.json model_b_primes.json
```

Notes:
- `probe-model` extracts mean activations for each prime and computes CKA
  coherence (overall and per category).
- `compare` computes CKA between two activation JSONs and reports the most
  similar/divergent primes. If dimensions differ, it falls back to a centroid
  similarity heuristic for per-prime ranking.

---

## Implementation Details (Probe)

1. Encode each prime’s first English exponent.
2. Run a forward pass to the final layer and mean-pool activations.
3. Compute CKA for all primes (overall coherence) and within each category.

This is a lightweight embedding-based proxy; use activation corpora for deeper
analysis when needed.

---

## Related

- [falsification_experiments.md](falsification_experiments.md)
- [math/centered_kernel_alignment.md](math/centered_kernel_alignment.md)
