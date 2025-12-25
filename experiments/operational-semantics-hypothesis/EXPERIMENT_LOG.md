# Operational Semantics Hypothesis: Experiment Log

**Principal Investigator**: Claude (AI Research Assistant)
**Date Started**: 2025-12-25
**Status**: In Progress

---

## 1. Hypothesis Summary

**Core Claim**: Mathematical concepts are encoded as structural relationships in LLM latent space, not as memorized symbolic associations. Specifically, the Pythagorean theorem (a² + b² = c²) should manifest as a geometric property of the embedding space.

**Predictions**:
1. Valid Pythagorean triples cluster together, separated from invalid near-misses
2. A consistent geometric transformation maps (a,b) pairs to their hypotenuses c
3. This structure generalizes to novel (unseen) triples
4. Cross-modal presentations (formula, word problem, geometric) converge
5. Arithmetic relationships manifest as geometric directions/distances

**Falsification Criteria**:
- Valid/invalid triples don't cluster → REJECT
- Novel triples fail but common ones succeed → PARTIAL (memorization)
- No consistent transformation → REJECT
- Cross-modal presentations diverge → REJECT

---

## 2. Experimental Design

### 2.1 Methodology

All experiments will use the validated ModelCypher CLI and MCP infrastructure. No custom scripts performing geometric calculations - only CLI commands that invoke the rigorously tested geometry modules.

**Key Principle**: The codebase enforces geodesic geometry and Fréchet means throughout. We trust this infrastructure rather than reimplementing it.

### 2.2 Models Under Test

| Model | Size | Quantization | Path |
|-------|------|--------------|------|
| Qwen2.5-0.5B | 0.5B | bf16 | /Volumes/CodeCypher/models/mlx-community/Qwen2.5-0.5B-Instruct-bf16 |
| Llama-3.2-3B | 3B | 4-bit | /Volumes/CodeCypher/models/mlx-community/Llama-3.2-3B-Instruct-4bit |
| Qwen2.5-3B | 3B | bf16 | /Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16 |
| Mistral-7B | 7B | 4-bit | /Volumes/CodeCypher/models/mlx-community/Mistral-7B-Instruct-v0.3-4bit |
| Qwen3-8B | 8B | 4-bit | /Volumes/CodeCypher/models/mlx-community/Qwen3-8B-4bit |

### 2.3 Relevant CLI Tools

See Section 3: Tool Inventory

---

## 3. Tool Inventory

### 3.1 Semantic Prime Probing
```bash
# Probe a model for 65 NSM semantic prime representations
mc geometry primes probe-model MODEL_PATH --output OUTPUT.json

# Compare semantic primes between two models (CKA similarity)
mc geometry primes compare ACTIVATIONS_A.json ACTIVATIONS_B.json
```

### 3.2 Multi-Atlas Layer Mapping (343 probes)
```bash
# Map layers between models using full 343-probe atlas
# Sources: sequence invariants (68), semantic primes (65),
#          computational gates (76), emotion concepts (32),
#          temporal (25), social (25), moral (30), compositional (22)
mc geometry invariant map-layers \
  --source MODEL_A --target MODEL_B \
  --scope multiAtlas \
  --triangulation

# View available probes
mc geometry invariant atlas-inventory
```

### 3.3 Structural Comparison
```bash
# Gromov-Wasserstein distance (structure without correspondence)
mc geometry metrics gromov-wasserstein SOURCE.json TARGET.json

# Topological fingerprint via persistent homology
mc geometry metrics topological-fingerprint POINTS.json

# Intrinsic dimension estimation (TwoNN)
mc geometry metrics intrinsic-dimension POINTS.json
```

### 3.4 Clustering
```bash
# DBSCAN clustering of manifold points
mc geometry manifold cluster --points POINTS.json --epsilon 0.3
```

---

## 4. Experimental Design

### Experiment 1: Cross-Model Invariance (Foundation Test)

**Rationale**: Before testing Pythagorean-specific claims, validate the foundational premise that concept representations are "invariant but twisted" across architectures. This has been validated repeatedly with the UnifiedAtlas.

**Method**:
1. Run `mc geometry invariant map-layers` between each model pair with `--scope multiAtlas`
2. Examine layer correspondence and triangulation scores
3. Compare alignment quality metrics

**Prediction**: Layer mapping should succeed with high triangulation scores, indicating that the same conceptual structure exists (rotated/scaled) across models.

**Success Criteria**:
- Layer mapping converges with < 0.35 collapse threshold
- Cross-domain triangulation multiplier > 1.2

---

### Experiment 2: Semantic Prime Comparison (CKA)

**Rationale**: Test if semantic primes (the atoms of human conceptual thought) are represented similarly across models.

**Method**:
1. Probe each model with `mc geometry primes probe-model`
2. Compare each pair with `mc geometry primes compare`
3. Record CKA similarity scores

**Prediction**: CKA > 0.6 for semantic primes across model pairs.

**Success Criteria**:
- Mean CKA > 0.6 across all model pairs
- Category-level coherence patterns consistent

---

### Experiment 3: Mathematical Invariant Structure

**Rationale**: Test if sequence invariants (Fibonacci, primes, logic patterns) show cross-model consistency. This connects to the Pythagorean hypothesis by testing if mathematical structure is encoded geometrically.

**Method**:
1. Run layer mapping with `--families fibonacci,primes,logic,arithmetic`
2. Compare sequence family alignment across models

**Prediction**: Mathematical invariants should show strong cross-model alignment since they represent formal relationships.

**Success Criteria**:
- Sequence invariant alignment > semantic prime alignment
- Logic family shows strongest consistency

---

## 5. Experiment Runs

### Run 1: Cross-Model Invariance (Foundation)
*Pending*

### Run 2: Semantic Prime CKA
*Pending*

### Run 3: Mathematical Invariants
*Pending*

---

## 6. Raw Output Archive

All raw CLI output logged verbatim in `raw_output/`:
- `cli_geometry_help.txt` - Full geometry command help
- `run_N_*.json` - Experiment N outputs
- `run_N_*.txt` - Experiment N console output

---

## 7. Analysis & Conclusions

*Pending experimental results*

---

## Appendix A: CLI Commands Used

All commands will be logged here with timestamps.

```bash
# [2025-12-25 08:XX] Tool inventory
poetry run mc geometry --help
poetry run mc geometry primes --help
poetry run mc geometry invariant --help
# ... (to be updated as experiments run)
```
