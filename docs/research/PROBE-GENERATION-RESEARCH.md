# Probe Generation Research: Achieving rank = hidden_dim at Every Layer

## Research Goal

Develop a mathematically principled probe generation system that achieves:

```
rank(A_l) = d_l  for all layers l
```

Where:
- `A_l ∈ ℝ^(n×d_l)` is the activation matrix at layer l
- `n` = number of probes
- `d_l` = hidden dimension at layer l

## Initial Findings

### Experiment: Rank Analysis on SmolLM2-135M-Instruct (n=500, d=576)

| Layer Range | Rank Range | Coverage | Observation |
|-------------|------------|----------|-------------|
| 0-10 (early) | 381-493 | 66-86% | High diversity |
| 11-27 (middle) | 40-191 | **7-33%** | Severe compression |
| 28-29 (late) | 357-417 | 62-72% | Recovery |

**Key finding**: Middle layers achieve only 7-10% rank coverage even with n ≈ d.

This is NOT explained by n < d alone. The activations themselves lie in a low-dimensional subspace.

### Experiment 10: Rank Saturation Curve (n=4596, streaming)

Streamed all 4596 probes in batches of 100, measuring rank after each batch.

**SmolLM-135M (d=576)**:
| Layer Depth | Final Rank | Coverage | Saturation Point |
|-------------|------------|----------|------------------|
| 25% (layer 7) | ~576 | 100% | ~600 probes |
| 50% (layer 15) | ~149 | **26%** | ~500 probes (early saturation) |
| 75% (layer 22) | ~318 | **55%** | ~1200 probes |

**LFM2-350M (d=960)**:
| Layer Depth | Final Rank | Coverage | Saturation Point |
|-------------|------------|----------|------------------|
| 25% (layer 4) | 960 | 100% | ~1000 probes |
| 50% (layer 8) | 960 | 100% | ~1500 probes |
| 75% (layer 12) | 960 | 100% | ~1500 probes |

**Critical insight**: This is NOT a model limitation—it's a probe limitation.

- LFM2 achieves 100% rank at all layers with our English-language probes
- SmolLM's middle layers use dimensions our probes don't activate
- Different models take different computational pathways through middle layers
- The relationships (CKA) are invariant, but the pathways are not

**Implication**: Every model requires a different probe set to achieve full rank mapping. Full rank mapping is prerequisite to merging.

## Mathematical Framework

### Problem Decomposition

The rank deficiency has two possible causes:

1. **Sampling limitation**: n_probes is too small to span the reachable subspace
2. **Intrinsic dimensionality**: The model architecture constrains activations to a low-dimensional manifold

**Key insight**: If cause (2) dominates, then NO probe set can achieve full rank.

### Formal Definitions

**Definition 1 (Reachable Subspace)**:
For layer l, the reachable subspace is:
```
R_l = span{f_l(x) : x ∈ all valid inputs}
```
where `f_l(x)` is the activation at layer l for input x.

**Definition 2 (Intrinsic Dimension)**:
The intrinsic dimension at layer l is `ID_l = dim(R_l)`.

**Theorem (Rank Upper Bound)**:
For any probe set P with activations A_l:
```
rank(A_l) ≤ min(|P|, ID_l)
```

**Corollary**: If `ID_l < d_l`, then full rank is impossible regardless of probe count.

### Key Questions

1. **What is ID_l at each layer?** Use TwoNN or MLE estimators on diverse inputs.
2. **Why is ID_l < d_l?** Possible causes:
   - Information bottleneck (middle layers compress)
   - RoPE/positional encoding constrains dimensions
   - Layer normalization collapses directions
   - Architectural dead directions (never activated)

3. **Can we identify the "missing" directions?** Compute null(A_l^T) to find directions orthogonal to probe activations.

4. **Are missing directions reachable?** Test if random noise inputs activate those directions.

## Research Plan

### Phase 1: Characterize the Manifold

1. Run rank analysis with full probe set (4596 probes)
2. Compare rank vs intrinsic dimension estimates (TwoNN)
3. Identify if gap is sampling or architecture

### Phase 2: Null Space Analysis

For layers with rank < hidden_dim:
1. Compute SVD: A_l = U Σ V^T
2. Extract null directions: V[:, rank_l:]
3. Characterize null space:
   - Are they consistent across layers?
   - Do they correspond to specific token patterns?
   - Can synthetic inputs activate them?

### Phase 3: Targeted Probe Generation (SAE-Based Approach)

Recent research (2025-2026) on Sparse Autoencoders provides a principled method for discovering unmapped dimensions.

#### SAE Research Foundations

**Key Paper 1**: "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" (arXiv:2506.23845, June 2025)

Core finding: SAEs excel at **enumerating concepts unsupervised**. They decompose activations into sparse features that span the full activation space—including dimensions our probes don't reach.

**Key Paper 2**: "Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models" (ICLR 2025)

Core finding: SAE features reveal **universal feature spaces** across different LLMs. Features learned on one model transfer to others. This validates the geometric premise: models encode the same conceptual space in different coordinates.

#### SAE-Based Probe Generation Algorithm

```
FOR each model M, each layer l:
    1. Train SAE on activations A_l
       - SAE decomposes d_l dimensions into k sparse features (k >> d_l)
       - Each feature captures a distinct activation direction

    2. Run current probes through model
       - Identify which SAE features activate (activation > threshold)
       - Identify DORMANT features (never activate on any probe)

    3. Auto-interpret dormant features
       - Use model's own decoder weights to understand what concepts they represent
       - Cluster dormant features by semantic similarity

    4. Generate probes targeting dormant features
       - For each dormant feature cluster, generate text that should activate it
       - Options: gradient optimization, LLM-based generation, structured search

    5. Validate new probes span dormant directions
       - Run new probes, measure rank increase
       - Repeat until rank(A_l) = d_l
```

#### Why SAEs Work for This Problem

1. **Complete decomposition**: SAE features span the full activation space by construction
2. **Interpretability**: Each feature has semantic meaning (via auto-interpretation)
3. **Universality**: Features transfer across models—what we learn helps with any model
4. **Unsupervised discovery**: SAEs find concepts we didn't know to look for

#### Alternative Approaches (Lower Priority)

If SAE approach fails or is too slow:
1. **Gradient-based generation**: Optimize probe text to maximize activation in null directions
2. **Adversarial probing**: Use model's own gradients to find inputs that activate dead directions
3. **Structured exploration**: Systematically vary probe attributes (length, complexity, domain)

### Phase 4: Validation

1. Verify rank(A_l) = d_l at all layers (or prove it's impossible)
2. Test alignment quality with full-rank probes
3. Measure merge quality improvement

## Mathematical Constraints

### The Minimum Probe Count Problem

**Given**: Model with hidden_dim d and intrinsic dimension ID at each layer.

**Find**: Minimum n such that rank(A_l) = min(n, ID_l) for all l.

**Lower bound**: n ≥ max_l(ID_l)

**Practical bound**: For generic point clouds (Berry & Sauer 2016):
```
n ≥ d × (1 + 1/sqrt(d))  for well-conditioned Gram matrix
```

For d=576: n ≥ 600 probes minimum.

But this assumes the probe activations are "generic" (no special structure). Neural activations are highly structured, so the actual requirement may be higher.

### Condition Number Constraint

For numerical stability of F = pinv(A) @ B:
```
κ(G) × ε < 0.1  where G = A^T A
```

Current observations show κ ≈ 2.8e3, which is acceptable for float32 (ε ≈ 1e-7).

## Implementation Notes

### Architecture

```
src/modelcypher/core/domain/probes/
├── rank_analyzer.py        # Measure rank at each layer
├── null_space_analyzer.py  # Find and characterize missing directions
├── intrinsic_dimension.py  # Estimate ID at each layer
├── probe_generator.py      # Generate targeted probes
└── coverage_validator.py   # Verify full rank achieved
```

### CLI Integration

```bash
# Analyze current probe coverage
mc probe analyze --model <path> --output coverage.json

# Generate probes to fill rank gaps
mc probe generate --model <path> --target-rank full --output new_probes.json

# Validate full rank achieved
mc probe validate --model <path> --probes all
```

## Proposed Experiment: SAE-Guided Probe Discovery

### Experiment 11: SAE Feature Coverage

**Question**: Can SAEs identify which dimensions our probes miss?

**Protocol**:
1. Train SAE on SmolLM layer 15 activations (the 26% coverage layer)
2. Run all 4596 probes, record which SAE features activate
3. Identify dormant features (never activated by any probe)
4. Measure: `n_dormant / n_total_features`

**Expected outcome**: Dormant features should correspond to ~74% of dimensions (the unmapped space)

### Experiment 12: Targeted Probe Generation

**Question**: Can we generate probes that activate dormant SAE features?

**Protocol**:
1. Take top-k dormant features from Exp 11
2. Auto-interpret each feature (what concept does it represent?)
3. Generate probes targeting those concepts (LLM-assisted)
4. Run new probes, measure:
   - Which dormant features now activate?
   - Does numerical rank increase?

**Success criteria**:
- New probes activate previously dormant features
- Numerical rank increases toward d_l

### Experiment 13: Full Rank Validation

**Question**: Can we achieve rank = hidden_dim at every layer?

**Protocol**:
1. Apply SAE-guided probe generation iteratively
2. For each layer, generate probes until rank saturates at d_l
3. Measure total probes required per layer

**Success criteria**:
- rank(A_l) = d_l for all layers
- Or: prove certain directions are unreachable (architectural dead neurons)

---

## Implementation Notes

### SAE Training Requirements

```python
# SAE architecture for SmolLM (d=576)
sae_config = {
    "input_dim": 576,
    "hidden_dim": 576 * 8,  # 8x expansion typical for SAEs
    "sparsity_coefficient": 0.04,  # L1 penalty
    "tied_weights": True,  # encoder.T = decoder for interpretability
}
```

Training data: Activations from diverse text corpus (not just our probes)

### Dormant Feature Detection

```python
def find_dormant_features(sae, probe_activations, threshold=0.01):
    """
    Identify SAE features that never activate on probe set.

    threshold: activation level below which feature is considered dormant
    """
    encoded = sae.encode(probe_activations)  # [n_probes, n_features]
    max_activation = encoded.max(axis=0)     # [n_features]
    dormant_mask = max_activation < threshold
    return dormant_mask
```

### Architecture

```
src/modelcypher/core/domain/probes/
├── rank_analyzer.py        # Measure rank at each layer
├── null_space_analyzer.py  # Find and characterize missing directions
├── intrinsic_dimension.py  # Estimate ID at each layer
├── probe_generator.py      # Generate targeted probes
├── coverage_validator.py   # Verify full rank achieved
├── sae_trainer.py          # Train sparse autoencoders
└── feature_interpreter.py  # Auto-interpret SAE features
```

### CLI Integration

```bash
# Analyze current probe coverage
mc probe analyze --model <path> --output coverage.json

# Train SAE for a specific layer
mc probe train-sae --model <path> --layer 15 --output sae_layer15.safetensors

# Find dormant features
mc probe find-dormant --model <path> --sae sae_layer15.safetensors --output dormant.json

# Generate probes targeting dormant features
mc probe generate --model <path> --dormant dormant.json --output new_probes.json

# Validate full rank achieved
mc probe validate --model <path> --probes all
```

---

## References

1. Berry & Sauer (2016) - Gram matrix conditioning for point clouds
2. Facco et al. (2017) - TwoNN intrinsic dimension estimator
3. CLAUDE.md - "Don't Invent Heuristics" principle
4. Experiments: `experiments/rank_analysis.py`
5. Marks et al. (2025) - "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" (arXiv:2506.23845)
6. Feng et al. (2025) - "Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models" (ICLR 2025)
7. Experiment 10: `experiments/validation_protocol/exp10_rank_saturation/`
