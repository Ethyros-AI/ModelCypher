# Research Map

ModelCypher implements geometric analysis tools that support research into the relationship between neural network representations and biological cognition. This document consolidates external research connections, implementation status, and future directions.

---

## Part I: Research Foundations

### The Platonic Representation Hypothesis

**Paper**: Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). *The Platonic Representation Hypothesis*. ICML 2024. ([PDF](references/arxiv/Huh_2024_Platonic_Representation.pdf), [arXiv:2405.07987](https://arxiv.org/abs/2405.07987))

**Core Claim**: Neural networks trained with different objectives on different data and modalities converge to a shared statistical model of reality in their representation spaces.

**ModelCypher Implementation**: Our CKA implementation (`src/modelcypher/core/domain/geometry/cka.py`) with Gram-based cross-dimensional comparison directly enables testing this hypothesis. `compute_cka_from_grams()` enables cross-dimensional comparison without projection or truncation.

---

### Blue Brain Project: Algebraic Topology of Neural Circuits

**Paper**: Reimann, M.W., et al. (2017). *Cliques of Neurons Bound into Cavities Provide a Missing Link between Structure and Function*. Frontiers in Computational Neuroscience.
**DOI**: [10.3389/fncom.2017.00048](https://doi.org/10.3389/fncom.2017.00048)

**Core Finding**: The brain contains multi-dimensional geometrical structures operating in as many as 11 dimensions. Neural circuits form high-dimensional simplicial complexes with topological cavities that appear during stimulus processing and then collapse.

**ModelCypher Implementation**: Our persistent homology implementation (`src/modelcypher/core/domain/geometry/topological_fingerprint.py`) computes the same Betti numbers used to characterize these structures.

---

### Brain-like Space: Unified Geometric Framework

**Paper**: Chen, S., et al. (2025). *A Unified Geometric Space Bridging AI Models and the Human Brain*. ([PDF](references/arxiv/Chen_2025_Unified_Geometric_Space_Bridging_AI_Models.pdf), [arXiv:2510.24342](https://arxiv.org/abs/2510.24342))

**Core Finding**: 151 Transformer-based models form a continuous arc-shaped geometry when mapped onto human functional brain networks.

**ModelCypher Implementation**: Our multi-model alignment (`src/modelcypher/core/domain/geometry/generalized_procrustes.py`) and curvature analysis (`src/modelcypher/core/domain/geometry/manifold_curvature.py`) enable positioning models in unified geometric spaces.

---

### Brain-AI Convergent Evolution

**Paper**: Shen, G., et al. (2025). *Alignment between Brains and AI: Evidence for Convergent Evolution across Modalities, Scales and Training Trajectories*. ([PDF](references/arxiv/Shen_2025_Alignment_Brains_AI_Evidence_Convergent_Evolution.pdf), [arXiv:2507.01966](https://arxiv.org/abs/2507.01966))

**Core Finding**: Analysis of 600+ AI models reveals that brain alignment *precedes* performance improvements during training. Language models show r=0.89 correlation between performance and brain alignment.

**ModelCypher Implementation**: Our intrinsic dimension tracking and training checkpoint analysis enable longitudinal geometry studies.

---

### Intrinsic Dimension and Abstraction Phases

**Papers**:
- Aghajanyan et al. (2021). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. ([arXiv:2012.13255](https://arxiv.org/abs/2012.13255))
- Cheng et al. (2025). *Emergence of a High-Dimensional Abstraction Phase in Language Transformers*. ([OpenReview](https://openreview.net/forum?id=0fD3iIBhlV))

**Core Findings**:
- Fine-tuning operates in very low intrinsic-dimension subspaces
- Mid-layer ID peaks correlate with abstraction and cross-model similarity
- Local ID decreases often precede capability gains

**ModelCypher Implementation**: `src/modelcypher/core/domain/geometry/intrinsic_dimension.py` provides TwoNN estimation with geodesic distances, per-layer dimension mapping, and deficiency detection.

---

### The Topology and Geometry of Neural Representations

**Paper**: Lin, B. & Kriegeskorte, N. (2024). *The topology and geometry of neural representations*. PNAS.
**DOI**: [10.1073/pnas.2317881121](https://doi.org/10.1073/pnas.2317881121)

**Core Finding**: Topological Representational Similarity Analysis (tRSA) provides robust comparison by focusing on topology rather than just geometry.

**ModelCypher Implementation**: Our Gromov-Wasserstein distance (`src/modelcypher/core/domain/geometry/gromov_wasserstein.py`) measures geometric similarity independent of coordinate systems.

---

### Cross-Species Neural Geometry

Evolution preserves representational geometry across primates despite differences in brain size and structure—suggesting the geometry is more fundamental than the hardware.

**ModelCypher Implementation**: Cross-model comparison tools enable testing whether this cross-species invariance extends to artificial systems.

---

### Wiring Cost and 3D Embedding Constraints

**Papers**:
- Rubinov, M. (2015). *Wiring cost and topological participation of the mouse brain connectome*. PNAS.
- PNAS (2024). *Human brain dynamics are shaped by rare long-range connections*.

**Core Finding**: Brain connectivity follows an exponential distance rule. Evolution minimizes "wiring cost," which **forces 3D embedding** of higher-dimensional optimal topologies.

**ModelCypher Implementation**: Our spatial 3D analysis (`src/modelcypher/core/domain/geometry/spatial_3d.py`) measures how models project concepts onto human-perceptual axes.

---

## Part II: Synthesis

These research threads converge on a unified hypothesis:

1. **Conceptual reality is intrinsically high-dimensional** (Blue Brain: 11+ dimensions)
2. **Physical brains are 3D projections** of this higher-dimensional manifold
3. **Neural networks converge to the same manifold** (Platonic Representation Hypothesis)
4. **The geometry is substrate-independent** (cross-species invariance, brain-AI alignment)
5. **Optimization naturally finds brain-like solutions** (convergent evolution)

### Implementation Summary

| Capability | File | Purpose |
|-----------|------|---------|
| Intrinsic Dimension | `intrinsic_dimension.py` | Measure actual dimensionality |
| CKA (Gram-based) | `cka.py` | Cross-dimensional comparison |
| Geodesic Distances | `riemannian_utils.py` | True manifold geometry |
| Persistent Homology | `topological_fingerprint.py` | Topological structure |
| Multi-Model Alignment | `generalized_procrustes.py` | Consensus geometry |
| Curvature Analysis | `manifold_curvature.py` | Manifold characterization |
| Gromov-Wasserstein | `gromov_wasserstein.py` | Coordinate-free comparison |

---

## Part III: Preliminary Measurements

The numeric summaries below are from local runs (2025-12-31) and are included as working notes.

### Dimensionality Collapse in SmolLM-360M

Using `mc geometry atlas dimensionality-study`:

| Layer | Mean Intrinsic Dimension |
|-------|-------------------------|
| 0     | 7.03                    |
| 4     | 6.08                    |
| 8     | 1.59                    |

**Collapse ratio**: 0.37 (63% reduction from peak to bottleneck)

This directly supports the "build then raze" hypothesis.

### Cross-Architecture Comparison: 6 Model Families

| Model | Architecture | Params | Bottleneck Dim | Cluster |
|-------|--------------|--------|----------------|---------|
| Qwen3-0.6B | Qwen3 | 600M | **1.52** | A |
| Qwen2.5-0.5B | Qwen2.5 | 500M | **1.56** | A |
| SmolLM-360M | EleutherAI | 360M | **1.59** | A |
| Llama-3.2-3B | Llama3 | 3B | **1.77** | A |
| Mistral-7B | Mistral | 7B | **2.56** | B |
| TinyLlama-1.1B | Llama | 1.1B | **2.72** | B |

**Key finding**: Two discrete bottleneck clusters rather than a continuum. Cluster membership is NOT determined by model size.

---

## Part IV: Testable Predictions

### P1: Quantized Bottleneck Clusters (CONFIRMED)
6 models tested, all fall into one of two clusters (1.6D or 2.6D).

### P2: Bottleneck Dimension is Scale-Invariant (CONFIRMED)
Model size does NOT predict bottleneck dimension.

### P3: Bottleneck Representations Are Cross-Architecturally Aligned
**Prediction**: CKA between bottleneck layers of different architectures > 0.7.
**Test**: `mc geometry baseline compare ModelA@bottleneck ModelB@bottleneck`

### P4: Bottleneck Position is Proportionally Consistent
**Prediction**: Bottleneck occurs at 40-60% of network depth across architectures.

### P5: Topological Invariants Match at Bottleneck
**Prediction**: Betti numbers (β₀, β₁, β₂) at bottleneck are similar across architectures.

### P6: Domain-Specific Structure Vanishes at Bottleneck
**Prediction**: All semantic domains converge to similar dimensionality at bottleneck.

### P7: Two Fundamental Representation Modes Exist

| Mode | Architecture | Bottleneck Position | Geometry |
|------|-------------|--------------------| ---------|
| Concentrated | SmolLM | 89% (final) | Lower orthogonality |
| Distributed | Qwen | 50% (middle) | High orthogonality |

Both modes converge to the same ~1.6D semantic bottleneck.

---

## Part V: Implementation Status

| Topic | External Reference | Status | ModelCypher Location |
|------|---------------------|--------|----------------------|
| WUDI interference | ICML 2025 | Implemented (metrics only) | `wudi_interference.py` |
| TSV-Merge | CVPR 2025 | Not implemented | — |
| Curvature signals | arXiv 2024 | Implemented (raw metrics) | `manifold_curvature.py` |
| Fisher/CAMEx | ICLR 2025 | Not implemented | — |
| Null-space filtering | MINGLE-like | Implemented | `geodesic_null_space.py` |
| Anchor-relative grafting | Moschella 2023 | Design note | See Future Directions below |

### Design Principles

- Parameter-space averaging and interpolation are not used for merging
- All thresholds are derived from data or machine epsilon
- Feature-space alignment transforms apply to activations; direct weight transforms require full layer basis change

---

## Part VI: Future Directions

### Thread 1: Anchor-Relative Concept Grafting

**The Problem**: Activation-space transforms (F = pinv(X_s) @ X_t) achieve CKA = 1.0 on probes, but applying F directly to weights breaks the target model.

**The Solution**: Anchor-relative coordinates (Moschella et al., 2023) provide a shared semantic address space. Alignment happens in anchor space, not feature space.

```
S_s = cos(A_s, C_s)       # Source anchor-relative representation
S_t = cos(A_t, C_t)       # Target anchor-relative representation
R = Procrustes(S_s, S_t)  # Align in anchor space
Delta_A = (density_weight * Delta_S) @ B  # Decode into target space
W_merged = W_target + P_null @ Delta_W    # Null-space constrained
```

**Implementation Touchpoints**: `relative_representation.py`, `gram_aligner.py`, `geodesic_null_space.py`

---

### Thread 2: Cross-LoRA Transfer

**The Dream**: Train a "coding adapter" for Llama-3 and reuse it on Qwen-2.5 without retraining.

**The Hypothesis**: Some fine-tuned behaviors correspond to transferable low-rank structure:
```
ΔW_target ≈ P^T · ΔW_source · P
```
Where P is the orthogonal Procrustes rotation derived from semantic primes.

**The Algorithm**:
1. Extract semantic prime activations for source and target
2. Find rotation R mapping source → target
3. Apply R to LoRA matrices A and B
4. Fine-tune on small calibration set

---

### Thread 3: Multi-Channel Architecture

ModelCypher's null-space projection and DeepSeek's Manifold-constrained Hyper-Connectivity (mHC) are mathematically related through invariant-preserving projections onto constrained manifolds.

| Property | Null-Space Projection | Birkhoff Projection (mHC) |
|----------|----------------------|---------------------------|
| Manifold | Orthogonal complement | Doubly stochastic matrices |
| Invariant | Boundary behavior | Total information flow |

Combined: multi-modal knowledge compression while maintaining CKA = 1.0.

---

### Thread 4: Experimental Insights from Geometry Probes

These patterns emerged from hypothesis-testing experiments and suggest extensions to existing infrastructure.

#### 4.1 Concepts as Probability Clouds

**Insight**: Using multiple phrasings per concept ("The number 5", "The value 5", "Consider 5") creates a distribution rather than a single point. This is more honest about what a "concept" actually is in latent space.

**Current State**: `ConceptVolume` exists in `riemannian_density.py` but isn't the default probing mode.

**Potential Extension**: Multi-prompt probing → ConceptVolume by default. Enables Mahalanobis distance (shape-aware) and Bhattacharyya overlap instead of point-to-point cosines.

#### 4.2 Relational Patterns Beyond Aggregate CKA

**Insight**: CKA says "overall structure matches" but doesn't reveal if specific relational patterns (hierarchies, composition triangles, oppositions) are preserved. Experiments testing whether `priv → G → pub` triangles have consistent structure suggest graph-level analysis.

**Potential Extension**: Relational pattern analyzer that checks if specific graph structures (A→B→C vs A→C directly) are preserved across models. Validates whether compositional reasoning transfers.

#### 4.3 Transformations as Near-Isometries

**Insight**: If a transformation preserves neighborhood structure (`geodesic(a_i, a_j) ≈ geodesic(T(a_i), T(a_j))`), it's a near-isometry—rotation/translation without distortion.

**Potential Extension**: LoRA isometry ratio. If a LoRA is near-isometric, it adds capability without warping existing structure. If not, it's overwriting. This distinguishes "extending" from "replacing."

#### 4.4 The "Generator IS the Transform" Pattern

**Insight**: If a model understands a conceptual transformation (like "double" or "negate"), that concept's embedding might BE the transformation direction. Experiments found `pub_concept - priv_concept` sometimes aligns with `generator_G_concept`.

**Potential Extension**: Given a conceptual operation ("make this more formal", "translate to code"), extract its direction by probing the concept itself, then apply it as a steering vector.

#### 4.5 Layer-wise ID for Probe Targeting

**Insight**: Intrinsic dimension follows entry-ramp → highway → exit-ramp:
- Early layers: High ID (building constraints)
- Mid layers: Low ID (maximum constraint, clearest structure)
- Late layers: Variable ID (task-specific)

**Current State**: Probing uses fixed layer or all layers.

**Potential Extension**: Adaptive layer selection based on ID profile. Focus geometry measurements on "highway" layers where structure is clearest.

#### 4.6 Geodesic Distance Reveals Hidden Structure

**Insight**: Experiments found that semantically related pairs are CLOSER in geodesic space than Euclidean distance suggests. Manifold structure surfaces relationships that raw cosine misses.

**Potential Extension**: Merge quality metrics using geodesic distance. If merged model has good CKA but wrong geodesic structure, generation will be incoherent despite passing aggregate tests.

---

### Thread 5: Novel Techniques from Script Mining (2026-01-29)

These techniques emerged from 284 research scripts (exp9-exp87) and show promise for integration.

#### 5.1 Distilled Logic Shapes

**Source**: `train_distilled_logic.py`

**Insight**: 10 perfect examples > 10,000 mediocre ones. The model needs to learn the **shape** of logic, not surface patterns.

**The 6 Logical Shapes**:
1. **PERCENTAGE INCREASE**: new = original + (original × percent)
2. **AVERAGE RATE**: total_output / total_input (NOT mean of rates)
3. **THRESHOLD CROSSING**: breakeven + 1 = first profitable
4. **INVERSE CHAIN**: work backwards, undo operations in reverse
5. **SEQUENTIAL OPERATIONS**: subtract first, THEN multiply
6. **REMAINING FIRST**: compute what's left BEFORE applying rate

**Implementation**: Create explicit templates for each shape with step-by-step reasoning. Repeat distilled examples 10x in training to give them weight.

#### 5.2 Counterfactual Sensitivity

**Source**: `counterfactual_sensitivity.py`, `geometric_knowledge_discovery.py`

**Insight**: Semantic invariance (paraphrase test) does NOT distinguish facts from opinions. But **counterfactual sensitivity** does.

**The Metric**:
- If model "knows" a fact, violating it changes the representation
- "2+2=4" vs "2+2=5" should be different IF the model knows math
- "Pizza is best" vs "Sushi is best" should be similar (both opinions)

**Results**:
- Factual statements: mean sensitivity ~0.25
- Opinion statements: mean sensitivity ~0.08
- Effect size: +0.94 (STRONG separation)

**Use Cases**:
- Detecting whether model has factual knowledge vs pattern matching
- Identifying truly missing capabilities vs disconnected ones
- Confidence calibration based on counterfactual stability

#### 5.3 Generation-Based Evaluation

**Source**: `exp86_proper_evaluation.py`, `exp87_generation_based_self_improvement.py`

**Insight**: Single-token evaluation creates a false ceiling at ~70%. Generation-based evaluation reveals true capability.

**The Gap**:
| Metric | Accuracy |
|--------|----------|
| Single-token | 70% |
| Generation-based | 90% |
| **Gap** | **+20pp** |

**Why It Matters**:
- Self-improvement using single-token metrics hits artificial ceiling
- Models reason correctly over multiple tokens but fail single-token prediction
- "Using only letters a,b,c" vs "using the full alphabet"

**Implementation**: Use `backend.generate()` with short max_tokens, check if expected substring appears in output.

#### 5.4 Geometry-Derived Training Parameters

**Source**: `geometry_derived_training.py`, `gsm8k_geometric_training.py`

**Insight**: ALL training parameters can be derived from geometry - no arbitrary constants needed.

**Derivations**:
```
Learning rate = 1 / (κ × scale)     # Stay in linear regime
Stop threshold = κ × √eps           # Achievable precision
Convergence = √eps                  # Numerical stability bound
```

Where:
- κ = condition number of Gram matrix
- scale = Frobenius norm of Gram matrix
- eps = machine epsilon for dtype

**Why It Works**: The geometry tells us how fast we can safely move (learning rate) and when we've reached the precision limit (stopping criterion).

---

### Research Roadmap

**Phase A**: Validate Anchor-Relative Transfer
- Test on same-architecture pairs (known to work)
- Test on cross-architecture pairs (LFM2-700M → LFM2-350M)

**Phase B**: Cross-LoRA Experiments
- Train coding adapter on Llama-3
- Project to Qwen-2.5 using Procrustes
- Measure rotation field roughness

**Phase C**: Multi-Channel World Model Compression
- Extract channel-specific activations from world models
- Compute per-channel null-space projections
- Learn doubly stochastic routing

---

## References

- Moschella et al. (2023). Relative representations enable zero-shot latent space communication.
- DeepSeek-AI. (2025). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880.
- Kornblith et al. (2019). Similarity of Neural Network Representations Revisited.
- ModelCypher Paper 0: The Shape of Knowledge (January 2026).

---

## Contributing

If you're working on related research, we welcome collaboration:

1. **Empirical validation**: Run our tools on your datasets
2. **Method development**: Extend our geometric primitives
3. **Cross-validation**: Compare our measurements to established baselines

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
