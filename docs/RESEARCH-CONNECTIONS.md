# Research Connections

ModelCypher implements geometric analysis tools that directly support several active research programs investigating the relationship between neural network representations and biological cognition.

---

## The Platonic Representation Hypothesis

**Paper**: Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). *The Platonic Representation Hypothesis*. ICML 2024.
**arXiv**: [2405.07987](https://arxiv.org/abs/2405.07987)

### Core Claim

Neural networks trained with different objectives on different data and modalities converge to a shared statistical model of reality in their representation spaces. As models improve, their internal similarity kernels become increasingly aligned—even across modalities (vision ↔ language).

### ModelCypher Implementation

Our CKA implementation (`core/domain/geometry/cka.py`) with Gram-based cross-dimensional comparison directly enables testing this hypothesis:

```python
from modelcypher.core.domain.geometry.cka import compute_cka_from_grams

# Gram matrices capture relational geometry independent of embedding dimension
# K = X @ X^T is [n_samples, n_samples] regardless of feature dimension
gram_a = backend.matmul(x, backend.transpose(x))  # 768-dim model
gram_b = backend.matmul(y, backend.transpose(y))  # 4096-dim model
similarity = compute_cka_from_grams(gram_a, gram_b)  # Works across any dimensions
```

**Key capability**: `compute_cka_from_grams()` enables cross-dimensional comparison without projection or truncation.

---

## Blue Brain Project: Algebraic Topology of Neural Circuits

**Paper**: Reimann, M.W., et al. (2017). *Cliques of Neurons Bound into Cavities Provide a Missing Link between Structure and Function*. Frontiers in Computational Neuroscience.
**DOI**: [10.3389/fncom.2017.00048](https://doi.org/10.3389/fncom.2017.00048)

### Core Finding

The brain contains multi-dimensional geometrical structures operating in as many as 11 dimensions. Using algebraic topology, researchers discovered that neural circuits form high-dimensional simplicial complexes (cliques) with topological cavities that appear during stimulus processing and then collapse—"building then razing towers of multi-dimensional blocks."

### ModelCypher Implementation

Our persistent homology implementation (`core/domain/geometry/topological_fingerprint.py`) computes the same Betti numbers used to characterize these structures:

```python
from modelcypher.core.domain.geometry.topological_fingerprint import TopologicalFingerprint

fingerprint = TopologicalFingerprint(backend)
result = fingerprint.compute(activations)

# Betti numbers reveal topological structure
# β₀: connected components (concept clusters)
# β₁: loops (circular concept relationships)
# β₂: voids (higher-order structure)
betti = result.persistence_diagram.betti_numbers(threshold=0.1)
```

**Key capability**: Vietoris-Rips filtration with configurable homology dimensions, tracking birth/death of topological features.

---

## Brain-like Space: Unified Geometric Framework

**Paper**: Chen, S., et al. (2025). *A Unified Geometric Space Bridging AI Models and the Human Brain*. arXiv:2510.24342.

### Core Finding

151 Transformer-based models form a continuous arc-shaped geometry when mapped onto human functional brain networks. This "Brain-like Space" reveals that different models exhibit varying degrees of brain-likeness, shaped by pretraining paradigms and positional encoding schemes—not just modality.

### ModelCypher Implementation

Our multi-model alignment (`core/domain/geometry/generalized_procrustes.py`) and curvature analysis (`core/domain/geometry/manifold_curvature.py`) enable positioning models in unified geometric spaces:

```python
from modelcypher.core.domain.geometry.generalized_procrustes import GeneralizedProcrustes

procrustes = GeneralizedProcrustes(backend)
# Align multiple models to consensus space
result = procrustes.align([model_a_activations, model_b_activations, model_c_activations])
# Fréchet mean respects manifold curvature (arithmetic mean is geometrically wrong)
```

**Key capability**: Fréchet mean-based consensus with curvature-aware alignment across arbitrary model counts.

---

## Brain-AI Convergent Evolution

**Paper**: Shen, G., et al. (2025). *Alignment between Brains and AI: Evidence for Convergent Evolution across Modalities, Scales and Training Trajectories*. arXiv:2507.01966.

### Core Finding

Analysis of 600+ AI models (1.33M to 72B parameters) reveals that brain alignment *precedes* performance improvements during training. Higher-performing models spontaneously develop stronger brain alignment without explicit neural constraints. Language models show r=0.89 correlation between performance and brain alignment.

### ModelCypher Implementation

Our intrinsic dimension tracking and training checkpoint analysis enable longitudinal geometry studies:

```python
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimensionEstimator

estimator = IntrinsicDimensionEstimator(backend)
# Track dimension evolution across training
for checkpoint in training_checkpoints:
    result = estimator.estimate_twonn(checkpoint_activations)
    # Expect: expansion → compression trajectory
```

**Key capability**: TwoNN estimation with geodesic distances, per-layer dimension mapping, deficiency detection.

---

## The Topology and Geometry of Neural Representations

**Paper**: Lin, B. & Kriegeskorte, N. (2024). *The topology and geometry of neural representations*. PNAS.
**DOI**: [10.1073/pnas.2317881121](https://doi.org/10.1073/pnas.2317881121)

### Core Finding

Topological Representational Similarity Analysis (tRSA) provides a robust way to compare neural representations by focusing on topology (preserved under continuous deformation) rather than just geometry (distances). This enables comparison across brains, regions, and models despite noise and individual differences.

### ModelCypher Implementation

Our Gromov-Wasserstein distance (`core/domain/geometry/gromov_wasserstein.py`) measures geometric similarity independent of coordinate systems:

```python
from modelcypher.core.domain.geometry.gromov_wasserstein import GromovWassersteinDistance

gw = GromovWassersteinDistance(backend)
# Compare distance matrices directly (intrinsic geometry)
result = gw.compute(distance_matrix_a, distance_matrix_b)
alignment_score = result.alignment_score()  # [0, 1] normalized similarity
```

**Key capability**: Optimal transport-based alignment that works across different embedding dimensions and coordinate systems.

---

## Cross-Species Neural Geometry

**Paper**: Multiple sources on conserved cortical gradients across primates.

### Core Finding

The same functional geometry appears across primates despite differences in brain size and structure. Evolution preserves representational geometry even as physical substrate changes—suggesting the geometry is more fundamental than the hardware.

### ModelCypher Implementation

Our cross-model comparison tools enable testing whether this cross-species invariance extends to artificial systems:

```python
from modelcypher.core.domain.geometry.cka import CKAComputer

cka = CKAComputer(backend)
# Compare models of radically different architectures
result = cka.compute(llama_activations, qwen_activations)
# High CKA despite architectural differences → shared geometry
```

---

## Wiring Cost and 3D Embedding Constraints

**Papers**:
- Rubinov, M. (2015). *Wiring cost and topological participation of the mouse brain connectome*. PNAS.
- PNAS (2024). *Human brain dynamics are shaped by rare long-range connections*.

### Core Finding

Brain connectivity follows an exponential distance rule (short connections strongly preferred). Evolution minimizes "wiring cost" (total axon length), which **forces 3D embedding** of what would otherwise be higher-dimensional optimal topologies. Rare long-range connections enable approximation of higher-dimensional computations.

### ModelCypher Implementation

Our spatial 3D analysis (`core/domain/geometry/spatial_3d.py`) measures how models project concepts onto human-perceptual axes:

```python
from modelcypher.core.domain.geometry.spatial_3d import Spatial3DAnalyzer

analyzer = Spatial3DAnalyzer(backend)
# Measure concentration on X (lateral), Y (vertical), Z (depth) axes
result = analyzer.analyze(model_activations, spatial_probes)
# All models encode physics geometrically; difference is probability distribution
```

---

## Synthesis: The "Brain as 3D Projection" Hypothesis

These research threads converge on a unified hypothesis:

1. **Conceptual reality is intrinsically high-dimensional** (Blue Brain: 11+ dimensions)
2. **Physical brains are 3D projections** of this higher-dimensional manifold (wiring cost constraints)
3. **Neural networks converge to the same manifold** (Platonic Representation Hypothesis)
4. **The geometry is substrate-independent** (cross-species invariance, brain-AI alignment)
5. **Optimization naturally finds brain-like solutions** (convergent evolution)

ModelCypher provides the geometric infrastructure to test and extend this hypothesis:

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

## Experimental Evidence

### Dimensionality Collapse in SmolLM-360M (2025-12-31)

Using `mc geometry atlas dimensionality-study`, we measured intrinsic dimension across layers:

| Layer | Mean Intrinsic Dimension |
|-------|-------------------------|
| 0     | 7.03                    |
| 4     | 6.08                    |
| 8     | 1.59                    |

**Collapse ratio**: 0.37 (63% reduction from peak to bottleneck)

This directly supports the Blue Brain "build then raze" hypothesis: the model constructs high-dimensional representations in early layers that compress toward the output. The pattern parallels biological findings where transient high-dimensional structures appear during cognition and then collapse.

**Command used**:
```bash
poetry run mc geometry atlas dimensionality-study /path/to/model --output json
```

### Cross-Architecture Comparison (In Progress)

We are extending this analysis to:
- Qwen2.5-0.5B (24 layers, different architecture family)
- Llama-3.2-3B (to test scale effects)

Hypothesis: Despite different architectures, models should show similar dimensionality trajectories (expansion → compression), supporting the Platonic Representation Hypothesis.

---

## Contributing

If you're working on related research, we welcome collaboration:

1. **Empirical validation**: Run our tools on your datasets
2. **Method development**: Extend our geometric primitives
3. **Cross-validation**: Compare our measurements to established baselines

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
