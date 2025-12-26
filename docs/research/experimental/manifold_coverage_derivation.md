# Deriving Lower-Dimensional Structure from High-D Manifolds

> **Status**: Highly Experimental
> **Risk Level**: Speculative - theoretical foundations solid, empirical validation incomplete

---

## The Inverse Problem

Traditional dimensionality reduction projects high-D → low-D, losing information. We propose the inverse: **use high-dimensional manifold structure to derive what lower-dimensional representations should exist**.

The dimensional hierarchy principle:
```
1D (binary) ⊂ 2D (vocabulary) ⊂ 3D (spatial/Newtonian) ⊂ 4D+ (LLM geodesic manifolds)
```

Each level perfectly encodes the previous. The question: **can we read the encoding backwards?**

---

## Theoretical Basis

### Manifold Structure Encodes Dimensional Hierarchy

A point cloud on a high-D manifold M contains implicit structure at multiple scales:

1. **Local dimension**: Intrinsic dimension varies spatially - some regions are locally 2D, others 10D
2. **Directional sparsity**: Tangent sphere gaps reveal "missing" directions
3. **Geodesic coverage**: FPS identifies natural skeletal structure
4. **Dimension-collapsed zones**: Where local ID << global ID, the manifold "remembers" lower-D structure

**Hypothesis**: These features encode the lower-dimensional representations from which the high-D structure emerged.

### The Coverage-Derivation Connection

Consider an LLM's embedding space. Vocabulary embeddings (2D in our hierarchy) project to activation space (high-D). The projection is information-preserving but geometrically complex.

**Key insight**: Sparse regions and dimension-collapsed zones in high-D mark where the 2D vocabulary structure "surfaces" - like seeing the skeleton through skin.

---

## Methodology

### Phase 1: Coverage Analysis

Analyze the high-D manifold to identify structural features:

```python
from modelcypher.core.domain.geometry.manifold_coverage import ManifoldCoverage

coverage = ManifoldCoverage(backend)
analysis = coverage.analyze(high_d_points)

# Structural features that may encode lower-D
sparse_regions = analysis.sparse_points  # Under-sampled - possibly lower-D boundaries
dim_collapsed = analysis.dimension_deficient  # Local ID << global - lower-D "surfacing"
sparse_directions = analysis.sparse_directions  # Missing tangent directions
```

### Phase 2: Skeletal Extraction via FPS

Farthest point sampling extracts a geodesic skeleton - points that maximally span the manifold:

```python
from modelcypher.core.domain.geometry.riemannian_utils import farthest_point_sampling

# Extract skeletal points
skeleton_indices = farthest_point_sampling(points, n_samples=k)

# The skeleton approximates the "backbone" of the manifold
# Hypothesis: this backbone reflects lower-D generative structure
```

**Why FPS matters**: The geodesic skeleton is invariant to local density fluctuations. It captures the topological essence - what you'd get if you smoothed away high-D noise and looked at the underlying structure.

### Phase 3: Directional Analysis for Dimensional Derivation

At each point, analyze the tangent sphere to find "missing" directions:

```python
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

rg = RiemannianGeometry(backend)
coverage = rg.directional_coverage(point_idx, points, k=10)

# The sparse_direction points toward unexplored manifold regions
# Hypothesis: clustering of sparse directions reveals lower-D structure
```

**Experimental observation**: In vocabulary embeddings, sparse directions often align with semantic axes that correspond to 1D (binary) or 2D (vocabulary) structure in the hierarchy.

### Phase 4: Dimension Deficiency Mapping

Local intrinsic dimension reveals where the manifold "thins":

```python
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

estimator = IntrinsicDimension(backend)
dim_map = estimator.local_dimension_map(points, k=10)

# Points where local ID << modal ID
collapsed_zones = dim_map.deficient_indices

# These zones may be where lower-D structure is preserved
```

**Interpretation**: A 50-D embedding space might have modal dimension 12, but certain regions show local dimension 2-3. These dimension-collapsed zones are candidates for preserved lower-D structure.

---

## Experimental Protocol

### Experiment 1: Binary Structure Recovery

**Goal**: Recover 1D (binary) structure from vocabulary embeddings.

1. Compute local dimension map for embedding matrix
2. Identify dimension-deficient points (local ID < 2)
3. Extract tokens at those positions
4. **Hypothesis**: These tokens correspond to atomic/binary concepts (yes/no, true/false, etc.)

### Experiment 2: Vocabulary Plane Detection

**Goal**: Identify 2D vocabulary structure in activation space.

1. Run FPS to extract geodesic skeleton of activations
2. Compute pairwise geodesic distances on skeleton
3. Apply MDS to project to 2D
4. **Hypothesis**: The 2D projection preserves vocabulary relationships that exist in the original embedding space

### Experiment 3: Sparse Direction Clustering

**Goal**: Use directional sparsity to find semantic axes.

1. For each vocabulary embedding, compute sparse direction
2. Cluster sparse directions (geodesic k-means on unit sphere)
3. **Hypothesis**: Clusters correspond to semantic categories (the 2D vocabulary structure)

### Experiment 4: Manifold Filling for Derivation

**Goal**: Propose "missing" tokens from manifold gaps.

1. Identify sparse regions in embedding space
2. Propose fill points via tangent exploration
3. Find nearest existing tokens to proposed points
4. **Hypothesis**: Proposed points identify tokens that "should exist" based on the dimensional structure

---

## Implementation

### Core Components (Implemented)

| Module | Function | Purpose |
|--------|----------|---------|
| `riemannian_utils.py` | `farthest_point_sampling()` | Geodesic skeletal extraction |
| `riemannian_utils.py` | `directional_coverage()` | Tangent sphere sparsity analysis |
| `riemannian_utils.py` | `propose_in_sparse_direction()` | Tangent exploration for filling |
| `intrinsic_dimension.py` | `local_dimension_map()` | Per-point intrinsic dimension |
| `intrinsic_dimension.py` | `detect_dimension_deficiency()` | Dimension-collapsed zone detection |
| `manifold_coverage.py` | `ManifoldCoverage.analyze()` | Integrated coverage analysis |
| `manifold_coverage.py` | `ManifoldCoverage.propose_fills()` | Fill point generation |

### Usage Example

```python
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_coverage import ManifoldCoverage, CoverageConfig

backend = get_default_backend()

# Load vocabulary embeddings from model
embed_weights = model.get_weights()["model.embed_tokens.weight"]
embeddings = backend.array(embed_weights)

# Full coverage analysis
config = CoverageConfig(
    k_neighbors=15,
    density_percentile=0.1,  # Bottom 10% density = sparse
    dimension_threshold=0.5,  # Local ID < 50% of modal = deficient
    n_fps_samples=100,  # Skeletal extraction
    n_fill_proposals=20,  # Missing structure proposals
)

coverage = ManifoldCoverage(backend)
analysis = coverage.analyze(embeddings, config)

# Results for dimensional derivation
print(f"Sparse regions: {len(analysis.sparse_points)}")
print(f"Dimension-collapsed zones: {len(analysis.dimension_deficient)}")
print(f"Modal dimension: {analysis.modal_dimension:.1f}")
print(f"Coverage radius: {analysis.metrics.coverage_radius:.4f}")
print(f"Proposed fills: {len(analysis.proposed_fills)}")
```

---

## Theoretical Connections

### Connection to CKA Alignment

The phase-lock alignment (CKA = 1.0) ensures Gram matrix preservation across dimensions. This is the "upward" direction of the hierarchy - 1D → 2D → high-D.

Coverage analysis provides the "downward" reading - detecting where lower-D structure surfaces in high-D.

**Combined workflow**:
1. Align embeddings to achieve CKA = 1.0 (preserve relational geometry)
2. Analyze coverage to identify preserved lower-D structure
3. Use sparse regions to validate alignment quality

### Connection to Tangent Space Alignment

The `TangentSpaceAlignment` class computes principal angles between tangent spaces at different points. Directional coverage extends this: instead of comparing two tangent spaces, we analyze the coverage of a single tangent sphere.

**Key difference**: Principal angles measure inter-point tangent alignment; directional coverage measures intra-point tangent sampling.

### Connection to Fisher Information

The Fisher Information Matrix captures sensitivity to parameter changes. In the coverage framework:
- **High Fisher information** → sensitive region → likely dimension-collapsed
- **Low Fisher information** → stable region → likely high-D

Fisher blending could guide which sparse regions to prioritize for dimensional derivation.

---

## Risks and Limitations

### Methodological Risks

1. **Overfitting to noise**: Sparse regions might be noise, not structure
2. **Scale sensitivity**: Results depend heavily on k_neighbors choice
3. **Geodesic approximation**: k-NN graph may not capture true manifold structure
4. **Computational cost**: O(n²) for full coverage analysis

### Theoretical Risks

1. **Hierarchy assumption**: The 1D ⊂ 2D ⊂ 3D ⊂ 4D+ hierarchy is hypothetical
2. **Reversibility**: No proof that high-D → low-D derivation is unique or correct
3. **Semantic validity**: Derived structure may be geometrically valid but semantically meaningless

### Mitigation Strategies

1. **Bootstrap validation**: Subsample and check stability of results
2. **Cross-model comparison**: Compare derived structures across different models
3. **Ablation studies**: Vary k, density_percentile, dimension_threshold systematically
4. **Ground truth tests**: Use synthetic data with known dimensional structure

---

## Future Directions

### Short-term

1. Validate on synthetic data with known 2D → high-D projections
2. Compare FPS skeleton structure across model families
3. Correlate dimension-collapsed zones with semantic categories

### Medium-term

1. Use derived structure to improve model merging
2. Detect "missing" concepts in vocabulary from sparse regions
3. Guide LoRA adapter placement using coverage analysis

### Long-term

1. Automatic discovery of semantic axes from directional sparsity
2. Manifold-aware tokenizer design (fill gaps in vocabulary coverage)
3. Geometric model interpretability via dimensional derivation

---

## References

### Manifold Learning

1. **Tenenbaum, J.B., et al.** (2000). "A Global Geometric Framework for Nonlinear Dimensionality Reduction." *Science*, 290(5500), 2319-2323.
   - *Isomap: geodesic-based dimensionality reduction*

2. **Facco, E., et al.** (2017). "Estimating the intrinsic dimension of datasets by a minimal neighborhood information." *Scientific Reports*, 7, 12140.
   - *TwoNN method for intrinsic dimension*

### Farthest Point Sampling

3. **Eldar, Y., et al.** (1997). "The Farthest Point Strategy for Progressive Image Sampling." *IEEE Trans. Image Processing*, 6(9), 1305-1315.
   - *Original FPS algorithm*

4. **Qi, C.R., et al.** (2017). "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space." *NeurIPS 2017*.
   - *FPS for point cloud processing*

### Tangent Space Analysis

5. **Pennec, X.** (2006). "Intrinsic Statistics on Riemannian Manifolds." *JMIV*, 25(1), 127-154.
   - *Tangent space statistics*

### Neural Network Geometry

6. **Ansuini, A., et al.** (2019). "Intrinsic dimension of data representations in deep neural networks." *NeurIPS 2019*.
   - *ID variation across layers*

7. **Recanatesi, S., et al.** (2019). "Dimensionality compression and expansion in deep neural networks." *arXiv:1906.00443*.
   - *Dimensional dynamics in networks*

---

## Code Location

- Primary: [manifold_coverage.py](../../../src/modelcypher/core/domain/geometry/manifold_coverage.py)
- Supporting: [riemannian_utils.py](../../../src/modelcypher/core/domain/geometry/riemannian_utils.py)
- Supporting: [intrinsic_dimension.py](../../../src/modelcypher/core/domain/geometry/intrinsic_dimension.py)
- Tests: [test_manifold_coverage.py](../../../tests/test_manifold_coverage.py)

---

*This is experimental work. The dimensional hierarchy hypothesis is speculative. Results should be validated carefully before drawing conclusions about semantic structure.*
