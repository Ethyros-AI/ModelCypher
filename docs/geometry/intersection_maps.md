# Theory: Intersection Maps & Semantic Overlap

> **Status**: Core Theory
> **Reference**: `src/modelcypher/core/domain/geometry/manifold_stitcher.py`

## The Concept

The **Intersection Map** is a formalization of the "overlapping knowledge" between two disparate Neural Networks. It is the geometric equivalent of a Venn Diagram for high-dimensional vector spaces.

### Fundamental Assumption
Two models trained on the same data (the internet) will learn to encode similar **semantic invariants** (concepts like "King", "Queen", "Dog"), even if they encode them in different locations or orientations.

Therefore, there exists a shared subspace $\mathcal{S}$ such that:
$$ \mathcal{M}_A \cap \mathcal{M}_B = \mathcal{S} $$

Where $\mathcal{M}_A$ and $\mathcal{M}_B$ are the knowledge manifolds of Model A and Model B.

## Computing the Map

We cannot compare weights directly. We must compare **activations** on identical inputs.

1.  **Probe**: Feed both models a "Key" input $X$ (e.g., "The quick brown fox...").
2.  **Act**: Capture activation vectors $a = M_A(X)$ and $b = M_B(X)$.
3.  **Correlate**: Compute the pairwise correlation matrix between the neurons of $A$ and $B$.

$$ C_{ij} = \text{corr}(a_i, b_j) $$

### The "Aligned" Subspace
We filter this matrix for strong correlations.
-   **Strong Correlation (> 0.7)**: Dimensions $i$ and $j$ represent the same concept. They are part of the Intersection.
-   **Weak Correlation**: Concepts unique to one model (or encoded in a basis we haven't found).

The `IntersectionMap` dataclass captures this:
```python
@dataclass
class IntersectionMap:
    source_model: str
    target_model: str
    dimension_correlations: dict[int, list[DimensionCorrelation]]
    overall_correlation: float
    aligned_dimension_count: int
```

## Layer-wise Dynamics

The Intersection Map evolves as data flows through the model depth.

1.  **Early Layers (0-5)**: High overlap. Basic syntax and token processing is universal.
2.  **Middle Layers (5-20)**: Divergence. Models process abstract reasoning differently. The Venn circles move apart.
3.  **Late Layers (20+)**: Convergence (sometimes). Models must agree on the final output token distribution, forcing manifolds to realign (logit bottleneck).

## Applications

Understanding the Intersection Map allows us to:
1.  **Merge Disparate Models**: Only averages weights in the $\mathcal{S}$ subspace, avoiding destructive interference in disjoint regions.
2.  **Transfer Learning**: Stitch an adapter trained on Model A onto Model B by aligning it to the intersection.
3.  **Drift Detection**: If the Intersection Map between a Base model and an RLHF model shrinks drastically, the RLHF process has caused "Catastrophic Forgetting" (destroyed knowledge).
