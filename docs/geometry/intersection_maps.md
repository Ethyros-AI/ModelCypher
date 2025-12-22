# Theory: Intersection Maps & Semantic Overlap

> **Status**: Core Theory
> **Reference**: `src/modelcypher/core/domain/geometry/manifold_stitcher.py`

## The Concept

The **Intersection Map** is a formalization of the "overlapping knowledge" between two disparate Neural Networks. It is the geometric equivalent of a Venn Diagram for high-dimensional vector spaces.

### Fundamental Assumption
Two models trained on the same data (the internet) will learn to encode similar **semantic invariants** (concepts like "King", "Queen", "Dog"), even if they encode them in different locations or orientations.

Therefore, there may exist an approximately shared subspace $\mathcal{S}$ such that:
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
-   **Strong Correlation (> 0.7)**: A common heuristic threshold suggesting dimensions $i$ and $j$ behave similarly on the probe corpus. (Thresholds should be calibrated per architecture/prompt set.)
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

1.  **Early Layers (0-5)**: Often higher overlap. Basic token/formatting features can be more similar across families.
2.  **Middle Layers (5-20)**: Divergence is common. Higher-level features can differ due to architecture, tokenizer, training mix, and fine-tuning.
3.  **Late Layers (20+)**: Convergence sometimes appears due to the shared objective of producing logits, but it is not guaranteed.

## Applications

Understanding the Intersection Map allows us to:
1.  **Merge Disparate Models**: Only averages weights in the $\mathcal{S}$ subspace, avoiding destructive interference in disjoint regions.
2.  **Transfer Learning**: Stitch an adapter trained on Model A onto Model B by aligning it to the intersection.
3.  **Drift Detection**: If the Intersection Map between a base model and a fine-tuned variant shrinks drastically, it may indicate representational drift or loss of overlap that warrants further investigation.
