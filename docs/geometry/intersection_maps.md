# Theory: Intersection Maps & Representation Overlap

> **Status**: Core Theory
> **References**:
> - `src/modelcypher/core/domain/geometry/manifold_stitcher.py`
> - `src/modelcypher/core/domain/geometry/intersection_similarity.py`

## The Concept

The **Intersection Map** is a diagnostic of **representation overlap** between two models under a fixed probe setup.

> **Analogy (intuition)**: a “Venn diagram” of overlap.
>
> **Operationalization**: overlap is computed from **activation fingerprints** (semantic prime probes), not from weights or raw logits.

### Working assumption

Two models trained on broadly similar data may encode partially similar features, even if they encode them in different coordinates.

Conceptually, there may exist an approximately shared subspace:

```
M_A ∩ M_B ≈ S
```

where M_A and M_B are activation manifolds induced by the probe corpus and capture method.

## Computing the Map

We do not compare weights directly. We compare **activation fingerprints** for the same probe set.

1. **Probe**: Run semantic-prime probes on both models.
2. **Fingerprint**: Build `ActivationFingerprint` per probe (activated dimensions per layer).
3. **Match**: For each layer, compare source/target activated dimensions using a similarity mode.

### Similarity Modes

The intersection map uses similarity metrics over sparse activation signatures:
- **JACCARD**: set overlap of activated dimensions.
- **WEIGHTED_JACCARD**: magnitude-weighted overlap.
- **CKA**: cosine² on sparse activation vectors.

Each source dimension is paired with the target dimension that yields the **best similarity** in that layer.

### Output Structure

```python
@dataclass
class IntersectionMap:
    source_model: str
    target_model: str
    dimension_correlations: dict[int, list[DimensionCorrelation]]
    raw_fingerprint_similarity: float
    aligned_dimension_count: int
    total_source_dims: int
    total_target_dims: int
    layer_confidences: list[LayerConfidence]
```

`raw_fingerprint_similarity` is **pre-alignment only**. Use CKA separately for post-alignment quality checks.

## Layer-wise Dynamics

The Intersection Map evolves across depth because fingerprints are tracked per layer:

1. **Early layers**: often higher overlap (shared token/formatting features).
2. **Middle layers**: divergence is common (architecture and data differences).
3. **Late layers**: convergence may appear due to shared logit objectives, but it is not assured.

## Applications

Understanding the Intersection Map allows us to:
1. **Guide merging**: focus on regions with measured overlap to reduce interference.
2. **Support transfer**: decide when adapter or feature transfer is plausible.
3. **Detect drift**: large drops in overlap can indicate representational drift that warrants follow-up evaluation.
