# DoRA Decomposition: Magnitude vs Direction

> Separating magnitude and direction to analyze fine-tuning geometry.

---

## Why This Matters for ModelCypher

DoRA shows that fine-tuning changes **magnitude** and **direction** differently.
ModelCypher uses this decomposition as an analysis tool to report how adapters
shift weight geometry, not to train DoRA adapters.

**In ModelCypher**: Implemented in `src/modelcypher/core/domain/geometry/dora_decomposition.py`
for magnitude/direction diagnostics on adapter deltas.

---

## Decomposition (Background)

Any weight matrix $W$ can be decomposed into magnitude and direction:

$$W = \|W\| \cdot \hat{W}$$

where $\|W\|$ is a magnitude (norm) and $\hat{W}$ is the unit-normalized direction.

---

## ModelCypher Analysis Metrics

For each layer, `DoRADecomposition` computes:
- **base_magnitude** and **current_magnitude** (geodesic norms)
- **magnitude_ratio**: $\|W_{current}\| / \|W_{base}\|$
- **direction_cosine**: geodesic cosine similarity
- **directional_drift**: $1 - \text{direction_cosine}$

Global metrics are parameter-count weighted averages:
- `overall_magnitude_change`
- `overall_directional_drift`
- `magnitude_to_direction_ratio`
- `dominant_change_type` (magnitude-dominated, direction-dominated, balanced, minimal)

All thresholds are derived from machine epsilon; there are no hardcoded cutoffs.

---

## Algorithm (ModelCypher)

1. For each weight tensor pair, compute geodesic norms and geodesic cosine.
2. Record per-layer magnitude and directional drift metrics.
3. Aggregate metrics weighted by parameter count.
4. Classify the dominant change type from the magnitude/direction ratio.

---

## Usage Example

```python
from modelcypher.core.domain.geometry.dora_decomposition import DoRADecomposition

analysis = DoRADecomposition().analyze_adapter(base_weights, adapted_weights)
print(analysis.overall_directional_drift)
print(analysis.dominant_change_type)
```

---

## Code Implementation

**Primary Location**: `src/modelcypher/core/domain/geometry/dora_decomposition.py`

**Key types**:
- `ChangeType`
- `MagnitudeDirectionMetrics`
- `DecompositionResult`
- `DoRADecomposition`

---

## Citations

1. **[Liu et al. (2024)](../../references/arxiv/Liu_2024_DoRA_WeightDecomposed_LowRank_Adaptation.pdf)**. "DoRA: Weight-Decomposed Low-Rank Adaptation." *ICML 2024*. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
2. **[Salimans & Kingma (2016)](../../references/arxiv/Salimans_2016_Weight_Normalization_Simple_Reparameterization_Accelerate_Training.pdf)**. "Weight Normalization: A Simple Reparameterization to Accelerate Training." *NeurIPS 2016*. [arXiv:1602.07868](https://arxiv.org/abs/1602.07868)
