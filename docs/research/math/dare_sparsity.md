# DARE: Drop And REscale (Sparsity Analysis)

> Sparse delta parameters for efficient model merging.

---

## Background (DARE in the Literature)

DARE starts from fine-tuned deltas:

$$\Delta\theta = \theta_{ft} - \theta_{pre}$$

The original algorithm randomly drops and rescales deltas:

$$M_i \sim \text{Bernoulli}(1 - p)$$
$$\tilde{\Delta\theta} = M \odot \Delta\theta$$
$$\hat{\Delta\theta} = \tilde{\Delta\theta} / (1 - p)$$
$$\theta_{DARE} = \theta_{pre} + \hat{\Delta\theta}$$

This preserves the expected delta while enforcing sparsity.

---

## ModelCypher Implementation

ModelCypher does **not** apply random drop/rescale. Instead, it analyzes delta
sparsity to quantify how much of a delta could be dropped without invoking
arbitrary thresholds. The implementation is deterministic and data-derived.

### Threshold Derivation (No Heuristics)

`DARESparsityAnalyzer` derives thresholds from the magnitude distribution:
- **Zero threshold**: `max_magnitude * machine_epsilon` (numerical noise floor)
- **Gap threshold**: largest relative gap in the sorted magnitude spectrum
- **Drop threshold**: `max(zero_threshold, gap_threshold)`

These thresholds yield:
- **effective_sparsity**: fraction of deltas below the drop threshold
- **essential_fraction**: `1 - effective_sparsity`

---

## Analysis Algorithm (ModelCypher)

1. Compute per-layer delta magnitudes.
2. Aggregate magnitudes and compute summary statistics.
3. Find the natural magnitude gap using `find_magnitude_gap_threshold`.
4. Report effective sparsity and per-layer metrics.

---

## Outputs

`SparsityAnalysis` includes:
- `effective_sparsity`: droppable fraction by data-derived threshold
- `essential_fraction`: retained fraction
- `per_layer_sparsity`: per-layer stats (mean, max, essential fraction)
- `magnitude_stats`: global distribution stats

---

## Usage Example

```python
from modelcypher.core.domain.geometry.dare_sparsity import DARESparsityAnalyzer

# delta_weights: dict[layer_name, list[float]] or backend arrays
analysis = DARESparsityAnalyzer.analyze(delta_weights)
print(analysis.effective_sparsity)
```

For backend arrays (GPU), use:

```python
analysis = DARESparsityAnalyzer.analyze_with_backend(delta_weights)
```

---

## Code Implementation

**Primary Location**: `src/modelcypher/core/domain/geometry/dare_sparsity.py`

**Key classes**:
- `LayerSparsityMetrics`
- `MagnitudeStatistics`
- `SparsityAnalysis`
- `DARESparsityAnalyzer`

---

## Citations

1. **[Yu et al. (2024)](../../references/arxiv/Yu_2023_Language_Models_are_Super_Mario_Absorbing.pdf)**. "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)." *ICML 2024*. [arXiv:2311.03099](https://arxiv.org/abs/2311.03099)
