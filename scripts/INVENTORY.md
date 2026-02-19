# Scripts Inventory

Active exploration scripts. For archived experiments, see `/Volumes/CodeCypher/archive/modelcypher-scripts/`.

---

## Current Scripts

| Script | Purpose | CLI Equivalent |
|--------|---------|----------------|
| `explore_expansion_trajectories.py` | Layer-wise norm tracking, expansion_ratio per task | `mc analyze spectral-trajectory` (different method) |
| `hidden_state_analysis.py` | EffDim, kurtosis, sparsity per layer | `mc analyze dimension-profile` (different method) |
| `layer_contribution_analysis.py` | Compression gate detection | **None - promote to CLI** |
| `final_layer_weight_analysis.py` | Weight matrix rank/sparsity | **None - promote to CLI** |
| `exp_soft_null_space.py` | Soft null-space projection experiments | N/A (experimental) |
| `spectral_capacity_domain_rank_investigation.py` | Weight spectral gap ratios at domain rank positions | `mc model capacity` (subset) |
| `spectral_energy_curves.py` | Full energy curves + inflection points from weight spectra | N/A (research) |
| `activation_spectral_analysis.py` | Activation-space SVD per domain group (Gram eigendecomp) | N/A (research) |

---

## Usage

```bash
# Layer-by-layer norm trajectory (expansion_ratio analysis)
poetry run python scripts/explore_expansion_trajectories.py --model /path/to/model

# Hidden state metrics (effective dimension, kurtosis)
poetry run python scripts/hidden_state_analysis.py --model /path/to/model

# Layer role analysis (which layers compress vs expand)
poetry run python scripts/layer_contribution_analysis.py data/experiments/trajectory_*.json

# Weight matrix analysis (effective rank, sparsity)
poetry run python scripts/final_layer_weight_analysis.py /path/to/model1 /path/to/model2
```

---

## Relationship to CLI

These scripts complement the existing CLI:

| CLI Command | Method | Script Alternative |
|-------------|--------|-------------------|
| `mc analyze spectral-trajectory` | SVD spectral entropy | `explore_expansion_trajectories.py` (L2 norms) |
| `mc analyze dimension-profile` | TwoNN intrinsic dimension | `hidden_state_analysis.py` (participation ratio) |
| `mc analyze expansion-ratio` | TwoNN peak/final ratio | `explore_expansion_trajectories.py` (norm peak/final) |

The scripts use simpler/faster methods that correlate with the CLI's more principled methods.
