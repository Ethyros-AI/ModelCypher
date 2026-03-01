# Scripts Inventory

Active exploration scripts. For archived experiments, see `/Volumes/CodeCypher/archive/modelcypher-scripts/`.

---

## Current Scripts

| Script | Purpose | CLI Equivalent |
|--------|---------|----------------|
| `layer_contribution_analysis.py` | Compression gate detection | **None - promote to CLI** |
| `spectral_capacity_domain_rank_investigation.py` | Weight spectral gap ratios at domain rank positions | `mc model capacity` (subset) |
| `spectral_energy_curves.py` | Full energy curves + inflection points from weight spectra | N/A (research) |
| `activation_spectral_analysis.py` | Activation-space SVD per domain group (Gram eigendecomp) | N/A (research) |

> **Note (2026-03-01):** `explore_expansion_trajectories.py`, `hidden_state_analysis.py`, `final_layer_weight_analysis.py`, and `exp_soft_null_space.py` were removed. Their functionality is covered by CLI commands (`mc analyze spectral-trajectory`, `mc analyze dimension-profile`). See `/Volumes/CodeCypher/archive/modelcypher-scripts/` for archived versions.

---

## Usage

```bash
# Layer role analysis (which layers compress vs expand)
poetry run python scripts/layer_contribution_analysis.py data/experiments/trajectory_*.json

# Weight spectral gap analysis
poetry run python scripts/spectral_capacity_domain_rank_investigation.py --model /path/to/model
```

---

## Relationship to CLI

These scripts complement the existing CLI:

| CLI Command | Method | Script Alternative |
|-------------|--------|-------------------|
| `mc analyze spectral-trajectory` | SVD spectral entropy | (archived: `explore_expansion_trajectories.py`) |
| `mc analyze dimension-profile` | TwoNN intrinsic dimension | (archived: `hidden_state_analysis.py`) |
| `mc model capacity` | Per-layer spectral budget | `spectral_capacity_domain_rank_investigation.py` |

Most early scripts have been superseded by CLI commands with more principled methods.
