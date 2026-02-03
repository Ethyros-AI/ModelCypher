#!/usr/bin/env python3
"""
Geometric analysis of MIT Open Density Limit Database.

This script applies our LLM-derived geometry tools to REAL tokamak data
from Alcator C-Mod to see if disruption precursors have geometric signatures.

The hypothesis: disruption precursors (density limit phase) should have
different geometric characteristics than stable plasma operation.

Run from ModelCypher root:
    python plasma/notebooks/02_mit_density_limit_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add plasma src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geometry_tools import (
    compute_expansion_ratio,
    compute_local_dimension,
    compute_spectral_entropy,
)


def load_mit_density_limit_data() -> pd.DataFrame:
    """Load the MIT Open Density Limit Database."""
    data_path = Path(__file__).parent.parent / "data" / "mit_density_limit" / "data" / "DL_DataFrame.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"MIT data not found at {data_path}. "
            "Run: cd plasma/data && git clone https://github.com/MIT-PSFC/open_density_limit_database.git mit_density_limit"
        )

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} time points from {df['discharge_ID'].nunique()} discharges")
    return df


def extract_shot_trajectory(df: pd.DataFrame, shot_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract trajectory for a single shot.

    Returns:
        trajectory: [T, D] array of diagnostic state
        time: [T] array of time points
        labels: [T] array of density_limit_phase labels
    """
    shot_df = df[df['discharge_ID'] == shot_id].sort_values('time')

    # Feature columns (exclude ID, time, and label)
    feature_cols = ['density', 'elongation', 'minor_radius',
                    'plasma_current', 'toroidal_B_field', 'triangularity']

    trajectory = shot_df[feature_cols].values
    time = shot_df['time'].values
    labels = shot_df['density_limit_phase'].values

    return trajectory, time, labels


def analyze_shot_geometry(trajectory: np.ndarray, time: np.ndarray, labels: np.ndarray) -> dict:
    """Compute geometric features for a shot."""
    # Normalize trajectory for stable numerical computation
    trajectory_norm = (trajectory - trajectory.mean(axis=0)) / (trajectory.std(axis=0) + 1e-10)

    # Compute geometric features
    expansion = compute_expansion_ratio(trajectory_norm, window_size=5)
    local_dim = compute_local_dimension(trajectory_norm, window_size=20, method="eigenvalue")
    spectral_ent = compute_spectral_entropy(trajectory_norm, window_size=20)

    # Split by label (stable=0, density_limit_precursor=1)
    # Note: expansion is one element shorter than trajectory
    stable_mask = labels[1:-1] == 0
    precursor_mask = labels[1:-1] == 1

    # Get valid (non-NaN) indices for dimension and entropy
    valid_dim = ~np.isnan(local_dim)
    valid_ent = ~np.isnan(spectral_ent)

    results = {
        "n_timesteps": len(trajectory),
        "n_stable": int(stable_mask.sum()),
        "n_precursor": int(precursor_mask.sum()),
    }

    # Expansion ratio statistics
    if stable_mask.any():
        results["expansion_stable_mean"] = float(np.mean(expansion[stable_mask]))
        results["expansion_stable_std"] = float(np.std(expansion[stable_mask]))
    if precursor_mask.any():
        results["expansion_precursor_mean"] = float(np.mean(expansion[precursor_mask]))
        results["expansion_precursor_std"] = float(np.std(expansion[precursor_mask]))

    # Local dimension statistics (adjust indices for dimension array)
    dim_stable_mask = valid_dim & (labels == 0)
    dim_precursor_mask = valid_dim & (labels == 1)

    if dim_stable_mask.any():
        results["dimension_stable_mean"] = float(np.nanmean(local_dim[dim_stable_mask]))
    if dim_precursor_mask.any():
        results["dimension_precursor_mean"] = float(np.nanmean(local_dim[dim_precursor_mask]))

    # Spectral entropy statistics
    ent_stable_mask = valid_ent & (labels == 0)
    ent_precursor_mask = valid_ent & (labels == 1)

    if ent_stable_mask.any():
        results["entropy_stable_mean"] = float(np.nanmean(spectral_ent[ent_stable_mask]))
    if ent_precursor_mask.any():
        results["entropy_precursor_mean"] = float(np.nanmean(spectral_ent[ent_precursor_mask]))

    return results


def main():
    print("=" * 70)
    print("GEOMETRIC ANALYSIS OF REAL TOKAMAK DATA")
    print("MIT Open Density Limit Database - Alcator C-Mod")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = load_mit_density_limit_data()

    # Basic statistics
    print(f"\n2. Data overview:")
    print(f"   Total time points: {len(df):,}")
    print(f"   Unique discharges: {df['discharge_ID'].nunique()}")
    print(f"   Stable points (label=0): {(df['density_limit_phase'] == 0).sum():,}")
    print(f"   Precursor points (label=1): {(df['density_limit_phase'] == 1).sum():,}")
    print(f"\n   Features: {', '.join(['density', 'elongation', 'minor_radius', 'plasma_current', 'toroidal_B_field', 'triangularity'])}")

    # Analyze individual shots
    print("\n3. Analyzing shots with both stable and precursor phases...")

    shot_ids = df['discharge_ID'].unique()

    # Find shots that have BOTH stable and precursor phases
    mixed_shots = []
    for shot_id in shot_ids:
        shot_df = df[df['discharge_ID'] == shot_id]
        if shot_df['density_limit_phase'].nunique() == 2:
            mixed_shots.append(shot_id)

    print(f"   Found {len(mixed_shots)} shots with both phases")

    # Analyze each shot
    all_results = []
    for shot_id in mixed_shots[:50]:  # Analyze first 50 mixed shots
        try:
            trajectory, time, labels = extract_shot_trajectory(df, shot_id)
            if len(trajectory) < 30:  # Skip very short shots
                continue
            results = analyze_shot_geometry(trajectory, time, labels)
            results["shot_id"] = shot_id
            all_results.append(results)
        except Exception as e:
            print(f"   Warning: Shot {shot_id} failed: {e}")

    print(f"   Successfully analyzed {len(all_results)} shots")

    # Aggregate results
    print("\n4. GEOMETRIC SIGNATURE COMPARISON")
    print("=" * 70)

    # Collect per-shot statistics
    expansion_stable = [r.get("expansion_stable_mean") for r in all_results if "expansion_stable_mean" in r]
    expansion_precursor = [r.get("expansion_precursor_mean") for r in all_results if "expansion_precursor_mean" in r]

    dimension_stable = [r.get("dimension_stable_mean") for r in all_results if "dimension_stable_mean" in r]
    dimension_precursor = [r.get("dimension_precursor_mean") for r in all_results if "dimension_precursor_mean" in r]

    entropy_stable = [r.get("entropy_stable_mean") for r in all_results if "entropy_stable_mean" in r]
    entropy_precursor = [r.get("entropy_precursor_mean") for r in all_results if "entropy_precursor_mean" in r]

    print("\n   EXPANSION RATIO (velocity of state change):")
    if expansion_stable and expansion_precursor:
        print(f"     Stable phase:    {np.mean(expansion_stable):.4f} +/- {np.std(expansion_stable):.4f}")
        print(f"     Precursor phase: {np.mean(expansion_precursor):.4f} +/- {np.std(expansion_precursor):.4f}")
        ratio = np.mean(expansion_precursor) / np.mean(expansion_stable) if np.mean(expansion_stable) != 0 else float('inf')
        print(f"     Ratio: {ratio:.2f}x")

    print("\n   LOCAL DIMENSION (degrees of freedom):")
    if dimension_stable and dimension_precursor:
        print(f"     Stable phase:    {np.mean(dimension_stable):.4f} +/- {np.std(dimension_stable):.4f}")
        print(f"     Precursor phase: {np.mean(dimension_precursor):.4f} +/- {np.std(dimension_precursor):.4f}")
        ratio = np.mean(dimension_precursor) / np.mean(dimension_stable) if np.mean(dimension_stable) != 0 else float('inf')
        print(f"     Ratio: {ratio:.2f}x")

    print("\n   SPECTRAL ENTROPY (complexity):")
    if entropy_stable and entropy_precursor:
        print(f"     Stable phase:    {np.mean(entropy_stable):.4f} +/- {np.std(entropy_stable):.4f}")
        print(f"     Precursor phase: {np.mean(entropy_precursor):.4f} +/- {np.std(entropy_precursor):.4f}")
        ratio = np.mean(entropy_precursor) / np.mean(entropy_stable) if np.mean(entropy_stable) != 0 else float('inf')
        print(f"     Ratio: {ratio:.2f}x")

    # Statistical significance
    print("\n5. Statistical Significance (t-test):")
    from scipy import stats

    if expansion_stable and expansion_precursor:
        t_stat, p_val = stats.ttest_ind(expansion_stable, expansion_precursor)
        print(f"   Expansion ratio: t={t_stat:.3f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")

    if dimension_stable and dimension_precursor:
        t_stat, p_val = stats.ttest_ind(dimension_stable, dimension_precursor)
        print(f"   Local dimension: t={t_stat:.3f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")

    if entropy_stable and entropy_precursor:
        t_stat, p_val = stats.ttest_ind(entropy_stable, entropy_precursor)
        print(f"   Spectral entropy: t={t_stat:.3f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")

    # Create visualization
    print("\n6. Generating visualization...")
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Expansion ratio
    if expansion_stable and expansion_precursor:
        axes[0].boxplot([expansion_stable, expansion_precursor], labels=['Stable', 'Precursor'])
        axes[0].set_ylabel('Expansion Ratio')
        axes[0].set_title('Expansion Ratio by Phase')

    # Local dimension
    if dimension_stable and dimension_precursor:
        axes[1].boxplot([dimension_stable, dimension_precursor], labels=['Stable', 'Precursor'])
        axes[1].set_ylabel('Local Dimension')
        axes[1].set_title('Local Dimension by Phase')

    # Spectral entropy
    if entropy_stable and entropy_precursor:
        axes[2].boxplot([entropy_stable, entropy_precursor], labels=['Stable', 'Precursor'])
        axes[2].set_ylabel('Spectral Entropy')
        axes[2].set_title('Spectral Entropy by Phase')

    plt.tight_layout()
    plt.savefig(output_dir / "mit_geometry_comparison.png", dpi=150)
    print(f"   Saved: {output_dir / 'mit_geometry_comparison.png'}")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("""
If the precursor phase shows significantly different geometric signatures
than stable operation, this validates our hypothesis:

  DISRUPTION PRECURSORS HAVE A GEOMETRIC SIGNATURE IN STATE SPACE

This means LLM-style representation learning could potentially identify
disruption trajectories before they become visible in traditional 3D
field diagnostics.

Next steps:
1. Train a transformer on plasma diagnostic sequences
2. Extract embeddings and analyze their geometry
3. Test if disruption trajectories are geometrically distinct
""")


if __name__ == "__main__":
    main()
