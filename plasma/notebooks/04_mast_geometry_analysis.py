#!/usr/bin/env python3
"""
High-dimensional geometry analysis of FAIR-MAST tokamak data.

This script applies our LLM-derived geometry tools to REAL high-dimensional
plasma diagnostic data from the MAST tokamak.

Key question: Do disruption precursors have geometric signatures when we
have sufficient dimensionality to capture the relational structure?

Run from ModelCypher root:
    python plasma/notebooks/04_mast_geometry_analysis.py
"""

import sys
from pathlib import Path

import numpy as np

# Check dependencies
try:
    import xarray as xr
    import s3fs
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xarray", "s3fs", "zarr", "aiohttp"])
    import xarray as xr
    import s3fs

# Add plasma src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geometry_tools import (
    compute_expansion_ratio,
    compute_local_dimension,
    compute_spectral_entropy,
)


# FAIR-MAST S3 configuration
ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast"
LEVEL = "level1"


def get_s3_filesystem():
    """Get anonymous S3 filesystem for FAIR-MAST."""
    return s3fs.S3FileSystem(anon=True, endpoint_url=ENDPOINT_URL)


def load_diagnostic(shot_id: int, diagnostic: str) -> xr.Dataset:
    """Load a diagnostic dataset for a shot."""
    url = f"{ENDPOINT_URL}/{BUCKET}/{LEVEL}/shots/{shot_id}.zarr/{diagnostic}"
    return xr.open_zarr(url)


def build_state_vector(shot_id: int, diagnostics: list[str] = ["amc", "efm"]) -> tuple[np.ndarray, np.ndarray]:
    """Build high-dimensional state vector from multiple diagnostics.

    Returns:
        trajectory: [T, D] array of state vectors
        time: [T] array of time points
    """
    arrays = []
    common_time = None

    for diag in diagnostics:
        try:
            ds = load_diagnostic(shot_id, diag)

            # Get time coordinate
            if 'time' in ds.coords:
                time = ds.coords['time'].values

                # Use first diagnostic's time as reference
                if common_time is None:
                    common_time = time

                # Collect 1D time-varying signals
                for var in ds.data_vars:
                    data = ds[var].values

                    # Only include 1D arrays matching time dimension
                    if len(data.shape) == 1 and len(data) == len(time):
                        arrays.append(data)

        except Exception as e:
            print(f"  Warning: Could not load {diag}: {e}")
            continue

    if not arrays:
        raise ValueError(f"No valid data for shot {shot_id}")

    # Stack into trajectory
    # Each column is a diagnostic channel
    trajectory = np.stack(arrays, axis=1)

    return trajectory, common_time


def analyze_shot_geometry(trajectory: np.ndarray, time: np.ndarray) -> dict:
    """Compute geometric features for a shot."""
    # Normalize trajectory for stable numerical computation
    mean = trajectory.mean(axis=0, keepdims=True)
    std = trajectory.std(axis=0, keepdims=True) + 1e-10
    trajectory_norm = (trajectory - mean) / std

    # Compute geometric features
    expansion = compute_expansion_ratio(trajectory_norm, window_size=10)
    local_dim = compute_local_dimension(trajectory_norm, window_size=50, method="eigenvalue")
    spectral_ent = compute_spectral_entropy(trajectory_norm, window_size=50)

    results = {
        "n_timesteps": len(trajectory),
        "n_dims": trajectory.shape[1],
        "time_range": (float(time[0]), float(time[-1])),
    }

    # Get valid (non-NaN) statistics
    valid_exp = ~np.isnan(expansion)
    valid_dim = ~np.isnan(local_dim)
    valid_ent = ~np.isnan(spectral_ent)

    if valid_exp.any():
        results["expansion_mean"] = float(np.nanmean(expansion))
        results["expansion_std"] = float(np.nanstd(expansion))
        results["expansion_max"] = float(np.nanmax(expansion))

    if valid_dim.any():
        results["dimension_mean"] = float(np.nanmean(local_dim))
        results["dimension_std"] = float(np.nanstd(local_dim))

    if valid_ent.any():
        results["entropy_mean"] = float(np.nanmean(spectral_ent))
        results["entropy_std"] = float(np.nanstd(spectral_ent))

    # Time series for detailed analysis
    results["expansion_series"] = expansion
    results["dimension_series"] = local_dim
    results["entropy_series"] = spectral_ent
    results["time"] = time

    return results


def detect_disruption_signature(results: dict) -> dict:
    """Look for geometric signatures that might indicate disruption.

    Based on hypothesis: disruptions should show:
    - Sharp increase in expansion ratio (acceleration of state change)
    - Change in local dimension (degrees of freedom changing)
    - Drop in spectral entropy (collapsing onto fewer modes)
    """
    expansion = results.get("expansion_series", np.array([]))
    dimension = results.get("dimension_series", np.array([]))
    entropy = results.get("entropy_series", np.array([]))
    time = results.get("time", np.array([]))

    if len(expansion) == 0:
        return {"signature_detected": False}

    # Look for sharp changes in the last portion of the shot
    # (disruptions typically happen near end)

    # Split into early (first 70%) and late (last 30%)
    split_idx = int(len(expansion) * 0.7)

    if split_idx < 10:
        return {"signature_detected": False}

    early_exp = expansion[:split_idx]
    late_exp = expansion[split_idx:]

    early_exp_valid = early_exp[~np.isnan(early_exp)]
    late_exp_valid = late_exp[~np.isnan(late_exp)]

    if len(early_exp_valid) < 5 or len(late_exp_valid) < 5:
        return {"signature_detected": False}

    # Calculate ratios
    early_mean = np.mean(early_exp_valid)
    late_mean = np.mean(late_exp_valid)

    expansion_ratio = late_mean / (early_mean + 1e-10)

    # Same for dimension if available
    dimension_ratio = 1.0
    if len(dimension) > split_idx:
        early_dim = dimension[:split_idx]
        late_dim = dimension[split_idx:]
        early_dim_valid = early_dim[~np.isnan(early_dim)]
        late_dim_valid = late_dim[~np.isnan(late_dim)]
        if len(early_dim_valid) > 5 and len(late_dim_valid) > 5:
            dimension_ratio = np.mean(late_dim_valid) / (np.mean(early_dim_valid) + 1e-10)

    # Detect if there's a significant change
    # These thresholds are exploratory
    signature = {
        "expansion_early_late_ratio": float(expansion_ratio),
        "dimension_early_late_ratio": float(dimension_ratio),
        "signature_detected": expansion_ratio > 1.5 or expansion_ratio < 0.5 or dimension_ratio > 1.3,
    }

    return signature


def main():
    print("=" * 70)
    print("HIGH-DIMENSIONAL GEOMETRY ANALYSIS OF MAST TOKAMAK DATA")
    print("Testing: Do disruption precursors have geometric signatures?")
    print("=" * 70)

    # Connect to S3
    print("\n1. Connecting to FAIR-MAST S3...")
    s3 = get_s3_filesystem()

    # List available shots
    print("\n2. Listing available shots...")
    shots_path = f"{BUCKET}/{LEVEL}/shots/"
    items = s3.ls(shots_path)[:50]  # Get first 50 shots

    shot_ids = []
    for item in items:
        name = item.split("/")[-1]
        if name.endswith(".zarr"):
            shot_id = int(name.replace(".zarr", ""))
            shot_ids.append(shot_id)

    shot_ids = sorted(shot_ids)[:20]  # Analyze first 20
    print(f"   Found shots: {shot_ids[:5]}... ({len(shot_ids)} total)")

    # Analyze each shot
    print("\n3. Building high-dimensional state vectors and analyzing geometry...")

    all_results = []

    for shot_id in shot_ids:
        print(f"\n   Shot {shot_id}:", end=" ")
        try:
            # Build state vector from diagnostics
            trajectory, time = build_state_vector(shot_id)
            print(f"shape={trajectory.shape}", end=" ")

            # Analyze geometry
            results = analyze_shot_geometry(trajectory, time)
            results["shot_id"] = shot_id

            # Look for disruption signatures
            signature = detect_disruption_signature(results)
            results.update(signature)

            # Remove large arrays from summary
            summary = {k: v for k, v in results.items()
                      if not isinstance(v, np.ndarray)}

            print(f"dim={summary.get('n_dims', '?')}", end=" ")
            print(f"exp={summary.get('expansion_mean', 0):.3f}", end=" ")
            print(f"local_dim={summary.get('dimension_mean', 0):.2f}", end=" ")

            if signature.get("signature_detected"):
                print(" ** SIGNATURE **", end="")
            print()

            all_results.append(summary)

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Summary statistics
    print("\n" + "=" * 70)
    print("4. SUMMARY STATISTICS")
    print("=" * 70)

    if all_results:
        dims = [r.get("n_dims", 0) for r in all_results if r.get("n_dims")]
        exps = [r.get("expansion_mean", 0) for r in all_results if r.get("expansion_mean")]
        loc_dims = [r.get("dimension_mean", 0) for r in all_results if r.get("dimension_mean")]
        entropies = [r.get("entropy_mean", 0) for r in all_results if r.get("entropy_mean")]

        print(f"\n   Shots analyzed: {len(all_results)}")
        print(f"   State vector dimensions: {np.mean(dims):.0f} +/- {np.std(dims):.0f}")
        print(f"   Expansion ratio: {np.mean(exps):.4f} +/- {np.std(exps):.4f}")
        print(f"   Local dimension: {np.mean(loc_dims):.2f} +/- {np.std(loc_dims):.2f}")
        print(f"   Spectral entropy: {np.mean(entropies):.4f} +/- {np.std(entropies):.4f}")

        # Count signatures
        n_signatures = sum(1 for r in all_results if r.get("signature_detected"))
        print(f"\n   Shots with potential signatures: {n_signatures}/{len(all_results)}")

        if n_signatures > 0:
            print("\n   Shots with signatures:")
            for r in all_results:
                if r.get("signature_detected"):
                    print(f"     Shot {r['shot_id']}: exp_ratio={r.get('expansion_early_late_ratio', 0):.2f}")

    print("\n" + "=" * 70)
    print("5. INTERPRETATION")
    print("=" * 70)
    print("""
This is exploratory analysis on a small sample. Key observations:

1. STATE DIMENSIONALITY: We now have ~40+ diagnostic channels instead of 6.
   This is closer to "full rank" for capturing plasma dynamics.

2. GEOMETRIC FEATURES: We can measure expansion ratio, local dimension, and
   spectral entropy throughout each shot.

3. DISRUPTION SIGNATURES: Shots with significant early-vs-late geometric
   changes may indicate disruption precursors.

CAVEATS:
- We don't have ground truth disruption labels from MAST
- Early shots (11695+) may not include disruptions
- Need to cross-reference with MAST disruption database

NEXT STEPS:
1. Find MAST shots that are known to disrupt
2. Compare geometry profiles: stable vs disrupted
3. Test if geometric signatures precede traditional indicators
""")


if __name__ == "__main__":
    main()
