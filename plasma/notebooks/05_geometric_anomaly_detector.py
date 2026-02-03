#!/usr/bin/env python3
"""
Unsupervised Geometric Anomaly Detection for MAST Tokamak Data.

Hypothesis: Disruptions have geometric precursors detectable without labels.

This script scans MAST shots for geometric anomalies:
- Expansion ratio spikes (sudden acceleration)
- Dimension changes (degrees of freedom shifting)
- Entropy drops (collapsing onto fewer modes)
- Late-shot geometric instability

Outputs a ranked list of "most geometrically unusual" shots.

Run from ModelCypher root:
    python plasma/notebooks/05_geometric_anomaly_detector.py
"""

import sys
from pathlib import Path
import json
from dataclasses import dataclass, asdict

import numpy as np

# Check dependencies
try:
    import xarray as xr
    import s3fs
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xarray", "s3fs", "zarr", "aiohttp", "-q"])
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


@dataclass
class AnomalyScore:
    """Geometric anomaly metrics for a shot."""
    shot_id: int
    n_channels: int
    n_timesteps: int

    # Basic statistics
    expansion_mean: float
    expansion_std: float
    expansion_max: float
    dimension_mean: float
    dimension_std: float
    entropy_mean: float
    entropy_std: float

    # Anomaly indicators
    expansion_spike_count: int      # Number of >3σ expansion events
    expansion_late_ratio: float     # Late vs early expansion ratio
    dimension_late_ratio: float     # Late vs early dimension ratio
    entropy_drop_max: float         # Largest entropy drop
    volatility_trend: float         # Is volatility increasing toward end?

    # Combined anomaly score (higher = more anomalous)
    anomaly_score: float = 0.0
    anomaly_reasons: str = ""


def load_shot_trajectory(shot_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and preprocess a shot's diagnostic data."""
    try:
        url = f"{ENDPOINT_URL}/{BUCKET}/{LEVEL}/shots/{shot_id}.zarr/amc"
        ds = xr.open_zarr(url)
        time = ds.coords['time'].values

        # Build state vector from valid channels
        arrays = []
        for var in ds.data_vars:
            data = ds[var].values
            if len(data.shape) == 1 and len(data) == len(time):
                nan_frac = np.isnan(data).mean()
                if nan_frac < 0.3:
                    if np.isnan(data).any():
                        mask = ~np.isnan(data)
                        if mask.sum() > 100:
                            data = np.interp(np.arange(len(data)), np.where(mask)[0], data[mask])
                    if np.std(data) > 1e-10:
                        arrays.append(data)

        if len(arrays) < 5:
            return None

        trajectory = np.stack(arrays, axis=1)

        # Normalize
        mean = trajectory.mean(axis=0)
        std = trajectory.std(axis=0) + 1e-10
        traj_norm = (trajectory - mean) / std

        # Downsample for speed (target ~500-1000 points)
        factor = max(1, len(traj_norm) // 800)
        traj_ds = traj_norm[::factor]
        time_ds = time[::factor]

        return traj_ds, time_ds

    except Exception as e:
        return None


def compute_anomaly_score(shot_id: int, trajectory: np.ndarray, time: np.ndarray) -> AnomalyScore:
    """Compute comprehensive anomaly metrics for a shot."""

    # Compute geometric features
    expansion = compute_expansion_ratio(trajectory, window_size=5)

    # Handle potential numerical issues in dimension/entropy
    try:
        local_dim = compute_local_dimension(trajectory, window_size=10, method="eigenvalue")
    except:
        local_dim = np.full(len(trajectory), np.nan)

    try:
        spectral_ent = compute_spectral_entropy(trajectory, window_size=10)
    except:
        spectral_ent = np.full(len(trajectory), np.nan)

    # Basic statistics
    exp_mean = float(np.nanmean(expansion))
    exp_std = float(np.nanstd(expansion))
    exp_max = float(np.nanmax(expansion))
    dim_mean = float(np.nanmean(local_dim))
    dim_std = float(np.nanstd(local_dim))
    ent_mean = float(np.nanmean(spectral_ent))
    ent_std = float(np.nanstd(spectral_ent))

    # Anomaly indicators

    # 1. Expansion spikes (>3σ events)
    threshold = exp_mean + 3 * exp_std
    spike_count = int(np.sum(expansion > threshold))

    # 2. Late vs early ratios (split at 70%)
    split = int(len(expansion) * 0.7)
    early_exp = expansion[:split]
    late_exp = expansion[split:]

    early_exp_valid = early_exp[~np.isnan(early_exp)]
    late_exp_valid = late_exp[~np.isnan(late_exp)]

    if len(early_exp_valid) > 5 and len(late_exp_valid) > 5:
        exp_late_ratio = float(np.mean(late_exp_valid) / (np.mean(early_exp_valid) + 1e-10))
    else:
        exp_late_ratio = 1.0

    # Dimension late ratio
    if len(local_dim) > split:
        early_dim = local_dim[:split]
        late_dim = local_dim[split:]
        early_dim_valid = early_dim[~np.isnan(early_dim)]
        late_dim_valid = late_dim[~np.isnan(late_dim)]
        if len(early_dim_valid) > 5 and len(late_dim_valid) > 5:
            dim_late_ratio = float(np.mean(late_dim_valid) / (np.mean(early_dim_valid) + 1e-10))
        else:
            dim_late_ratio = 1.0
    else:
        dim_late_ratio = 1.0

    # 3. Entropy drops (largest single-step decrease)
    ent_diff = np.diff(spectral_ent)
    ent_drop_max = float(-np.nanmin(ent_diff)) if len(ent_diff) > 0 else 0.0

    # 4. Volatility trend (is expansion std increasing toward end?)
    window = min(50, len(expansion) // 4)
    if window > 5:
        early_vol = np.nanstd(expansion[:window*2])
        late_vol = np.nanstd(expansion[-window*2:])
        volatility_trend = float(late_vol / (early_vol + 1e-10))
    else:
        volatility_trend = 1.0

    # Combined anomaly score
    # Higher = more anomalous
    reasons = []
    score = 0.0

    # Expansion spikes
    if spike_count > 5:
        score += spike_count * 0.5
        reasons.append(f"{spike_count} expansion spikes")

    # Late expansion increase
    if exp_late_ratio > 1.5:
        score += (exp_late_ratio - 1) * 2
        reasons.append(f"late expansion {exp_late_ratio:.1f}x")

    # Dimension change
    if abs(dim_late_ratio - 1) > 0.3:
        score += abs(dim_late_ratio - 1) * 3
        reasons.append(f"dimension shift {dim_late_ratio:.2f}x")

    # Entropy drop
    if ent_drop_max > ent_std * 2:
        score += ent_drop_max * 2
        reasons.append(f"entropy drop {ent_drop_max:.2f}")

    # Volatility increase
    if volatility_trend > 2.0:
        score += volatility_trend
        reasons.append(f"volatility increase {volatility_trend:.1f}x")

    # Very high max expansion
    if exp_max > 5.0:
        score += exp_max * 0.3
        reasons.append(f"max expansion {exp_max:.1f}")

    return AnomalyScore(
        shot_id=shot_id,
        n_channels=trajectory.shape[1],
        n_timesteps=trajectory.shape[0],
        expansion_mean=exp_mean,
        expansion_std=exp_std,
        expansion_max=exp_max,
        dimension_mean=dim_mean,
        dimension_std=dim_std,
        entropy_mean=ent_mean,
        entropy_std=ent_std,
        expansion_spike_count=spike_count,
        expansion_late_ratio=exp_late_ratio,
        dimension_late_ratio=dim_late_ratio,
        entropy_drop_max=ent_drop_max,
        volatility_trend=volatility_trend,
        anomaly_score=score,
        anomaly_reasons="; ".join(reasons) if reasons else "normal",
    )


def main():
    print("=" * 70)
    print("UNSUPERVISED GEOMETRIC ANOMALY DETECTION")
    print("Scanning MAST shots for disruption candidates")
    print("=" * 70)

    # Connect to S3 and get shot list
    print("\n1. Connecting to FAIR-MAST...")
    s3 = s3fs.S3FileSystem(anon=True, endpoint_url=ENDPOINT_URL)

    shots_path = f"{BUCKET}/{LEVEL}/shots/"
    items = s3.ls(shots_path)

    shot_ids = []
    for item in items:
        name = item.split("/")[-1]
        if name.endswith(".zarr"):
            shot_id = int(name.replace(".zarr", ""))
            shot_ids.append(shot_id)

    shot_ids = sorted(shot_ids)
    print(f"   Total shots available: {len(shot_ids)}")
    print(f"   Range: {shot_ids[0]} to {shot_ids[-1]}")

    # Sample shots for analysis (focus on later shots with better data)
    # Take every Nth shot from the later portion
    late_shots = [s for s in shot_ids if s > 25000]
    sample_step = max(1, len(late_shots) // 100)
    sample_shots = late_shots[::sample_step][:100]

    print(f"\n2. Sampling {len(sample_shots)} shots for analysis...")
    print(f"   Sample range: {sample_shots[0]} to {sample_shots[-1]}")

    # Analyze each shot
    print("\n3. Computing geometric anomaly scores...")

    results = []
    failed = 0

    for i, shot_id in enumerate(sample_shots):
        print(f"\r   Processing {i+1}/{len(sample_shots)}: shot {shot_id}...", end="", flush=True)

        data = load_shot_trajectory(shot_id)
        if data is None:
            failed += 1
            continue

        trajectory, time = data
        score = compute_anomaly_score(shot_id, trajectory, time)
        results.append(score)

    print(f"\n   Analyzed: {len(results)}, Failed: {failed}")

    # Rank by anomaly score
    results.sort(key=lambda x: x.anomaly_score, reverse=True)

    # Output results
    print("\n" + "=" * 70)
    print("4. TOP ANOMALOUS SHOTS (potential disruption candidates)")
    print("=" * 70)

    print(f"\n{'Rank':<5} {'Shot':<8} {'Score':<8} {'Reasons'}")
    print("-" * 70)

    for i, r in enumerate(results[:20]):
        print(f"{i+1:<5} {r.shot_id:<8} {r.anomaly_score:<8.2f} {r.anomaly_reasons}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("5. POPULATION STATISTICS")
    print("=" * 70)

    scores = [r.anomaly_score for r in results]
    print(f"\n   Anomaly score distribution:")
    print(f"     Mean: {np.mean(scores):.2f}")
    print(f"     Std:  {np.std(scores):.2f}")
    print(f"     Max:  {np.max(scores):.2f}")

    # Count by threshold
    high_anomaly = sum(1 for s in scores if s > 5)
    medium_anomaly = sum(1 for s in scores if 2 < s <= 5)
    low_anomaly = sum(1 for s in scores if s <= 2)

    print(f"\n   Anomaly categories:")
    print(f"     High (score > 5):    {high_anomaly} shots ({high_anomaly/len(results)*100:.1f}%)")
    print(f"     Medium (2 < score ≤ 5): {medium_anomaly} shots ({medium_anomaly/len(results)*100:.1f}%)")
    print(f"     Normal (score ≤ 2):  {low_anomaly} shots ({low_anomaly/len(results)*100:.1f}%)")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "anomaly_candidates.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n   Full results saved to: {output_file}")

    # Save top candidates
    top_file = output_dir / "top_anomaly_candidates.txt"
    with open(top_file, "w") as f:
        f.write("MAST Geometric Anomaly Detection Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Top 20 shots ranked by geometric anomaly score:\n")
        f.write("(Higher score = more geometrically unusual)\n\n")
        for i, r in enumerate(results[:20]):
            f.write(f"{i+1}. Shot {r.shot_id}: score={r.anomaly_score:.2f}\n")
            f.write(f"   Reasons: {r.anomaly_reasons}\n")
            f.write(f"   Expansion: {r.expansion_mean:.3f}±{r.expansion_std:.3f}, max={r.expansion_max:.2f}\n")
            f.write(f"   Dimension: {r.dimension_mean:.2f}±{r.dimension_std:.2f}\n")
            f.write(f"   Late ratios: exp={r.expansion_late_ratio:.2f}x, dim={r.dimension_late_ratio:.2f}x\n")
            f.write("\n")
    print(f"   Top candidates saved to: {top_file}")

    print("\n" + "=" * 70)
    print("6. INTERPRETATION")
    print("=" * 70)
    print("""
These shots showed unusual geometric signatures:
- Expansion spikes: sudden acceleration of state change
- Late-shot instability: geometry changing near end
- Dimension shifts: degrees of freedom changing
- Entropy drops: collapsing onto fewer modes

HYPOTHESIS: High-anomaly shots are disruption candidates.

To validate:
1. Cross-reference top shots against MAST disruption database
2. If correlation exists, geometry predicts disruptions unsupervised
3. Examine detailed trajectories of top anomalies
""")

    return results


if __name__ == "__main__":
    results = main()
