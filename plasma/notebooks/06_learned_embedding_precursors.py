#!/usr/bin/env python3
"""
Test if learned embeddings give earlier disruption warnings than raw diagnostics.

Approach:
1. Train transformer on STABLE shots (learn normal plasma dynamics)
2. Extract embeddings for DISRUPTED shots
3. Compare geometry: raw diagnostics vs learned embeddings
4. Measure: do embeddings show anomalies earlier?

The hypothesis: a model trained on normal dynamics will show
larger geometric anomalies earlier when processing abnormal (pre-disruption) dynamics.
"""

import sys
from pathlib import Path

import numpy as np

# Check PyTorch
try:
    import torch
    print(f"PyTorch available: {torch.__version__}")
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        print("Using Apple Silicon GPU")
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except ImportError:
    print("PyTorch not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "-q"])
    import torch
    DEVICE = "cpu"

import xarray as xr

# Add local modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plasma_transformer import PlasmaTransformer, train_on_shots, extract_embedding_trajectory
from geometry_tools import compute_expansion_ratio


def load_shot_trajectory(shot_id: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Load trajectory and find disruption time if applicable."""
    try:
        url = f"https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot_id}.zarr/amc"
        ds = xr.open_zarr(url)
        time = ds.coords['time'].values

        # Get plasma current for disruption detection
        Ip = ds['plasma_current'].values
        if np.isnan(Ip).any():
            mask = ~np.isnan(Ip)
            if mask.sum() > 100:
                Ip = np.interp(np.arange(len(Ip)), np.where(mask)[0], Ip[mask])

        # Build state vector
        arrays = []
        for var in ds.data_vars:
            data = ds[var].values
            if len(data.shape) == 1 and len(data) == len(time):
                nan_frac = np.isnan(data).mean()
                if nan_frac < 0.3:
                    if np.isnan(data).any():
                        m = ~np.isnan(data)
                        if m.sum() > 100:
                            data = np.interp(np.arange(len(data)), np.where(m)[0], data[m])
                    if np.std(data) > 1e-10:
                        arrays.append(data)

        if len(arrays) < 10:
            return None

        trajectory = np.stack(arrays, axis=1).astype(np.float32)

        # Find disruption time (if any)
        Ip_max = np.max(np.abs(Ip))
        if Ip_max < 100:  # No plasma
            return None

        threshold = 0.1 * Ip_max
        plasma_indices = np.where(np.abs(Ip) > threshold)[0]
        if len(plasma_indices) < 100:
            return None

        disruption_time = time[plasma_indices[-1]]

        # Downsample for manageable size
        factor = max(1, len(trajectory) // 2000)
        return trajectory[::factor], time[::factor], disruption_time

    except Exception as e:
        print(f"  Error loading {shot_id}: {e}")
        return None


def analyze_precursor_timing(
    trajectory: np.ndarray,
    time: np.ndarray,
    disruption_time: float,
    label: str,
) -> dict:
    """Analyze when geometric anomalies appear relative to disruption."""

    # Normalize
    mean = trajectory.mean(axis=0, keepdims=True)
    std = trajectory.std(axis=0, keepdims=True) + 1e-10
    traj_norm = (trajectory - mean) / std

    # Compute expansion
    expansion = compute_expansion_ratio(traj_norm, window_size=5)
    time_exp = time[1:-1] if len(expansion) == len(time) - 2 else time[:len(expansion)]

    # Find plasma phase (before disruption)
    pre_disruption = time_exp < disruption_time

    # Baseline from first half of plasma phase
    plasma_start = time_exp[pre_disruption][0] if pre_disruption.any() else time_exp[0]
    mid_time = plasma_start + (disruption_time - plasma_start) * 0.5

    baseline_mask = (time_exp >= plasma_start) & (time_exp < mid_time)
    if baseline_mask.sum() < 10:
        return {"label": label, "error": "insufficient baseline"}

    baseline = expansion[baseline_mask]
    threshold_2sigma = np.nanmean(baseline) + 2 * np.nanstd(baseline)
    threshold_3sigma = np.nanmean(baseline) + 3 * np.nanstd(baseline)

    # Find first crossing after baseline period
    post_baseline = time_exp >= mid_time
    pre_disruption_mask = post_baseline & (time_exp < disruption_time)

    crossings_2sigma = np.where((expansion > threshold_2sigma) & pre_disruption_mask)[0]
    crossings_3sigma = np.where((expansion > threshold_3sigma) & pre_disruption_mask)[0]

    result = {
        "label": label,
        "baseline_mean": float(np.nanmean(baseline)),
        "baseline_std": float(np.nanstd(baseline)),
        "threshold_2sigma": float(threshold_2sigma),
        "threshold_3sigma": float(threshold_3sigma),
    }

    if len(crossings_2sigma) > 0:
        first_2sigma = time_exp[crossings_2sigma[0]]
        result["lead_time_2sigma_ms"] = float((disruption_time - first_2sigma) * 1000)
    else:
        result["lead_time_2sigma_ms"] = None

    if len(crossings_3sigma) > 0:
        first_3sigma = time_exp[crossings_3sigma[0]]
        result["lead_time_3sigma_ms"] = float((disruption_time - first_3sigma) * 1000)
    else:
        result["lead_time_3sigma_ms"] = None

    return result


def main():
    print("=" * 70)
    print("LEARNED EMBEDDINGS FOR EARLY DISRUPTION DETECTION")
    print("=" * 70)

    # Define shot sets
    # Stable shots (for training) - high Ip, clean termination
    stable_shots = [30473, 30460, 30440, 30420, 30400]

    # Disrupted shots (for testing) - confirmed disruptions
    disrupted_shots = [27177, 27499, 29484, 28298]

    # Load stable shots for training
    print("\n1. Loading stable shots for training...")
    train_trajectories = []
    for shot_id in stable_shots:
        print(f"   Loading {shot_id}...", end=" ")
        data = load_shot_trajectory(shot_id)
        if data:
            traj, time, _ = data
            train_trajectories.append(traj)
            print(f"shape {traj.shape}")
        else:
            print("failed")

    if len(train_trajectories) < 2:
        print("Not enough training data!")
        return

    # Get diagnostic dimension
    diagnostic_dim = train_trajectories[0].shape[1]
    print(f"\n   Diagnostic dimension: {diagnostic_dim}")
    print(f"   Training shots: {len(train_trajectories)}")

    # Create and train model
    print("\n2. Training transformer on normal plasma dynamics...")
    model = PlasmaTransformer(
        diagnostic_dim=diagnostic_dim,
        embed_dim=64,  # Smaller embedding for faster training
        num_heads=4,
        num_layers=2,
        max_seq_len=500,
    )

    losses = train_on_shots(
        model,
        train_trajectories,
        epochs=30,
        lr=1e-3,
        seq_len=200,
        batch_size=16,
        device=DEVICE,
        verbose=True,
    )

    print(f"   Final loss: {losses[-1]:.6f}")

    # Analyze disrupted shots
    print("\n3. Analyzing disrupted shots...")
    print("   Comparing: raw diagnostics vs learned embeddings")

    results = []

    for shot_id in disrupted_shots:
        print(f"\n   Shot {shot_id}:")
        data = load_shot_trajectory(shot_id)
        if not data:
            continue

        trajectory, time, disruption_time = data
        print(f"     Trajectory: {trajectory.shape}, disruption at {disruption_time:.3f}s")

        # Analyze RAW diagnostics
        raw_result = analyze_precursor_timing(trajectory, time, disruption_time, "raw")

        # Extract EMBEDDINGS
        print("     Extracting embeddings...", end=" ")
        try:
            embeddings = extract_embedding_trajectory(model, trajectory, device=DEVICE)
            print(f"shape {embeddings.shape}")

            # Analyze EMBEDDING geometry
            emb_result = analyze_precursor_timing(embeddings, time[:len(embeddings)], disruption_time, "embedding")
        except Exception as e:
            print(f"failed: {e}")
            emb_result = {"label": "embedding", "error": str(e)}

        results.append({
            "shot_id": shot_id,
            "raw": raw_result,
            "embedding": emb_result,
        })

        # Print comparison
        raw_lead = raw_result.get("lead_time_2sigma_ms")
        emb_lead = emb_result.get("lead_time_2sigma_ms")

        print(f"     Raw lead time (2σ):       {raw_lead:.0f} ms" if raw_lead else "     Raw: no anomaly detected")
        print(f"     Embedding lead time (2σ): {emb_lead:.0f} ms" if emb_lead else "     Embedding: no anomaly detected")

        if raw_lead and emb_lead:
            improvement = emb_lead - raw_lead
            print(f"     >>> IMPROVEMENT: {improvement:+.0f} ms <<<")

    # Summary
    print("\n" + "=" * 70)
    print("4. SUMMARY: RAW vs LEARNED EMBEDDINGS")
    print("=" * 70)

    print(f"\n{'Shot':<10} {'Raw (2σ)':<15} {'Embedding (2σ)':<15} {'Improvement':<15}")
    print("-" * 55)

    improvements = []
    for r in results:
        shot = r["shot_id"]
        raw_lead = r["raw"].get("lead_time_2sigma_ms")
        emb_lead = r["embedding"].get("lead_time_2sigma_ms")

        raw_str = f"{raw_lead:.0f} ms" if raw_lead else "N/A"
        emb_str = f"{emb_lead:.0f} ms" if emb_lead else "N/A"

        if raw_lead and emb_lead:
            imp = emb_lead - raw_lead
            imp_str = f"{imp:+.0f} ms"
            improvements.append(imp)
        else:
            imp_str = "N/A"

        print(f"{shot:<10} {raw_str:<15} {emb_str:<15} {imp_str:<15}")

    if improvements:
        print(f"\nMean improvement: {np.mean(improvements):+.0f} ms")
        print(f"Max improvement:  {np.max(improvements):+.0f} ms")

        if np.mean(improvements) > 0:
            print("\n*** LEARNED EMBEDDINGS PROVIDE EARLIER WARNING ***")
        else:
            print("\n*** No significant improvement from embeddings ***")
            print("    (May need more training data or larger model)")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If embeddings show earlier anomalies than raw diagnostics:
  - The model learned a representation where disruption precursors are amplified
  - Nonlinear combinations of diagnostics diverge before individual channels
  - This is exactly what we hypothesized from LLM geometry

If no improvement:
  - May need more training data (more stable shots)
  - May need larger model (more capacity to learn dynamics)
  - May need different architecture (attention to long-range dependencies)

Either way, this demonstrates the approach:
  Train on normal → Detect abnormal through geometric divergence
""")


if __name__ == "__main__":
    main()
