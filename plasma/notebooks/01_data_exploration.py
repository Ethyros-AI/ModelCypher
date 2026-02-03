#!/usr/bin/env python3
"""
Initial exploration of plasma diagnostic data.

This script:
1. Loads sample plasma shots (synthetic for now)
2. Visualizes diagnostic trajectories
3. Computes basic statistics
4. Tests the data pipeline

Run from ModelCypher root:
    python plasma/notebooks/01_data_exploration.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add plasma src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import create_synthetic_shot, PlasmaShot


def plot_diagnostic_trajectory(shot: PlasmaShot, save_path: Path | None = None):
    """Plot diagnostic channels over time."""
    fig, axes = plt.subplots(len(shot.diagnostics), 1, figsize=(12, 2 * len(shot.diagnostics)))

    for ax, (name, data) in zip(axes, shot.diagnostics.items()):
        # Plot each channel as a line
        for i in range(min(data.shape[1], 5)):  # Limit to 5 channels per diagnostic
            ax.plot(shot.time, data[:, i], alpha=0.7, label=f"ch{i}")
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)

        # Mark disruption time if applicable
        if shot.disruption_time is not None:
            ax.axvline(shot.disruption_time, color="red", linestyle="--", label="disruption")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Shot {shot.shot_id} - {'DISRUPTED' if shot.disrupted else 'STABLE'}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def compute_trajectory_statistics(shot: PlasmaShot) -> dict:
    """Compute basic statistics on the trajectory."""
    traj = shot.get_trajectory()

    # Compute step-to-step changes
    deltas = np.diff(traj, axis=0)
    step_norms = np.linalg.norm(deltas, axis=1)

    # Compute local covariance (sliding window)
    window_size = 50
    local_dims = []
    for t in range(0, len(traj) - window_size, window_size // 2):
        window = traj[t:t + window_size]
        cov = np.cov(window.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        # Effective dimension from eigenvalue distribution
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        p = eigenvalues / eigenvalues.sum()
        eff_dim = np.exp(-np.sum(p * np.log(p)))
        local_dims.append(eff_dim)

    return {
        "n_timesteps": shot.n_timesteps,
        "state_dim": shot.state_dim,
        "mean_step_norm": float(np.mean(step_norms)),
        "max_step_norm": float(np.max(step_norms)),
        "mean_local_dim": float(np.mean(local_dims)),
        "local_dim_std": float(np.std(local_dims)),
        "disrupted": shot.disrupted,
    }


def compare_stable_vs_disrupted(n_samples: int = 10):
    """Compare statistics between stable and disrupted shots."""
    stable_stats = []
    disrupted_stats = []

    for i in range(n_samples):
        stable = create_synthetic_shot(disrupted=False, seed=i)
        stable_stats.append(compute_trajectory_statistics(stable))

        disrupted = create_synthetic_shot(disrupted=True, seed=i + 1000)
        disrupted_stats.append(compute_trajectory_statistics(disrupted))

    print("\n=== Stable vs Disrupted Comparison ===\n")

    metrics = ["mean_step_norm", "max_step_norm", "mean_local_dim", "local_dim_std"]
    for metric in metrics:
        stable_vals = [s[metric] for s in stable_stats]
        disrupted_vals = [s[metric] for s in disrupted_stats]
        print(f"{metric}:")
        print(f"  Stable:    {np.mean(stable_vals):.4f} +/- {np.std(stable_vals):.4f}")
        print(f"  Disrupted: {np.mean(disrupted_vals):.4f} +/- {np.std(disrupted_vals):.4f}")
        print()


def main():
    print("=" * 60)
    print("Plasma Data Exploration")
    print("=" * 60)

    # Create output directory
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Generate and visualize sample shots
    print("\n1. Generating synthetic shots...")

    stable_shot = create_synthetic_shot(disrupted=False, seed=42)
    print(f"   Stable: {stable_shot.n_timesteps} timesteps, {stable_shot.state_dim} dimensions")

    disrupted_shot = create_synthetic_shot(disrupted=True, seed=43)
    print(f"   Disrupted: {disrupted_shot.n_timesteps} timesteps, disruption at t={disrupted_shot.disruption_time:.2f}s")

    # Plot trajectories
    print("\n2. Plotting diagnostic trajectories...")
    plot_diagnostic_trajectory(stable_shot, output_dir / "stable_trajectory.png")
    plot_diagnostic_trajectory(disrupted_shot, output_dir / "disrupted_trajectory.png")

    # Compute statistics
    print("\n3. Computing trajectory statistics...")
    stable_stats = compute_trajectory_statistics(stable_shot)
    disrupted_stats = compute_trajectory_statistics(disrupted_shot)

    print("\n   Stable shot:")
    for k, v in stable_stats.items():
        print(f"     {k}: {v}")

    print("\n   Disrupted shot:")
    for k, v in disrupted_stats.items():
        print(f"     {k}: {v}")

    # Compare populations
    print("\n4. Comparing stable vs disrupted populations...")
    compare_stable_vs_disrupted(n_samples=20)

    print("\n" + "=" * 60)
    print("Exploration complete. Results in:", output_dir)
    print("=" * 60)

    # Key observation for next steps
    print("""
NEXT STEPS:
-----------
1. Acquire real DisruptionBench data
2. Adapt geometry tools (expansion_ratio, spectral_entropy) to plasma trajectories
3. Look for geometric signatures that distinguish stable from disrupted shots
4. Train representation model on diagnostic sequences

The synthetic data shows that even random walks have distinguishable
statistics between stable and disrupted. The question is whether
real plasma data has geometric structure that makes disruptions
predictable in embedding space.
""")


if __name__ == "__main__":
    main()
