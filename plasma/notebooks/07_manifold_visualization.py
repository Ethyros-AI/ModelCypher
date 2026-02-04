#!/usr/bin/env python3
"""
3D visualization of plasma trajectories in PCA manifold space.

Shows:
1. Stable vs disrupted trajectories projected into PC1-PC2-PC3
2. Gradient arrows pointing back to stability
3. Where disruptions occur relative to the manifold
4. Animated trajectory evolution (optional)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add local modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geometry_tools import (
    compute_pca_manifold,
    compute_gradient_to_manifold,
    compute_trajectory_manifold_analysis,
)
from manifold_interpreter import interpret_pc_loadings, summarize_manifold


def load_mast_shot(shot_id: int) -> tuple[np.ndarray, np.ndarray, list[str], float | None]:
    """Load MAST shot trajectory from FAIR-MAST S3.

    Returns:
        trajectory: [T, D] array of states
        time: [T] array of time points
        diagnostic_names: List of diagnostic channel names
        disruption_time: Time of disruption (if detected) or None
    """
    import xarray as xr

    url = f"https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot_id}.zarr/amc"
    ds = xr.open_zarr(url)
    time = ds.coords['time'].values

    # Get plasma current for disruption detection
    Ip = ds['plasma_current'].values
    if np.isnan(Ip).any():
        mask = ~np.isnan(Ip)
        if mask.sum() > 100:
            Ip = np.interp(np.arange(len(Ip)), np.where(mask)[0], Ip[mask])

    # Build state vector with named diagnostics
    arrays = []
    names = []
    for var in sorted(ds.data_vars):
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
                    names.append(var)

    if len(arrays) < 10:
        raise ValueError(f"Shot {shot_id}: insufficient diagnostics ({len(arrays)})")

    trajectory = np.stack(arrays, axis=1).astype(np.float32)

    # Detect disruption (Ip collapse)
    Ip_max = np.max(np.abs(Ip))
    if Ip_max < 100:
        raise ValueError(f"Shot {shot_id}: no plasma detected")

    threshold = 0.1 * Ip_max
    plasma_indices = np.where(np.abs(Ip) > threshold)[0]
    if len(plasma_indices) < 100:
        raise ValueError(f"Shot {shot_id}: insufficient plasma duration")

    # Check for sudden Ip drop (disruption signature)
    disruption_time = None
    Ip_diff = np.diff(Ip[plasma_indices])
    large_drops = np.where(Ip_diff < -0.3 * Ip_max)[0]
    if len(large_drops) > 0:
        disruption_idx = plasma_indices[large_drops[0] + 1]
        disruption_time = time[disruption_idx]

    # Downsample for visualization
    factor = max(1, len(trajectory) // 2000)
    return trajectory[::factor], time[::factor], names, disruption_time


def plot_3d_manifold(
    stable_trajs: list[np.ndarray],
    disrupted_trajs: list[np.ndarray],
    manifold,
    stable_ids: list[str] | None = None,
    disrupted_ids: list[str] | None = None,
    disruption_times: list[float | None] | None = None,
    time_arrays: list[np.ndarray] | None = None,
    output_path: str | None = None,
):
    """Create 3D scatter plot of trajectories in PC space.

    Args:
        stable_trajs: List of stable shot trajectories [T, D]
        disrupted_trajs: List of disrupted shot trajectories [T, D]
        manifold: PCAManifold fitted on stable shots
        stable_ids: Optional shot IDs for labels
        disrupted_ids: Optional shot IDs for labels
        disruption_times: Time of disruption for each disrupted shot
        time_arrays: Time arrays for each trajectory
        output_path: Save figure to this path (optional)
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Project stable shots - color by time progression
    for i, traj in enumerate(stable_trajs):
        pc_traj = manifold.transform(traj)
        label = f"Stable {stable_ids[i]}" if stable_ids and i == 0 else None
        if i == 0:
            ax.scatter(pc_traj[:, 0], pc_traj[:, 1], pc_traj[:, 2],
                      c='blue', alpha=0.3, s=1, label="Stable shots")
        else:
            ax.scatter(pc_traj[:, 0], pc_traj[:, 1], pc_traj[:, 2],
                      c='blue', alpha=0.3, s=1)

    # Project disrupted shots - color by time, mark disruption point
    colors_disrupted = plt.cm.Reds(np.linspace(0.3, 1.0, len(disrupted_trajs)))

    for i, traj in enumerate(disrupted_trajs):
        pc_traj = manifold.transform(traj)

        # Find disruption index if we have timing info
        disrupt_idx = None
        if disruption_times and time_arrays and i < len(disruption_times):
            dt = disruption_times[i]
            if dt is not None and i < len(time_arrays):
                t = time_arrays[i]
                disrupt_idx = np.searchsorted(t, dt)
                disrupt_idx = min(disrupt_idx, len(pc_traj) - 1)

        # Plot trajectory
        label = f"Disrupted {disrupted_ids[i]}" if disrupted_ids else f"Disrupted {i+1}"
        ax.plot(pc_traj[:, 0], pc_traj[:, 1], pc_traj[:, 2],
               color=colors_disrupted[i], alpha=0.6, linewidth=0.5,
               label=label if i < 3 else None)

        # Mark disruption point
        if disrupt_idx is not None:
            ax.scatter([pc_traj[disrupt_idx, 0]], [pc_traj[disrupt_idx, 1]],
                      [pc_traj[disrupt_idx, 2]], c='red', s=100, marker='X',
                      edgecolors='black', linewidth=1,
                      label='Disruption point' if i == 0 else None)

            # Draw gradient arrow at disruption point
            grad = compute_gradient_to_manifold(traj[disrupt_idx], manifold)
            grad_pc = manifold.transform(traj[disrupt_idx] + grad.gradient_direction * grad.distance * 0.5)
            ax.quiver(pc_traj[disrupt_idx, 0], pc_traj[disrupt_idx, 1], pc_traj[disrupt_idx, 2],
                     grad_pc[0] - pc_traj[disrupt_idx, 0],
                     grad_pc[1] - pc_traj[disrupt_idx, 1],
                     grad_pc[2] - pc_traj[disrupt_idx, 2],
                     color='green', alpha=0.8, arrow_length_ratio=0.3,
                     label='Gradient to stability' if i == 0 else None)

    ax.set_xlabel(f'PC1 ({manifold.explained_variance_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({manifold.explained_variance_ratio[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({manifold.explained_variance_ratio[2]*100:.1f}%)')
    ax.set_title('Plasma Trajectories in PCA Manifold Space\n(Stable = blue, Disrupted = red)')
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


def plot_manifold_distance_timeseries(
    trajectories: list[np.ndarray],
    time_arrays: list[np.ndarray],
    manifold,
    labels: list[str],
    disruption_times: list[float | None] | None = None,
    output_path: str | None = None,
):
    """Plot manifold distance over time for multiple shots.

    Args:
        trajectories: List of [T, D] arrays
        time_arrays: List of [T] time arrays
        manifold: PCAManifold
        labels: Shot labels
        disruption_times: Disruption time for each shot (None if stable)
        output_path: Save path (optional)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Top: stable shots
    # Bottom: disrupted shots
    stable_ax, disrupted_ax = axes

    for i, (traj, time, label) in enumerate(zip(trajectories, time_arrays, labels)):
        analysis = compute_trajectory_manifold_analysis(traj, manifold, time)
        distances = analysis['distances']

        dt = disruption_times[i] if disruption_times else None

        if dt is None:
            # Stable shot
            stable_ax.plot(time[:len(distances)], distances, alpha=0.7, label=label)
        else:
            # Disrupted shot
            disrupted_ax.plot(time[:len(distances)], distances, alpha=0.7, label=label)
            disrupted_ax.axvline(dt, color='red', linestyle='--', alpha=0.5)

    stable_ax.set_ylabel('Manifold Distance')
    stable_ax.set_title('Stable Shots: Distance from Learned Manifold')
    stable_ax.legend(fontsize=8)
    stable_ax.grid(True, alpha=0.3)

    disrupted_ax.set_xlabel('Time (s)')
    disrupted_ax.set_ylabel('Manifold Distance')
    disrupted_ax.set_title('Disrupted Shots: Distance from Learned Manifold\n(red dashed = disruption time)')
    disrupted_ax.legend(fontsize=8)
    disrupted_ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, axes


def plot_gradient_decomposition(
    traj: np.ndarray,
    time: np.ndarray,
    manifold,
    disruption_time: float | None = None,
    n_features_to_show: int = 5,
    output_path: str | None = None,
):
    """Plot which diagnostics drive manifold deviation over time.

    Args:
        traj: [T, D] trajectory
        time: [T] time array
        manifold: PCAManifold with diagnostic_names
        disruption_time: Time of disruption (optional)
        n_features_to_show: Number of top features to plot
        output_path: Save path (optional)
    """
    T = len(traj)

    # Compute gradient at each timestep
    distances = np.zeros(T)
    feature_deviations = np.zeros((T, manifold.n_features))

    for t in range(T):
        grad = compute_gradient_to_manifold(traj[t], manifold)
        distances[t] = grad.distance
        feature_deviations[t] = np.abs(grad.gradient_by_feature)

    # Find globally most important features
    mean_importance = feature_deviations.mean(axis=0)
    top_features = np.argsort(mean_importance)[::-1][:n_features_to_show]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: total distance
    axes[0].plot(time, distances, 'k-', linewidth=1.5, label='Total distance')
    if disruption_time:
        axes[0].axvline(disruption_time, color='red', linestyle='--', alpha=0.7, label='Disruption')
    axes[0].set_ylabel('Manifold Distance')
    axes[0].set_title('Total Distance from Stable Manifold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: per-feature contributions
    for feat_idx in top_features:
        name = manifold.diagnostic_names[feat_idx] if manifold.diagnostic_names else f"Feature {feat_idx}"
        axes[1].plot(time, feature_deviations[:, feat_idx], label=name, alpha=0.7)

    if disruption_time:
        axes[1].axvline(disruption_time, color='red', linestyle='--', alpha=0.7)

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('|Deviation|')
    axes[1].set_title(f'Top {n_features_to_show} Diagnostics Driving Manifold Deviation')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, axes


def main():
    print("=" * 70)
    print("PLASMA MANIFOLD VISUALIZATION")
    print("=" * 70)

    # Define shots
    stable_shots = [30473, 30460, 30440, 30420, 30400]
    disrupted_shots = [27177, 27499, 29484, 28298]

    # Load stable shots
    print("\n1. Loading stable shots...")
    stable_trajs = []
    stable_times = []
    stable_names = None

    for shot_id in stable_shots:
        print(f"   {shot_id}...", end=" ")
        try:
            traj, time, names, _ = load_mast_shot(shot_id)
            stable_trajs.append(traj)
            stable_times.append(time)
            if stable_names is None:
                stable_names = names
            print(f"OK ({traj.shape})")
        except Exception as e:
            print(f"FAILED: {e}")

    if len(stable_trajs) < 2:
        print("Not enough stable shots loaded!")
        return

    # Fit manifold on stable shots
    print("\n2. Fitting PCA manifold on stable shots...")
    manifold = compute_pca_manifold(stable_trajs, n_components=10, diagnostic_names=stable_names)
    print(f"   Components: {manifold.n_components}")
    print(f"   Variance explained: {manifold.explained_variance_ratio.sum()*100:.1f}%")
    print(f"   Top 3 PCs: {manifold.explained_variance_ratio[:3]*100}")

    # Print manifold summary
    print("\n" + summarize_manifold(manifold, n_pcs=3))

    # Load disrupted shots
    print("\n3. Loading disrupted shots...")
    disrupted_trajs = []
    disrupted_times = []
    disruption_times_list = []

    for shot_id in disrupted_shots:
        print(f"   {shot_id}...", end=" ")
        try:
            traj, time, _, dt = load_mast_shot(shot_id)
            disrupted_trajs.append(traj)
            disrupted_times.append(time)
            disruption_times_list.append(dt)
            print(f"OK ({traj.shape}), disruption at {dt:.3f}s" if dt else f"OK ({traj.shape})")
        except Exception as e:
            print(f"FAILED: {e}")

    if len(disrupted_trajs) == 0:
        print("No disrupted shots loaded!")
        return

    # Create output directory
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Plot 3D manifold
    print("\n4. Creating 3D manifold visualization...")
    fig, ax = plot_3d_manifold(
        stable_trajs, disrupted_trajs, manifold,
        stable_ids=[str(s) for s in stable_shots[:len(stable_trajs)]],
        disrupted_ids=[str(s) for s in disrupted_shots[:len(disrupted_trajs)]],
        disruption_times=disruption_times_list,
        time_arrays=disrupted_times,
        output_path=str(output_dir / "manifold_3d.png"),
    )

    # Plot distance time series
    print("\n5. Creating distance time series...")
    all_trajs = stable_trajs + disrupted_trajs
    all_times = stable_times + disrupted_times
    all_labels = [f"Stable {s}" for s in stable_shots[:len(stable_trajs)]] + \
                 [f"Disrupted {s}" for s in disrupted_shots[:len(disrupted_trajs)]]
    all_disruption_times = [None] * len(stable_trajs) + disruption_times_list

    fig, axes = plot_manifold_distance_timeseries(
        all_trajs, all_times, manifold, all_labels,
        disruption_times=all_disruption_times,
        output_path=str(output_dir / "manifold_distances.png"),
    )

    # Plot gradient decomposition for a disrupted shot
    if len(disrupted_trajs) > 0:
        print("\n6. Creating gradient decomposition plot...")
        fig, axes = plot_gradient_decomposition(
            disrupted_trajs[0], disrupted_times[0], manifold,
            disruption_time=disruption_times_list[0],
            n_features_to_show=8,
            output_path=str(output_dir / "gradient_decomposition.png"),
        )

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - manifold_3d.png: 3D PC space visualization")
    print("  - manifold_distances.png: Distance time series")
    print("  - gradient_decomposition.png: Feature-level gradient analysis")

    plt.show()


if __name__ == "__main__":
    main()
