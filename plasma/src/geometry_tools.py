"""
Geometric analysis tools for plasma trajectories.

Adapts the LLM geometry tools from ModelCypher for plasma state analysis.
The key insight: plasma diagnostic trajectories through time are analogous
to activation trajectories through layers in LLMs.

| LLM Concept | Plasma Analog |
|-------------|---------------|
| Token | Time slice |
| Layer | Not applicable (single "layer") |
| Activation vector | Diagnostic state vector |
| Layer trajectory | Temporal trajectory |
| Attention pattern | Correlation structure |
"""

import numpy as np
from dataclasses import dataclass
from typing import Sequence

# Import plasma data structures
from data_loader import PlasmaShot


@dataclass
class GeometricProfile:
    """Geometric analysis of a plasma trajectory."""
    shot_id: str

    # Per-timestep metrics
    time: np.ndarray  # [T]
    expansion_ratio: np.ndarray  # [T-1] ratio of output/input norms
    local_dimension: np.ndarray  # [T] effective dimension at each timestep
    spectral_entropy: np.ndarray  # [T] entropy of local covariance spectrum

    # Global metrics
    mean_expansion: float
    expansion_variance: float
    mean_dimension: float
    dimension_variance: float

    # Disruption-specific
    pre_disruption_signature: dict | None  # Metrics in window before disruption


def compute_expansion_ratio(
    trajectory: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:
    """Compute expansion ratio along trajectory.

    For LLMs, expansion_ratio = ||output|| / ||input|| per layer.
    For plasma, we compute the ratio of trajectory "speed" at each step,
    measuring how much the state is changing.

    Args:
        trajectory: [T, D] array of states
        window_size: Window for smoothing

    Returns:
        [T-1] array of expansion ratios
    """
    # Compute step-to-step changes
    deltas = np.diff(trajectory, axis=0)
    step_norms = np.linalg.norm(deltas, axis=1)

    # Expansion ratio: current step norm / previous step norm
    # (analogous to output/input norm ratio)
    expansion = step_norms[1:] / (step_norms[:-1] + 1e-10)

    # Optional smoothing
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        expansion = np.convolve(expansion, kernel, mode='same')

    return expansion


def compute_local_dimension(
    trajectory: np.ndarray,
    window_size: int = 50,
    method: str = "eigenvalue",
) -> np.ndarray:
    """Compute local intrinsic dimension along trajectory.

    For LLMs, we measure intrinsic dimension of activations at each layer.
    For plasma, we measure the effective dimensionality of the local
    state space around each time point.

    Args:
        trajectory: [T, D] array of states
        window_size: Window size for local covariance estimation
        method: "eigenvalue" (spectral) or "mle" (maximum likelihood)

    Returns:
        [T] array of local dimensions (NaN at boundaries)
    """
    T, D = trajectory.shape
    dimensions = np.full(T, np.nan)
    half_window = window_size // 2

    for t in range(half_window, T - half_window):
        window = trajectory[t - half_window:t + half_window]

        if method == "eigenvalue":
            # Eigenvalue-based effective dimension
            cov = np.cov(window.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            p = eigenvalues / eigenvalues.sum()
            dimensions[t] = np.exp(-np.sum(p * np.log(p)))

        elif method == "mle":
            # MLE intrinsic dimension (Levina-Bickel)
            # Compute k-NN distances
            from scipy.spatial.distance import pdist, squareform
            dists = squareform(pdist(window))
            k = min(10, len(window) - 1)
            knn_dists = np.sort(dists, axis=1)[:, 1:k+1]  # Exclude self

            # MLE estimate
            log_ratios = np.log(knn_dists[:, -1:] / knn_dists[:, :-1])
            dimensions[t] = (k - 1) / np.mean(log_ratios.sum(axis=1))

    return dimensions


def compute_spectral_entropy(
    trajectory: np.ndarray,
    window_size: int = 50,
) -> np.ndarray:
    """Compute spectral entropy of local covariance.

    Spectral entropy measures the "flatness" of the eigenvalue spectrum.
    High entropy = uniform eigenvalues = high effective dimension.
    Low entropy = concentrated eigenvalues = low effective dimension.

    For LLMs, this captures whether processing is "focused" or "diffuse".
    For plasma, it captures whether the dynamics are confined to a
    low-dimensional subspace or spread across many modes.

    Args:
        trajectory: [T, D] array of states
        window_size: Window size for local covariance estimation

    Returns:
        [T] array of spectral entropies (NaN at boundaries)
    """
    T, D = trajectory.shape
    entropy = np.full(T, np.nan)
    half_window = window_size // 2

    for t in range(half_window, T - half_window):
        window = trajectory[t - half_window:t + half_window]
        cov = np.cov(window.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        p = eigenvalues / eigenvalues.sum()
        entropy[t] = -np.sum(p * np.log(p))

    return entropy


def compute_jacobian_approximation(
    trajectory: np.ndarray,
    window_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate local Jacobian of the dynamics.

    For LLMs, the Jacobian captures how perturbations propagate through layers.
    For plasma, we approximate the local linearized dynamics:
    x(t+1) ≈ J(t) @ x(t) + c

    We estimate J from local trajectory data using least squares.

    Args:
        trajectory: [T, D] array of states

    Returns:
        jacobian_ranks: [T] effective rank of local Jacobian
        jacobian_norms: [T] spectral norm of local Jacobian
    """
    T, D = trajectory.shape
    ranks = np.full(T, np.nan)
    norms = np.full(T, np.nan)
    half_window = window_size // 2

    for t in range(half_window, T - half_window - 1):
        # Get window of states and their successors
        X = trajectory[t - half_window:t + half_window]  # [W, D]
        Y = trajectory[t - half_window + 1:t + half_window + 1]  # [W, D]

        # Least squares: Y = X @ J.T + c
        # Solve for J using pseudoinverse
        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)

        try:
            J = np.linalg.lstsq(X_centered, Y_centered, rcond=None)[0].T
            s = np.linalg.svd(J, compute_uv=False)

            # Effective rank
            s_normalized = s / (s.sum() + 1e-10)
            ranks[t] = np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-10)))

            # Spectral norm
            norms[t] = s[0]
        except np.linalg.LinAlgError:
            pass

    return ranks, norms


def analyze_shot(
    shot: PlasmaShot,
    window_size: int = 50,
) -> GeometricProfile:
    """Compute full geometric profile of a plasma shot.

    Args:
        shot: PlasmaShot to analyze
        window_size: Window size for local computations

    Returns:
        GeometricProfile with all computed metrics
    """
    trajectory = shot.get_trajectory()

    # Compute metrics
    expansion = compute_expansion_ratio(trajectory, window_size=10)
    local_dim = compute_local_dimension(trajectory, window_size=window_size)
    spectral_ent = compute_spectral_entropy(trajectory, window_size=window_size)

    # Pre-disruption signature
    pre_disruption = None
    if shot.disrupted and shot.disruption_time is not None:
        # Find disruption index
        disrupt_idx = np.searchsorted(shot.time, shot.disruption_time)

        # Analyze window before disruption
        pre_window = max(0, disrupt_idx - 100)
        if disrupt_idx > pre_window + 10:
            pre_expansion = expansion[pre_window:disrupt_idx-1]
            pre_dim = local_dim[pre_window:disrupt_idx]
            pre_entropy = spectral_ent[pre_window:disrupt_idx]

            pre_disruption = {
                "expansion_trend": float(np.polyfit(range(len(pre_expansion)), pre_expansion, 1)[0]),
                "dimension_trend": float(np.polyfit(range(len(pre_dim[~np.isnan(pre_dim)])),
                                                     pre_dim[~np.isnan(pre_dim)], 1)[0]),
                "mean_expansion": float(np.nanmean(pre_expansion)),
                "max_expansion": float(np.nanmax(pre_expansion)),
                "entropy_drop": float(spectral_ent[pre_window] - spectral_ent[disrupt_idx-1])
                                if not np.isnan(spectral_ent[pre_window]) else None,
            }

    return GeometricProfile(
        shot_id=shot.shot_id,
        time=shot.time,
        expansion_ratio=expansion,
        local_dimension=local_dim,
        spectral_entropy=spectral_ent,
        mean_expansion=float(np.nanmean(expansion)),
        expansion_variance=float(np.nanvar(expansion)),
        mean_dimension=float(np.nanmean(local_dim)),
        dimension_variance=float(np.nanvar(local_dim)),
        pre_disruption_signature=pre_disruption,
    )


def compare_profiles(
    profiles: Sequence[GeometricProfile],
    group_by: str = "disrupted",
) -> dict:
    """Compare geometric profiles across shots.

    Args:
        profiles: List of GeometricProfile objects
        group_by: Grouping criterion

    Returns:
        Dict with comparison statistics
    """
    # Group profiles
    groups = {}
    for p in profiles:
        key = p.pre_disruption_signature is not None  # Proxy for disrupted
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    # Compute statistics per group
    results = {}
    for group_name, group_profiles in groups.items():
        label = "disrupted" if group_name else "stable"
        results[label] = {
            "n_shots": len(group_profiles),
            "mean_expansion": np.mean([p.mean_expansion for p in group_profiles]),
            "mean_expansion_var": np.mean([p.expansion_variance for p in group_profiles]),
            "mean_dimension": np.mean([p.mean_dimension for p in group_profiles]),
            "mean_dimension_var": np.mean([p.dimension_variance for p in group_profiles]),
        }

        # Pre-disruption signatures (for disrupted group only)
        if group_name:
            pre_sigs = [p.pre_disruption_signature for p in group_profiles
                       if p.pre_disruption_signature is not None]
            if pre_sigs:
                results[label]["pre_disruption"] = {
                    "mean_expansion_trend": np.mean([s["expansion_trend"] for s in pre_sigs]),
                    "mean_dimension_trend": np.mean([s["dimension_trend"] for s in pre_sigs]),
                    "mean_max_expansion": np.mean([s["max_expansion"] for s in pre_sigs]),
                }

    return results


if __name__ == "__main__":
    from data_loader import create_synthetic_shot

    print("Testing geometry tools on synthetic plasma data...")

    # Create test shots
    stable = create_synthetic_shot(disrupted=False, seed=42)
    disrupted = create_synthetic_shot(disrupted=True, seed=43)

    # Analyze
    stable_profile = analyze_shot(stable)
    disrupted_profile = analyze_shot(disrupted)

    print("\nStable shot profile:")
    print(f"  Mean expansion: {stable_profile.mean_expansion:.4f}")
    print(f"  Expansion variance: {stable_profile.expansion_variance:.4f}")
    print(f"  Mean dimension: {stable_profile.mean_dimension:.4f}")

    print("\nDisrupted shot profile:")
    print(f"  Mean expansion: {disrupted_profile.mean_expansion:.4f}")
    print(f"  Expansion variance: {disrupted_profile.expansion_variance:.4f}")
    print(f"  Mean dimension: {disrupted_profile.mean_dimension:.4f}")

    if disrupted_profile.pre_disruption_signature:
        print("\n  Pre-disruption signature:")
        for k, v in disrupted_profile.pre_disruption_signature.items():
            print(f"    {k}: {v}")

    # Compare populations
    print("\n" + "="*60)
    print("Population comparison (20 shots each):")
    print("="*60)

    profiles = []
    for i in range(20):
        profiles.append(analyze_shot(create_synthetic_shot(disrupted=False, seed=i)))
        profiles.append(analyze_shot(create_synthetic_shot(disrupted=True, seed=i+1000)))

    comparison = compare_profiles(profiles)
    for group, stats in comparison.items():
        print(f"\n{group.upper()}:")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2:.4f}")
            else:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
