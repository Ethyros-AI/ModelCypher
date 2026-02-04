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
from dataclasses import dataclass, field
from typing import Sequence

# Import plasma data structures
from data_loader import PlasmaShot


@dataclass
class PCAManifold:
    """PCA model for stable plasma manifold with interpretable components.

    Attributes:
        components: Principal component vectors [n_components, D]
        mean: Mean state used for centering [D]
        explained_variance: Variance explained by each PC [n_components]
        explained_variance_ratio: Fraction of variance per PC [n_components]
        n_components: Number of principal components
        n_features: Original feature dimension
        diagnostic_names: Optional names for each feature
    """
    components: np.ndarray  # [n_components, D]
    mean: np.ndarray  # [D]
    explained_variance: np.ndarray  # [n_components]
    explained_variance_ratio: np.ndarray  # [n_components]
    n_components: int
    n_features: int
    diagnostic_names: list[str] | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project states into PC space.

        Args:
            X: States [N, D] or single state [D]

        Returns:
            PC coordinates [N, n_components] or [n_components]
        """
        X = np.atleast_2d(X)
        centered = X - self.mean
        projected = centered @ self.components.T
        return projected.squeeze() if X.shape[0] == 1 else projected

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct states from PC coordinates.

        Args:
            Z: PC coordinates [N, n_components] or [n_components]

        Returns:
            Reconstructed states [N, D] or [D]
        """
        Z = np.atleast_2d(Z)
        reconstructed = Z @ self.components + self.mean
        return reconstructed.squeeze() if Z.shape[0] == 1 else reconstructed

    def get_loadings(self, pc_idx: int = 0) -> np.ndarray:
        """Get loadings (feature weights) for a specific PC.

        Args:
            pc_idx: Which principal component (0-indexed)

        Returns:
            Loadings [D] showing each feature's contribution
        """
        return self.components[pc_idx]

    def get_top_features(self, pc_idx: int = 0, n_top: int = 5) -> list[tuple[int, float, str | None]]:
        """Get features with highest absolute loadings for a PC.

        Args:
            pc_idx: Which principal component
            n_top: Number of top features to return

        Returns:
            List of (feature_idx, loading, name) tuples, sorted by |loading|
        """
        loadings = self.components[pc_idx]
        sorted_indices = np.argsort(np.abs(loadings))[::-1][:n_top]

        results = []
        for idx in sorted_indices:
            name = self.diagnostic_names[idx] if self.diagnostic_names else None
            results.append((int(idx), float(loadings[idx]), name))
        return results


@dataclass
class ManifoldGradient:
    """Gradient information for a state relative to the stable manifold.

    Attributes:
        state: Original state vector [D]
        distance: Mahalanobis distance from manifold
        reconstruction_error: L2 distance to nearest point on manifold
        gradient_direction: Unit vector pointing back to manifold [D]
        gradient_by_feature: Gradient decomposed by feature [D]
        pc_coordinates: Position in PC space [n_components]
        pc_residual: Component in null space of PCA [D]
        top_driving_features: Features most responsible for deviation
    """
    state: np.ndarray
    distance: float
    reconstruction_error: float
    gradient_direction: np.ndarray
    gradient_by_feature: np.ndarray
    pc_coordinates: np.ndarray
    pc_residual: np.ndarray
    top_driving_features: list[tuple[int, float, str | None]]


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


def compute_pca_manifold(
    trajectories: list[np.ndarray],
    n_components: int = 10,
    diagnostic_names: list[str] | None = None,
) -> PCAManifold:
    """Fit PCA on trajectories to define stable plasma manifold.

    Concatenates all trajectories, centers, and computes principal components.
    The resulting manifold captures the main directions of variation in
    stable plasma operation.

    Args:
        trajectories: List of [T, D] trajectory arrays (e.g., from stable shots)
        n_components: Number of principal components to retain
        diagnostic_names: Optional names for each of the D features

    Returns:
        PCAManifold object with fitted components and loadings
    """
    # Concatenate all trajectories
    all_states = np.vstack(trajectories)
    n_samples, n_features = all_states.shape

    # Compute mean for centering
    mean = all_states.mean(axis=0)
    centered = all_states - mean

    # SVD for PCA (more numerically stable than eigendecomposition)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Explained variance (singular values squared, normalized)
    explained_variance = (S ** 2) / (n_samples - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance

    # Keep only requested components
    n_components = min(n_components, len(S))
    components = Vt[:n_components]  # [n_components, D]

    return PCAManifold(
        components=components,
        mean=mean,
        explained_variance=explained_variance[:n_components],
        explained_variance_ratio=explained_variance_ratio[:n_components],
        n_components=n_components,
        n_features=n_features,
        diagnostic_names=diagnostic_names,
    )


def compute_gradient_to_manifold(
    state: np.ndarray,
    pca_model: PCAManifold,
    variance_scale: np.ndarray | None = None,
    n_top_features: int = 5,
) -> ManifoldGradient:
    """Compute gradient direction from state back to stable manifold.

    Given a plasma state and a PCA manifold of stable operation, compute:
    1. Distance from the manifold (reconstruction error + Mahalanobis)
    2. Direction vector pointing back to the manifold
    3. Which diagnostics are most responsible for the deviation

    Args:
        state: Current plasma state [D]
        pca_model: PCAManifold fitted on stable trajectories
        variance_scale: Optional per-feature variance for Mahalanobis [D]
        n_top_features: Number of top features to report

    Returns:
        ManifoldGradient with distance, direction, and feature attribution
    """
    state = np.asarray(state).flatten()

    # Project to PC space
    pc_coords = pca_model.transform(state)

    # Reconstruct on manifold
    reconstruction = pca_model.inverse_transform(pc_coords)

    # Residual (what's not captured by manifold)
    residual = state - reconstruction
    reconstruction_error = np.linalg.norm(residual)

    # Gradient direction: points from state toward manifold
    # This is the negative of the residual, normalized
    if reconstruction_error > 1e-10:
        gradient_direction = -residual / reconstruction_error
    else:
        gradient_direction = np.zeros_like(residual)

    # Mahalanobis distance in PC space (if variance provided)
    if variance_scale is not None:
        # Scale by explained variance in each PC direction
        pc_variance = pca_model.explained_variance
        mahal_squared = np.sum((pc_coords ** 2) / (pc_variance + 1e-10))
        distance = np.sqrt(mahal_squared) + reconstruction_error
    else:
        # Use reconstruction error as distance
        distance = reconstruction_error

    # Gradient decomposed by feature (residual per feature)
    gradient_by_feature = -residual  # Sign: positive means "increase this"

    # Find top features driving deviation (largest |residual|)
    sorted_indices = np.argsort(np.abs(residual))[::-1][:n_top_features]
    top_features = []
    for idx in sorted_indices:
        name = pca_model.diagnostic_names[idx] if pca_model.diagnostic_names else None
        top_features.append((int(idx), float(residual[idx]), name))

    return ManifoldGradient(
        state=state,
        distance=float(distance),
        reconstruction_error=float(reconstruction_error),
        gradient_direction=gradient_direction,
        gradient_by_feature=gradient_by_feature,
        pc_coordinates=pc_coords,
        pc_residual=residual,
        top_driving_features=top_features,
    )


def compute_trajectory_manifold_analysis(
    trajectory: np.ndarray,
    pca_model: PCAManifold,
    time: np.ndarray | None = None,
) -> dict:
    """Analyze full trajectory relative to stable manifold.

    For each timestep, compute distance and gradient direction to manifold.
    Returns time series of manifold distances and aggregate statistics.

    Args:
        trajectory: [T, D] array of states
        pca_model: PCAManifold fitted on stable trajectories
        time: Optional [T] array of time points

    Returns:
        Dict with:
        - distances: [T] array of manifold distances
        - pc_trajectory: [T, n_components] trajectory in PC space
        - reconstruction_errors: [T] array
        - mean_distance: Average distance from manifold
        - max_distance: Maximum distance
        - distance_trend: Linear trend in distance (positive = drifting away)
    """
    T, D = trajectory.shape

    distances = np.zeros(T)
    reconstruction_errors = np.zeros(T)
    pc_trajectory = np.zeros((T, pca_model.n_components))

    for t in range(T):
        grad = compute_gradient_to_manifold(trajectory[t], pca_model)
        distances[t] = grad.distance
        reconstruction_errors[t] = grad.reconstruction_error
        pc_trajectory[t] = grad.pc_coordinates

    # Compute trend (positive = moving away from manifold)
    t_axis = np.arange(T)
    valid = ~np.isnan(distances)
    if valid.sum() > 10:
        trend_coef = np.polyfit(t_axis[valid], distances[valid], 1)[0]
    else:
        trend_coef = np.nan

    result = {
        "distances": distances,
        "pc_trajectory": pc_trajectory,
        "reconstruction_errors": reconstruction_errors,
        "mean_distance": float(np.nanmean(distances)),
        "max_distance": float(np.nanmax(distances)),
        "distance_trend": float(trend_coef),
    }

    if time is not None:
        result["time"] = time

    return result


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

    # Test PCA manifold and gradient
    print("\n" + "="*60)
    print("Testing PCA manifold extraction and gradient computation:")
    print("="*60)

    # Create stable training set
    stable_trajs = [create_synthetic_shot(disrupted=False, seed=i).get_trajectory()
                    for i in range(10)]

    # Fit manifold
    manifold = compute_pca_manifold(stable_trajs, n_components=5)
    print(f"\nFitted PCA manifold:")
    print(f"  Features: {manifold.n_features}")
    print(f"  Components: {manifold.n_components}")
    print(f"  Explained variance: {manifold.explained_variance_ratio.sum()*100:.1f}%")
    print(f"  Top 3 PCs: {manifold.explained_variance_ratio[:3]*100}")

    # Test gradient on disrupted shot
    disrupted_traj = create_synthetic_shot(disrupted=True, seed=99).get_trajectory()
    analysis = compute_trajectory_manifold_analysis(disrupted_traj, manifold)

    print(f"\nDisrupted shot manifold analysis:")
    print(f"  Mean distance: {analysis['mean_distance']:.4f}")
    print(f"  Max distance: {analysis['max_distance']:.4f}")
    print(f"  Distance trend: {analysis['distance_trend']:.6f}")

    # Check gradient at a specific point
    grad = compute_gradient_to_manifold(disrupted_traj[800], manifold)
    print(f"\nGradient at t=800 (pre-disruption):")
    print(f"  Distance: {grad.distance:.4f}")
    print(f"  Reconstruction error: {grad.reconstruction_error:.4f}")
    print(f"  Top driving features: {[(f[0], f'{f[1]:.3f}') for f in grad.top_driving_features[:3]]}")
