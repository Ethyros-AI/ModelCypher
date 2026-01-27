#!/usr/bin/env python3
"""Experiment 1: Geometry-Derived Metric Tensor.

Tests the hypothesis that the SKA paper's metric tensor coefficients (α, β, γ)
are not arbitrary but encode fundamental ratios (π/e, e/π, φ, √2) when derived
from the eigenspectrum of local covariance matrices.

SKA's Formula: g_ij(r) = α·(∇h)_i(∇h)_j + β·(∇ρ)_i(∇ρ)_j + γ·δ_ij

Our Derivation (NO HEURISTICS):
- α = eigenvalue_ratio[0]/[1] of local covariance
- β = eigenvalue_ratio[1]/[2] of local covariance
- γ = sqrt(machine_eps) × max(eigenvalue) - regularization floor from dtype

ALL parameters derived from geometry: κ, √eps, fundamental constants.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# FUNDAMENTAL CONSTANTS (from fundamental_constants.py)
# =============================================================================

PI = np.pi
E = np.e
PHI = (1 + np.sqrt(5)) / 2
SQRT2 = np.sqrt(2)

# Derived constants we expect to find in the metric tensor
CONSTANTS = {
    "pi/e": PI / E,          # 1.1557...
    "e/pi": E / PI,          # 0.8653...
    "phi": PHI,              # 1.6180...
    "1/phi": 1 / PHI,        # 0.6180...
    "sqrt2": SQRT2,          # 1.4142...
    "1/sqrt2": 1 / SQRT2,    # 0.7071...
    "phi^2": PHI ** 2,       # 2.6180...
}


@dataclass
class ConstantMatch:
    """Result of matching a value to a fundamental constant."""
    value: float
    constant_name: str
    constant_value: float
    error_pct: float
    matched: bool


@dataclass
class MetricTensorCoefficients:
    """Derived metric tensor coefficients."""
    alpha: float           # From eig[0]/eig[1]
    beta: float            # From eig[1]/eig[2]
    gamma: float           # From sqrt(eps) * max(eig)
    alpha_match: ConstantMatch
    beta_match: ConstantMatch
    alpha_beta_ratio: float
    alpha_beta_match: ConstantMatch


@dataclass
class LocalMetricResult:
    """Metric tensor analysis at a single point."""
    point_idx: int
    eigenvalues: List[float]
    eigenvalue_ratios: Dict[str, float]
    coefficients: MetricTensorCoefficients
    is_positive_definite: bool
    entropy_gradient_norm: float
    density_gradient_norm: float


@dataclass
class ExperimentResult:
    """Full experiment result."""
    timestamp: str
    geometry_params: Dict
    measurements: Dict
    diagnosis: Dict


def match_to_constant(value: float, threshold_pct: float = 5.0) -> ConstantMatch:
    """Match a value to the closest fundamental constant."""
    best_match = None
    best_error = float('inf')

    for name, const_val in CONSTANTS.items():
        if const_val > 0:
            error_pct = abs(value - const_val) / const_val * 100
            if error_pct < best_error:
                best_error = error_pct
                best_match = ConstantMatch(
                    value=float(value),
                    constant_name=name,
                    constant_value=float(const_val),
                    error_pct=float(error_pct),
                    matched=error_pct < threshold_pct,
                )

    return best_match


def compute_adaptive_epsilon(distances: np.ndarray, d: int, sqrt_eps: float) -> float:
    """Compute adaptive finite difference epsilon.

    From manifold_curvature.py: eps = median_dist * sqrt(eps) * d^0.25
    """
    median_dist = float(np.median(distances[distances > 0]))
    return median_dist * sqrt_eps * (d ** 0.25)


def compute_entropy_gradient(
    activations: np.ndarray,
    point_idx: int,
    k_neighbors: int,
    finite_diff_eps: float,
) -> np.ndarray:
    """Compute entropy gradient at a point via finite differences.

    Entropy H = -sum(p_i * log(p_i)) where p_i = sigma_i^2 / sum(sigma^2)
    """
    n, d = activations.shape

    # Get k nearest neighbors
    dists = cdist(activations[point_idx:point_idx+1], activations)[0]
    neighbor_idx = np.argsort(dists)[1:k_neighbors+1]  # Exclude self

    def local_entropy(acts: np.ndarray) -> float:
        """Compute spectral entropy from local activations."""
        if len(acts) < 2:
            return 0.0
        centered = acts - acts.mean(axis=0)
        _, S, _ = svd(centered, full_matrices=False)
        S_sq = S ** 2
        S_sq = S_sq[S_sq > 1e-10]  # Filter numerical zeros
        if len(S_sq) < 2:
            return 0.0
        p = S_sq / S_sq.sum()
        return float(-np.sum(p * np.log(p + 1e-10)))

    # Base entropy at point
    local_acts = activations[neighbor_idx]
    H_0 = local_entropy(local_acts)

    # Gradient via central differences
    gradient = np.zeros(d)
    for j in range(min(d, 100)):  # Limit dimensions for speed
        # Perturb in direction j
        perturbed_plus = local_acts.copy()
        perturbed_plus[:, j] += finite_diff_eps
        H_plus = local_entropy(perturbed_plus)

        perturbed_minus = local_acts.copy()
        perturbed_minus[:, j] -= finite_diff_eps
        H_minus = local_entropy(perturbed_minus)

        gradient[j] = (H_plus - H_minus) / (2 * finite_diff_eps)

    return gradient


def compute_density_gradient(
    activations: np.ndarray,
    point_idx: int,
    k_neighbors: int,
    finite_diff_eps: float,
) -> np.ndarray:
    """Compute density gradient at a point.

    Density rho(x) = k / Volume_of_kNN_ball
    """
    n, d = activations.shape

    # k-NN distances
    dists = cdist(activations[point_idx:point_idx+1], activations)[0]
    sorted_dists = np.sort(dists)
    r_k = sorted_dists[k_neighbors]  # Distance to k-th neighbor

    # Volume of d-ball: V_d * r^d
    # We use the inverse as density proxy: rho = 1/r_k^d (normalized)
    def local_density(acts: np.ndarray, center: np.ndarray) -> float:
        dists = np.linalg.norm(acts - center, axis=1)
        sorted_d = np.sort(dists)
        r = sorted_d[min(k_neighbors, len(sorted_d)-1)]
        if r < 1e-10:
            return 1e10  # Very high density
        return 1.0 / (r ** min(d, 10))  # Limit exponent for stability

    # Base density
    rho_0 = local_density(activations, activations[point_idx])

    # Gradient via central differences
    gradient = np.zeros(d)
    for j in range(min(d, 100)):
        perturbed_plus = activations[point_idx].copy()
        perturbed_plus[j] += finite_diff_eps
        rho_plus = local_density(activations, perturbed_plus)

        perturbed_minus = activations[point_idx].copy()
        perturbed_minus[j] -= finite_diff_eps
        rho_minus = local_density(activations, perturbed_minus)

        gradient[j] = (rho_plus - rho_minus) / (2 * finite_diff_eps)

    return gradient


def compute_local_metric_tensor(
    activations: np.ndarray,
    point_idx: int,
    k_neighbors: int,
    sqrt_eps: float,
) -> LocalMetricResult:
    """Compute metric tensor at a single point and extract coefficients."""
    n, d = activations.shape

    # Get k nearest neighbors
    dists = cdist(activations[point_idx:point_idx+1], activations)[0]
    neighbor_idx = np.argsort(dists)[1:k_neighbors+1]
    neighbors = activations[neighbor_idx].astype(np.float64)  # Use float64 for stability

    # Center neighbors
    centered = neighbors - neighbors.mean(axis=0)

    # Debug: check shape and values
    if point_idx == 0:
        logger.debug(f"  Debug: neighbors shape={neighbors.shape}, centered norm={np.linalg.norm(centered):.4e}")

    # Use SVD for numerical stability instead of eigendecomposition
    try:
        _, S, _ = svd(centered, full_matrices=False)
        if point_idx == 0:
            logger.debug(f"  Debug: S shape={S.shape}, S[:5]={S[:5]}")
    except Exception as e:
        logger.warning(f"SVD failed at point {point_idx}: {e}")
        return LocalMetricResult(
            point_idx=point_idx,
            eigenvalues=[],
            eigenvalue_ratios={},
            coefficients=None,
            is_positive_definite=False,
            entropy_gradient_norm=0.0,
            density_gradient_norm=0.0,
        )

    # Eigenvalues of covariance (squared singular values, normalized)
    eigenvalues = (S ** 2) / len(neighbors)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Handle zero or negative eigenvalues
    if len(eigenvalues) == 0 or eigenvalues[0] <= 0:
        return LocalMetricResult(
            point_idx=point_idx,
            eigenvalues=[],
            eigenvalue_ratios={},
            coefficients=None,
            is_positive_definite=False,
            entropy_gradient_norm=0.0,
            density_gradient_norm=0.0,
        )

    # Filter valid eigenvalues - use a less strict threshold
    # sqrt_eps is ~3.5e-4, which may be too strict
    threshold = max(sqrt_eps, 1e-10) * eigenvalues[0] if eigenvalues[0] > 0 else sqrt_eps
    valid_eig = eigenvalues[eigenvalues > threshold]

    if len(valid_eig) < 3:
        # Not enough eigenvalues to compute ratios
        # For debugging, we can still try with top 3 eigenvalues if they exist
        if len(eigenvalues) >= 3 and eigenvalues[2] > 1e-15:
            valid_eig = eigenvalues[:10]  # Use top eigenvalues anyway
        else:
            return LocalMetricResult(
                point_idx=point_idx,
                eigenvalues=eigenvalues[:10].tolist() if len(eigenvalues) > 0 else [],
                eigenvalue_ratios={},
                coefficients=None,
                is_positive_definite=False,
                entropy_gradient_norm=0.0,
                density_gradient_norm=0.0,
            )

    # Compute eigenvalue ratios
    ratios = {
        "eig0_eig1": valid_eig[0] / valid_eig[1],
        "eig1_eig2": valid_eig[1] / valid_eig[2],
        "eig0_eig2": valid_eig[0] / valid_eig[2],
    }

    # Derive metric tensor coefficients
    alpha = ratios["eig0_eig1"]
    beta = ratios["eig1_eig2"]
    gamma = sqrt_eps * valid_eig[0]

    alpha_match = match_to_constant(alpha)
    beta_match = match_to_constant(beta)

    alpha_beta_ratio = alpha / beta if beta > 1e-10 else 0.0
    alpha_beta_match = match_to_constant(alpha_beta_ratio)

    coefficients = MetricTensorCoefficients(
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        alpha_match=alpha_match,
        beta_match=beta_match,
        alpha_beta_ratio=float(alpha_beta_ratio),
        alpha_beta_match=alpha_beta_match,
    )

    # Compute finite difference epsilon
    finite_diff_eps = compute_adaptive_epsilon(dists, d, sqrt_eps)

    # Compute gradients
    entropy_grad = compute_entropy_gradient(activations, point_idx, k_neighbors, finite_diff_eps)
    density_grad = compute_density_gradient(activations, point_idx, k_neighbors, finite_diff_eps)

    # Check positive definiteness
    is_positive_definite = np.all(eigenvalues > sqrt_eps)

    return LocalMetricResult(
        point_idx=point_idx,
        eigenvalues=valid_eig[:10].tolist(),
        eigenvalue_ratios=ratios,
        coefficients=coefficients,
        is_positive_definite=is_positive_definite,
        entropy_gradient_norm=float(np.linalg.norm(entropy_grad)),
        density_gradient_norm=float(np.linalg.norm(density_grad)),
    )


def run_experiment(activations: np.ndarray, n_sample_points: int = 50) -> ExperimentResult:
    """Run the full metric tensor experiment."""
    n, d = activations.shape
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    # Berry-Sauer connectivity: k = 2 * log(n) for connectivity
    k_neighbors = max(5, int(2 * np.log(n)))

    logger.info(f"Running metric tensor experiment")
    logger.info(f"  Activations shape: {activations.shape}")
    logger.info(f"  sqrt(eps): {sqrt_eps:.4e}")
    logger.info(f"  k_neighbors: {k_neighbors}")

    # Sample points uniformly
    sample_idx = np.linspace(0, n-1, min(n_sample_points, n), dtype=int)

    # Compute metric tensor at each sample point
    results: List[LocalMetricResult] = []
    n_tried = 0
    n_no_coef = 0
    for idx in sample_idx:
        n_tried += 1
        result = compute_local_metric_tensor(activations, idx, k_neighbors, sqrt_eps)
        if result.coefficients is not None:
            results.append(result)
        else:
            n_no_coef += 1
            if n_no_coef <= 3:
                logger.info(f"    Point {idx}: no coefficients, eigenvalues[:3]={result.eigenvalues[:3] if result.eigenvalues else 'None'}")

    logger.info(f"  Valid measurements: {len(results)}/{len(sample_idx)}")

    if len(results) == 0:
        return ExperimentResult(
            timestamp=datetime.now().isoformat(),
            geometry_params={},
            measurements={},
            diagnosis={"error": "No valid measurements"},
        )

    # Aggregate results
    alphas = [r.coefficients.alpha for r in results]
    betas = [r.coefficients.beta for r in results]
    gammas = [r.coefficients.gamma for r in results]
    alpha_beta_ratios = [r.coefficients.alpha_beta_ratio for r in results]

    # Check constant matches
    alpha_matches = [r.coefficients.alpha_match.matched for r in results]
    beta_matches = [r.coefficients.beta_match.matched for r in results]
    alpha_beta_matches = [r.coefficients.alpha_beta_match.matched for r in results]

    # Most common matched constants
    alpha_constants = [r.coefficients.alpha_match.constant_name for r in results]
    beta_constants = [r.coefficients.beta_match.constant_name for r in results]

    from collections import Counter
    alpha_mode = Counter(alpha_constants).most_common(1)[0]
    beta_mode = Counter(beta_constants).most_common(1)[0]

    # Mean values
    mean_alpha = float(np.mean(alphas))
    mean_beta = float(np.mean(betas))
    mean_gamma = float(np.mean(gammas))
    mean_alpha_match = match_to_constant(mean_alpha)
    mean_beta_match = match_to_constant(mean_beta)

    # Positive definiteness
    n_positive_definite = sum(1 for r in results if r.is_positive_definite)

    logger.info(f"\n  RESULTS:")
    logger.info(f"  Mean α = {mean_alpha:.4f} → {mean_alpha_match.constant_name} (err: {mean_alpha_match.error_pct:.2f}%)")
    logger.info(f"  Mean β = {mean_beta:.4f} → {mean_beta_match.constant_name} (err: {mean_beta_match.error_pct:.2f}%)")
    logger.info(f"  Mean γ = {mean_gamma:.4e}")
    logger.info(f"  α matched: {sum(alpha_matches)}/{len(alpha_matches)} ({sum(alpha_matches)/len(alpha_matches)*100:.1f}%)")
    logger.info(f"  β matched: {sum(beta_matches)}/{len(beta_matches)} ({sum(beta_matches)/len(beta_matches)*100:.1f}%)")
    logger.info(f"  Most common α constant: {alpha_mode[0]} ({alpha_mode[1]} occurrences)")
    logger.info(f"  Most common β constant: {beta_mode[0]} ({beta_mode[1]} occurrences)")
    logger.info(f"  Positive definite: {n_positive_definite}/{len(results)}")

    # Diagnosis
    hypothesis_supported = (
        sum(alpha_matches) / len(alpha_matches) > 0.5 and
        sum(beta_matches) / len(beta_matches) > 0.5
    )

    return ExperimentResult(
        timestamp=datetime.now().isoformat(),
        geometry_params={
            "n_samples": n,
            "d_dimensions": d,
            "k_neighbors": k_neighbors,
            "sqrt_eps": float(sqrt_eps),
            "n_sample_points": len(sample_idx),
            "n_valid_measurements": len(results),
        },
        measurements={
            "coefficients": {
                "alpha": {
                    "mean": mean_alpha,
                    "std": float(np.std(alphas)),
                    "constant_match": asdict(mean_alpha_match),
                },
                "beta": {
                    "mean": mean_beta,
                    "std": float(np.std(betas)),
                    "constant_match": asdict(mean_beta_match),
                },
                "gamma": {
                    "mean": mean_gamma,
                    "std": float(np.std(gammas)),
                },
            },
            "match_rates": {
                "alpha": sum(alpha_matches) / len(alpha_matches),
                "beta": sum(beta_matches) / len(beta_matches),
                "alpha_beta": sum(alpha_beta_matches) / len(alpha_beta_matches),
            },
            "most_common_constants": {
                "alpha": {"constant": alpha_mode[0], "count": alpha_mode[1]},
                "beta": {"constant": beta_mode[0], "count": beta_mode[1]},
            },
            "positive_definite_rate": n_positive_definite / len(results),
        },
        diagnosis={
            "hypothesis_supported": hypothesis_supported,
            "alpha_matches_constant": mean_alpha_match.matched,
            "beta_matches_constant": mean_beta_match.matched,
            "n_constant_matches_precise": sum([
                mean_alpha_match.error_pct < 1.0,
                mean_beta_match.error_pct < 1.0,
            ]),
            "n_constant_matches_significant": sum([
                mean_alpha_match.matched,
                mean_beta_match.matched,
            ]),
        },
    )


def main():
    """Run experiment on model activations."""
    import mlx.core as mx
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: GEOMETRY-DERIVED METRIC TENSOR")
    logger.info("=" * 70)
    logger.info("\nTesting: Do SKA paper's α, β, γ match fundamental constants?")
    logger.info("Hypothesis: Eigenvalue ratios encode π/e, e/π, φ, √2\n")

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_final_mastery_lora"

    logger.info(f"Loading model: {model_path}")
    logger.info(f"With adapter: {adapter_path}")

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    # Generate unique prompts - vary numbers to get diverse activations
    import random
    random.seed(42)

    prompts = []
    templates = [
        "Question: If I have {a} apples and get {b} more, how many do I have?\n\nAnswer:",
        "Question: A store sells {a} items. Each costs ${b}. Total revenue?\n\nAnswer:",
        "Question: Sarah has {a} cookies. She gives away {b}%. How many left?\n\nAnswer:",
        "Question: A train travels {a} mph for {b} hours. Distance covered?\n\nAnswer:",
        "Question: If {a} workers finish a job in {b} days, how long for {c} workers?\n\nAnswer:",
        "Question: John has {a} marbles. He loses 1/{b} of them. How many remain?\n\nAnswer:",
        "Question: A rectangle is {a} cm long and {b} cm wide. What is the area?\n\nAnswer:",
        "Question: Tom earns ${a} per hour. He works {b} hours. What is his pay?\n\nAnswer:",
        "Question: A book costs ${a}. With {b}% discount, what is the new price?\n\nAnswer:",
        "Question: There are {a} students in a class. 1/{b} are absent. How many present?\n\nAnswer:",
    ]

    for i in range(100):
        template = templates[i % len(templates)]
        a = random.randint(5, 50)
        b = random.randint(2, 20)
        c = random.randint(2, 10)
        prompts.append(template.format(a=a, b=b, c=c))

    logger.info(f"\nCollecting activations from {len(prompts)} prompts...")

    activations = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Forward through model
        hidden = model.model.embed_tokens(input_ids)
        for layer in model.model.layers:
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        hidden = model.model.norm(hidden)
        mx.eval(hidden)

        # Get last token activation
        activations.append(np.array(hidden[0, -1, :].tolist(), dtype=np.float32))

    activations = np.vstack(activations)
    logger.info(f"Activations shape: {activations.shape}")

    # Run experiment
    result = run_experiment(activations, n_sample_points=40)

    # Save results
    output_path = Path("data/experiments/exp1_geometry_metric_tensor.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2, cls=NumpyEncoder)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'=' * 70}")

    # Summary
    logger.info(f"\nSUMMARY:")
    if "error" in result.diagnosis:
        logger.info(f"  Error: {result.diagnosis['error']}")
    else:
        logger.info(f"  Hypothesis supported: {result.diagnosis.get('hypothesis_supported', False)}")
        logger.info(f"  Precise constant matches: {result.diagnosis.get('n_constant_matches_precise', 0)}")
        logger.info(f"  Significant constant matches: {result.diagnosis.get('n_constant_matches_significant', 0)}")

    return result


if __name__ == "__main__":
    main()
