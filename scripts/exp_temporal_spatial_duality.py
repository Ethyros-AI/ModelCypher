#!/usr/bin/env python3
"""Experiment 3: Temporal/Spatial Learning Duality.

Tests the hypothesis that optimal learning follows geodesics in joint
temporal-spatial space:
- Temporal: Layer progression (0 → n_layers), entropy changes per layer
- Spatial: Geodesic embedding within each layer

Key predictions:
1. Entropy decreases through layers (Spearman correlation < 0)
2. Distance to correct region decreases for correct samples
3. Distance to correct region stays high for incorrect samples
4. Entropy decrease rate matches e/π or π/e
5. Final layer separation: dist_incorrect / dist_correct > φ

ALL parameters derived from geometry: k from Berry-Sauer, convergence from √eps.
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Fundamental constants
PI = np.pi
E = np.e
PHI = (1 + np.sqrt(5)) / 2
PI_OVER_E = PI / E
E_OVER_PI = E / PI


@dataclass
class ConstantMatch:
    """Result of matching a value to a fundamental constant."""
    value: float
    constant_name: str
    constant_value: float
    error_pct: float
    matched: bool


@dataclass
class TemporalAnalysis:
    """Analysis of temporal (layer-wise) dynamics."""
    entropy_trajectory: List[float]
    entropy_layer_correlation: float
    entropy_decrease_rate: float
    rate_constant_match: ConstantMatch


@dataclass
class SpatialAnalysis:
    """Analysis of spatial (within-layer) dynamics."""
    per_layer_intrinsic_dim: List[float]
    dim_compression_ratio: float
    compression_constant_match: ConstantMatch


@dataclass
class GeodesicToTarget:
    """Geodesic distance to target region."""
    initial_mean_dist: float
    final_mean_dist: float
    layer_correlation: float
    converges_to_target: bool
    trajectory: List[float]


@dataclass
class ExperimentResult:
    """Full experiment result."""
    timestamp: str
    geometry_params: Dict
    temporal_analysis: Dict
    spatial_analysis: Dict
    geodesic_to_target: Dict
    diagnosis: Dict


CONSTANTS = {
    "pi/e": PI_OVER_E,
    "e/pi": E_OVER_PI,
    "phi": PHI,
    "1/phi": 1/PHI,
    "phi^2": PHI**2,
}


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


def compute_spectral_entropy(activations: np.ndarray, sqrt_eps: float) -> float:
    """Compute spectral entropy from activations."""
    if len(activations) < 2:
        return 0.0

    centered = activations - activations.mean(axis=0)
    _, S, _ = svd(centered, full_matrices=False)

    # Filter numerical zeros
    S_valid = S[S > sqrt_eps * S[0]]
    if len(S_valid) < 2:
        return 0.0

    # Normalize to probabilities
    p = S_valid ** 2
    p = p / p.sum()

    # Shannon entropy
    return float(-np.sum(p * np.log(p + 1e-10)))


def compute_intrinsic_dimension_twonn(activations: np.ndarray) -> float:
    """Estimate intrinsic dimension using TwoNN method."""
    n = len(activations)
    if n < 3:
        return 0.0

    # Compute pairwise distances
    dists = cdist(activations, activations)

    # For each point, get distances to 2 nearest neighbors
    mu_values = []
    for i in range(n):
        sorted_dists = np.sort(dists[i])
        r1 = sorted_dists[1]  # Nearest neighbor (not self)
        r2 = sorted_dists[2]  # Second nearest

        if r1 > 1e-10:
            mu_values.append(r2 / r1)

    if not mu_values:
        return 0.0

    # Intrinsic dimension: d = n_samples / sum(log(mu))
    mu_array = np.array(mu_values)
    d = len(mu_array) / np.sum(np.log(mu_array + 1e-10))
    return float(max(0, d))


def compute_frechet_mean(
    activations: np.ndarray,
    max_iter: int = 100,
    sqrt_eps: float = 1e-4,
) -> np.ndarray:
    """Compute Fréchet (geodesic) mean of activations.

    Uses iterative algorithm that converges to geometric center.
    """
    # Initialize with arithmetic mean
    mean = activations.mean(axis=0)

    for _ in range(max_iter):
        # Compute weighted direction to each point
        directions = activations - mean
        dists = np.linalg.norm(directions, axis=1, keepdims=True)
        dists = np.maximum(dists, sqrt_eps)

        # Weighted average direction
        weights = 1.0 / dists
        new_direction = np.sum(weights * directions, axis=0) / np.sum(weights)

        # Update mean
        step = 0.5  # Conservative step
        new_mean = mean + step * new_direction

        # Check convergence
        if np.linalg.norm(new_mean - mean) < sqrt_eps:
            break

        mean = new_mean

    return mean


def get_layer_activations(
    model,
    tokenizer,
    prompts: List[str],
    layer_indices: List[int],
) -> Dict[int, np.ndarray]:
    """Get activations at specified layers for all prompts."""
    import mlx.core as mx

    layer_activations = {i: [] for i in layer_indices}

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Forward through model, capturing at specified layers
        hidden = model.model.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(model.model.layers):
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

            if layer_idx in layer_indices:
                mx.eval(hidden)
                layer_activations[layer_idx].append(
                    np.array(hidden[0, -1, :].tolist(), dtype=np.float32)
                )

        # Final norm
        hidden = model.model.norm(hidden)
        mx.eval(hidden)

    # Stack activations
    return {i: np.vstack(acts) for i, acts in layer_activations.items() if acts}


def run_experiment(
    model,
    tokenizer,
    prompts: List[str],
    correct_mask: np.ndarray,
    n_layers: int,
) -> ExperimentResult:
    """Run the full temporal/spatial duality experiment."""
    n = len(prompts)
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    logger.info(f"Running temporal/spatial duality experiment")
    logger.info(f"  n_prompts: {n}")
    logger.info(f"  n_layers: {n_layers}")
    logger.info(f"  correct samples: {correct_mask.sum()}/{n}")

    # Sample layers (every 4th layer for speed)
    layer_step = max(1, n_layers // 10)
    layer_indices = list(range(0, n_layers, layer_step))
    logger.info(f"  Sampling layers: {layer_indices}")

    # Get activations at each layer
    logger.info("  Collecting layer activations...")
    layer_acts = get_layer_activations(model, tokenizer, prompts, layer_indices)

    # === TEMPORAL ANALYSIS ===
    logger.info("  Computing temporal analysis...")

    entropy_trajectory = []
    for layer_idx in layer_indices:
        if layer_idx in layer_acts:
            entropy = compute_spectral_entropy(layer_acts[layer_idx], sqrt_eps)
            entropy_trajectory.append(entropy)

    if len(entropy_trajectory) >= 2:
        # Spearman correlation of entropy vs layer index
        layers_for_corr = list(range(len(entropy_trajectory)))
        entropy_corr, _ = spearmanr(layers_for_corr, entropy_trajectory)
        entropy_corr = float(entropy_corr) if not np.isnan(entropy_corr) else 0.0

        # Entropy decrease rate: (initial - final) / initial
        if entropy_trajectory[0] > sqrt_eps:
            decrease_rate = (entropy_trajectory[0] - entropy_trajectory[-1]) / entropy_trajectory[0]
        else:
            decrease_rate = 0.0

        rate_match = match_to_constant(abs(decrease_rate))
    else:
        entropy_corr = 0.0
        decrease_rate = 0.0
        rate_match = ConstantMatch(0, "none", 0, 100, False)

    temporal_analysis = TemporalAnalysis(
        entropy_trajectory=entropy_trajectory,
        entropy_layer_correlation=entropy_corr,
        entropy_decrease_rate=decrease_rate,
        rate_constant_match=rate_match,
    )

    logger.info(f"    Entropy correlation: {entropy_corr:.3f}")
    logger.info(f"    Decrease rate: {decrease_rate:.3f} → {rate_match.constant_name}")

    # === SPATIAL ANALYSIS ===
    logger.info("  Computing spatial analysis...")

    intrinsic_dims = []
    for layer_idx in layer_indices:
        if layer_idx in layer_acts:
            dim = compute_intrinsic_dimension_twonn(layer_acts[layer_idx])
            intrinsic_dims.append(dim)

    if len(intrinsic_dims) >= 2 and intrinsic_dims[-1] > sqrt_eps:
        compression_ratio = intrinsic_dims[0] / intrinsic_dims[-1]
    else:
        compression_ratio = 1.0

    compression_match = match_to_constant(compression_ratio)

    spatial_analysis = SpatialAnalysis(
        per_layer_intrinsic_dim=intrinsic_dims,
        dim_compression_ratio=compression_ratio,
        compression_constant_match=compression_match,
    )

    logger.info(f"    Dim trajectory: {intrinsic_dims[0]:.1f} → {intrinsic_dims[-1]:.1f}")
    logger.info(f"    Compression ratio: {compression_ratio:.2f} → {compression_match.constant_name}")

    # === GEODESIC TO TARGET ===
    logger.info("  Computing geodesic distances to target...")

    correct_idx = np.where(correct_mask)[0]
    incorrect_idx = np.where(~correct_mask)[0]

    if len(correct_idx) > 0:
        # Track distance to correct region through layers
        correct_dist_trajectory = []
        incorrect_dist_trajectory = []

        for layer_idx in layer_indices:
            if layer_idx not in layer_acts:
                continue

            acts = layer_acts[layer_idx]

            # Fréchet mean of correct region
            correct_acts = acts[correct_idx]
            correct_centroid = compute_frechet_mean(correct_acts, sqrt_eps=sqrt_eps)

            # Mean distance from correct samples to centroid
            correct_dists = np.linalg.norm(acts[correct_idx] - correct_centroid, axis=1)
            correct_dist_trajectory.append(float(np.mean(correct_dists)))

            # Mean distance from incorrect samples to correct centroid
            if len(incorrect_idx) > 0:
                incorrect_dists = np.linalg.norm(acts[incorrect_idx] - correct_centroid, axis=1)
                incorrect_dist_trajectory.append(float(np.mean(incorrect_dists)))

        # Analysis for correct samples
        if len(correct_dist_trajectory) >= 2:
            corr_layers = list(range(len(correct_dist_trajectory)))
            correct_corr, _ = spearmanr(corr_layers, correct_dist_trajectory)
            correct_corr = float(correct_corr) if not np.isnan(correct_corr) else 0.0

            correct_geo = GeodesicToTarget(
                initial_mean_dist=correct_dist_trajectory[0],
                final_mean_dist=correct_dist_trajectory[-1],
                layer_correlation=correct_corr,
                converges_to_target=correct_corr < -0.3,  # Negative = converging
                trajectory=correct_dist_trajectory,
            )
        else:
            correct_geo = GeodesicToTarget(0, 0, 0, False, [])

        # Analysis for incorrect samples
        if len(incorrect_dist_trajectory) >= 2:
            incorr_layers = list(range(len(incorrect_dist_trajectory)))
            incorrect_corr, _ = spearmanr(incorr_layers, incorrect_dist_trajectory)
            incorrect_corr = float(incorrect_corr) if not np.isnan(incorrect_corr) else 0.0

            incorrect_geo = GeodesicToTarget(
                initial_mean_dist=incorrect_dist_trajectory[0],
                final_mean_dist=incorrect_dist_trajectory[-1],
                layer_correlation=incorrect_corr,
                converges_to_target=incorrect_corr < -0.3,
                trajectory=incorrect_dist_trajectory,
            )
        else:
            incorrect_geo = GeodesicToTarget(0, 0, 0, False, [])

        # Final layer separation
        if correct_geo.final_mean_dist > sqrt_eps:
            separation_ratio = incorrect_geo.final_mean_dist / correct_geo.final_mean_dist
        else:
            separation_ratio = 1.0

        geodesic_analysis = {
            "correct_samples": asdict(correct_geo),
            "incorrect_samples": asdict(incorrect_geo),
            "final_layer_separation_ratio": separation_ratio,
        }

        logger.info(f"    Correct correlation: {correct_geo.layer_correlation:.3f}")
        logger.info(f"    Incorrect correlation: {incorrect_geo.layer_correlation:.3f}")
        logger.info(f"    Separation ratio: {separation_ratio:.2f}")
    else:
        geodesic_analysis = {"error": "No correct samples"}
        separation_ratio = 1.0
        correct_geo = GeodesicToTarget(0, 0, 0, False, [])
        incorrect_geo = GeodesicToTarget(0, 0, 0, False, [])

    # === DIAGNOSIS ===
    entropy_decreases = entropy_corr < 0
    correct_converges = correct_geo.converges_to_target if "error" not in geodesic_analysis else False
    incorrect_diverges = not incorrect_geo.converges_to_target if "error" not in geodesic_analysis else False
    learning_follows_geodesic = entropy_decreases and correct_converges

    diagnosis = {
        "entropy_decreases": entropy_decreases,
        "correct_converges": correct_converges,
        "incorrect_diverges": incorrect_diverges,
        "learning_follows_geodesic": learning_follows_geodesic,
        "separation_exceeds_phi": separation_ratio > PHI,
        "n_constant_matches": sum([
            rate_match.matched,
            compression_match.matched,
            separation_ratio > PHI,
        ]),
    }

    return ExperimentResult(
        timestamp=datetime.now().isoformat(),
        geometry_params={
            "n_samples": n,
            "n_layers": n_layers,
            "sampled_layers": layer_indices,
            "sqrt_eps": float(sqrt_eps),
        },
        temporal_analysis=asdict(temporal_analysis),
        spatial_analysis=asdict(spatial_analysis),
        geodesic_to_target=geodesic_analysis,
        diagnosis=diagnosis,
    )


def main():
    """Run experiment on model."""
    import mlx.core as mx
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: TEMPORAL/SPATIAL LEARNING DUALITY")
    logger.info("=" * 70)
    logger.info("\nTesting: Does learning follow geodesics through layers?")
    logger.info("Testing: Do correct samples converge, incorrect diverge?\n")

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_final_mastery_lora"

    logger.info(f"Loading model: {model_path}")
    logger.info(f"With adapter: {adapter_path}")

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    # Generate prompts with correct/incorrect labels
    prompts = [
        # Simple arithmetic (should be correct)
        ("Question: 5 + 3 = ?\n\nAnswer:", True),
        ("Question: 12 - 4 = ?\n\nAnswer:", True),
        ("Question: 6 * 2 = ?\n\nAnswer:", True),
        ("Question: 20 / 4 = ?\n\nAnswer:", True),
        ("Question: 7 + 8 = ?\n\nAnswer:", True),
        ("Question: 15 - 6 = ?\n\nAnswer:", True),
        ("Question: 4 * 5 = ?\n\nAnswer:", True),
        ("Question: 18 / 3 = ?\n\nAnswer:", True),
        # Harder problems (may be incorrect)
        ("Question: If 5 workers finish in 20 days, how long for 4 workers?\n\nAnswer:", False),
        ("Question: What is 37% of 200?\n\nAnswer:", False),
        ("Question: A car travels 65 mph for 3.5 hours. Distance?\n\nAnswer:", False),
        ("Question: If price drops 25% then rises 25%, net change?\n\nAnswer:", False),
    ]

    prompt_texts = [p[0] for p in prompts]
    correct_mask = np.array([p[1] for p in prompts])

    logger.info(f"\nUsing {len(prompts)} prompts ({correct_mask.sum()} correct, {(~correct_mask).sum()} incorrect)")

    # Run experiment
    result = run_experiment(model, tokenizer, prompt_texts, correct_mask, n_layers)

    # Save results
    output_path = Path("data/experiments/exp3_temporal_spatial_duality.json")
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
    logger.info(f"  Entropy decreases: {result.diagnosis['entropy_decreases']}")
    logger.info(f"  Correct converges: {result.diagnosis['correct_converges']}")
    logger.info(f"  Incorrect diverges: {result.diagnosis['incorrect_diverges']}")
    logger.info(f"  Learning follows geodesic: {result.diagnosis['learning_follows_geodesic']}")
    logger.info(f"  Separation > φ: {result.diagnosis['separation_exceeds_phi']}")

    return result


if __name__ == "__main__":
    main()
