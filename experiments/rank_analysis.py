# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rank Analysis Experiment: Measure activation matrix rank at every layer.

The goal: understand where rank(A) < hidden_dim and by how much.

This is step 1 of the research direction to achieve rank = hidden_dim at every layer.

Mathematical framework:
- A_l ∈ ℝ^(n×d) is the activation matrix at layer l (n probes, d hidden dimensions)
- We want rank(A_l) = d for all l
- Numerical rank: count of singular values σ > σ_max × sqrt(machine_epsilon)

Questions this experiment answers:
1. What is rank(A_l) at each layer for current probes?
2. How does rank deficit (d - rank) vary across layers?
3. What is the condition number at each layer?
4. What directions are in the null space (missing)?
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class LayerRankAnalysis:
    """Rank analysis for a single layer."""

    layer_idx: int
    hidden_dim: int
    n_samples: int

    # Numerical rank (σ > σ_max × sqrt(eps))
    numerical_rank: int

    # Rank deficit: how many directions are missing?
    rank_deficit: int  # hidden_dim - numerical_rank

    # Condition number (max_σ / min_σ among non-zero singular values)
    condition_number: float

    # Singular value spectrum (for analysis)
    singular_values: list[float] = field(default_factory=list)

    # Indices of "dead" directions (below threshold)
    dead_direction_count: int = 0

    @property
    def coverage_ratio(self) -> float:
        """Fraction of hidden dimensions covered by probes."""
        return self.numerical_rank / self.hidden_dim if self.hidden_dim > 0 else 0.0

    @property
    def is_full_rank(self) -> bool:
        """True if rank equals hidden_dim."""
        return self.numerical_rank >= self.hidden_dim


@dataclass
class RankAnalysisResult:
    """Complete rank analysis across all layers."""

    model_path: str
    total_probes: int
    hidden_dim: int  # Model's hidden dimension
    n_layers: int

    # Per-layer analysis
    layer_analyses: list[LayerRankAnalysis] = field(default_factory=list)

    # Summary statistics
    min_rank: int = 0
    max_rank: int = 0
    mean_rank: float = 0.0
    worst_layer_idx: int = 0
    worst_coverage: float = 0.0

    # How many layers achieve full rank?
    full_rank_layers: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_path": self.model_path,
            "total_probes": self.total_probes,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "summary": {
                "min_rank": self.min_rank,
                "max_rank": self.max_rank,
                "mean_rank": self.mean_rank,
                "worst_layer_idx": self.worst_layer_idx,
                "worst_coverage": self.worst_coverage,
                "full_rank_layers": self.full_rank_layers,
                "full_rank_fraction": self.full_rank_layers / self.n_layers if self.n_layers > 0 else 0.0,
            },
            "layers": [
                {
                    "layer_idx": la.layer_idx,
                    "hidden_dim": la.hidden_dim,
                    "n_samples": la.n_samples,
                    "numerical_rank": la.numerical_rank,
                    "rank_deficit": la.rank_deficit,
                    "coverage_ratio": la.coverage_ratio,
                    "condition_number": la.condition_number,
                    "is_full_rank": la.is_full_rank,
                    "dead_direction_count": la.dead_direction_count,
                    # Include top singular values for analysis
                    "top_10_singular_values": la.singular_values[:10] if la.singular_values else [],
                    "bottom_10_singular_values": la.singular_values[-10:] if la.singular_values else [],
                }
                for la in self.layer_analyses
            ],
        }


def compute_numerical_rank(
    activations: "Array",
    backend: "Backend",
) -> tuple[int, float, list[float], int]:
    """Compute numerical rank from SVD.

    Returns:
        (numerical_rank, condition_number, singular_values, dead_count)
    """
    from modelcypher.core.domain.geometry.numerical_stability import (
        geodesic_svd,
        machine_epsilon,
        sqrt_scalar,
    )

    b = backend

    # Compute SVD
    _, S, _ = geodesic_svd(b, activations)
    b.eval(S)

    n_singular = int(S.shape[0])
    if n_singular == 0:
        return 0, float("inf"), [], 0

    # Convert to list for analysis
    singular_values = [float(b.to_scalar(S[i])) for i in range(n_singular)]

    # Numerical rank threshold: σ_max × sqrt(machine_epsilon)
    eps = machine_epsilon(b, activations)
    threshold = singular_values[0] * sqrt_scalar(eps, b) if singular_values[0] > 0 else eps

    # Count singular values above threshold
    numerical_rank = sum(1 for s in singular_values if s > threshold)
    dead_count = n_singular - numerical_rank

    # Condition number (among non-zero singular values)
    nonzero_singular = [s for s in singular_values if s > threshold]
    if len(nonzero_singular) >= 2:
        condition_number = nonzero_singular[0] / nonzero_singular[-1]
    elif len(nonzero_singular) == 1:
        condition_number = 1.0
    else:
        condition_number = float("inf")

    return numerical_rank, condition_number, singular_values, dead_count


def analyze_layer_activations(
    activations: "Array",
    layer_idx: int,
    backend: "Backend",
) -> LayerRankAnalysis:
    """Analyze rank for a single layer's activations."""
    b = backend

    n_samples, hidden_dim = int(activations.shape[0]), int(activations.shape[1])

    numerical_rank, condition_number, singular_values, dead_count = compute_numerical_rank(
        activations, b
    )

    rank_deficit = hidden_dim - numerical_rank

    return LayerRankAnalysis(
        layer_idx=layer_idx,
        hidden_dim=hidden_dim,
        n_samples=n_samples,
        numerical_rank=numerical_rank,
        rank_deficit=rank_deficit,
        condition_number=condition_number,
        singular_values=singular_values,
        dead_direction_count=dead_count,
    )


def run_rank_analysis(
    model_path: str,
    max_probes: int | None = None,
    output_path: str | None = None,
) -> RankAnalysisResult:
    """Run rank analysis on a model.

    Args:
        model_path: Path to the model directory.
        max_probes: Maximum number of probes to use (None = all).
        output_path: Path to save JSON results.

    Returns:
        RankAnalysisResult with per-layer analysis.
    """
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.probe_loader import load_all_probes
    from modelcypher.adapters.hf_hub import HfHubAdapter
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    backend = get_default_backend()
    logger.info("Backend: %s", type(backend).__name__)

    # Load model using mlx_lm directly
    logger.info("Loading model from %s", model_path)
    from mlx_lm import load as mlx_load
    import mlx.core as mx
    import mlx.nn as nn

    model, tokenizer = mlx_load(model_path)

    # Get hidden dimension from model config or by inspection
    hidden_dim = 0
    n_layers = 0

    # Try to get from model config first
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
        # Get hidden dim from first layer's norm weight
        first_layer = model.model.layers[0]
        if hasattr(first_layer, "input_layernorm") and hasattr(first_layer.input_layernorm, "weight"):
            hidden_dim = first_layer.input_layernorm.weight.shape[0]
        elif hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
            # Fall back to q_proj shape
            hidden_dim = first_layer.self_attn.q_proj.weight.shape[1]

    logger.info("Hidden dimension: %d", hidden_dim)
    logger.info("Number of layers: %d", n_layers)

    if hidden_dim == 0 or n_layers == 0:
        raise RuntimeError(f"Could not determine model dimensions: hidden_dim={hidden_dim}, n_layers={n_layers}")

    # Load probes
    probes = load_all_probes()
    logger.info("Total probes available: %d", len(probes))

    # Filter probes with valid text
    from modelcypher.core.use_cases.merge.stages.probe_helpers import _select_probe_text
    valid_probes = []
    for probe in probes:
        text = _select_probe_text(probe)
        if text:
            valid_probes.append((probe, text))

    if max_probes and max_probes < len(valid_probes):
        valid_probes = valid_probes[:max_probes]

    logger.info("Using %d probes", len(valid_probes))

    # Collect activations
    logger.info("Collecting activations...")
    activation_provider = MLXActivationProvider()

    probe_texts = [text for _, text in valid_probes]
    batch_result = activation_provider.collect_probe_activations_batch(
        model, tokenizer, probe_texts
    )

    # Analyze each layer
    logger.info("Analyzing rank at each layer...")
    layer_analyses = []

    for layer_idx in range(n_layers):
        # Collect activations for this layer across all probes
        layer_activations = []
        for probe_idx in range(len(valid_probes)):
            if layer_idx in batch_result.hidden[probe_idx]:
                act = batch_result.hidden[probe_idx][layer_idx]
                layer_activations.append(act)

        if not layer_activations:
            logger.warning("No activations for layer %d", layer_idx)
            continue

        # Stack into [n_probes, hidden_dim]
        stacked = backend.stack(layer_activations, axis=0)
        backend.eval(stacked)

        analysis = analyze_layer_activations(stacked, layer_idx, backend)
        layer_analyses.append(analysis)

        logger.info(
            "Layer %d: rank=%d/%d (%.1f%%), deficit=%d, κ=%.2e",
            layer_idx,
            analysis.numerical_rank,
            analysis.hidden_dim,
            100 * analysis.coverage_ratio,
            analysis.rank_deficit,
            analysis.condition_number,
        )

    # Compute summary statistics
    ranks = [la.numerical_rank for la in layer_analyses]
    coverages = [la.coverage_ratio for la in layer_analyses]

    result = RankAnalysisResult(
        model_path=model_path,
        total_probes=len(valid_probes),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        layer_analyses=layer_analyses,
        min_rank=min(ranks) if ranks else 0,
        max_rank=max(ranks) if ranks else 0,
        mean_rank=sum(ranks) / len(ranks) if ranks else 0.0,
        worst_layer_idx=coverages.index(min(coverages)) if coverages else 0,
        worst_coverage=min(coverages) if coverages else 0.0,
        full_rank_layers=sum(1 for la in layer_analyses if la.is_full_rank),
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("RANK ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info("Model: %s", model_path)
    logger.info("Hidden dimension: %d", hidden_dim)
    logger.info("Total probes: %d", len(valid_probes))
    logger.info("Number of layers: %d", n_layers)
    logger.info("-" * 60)
    logger.info("Min rank across layers: %d (%.1f%% of hidden_dim)", result.min_rank, 100 * result.min_rank / hidden_dim)
    logger.info("Max rank across layers: %d (%.1f%% of hidden_dim)", result.max_rank, 100 * result.max_rank / hidden_dim)
    logger.info("Mean rank across layers: %.1f (%.1f%% of hidden_dim)", result.mean_rank, 100 * result.mean_rank / hidden_dim)
    logger.info("-" * 60)
    logger.info("Layers with full rank: %d/%d (%.1f%%)", result.full_rank_layers, n_layers, 100 * result.full_rank_layers / n_layers)
    logger.info("Worst layer: %d with %.1f%% coverage", result.worst_layer_idx, 100 * result.worst_coverage)
    logger.info("=" * 60)

    # What this means for probe generation
    if result.full_rank_layers < n_layers:
        deficit = hidden_dim - result.min_rank
        logger.info("")
        logger.info("IMPLICATION FOR PROBE GENERATION:")
        logger.info("  Current probes leave %d directions unactivated at worst layer.", deficit)
        logger.info("  To achieve full rank, we need probes that activate these directions.")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("  1. Analyze the null space at deficient layers")
        logger.info("  2. Determine if missing directions are reachable")
        logger.info("  3. Generate targeted probes for missing directions")
    else:
        logger.info("")
        logger.info("GOOD NEWS: All layers achieve full rank with current probes!")

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info("Results saved to %s", output_path)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze activation matrix rank at every layer."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of the model to analyze.",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=None,
        help="Maximum number of probes to use (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/rank_analysis.json",
        help="Output path for JSON results.",
    )

    args = parser.parse_args()

    run_rank_analysis(
        model_path=args.model,
        max_probes=args.max_probes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
