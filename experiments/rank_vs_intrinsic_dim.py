# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare Numerical Rank vs Intrinsic Dimension at Each Layer.

The key question: Is rank deficiency due to sampling (n < ID) or architecture (ID < hidden_dim)?

This experiment computes:
1. Numerical rank from SVD (what we can currently cover with probes)
2. Intrinsic dimension from TwoNN (what the manifold actually contains)

If rank ≈ ID: Our probes cover the reachable subspace. Adding more won't help.
If rank < ID: We need more/better probes to span the manifold.
If rank > ID: Impossible by definition (rank ≤ ID ≤ hidden_dim).

The goal becomes: rank = intrinsic_dim at every layer (not hidden_dim).
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
class LayerDimensionAnalysis:
    """Dimension analysis for a single layer."""

    layer_idx: int
    hidden_dim: int
    n_samples: int

    # Numerical rank from SVD
    numerical_rank: int

    # Intrinsic dimension from TwoNN
    intrinsic_dim: float
    id_usable_samples: int

    # Comparison metrics
    rank_to_id_ratio: float  # rank / ID - should be ≈ 1.0 if well covered
    rank_to_hidden_ratio: float  # rank / hidden_dim
    id_to_hidden_ratio: float  # ID / hidden_dim (compression factor)

    @property
    def is_fully_covered(self) -> bool:
        """True if rank ≈ intrinsic dimension (within 10%)."""
        if self.intrinsic_dim < 1:
            return True
        return self.rank_to_id_ratio >= 0.9

    @property
    def compression_factor(self) -> float:
        """How much the representation is compressed at this layer."""
        return 1.0 - self.id_to_hidden_ratio if self.id_to_hidden_ratio > 0 else 0.0


@dataclass
class DimensionComparisonResult:
    """Complete dimension analysis across all layers."""

    model_path: str
    total_probes: int
    hidden_dim: int
    n_layers: int

    # Per-layer analysis
    layer_analyses: list[LayerDimensionAnalysis] = field(default_factory=list)

    # Summary
    mean_intrinsic_dim: float = 0.0
    min_intrinsic_dim: float = 0.0
    max_intrinsic_dim: float = 0.0
    bottleneck_layer: int = 0  # Layer with lowest ID
    layers_fully_covered: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_path": self.model_path,
            "total_probes": self.total_probes,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "summary": {
                "mean_intrinsic_dim": self.mean_intrinsic_dim,
                "min_intrinsic_dim": self.min_intrinsic_dim,
                "max_intrinsic_dim": self.max_intrinsic_dim,
                "bottleneck_layer": self.bottleneck_layer,
                "layers_fully_covered": self.layers_fully_covered,
                "coverage_fraction": self.layers_fully_covered / self.n_layers if self.n_layers > 0 else 0.0,
            },
            "layers": [
                {
                    "layer_idx": la.layer_idx,
                    "hidden_dim": la.hidden_dim,
                    "n_samples": la.n_samples,
                    "numerical_rank": la.numerical_rank,
                    "intrinsic_dim": la.intrinsic_dim,
                    "rank_to_id_ratio": la.rank_to_id_ratio,
                    "rank_to_hidden_ratio": la.rank_to_hidden_ratio,
                    "id_to_hidden_ratio": la.id_to_hidden_ratio,
                    "compression_factor": la.compression_factor,
                    "is_fully_covered": la.is_fully_covered,
                }
                for la in self.layer_analyses
            ],
        }


def compute_numerical_rank(
    activations: "Array",
    backend: "Backend",
) -> int:
    """Compute numerical rank from SVD."""
    from modelcypher.core.domain.geometry.numerical_stability import (
        geodesic_svd,
        machine_epsilon,
        sqrt_scalar,
    )

    _, S, _ = geodesic_svd(backend, activations)
    backend.eval(S)

    n_singular = int(S.shape[0])
    if n_singular == 0:
        return 0

    singular_values = [float(backend.to_scalar(S[i])) for i in range(n_singular)]

    eps = machine_epsilon(backend, activations)
    threshold = singular_values[0] * sqrt_scalar(eps, backend) if singular_values[0] > 0 else eps

    return sum(1 for s in singular_values if s > threshold)


def compute_intrinsic_dim(
    activations: "Array",
    backend: "Backend",
) -> tuple[float, int]:
    """Compute intrinsic dimension using TwoNN."""
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    estimator = IntrinsicDimension(backend)
    try:
        result = estimator.compute(activations, with_ci=False)
        return result.intrinsic_dimension, result.usable_count
    except Exception as e:
        logger.warning("TwoNN estimation failed: %s", e)
        return 0.0, 0


def run_comparison(
    model_path: str,
    max_probes: int | None = None,
    output_path: str | None = None,
) -> DimensionComparisonResult:
    """Run rank vs intrinsic dimension comparison."""
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.probe_loader import load_all_probes
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    backend = get_default_backend()
    logger.info("Backend: %s", type(backend).__name__)

    # Load model
    logger.info("Loading model from %s", model_path)
    from mlx_lm import load as mlx_load
    import mlx.core as mx

    model, tokenizer = mlx_load(model_path)

    # Get dimensions
    hidden_dim = 0
    n_layers = 0
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
        first_layer = model.model.layers[0]
        if hasattr(first_layer, "input_layernorm") and hasattr(first_layer.input_layernorm, "weight"):
            hidden_dim = first_layer.input_layernorm.weight.shape[0]
        elif hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
            hidden_dim = first_layer.self_attn.q_proj.weight.shape[1]

    logger.info("Hidden dimension: %d", hidden_dim)
    logger.info("Number of layers: %d", n_layers)

    if hidden_dim == 0 or n_layers == 0:
        raise RuntimeError(f"Could not determine model dimensions")

    # Load probes
    probes = load_all_probes()
    logger.info("Total probes available: %d", len(probes))

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
    logger.info("Analyzing each layer (rank vs intrinsic dimension)...")
    layer_analyses = []

    for layer_idx in range(n_layers):
        layer_activations = []
        for probe_idx in range(len(valid_probes)):
            if layer_idx in batch_result.hidden[probe_idx]:
                act = batch_result.hidden[probe_idx][layer_idx]
                layer_activations.append(act)

        if not layer_activations:
            logger.warning("No activations for layer %d", layer_idx)
            continue

        stacked = backend.stack(layer_activations, axis=0)
        backend.eval(stacked)

        n_samples = int(stacked.shape[0])
        layer_hidden_dim = int(stacked.shape[1])

        # Compute numerical rank
        rank = compute_numerical_rank(stacked, backend)

        # Compute intrinsic dimension
        id_val, id_usable = compute_intrinsic_dim(stacked, backend)

        # Compute ratios
        rank_to_id = rank / id_val if id_val > 0 else 0.0
        rank_to_hidden = rank / layer_hidden_dim if layer_hidden_dim > 0 else 0.0
        id_to_hidden = id_val / layer_hidden_dim if layer_hidden_dim > 0 else 0.0

        analysis = LayerDimensionAnalysis(
            layer_idx=layer_idx,
            hidden_dim=layer_hidden_dim,
            n_samples=n_samples,
            numerical_rank=rank,
            intrinsic_dim=id_val,
            id_usable_samples=id_usable,
            rank_to_id_ratio=rank_to_id,
            rank_to_hidden_ratio=rank_to_hidden,
            id_to_hidden_ratio=id_to_hidden,
        )
        layer_analyses.append(analysis)

        logger.info(
            "Layer %2d: rank=%3d, ID=%.1f, rank/ID=%.2f, ID/d=%.2f (compression=%.1f%%)",
            layer_idx,
            rank,
            id_val,
            rank_to_id,
            id_to_hidden,
            100 * analysis.compression_factor,
        )

    # Compute summary
    intrinsic_dims = [la.intrinsic_dim for la in layer_analyses if la.intrinsic_dim > 0]
    result = DimensionComparisonResult(
        model_path=model_path,
        total_probes=len(valid_probes),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        layer_analyses=layer_analyses,
        mean_intrinsic_dim=sum(intrinsic_dims) / len(intrinsic_dims) if intrinsic_dims else 0.0,
        min_intrinsic_dim=min(intrinsic_dims) if intrinsic_dims else 0.0,
        max_intrinsic_dim=max(intrinsic_dims) if intrinsic_dims else 0.0,
        bottleneck_layer=min(range(len(layer_analyses)), key=lambda i: layer_analyses[i].intrinsic_dim) if layer_analyses else 0,
        layers_fully_covered=sum(1 for la in layer_analyses if la.is_fully_covered),
    )

    # Print summary
    logger.info("=" * 70)
    logger.info("RANK VS INTRINSIC DIMENSION SUMMARY")
    logger.info("=" * 70)
    logger.info("Model: %s", model_path)
    logger.info("Hidden dimension: %d", hidden_dim)
    logger.info("Total probes: %d", len(valid_probes))
    logger.info("-" * 70)
    logger.info("Mean intrinsic dimension: %.1f (%.1f%% of hidden_dim)", result.mean_intrinsic_dim, 100 * result.mean_intrinsic_dim / hidden_dim)
    logger.info("Min intrinsic dimension: %.1f at layer %d (%.1f%% compression)", result.min_intrinsic_dim, result.bottleneck_layer, 100 * (1 - result.min_intrinsic_dim / hidden_dim))
    logger.info("Max intrinsic dimension: %.1f (%.1f%% of hidden_dim)", result.max_intrinsic_dim, 100 * result.max_intrinsic_dim / hidden_dim)
    logger.info("-" * 70)
    logger.info("Layers fully covered (rank ≈ ID): %d/%d (%.1f%%)", result.layers_fully_covered, n_layers, 100 * result.layers_fully_covered / n_layers)
    logger.info("=" * 70)

    # Key insight
    logger.info("")
    logger.info("KEY INSIGHT:")
    if result.min_intrinsic_dim < hidden_dim * 0.5:
        logger.info("  The model compresses to %.1f dimensions at layer %d (%.0f%% compression).",
                    result.min_intrinsic_dim, result.bottleneck_layer, 100 * (1 - result.min_intrinsic_dim / hidden_dim))
        logger.info("  Full rank (rank=hidden_dim) is IMPOSSIBLE at bottleneck layers.")
        logger.info("  The achievable goal is: rank = intrinsic_dim at every layer.")
    else:
        logger.info("  Intrinsic dimension is close to hidden_dim. Full rank may be achievable.")

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
        description="Compare numerical rank vs intrinsic dimension at each layer."
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
        default=500,
        help="Maximum number of probes to use (default: 500).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/rank_vs_id.json",
        help="Output path for JSON results.",
    )

    args = parser.parse_args()

    run_comparison(
        model_path=args.model,
        max_probes=args.max_probes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
