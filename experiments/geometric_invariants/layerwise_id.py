#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# AGPL-3.0-or-later
"""Measure intrinsic dimension across network layers.

This script measures ID at multiple depths (0%, 25%, 50%, 75%, 100%)
to understand how the representation manifold changes through the network.

Hypothesis: ID profile reveals compression pattern:
- Early layers: Higher ID (feature extraction)
- Middle layers: Peak ID (representation)
- Late layers: Lower ID (compression for output)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class LayerIDResult:
    """Intrinsic dimension at a specific layer."""

    layer_idx: int
    layer_pct: float  # Position as percentage (0-100)
    intrinsic_dimension: float
    ci_lower: float | None
    ci_upper: float | None


@dataclass
class LayerwiseIDProfile:
    """Complete layer-wise ID profile for a model."""

    model_path: str
    model_name: str
    hidden_dim: int
    num_layers: int
    n_probes: int

    # Layer-wise measurements
    layer_results: list[LayerIDResult]

    # Summary statistics
    min_id: float
    max_id: float
    mean_id: float
    peak_layer_pct: float  # Layer with highest ID

    elapsed_seconds: float


def measure_layerwise_id(
    model_path: Path,
    n_probes: int = 300,
    layer_pcts: list[float] | None = None,
) -> LayerwiseIDProfile:
    """Measure intrinsic dimension at multiple layers."""
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends import get_backend
    from modelcypher.cli.commands.geometry.helpers import (
        extract_anchor_activations,
        resolve_model_backbone,
    )
    from modelcypher.core.domain._backend import set_default_backend
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    if layer_pcts is None:
        layer_pcts = [0, 25, 50, 75, 100]

    start_time = time.perf_counter()

    # Initialize backend
    backend = get_backend("mlx")
    set_default_backend(backend)

    # Load model
    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load_model_for_training(str(model_path))

    # Resolve backbone
    embed_tokens, layers, norm = resolve_model_backbone(
        model, getattr(model, "model_type", None)
    )

    if embed_tokens is None:
        raise ValueError(f"Could not resolve backbone for {model_path}")

    hidden_dim = int(embed_tokens.weight.shape[-1])
    num_layers = len(layers)

    logger.info(f"Model: hidden_dim={hidden_dim}, layers={num_layers}")

    # Get probe words
    all_probes = UnifiedAtlasInventory.all_probes()
    probe_words = [p.name for p in all_probes[:n_probes * 2]]

    # Measure ID at each layer percentage
    layer_results = []
    id_estimator = IntrinsicDimension(backend)

    for pct in layer_pcts:
        # Convert percentage to layer index
        layer_idx = int((pct / 100) * (num_layers - 1))
        layer_idx = max(0, min(layer_idx, num_layers - 1))

        logger.info(f"Measuring layer {layer_idx} ({pct}%)...")

        # Collect activations at this layer
        activations_map = extract_anchor_activations(
            probe_words,
            tokenizer,
            embed_tokens,
            layers,
            norm,
            layer_idx,
            backend,
        )

        valid_words = [w for w in probe_words if w in activations_map][:n_probes]
        # TwoNN ID estimation requires at least 3 samples for k=2 nearest neighbors
        # Using k+1=3 as the mathematical minimum
        min_samples_for_twonn = 3
        if len(valid_words) < min_samples_for_twonn:
            logger.warning(f"Only {len(valid_words)} valid probes at layer {layer_idx} (need >= {min_samples_for_twonn})")
            continue

        activations = backend.stack([activations_map[w] for w in valid_words], axis=0)
        activations = backend.astype(activations, "float32")
        backend.eval(activations)

        # Compute ID
        id_result = id_estimator.compute(activations, with_ci=True)

        layer_results.append(LayerIDResult(
            layer_idx=layer_idx,
            layer_pct=pct,
            intrinsic_dimension=id_result.intrinsic_dimension,
            ci_lower=id_result.ci.lower if id_result.ci else None,
            ci_upper=id_result.ci.upper if id_result.ci else None,
        ))

        logger.info(f"  Layer {layer_idx} ({pct}%): ID = {id_result.intrinsic_dimension:.2f}")

    # Compute summary statistics
    ids = [r.intrinsic_dimension for r in layer_results]
    min_id = min(ids)
    max_id = max(ids)
    mean_id = sum(ids) / len(ids)
    peak_idx = ids.index(max_id)
    peak_layer_pct = layer_results[peak_idx].layer_pct

    elapsed = time.perf_counter() - start_time

    return LayerwiseIDProfile(
        model_path=str(model_path),
        model_name=model_path.name,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_probes=len(valid_words),
        layer_results=layer_results,
        min_id=min_id,
        max_id=max_id,
        mean_id=mean_id,
        peak_layer_pct=peak_layer_pct,
        elapsed_seconds=elapsed,
    )


def main():
    parser = argparse.ArgumentParser(description="Measure layer-wise intrinsic dimension")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or name",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=300,
        help="Number of probe words",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/layerwise_id.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    import os
    model_path = Path(args.model)
    if not model_path.exists():
        # Check environment variable, then HuggingFace cache
        model_base = os.environ.get(
            "MODELCYPHER_MODEL_PATH",
            str(Path.home() / ".cache/huggingface/hub")
        )
        model_path = Path(model_base) / args.model

    profile = measure_layerwise_id(model_path, n_probes=args.n_probes)

    # Save results
    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "experiment": "layerwise_intrinsic_dimension",
        "timestamp": datetime.now().isoformat(),
        "profile": asdict(profile),
    }

    # Convert dataclass list to dict list
    output_data["profile"]["layer_results"] = [asdict(r) for r in profile.layer_results]

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Log summary (raw measurements only)
    logger.info("=" * 60)
    logger.info("LAYER-WISE INTRINSIC DIMENSION PROFILE")
    logger.info("=" * 60)
    logger.info(f"model: {profile.model_name}")
    logger.info(f"hidden_dim: {profile.hidden_dim}")
    logger.info(f"num_layers: {profile.num_layers}")
    logger.info(f"n_probes: {profile.n_probes}")
    for r in profile.layer_results:
        ci_str = ""
        if r.ci_lower is not None and r.ci_upper is not None:
            ci_str = f" ci=[{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
        logger.info(f"layer_{r.layer_idx} ({r.layer_pct:.1f}%): id={r.intrinsic_dimension:.4f}{ci_str}")
    logger.info(f"min_id: {profile.min_id:.4f}")
    logger.info(f"max_id: {profile.max_id:.4f}")
    logger.info(f"mean_id: {profile.mean_id:.4f}")
    logger.info(f"peak_layer_pct: {profile.peak_layer_pct:.1f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
