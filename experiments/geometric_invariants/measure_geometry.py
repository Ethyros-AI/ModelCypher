#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# AGPL-3.0-or-later
"""
Measure geometric properties of LLM representation manifolds.

This script measures:
1. Intrinsic dimension (TwoNN estimator)
2. Sectional curvature (geodesic-based)
3. Cross-model CKA alignment

Usage:
    poetry run python experiments/geometric_invariants/measure_geometry.py \
        --models /path/to/model1 /path/to/model2 \
        --output results/experiment_001.json
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
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelGeometry:
    """Geometric measurements for a single model."""

    model_path: str
    model_name: str
    hidden_dim: int
    num_layers: int
    layer_measured: int

    # Intrinsic dimension
    intrinsic_dimension: float
    intrinsic_dimension_ci_lower: float | None
    intrinsic_dimension_ci_upper: float | None

    # Curvature
    mean_sectional_curvature: float
    std_sectional_curvature: float
    curvature_sign: str  # positive, negative, mixed, flat

    # Probe statistics
    n_probes: int
    probe_source: str

    # Timing
    measurement_time_seconds: float


@dataclass
class PairwiseAlignment:
    """CKA alignment between two models."""

    model_a: str
    model_b: str

    # Linear CKA (Euclidean Gram)
    linear_cka_raw: float
    linear_cka_aligned: float

    # Geodesic CKA (RBF on geodesic distances)
    geodesic_cka_raw: float
    geodesic_cka_aligned: float

    # Alignment metadata
    n_samples: int
    rank_bound: int
    is_full_rank: bool

    # Timing
    alignment_time_seconds: float


@dataclass
class ExperimentResult:
    """Complete experiment result."""

    experiment_id: str
    timestamp: str
    hypothesis: str

    # Per-model measurements
    model_geometries: list[dict[str, Any]]

    # Pairwise alignments
    pairwise_alignments: list[dict[str, Any]]

    # Summary statistics
    summary: dict[str, Any]


def get_model_paths() -> dict[str, Path]:
    """Get paths to available models.

    Uses MODELCYPHER_MODEL_PATH environment variable if set,
    otherwise falls back to HuggingFace cache.
    """
    import os
    models = {}

    # Check environment variable first, then HuggingFace cache
    model_base = os.environ.get("MODELCYPHER_MODEL_PATH")
    if model_base:
        mlx_base = Path(model_base)
    else:
        mlx_base = Path.home() / ".cache/huggingface/hub"

    if mlx_base.exists():
        for model_dir in mlx_base.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                # Skip adapters
                if "adapter" in model_dir.name.lower():
                    continue
                models[model_dir.name] = model_dir

    return models


def measure_model_geometry(
    model_path: Path,
    backend,
    n_probes: int = 200,
) -> ModelGeometry:
    """Measure geometric properties of a single model."""
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.cli.commands.geometry.helpers import (
        extract_anchor_activations,
        resolve_model_backbone,
    )
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
    from modelcypher.core.domain.geometry.manifold_curvature import SectionalCurvatureEstimator

    start_time = time.perf_counter()
    logger.info(f"Loading model: {model_path}")

    model, tokenizer = load_model_for_training(str(model_path))

    # Get model architecture
    embed_tokens, layers, norm = resolve_model_backbone(
        model, getattr(model, "model_type", None)
    )

    if embed_tokens is None:
        raise ValueError(f"Could not resolve backbone for {model_path}")

    hidden_dim = int(embed_tokens.weight.shape[-1])
    num_layers = len(layers)
    layer_idx = num_layers // 2  # Middle layer

    logger.info(f"Model: hidden_dim={hidden_dim}, layers={num_layers}, measuring layer {layer_idx}")

    # Get probe words from semantic atlas
    all_probes = UnifiedAtlasInventory.all_probes()
    probe_words = [p.name for p in all_probes[:n_probes * 2]]  # Get extras for filtering

    # Collect activations
    logger.info(f"Collecting activations for {len(probe_words)} probe candidates...")
    activations_map = extract_anchor_activations(
        probe_words,
        tokenizer,
        embed_tokens,
        layers,
        norm,
        layer_idx,
        backend,
    )

    # Stack activations
    valid_words = [w for w in probe_words if w in activations_map][:n_probes]
    # TwoNN ID estimation requires at least k+1=3 samples for k=2 nearest neighbors
    min_samples_for_twonn = 3
    if len(valid_words) < min_samples_for_twonn:
        raise ValueError(f"Only {len(valid_words)} valid probes, need at least {min_samples_for_twonn}")

    activations = backend.stack([activations_map[w] for w in valid_words], axis=0)
    activations = backend.astype(activations, "float32")
    backend.eval(activations)

    logger.info(f"Collected {len(valid_words)} activations, shape: {activations.shape}")

    # Measure intrinsic dimension
    logger.info("Computing intrinsic dimension...")
    id_estimator = IntrinsicDimension(backend)
    id_result = id_estimator.compute(activations, with_ci=True)

    # Measure sectional curvature
    mean_curv = 0.0
    std_curv = 0.0
    curv_sign = "unknown"
    try:
        logger.info("Computing sectional curvature...")
        curv_estimator = SectionalCurvatureEstimator()
        curv_result = curv_estimator.estimate_manifold_profile(activations)
        mean_curv = curv_result.global_mean
        std_curv = curv_result.global_variance ** 0.5
        curv_sign = curv_result.dominant_sign.value
    except Exception as e:
        logger.warning(f"Curvature estimation failed: {e}")

    elapsed = time.perf_counter() - start_time

    return ModelGeometry(
        model_path=str(model_path),
        model_name=model_path.name,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        layer_measured=layer_idx,
        intrinsic_dimension=id_result.intrinsic_dimension,
        intrinsic_dimension_ci_lower=id_result.ci.lower if id_result.ci else None,
        intrinsic_dimension_ci_upper=id_result.ci.upper if id_result.ci else None,
        mean_sectional_curvature=mean_curv,
        std_sectional_curvature=std_curv,
        curvature_sign=curv_sign,
        n_probes=len(valid_words),
        probe_source="UnifiedAtlasInventory",
        measurement_time_seconds=elapsed,
    )


def measure_pairwise_alignment(
    model_a_path: Path,
    model_b_path: Path,
    backend,
    n_samples: int = 500,
) -> PairwiseAlignment:
    """Measure CKA alignment between two models."""
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.cli.commands.geometry.helpers import (
        extract_anchor_activations,
        resolve_model_backbone,
    )
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    from modelcypher.core.domain.geometry.cka import compute_cka, compute_linear_cka
    from modelcypher.core.domain.geometry.gram_aligner import find_alignment

    start_time = time.perf_counter()
    logger.info(f"Measuring alignment: {model_a_path.name} <-> {model_b_path.name}")

    # Load both models
    model_a, tok_a = load_model_for_training(str(model_a_path))
    model_b, tok_b = load_model_for_training(str(model_b_path))

    embed_a, layers_a, norm_a = resolve_model_backbone(model_a, getattr(model_a, "model_type", None))
    embed_b, layers_b, norm_b = resolve_model_backbone(model_b, getattr(model_b, "model_type", None))

    if embed_a is None or embed_b is None:
        raise ValueError("Could not resolve model backbones")

    d_a = int(embed_a.weight.shape[-1])
    d_b = int(embed_b.weight.shape[-1])
    layer_a = len(layers_a) // 2
    layer_b = len(layers_b) // 2

    # Get probe words
    all_probes = UnifiedAtlasInventory.all_probes()
    probe_words = [p.name for p in all_probes[:n_samples * 2]]

    # Collect activations from both models
    acts_a = extract_anchor_activations(probe_words, tok_a, embed_a, layers_a, norm_a, layer_a, backend)
    acts_b = extract_anchor_activations(probe_words, tok_b, embed_b, layers_b, norm_b, layer_b, backend)

    # Find common words
    common = [w for w in probe_words if w in acts_a and w in acts_b][:n_samples]
    if len(common) < 50:
        raise ValueError(f"Only {len(common)} common words")

    source = backend.stack([acts_a[w] for w in common], axis=0)
    target = backend.stack([acts_b[w] for w in common], axis=0)
    source = backend.astype(source, "float32")
    target = backend.astype(target, "float32")
    backend.eval(source, target)

    n = len(common)
    d_min = min(d_a, d_b)
    rank_bound = min(n, d_a, d_b)
    is_full_rank = rank_bound >= d_min

    # Raw CKA (before alignment)
    linear_raw = compute_linear_cka(source, target, backend)
    geo_raw = compute_cka(source, target, backend)

    # Find alignment
    alignment = find_alignment(source, target, backend)
    aligned_source = backend.matmul(source, alignment.feature_transform)
    backend.eval(aligned_source)

    # Aligned CKA
    linear_aligned = compute_linear_cka(aligned_source, target, backend)
    geo_aligned = compute_cka(aligned_source, target, backend)

    elapsed = time.perf_counter() - start_time

    return PairwiseAlignment(
        model_a=model_a_path.name,
        model_b=model_b_path.name,
        linear_cka_raw=linear_raw,
        linear_cka_aligned=linear_aligned,
        geodesic_cka_raw=geo_raw.cka if geo_raw.is_valid else 0.0,
        geodesic_cka_aligned=geo_aligned.cka if geo_aligned.is_valid else 0.0,
        n_samples=n,
        rank_bound=rank_bound,
        is_full_rank=is_full_rank,
        alignment_time_seconds=elapsed,
    )


def main():
    parser = argparse.ArgumentParser(description="Measure LLM geometric invariants")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="Model paths or names to measure. If empty, uses all available.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiment.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=200,
        help="Number of probe words for geometry measurement",
    )
    parser.add_argument(
        "--skip-pairwise",
        action="store_true",
        help="Skip pairwise CKA measurements",
    )
    args = parser.parse_args()

    # Initialize backend
    from modelcypher.backends import get_backend
    from modelcypher.core.domain._backend import set_default_backend

    backend = get_backend("mlx")
    set_default_backend(backend)

    # Get model paths
    available_models = get_model_paths()
    logger.info(f"Found {len(available_models)} available models")

    if args.models:
        # Use specified models
        model_paths = []
        for m in args.models:
            if m in available_models:
                model_paths.append(available_models[m])
            elif Path(m).exists():
                model_paths.append(Path(m))
            else:
                logger.warning(f"Model not found: {m}")
    else:
        # Use small models for quick test
        small_models = ["LFM2-350M-MLX-bf16", "Qwen2.5-Coder-0.5B-Instruct-bf16"]
        model_paths = [available_models[m] for m in small_models if m in available_models]

    if not model_paths:
        logger.error("No models found!")
        sys.exit(1)

    logger.info(f"Measuring {len(model_paths)} models: {[p.name for p in model_paths]}")

    # Measure each model
    geometries = []
    for model_path in model_paths:
        try:
            geom = measure_model_geometry(model_path, backend, n_probes=args.n_probes)
            geometries.append(asdict(geom))
            logger.info(
                f"  {geom.model_name}: ID={geom.intrinsic_dimension:.1f}, "
                f"curvature={geom.mean_sectional_curvature:.4f} ({geom.curvature_sign})"
            )
        except Exception as e:
            logger.error(f"Failed to measure {model_path}: {e}")

    # Pairwise alignments
    alignments = []
    if not args.skip_pairwise and len(model_paths) >= 2:
        for i, model_a in enumerate(model_paths):
            for model_b in model_paths[i + 1:]:
                try:
                    align = measure_pairwise_alignment(model_a, model_b, backend)
                    alignments.append(asdict(align))
                    logger.info(
                        f"  {align.model_a} <-> {align.model_b}: "
                        f"Linear CKA={align.linear_cka_aligned:.4f}, "
                        f"Geodesic CKA={align.geodesic_cka_aligned:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Failed alignment {model_a.name} <-> {model_b.name}: {e}")

    # Compute summary
    if geometries:
        ids = [g["intrinsic_dimension"] for g in geometries]
        curvs = [g["mean_sectional_curvature"] for g in geometries]
        summary = {
            "n_models": len(geometries),
            "intrinsic_dimension_mean": sum(ids) / len(ids),
            "intrinsic_dimension_std": (sum((x - sum(ids)/len(ids))**2 for x in ids) / len(ids)) ** 0.5,
            "curvature_mean": sum(curvs) / len(curvs),
        }
        if alignments:
            ckas = [a["geodesic_cka_aligned"] for a in alignments]
            summary["mean_geodesic_cka"] = sum(ckas) / len(ckas)
    else:
        summary = {}

    # Build result
    result = ExperimentResult(
        experiment_id=f"geom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        hypothesis="All LLMs encode the same invariant geometric shape",
        model_geometries=geometries,
        pairwise_alignments=alignments,
        summary=summary,
    )

    # Save results
    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Log summary (raw measurements only)
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for g in geometries:
        logger.info(f"model: {g['model_name']}")
        logger.info(f"  hidden_dim: {g['hidden_dim']}")
        logger.info(f"  num_layers: {g['num_layers']}")
        logger.info(f"  intrinsic_dimension: {g['intrinsic_dimension']:.4f}")
        logger.info(f"  mean_sectional_curvature: {g['mean_sectional_curvature']:.6f}")
        logger.info(f"  curvature_sign: {g['curvature_sign']}")

    if alignments:
        logger.info("PAIRWISE ALIGNMENTS:")
        for a in alignments:
            logger.info(f"  {a['model_a']} <-> {a['model_b']}:")
            logger.info(f"    linear_cka_aligned: {a['linear_cka_aligned']:.6f}")
            logger.info(f"    geodesic_cka_aligned: {a['geodesic_cka_aligned']:.6f}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
