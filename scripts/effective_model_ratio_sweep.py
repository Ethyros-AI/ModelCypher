#!/usr/bin/env python3
"""Effective Model Ratio sweep across model scales.

Measures sum(tail_dims) / sum(full_rank) for each model — the fraction
of each model's spectral capacity that is null space.

Hypothesis: EMR increases with model size.  If true, larger models have
proportionally MORE unused capacity, supporting the thesis that we don't
need more parameters, we need to use the activation space we have.

Cross-architecture comparison is built in: LFM2 (hybrid attention/convolution)
vs Qwen3 (standard transformer).

Usage:
    poetry run python scripts/effective_model_ratio_sweep.py
    poetry run python scripts/effective_model_ratio_sweep.py --models /path/to/m1 /path/to/m2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.backends import initialize_default_backend
from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter

DEFAULT_MODELS = [
    "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
    "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
]

OUTPUT_DIR = Path("results/effective_model_ratio")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("emr_sweep")


def _clear_gpu_cache() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _analyze_model(
    model_path: str,
    backend: Any,
    adapter: MLXTrainingAdapter,
) -> dict[str, Any]:
    """Load model, run streaming geometry, return results, release model."""
    logger.info("Loading %s", model_path)
    model, tokenizer = backend.load_model(model_path)

    logger.info("Running streaming geometry analysis (randomized SVD)...")
    geometries = adapter.analyze_model_geometry_streaming(
        model,
        use_randomized=True,
        randomized_kwargs={"seed": 42},
    )

    # Aggregate metrics
    total_tail = 0
    total_structural = 0
    total_full_rank = 0
    n_targetable = 0
    per_layer: dict[str, dict[str, Any]] = {}

    for key, geom in geometries.items():
        total_tail += geom.tail_dims
        total_structural += int(geom.shannon_effective_rank)
        total_full_rank += geom.full_rank
        if geom.is_targetable:
            n_targetable += 1

        per_layer[key] = {
            "shape": list(geom.shape),
            "full_rank": geom.full_rank,
            "tail_dims": geom.tail_dims,
            "shannon_effective_rank": geom.shannon_effective_rank,
            "sigma_max": geom.sigma_max,
            "sigma_k": geom.sigma_k,
            "spectral_gap": geom.spectral_gap,
            "decay_ratio": geom.decay_ratio,
            "is_targetable": geom.is_targetable,
        }

    emr = total_tail / total_full_rank if total_full_rank > 0 else 0.0

    result = {
        "model_path": model_path,
        "model_id": Path(model_path).name,
        "n_matrices": len(geometries),
        "n_targetable": n_targetable,
        "total_tail_dims": total_tail,
        "total_structural_rank": total_structural,
        "total_full_rank": total_full_rank,
        "effective_model_ratio": emr,
        "per_layer": per_layer,
    }

    logger.info(
        "%s: EMR=%.4f, tail=%d, structural=%d, full_rank=%d, targetable=%d/%d",
        Path(model_path).name,
        emr,
        total_tail,
        total_structural,
        total_full_rank,
        n_targetable,
        len(geometries),
    )

    # Release model
    del model, tokenizer
    gc.collect()
    _clear_gpu_cache()

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Effective Model Ratio sweep across model scales.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model paths to analyze (default: LFM2 350M/700M/1.2B + Qwen3-8B).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Validate model paths
    for model_path in args.models:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

    backend = initialize_default_backend()
    adapter = MLXTrainingAdapter(backend)

    results: list[dict[str, Any]] = []

    for model_path in args.models:
        result = _analyze_model(model_path, backend, adapter)
        results.append(result)

    # Write output
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "n_models": len(results),
        "models": results,
        "summary": [
            {
                "model_id": r["model_id"],
                "effective_model_ratio": r["effective_model_ratio"],
                "total_tail_dims": r["total_tail_dims"],
                "total_full_rank": r["total_full_rank"],
                "n_targetable": r["n_targetable"],
                "n_matrices": r["n_matrices"],
            }
            for r in results
        ],
    }

    output_path = output_dir / "effective_model_ratio_sweep.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results written to %s", output_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("EFFECTIVE MODEL RATIO SWEEP")
    print("=" * 80)
    print(
        f"{'Model':<40} {'EMR':>8} {'Tail':>8} {'Full':>8} "
        f"{'Target':>8} {'Total':>8}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['model_id']:<40} {r['effective_model_ratio']:>8.4f} "
            f"{r['total_tail_dims']:>8d} {r['total_full_rank']:>8d} "
            f"{r['n_targetable']:>8d} {r['n_matrices']:>8d}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
