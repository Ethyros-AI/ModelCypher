# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Geometry table emitter for LKM validation protocol.

Wraps analyze_weight_geometries() into a JSON emitter matching the LKM
manifest schema. Produces a geometry_table.json with per-layer spectral
geometry and summary statistics.

Usage:
    poetry run python scripts/lkm/emit_geometry_table.py \\
        --model /path/to/model \\
        --output-dir results/lora_memory_capacity_validation/model-id/
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_geometry_table(
    model_id: str,
    model_family: str,
    dtype: str,
    geometries: dict[str, Any],
) -> dict[str, Any]:
    """Build geometry table dict matching the LKM manifest schema.

    Pure Python function -- no backend dependency. Takes geometry objects
    (anything with the required attributes) and formats them into the
    schema expected by the LKM validation protocol.

    Args:
        model_id: Identifier for the model (e.g. "LFM2-350M-bf16").
        model_family: Model family name (e.g. "LFM2", "Qwen3.5").
        dtype: Data type string (e.g. "bf16", "float32").
        geometries: Dict mapping layer_key -> object with attributes:
            shape, full_rank, effective_rank, tail_dims, sigma_max,
            sigma_k, spectral_gap, shannon_effective_rank.

    Returns:
        Dict matching the LKM geometry_table.json schema.
    """
    layers: list[dict[str, Any]] = []

    for layer_key, geom in geometries.items():
        sigma_max = float(geom.sigma_max)
        sigma_k = float(geom.sigma_k)

        if sigma_k == 0.0:
            condition_number = math.inf
        else:
            condition_number = sigma_max / sigma_k

        layers.append({
            "layer_key": layer_key,
            "shape": list(geom.shape),
            "full_rank": int(geom.full_rank),
            "effective_rank": int(geom.effective_rank),
            "shannon_effective_rank": float(geom.shannon_effective_rank),
            "tail_dims": int(geom.tail_dims),
            "sigma_max": sigma_max,
            "sigma_k": sigma_k,
            "spectral_gap": float(geom.spectral_gap),
            "condition_number": condition_number,
        })

    total_layers = len(layers)
    total_tail_dims = sum(layer["tail_dims"] for layer in layers)
    layers_with_capacity = sum(1 for layer in layers if layer["tail_dims"] > 0)

    return {
        "model_id": model_id,
        "model_family": model_family,
        "dtype": dtype,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "layers": layers,
        "summary": {
            "total_layers": total_layers,
            "total_tail_dims": total_tail_dims,
            "mean_tail_dims": total_tail_dims / total_layers if total_layers > 0 else 0.0,
            "layers_with_capacity": layers_with_capacity,
            "layers_without_capacity": total_layers - layers_with_capacity,
        },
    }


def _flatten_params(params) -> dict:
    """Flatten model parameters to flat key -> 2D array mapping.

    Uses mlx.utils.tree_flatten to handle arbitrarily nested parameter dicts.
    Only includes 2D arrays (weight matrices).
    """
    import mlx.utils

    flat = {}
    for key, value in mlx.utils.tree_flatten(params):
        if hasattr(value, "shape") and value.ndim == 2:
            flat[key] = value
    return flat


def main() -> None:
    """CLI entry point for geometry table emission."""
    parser = argparse.ArgumentParser(
        description="Emit geometry table JSON for LKM validation protocol."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory (default: "
            "results/lora_memory_capacity_validation/<model_id>/)."
        ),
    )

    args = parser.parse_args()
    model_path = Path(args.model)
    model_id = model_path.name

    # Determine output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/lora_memory_capacity_validation") / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from mlx_lm import load as mlx_load
    print(f"Loading model from {model_path}...")
    model, _ = mlx_load(str(model_path))

    # Flatten parameters to get 2D weight matrices
    params = dict(model.parameters())
    flat_weights = _flatten_params(params)
    print(f"Found {len(flat_weights)} weight matrices.")

    # Analyze geometries
    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.training.geometric_lora import (
        analyze_weight_geometries,
    )

    backend = MLXBackend()
    print("Analyzing weight geometries (this may take a while)...")
    geometries = analyze_weight_geometries(flat_weights, backend)

    # Detect model family from model_id
    model_id_lower = model_id.lower()
    if "lfm" in model_id_lower:
        model_family = "LFM2"
    elif "qwen" in model_id_lower:
        model_family = "Qwen"
    elif "deepseek" in model_id_lower:
        model_family = "DeepSeek"
    else:
        model_family = "unknown"

    # Detect dtype from model_id
    if "bf16" in model_id_lower:
        dtype = "bf16"
    elif "fp16" in model_id_lower or "float16" in model_id_lower:
        dtype = "fp16"
    elif "fp32" in model_id_lower or "float32" in model_id_lower:
        dtype = "float32"
    else:
        dtype = "unknown"

    # Build table
    table = build_geometry_table(
        model_id=model_id,
        model_family=model_family,
        dtype=dtype,
        geometries=geometries,
    )

    # Write JSON
    output_path = output_dir / "geometry_table.json"
    with open(output_path, "w") as f:
        json.dump(table, f, indent=2, default=_json_default)
    print(f"Wrote geometry table ({table['summary']['total_layers']} layers) to {output_path}")


def _json_default(obj: Any) -> Any:
    """Handle non-serializable values in JSON output."""
    if isinstance(obj, float) and math.isinf(obj):
        return "Infinity"
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
