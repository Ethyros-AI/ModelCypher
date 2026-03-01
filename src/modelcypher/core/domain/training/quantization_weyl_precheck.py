# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Quantization Weyl precheck for FP-vs-quantized weight pairs.

For each matching layer:
- measure quantization perturbation ||E_q||_2 where E_q = W_fp - W_q
- derive structural spectral gap from full-precision weights
- evaluate Weyl crossing condition: ||E_q||_2 < gap/2

Weyl (1912): singular-value order is preserved under perturbation E when
||E||_2 is below half the adjacent singular-value gap at the structural
boundary.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.training.geometric_lora import compute_layer_geometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _spectral_norm(matrix: Any, backend: "Backend") -> float:
    """Exact spectral norm via top singular value."""
    M = backend.astype(matrix, "float32")
    backend.eval(M)
    singular_values = backend.svd(M, compute_uv=False)
    backend.eval(singular_values)
    if int(singular_values.shape[0]) <= 0:
        return 0.0
    return float(backend.to_scalar(singular_values[0]))


def run_quantization_weyl_precheck(
    *,
    fp_weights: dict[str, Any],
    quantized_weights: dict[str, Any],
    backend: "Backend",
) -> dict[str, Any]:
    """Measure Weyl crossing risk for quantization perturbations.

    Returns a JSON-serializable payload with per-layer diagnostics and
    aggregate crossing status.
    """
    common_layers = sorted(set(fp_weights.keys()) & set(quantized_weights.keys()))
    per_layer: list[dict[str, Any]] = []
    n_crossing = 0
    max_ratio = 0.0

    for layer_key in common_layers:
        fp_weight = fp_weights[layer_key]
        q_weight = quantized_weights[layer_key]

        geometry = compute_layer_geometry(fp_weight, layer_key, backend)
        spectral_gap = float(geometry.spectral_gap)
        gap_half = spectral_gap / 2.0

        error = backend.astype(fp_weight, "float32") - backend.astype(q_weight, "float32")
        backend.eval(error)
        error_norm = _spectral_norm(error, backend)

        if gap_half > 0.0:
            error_over_gap_half = error_norm / gap_half
            crossing = error_norm >= gap_half
        else:
            error_over_gap_half = float("inf") if error_norm > 0.0 else 0.0
            crossing = error_norm > 0.0

        if crossing:
            n_crossing += 1
        if math.isfinite(error_over_gap_half):
            max_ratio = max(max_ratio, error_over_gap_half)

        per_layer.append(
            {
                "layer_key": layer_key,
                "shape": [int(geometry.shape[0]), int(geometry.shape[1])],
                "error_norm_2": error_norm,
                "spectral_gap": spectral_gap,
                "gap_half": gap_half,
                "error_over_gap_half": error_over_gap_half,
                "crossing": crossing,
            }
        )

    n_layers = len(common_layers)
    non_crossing_layers = n_layers - n_crossing
    all_non_crossing = (n_layers > 0) and (n_crossing == 0)

    return {
        "n_layers": n_layers,
        "n_crossing": n_crossing,
        "n_non_crossing": non_crossing_layers,
        "all_non_crossing": all_non_crossing,
        "max_error_over_gap_half": max_ratio,
        "crossing_layers": [row["layer_key"] for row in per_layer if row["crossing"]],
        "per_layer": per_layer,
    }


__all__ = ["run_quantization_weyl_precheck"]
