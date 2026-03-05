# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Activation-aware quantization frontier precheck.

This precheck compares hidden-output probe activations from a quantized model and
its full-precision reference on the same probe texts.

The operator is intentionally telemetry-rich and gate-light:
- it validates whether centered-Gram geometry is measurable on probe activations
- it does not impose a universal severity threshold
- raw Weyl crossing is preserved as nested telemetry only

`hidden_probe_d_eff` is derived from the hidden-output probe spectrum and is not
numerically comparable to the input-covariance `D_eff` reported in the
quantization deep-dive experiments.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Mapping

from modelcypher.core.domain.geometry.cka import (
    compute_gram_perturbation_ratio,
    compute_linear_cka_from_activations,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

_OPERATOR = "quantization_frontier_precheck_v1"


def _spectral_norm(matrix: Any, backend: "Backend") -> float:
    M = backend.astype(matrix, "float32")
    backend.eval(M)
    singular_values = backend.svd(M, compute_uv=False)
    backend.eval(singular_values)
    if int(singular_values.shape[0]) <= 0:
        return 0.0
    return float(backend.to_scalar(singular_values[0]))


def _center_rows(matrix: Any, backend: "Backend") -> Any:
    row_mean = backend.mean(matrix, axis=0)
    centered = matrix - row_mean
    backend.eval(centered)
    return centered


def _frobenius_norm_sq(matrix: Any, backend: "Backend") -> float:
    value = backend.sum(matrix * matrix)
    backend.eval(value)
    return float(backend.to_scalar(value))


def _descending_spectrum(centered: Any, backend: "Backend") -> tuple[Any, list[float]]:
    singular_values = backend.svd(centered, compute_uv=False)
    backend.eval(singular_values)
    singular_list = backend.tolist(singular_values)
    if isinstance(singular_list, (int, float)):
        singular_values_py = [float(singular_list)]
    else:
        singular_values_py = [float(v) for v in singular_list]
    singular_values_py.sort(reverse=True)
    return singular_values, singular_values_py


def _covariance_eigenvalues_from_singulars(
    singular_values: list[float],
    n_probes: int,
) -> list[float]:
    denom = max(1, n_probes - 1)
    eigenvalues = [(sv * sv) / float(denom) for sv in singular_values]
    return [max(ev, 0.0) for ev in eigenvalues]


def _participation_ratio(eigenvalues: list[float]) -> float:
    total = sum(eigenvalues)
    total_sq = sum(ev * ev for ev in eigenvalues)
    if total <= 0.0 or total_sq <= 0.0:
        return 0.0
    return (total * total) / total_sq


def _append_failure(failure_modes: list[str], failure_mode: str) -> None:
    if failure_mode not in failure_modes:
        failure_modes.append(failure_mode)


def run_quantization_frontier_precheck_v1(
    *,
    fp_activations: Mapping[int, Any],
    quantized_activations: Mapping[int, Any],
    n_probes: int,
    backend: "Backend",
    raw_weyl: dict[str, Any] | None = None,
    subspace_source: str = "hidden_probe_output",
) -> dict[str, Any]:
    """Reduce paired probe activations to quantization-frontier telemetry."""
    payload: dict[str, Any] = {
        "operator": _OPERATOR,
        "valid": False,
        "failure_modes": [],
        "subspace_source": subspace_source,
        "n_probes": int(n_probes),
        "n_layers": 0,
        "n_overlapping_layers": 0,
        "min_cka": None,
        "mean_cka": None,
        "per_layer_cka": {},
        "per_layer_gram_epsilon": {},
        "per_layer_cka_bound": {},
        "per_layer_hidden_probe_eigenvalues": {},
        "per_layer_hidden_probe_d_eff": {},
        "per_layer_hidden_probe_k_eff": {},
        "per_layer_hidden_probe_gap_eff": {},
        "per_layer_hidden_probe_rho_out": {},
        "raw_weyl": raw_weyl,
    }
    failure_modes: list[str] = payload["failure_modes"]

    if n_probes < 2:
        _append_failure(failure_modes, "insufficient_probes")
        return payload

    common_layers = sorted(set(fp_activations.keys()) & set(quantized_activations.keys()))
    payload["n_overlapping_layers"] = len(common_layers)
    if not common_layers:
        _append_failure(failure_modes, "no_overlapping_layers")
        return payload

    eps = float(backend.finfo().eps)
    sqrt_eps = math.sqrt(eps)

    cka_scores: dict[int, float] = {}
    gram_epsilons: dict[int, float] = {}
    cka_bounds: dict[int, float] = {}
    eigenvalues_by_layer: dict[int, list[float]] = {}
    d_eff_by_layer: dict[int, float] = {}
    k_eff_by_layer: dict[int, int] = {}
    gap_eff_by_layer: dict[int, float | None] = {}
    rho_out_by_layer: dict[int, float | None] = {}

    degenerate_layers = 0
    nonfinite_layers = 0

    for layer_idx in common_layers:
        fp_stack = backend.astype(fp_activations[layer_idx], "float32")
        q_stack = backend.astype(quantized_activations[layer_idx], "float32")
        backend.eval(fp_stack, q_stack)

        if int(fp_stack.shape[0]) != n_probes or int(q_stack.shape[0]) != n_probes:
            continue

        fp_centered = _center_rows(fp_stack, backend)
        q_centered = _center_rows(q_stack, backend)
        fp_norm_sq = _frobenius_norm_sq(fp_centered, backend)
        q_norm_sq = _frobenius_norm_sq(q_centered, backend)

        if (
            not math.isfinite(fp_norm_sq)
            or not math.isfinite(q_norm_sq)
        ):
            nonfinite_layers += 1
            continue

        if fp_norm_sq <= sqrt_eps or q_norm_sq <= sqrt_eps:
            degenerate_layers += 1
            continue

        singular_values_arr, singular_values = _descending_spectrum(fp_centered, backend)
        eigenvalues = _covariance_eigenvalues_from_singulars(singular_values, n_probes)
        d_eff = _participation_ratio(eigenvalues)
        if d_eff > 0.0:
            k_eff = max(1, int(math.ceil(d_eff)))
        else:
            k_eff = 0
        gap_eff = None
        if 1 <= k_eff < len(singular_values):
            gap_eff = max(0.0, singular_values[k_eff - 1] - singular_values[k_eff])

        delta_y = q_stack - fp_stack
        backend.eval(delta_y)
        rho_out = None
        if gap_eff is not None and gap_eff > 0.0:
            rho_out = _spectral_norm(delta_y, backend) / gap_eff

        cka = compute_linear_cka_from_activations(fp_stack, q_stack, backend)
        gram_epsilon, cka_bound = compute_gram_perturbation_ratio(
            fp_stack,
            q_stack,
            backend,
        )

        metrics_to_check = [cka, gram_epsilon, cka_bound, d_eff]
        if gap_eff is not None:
            metrics_to_check.append(gap_eff)
        if rho_out is not None:
            metrics_to_check.append(rho_out)
        if not all(math.isfinite(metric) for metric in metrics_to_check):
            nonfinite_layers += 1
            continue

        cka_scores[layer_idx] = cka
        gram_epsilons[layer_idx] = gram_epsilon
        cka_bounds[layer_idx] = cka_bound
        eigenvalues_by_layer[layer_idx] = eigenvalues
        d_eff_by_layer[layer_idx] = d_eff
        k_eff_by_layer[layer_idx] = k_eff
        gap_eff_by_layer[layer_idx] = gap_eff
        rho_out_by_layer[layer_idx] = rho_out
        payload["n_layers"] = payload["n_layers"] + 1
        backend.eval(singular_values_arr)

    if not cka_scores:
        if degenerate_layers > 0:
            _append_failure(failure_modes, "degenerate_centered_gram")
        if nonfinite_layers > 0:
            _append_failure(failure_modes, "nonfinite_metric")
        if not failure_modes:
            _append_failure(failure_modes, "no_overlapping_layers")
        return payload

    payload["valid"] = True
    payload["min_cka"] = min(cka_scores.values())
    payload["mean_cka"] = sum(cka_scores.values()) / float(len(cka_scores))
    payload["per_layer_cka"] = cka_scores
    payload["per_layer_gram_epsilon"] = gram_epsilons
    payload["per_layer_cka_bound"] = cka_bounds
    payload["per_layer_hidden_probe_eigenvalues"] = eigenvalues_by_layer
    payload["per_layer_hidden_probe_d_eff"] = d_eff_by_layer
    payload["per_layer_hidden_probe_k_eff"] = k_eff_by_layer
    payload["per_layer_hidden_probe_gap_eff"] = gap_eff_by_layer
    payload["per_layer_hidden_probe_rho_out"] = rho_out_by_layer
    return payload


__all__ = ["run_quantization_frontier_precheck_v1"]
