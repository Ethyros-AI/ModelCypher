# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""
Deviation Measurement for Model Merging.

Philosophy: If null-space projection is working correctly, the geometry
constrains deviation by construction. We don't need thresholds or safety
checks - the math handles it.

This module provides:
1. **Measurement**: Report deviation for transparency and logging
2. **Scale derivation**: Compute geometrically-correct scales from SVD
3. **No gating**: Never block operations based on thresholds

The geometry determines everything. Thresholds are vestigial.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass
class DeviationMeasurement:
    """Measurement of deviation from baseline (informational only).

    This is purely observational - not a safety check. If null-space projection
    is working correctly, the geometry constrains deviation by construction.
    We report measurements for transparency, not for gating decisions.
    """

    # Current deviation from baseline (L2 norm)
    deviation: float

    # Baseline weight norm for context
    baseline_norm: float

    # Deviation as percentage of baseline norm
    deviation_percent: float

    # Condition number of baseline (measures sensitivity)
    condition_number: float


class DeviationTracker:
    """
    Tracks and measures deviation from baseline for merging operations.

    Philosophy: If null-space projection is working correctly, the geometry
    constrains deviation by construction. We don't need thresholds or safety
    checks - the math handles it. This class provides:

    1. **Measurement**: Report deviation for transparency and logging
    2. **Scale derivation**: Compute geometrically-correct scales from data
    3. **No gating**: Never block operations based on thresholds

    The geometry determines everything.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

        # Baseline tracking
        self._baseline_weights: dict[str, Any] = {}
        self._baseline_norms: dict[str, float] = {}
        self._baseline_condition_numbers: dict[str, float] = {}

    def _compute_weight_norm(self, weights: dict[str, Any]) -> float:
        """Compute total geodesic Frobenius-like norm of all weight tensors.

        Uses geodesic norms (k-NN graph shortest paths) to properly account
        for manifold curvature in the weight space.
        """
        backend = self._backend
        total_sq = backend.array(0.0)

        for v in weights.values():
            w = backend.array(v)
            shape = backend.shape(w)

            if len(shape) >= 2:
                # Reshape to 2D if needed (flatten all but last dim)
                if len(shape) > 2:
                    n_rows = 1
                    for dim in shape[:-1]:
                        n_rows *= dim
                    w = backend.reshape(w, (n_rows, shape[-1]))
                    shape = backend.shape(w)

                # Use geodesic norms if we have enough rows for k-NN graph
                if shape[0] >= 2:
                    geo_norms_arr = geodesic_norms(w, backend, use_cache=False)
                    backend.eval(geo_norms_arr)
                    w_sq = backend.sum(geo_norms_arr * geo_norms_arr)
                else:
                    # Single row: fall back to Euclidean
                    w_sq = backend.sum(w * w)
            else:
                # 1D tensor: use Euclidean
                w_sq = backend.sum(w * w)

            total_sq = total_sq + w_sq
            backend.eval(total_sq)

        return float(backend.sqrt(total_sq))

    def _singular_values(self, matrix: Any) -> Any | None:
        backend = self._backend
        arr = backend.array(matrix)
        shape = backend.shape(arr)
        if len(shape) < 2:
            return None
        if len(shape) > 2:
            total_rows = 1
            for dim in shape[:-1]:
                total_rows *= dim
            arr = backend.reshape(arr, (total_rows, shape[-1]))
        arr = backend.astype(arr, "float32")
        backend.eval(arr)
        m = int(arr.shape[0])
        n = int(arr.shape[1])
        if m == 0 or n == 0:
            return backend.zeros((0,), dtype="float32")
        if m >= n:
            gram = backend.matmul(backend.transpose(arr), arr)
        else:
            gram = backend.matmul(arr, backend.transpose(arr))
        backend.eval(gram)
        eigenvalues, _ = backend.eigh(gram)
        backend.eval(eigenvalues)
        eigenvalues = backend.maximum(eigenvalues, backend.zeros_like(eigenvalues))
        s = backend.sqrt(eigenvalues)
        backend.eval(s)
        return s

    def _compute_condition_number(self, weights: dict[str, Any]) -> float:
        """Compute effective condition number from weight matrices via Gram spectra.

        The condition number κ = σ_max / σ_min measures matrix sensitivity to
        perturbations. For a collection of weight matrices, we compute:
            κ_eff = max(σ_max across all) / min(σ_min across all)
        """
        backend = self._backend
        eps = math.sqrt(sys.float_info.epsilon)

        sigma_max_global = 0.0
        sigma_min_global = float("inf")

        for key, v in weights.items():
            try:
                s = self._singular_values(v)
                if s is None:
                    continue
                s_max = float(backend.max(s))
                s_nonzero = backend.where(
                    s > eps,
                    s,
                    backend.full_like(s, float("inf")),
                )
                s_min = float(backend.min(s_nonzero))

                if s_max > sigma_max_global:
                    sigma_max_global = s_max
                if s_min < sigma_min_global and s_min > eps:
                    sigma_min_global = s_min

            except Exception:
                logger.debug("Spectrum solve failed for weight '%s', skipping", key)
                continue

        if sigma_max_global <= eps or sigma_min_global == float("inf"):
            logger.warning("Could not compute condition number, using fallback")
            return 1e4

        if sigma_min_global <= eps:
            sigma_min_global = eps

        return max(1.0, sigma_max_global / sigma_min_global)

    def record_baseline(self, weights: dict[str, Any], name: str = "default") -> None:
        """Record baseline weights for deviation measurement."""
        backend = self._backend
        self._baseline_weights[name] = {
            k: backend.array(v) for k, v in weights.items()
        }

        weight_norm = self._compute_weight_norm(weights)
        self._baseline_norms[name] = weight_norm

        condition_number = self._compute_condition_number(weights)
        self._baseline_condition_numbers[name] = condition_number

        logger.info(
            f"Recorded baseline '{name}': ||W||_F={weight_norm:.1f}, κ={condition_number:.1f}"
        )

    def compute_deviation(
        self,
        current_weights: dict[str, Any],
        baseline_name: str = "default",
    ) -> float:
        """Compute geodesic deviation from baseline.

        Uses geodesic norms (k-NN graph shortest paths) to properly account
        for manifold curvature in the weight delta.
        """
        if baseline_name not in self._baseline_weights:
            logger.warning(f"No baseline '{baseline_name}' recorded")
            return 0.0

        backend = self._backend
        baseline = self._baseline_weights[baseline_name]

        total_deviation_sq = backend.array(0.0)

        for key in current_weights:
            if key in baseline:
                current = backend.array(current_weights[key])
                base = baseline[key]

                delta = current - base
                shape = backend.shape(delta)

                if len(shape) >= 2:
                    # Reshape to 2D if needed
                    if len(shape) > 2:
                        n_rows = 1
                        for dim in shape[:-1]:
                            n_rows *= dim
                        delta = backend.reshape(delta, (n_rows, shape[-1]))
                        shape = backend.shape(delta)

                    # Use geodesic norms if we have enough rows
                    if shape[0] >= 2:
                        geo_norms_arr = geodesic_norms(delta, backend, use_cache=False)
                        backend.eval(geo_norms_arr)
                        deviation_sq = backend.sum(geo_norms_arr * geo_norms_arr)
                    else:
                        deviation_sq = backend.sum(delta * delta)
                else:
                    deviation_sq = backend.sum(delta * delta)

                total_deviation_sq = total_deviation_sq + deviation_sq
                backend.eval(total_deviation_sq)

        return float(backend.sqrt(total_deviation_sq))

    def measure(
        self,
        merged_weights: dict[str, Any],
        baseline_name: str = "default",
    ) -> DeviationMeasurement:
        """Measure deviation after merge (informational only)."""
        deviation = self.compute_deviation(merged_weights, baseline_name)
        baseline_norm = self._baseline_norms.get(baseline_name, 0.0)
        condition_number = self._baseline_condition_numbers.get(baseline_name, 1.0)

        deviation_percent = (deviation / baseline_norm * 100) if baseline_norm > 0 else 0.0

        return DeviationMeasurement(
            deviation=deviation,
            baseline_norm=baseline_norm,
            deviation_percent=deviation_percent,
            condition_number=condition_number,
        )

    def compute_delta_magnitude(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
    ) -> float:
        """Compute geodesic magnitude of delta between source and target.

        Uses geodesic norms (k-NN graph shortest paths) to properly account
        for manifold curvature in the weight delta.
        """
        backend = self._backend
        delta_sq = backend.array(0.0)

        for key in source_weights:
            if key in target_weights:
                src = backend.array(source_weights[key])
                tgt = backend.array(target_weights[key])
                delta = src - tgt
                shape = backend.shape(delta)

                if len(shape) >= 2:
                    # Reshape to 2D if needed
                    if len(shape) > 2:
                        n_rows = 1
                        for dim in shape[:-1]:
                            n_rows *= dim
                        delta = backend.reshape(delta, (n_rows, shape[-1]))
                        shape = backend.shape(delta)

                    # Use geodesic norms if we have enough rows
                    if shape[0] >= 2:
                        geo_norms_arr = geodesic_norms(delta, backend, use_cache=False)
                        backend.eval(geo_norms_arr)
                        delta_contrib = backend.sum(geo_norms_arr * geo_norms_arr)
                    else:
                        delta_contrib = backend.sum(delta * delta)
                else:
                    delta_contrib = backend.sum(delta * delta)

                delta_sq = delta_sq + delta_contrib
                backend.eval(delta_sq)

        return float(backend.sqrt(delta_sq))

    def derive_scale(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        target_activations: Any,
    ) -> float:
        """Derive scale from null-space capacity via Gram spectra.

        Formula: scale = null_space_capacity / delta_magnitude

        The null-space capacity is the energy in dimensions beyond the
        effective dimensionality of the target activations.
        """
        backend = self._backend

        # Compute delta magnitude
        delta_magnitude = self.compute_delta_magnitude(source_weights, target_weights)
        if delta_magnitude <= 0:
            return 1.0

        # Compute null-space capacity from target activations
        activations = backend.array(target_activations)

        # Handle different shapes
        if len(activations.shape) == 1:
            activations = backend.reshape(activations, (1, -1))
        elif len(activations.shape) > 2:
            # Flatten all but last dimension
            total_rows = 1
            for dim in activations.shape[:-1]:
                total_rows *= dim
            activations = backend.reshape(activations, (total_rows, activations.shape[-1]))

        try:
            S = self._singular_values(activations)
            if S is None:
                return 1.0
        except Exception:
            logger.warning("Spectrum solve failed for activations, using scale=1.0")
            return 1.0

        # Effective dimensionality: d_eff = (Σσ)² / Σσ²
        eps = math.sqrt(sys.float_info.epsilon)
        sum_s = float(backend.sum(S))
        sum_s_sq = float(backend.sum(S * S))

        if sum_s_sq < eps:
            return 1.0

        d_eff = (sum_s * sum_s) / sum_s_sq
        d_eff_int = int(math.ceil(d_eff))

        # Null-space capacity: energy in dimensions beyond d_eff
        n_dims = int(S.shape[0])
        if d_eff_int >= n_dims:
            # All dimensions are "occupied" - use small scale
            null_capacity = float(backend.min(S))
        else:
            # Sum of singular values in null dimensions
            null_s = S[d_eff_int:]
            null_capacity = float(backend.sum(null_s))

        if null_capacity < eps:
            null_capacity = eps

        # Scale = capacity / magnitude
        scale = null_capacity / delta_magnitude

        # Clamp to reasonable range
        return min(1.0, max(0.01, scale))


# Backward compatibility alias
DeviationBudget = DeviationTracker
