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
Gradient Smoothness Estimator.

Computes per-layer gradient smoothness metrics from per-sample gradients.
Smoothness is defined using gradient SNR and variance.

Ported 1:1 from the reference Swift implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    precision_dtype,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class LayerGradientQuality:
    variance: float
    snr: float
    mean_norm: float
    sample_count: int


class GradientSmoothnessEstimator:
    """Computes per-layer gradient quality metrics."""

    @staticmethod
    def gradient_quality(
        per_sample_gradients: "list[dict[str, Array]]",
        backend: "Backend | None" = None,
    ) -> LayerGradientQuality | None:
        """Compute gradient quality metrics for one parameter group.

        This is the direct helper for estimating gradient noise scale:
            B_crit = Var(g) / ||E[g]||^2 = 1 / SNR
        where SNR is returned in ``LayerGradientQuality.snr``.
        """
        return GradientSmoothnessEstimator._compute_gradient_quality(
            per_sample_gradients=per_sample_gradients,
            backend=backend,
        )

    @staticmethod
    def per_layer_quality(
        per_sample_gradients: "list[dict[str, Array]]",
        backend: "Backend | None" = None,
    ) -> dict[int, LayerGradientQuality]:
        """
        Computes per-layer gradient quality by grouping parameter gradients per transformer layer.
        per_sample_gradients: List of dictionaries (one per sample), where keys are param names and values are gradients.
        """
        if len(per_sample_gradients) <= 1:
            return {}

        # Group gradients by layer
        per_layer_samples: "dict[int, list[dict[str, Array]]]" = {}

        for sample_grads in per_sample_gradients:
            layer_bucket: "dict[int, dict[str, Array]]" = {}

            for key, grad in sample_grads.items():
                layer_index = GradientSmoothnessEstimator._extract_layer_index_from_key(key)
                if layer_index is not None:
                    if layer_index not in layer_bucket:
                        layer_bucket[layer_index] = {}
                    layer_bucket[layer_index][key] = grad

            for layer, grads in layer_bucket.items():
                if layer not in per_layer_samples:
                    per_layer_samples[layer] = []
                per_layer_samples[layer].append(grads)

        # Compute quality for each layer
        results: dict[int, LayerGradientQuality] = {}
        for layer, samples in per_layer_samples.items():
            quality = GradientSmoothnessEstimator._compute_gradient_quality(samples, backend)
            if quality:
                results[layer] = quality

        return results

    @staticmethod
    def _compute_gradient_quality(
        per_sample_gradients: "list[dict[str, Array]]",
        backend: "Backend | None" = None,
    ) -> LayerGradientQuality | None:
        """
        Computes gradient quality metrics for a set of samples (implicitly representing one layer or group).
        This logic mirrors HessianEstimator.gradientQuality in Swift.
        """
        if len(per_sample_gradients) < 2:
            return None

        b = backend or get_default_backend()
        eps = machine_epsilon(b, b.array([1.0], dtype=precision_dtype(b)))

        # Flatten all gradients for each sample into a single vector (conceptually)
        # or compute norms/variances per parameter and aggregate.
        # Gradient smoothness: compute variance of per-sample gradient vectors.

        # 1. Compute Mean Gradient
        # Sum all sample gradients
        sum_grad: "dict[str, Array]" = {}
        count = len(per_sample_gradients)

        for sample in per_sample_gradients:
            for k, v in sample.items():
                if k not in sum_grad:
                    sum_grad[k] = b.zeros_like(v)
                sum_grad[k] = sum_grad[k] + v

        mean_grad = {k: v / count for k, v in sum_grad.items()}

        # 2. Compute Mean Norm (E[||g||])
        total_norm_sum = 0.0
        for sample in per_sample_gradients:
            flats = [b.reshape(v, (-1,)) for v in sample.values()]
            if not flats:
                continue
            concat = b.concatenate(flats, axis=0)
            norm_arr = geodesic_norms(b.reshape(concat, (1, -1)), b)
            b.eval(norm_arr)
            total_norm_sum += float(b.to_scalar(norm_arr))

        mean_norm = total_norm_sum / count

        # 3. Compute Variance (E[||g - E[g]||^2])
        # Variance of the gradient vector
        variance_sum = 0.0
        for sample in per_sample_gradients:
            flats = []
            for k, v in sample.items():
                if k in mean_grad:
                    diff = v - mean_grad[k]
                    flats.append(b.reshape(diff, (-1,)))
            if not flats:
                continue
            concat = b.concatenate(flats, axis=0)
            diff_norm_arr = geodesic_norms(b.reshape(concat, (1, -1)), b)
            b.eval(diff_norm_arr)
            diff_norm = float(b.to_scalar(diff_norm_arr))
            variance_sum += diff_norm * diff_norm

        variance = variance_sum / (count - 1)

        # 4. SNR = MeanSquaredNorm / Variance? Or MeanNorm^2 / Variance?
        # Typically SNR = ||E[g]||^2 / Tr(Var(g))
        # Here we use: SNR = ||mean_grad||^2 / variance

        mean_flats = [b.reshape(v, (-1,)) for v in mean_grad.values()]
        mean_grad_norm_sq = 0.0
        if mean_flats:
            mean_concat = b.concatenate(mean_flats, axis=0)
            mean_norm_arr = geodesic_norms(b.reshape(mean_concat, (1, -1)), b)
            b.eval(mean_norm_arr)
            mean_norm = float(b.to_scalar(mean_norm_arr))
            mean_grad_norm_sq = mean_norm * mean_norm

        snr = mean_grad_norm_sq / (variance + eps)

        return LayerGradientQuality(
            variance=variance, snr=snr, mean_norm=mean_norm, sample_count=count
        )

    @staticmethod
    def _extract_layer_index_from_key(key: str) -> int | None:
        idx = GradientSmoothnessEstimator._parse_index(after=".layers.", in_str=key)
        if idx is not None:
            return idx

        idx = GradientSmoothnessEstimator._parse_index(after=".h.", in_str=key)
        if idx is not None:
            return idx

        idx = GradientSmoothnessEstimator._parse_index(after=".blocks.", in_str=key)
        if idx is not None:
            return idx

        idx = GradientSmoothnessEstimator._parse_index(after=".block.", in_str=key)
        if idx is not None:
            return idx

        return None

    @staticmethod
    def _parse_index(after: str, in_str: str) -> int | None:
        try:
            parts = in_str.split(after)
            if len(parts) > 1:
                suffix = parts[1]
                digits = ""
                for ch in suffix:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    return int(digits)
        except Exception:
            return None
        return None
