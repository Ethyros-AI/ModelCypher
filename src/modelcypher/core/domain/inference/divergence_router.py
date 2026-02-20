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

"""Activation-divergence measurements for experimental adapter routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.logit_divergence_calculator import LogitDivergenceCalculator
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.inference.adapter_routing import (
    LayerRoutingMeasurement,
    LayerRoutingSnapshot,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class LayerDivergenceComputer:
    """Compute per-layer divergence signals between base and adapted activations."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._kl_calc = LogitDivergenceCalculator(self._backend)

    def compute_layer_divergence(
        self,
        base_activation: "Array",
        adapted_activation: "Array",
        layer_index: int,
        adapter_id: str,
    ) -> LayerRoutingMeasurement:
        """Compute KL, norm ratio, and cosine similarity for one layer."""
        b = self._backend

        base_vector = self._last_token_vector(base_activation)
        adapted_vector = self._last_token_vector(adapted_activation)
        if base_vector.shape != adapted_vector.shape:
            raise ValueError(
                "Activation shape mismatch for routing: "
                f"base={base_vector.shape}, adapted={adapted_vector.shape}",
            )

        adapted_magnitude = b.abs(adapted_vector)
        base_magnitude = b.abs(base_vector)
        kl_divergence = self._kl_calc.kl_divergence(adapted_magnitude, base_magnitude)

        base_norm_tensor = b.norm(base_vector)
        adapted_norm_tensor = b.norm(adapted_vector)
        dot_tensor = b.dot(adapted_vector, base_vector)
        b.eval(base_norm_tensor, adapted_norm_tensor, dot_tensor)

        base_norm = float(b.to_scalar(base_norm_tensor))
        adapted_norm = float(b.to_scalar(adapted_norm_tensor))
        dot_value = float(b.to_scalar(dot_tensor))

        eps = division_epsilon(b, base_vector)
        norm_denom = base_norm if base_norm > eps else eps
        activation_norm_ratio = adapted_norm / norm_denom

        cosine_denom = base_norm * adapted_norm
        if cosine_denom <= eps:
            cosine_similarity = 0.0
        else:
            cosine_similarity = dot_value / cosine_denom
            cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        return LayerRoutingMeasurement(
            layer_index=layer_index,
            adapter_id=adapter_id,
            kl_divergence=kl_divergence,
            activation_norm_ratio=activation_norm_ratio,
            cosine_similarity=cosine_similarity,
        )

    def compute_routing_snapshot(
        self,
        base_activation: "Array",
        adapted_activations: dict[str, "Array"],
        layer_index: int,
    ) -> LayerRoutingSnapshot:
        """Compute per-adapter divergence measurements for a single layer."""
        b = self._backend
        base_vector = self._last_token_vector(base_activation)
        base_norm_tensor = b.norm(base_vector)
        b.eval(base_norm_tensor)
        base_activation_norm = float(b.to_scalar(base_norm_tensor))

        measurements = tuple(
            self.compute_layer_divergence(
                base_activation=base_activation,
                adapted_activation=adapted_activations[adapter_id],
                layer_index=layer_index,
                adapter_id=adapter_id,
            )
            for adapter_id in sorted(adapted_activations)
        )

        return LayerRoutingSnapshot(
            layer_index=layer_index,
            measurements=measurements,
            base_activation_norm=base_activation_norm,
        )

    def _last_token_vector(self, activation: "Array") -> "Array":
        """Extract last-token hidden vector from [b,s,h], [s,h], or [h] activation."""
        ndim = activation.ndim
        if ndim == 3:
            return activation[0, -1]
        if ndim == 2:
            return activation[-1]
        if ndim == 1:
            return activation
        raise ValueError(f"Unsupported activation rank for routing: ndim={ndim}")


__all__ = ["LayerDivergenceComputer"]

