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

"""Manifold entropy measurement for neural network representations.

This module provides entropy measures across the manifold of neural network
activations. The entropy aggregates pure geometric measurements:

1. Layer-wise intrinsic dimension (TwoNN)
2. Effective rank (Shannon entropy-based)
3. Spectral entropy

This is READ-ONLY measurement - no weights are modified here.

References:
    - Facco et al. (2017) - TwoNN intrinsic dimension estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class LayerEntropyResult:
    """Entropy measurements for a single layer."""

    layer_idx: int
    intrinsic_dimension: float
    effective_rank: float  # Shannon effective rank (entropy-based)
    spectral_entropy: float
    sample_count: int

    @property
    def dimension_ratio(self) -> float:
        """Ratio of intrinsic dimension to effective rank.

        Values near 1.0 indicate consistent dimensionality.
        """
        if self.effective_rank > 0:
            return self.intrinsic_dimension / self.effective_rank
        return 0.0


@dataclass
class ManifoldEntropyResult:
    """Complete entropy measurement across the full manifold."""

    # Aggregate entropy (sum of spectral entropies)
    total_entropy: float

    # Per-layer breakdown
    layer_entropies: Dict[int, LayerEntropyResult] = field(default_factory=dict)

    # Summary statistics
    mean_intrinsic_dimension: float = 0.0
    mean_effective_rank: float = 0.0


class ManifoldEntropy:
    """Compute aggregate entropy across the full manifold.

    This provides pure geometric measurements of neural network activations.

    Usage:
        entropy = ManifoldEntropy(backend)
        result = entropy.compute_from_activations(layer_activations)
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._id_estimator = IntrinsicDimension(self._backend)
        self._eff_rank = EffectiveRank(self._backend)

    def compute_layer_entropy(
        self,
        activations: "Array",
        layer_idx: int,
    ) -> LayerEntropyResult:
        """Compute entropy for a single layer's activations.

        Args:
            activations: [n_samples, features] activation matrix
            layer_idx: Layer index for identification

        Returns:
            LayerEntropyResult with intrinsic dimension, effective rank, etc.
        """
        b = self._backend
        arr = b.array(activations) if not hasattr(activations, "shape") else activations
        b.eval(arr)

        n_samples = int(arr.shape[0])
        if n_samples < 4:
            return LayerEntropyResult(
                layer_idx=layer_idx,
                intrinsic_dimension=0.0,
                effective_rank=0.0,
                spectral_entropy=0.0,
                sample_count=n_samples,
            )

        # Compute intrinsic dimension (TwoNN)
        try:
            id_result = self._id_estimator.compute(arr)
            intrinsic_dim = id_result.intrinsic_dimension
        except Exception:
            intrinsic_dim = 0.0

        # Compute effective rank (Shannon entropy-based)
        eff_result = self._eff_rank.compute(arr)
        effective_rank = eff_result.shannon_effective_rank
        spectral_entropy = eff_result.spectral_entropy

        return LayerEntropyResult(
            layer_idx=layer_idx,
            intrinsic_dimension=intrinsic_dim,
            effective_rank=effective_rank,
            spectral_entropy=spectral_entropy,
            sample_count=n_samples,
        )

    def compute_from_activations(
        self,
        layer_activations: Dict[int, "Array"],
    ) -> ManifoldEntropyResult:
        """Compute manifold entropy from layer activations.

        Args:
            layer_activations: Dict mapping layer_idx to activation arrays
                              Each array is [n_samples, features]

        Returns:
            ManifoldEntropyResult with aggregate entropy and per-layer breakdown
        """
        layer_entropies: Dict[int, LayerEntropyResult] = {}
        total_spectral_entropy = 0.0
        total_id = 0.0
        total_rank = 0.0

        for layer_idx, activations in layer_activations.items():
            layer_result = self.compute_layer_entropy(activations, layer_idx)
            layer_entropies[layer_idx] = layer_result
            total_spectral_entropy += layer_result.spectral_entropy
            total_id += layer_result.intrinsic_dimension
            total_rank += layer_result.effective_rank

        n_layers = len(layer_entropies)
        mean_id = total_id / n_layers if n_layers > 0 else 0.0
        mean_rank = total_rank / n_layers if n_layers > 0 else 0.0

        return ManifoldEntropyResult(
            total_entropy=total_spectral_entropy,
            layer_entropies=layer_entropies,
            mean_intrinsic_dimension=mean_id,
            mean_effective_rank=mean_rank,
        )

    def compute_entropy_delta(
        self,
        before: ManifoldEntropyResult,
        after: ManifoldEntropyResult,
    ) -> float:
        """Compute the change in entropy.

        Returns:
            Positive value if entropy decreased,
            negative if increased.
        """
        return before.total_entropy - after.total_entropy


def compute_manifold_entropy(
    layer_activations: Dict[int, "Array"],
    backend: "Backend | None" = None,
) -> ManifoldEntropyResult:
    """Convenience function to compute manifold entropy.

    Args:
        layer_activations: Dict mapping layer_idx to activation arrays
        backend: Backend to use

    Returns:
        ManifoldEntropyResult
    """
    entropy = ManifoldEntropy(backend)
    return entropy.compute_from_activations(layer_activations)


__all__ = [
    # Main class
    "ManifoldEntropy",
    # Result types
    "ManifoldEntropyResult",
    "LayerEntropyResult",
    # Convenience function
    "compute_manifold_entropy",
]
