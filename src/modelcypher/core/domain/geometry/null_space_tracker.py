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
Null-Space Tracker - Track used vs available dimensions per layer.

Maintains per-layer activation buffers and summarizes used vs available
dimensions based on variance/SVD. Supports observation and projection modes.

References:
    - GNSP: Gradient Null Space Projection (arXiv:2507.19839)
    - PNSP: Primary Null Space Projection (ScienceDirect 2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.activation_buffer import ActivationBuffer

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class NullSpaceState:
    """State of null-space availability for a layer or model.

    Attributes:
        layer_id: Layer index (-1 for model-wide summary).
        hidden_dim: Dimension of the activation space.
        n_samples: Number of activation samples in the buffer.
        buffer_size: Maximum samples retained in the buffer.
        coverage_ratio: n_samples / hidden_dim.
        used_rank: Number of dimensions actively used (high variance).
        null_rank: Number of dimensions available (low variance).
        capacity_fraction: Fraction of space available (null_rank / hidden_dim).
        total_variance: Sum of all eigenvalues (total activation energy).
        null_variance: Sum of null-space eigenvalues (available capacity).
        svd_update_count: Number of times SVD has been updated.
    """

    layer_id: int
    hidden_dim: int
    n_samples: int
    buffer_size: int
    coverage_ratio: float
    used_rank: int
    null_rank: int
    capacity_fraction: float
    total_variance: float
    null_variance: float
    svd_update_count: int

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "layer_id": self.layer_id,
            "hidden_dim": self.hidden_dim,
            "n_samples": self.n_samples,
            "buffer_size": self.buffer_size,
            "coverage_ratio": self.coverage_ratio,
            "used_rank": self.used_rank,
            "null_rank": self.null_rank,
            "capacity_fraction": self.capacity_fraction,
            "total_variance": self.total_variance,
            "null_variance": self.null_variance,
            "svd_update_count": self.svd_update_count,
        }


class NullSpaceTracker:
    """Tracks null-space availability across model layers.

    Maintains per-layer activation buffers and computes null-space
    statistics for knowledge encoding.

    Usage:
        tracker = NullSpaceTracker(n_layers=32, hidden_dim=4096)

        for batch in data_stream:
            for layer_id, activation in enumerate(layer_activations):
                tracker.add_activation(layer_id, activation)

            if tracker.should_update():
                tracker.update_all_layers()

        # Get null-space projector for a specific layer
        projector = tracker.get_null_projector(layer_id=16)
        safe_delta = projector @ weight_delta  # Project update to null-space
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        backend: Backend | None = None,
    ) -> None:
        """Initialize the null-space tracker.

        Args:
            n_layers: Number of transformer layers to track.
            hidden_dim: Dimension of activation vectors.
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim

        # Create per-layer buffers
        self._buffers: list[ActivationBuffer] = [
            ActivationBuffer(
                hidden_dim=hidden_dim,
                backend=self._backend,
            )
            for _ in range(n_layers)
        ]

        # Track total samples added
        self._total_samples = 0

    def add_activation(self, layer_id: int, activation: Array) -> None:
        """Add an activation to a specific layer's buffer.

        Args:
            layer_id: Index of the transformer layer.
            activation: Activation vector [hidden_dim].
        """
        if layer_id < 0 or layer_id >= self._n_layers:
            raise ValueError(f"Layer {layer_id} out of range [0, {self._n_layers})")

        self._buffers[layer_id].add(activation)
        self._total_samples += 1

    def add_all_layers(self, activations: dict[int, Array]) -> None:
        """Add activations for multiple layers at once.

        Args:
            activations: Mapping of layer_id -> activation vector.
        """
        for layer_id, activation in activations.items():
            self.add_activation(layer_id, activation)

    def should_update(self) -> bool:
        """Check if any layer should update its SVD."""
        return any(buf.should_update_svd() for buf in self._buffers)

    def update_layer(self, layer_id: int) -> None:
        """Update SVD for a specific layer.

        Args:
            layer_id: Index of the layer to update.
        """
        self._buffers[layer_id].update_svd()

    def update_all_layers(self) -> None:
        """Update SVD for all layers that need it."""
        for buf in self._buffers:
            if buf.should_update_svd():
                buf.update_svd()

    def get_layer_state(self, layer_id: int) -> NullSpaceState:
        """Get null-space state for a specific layer.

        Args:
            layer_id: Index of the layer.

        Returns:
            NullSpaceState with rank and capacity information.
        """
        buf = self._buffers[layer_id]
        stats = buf.get_stats()

        # Get singular values for variance computation
        singular_values = buf.get_singular_values()
        n_samples = stats.n_samples
        buffer_size = buf.buffer_size
        coverage_ratio = n_samples / self._hidden_dim if self._hidden_dim > 0 else 0.0
        if singular_values is None:
            return NullSpaceState(
                layer_id=layer_id,
                hidden_dim=self._hidden_dim,
                n_samples=n_samples,
                buffer_size=buffer_size,
                coverage_ratio=coverage_ratio,
                used_rank=0,
                null_rank=self._hidden_dim,
                capacity_fraction=1.0,
                total_variance=stats.total_variance,
                null_variance=stats.total_variance,
                svd_update_count=stats.svd_update_count,
            )

        b = self._backend

        # Compute null-space variance
        used_rank = stats.svd_rank
        null_rank = self._hidden_dim - used_rank

        # Total variance is sum of all singular values (eigenvalues of covariance)
        total_var = b.sum(singular_values)
        b.eval(total_var)
        total_variance = float(b.to_scalar(total_var))

        # Null variance is sum of small singular values
        if null_rank > 0:
            null_indices = b.arange(used_rank, self._hidden_dim)
            # Handle case where SVD might have fewer values than hidden_dim
            n_svd = int(singular_values.shape[0])
            if used_rank < n_svd:
                actual_null_indices = b.arange(used_rank, min(n_svd, self._hidden_dim))
                null_values = b.take(singular_values, actual_null_indices, axis=0)
                null_var = b.sum(null_values)
                b.eval(null_var)
                null_variance = float(b.to_scalar(null_var))
            else:
                null_variance = 0.0
        else:
            null_variance = 0.0

        capacity_fraction = null_rank / self._hidden_dim if self._hidden_dim > 0 else 0.0

        return NullSpaceState(
            layer_id=layer_id,
            hidden_dim=self._hidden_dim,
            n_samples=n_samples,
            buffer_size=buffer_size,
            coverage_ratio=coverage_ratio,
            used_rank=used_rank,
            null_rank=null_rank,
            capacity_fraction=capacity_fraction,
            total_variance=total_variance,
            null_variance=null_variance,
            svd_update_count=stats.svd_update_count,
        )

    def get_model_state(self) -> NullSpaceState:
        """Get aggregate null-space state across all layers.

        Returns:
            NullSpaceState with model-wide averages (layer_id=-1).
        """
        layer_states = [self.get_layer_state(i) for i in range(self._n_layers)]

        if not layer_states:
            return NullSpaceState(
                layer_id=-1,
                hidden_dim=self._hidden_dim,
                n_samples=0,
                buffer_size=self._hidden_dim + 1,
                coverage_ratio=0.0,
                used_rank=0,
                null_rank=self._hidden_dim,
                capacity_fraction=1.0,
                total_variance=0.0,
                null_variance=0.0,
                svd_update_count=0,
            )

        # Average across layers
        avg_used = sum(s.used_rank for s in layer_states) / len(layer_states)
        avg_null = sum(s.null_rank for s in layer_states) / len(layer_states)
        avg_capacity = sum(s.capacity_fraction for s in layer_states) / len(layer_states)
        avg_samples = sum(s.n_samples for s in layer_states) / len(layer_states)
        avg_coverage = sum(s.coverage_ratio for s in layer_states) / len(layer_states)
        avg_svd_updates = sum(s.svd_update_count for s in layer_states) / len(
            layer_states
        )
        total_var = sum(s.total_variance for s in layer_states)
        null_var = sum(s.null_variance for s in layer_states)

        return NullSpaceState(
            layer_id=-1,
            hidden_dim=self._hidden_dim,
            n_samples=int(avg_samples),
            buffer_size=layer_states[0].buffer_size,
            coverage_ratio=avg_coverage,
            used_rank=int(avg_used),
            null_rank=int(avg_null),
            capacity_fraction=avg_capacity,
            total_variance=total_var,
            null_variance=null_var,
            svd_update_count=int(avg_svd_updates),
        )

    def get_null_basis(self, layer_id: int) -> Array | None:
        """Get null-space basis vectors for a layer.

        The null-space basis consists of orthonormal directions that are
        minimally used by current activations. These are safe directions
        for encoding new knowledge without interference.

        Args:
            layer_id: Index of the layer.

        Returns:
            Basis matrix [null_rank, hidden_dim] or None if not ready.
            Each row is a unit vector in the null-space.
        """
        if layer_id < 0 or layer_id >= self._n_layers:
            return None

        buf = self._buffers[layer_id]
        return buf.get_null_directions()

    def get_null_projector(self, layer_id: int) -> Array | None:
        """Get null-space projection matrix for a layer.

        The projector P_null satisfies:
            - P_null @ v is in the null-space for any v
            - P_null is idempotent: P_null @ P_null = P_null
            - P_null preserves null-space: P_null @ null_vec = null_vec

        Args:
            layer_id: Index of the layer.

        Returns:
            Projection matrix [hidden_dim, hidden_dim] or None if not ready.
        """
        null_dirs = self.get_null_basis(layer_id)

        if null_dirs is None:
            return None

        b = self._backend

        # Projector: P = V_null @ V_null^T
        # where V_null has null-space directions as rows [k, d]
        projector = b.matmul(b.transpose(null_dirs), null_dirs)  # [d, d]
        b.eval(projector)

        return projector

    def get_variance_weights(self, layer_id: int) -> Array | None:
        """Get variance-based weights for transfer.

        These weights are used for variance-weighted null-space projection
        (the approach used in model merging).

        High variance = dense direction = scale down transfer
        Low variance = sparse direction = allow transfer

        Args:
            layer_id: Index of the layer.

        Returns:
            Weight vector [hidden_dim] in [0, 1] or None if not ready.
        """
        buf = self._buffers[layer_id]
        variance = buf.get_variance()

        if buf.current_size < 2:
            return None

        b = self._backend

        # Normalize variance to [0, 1] range
        max_var = b.max(variance)
        b.eval(max_var)

        if float(b.to_scalar(max_var)) == 0:
            return b.ones((self._hidden_dim,))

        # High variance -> low weight (don't transfer there)
        # Low variance -> high weight (transfer there)
        weights = 1.0 - (variance / max_var)

        b.eval(weights)
        return weights

    def project_to_null_space(
        self,
        layer_id: int,
        delta: Array,
        use_variance_weighting: bool = True,
    ) -> Array | None:
        """Project a weight delta to the null-space.

        Args:
            layer_id: Index of the layer.
            delta: Weight delta to project [out_dim, in_dim] or [in_dim].
            use_variance_weighting: If True, use variance weights instead of
                hard projection.

        Returns:
            Projected delta or None if not ready.
        """
        b = self._backend

        if use_variance_weighting:
            weights = self.get_variance_weights(layer_id)
            if weights is None:
                return None

            # Apply variance-based scaling
            # For 2D: scale columns (input dimensions)
            if delta.ndim == 2:
                projected = delta * weights[None, :]
            else:
                projected = delta * weights

        else:
            projector = self.get_null_projector(layer_id)
            if projector is None:
                return None

            # For 2D: project each row
            if delta.ndim == 2:
                projected = b.matmul(delta, projector)
            else:
                projected = b.matmul(delta[None, :], projector)[0]

        b.eval(projected)
        return projected

    def reset(self) -> None:
        """Reset all buffers and statistics."""
        for buf in self._buffers:
            buf.reset()
        self._total_samples = 0

    def reset_layer(self, layer_id: int) -> None:
        """Reset a specific layer's buffer."""
        self._buffers[layer_id].reset()

    @property
    def n_layers(self) -> int:
        """Number of layers being tracked."""
        return self._n_layers

    @property
    def hidden_dim(self) -> int:
        """Activation dimension."""
        return self._hidden_dim

    @property
    def total_samples(self) -> int:
        """Total activation samples added across all layers."""
        return self._total_samples

    def get_layer_buffer(self, layer_id: int) -> ActivationBuffer:
        """Get the activation buffer for a specific layer.

        Args:
            layer_id: Index of the layer.

        Returns:
            The ActivationBuffer for that layer.
        """
        return self._buffers[layer_id]
