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
Activation Buffer - Rolling buffer with incremental statistics for null-space tracking.

This module maintains a rolling buffer of recent activations and computes
statistics needed for null-space identification:
1. Running mean and covariance
2. Incremental SVD for principal directions
3. Variance per dimension (for null-space detection)

The key insight: We don't need exact null-space computation at every step.
An incrementally-maintained approximation is sufficient for detecting which
directions are "in use" vs "available" for knowledge encoding.

Algorithm:
    1. Maintain rolling buffer of N most recent activations
    2. Track running mean: μ_new = μ_old + (x - μ_old) / n
    3. Track running covariance: C_new = C_old + (x - μ_old)(x - μ_new)^T / n
    4. Periodically update SVD: U, S, V = svd(C)
    5. Null-space ≈ directions with S[i] < threshold

Memory efficiency:
    - Store only N activations, not full history
    - Covariance matrix is [d, d] regardless of buffer size
    - SVD is O(d³) but only computed periodically

References:
    - Incremental SVD: Brand (2006) "Fast low-rank modifications of SVD"
    - Welford's algorithm for running variance
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class BufferStats:
    """Statistics computed from the activation buffer.

    Attributes:
        n_samples: Number of samples currently in buffer.
        mean: Running mean [hidden_dim].
        variance: Per-dimension variance [hidden_dim].
        total_variance: Sum of all variances (trace of covariance).
        svd_rank: Effective rank from SVD (singular values > threshold).
        svd_update_count: Number of SVD updates performed.
    """

    n_samples: int
    mean: "Array"
    variance: "Array"
    total_variance: float
    svd_rank: int
    svd_update_count: int


class ActivationBuffer:
    """Rolling buffer of activations with incremental statistics.

    Maintains a fixed-size buffer of recent activations and computes
    running statistics needed for null-space tracking.

    Usage:
        buffer = ActivationBuffer(buffer_size=1024, hidden_dim=4096)

        for activation in inference_loop:
            buffer.add(activation)

            if buffer.should_update_svd():
                buffer.update_svd()

            stats = buffer.get_stats()
            # Use stats.variance to identify null-space directions
    """

    def __init__(
        self,
        buffer_size: int,
        hidden_dim: int,
        svd_update_frequency: int | None = None,
        backend: Backend | None = None,
    ) -> None:
        """Initialize the activation buffer.

        Args:
            buffer_size: Maximum number of activations to store.
            hidden_dim: Dimension of activation vectors.
            svd_update_frequency: How often to recompute SVD (in samples).
                None = use rank change detection instead of fixed frequency.
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._buffer_size = buffer_size
        self._hidden_dim = hidden_dim
        self._svd_update_frequency_config = svd_update_frequency

        # Derive SVD update frequency if not set:
        # Use sqrt(hidden_dim) as the update frequency - this captures the
        # scale at which rank changes are meaningful. Smaller dims need
        # more frequent updates; larger dims are more stable.
        if svd_update_frequency is not None:
            self._svd_update_frequency = svd_update_frequency
        else:
            # sqrt(hidden_dim) is the natural scale for rank stability
            # Minimum 8 samples between updates for stability, max 128 for responsiveness
            self._svd_update_frequency = max(8, min(128, int(hidden_dim**0.5)))

        # Track previous rank for change detection
        self._previous_svd_rank = 0

        # Rolling buffer of activations
        self._buffer: deque[Array] = deque(maxlen=buffer_size)

        # Running statistics (Welford's algorithm)
        self._n_samples = 0
        self._mean = self._backend.zeros((hidden_dim,))
        self._m2 = self._backend.zeros((hidden_dim,))  # Sum of squared deviations

        # Covariance matrix (for SVD)
        self._covariance: Array | None = None
        self._covariance_dirty = True

        # SVD results
        self._svd_u: Array | None = None
        self._svd_s: Array | None = None
        self._svd_v: Array | None = None
        self._svd_rank = 0
        self._svd_update_count = 0
        self._samples_since_svd = 0

    def add(self, activation: Array) -> None:
        """Add an activation to the buffer.

        Updates running statistics incrementally.

        Args:
            activation: Activation vector [hidden_dim] or [1, hidden_dim].
        """
        b = self._backend

        # Ensure 1D
        if activation.ndim > 1:
            activation = b.reshape(activation, (-1,))

        # If buffer is full, we need to remove the oldest and update stats
        if len(self._buffer) >= self._buffer_size:
            old_activation = self._buffer[0]
            self._remove_from_stats(old_activation)

        # Add to buffer
        self._buffer.append(activation)

        # Update running statistics (Welford's algorithm)
        self._n_samples += 1
        delta = activation - self._mean
        self._mean = self._mean + delta / float(self._n_samples)
        delta2 = activation - self._mean
        self._m2 = self._m2 + delta * delta2

        b.eval(self._mean, self._m2)

        # Mark covariance as needing update
        self._covariance_dirty = True
        self._samples_since_svd += 1

    def _remove_from_stats(self, activation: Array) -> None:
        """Remove an activation's contribution from running statistics.

        This is the reverse of the Welford update, needed when the buffer
        rolls over.

        Args:
            activation: Activation being removed from buffer.
        """
        if self._n_samples <= 1:
            # Reset to initial state
            self._mean = self._backend.zeros((self._hidden_dim,))
            self._m2 = self._backend.zeros((self._hidden_dim,))
            self._n_samples = 0
            return

        b = self._backend

        # Reverse Welford update
        # This is approximate when buffer is full - we'd need to store
        # more history for exact removal. For our purposes, the approximation
        # is acceptable since we're tracking broad statistical properties.
        delta2 = activation - self._mean
        new_mean = (self._mean * self._n_samples - activation) / float(
            self._n_samples - 1
        )
        delta = activation - new_mean
        self._m2 = self._m2 - delta * delta2

        # Clamp M2 to non-negative (numerical stability)
        self._m2 = b.maximum(self._m2, b.zeros_like(self._m2))

        self._mean = new_mean
        self._n_samples -= 1

        b.eval(self._mean, self._m2)

    def get_variance(self) -> Array:
        """Get per-dimension variance.

        Returns:
            Variance vector [hidden_dim].
        """
        if self._n_samples < 2:
            return self._backend.zeros((self._hidden_dim,))

        variance = self._m2 / float(self._n_samples - 1)
        self._backend.eval(variance)
        return variance

    def get_covariance(self) -> Array:
        """Compute full covariance matrix.

        This is expensive O(n * d²) but needed for SVD.
        Results are cached until new samples arrive.

        Returns:
            Covariance matrix [hidden_dim, hidden_dim].
        """
        if not self._covariance_dirty and self._covariance is not None:
            return self._covariance

        b = self._backend

        if len(self._buffer) < 2:
            self._covariance = b.zeros((self._hidden_dim, self._hidden_dim))
            self._covariance_dirty = False
            return self._covariance

        # Stack activations and center them
        activations = b.stack(list(self._buffer), axis=0)  # [n, d]
        centered = activations - self._mean[None, :]  # [n, d]

        # Covariance: (1/(n-1)) * X^T @ X
        n = len(self._buffer)
        self._covariance = b.matmul(b.transpose(centered), centered) / float(n - 1)

        b.eval(self._covariance)
        self._covariance_dirty = False
        return self._covariance

    def should_update_svd(self) -> bool:
        """Check if SVD should be recomputed.

        Uses frequency-based triggering with sqrt(hidden_dim) as the natural
        scale for rank stability checks.
        """
        return self._samples_since_svd >= self._svd_update_frequency

    def update_svd(self) -> None:
        """Recompute SVD of covariance matrix.

        Updates internal SVD state for null-space tracking.
        """
        b = self._backend

        if len(self._buffer) < 2:
            return

        covariance = self.get_covariance()

        # Full SVD
        # For symmetric positive semi-definite covariance, eigendecomposition
        # is equivalent and may be faster
        try:
            # Use eigendecomposition for symmetric matrix
            eigenvalues, eigenvectors = b.eigh(covariance)

            # Sort by descending eigenvalue (eigh returns ascending)
            n = int(eigenvalues.shape[0])
            reverse_idx = b.arange(n - 1, -1, -1)
            self._svd_s = b.take(eigenvalues, reverse_idx, axis=0)
            self._svd_v = b.take(eigenvectors, reverse_idx, axis=1)
            self._svd_u = self._svd_v  # For symmetric matrices, U = V

        except Exception:
            # Fall back to SVD if eigendecomposition fails
            self._svd_u, self._svd_s, vt = b.svd(covariance, full_matrices=False)
            self._svd_v = b.transpose(vt)

        b.eval(self._svd_u, self._svd_s, self._svd_v)

        # Compute effective rank (singular values > machine epsilon * max)
        from modelcypher.core.domain.geometry.numerical_stability import (
            machine_epsilon,
        )

        eps = machine_epsilon(b, self._svd_s)
        max_s = b.max(self._svd_s)
        b.eval(max_s)
        threshold = float(b.to_scalar(max_s)) * eps

        # Count significant singular values
        significant = b.sum(
            b.astype(self._svd_s > b.array([threshold]), self._svd_s.dtype)
        )
        b.eval(significant)
        self._svd_rank = int(b.to_scalar(significant))

        self._svd_update_count += 1
        self._samples_since_svd = 0

    def get_stats(self) -> BufferStats:
        """Get current buffer statistics.

        Returns:
            BufferStats with mean, variance, rank information.
        """
        variance = self.get_variance()
        total_var = self._backend.sum(variance)
        self._backend.eval(total_var)

        return BufferStats(
            n_samples=len(self._buffer),
            mean=self._mean,
            variance=variance,
            total_variance=float(self._backend.to_scalar(total_var)),
            svd_rank=self._svd_rank,
            svd_update_count=self._svd_update_count,
        )

    def get_principal_directions(self, k: int | None = None) -> Array | None:
        """Get top-k principal directions (high-variance).

        Args:
            k: Number of directions to return. None = all.

        Returns:
            Principal directions [k, hidden_dim] or None if SVD not computed.
        """
        if self._svd_v is None:
            return None

        b = self._backend
        if k is None:
            return b.transpose(self._svd_v)  # [d, d] -> [d, d] (columns are eigenvectors)

        # Take top k columns
        top_k = b.take(self._svd_v, b.arange(k), axis=1)  # [d, k]
        return b.transpose(top_k)  # [k, d]

    def get_null_directions(self, k: int | None = None) -> Array | None:
        """Get bottom-k directions (low-variance, null-space approximation).

        These are directions that are minimally used by current activations,
        hence available for new knowledge encoding.

        Args:
            k: Number of directions to return. None = all below rank.

        Returns:
            Null-space directions [k, hidden_dim] or None if SVD not computed.
        """
        if self._svd_v is None or self._svd_rank >= self._hidden_dim:
            return None

        b = self._backend

        # Null directions are the last (d - rank) columns
        null_start = self._svd_rank
        n_null = self._hidden_dim - null_start

        if k is not None:
            n_null = min(k, n_null)

        null_indices = b.arange(null_start, null_start + n_null)
        null_dirs = b.take(self._svd_v, null_indices, axis=1)  # [d, n_null]
        return b.transpose(null_dirs)  # [n_null, d]

    def get_singular_values(self) -> Array | None:
        """Get singular values from last SVD.

        Returns:
            Singular values [hidden_dim] or None if SVD not computed.
        """
        return self._svd_s

    def reset(self) -> None:
        """Reset buffer and all statistics."""
        self._buffer.clear()
        self._n_samples = 0
        self._mean = self._backend.zeros((self._hidden_dim,))
        self._m2 = self._backend.zeros((self._hidden_dim,))
        self._covariance = None
        self._covariance_dirty = True
        self._svd_u = None
        self._svd_s = None
        self._svd_v = None
        self._svd_rank = 0
        self._previous_svd_rank = 0
        self._svd_update_count = 0
        self._samples_since_svd = 0

    @property
    def buffer_size(self) -> int:
        """Maximum buffer capacity."""
        return self._buffer_size

    @property
    def current_size(self) -> int:
        """Current number of samples in buffer."""
        return len(self._buffer)

    @property
    def hidden_dim(self) -> int:
        """Activation dimension."""
        return self._hidden_dim

    @property
    def is_full(self) -> bool:
        """Whether buffer is at capacity."""
        return len(self._buffer) >= self._buffer_size
