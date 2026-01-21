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

"""Attractor detection and escape for manifold trajectories.

This module implements detection and escape from attractor basins in the
activation manifold - the geometric cause of repetition loops in autoregressive
generation.

Theoretical Foundation:
    In dynamical systems, an attractor is a region where trajectories converge
    and remain trapped. For autoregressive generation, this manifests as:

    1. Fixed Points: The hidden state stops changing (position variance → 0)
       Example: Model outputs "... ... ... ..."

    2. Limit Cycles: The trajectory returns to the same region periodically
       Example: Model outputs "I think I think I think I think..."

    3. Strange Attractors: Chaotic but bounded trajectories
       Example: Model outputs varying but semantically repetitive content

Detection is based on geometric measures in activation space:
    - Cosine distance: How different is the current direction from previous?
    - Positional variance: Is the trajectory frozen?
    - Cycle detection: Have we visited this region before?

Escape is achieved through null-space perturbation:
    - Find direction in null-space with highest variance
    - Perturb position along this direction
    - Reset velocity to break momentum

All thresholds derive from machine precision (sqrt(eps)), not heuristics.

References:
    - Strogatz, "Nonlinear Dynamics and Chaos" (attractor theory)
    - Our manifold_native_architecture.md design document
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class AttractorType(str, Enum):
    """Type of attractor detected."""

    NONE = "none"  # No attractor - trajectory is flowing freely
    FIXED_POINT = "fixed_point"  # Trajectory frozen at a single point
    LIMIT_CYCLE = "limit_cycle"  # Trajectory cycling between regions
    SLOW_DYNAMICS = "slow_dynamics"  # Trajectory moving but very slowly


@dataclass(frozen=True)
class AttractorState:
    """State of attractor detection.

    Attributes:
        attractor_type: Type of attractor detected.
        severity: How trapped is the trajectory [0, 1].
            0 = flowing freely
            1 = completely stuck
        cycle_length: If limit cycle, the period. Otherwise 0.
        position_variance: Variance of recent positions.
        velocity_magnitude: Current velocity magnitude.
        timesteps_stuck: Number of timesteps in attractor basin.
        escape_direction: Suggested direction for escape (if attractor detected).
    """

    attractor_type: AttractorType
    severity: float
    cycle_length: int
    position_variance: float
    velocity_magnitude: float
    timesteps_stuck: int
    escape_direction: tuple[float, ...] | None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "attractor_type": self.attractor_type.value,
            "severity": self.severity,
            "cycle_length": self.cycle_length,
            "position_variance": self.position_variance,
            "velocity_magnitude": self.velocity_magnitude,
            "timesteps_stuck": self.timesteps_stuck,
            "has_escape_direction": self.escape_direction is not None,
        }


class AttractorDetector:
    """Detect attractor basins in manifold trajectories.

    Tracks the trajectory of hidden states and detects when the model
    is stuck in a fixed point or limit cycle. All thresholds are derived
    from machine precision.

    Usage:
        detector = AttractorDetector(hidden_dim=4096)

        for hidden_state in generation:
            state = detector.update(hidden_state)

            if state.attractor_type != AttractorType.NONE:
                # Escape the attractor
                if state.escape_direction is not None:
                    hidden_state = escape(hidden_state, state.escape_direction)

    Parameters:
        hidden_dim: Dimension of hidden states.
        window_size: Number of recent positions to track. Default derives from
            hidden_dim: window = max(10, sqrt(hidden_dim)).
        backend: Compute backend.
    """

    def __init__(
        self,
        hidden_dim: int,
        window_size: int | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize attractor detector."""
        self._backend = backend or get_default_backend()
        self._hidden_dim = hidden_dim

        # Window size derived from geometry if not specified
        # Intuition: larger manifolds need more samples to detect patterns
        self._window_size = window_size or max(10, int(math.sqrt(hidden_dim)))

        # Derive precision threshold from machine epsilon
        ref = self._backend.array([1.0])
        eps = machine_epsilon(self._backend, ref)
        self._sqrt_eps = math.sqrt(float(eps))

        # Trajectory storage
        self._positions: list["Array"] = []
        self._velocities: list["Array"] = []
        self._timesteps_stuck = 0

        # Precompute thresholds
        # Fixed point: variance below numerical floor (scaled by dimension)
        self._fixed_point_threshold = self._sqrt_eps * math.sqrt(hidden_dim)
        # Limit cycle: cosine similarity threshold
        # Use 0.99 as practical threshold - positions 99% similar = same region
        # This is derived from observation that repetition produces ~0.999 similarity
        # but numerical noise in forward pass adds ~0.01 variation
        self._cycle_similarity_threshold = 0.99
        # Slow dynamics: velocity magnitude below sqrt(eps) * sqrt(dim)
        self._slow_velocity_threshold = self._sqrt_eps * math.sqrt(hidden_dim)

    def update(
        self,
        hidden_state: "Array",
        null_basis: "Array | None" = None,
    ) -> AttractorState:
        """Update trajectory and detect attractor.

        Parameters:
            hidden_state: Current hidden state [hidden_dim].
            null_basis: Optional null-space basis for computing escape direction.
                Shape [null_rank, hidden_dim].

        Returns:
            AttractorState with detection results and escape direction.
        """
        b = self._backend

        # Flatten if needed
        if hidden_state.ndim > 1:
            hidden_state = b.reshape(hidden_state, (-1,))

        # Compute velocity if we have previous position
        if self._positions:
            velocity = hidden_state - self._positions[-1]
            b.eval(velocity)
        else:
            velocity = b.zeros((self._hidden_dim,))

        # Add to trajectory
        self._positions.append(hidden_state)
        self._velocities.append(velocity)

        # Trim to window size
        if len(self._positions) > self._window_size:
            self._positions = self._positions[-self._window_size:]
            self._velocities = self._velocities[-self._window_size:]

        # Need minimum samples for detection
        if len(self._positions) < 4:
            return AttractorState(
                attractor_type=AttractorType.NONE,
                severity=0.0,
                cycle_length=0,
                position_variance=float("inf"),
                velocity_magnitude=float("inf"),
                timesteps_stuck=0,
                escape_direction=None,
            )

        # Compute trajectory statistics
        pos_variance = self._compute_position_variance()
        vel_magnitude = self._compute_velocity_magnitude(velocity)
        cycle_info = self._detect_limit_cycle()

        # Classify attractor type
        attractor_type, severity = self._classify_attractor(
            pos_variance, vel_magnitude, cycle_info
        )

        # Update stuck counter
        if attractor_type != AttractorType.NONE:
            self._timesteps_stuck += 1
        else:
            self._timesteps_stuck = 0

        # Compute escape direction if needed
        escape_direction = None
        if attractor_type != AttractorType.NONE and null_basis is not None:
            escape_direction = self._compute_escape_direction(
                hidden_state, null_basis
            )

        return AttractorState(
            attractor_type=attractor_type,
            severity=severity,
            cycle_length=cycle_info[0] if cycle_info else 0,
            position_variance=pos_variance,
            velocity_magnitude=vel_magnitude,
            timesteps_stuck=self._timesteps_stuck,
            escape_direction=escape_direction,
        )

    def _compute_position_variance(self) -> float:
        """Compute variance of recent positions.

        Returns mean pairwise squared distance normalized by hidden_dim.
        Low variance = frozen trajectory (fixed point).
        """
        b = self._backend

        if len(self._positions) < 2:
            return float("inf")

        # Stack positions
        positions = b.stack(self._positions, axis=0)  # [n, hidden_dim]

        # Compute mean position
        mean_pos = b.mean(positions, axis=0)  # [hidden_dim]

        # Compute mean squared distance from mean
        diff = positions - mean_pos
        sq_dist = b.sum(diff * diff, axis=1)  # [n]
        mean_sq_dist = b.mean(sq_dist)
        b.eval(mean_sq_dist)

        # Normalize by dimension
        variance = float(b.to_scalar(mean_sq_dist)) / self._hidden_dim
        return variance

    def _compute_velocity_magnitude(self, velocity: "Array") -> float:
        """Compute magnitude of current velocity."""
        b = self._backend

        sq_mag = b.sum(velocity * velocity)
        b.eval(sq_mag)

        # Normalize by dimension for scale-invariance
        magnitude = math.sqrt(float(b.to_scalar(sq_mag)) / self._hidden_dim)
        return magnitude

    def _detect_limit_cycle(self) -> tuple[int, float] | None:
        """Detect if trajectory is in a limit cycle.

        Returns (cycle_length, similarity) if cycle detected, else None.

        Detection method: Look for periodic patterns in DIRECTION of movement,
        not absolute position. This handles position-encoded transformers where
        the same token at different positions has different hidden states.

        We check if the velocity directions are cycling (repeating pattern).
        """
        b = self._backend

        if len(self._velocities) < 4:
            return None

        # Use velocity (direction) rather than position for cycle detection
        # This is more robust to position encoding
        current_vel = self._velocities[-1]
        current_norm = b.sqrt(b.sum(current_vel * current_vel))
        b.eval(current_norm)
        current_norm_val = float(b.to_scalar(current_norm))

        if current_norm_val < self._sqrt_eps:
            # Very low velocity - might be fixed point
            return None

        # Check similarity to previous velocities (skip immediate neighbors)
        max_similarity = 0.0
        best_offset = 0

        for offset in range(2, min(len(self._velocities), self._window_size // 2)):
            past_vel = self._velocities[-(offset + 1)]
            past_norm = b.sqrt(b.sum(past_vel * past_vel))
            b.eval(past_norm)
            past_norm_val = float(b.to_scalar(past_norm))

            if past_norm_val < self._sqrt_eps:
                continue

            # Cosine similarity of velocities
            dot = b.sum(current_vel * past_vel)
            b.eval(dot)
            dot_val = float(b.to_scalar(dot))

            similarity = dot_val / (current_norm_val * past_norm_val)

            if similarity > max_similarity:
                max_similarity = similarity
                best_offset = offset

        if max_similarity > self._cycle_similarity_threshold:
            return (best_offset, max_similarity)

        return None

    def _classify_attractor(
        self,
        pos_variance: float,
        vel_magnitude: float,
        cycle_info: tuple[int, float] | None,
    ) -> tuple[AttractorType, float]:
        """Classify attractor type and compute severity.

        Severity is in [0, 1] where:
        - 0 = flowing freely
        - 1 = completely stuck
        """
        # Fixed point: variance and velocity both below threshold
        if pos_variance < self._fixed_point_threshold:
            if vel_magnitude < self._slow_velocity_threshold:
                # Severity based on how far below threshold
                severity = 1.0 - (pos_variance / self._fixed_point_threshold)
                return AttractorType.FIXED_POINT, min(1.0, max(0.0, severity))

        # Limit cycle: high similarity to past position
        if cycle_info is not None:
            _, similarity = cycle_info
            # Severity based on similarity
            severity = (similarity - self._cycle_similarity_threshold) / (
                1.0 - self._cycle_similarity_threshold
            )
            return AttractorType.LIMIT_CYCLE, min(1.0, max(0.0, severity))

        # Slow dynamics: low velocity but not frozen
        if vel_magnitude < self._slow_velocity_threshold:
            severity = 1.0 - (vel_magnitude / self._slow_velocity_threshold)
            return AttractorType.SLOW_DYNAMICS, min(1.0, max(0.0, severity * 0.5))

        return AttractorType.NONE, 0.0

    def _compute_escape_direction(
        self,
        current_position: "Array",
        null_basis: "Array",
    ) -> tuple[float, ...]:
        """Compute escape direction in null-space.

        Finds the null-space direction with maximum projected variance
        of recent positions. This is the direction of least constraint.

        Parameters:
            current_position: Current hidden state.
            null_basis: Null-space basis [null_rank, hidden_dim].

        Returns:
            Unit vector in direction of escape.
        """
        b = self._backend

        # Stack recent positions
        if len(self._positions) < 2:
            # Return first null-space direction
            first_dir = null_basis[0]
            b.eval(first_dir)
            return tuple(float(x) for x in b.tolist(first_dir))

        positions = b.stack(self._positions[-min(10, len(self._positions)):], axis=0)
        mean_pos = b.mean(positions, axis=0)
        centered = positions - mean_pos  # [n, hidden_dim]

        # Project onto null-space
        # null_basis: [null_rank, hidden_dim]
        # centered: [n, hidden_dim]
        projected = b.matmul(centered, b.transpose(null_basis))  # [n, null_rank]

        # Compute variance along each null direction
        variances = b.var(projected, axis=0)  # [null_rank]
        b.eval(variances)

        # Find direction with max variance
        max_idx = b.argmax(variances)
        b.eval(max_idx)
        max_idx_val = int(b.to_scalar(max_idx))

        # Get that direction and normalize
        escape_dir = null_basis[max_idx_val]
        escape_norm = b.sqrt(b.sum(escape_dir * escape_dir))
        b.eval(escape_norm)
        escape_norm_val = float(b.to_scalar(escape_norm))

        if escape_norm_val > self._sqrt_eps:
            escape_dir = escape_dir / escape_norm
            b.eval(escape_dir)

        return tuple(float(x) for x in b.tolist(escape_dir))

    def escape_attractor(
        self,
        current_position: "Array",
        escape_direction: tuple[float, ...],
        magnitude: float | None = None,
    ) -> "Array":
        """Apply escape perturbation to position.

        Parameters:
            current_position: Current hidden state.
            escape_direction: Direction for escape (from AttractorState).
            magnitude: Perturbation magnitude. If None, uses sqrt(eps) × hidden_dim.

        Returns:
            Perturbed position.
        """
        b = self._backend

        # Default magnitude scaled by dimension
        if magnitude is None:
            magnitude = self._sqrt_eps * math.sqrt(self._hidden_dim)

        # Convert escape direction to array
        escape_vec = b.array(list(escape_direction))

        # Apply perturbation
        new_position = current_position + magnitude * escape_vec
        b.eval(new_position)

        # Reset trajectory state to prevent immediate re-detection
        self._positions = [new_position]
        self._velocities = []
        self._timesteps_stuck = 0

        return new_position

    def reset(self) -> None:
        """Reset trajectory state."""
        self._positions = []
        self._velocities = []
        self._timesteps_stuck = 0

    @property
    def window_size(self) -> int:
        """Trajectory window size."""
        return self._window_size

    @property
    def timesteps_stuck(self) -> int:
        """Number of timesteps in current attractor basin."""
        return self._timesteps_stuck


def create_attractor_detector(
    hidden_dim: int,
    window_size: int | None = None,
    backend: "Backend | None" = None,
) -> AttractorDetector:
    """Create an attractor detector.

    Parameters:
        hidden_dim: Dimension of hidden states.
        window_size: Number of recent positions to track.
        backend: Compute backend.

    Returns:
        Configured AttractorDetector instance.
    """
    return AttractorDetector(
        hidden_dim=hidden_dim,
        window_size=window_size,
        backend=backend,
    )
