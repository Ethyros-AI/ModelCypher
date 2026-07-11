"""Token-trajectory contextual curvature from King et al. (2026)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class ContextualCurvatureProfile:
    """Angle and backward-window curvature values for valid token positions."""

    turning_angles_radians: "Array"
    contextual_curvature_radians: "Array"
    token_positions: tuple[int, ...]
    window_size: int


def compute_contextual_curvature(
    positions: "Array",
    *,
    backend: "Backend",
    window_size: int,
) -> ContextualCurvatureProfile:
    """Compute the paper's backward-window mean of adjacent displacement angles.

    For token position ``k``, first differences are ``v_i = x_(i+1) - x_i``
    and turning angles are ``c_i = arccos(<v_(i+1), v_i> / norms)``. The
    contextual value averages ``c_i`` for ``i = k-window_size-1, ..., k-2``.
    King et al. (2026, arXiv:2604.23985, section 2.3) used ``window_size=3``;
    callers pass the value explicitly so the replication choice cannot become
    an implicit product constant.
    """
    if len(positions.shape) != 2:
        raise ValueError("Contextual curvature requires [tokens, hidden_dim] positions")
    if window_size <= 0:
        raise ValueError("Contextual curvature window_size must be positive")

    token_count = int(positions.shape[0])
    first_valid_position = window_size + 1
    if token_count <= first_valid_position:
        raise ValueError(
            "Contextual curvature requires more token positions than window_size + 1"
        )

    velocities = positions[1:] - positions[:-1]
    left = velocities[:-1]
    right = velocities[1:]
    dot = backend.sum(left * right, axis=1)
    left_norm = backend.sqrt(backend.sum(left * left, axis=1))
    right_norm = backend.sqrt(backend.sum(right * right, axis=1))
    denominator = left_norm * right_norm
    eps = division_epsilon(backend, denominator)
    cosine = backend.clip(dot / backend.maximum(denominator, eps), -1.0, 1.0)
    turning_angles = backend.arccos(cosine)

    token_positions = tuple(range(first_valid_position, token_count))
    contextual_values = []
    for token_position in token_positions:
        start = token_position - window_size - 1
        stop = token_position - 1
        contextual_values.append(backend.mean(turning_angles[start:stop]))
    contextual = backend.stack(contextual_values, axis=0)
    backend.eval(turning_angles, contextual)

    return ContextualCurvatureProfile(
        turning_angles_radians=turning_angles,
        contextual_curvature_radians=contextual,
        token_positions=token_positions,
        window_size=window_size,
    )
