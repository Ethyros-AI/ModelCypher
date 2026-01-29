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

"""Deprecated multi-scale perturbation system.

Multi-scale exploration relies on heuristic schedules and patience counters.
This contradicts the "no heuristics" constraint. Use geometry-derived scaling
via compute_geometry_derived_scale instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@dataclass
class ScaleState:
    """Track the state of multi-scale exploration."""

    # Current scale index in the schedule
    current_scale_idx: int = 0

    # Number of rounds at current scale without improvement
    rounds_without_improvement: int = 0

    # Best entropy seen at current scale
    best_entropy_at_scale: float = float("inf")

    # History of (scale, entropy_delta) for adaptive scheduling
    scale_history: List[Tuple[float, float]] = field(default_factory=list)

    # Number of times we've cycled through all scales
    cycle_count: int = 0

    # Whether we're in "escape mode" (large scales)
    escape_mode: bool = False


class MultiScalePerturbation:
    """Adaptive multi-scale perturbation for escaping local minima.

    Perturbation scales follow a geometric progression:
    - Finest: sqrt(machine_epsilon) ≈ 3e-4 for float32
    - Coarsest: 0.1 (aggressive restructuring)

    The system:
    1. Starts at medium scale
    2. When stuck, tries larger scales (escape mode)
    3. When improving, tries finer scales (optimization mode)
    4. Tracks which scales work best for each layer

    Deprecated: use compute_geometry_derived_scale instead.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        n_scales: int = 7,
        patience_per_scale: int = 3,
        min_scale: Optional[float] = None,
        max_scale: float = 0.1,
    ) -> None:
        """Initialize multi-scale perturbation.

        Args:
            backend: Computational backend
            n_scales: Number of scales in the schedule
            patience_per_scale: Rounds without improvement before changing scale
            min_scale: Minimum scale (default: sqrt(machine_epsilon))
            max_scale: Maximum scale for aggressive exploration
        """
        raise RuntimeError(
            "MultiScalePerturbation is heuristic. "
            "Use compute_geometry_derived_scale for geometry-derived scaling."
        )
        self._backend = backend or get_default_backend()
        self._n_scales = n_scales
        self._patience = patience_per_scale

        # Compute scale schedule
        # Using sqrt(eps) as minimum for numerical stability
        if min_scale is None:
            # sqrt(1e-7) ≈ 3.16e-4 for float32
            min_scale = math.sqrt(1e-7)

        # Geometric progression from min to max
        # scales[i] = min_scale * (max_scale/min_scale)^(i/(n-1))
        ratio = max_scale / min_scale
        self._scales = [
            min_scale * (ratio ** (i / (n_scales - 1)))
            for i in range(n_scales)
        ]

        # Start in the middle of the scale range
        self._state = ScaleState(current_scale_idx=n_scales // 2)

    @property
    def scales(self) -> List[float]:
        """Get the scale schedule."""
        return self._scales.copy()

    @property
    def current_scale(self) -> float:
        """Get the current perturbation scale."""
        return self._scales[self._state.current_scale_idx]

    @property
    def state(self) -> ScaleState:
        """Get the current state."""
        return self._state

    def get_current_scale(self) -> float:
        """Get the current perturbation scale."""
        return self.current_scale

    def get_exploration_scales(self, n: int = 3) -> List[float]:
        """Get multiple scales for parallel exploration.

        Returns scales around the current one for efficient search.

        Args:
            n: Number of scales to return (centered on current)

        Returns:
            List of scales to try in parallel
        """
        idx = self._state.current_scale_idx
        half = n // 2

        start = max(0, idx - half)
        end = min(self._n_scales, start + n)
        start = max(0, end - n)  # Adjust if we hit the end

        return [self._scales[i] for i in range(start, end)]

    def update(self, entropy_delta: float, entropy_after: float) -> None:
        """Update state based on round result.

        Args:
            entropy_delta: Entropy reduction (positive = improvement)
            entropy_after: Entropy after applying direction
        """
        scale = self.current_scale
        self._state.scale_history.append((scale, entropy_delta))

        # Track improvement at this scale
        if entropy_delta > 0:
            self._state.rounds_without_improvement = 0

            # Update best entropy at this scale
            if entropy_after < self._state.best_entropy_at_scale:
                self._state.best_entropy_at_scale = entropy_after

            # If improving, try finer scales (optimization mode)
            if self._state.current_scale_idx > 0:
                self._move_to_scale(self._state.current_scale_idx - 1)
            self._state.escape_mode = False
        else:
            self._state.rounds_without_improvement += 1

            # If stuck at this scale, try larger scales (escape mode)
            if self._state.rounds_without_improvement >= self._patience:
                self._try_escape()

    def _move_to_scale(self, idx: int) -> None:
        """Move to a new scale index."""
        self._state.current_scale_idx = max(0, min(self._n_scales - 1, idx))
        self._state.rounds_without_improvement = 0
        self._state.best_entropy_at_scale = float("inf")

    def _try_escape(self) -> None:
        """Try to escape local minimum by using larger scales."""
        if self._state.current_scale_idx < self._n_scales - 1:
            # Move to larger scale
            self._move_to_scale(self._state.current_scale_idx + 1)
            self._state.escape_mode = True
        else:
            # Already at largest scale, cycle back through
            self._state.cycle_count += 1
            self._move_to_scale(self._n_scales // 2)  # Back to middle
            self._state.escape_mode = False

    def reset(self) -> None:
        """Reset state for a new alignment run."""
        self._state = ScaleState(current_scale_idx=self._n_scales // 2)

    def get_best_scale_for_layer(self, layer_idx: int) -> float:
        """Get the scale that worked best historically for a layer.

        This is a simple heuristic - in the future we could track
        per-layer performance separately.

        Args:
            layer_idx: Layer index (currently unused)

        Returns:
            Best performing scale based on history
        """
        if not self._state.scale_history:
            return self.current_scale

        # Group by scale, find scale with best average improvement
        scale_to_deltas: dict[float, list[float]] = {}
        for scale, delta in self._state.scale_history:
            if scale not in scale_to_deltas:
                scale_to_deltas[scale] = []
            scale_to_deltas[scale].append(delta)

        best_scale = self.current_scale
        best_avg = float("-inf")

        for scale, deltas in scale_to_deltas.items():
            avg = sum(deltas) / len(deltas)
            if avg > best_avg:
                best_avg = avg
                best_scale = scale

        return best_scale

    def should_continue(self, max_cycles: int = 3) -> bool:
        """Check if we should continue exploring.

        Args:
            max_cycles: Maximum complete cycles through scales

        Returns:
            True if we haven't exhausted exploration
        """
        return self._state.cycle_count < max_cycles

    def get_status(self) -> str:
        """Get human-readable status."""
        mode = "escape" if self._state.escape_mode else "optimize"
        return (
            f"Scale: {self.current_scale:.2e} ({mode} mode), "
            f"Rounds stuck: {self._state.rounds_without_improvement}, "
            f"Cycles: {self._state.cycle_count}"
        )


__all__ = [
    "MultiScalePerturbation",
    "ScaleState",
]
