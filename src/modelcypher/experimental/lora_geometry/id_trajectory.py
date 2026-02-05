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

"""Intrinsic dimension trajectory measurement during LoRA training.

Tracks how intrinsic dimension evolves during training to understand
the geometric dynamics of adapter convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
    TwoNNEstimate,
)
from modelcypher.experimental.lora_geometry.statistics import compute_kendall_tau

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class IDTrajectoryPoint:
    """A single point in the ID trajectory.

    Attributes:
        epoch: Training epoch (0-indexed).
        intrinsic_dimension: Estimated ID at this epoch.
        ci_lower: Lower bound of CI.
        ci_upper: Upper bound of CI.
        usable_count: Number of samples used for estimation.
        step: Optional training step within epoch.
    """

    epoch: int
    intrinsic_dimension: float
    ci_lower: float
    ci_upper: float
    usable_count: int
    step: int | None = None


@dataclass
class IDTrajectory:
    """Complete intrinsic dimension trajectory during training.

    Attributes:
        points: List of trajectory points ordered by epoch.
        kendall_tau: Kendall's tau measuring monotonicity.
        spike_epochs: Epochs where ID exits previous CI band.
        adapter_id: Identifier of the adapter being trained.
        probe_set_size: Number of probes used for ID estimation.
    """

    points: list[IDTrajectoryPoint] = field(default_factory=list)
    kendall_tau: float | None = None
    spike_epochs: list[int] = field(default_factory=list)
    adapter_id: str = ""
    probe_set_size: int = 0

    def add_point(self, point: IDTrajectoryPoint) -> None:
        """Add a trajectory point and update analysis."""
        self.points.append(point)
        self._update_analysis()

    def _update_analysis(self) -> None:
        """Recompute analysis metrics."""
        if len(self.points) < 2:
            self.kendall_tau = None
            self.spike_epochs = []
            return

        # Compute Kendall tau for monotonicity
        epochs = [p.epoch for p in self.points]
        ids = [p.intrinsic_dimension for p in self.points]
        self.kendall_tau = compute_kendall_tau(epochs, ids)

        # Detect spikes: epochs where ID exits previous CI band
        self.spike_epochs = []
        for i in range(1, len(self.points)):
            prev = self.points[i - 1]
            curr = self.points[i]

            # Check if current ID is outside previous CI
            if curr.intrinsic_dimension < prev.ci_lower:
                self.spike_epochs.append(curr.epoch)
            elif curr.intrinsic_dimension > prev.ci_upper:
                self.spike_epochs.append(curr.epoch)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "adapter_id": self.adapter_id,
            "probe_set_size": self.probe_set_size,
            "kendall_tau": self.kendall_tau,
            "spike_epochs": self.spike_epochs,
            "trajectory": [
                {
                    "epoch": p.epoch,
                    "step": p.step,
                    "id": p.intrinsic_dimension,
                    "ci_lower": p.ci_lower,
                    "ci_upper": p.ci_upper,
                    "usable_count": p.usable_count,
                }
                for p in self.points
            ],
        }


def measure_id_at_checkpoint(
    activations: "Array",
    epoch: int,
    step: int | None = None,
    use_convergence: bool = True,
    backend: "Backend | None" = None,
) -> IDTrajectoryPoint:
    """Measure intrinsic dimension at a training checkpoint.

    Args:
        activations: Probe activations at this checkpoint [n_probes, hidden_dim].
        epoch: Current epoch number.
        step: Optional step within epoch.
        use_convergence: Whether to use compute_with_convergence().
        backend: Compute backend.

    Returns:
        IDTrajectoryPoint with dimension estimate and CI.
    """
    if backend is None:
        backend = get_default_backend()

    estimator = IntrinsicDimension(backend)

    if use_convergence:
        estimate = estimator.compute_with_convergence(activations, with_ci=True)
    else:
        estimate = estimator.compute(activations, with_ci=True)

    ci_lower = estimate.ci.lower if estimate.ci else estimate.intrinsic_dimension
    ci_upper = estimate.ci.upper if estimate.ci else estimate.intrinsic_dimension

    return IDTrajectoryPoint(
        epoch=epoch,
        intrinsic_dimension=estimate.intrinsic_dimension,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        usable_count=estimate.usable_count,
        step=step,
    )


class IDTrajectoryTracker:
    """Tracker for intrinsic dimension during training.

    Usage:
        tracker = IDTrajectoryTracker(probe_activations, adapter_id)

        # During training loop:
        for epoch in range(n_epochs):
            train_one_epoch(model, adapter)
            activations = collect_activations(model, probes)
            tracker.record(epoch, activations)

        trajectory = tracker.get_trajectory()
    """

    def __init__(
        self,
        probe_set_size: int,
        adapter_id: str = "",
        use_convergence: bool = True,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize tracker.

        Args:
            probe_set_size: Number of probes to use (for metadata).
            adapter_id: Identifier of the adapter.
            use_convergence: Whether to use compute_with_convergence().
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._use_convergence = use_convergence
        self._trajectory = IDTrajectory(
            adapter_id=adapter_id,
            probe_set_size=probe_set_size,
        )

    def record(
        self,
        epoch: int,
        activations: "Array",
        step: int | None = None,
    ) -> IDTrajectoryPoint:
        """Record ID at a checkpoint.

        Args:
            epoch: Current epoch.
            activations: Probe activations [n_probes, hidden_dim].
            step: Optional step within epoch.

        Returns:
            The recorded trajectory point.
        """
        point = measure_id_at_checkpoint(
            activations=activations,
            epoch=epoch,
            step=step,
            use_convergence=self._use_convergence,
            backend=self._backend,
        )
        self._trajectory.add_point(point)
        return point

    def get_trajectory(self) -> IDTrajectory:
        """Get the complete trajectory."""
        return self._trajectory

    def is_converging(self, window: int = 3, threshold: float = 0.1) -> bool:
        """Check if ID trajectory is converging (stabilizing).

        Args:
            window: Number of recent points to consider.
            threshold: Max relative change for convergence.

        Returns:
            True if recent changes are below threshold.
        """
        if len(self._trajectory.points) < window:
            return False

        recent = self._trajectory.points[-window:]
        ids = [p.intrinsic_dimension for p in recent]

        # Check relative changes
        for i in range(1, len(ids)):
            if ids[i - 1] > 0:
                rel_change = abs(ids[i] - ids[i - 1]) / ids[i - 1]
                if rel_change > threshold:
                    return False

        return True
