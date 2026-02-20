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

"""Dimensional expansion/contraction monitor for training loops.

Tracks intrinsic dimension trajectories per epoch to detect whether
training is causing the model to expand (learn richer structure) or
contract (collapse to template retrieval).

Uses existing TwoNN infrastructure from ``intrinsic_dimension.py``
and ``id_trajectory_analysis.py``.

Pure Python + Backend protocol. No framework imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.id_trajectory_analysis import (
    IDTrajectoryAnalysis,
    analyze_id_trajectory,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class DimensionalSnapshot:
    """Per-epoch snapshot of intrinsic dimension trajectory."""

    epoch: int
    expansion_ratio: float
    peak_dim: float
    final_dim: float
    peak_layer: int
    smoothness: float
    hidden_dim: int = 0

    @property
    def final_used_fraction(self) -> float:
        """Fraction of final-layer ambient space used by the manifold."""
        if self.hidden_dim <= 0:
            return float("nan")
        return self.final_dim / float(self.hidden_dim)

    @property
    def final_null_fraction(self) -> float:
        """Fraction of final-layer ambient space still available (null space)."""
        used = self.final_used_fraction
        if used != used:
            return float("nan")
        return 1.0 - used


@dataclass(frozen=True)
class DimensionalTrend:
    """Trend analysis over multiple snapshots."""

    baseline_expansion_ratio: float
    current_expansion_ratio: float
    delta: float  # current - baseline
    is_contracting: bool  # significant contraction detected


def compute_expansion_from_activations(
    layer_activations: dict[int, "Array"],
    backend: "Backend",
    epoch: int,
) -> DimensionalSnapshot | None:
    """Compute dimensional snapshot from per-layer activations.

    Parameters
    ----------
    layer_activations
        ``{layer_idx: Array[batch, seq, hidden]}`` from
        ``backend.collect_hidden_activations()``.
    backend
        Backend protocol for array ops.
    epoch
        Current epoch number.

    Returns
    -------
    DimensionalSnapshot or None if computation fails.
    """
    if not layer_activations:
        return None

    # Reshape each layer: [batch, seq, hidden] → [batch*seq, hidden]
    sorted_layers = sorted(layer_activations.keys())
    point_clouds: list["Array"] = []

    hidden_dims: dict[int, int] = {}
    for layer_idx in sorted_layers:
        act = layer_activations[layer_idx]  # [batch, seq, hidden]
        shape = act.shape
        if len(shape) == 3:
            batch, seq, hidden = shape
            reshaped = backend.reshape(act, (batch * seq, hidden))
            hidden_dims[layer_idx] = int(hidden)
        elif len(shape) == 2:
            reshaped = act
            hidden_dims[layer_idx] = int(shape[1])
        else:
            return None
        point_clouds.append(reshaped)

    # Batch TwoNN estimation across all layers
    id_estimator = IntrinsicDimension(backend)
    estimates = id_estimator.batch_compute(point_clouds)

    # Extract ID trajectory
    id_trajectory: list[float] = []
    for est in estimates:
        if est is not None:
            id_trajectory.append(est.intrinsic_dimension)
        else:
            id_trajectory.append(float("nan"))

    if not any(d == d for d in id_trajectory):  # all NaN
        return None

    analysis: IDTrajectoryAnalysis = analyze_id_trajectory(id_trajectory)

    final_layer = sorted_layers[-1]
    final_hidden_dim = hidden_dims.get(final_layer, 0)

    return DimensionalSnapshot(
        epoch=epoch,
        expansion_ratio=analysis.expansion_ratio,
        peak_dim=analysis.peak_dim,
        final_dim=analysis.final_dim,
        peak_layer=analysis.peak_layer,
        smoothness=analysis.smoothness,
        hidden_dim=final_hidden_dim,
    )


def compute_null_space_recruitment(
    baseline: DimensionalSnapshot,
    current: DimensionalSnapshot,
) -> float:
    """Measure null-space recruitment as change in used fraction.

    Positive values mean training recruited additional dimensions into use.
    Negative values mean contraction back into fewer used dimensions.
    """
    baseline_used = baseline.final_used_fraction
    current_used = current.final_used_fraction
    if baseline_used != baseline_used or current_used != current_used:
        return float("nan")
    return current_used - baseline_used


def assess_trend(
    snapshots: list[DimensionalSnapshot],
) -> DimensionalTrend:
    """Assess expansion/contraction trend from snapshot history.

    Uses the first snapshot as baseline. Detects contraction when current
    expansion_ratio drops below baseline by more than 1 SE of the series.

    Parameters
    ----------
    snapshots
        Chronologically ordered list of snapshots (len >= 2).

    Returns
    -------
    DimensionalTrend
    """
    if len(snapshots) < 2:
        er = snapshots[0].expansion_ratio if snapshots else 0.0
        return DimensionalTrend(
            baseline_expansion_ratio=er,
            current_expansion_ratio=er,
            delta=0.0,
            is_contracting=False,
        )

    baseline_er = snapshots[0].expansion_ratio
    current_er = snapshots[-1].expansion_ratio
    delta = current_er - baseline_er

    # Standard error of expansion_ratio across history
    ers = [s.expansion_ratio for s in snapshots if s.expansion_ratio == s.expansion_ratio]
    if len(ers) >= 2:
        mean_er = sum(ers) / len(ers)
        variance = sum((e - mean_er) ** 2 for e in ers) / (len(ers) - 1)
        se = math.sqrt(variance / len(ers))
    else:
        se = 0.0

    # Contraction: current dropped below baseline by more than 1 SE
    is_contracting = delta < -se if se > 0 else delta < 0

    return DimensionalTrend(
        baseline_expansion_ratio=baseline_er,
        current_expansion_ratio=current_er,
        delta=delta,
        is_contracting=is_contracting,
    )
