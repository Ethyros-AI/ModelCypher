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

"""Scalar summary extraction from intrinsic dimension trajectories.

Pure Python — depends only on ``statistics.spearman_correlation``.

Extracts expansion_ratio, peak layer, and smoothness from a per-layer
intrinsic dimension trajectory.

Extracted from rl_geometric_attractor_experiment.py and
rl_processing_geometry_experiment.py where this logic was duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.statistics import spearman_correlation


@dataclass(frozen=True)
class IDTrajectoryAnalysis:
    """Scalar summary of an intrinsic dimension trajectory."""

    expansion_ratio: float
    peak_layer: int
    peak_dim: float
    final_dim: float
    smoothness: float  # Spearman(layer_order, id_values)


def analyze_id_trajectory(id_trajectory: list[float]) -> IDTrajectoryAnalysis:
    """Extract scalar summary from an intrinsic dimension trajectory.

    Parameters
    ----------
    id_trajectory : list[float]
        Per-layer intrinsic dimension values. May contain NaN.

    Returns
    -------
    IDTrajectoryAnalysis
        Expansion ratio, peak location, and smoothness.
    """
    valid_ids = [d for d in id_trajectory if d == d and d > 0]  # filter NaN

    if len(valid_ids) >= 2:
        peak_dim = max(valid_ids)
        peak_layer = id_trajectory.index(peak_dim)
        final_dim = valid_ids[-1] if valid_ids[-1] == valid_ids[-1] else valid_ids[-2]
        expansion_ratio = peak_dim / final_dim if final_dim > 0 else float("nan")
        smoothness = spearman_correlation(
            list(range(len(id_trajectory))), id_trajectory
        )
    else:
        peak_dim = 0.0
        peak_layer = -1
        final_dim = 0.0
        expansion_ratio = float("nan")
        smoothness = 0.0

    return IDTrajectoryAnalysis(
        expansion_ratio=expansion_ratio,
        peak_layer=peak_layer,
        peak_dim=peak_dim,
        final_dim=final_dim,
        smoothness=smoothness,
    )
