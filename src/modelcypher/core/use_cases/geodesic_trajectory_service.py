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

"""Geodesic trajectory analysis for Chain-of-Thought reasoning.

Experimental. Measures whether token-by-token activation trajectories
follow geodesics on the activation manifold or cut through voids.

Pipeline: generate text -> collect per-token trajectories -> compute
geodesic vs euclidean distances between consecutive tokens -> report
deviation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class GeodesicTrajectoryResult:
    """Geodesic trajectory analysis for a generation."""

    token_count: int
    layer_analyzed: int

    # Per-consecutive-step geodesic deviation: 0.0 = straight line, >0 = curved
    step_deviations: list[float]

    # Summary statistics
    mean_deviation: float
    max_deviation: float

    # Intrinsic dimension of the trajectory point cloud
    intrinsic_dimension: float

    # Total geodesic path length vs straight-line (first->last) distance
    # 1.0 = straight line, >1.0 = curved path
    path_length_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "layer_analyzed": self.layer_analyzed,
            "step_deviations": self.step_deviations,
            "mean_deviation": self.mean_deviation,
            "max_deviation": self.max_deviation,
            "intrinsic_dimension": self.intrinsic_dimension,
            "path_length_ratio": self.path_length_ratio,
        }


class GeodesicTrajectoryService:
    """Measure geodesic properties of activation trajectories.

    Experimental. Does Chain-of-Thought trace geodesics on the activation
    manifold, or cut through topological voids?
    """

    def __init__(
        self,
        backend: "Backend",
        activation_provider: "ActivationProvider",
    ) -> None:
        self._backend = backend
        self._activation_provider = activation_provider

    def measure(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        target_layer: int | None = None,
    ) -> GeodesicTrajectoryResult:
        """Measure geodesic properties of a text's activation trajectory.

        Args:
            model: Loaded model.
            tokenizer: Tokenizer for the model.
            text: Full text to analyze (prompt + response).
            target_layer: Layer to analyze. Defaults to last layer.

        Returns:
            GeodesicTrajectoryResult with per-step deviations and summary.
        """
        b = self._backend

        # 1. Collect trajectory via activation provider
        trajectory = self._activation_provider.collect_trajectory_batch(
            model, tokenizer, [text]
        )

        # 2. Select target layer
        available_layers = sorted(trajectory.positions.keys())
        if not available_layers:
            raise ValueError("No trajectory positions collected")

        if target_layer is None:
            target_layer = available_layers[-1]
        elif target_layer not in trajectory.positions:
            raise ValueError(
                f"Layer {target_layer} not in trajectory. "
                f"Available: {available_layers}"
            )

        positions = trajectory.positions[target_layer]  # [n_tokens, hidden_dim]
        n_tokens = trajectory.total_tokens

        if n_tokens < 3:
            raise ValueError(
                f"Need >= 3 tokens for geodesic analysis, got {n_tokens}"
            )

        # 3. Compute geodesic distance matrix
        rg = RiemannianGeometry(backend=b)
        geo_result = rg.geodesic_distances(positions)
        geo_dist = geo_result.distances  # [n, n]

        # 4. Compute euclidean distances between consecutive tokens
        step_deviations: list[float] = []
        geodesic_steps: list[float] = []

        for t in range(n_tokens - 1):
            # Euclidean distance between consecutive tokens
            diff = b.subtract(positions[t + 1 : t + 2], positions[t : t + 1])
            eucl = b.to_scalar(b.sqrt(b.sum(b.multiply(diff, diff))))

            # Geodesic distance between consecutive tokens
            geo = b.to_scalar(geo_dist[t][t + 1])

            geodesic_steps.append(geo)

            # Deviation: how much longer the geodesic is vs euclidean
            if eucl > 0:
                deviation = (geo / eucl) - 1.0
            else:
                deviation = 0.0
            step_deviations.append(deviation)

        # 5. Path length ratio: sum of geodesic steps / geodesic(first, last)
        total_path = sum(geodesic_steps)
        endpoint_geo = b.to_scalar(geo_dist[0][n_tokens - 1])

        if endpoint_geo > 0:
            path_length_ratio = total_path / endpoint_geo
        else:
            path_length_ratio = 1.0

        # 6. Intrinsic dimension of the trajectory point cloud
        dim_est = IntrinsicDimension(backend=b)
        two_nn = dim_est.compute(positions)
        intrinsic_dim = two_nn.intrinsic_dimension

        # 7. Summary
        mean_dev = sum(step_deviations) / len(step_deviations) if step_deviations else 0.0
        max_dev = max(step_deviations) if step_deviations else 0.0

        return GeodesicTrajectoryResult(
            token_count=n_tokens,
            layer_analyzed=target_layer,
            step_deviations=step_deviations,
            mean_deviation=mean_dev,
            max_deviation=max_dev,
            intrinsic_dimension=intrinsic_dim,
            path_length_ratio=path_length_ratio,
        )
