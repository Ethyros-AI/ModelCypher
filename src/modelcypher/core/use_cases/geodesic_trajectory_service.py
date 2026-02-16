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

    # Token annotation (when annotate_tokens=True)
    token_labels: list[str] | None = None
    annotated_steps: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "token_count": self.token_count,
            "layer_analyzed": self.layer_analyzed,
            "step_deviations": self.step_deviations,
            "mean_deviation": self.mean_deviation,
            "max_deviation": self.max_deviation,
            "intrinsic_dimension": self.intrinsic_dimension,
            "path_length_ratio": self.path_length_ratio,
        }
        if self.token_labels is not None:
            d["token_labels"] = self.token_labels
        if self.annotated_steps is not None:
            d["annotated_steps"] = self.annotated_steps
        return d


@dataclass
class CategoryGeodesicSummary:
    """Aggregated geodesic metrics for a prompt category."""

    category: str
    prompt_count: int
    mean_deviation: float
    mean_path_length_ratio: float
    mean_intrinsic_dimension: float
    results: list[GeodesicTrajectoryResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "prompt_count": self.prompt_count,
            "mean_deviation": self.mean_deviation,
            "mean_path_length_ratio": self.mean_path_length_ratio,
            "mean_intrinsic_dimension": self.mean_intrinsic_dimension,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class GeodesicComparisonResult:
    """Comparative geodesic analysis across prompt categories."""

    categories: list[CategoryGeodesicSummary]
    model_path: str
    layer_analyzed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "layer_analyzed": self.layer_analyzed,
            "categories": [c.to_dict() for c in self.categories],
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
        annotate_tokens: bool = False,
    ) -> GeodesicTrajectoryResult:
        """Measure geodesic properties of a text's activation trajectory.

        Args:
            model: Loaded model.
            tokenizer: Tokenizer for the model.
            text: Full text to analyze (prompt + response).
            target_layer: Layer to analyze. Defaults to last layer.
            annotate_tokens: If True, include per-token labels and
                annotated step dicts in the result.

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

        # 8. Token annotation (optional)
        token_labels: list[str] | None = None
        annotated_steps: list[dict[str, Any]] | None = None

        if annotate_tokens and tokenizer is not None:
            try:
                token_ids = tokenizer.encode(text)
                if not isinstance(token_ids, list):
                    token_ids = list(token_ids)
                token_labels = [
                    tokenizer.decode([tid]) for tid in token_ids[:n_tokens]
                ]
                annotated_steps = []
                for t in range(len(step_deviations)):
                    annotated_steps.append({
                        "from": token_labels[t] if t < len(token_labels) else "?",
                        "to": token_labels[t + 1] if t + 1 < len(token_labels) else "?",
                        "deviation": step_deviations[t],
                    })
            except Exception:
                logger.debug("Token annotation failed", exc_info=True)

        return GeodesicTrajectoryResult(
            token_count=n_tokens,
            layer_analyzed=target_layer,
            step_deviations=step_deviations,
            mean_deviation=mean_dev,
            max_deviation=max_dev,
            intrinsic_dimension=intrinsic_dim,
            path_length_ratio=path_length_ratio,
            token_labels=token_labels,
            annotated_steps=annotated_steps,
        )

    def measure_batch(
        self,
        model: Any,
        tokenizer: Any,
        categorized_prompts: dict[str, list[str]],
        model_path: str = "",
        target_layer: int | None = None,
        annotate_tokens: bool = False,
    ) -> GeodesicComparisonResult:
        """Run geodesic analysis across categorized prompt sets.

        Args:
            model: Loaded model.
            tokenizer: Tokenizer for the model.
            categorized_prompts: Dict mapping category name to list of prompts.
            model_path: Model path string for the result metadata.
            target_layer: Layer to analyze. Defaults to last layer.
            annotate_tokens: If True, annotate tokens in per-prompt results.

        Returns:
            GeodesicComparisonResult with per-category summaries.
        """
        categories: list[CategoryGeodesicSummary] = []
        resolved_layer = target_layer

        for cat_name, prompts in categorized_prompts.items():
            results: list[GeodesicTrajectoryResult] = []
            for prompt in prompts:
                try:
                    r = self.measure(
                        model, tokenizer, prompt, target_layer, annotate_tokens,
                    )
                    results.append(r)
                    if resolved_layer is None:
                        resolved_layer = r.layer_analyzed
                except Exception:
                    logger.warning(
                        "Skipping prompt in %s: %s", cat_name, prompt[:40],
                        exc_info=True,
                    )

            if results:
                categories.append(CategoryGeodesicSummary(
                    category=cat_name,
                    prompt_count=len(results),
                    mean_deviation=sum(r.mean_deviation for r in results) / len(results),
                    mean_path_length_ratio=sum(r.path_length_ratio for r in results) / len(results),
                    mean_intrinsic_dimension=sum(r.intrinsic_dimension for r in results) / len(results),
                    results=results,
                ))

        return GeodesicComparisonResult(
            categories=categories,
            model_path=model_path,
            layer_analyzed=resolved_layer or 0,
        )
