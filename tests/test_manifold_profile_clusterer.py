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

from __future__ import annotations

from uuid import uuid4

from modelcypher.core.domain.geometry.manifold_clusterer import Configuration, ManifoldClusterer
from modelcypher.core.domain.geometry.manifold_profile import ManifoldPoint, ManifoldRegion


def _make_point(
    mean_entropy: float, mean_gate_confidence: float, prompt_hash: str
) -> ManifoldPoint:
    return ManifoldPoint(
        id=uuid4(),
        mean_entropy=mean_entropy,
        entropy_variance=0.1,
        first_token_entropy=mean_entropy,
        gate_count=2,
        mean_gate_confidence=mean_gate_confidence,
        dominant_gate_category=0.0,
        entropy_path_correlation=0.0,
        assessment_strength=0.5,
        prompt_hash=prompt_hash,
    )


def test_region_classification() -> None:
    point = ManifoldPoint(
        id=uuid4(),
        mean_entropy=1.0,
        entropy_variance=0.1,
        first_token_entropy=1.0,
        gate_count=1,
        mean_gate_confidence=0.9,
        dominant_gate_category=0.0,
        entropy_path_correlation=0.0,
        assessment_strength=0.5,
        prompt_hash="p",
    )
    assert ManifoldRegion.classify(point) == ManifoldRegion.RegionType.safe


def test_clusterer_groups_identical_points() -> None:
    points = []
    for i in range(5):
        points.append(
            ManifoldPoint(
                id=uuid4(),
                mean_entropy=1.0,
                entropy_variance=0.1,
                first_token_entropy=1.0,
                gate_count=1,
                mean_gate_confidence=0.9,
                dominant_gate_category=0.0,
                entropy_path_correlation=0.0,
                assessment_strength=0.5,
                prompt_hash=f"p{i}",
            )
        )

    clusterer = ManifoldClusterer(configuration=Configuration(epsilon=0.1, min_points=3))
    result = clusterer.cluster(points)
    assert len(result.regions) == 1
    assert result.regions[0].member_count == 5
    assert result.noise_points == []
