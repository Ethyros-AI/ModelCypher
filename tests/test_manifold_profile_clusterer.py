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
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_profile import (
    ManifoldPoint,
    ManifoldRegion,
    RegionClassificationConfig,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
)


def _make_point(
    mean_entropy: float, mean_gate_similarity: float, prompt_hash: str
) -> ManifoldPoint:
    return ManifoldPoint(
        id=uuid4(),
        mean_entropy=mean_entropy,
        entropy_variance=0.1,
        first_token_entropy=mean_entropy,
        gate_count=2,
        mean_gate_similarity=mean_gate_similarity,
        dominant_gate_category=0.0,
        entropy_path_correlation=0.0,
        assessment_strength=0.5,
        prompt_hash=prompt_hash,
    )


def test_region_classification() -> None:
    backend = get_default_backend()
    entropies = [1.0, 1.1, 5.0, 5.2]
    variances = [0.05, 0.06, 0.5, 0.6]
    coherences = [0.9, 0.88, 0.2, 0.25]
    eps = division_epsilon(backend, backend.array(entropies))
    entropy_threshold = find_magnitude_gap_threshold(sorted(entropies), eps=eps)
    variance_threshold = find_magnitude_gap_threshold(sorted(variances), eps=eps)
    coherence_threshold = find_magnitude_gap_threshold(sorted(coherences), eps=eps)
    entropy_low = max(v for v in entropies if v <= entropy_threshold)
    entropy_high = min(v for v in entropies if v > entropy_threshold)
    variance_low = max(v for v in variances if v <= variance_threshold)
    variance_high = min(v for v in variances if v > variance_threshold)
    coherence_low = max(v for v in coherences if v <= coherence_threshold)
    coherence_high = min(v for v in coherences if v > coherence_threshold)
    config = RegionClassificationConfig(
        low_entropy=entropy_low,
        high_entropy=entropy_high,
        low_variance=variance_low,
        high_variance=variance_high,
        high_coherence=coherence_high,
        low_coherence=coherence_low,
    )
    point = ManifoldPoint(
        id=uuid4(),
        mean_entropy=min(entropies),
        entropy_variance=min(variances),
        first_token_entropy=1.0,
        gate_count=1,
        mean_gate_similarity=max(coherences),
        dominant_gate_category=0.0,
        entropy_path_correlation=0.0,
        assessment_strength=0.5,
        prompt_hash="p",
    )
    assert ManifoldRegion.classify(point, config=config) == ManifoldRegion.RegionCharacter.DENSE


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
                mean_gate_similarity=0.9,
                dominant_gate_category=0.0,
                entropy_path_correlation=0.0,
                assessment_strength=0.5,
                prompt_hash=f"p{i}",
            )
        )

    clusterer = ManifoldClusterer(configuration=Configuration())
    result = clusterer.cluster(points)
    assert len(result.regions) == 1
    assert result.regions[0].member_count == 5
    assert result.noise_points == []
