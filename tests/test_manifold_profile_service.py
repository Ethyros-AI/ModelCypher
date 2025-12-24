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

from modelcypher.adapters.local_manifold_profile_store import LocalManifoldProfileStore, ManifoldProfilePaths
from modelcypher.core.domain.geometry.manifold_clusterer import Configuration as ClustererConfiguration
from modelcypher.core.domain.geometry.manifold_profile import ManifoldPoint
from modelcypher.core.use_cases.manifold_profile_service import ManifoldProfileService


def _safe_point(prompt_hash: str = "prompt") -> ManifoldPoint:
    return ManifoldPoint(
        id=uuid4(),
        mean_entropy=1.0,
        entropy_variance=0.1,
        first_token_entropy=0.8,
        gate_count=5,
        mean_gate_confidence=0.9,
        dominant_gate_category=0.1,
        entropy_path_correlation=0.0,
        assessment_strength=0.5,
        prompt_hash=prompt_hash,
    )


def test_service_clustering_and_intervention(tmp_path) -> None:
    store = LocalManifoldProfileStore(ManifoldProfilePaths(base_path=tmp_path))
    config = ManifoldProfileService.Configuration(
        clustering_threshold=1,
        clusterer_config=ClustererConfiguration(epsilon=1.0, min_points=1, compute_intrinsic_dimension=False),
    )
    service = ManifoldProfileService(store=store, configuration=config)

    point = _safe_point()
    service.record_point(point, model_id="model-safe", model_name="Model Safe")
    service.flush_pending_points(model_id="model-safe", model_name="Model Safe")

    profile = service.get_profile("model-safe")
    assert profile is not None
    assert len(profile.regions) == 1

    suggestion = service.suggest_intervention(point, model_id="model-safe")
    assert suggestion.level == 0
    assert "safe region" in suggestion.reason

    report = service.generate_report("model-safe")
    assert "Manifold Profile Report" in report
