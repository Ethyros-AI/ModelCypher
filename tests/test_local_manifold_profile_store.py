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

from datetime import datetime, timedelta
from uuid import uuid4

from modelcypher.adapters.local_manifold_profile_store import (
    LocalManifoldProfileStore,
    ManifoldProfilePaths,
)
from modelcypher.core.domain.geometry.manifold_profile import ManifoldPoint, ManifoldProfile


def _make_point(prompt_hash: str = "prompt") -> ManifoldPoint:
    return ManifoldPoint(
        id=uuid4(),
        mean_entropy=1.2,
        entropy_variance=0.4,
        first_token_entropy=0.9,
        gate_count=4,
        mean_gate_confidence=0.8,
        dominant_gate_category=0.1,
        entropy_path_correlation=0.0,
        assessment_strength=0.5,
        prompt_hash=prompt_hash,
    )


def test_store_save_load_list(tmp_path) -> None:
    paths = ManifoldProfilePaths(base_path=tmp_path)
    store = LocalManifoldProfileStore(paths)

    point = _make_point()
    profile = ManifoldProfile(
        id=uuid4(),
        model_id="model-a",
        model_name="Model A",
        recent_points=[point],
        total_point_count=1,
        updated_at=datetime.utcnow() - timedelta(hours=1),
    )
    store.save(profile)

    later_profile = ManifoldProfile(
        id=uuid4(),
        model_id="model-b",
        model_name="Model B",
        recent_points=[_make_point(prompt_hash="later")],
        total_point_count=1,
        updated_at=datetime.utcnow(),
    )
    store.save(later_profile)

    loaded = store.load("model-a")
    assert loaded is not None
    assert loaded.model_id == "model-a"
    assert loaded.total_point_count == 1

    profiles = store.list()
    assert [item.model_id for item in profiles] == ["model-b", "model-a"]
    assert paths.index.exists()


def test_store_add_point_and_statistics(tmp_path) -> None:
    store = LocalManifoldProfileStore(ManifoldProfilePaths(base_path=tmp_path))
    store.add_point(_make_point(), model_id="model-c", model_name="Model C")
    profile = store.load("model-c")
    assert profile is not None
    assert profile.total_point_count == 1

    stats = store.get_statistics("model-c")
    assert stats is not None
    assert stats.total_points == 1
