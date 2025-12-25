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

from pathlib import Path

from modelcypher.core.domain.geometry.refusal_direction_cache import RefusalDirectionCache
from modelcypher.core.domain.geometry.refusal_direction_detector import (
    Configuration,
    RefusalDirectionDetector,
)


def test_refusal_direction_compute_and_distance() -> None:
    harmful = [[1.0, 0.0], [1.0, 0.0]]
    harmless = [[0.0, 1.0], [0.0, 1.0]]
    config = Configuration(activation_difference_threshold=0.01, normalize_direction=True)
    direction = RefusalDirectionDetector.compute_direction(
        harmful_activations=harmful,
        harmless_activations=harmless,
        configuration=config,
        layer_index=3,
        model_id="model-x",
    )
    assert direction is not None
    assert direction.hidden_size == 2
    assert direction.strength > 0

    metrics = RefusalDirectionDetector.measure_distance(
        activation=[1.0, -1.0],
        refusal_direction=direction,
        previous_projection=None,
        token_index=0,
    )
    assert metrics is not None
    # Activation [1,-1] is parallel to refusal direction [1,-1] (from harmful-harmless)
    # Distance should be ~0 (on the direction line)
    # Projection should be positive (same direction, not opposite)
    assert abs(metrics.distance_to_refusal) < 1e-6
    assert metrics.projection_magnitude > 0  # Same direction as refusal, not opposite


def test_refusal_direction_cache_roundtrip(tmp_path) -> None:
    harmful = [[1.0, 0.0]]
    harmless = [[0.0, 1.0]]
    config = Configuration(activation_difference_threshold=0.01, normalize_direction=True)
    direction = RefusalDirectionDetector.compute_direction(
        harmful_activations=harmful,
        harmless_activations=harmless,
        configuration=config,
        layer_index=1,
        model_id="model-cache",
    )
    assert direction is not None

    cache_dir = tmp_path / "cache"
    cache = RefusalDirectionCache(cache_directory=cache_dir)
    cache.save(direction, model_path=Path("/models/model-cache"))
    loaded = cache.load(model_path=Path("/models/model-cache"))
    assert loaded is not None
    assert loaded.model_id == "model-cache"
