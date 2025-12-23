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
    assert abs(metrics.distance_to_refusal) < 1e-6
    assert metrics.assessment.value == "likelyToRefuse"


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
