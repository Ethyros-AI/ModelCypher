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


# =============================================================================
# Stability Tests
# =============================================================================


class TestRefusalDirectionStability:
    """Tests for stability properties of refusal direction computation."""

    def test_direction_deterministic_with_same_data(self) -> None:
        """Same input data should produce same direction."""
        harmful = [[1.0, 0.0, 0.5], [1.0, 0.1, 0.4]]
        harmless = [[0.0, 1.0, -0.5], [0.1, 0.9, -0.4]]
        config = Configuration(activation_difference_threshold=0.01, normalize_direction=True)

        dir1 = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )
        dir2 = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )

        assert dir1 is not None
        assert dir2 is not None
        # Same direction vector
        for i in range(len(dir1.direction)):
            assert abs(dir1.direction[i] - dir2.direction[i]) < 1e-10

    def test_direction_robust_to_sample_ordering(self) -> None:
        """Direction should be robust to ordering of samples."""
        harmful = [[1.0, 0.0], [1.1, 0.1], [0.9, -0.1]]
        harmless = [[0.0, 1.0], [0.1, 0.9], [-0.1, 1.1]]

        harmful_reordered = [harmful[2], harmful[0], harmful[1]]
        harmless_reordered = [harmless[2], harmless[0], harmless[1]]

        config = Configuration(activation_difference_threshold=0.01, normalize_direction=True)

        dir1 = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )
        dir2 = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful_reordered,
            harmless_activations=harmless_reordered,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )

        assert dir1 is not None
        assert dir2 is not None
        # Directions should be similar (Fréchet mean is order-independent)
        dot = sum(dir1.direction[i] * dir2.direction[i] for i in range(len(dir1.direction)))
        # Either same direction or opposite (|dot| close to 1)
        assert abs(abs(dot) - 1.0) < 0.3


# =============================================================================
# Orthogonality Tests
# =============================================================================


class TestRefusalDirectionOrthogonality:
    """Tests for orthogonality properties."""

    def test_refusal_direction_separates_harmful_harmless(self) -> None:
        """Refusal direction should separate harmful from harmless activations."""
        # Create clear separation in first dimension
        harmful = [[2.0, 0.5], [2.1, 0.6], [1.9, 0.4]]
        harmless = [[-2.0, 0.5], [-1.9, 0.6], [-2.1, 0.4]]

        config = Configuration(activation_difference_threshold=0.01, normalize_direction=True)
        direction = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )

        assert direction is not None

        # Project harmful points onto direction - should be positive
        for h in harmful:
            proj = sum(h[i] * direction.direction[i] for i in range(len(h)))
            assert proj > 0

        # Project harmless points onto direction - should be negative
        for l in harmless:
            proj = sum(l[i] * direction.direction[i] for i in range(len(l)))
            assert proj < 0

    def test_projection_towards_refusal_detected(self) -> None:
        """Moving toward refusal direction should be detected."""
        harmful = [[1.0, 0.0]]
        harmless = [[-1.0, 0.0]]
        config = Configuration(activation_difference_threshold=0.01, normalize_direction=True)
        direction = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )
        assert direction is not None

        # Start at harmless-ish point
        metrics1 = RefusalDirectionDetector.measure_distance(
            activation=[-0.5, 0.0],
            refusal_direction=direction,
            previous_projection=None,
            token_index=0,
        )

        # Move toward harmful
        metrics2 = RefusalDirectionDetector.measure_distance(
            activation=[0.5, 0.0],
            refusal_direction=direction,
            previous_projection=metrics1.projection_magnitude,
            token_index=1,
        )

        # Second projection should be larger (more toward refusal)
        assert metrics2.projection_magnitude > metrics1.projection_magnitude
        # And should be flagged as approaching refusal
        assert metrics2.is_approaching_refusal


# =============================================================================
# Known Example Tests
# =============================================================================


class TestRefusalDirectionKnownExamples:
    """Tests on known synthetic examples with expected behavior."""

    def test_perpendicular_activation_high_distance(self) -> None:
        """Activation perpendicular to refusal direction has high distance."""
        harmful = [[1.0, 0.0]]
        harmless = [[0.0, 0.0]]  # Direction is just [1, 0]
        config = Configuration(activation_difference_threshold=0.001, normalize_direction=True)
        direction = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )
        assert direction is not None

        # Perpendicular activation (0, 1) - along y-axis
        metrics = RefusalDirectionDetector.measure_distance(
            activation=[0.0, 1.0],
            refusal_direction=direction,
            previous_projection=None,
            token_index=0,
        )

        # Distance should be 1.0 (perpendicular to direction)
        assert abs(metrics.distance_to_refusal - 1.0) < 0.1
        # Projection should be near zero
        assert abs(metrics.projection_magnitude) < 0.1

    def test_direction_normalized_when_configured(self) -> None:
        """Direction should be normalized when normalize_direction=True."""
        harmful = [[2.0, 0.0]]
        harmless = [[0.0, 0.0]]
        config = Configuration(activation_difference_threshold=0.001, normalize_direction=True)
        direction = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )
        assert direction is not None

        # Direction magnitude should be 1.0 (normalized)
        magnitude = sum(d * d for d in direction.direction) ** 0.5
        assert abs(magnitude - 1.0) < 1e-6

    def test_direction_strength_reflects_separation(self) -> None:
        """Direction strength should reflect how well it separates."""
        # Well-separated data
        harmful_sep = [[5.0, 0.0], [5.0, 0.0]]
        harmless_sep = [[-5.0, 0.0], [-5.0, 0.0]]

        # Less separated data (but still separable)
        harmful_less = [[2.0, 0.0], [2.0, 0.1]]
        harmless_less = [[-1.0, 0.0], [-1.0, 0.1]]

        config = Configuration(activation_difference_threshold=0.001, normalize_direction=True)

        dir_sep = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful_sep,
            harmless_activations=harmless_sep,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )
        dir_less = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful_less,
            harmless_activations=harmless_less,
            configuration=config,
            layer_index=3,
            model_id="model-x",
        )

        assert dir_sep is not None
        assert dir_less is not None
        # Well-separated should have higher strength
        assert dir_sep.strength > dir_less.strength
