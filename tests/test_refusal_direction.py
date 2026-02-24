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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.refusal_direction_cache import RefusalDirectionCache
from modelcypher.core.domain.geometry.refusal_direction_detector import RefusalDirectionDetector


def test_refusal_direction_compute_and_distance() -> None:
    harmful = [[1.0, 0.0], [1.0, 0.0]]
    harmless = [[0.0, 1.0], [0.0, 1.0]]
    direction = RefusalDirectionDetector.compute_direction(
        harmful_activations=harmful,
        harmless_activations=harmless,
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
    eps = division_epsilon(get_default_backend(), get_default_backend().array([1.0]))
    assert abs(metrics.distance_to_refusal) < eps
    assert metrics.projection_magnitude > 0  # Same direction as refusal, not opposite


def test_refusal_direction_cache_roundtrip(tmp_path) -> None:
    harmful = [[1.0, 0.0]]
    harmless = [[0.0, 1.0]]
    direction = RefusalDirectionDetector.compute_direction(
        harmful_activations=harmful,
        harmless_activations=harmless,
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
        dir1 = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            layer_index=3,
            model_id="model-x",
        )
        dir2 = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            layer_index=3,
            model_id="model-x",
        )

        assert dir1 is not None
        assert dir2 is not None
        # Same direction vector
        b = get_default_backend()
        eps = division_epsilon(b, b.array(dir1.direction))
        for i in range(len(dir1.direction)):
            assert abs(dir1.direction[i] - dir2.direction[i]) < eps

    def test_direction_robust_to_sample_ordering(self) -> None:
        """Direction should be robust to ordering of samples."""
        harmful = [[1.0, 0.0], [1.1, 0.1], [0.9, -0.1]]
        harmless = [[0.0, 1.0], [0.1, 0.9], [-0.1, 1.1]]

        harmful_reordered = [harmful[2], harmful[0], harmful[1]]
        harmless_reordered = [harmless[2], harmless[0], harmless[1]]

        dir1 = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            layer_index=3,
            model_id="model-x",
        )
        dir2 = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful_reordered,
            harmless_activations=harmless_reordered,
            layer_index=3,
            model_id="model-x",
        )

        assert dir1 is not None
        assert dir2 is not None
        # Directions should be similar (Fréchet mean is order-independent)
        dot = sum(dir1.direction[i] * dir2.direction[i] for i in range(len(dir1.direction)))
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([1.0]))
        # Either same direction or opposite (|dot| close to 1)
        assert abs(abs(dot) - 1.0) <= eps


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

        direction = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
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
        direction = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
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
        direction = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
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

        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([1.0]))
        # Distance should be 1.0 (perpendicular to direction)
        assert abs(metrics.distance_to_refusal - 1.0) <= eps
        # Projection should be near zero
        assert abs(metrics.projection_magnitude) <= eps

    def test_direction_normalized(self) -> None:
        """Direction should be normalized."""
        harmful = [[2.0, 0.0]]
        harmless = [[0.0, 0.0]]
        direction = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful,
            harmless_activations=harmless,
            layer_index=3,
            model_id="model-x",
        )
        assert direction is not None

        # Direction magnitude should be 1.0 (normalized)
        magnitude = sum(d * d for d in direction.direction) ** 0.5
        eps = division_epsilon(get_default_backend(), get_default_backend().array([1.0]))
        assert abs(magnitude - 1.0) < eps

    def test_direction_strength_reflects_separation(self) -> None:
        """Direction strength should reflect how well it separates."""
        # Well-separated data
        harmful_sep = [[5.0, 0.0], [5.0, 0.0]]
        harmless_sep = [[-5.0, 0.0], [-5.0, 0.0]]

        # Less separated data (but still separable)
        harmful_less = [[2.0, 0.0], [2.0, 0.1]]
        harmless_less = [[-1.0, 0.0], [-1.0, 0.1]]

        dir_sep = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful_sep,
            harmless_activations=harmless_sep,
            layer_index=3,
            model_id="model-x",
        )
        dir_less = RefusalDirectionDetector.compute_direction(
            harmful_activations=harmful_less,
            harmless_activations=harmless_less,
            layer_index=3,
            model_id="model-x",
        )

        assert dir_sep is not None
        assert dir_less is not None
        # Well-separated should have higher strength
        assert dir_sep.strength > dir_less.strength
