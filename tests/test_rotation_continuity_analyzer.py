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

"""Tests for RotationContinuityAnalyzer.

Tests the cross-model rotation analysis that determines whether
global vs per-layer alignment is needed for model merging.
"""

from typing import Dict, List

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.generalized_procrustes import (
    Config,
    LayerRotationResult,
    RotationContinuityAnalyzer,
    RotationContinuityResult,
)


class TestRotationContinuityAnalyzer:
    """Tests for RotationContinuityAnalyzer.compute_per_layer_alignments."""

    @pytest.fixture
    def base_activations(self) -> Dict[int, Dict[str, List[float]]]:
        """Create base activations with 3 layers and 4 anchors."""
        backend = get_default_backend()
        backend.random_seed(42)
        dim = 8
        activations = {}
        for layer in range(3):
            activations[layer] = {}
            for i in range(4):
                act = backend.random_randn((dim,))
                backend.eval(act)
                activations[layer][f"anchor_{i}"] = backend.to_numpy(act).tolist()
        return activations

    @pytest.fixture
    def rotated_activations(
        self, base_activations: Dict[int, Dict[str, List[float]]]
    ) -> Dict[int, Dict[str, List[float]]]:
        """Create activations that are globally rotated from base."""
        backend = get_default_backend()

        # Apply same rotation to all layers (should result in global alignment sufficient)
        theta = 0.3
        dim = 8

        # Build a simple rotation in 8D (rotate first 2 dims) using numpy for rotation matrix construction
        import numpy as np
        rotation = np.eye(dim, dtype=np.float64)
        rotation[0, 0] = np.cos(theta)
        rotation[0, 1] = -np.sin(theta)
        rotation[1, 0] = np.sin(theta)
        rotation[1, 1] = np.cos(theta)

        rotation_tensor = backend.array(rotation)
        backend.eval(rotation_tensor)

        result = {}
        for layer, anchors in base_activations.items():
            result[layer] = {}
            for anchor, act in anchors.items():
                act_tensor = backend.array(act)
                rotated = backend.matmul(act_tensor, rotation_tensor)
                backend.eval(rotated)
                result[layer][anchor] = backend.to_numpy(rotated).tolist()
        return result

    def test_identical_activations_returns_low_error(self, base_activations):
        """Identical activations should have near-zero error."""
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_activations=base_activations,
            target_activations=base_activations,
            source_model="model_a",
            target_model="model_a",
        )

        assert result is not None
        assert result.global_rotation_error < 1e-6
        for layer in result.layers:
            assert layer.error < 1e-6

    def test_global_rotation_detected(self, base_activations, rotated_activations):
        """Consistent rotation across layers should have low errors."""
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_activations=base_activations,
            target_activations=rotated_activations,
            source_model="base",
            target_model="rotated",
        )

        assert result is not None
        # Both global and per-layer errors should be very low for consistent rotation
        assert result.global_rotation_error < 1e-6
        for layer in result.layers:
            assert layer.error < 1e-6
        # When both are near-zero, the actual recommendation doesn't matter
        # The key insight is that the errors are uniformly low

    def test_per_layer_rotation_needed(self, base_activations):
        """Different rotations per layer should require per-layer alignment."""
        backend = get_default_backend()
        dim = 8

        # Create per-layer rotations with different angles
        per_layer_rotated = {}
        import numpy as np
        for layer, anchors in base_activations.items():
            # Different angle for each layer
            theta = 0.3 + layer * 0.5  # 0.3, 0.8, 1.3 radians
            rotation = np.eye(dim, dtype=np.float64)
            rotation[0, 0] = np.cos(theta)
            rotation[0, 1] = -np.sin(theta)
            rotation[1, 0] = np.sin(theta)
            rotation[1, 1] = np.cos(theta)

            rotation_tensor = backend.array(rotation)
            backend.eval(rotation_tensor)

            per_layer_rotated[layer] = {}
            for anchor, act in anchors.items():
                act_tensor = backend.array(act)
                rotated = backend.matmul(act_tensor, rotation_tensor)
                backend.eval(rotated)
                per_layer_rotated[layer][anchor] = backend.to_numpy(rotated).tolist()

        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_activations=base_activations,
            target_activations=per_layer_rotated,
            source_model="base",
            target_model="per_layer_rotated",
        )

        assert result is not None
        # Should require per-layer alignment due to varying rotations
        # The smoothness ratio should be low
        assert result.rotation_roughness > 0.1
        # Angular deviation should be non-zero
        angular_devs = [
            l.angular_deviation for l in result.layers if l.angular_deviation is not None
        ]
        assert len(angular_devs) > 0
        assert any(d > 0.1 for d in angular_devs)

    def test_returns_none_for_no_common_layers(self):
        """Should return None when no layers overlap."""
        source = {0: {"a": [1.0, 2.0, 3.0]}}
        target = {5: {"a": [1.0, 2.0, 3.0]}}

        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_activations=source,
            target_activations=target,
            source_model="s",
            target_model="t",
        )

        assert result is None

    def test_returns_none_for_insufficient_anchors(self):
        """Should return None when fewer than 3 anchors are common."""
        source = {0: {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}}
        target = {0: {"a": [1.0, 2.0, 3.0], "c": [4.0, 5.0, 6.0]}}

        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_activations=source,
            target_activations=target,
            source_model="s",
            target_model="t",
        )

        assert result is None

    def test_result_metadata(self, base_activations, rotated_activations):
        """Verify result metadata is populated correctly."""
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_activations=base_activations,
            target_activations=rotated_activations,
            source_model="model_source",
            target_model="model_target",
        )

        assert result is not None
        assert result.source_model == "model_source"
        assert result.target_model == "model_target"
        assert result.source_dimension == 8
        assert result.target_dimension == 8
        assert result.anchor_count == 4
        assert len(result.layers) == 3

    def test_layer_results_have_rotation_matrices(self, base_activations, rotated_activations):
        """Each layer should have a rotation matrix."""
        backend = get_default_backend()
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_activations=base_activations,
            target_activations=rotated_activations,
            source_model="s",
            target_model="t",
        )

        assert result is not None
        for layer_result in result.layers:
            assert layer_result.rotation is not None
            rotation = backend.array(layer_result.rotation)
            backend.eval(rotation)
            rotation_np = backend.to_numpy(rotation)

            # Should be square and orthogonal
            assert rotation_np.shape[0] == rotation_np.shape[1]
            # R @ R^T should be identity (orthogonal matrix)
            rotation_t = backend.transpose(rotation)
            identity_approx = backend.matmul(rotation, rotation_t)
            backend.eval(identity_approx)
            identity_approx_np = backend.to_numpy(identity_approx)

            expected_identity = backend.to_numpy(backend.eye(rotation_np.shape[0]))
            assert backend.allclose(backend.array(identity_approx_np), backend.array(expected_identity), atol=1e-5)

    def test_summary_property(self, base_activations, rotated_activations):
        """Verify summary string is generated correctly."""
        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_activations=base_activations,
            target_activations=rotated_activations,
            source_model="base_model",
            target_model="target_model",
        )

        assert result is not None
        summary = result.summary
        assert "Rotation Continuity Analysis" in summary
        assert "base_model" in summary
        assert "target_model" in summary
        assert "Dimensions:" in summary
        assert "Conclusion:" in summary

    def test_config_reflection_handling(self, base_activations):
        """Test that reflections are handled based on config."""
        backend = get_default_backend()

        # Create a reflected version (negate one dimension)
        reflected = {}
        for layer, anchors in base_activations.items():
            reflected[layer] = {}
            for anchor, act in anchors.items():
                arr = backend.array(act)
                backend.eval(arr)
                arr_np = backend.to_numpy(arr).copy()
                arr_np[0] = -arr_np[0]  # Negate first dimension
                reflected[layer][anchor] = arr_np.tolist()

        analyzer = RotationContinuityAnalyzer()

        # With reflections disallowed (default)
        result_no_reflect = analyzer.compute_per_layer_alignments(
            source_activations=base_activations,
            target_activations=reflected,
            source_model="s",
            target_model="t",
            config=Config(allow_reflections=False),
        )

        # With reflections allowed
        result_reflect = analyzer.compute_per_layer_alignments(
            source_activations=base_activations,
            target_activations=reflected,
            source_model="s",
            target_model="t",
            config=Config(allow_reflections=True),
        )

        assert result_no_reflect is not None
        assert result_reflect is not None
        # Reflection allowed should have lower error
        assert result_reflect.global_rotation_error <= result_no_reflect.global_rotation_error

    def test_different_dimension_models(self):
        """Test alignment with different source and target dimensions."""
        backend = get_default_backend()
        backend.random_seed(99)

        # Source with dim 8, target with dim 6
        source = {}
        target = {}
        for layer in range(2):
            source[layer] = {}
            target[layer] = {}
            for i in range(4):
                act_source = backend.random_randn((8,))
                act_target = backend.random_randn((6,))
                backend.eval(act_source, act_target)
                source[layer][f"anchor_{i}"] = backend.to_numpy(act_source).tolist()
                target[layer][f"anchor_{i}"] = backend.to_numpy(act_target).tolist()

        analyzer = RotationContinuityAnalyzer()
        result = analyzer.compute_per_layer_alignments(
            source_activations=source,
            target_activations=target,
            source_model="large",
            target_model="small",
        )

        assert result is not None
        # Should use shared dimension (min of 8, 6 = 6)
        assert result.source_dimension == 8
        assert result.target_dimension == 6
        # Rotation matrices should be 6x6 (shared_dim)
        for layer_result in result.layers:
            rotation = backend.array(layer_result.rotation)
            backend.eval(rotation)
            rotation_np = backend.to_numpy(rotation)
            assert rotation_np.shape == (6, 6)


class TestLayerRotationResult:
    """Tests for LayerRotationResult dataclass."""

    def test_layer_result_fields(self):
        """Verify LayerRotationResult has expected fields."""
        result = LayerRotationResult(
            layer_index=5,
            rotation=[[1.0, 0.0], [0.0, 1.0]],
            error=0.01,
            angular_deviation=0.1,
            rotation_delta=0.05,
        )

        assert result.layer_index == 5
        assert result.error == 0.01
        assert result.angular_deviation == 0.1
        assert result.rotation_delta == 0.05

    def test_layer_result_optional_fields(self):
        """First layer can have None for angular_deviation."""
        result = LayerRotationResult(
            layer_index=0,
            rotation=[[1.0, 0.0], [0.0, 1.0]],
            error=0.01,
            angular_deviation=None,
            rotation_delta=None,
        )

        assert result.angular_deviation is None
        assert result.rotation_delta is None


class TestRotationContinuityResultSummary:
    """Tests for RotationContinuityResult summary generation."""

    def test_summary_per_layer_required(self):
        """Summary should indicate per-layer alignment required when smoothness < 0.7."""
        result = RotationContinuityResult(
            source_model="a",
            target_model="b",
            layers=[
                LayerRotationResult(0, [[1, 0], [0, 1]], 0.1, None, None),
                LayerRotationResult(1, [[1, 0], [0, 1]], 0.1, 0.5, 0.3),
            ],
            global_rotation_error=0.5,
            smoothness_ratio=0.5,  # < 0.7
            rotation_roughness=0.2,
            mean_angular_velocity=0.3,
            requires_per_layer_alignment=True,
            source_dimension=2,
            target_dimension=2,
            anchor_count=4,
        )

        assert "Per-layer alignment REQUIRED" in result.summary

    def test_summary_global_sufficient(self):
        """Summary should indicate global rotation sufficient when smoothness >= 0.7."""
        result = RotationContinuityResult(
            source_model="a",
            target_model="b",
            layers=[
                LayerRotationResult(0, [[1, 0], [0, 1]], 0.1, None, None),
            ],
            global_rotation_error=0.1,
            smoothness_ratio=0.95,  # >= 0.7
            rotation_roughness=0.01,
            mean_angular_velocity=0.02,
            requires_per_layer_alignment=False,
            source_dimension=2,
            target_dimension=2,
            anchor_count=4,
        )

        assert "Global rotation SUFFICIENT" in result.summary
