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

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.affine_stitching_layer import (
    AffineStitchingLayer,
    AnchorPair,
    BackendAffineStitchingLayer,
    Config,
    get_affine_stitching_layer,
)
from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorActivation,
    AnchorMetadata,
    ConceptResponseMatrix,
)


def test_train_min_samples() -> None:
    data = [
        AnchorPair(source_activation=[1.0, 0.0], target_activation=[1.0, 0.0]),
        AnchorPair(source_activation=[0.0, 1.0], target_activation=[0.0, 1.0]),
    ]
    config = Config(min_samples=3, max_iterations=10)
    assert AffineStitchingLayer.train(data, config=config) is None


def test_train_identity_mapping() -> None:
    data = [
        AnchorPair(source_activation=[1.0, 0.0], target_activation=[1.0, 0.0]),
        AnchorPair(source_activation=[0.0, 1.0], target_activation=[0.0, 1.0]),
        AnchorPair(source_activation=[1.0, 1.0], target_activation=[1.0, 1.0]),
    ]
    config = Config(
        min_samples=1,
        max_iterations=5,
        weight_decay=0.0,
        use_momentum=False,
        use_procrustes_warm_start=True,
    )
    result = AffineStitchingLayer.train(data, config=config)
    assert result is not None
    assert result.forward_error == pytest.approx(0.0, abs=1e-6)
    assert result.backward_error == pytest.approx(0.0, abs=1e-6)
    assert result.weights[0][0] == pytest.approx(1.0, abs=1e-5)
    assert result.weights[1][1] == pytest.approx(1.0, abs=1e-5)
    assert result.weights[0][1] == pytest.approx(0.0, abs=1e-5)
    assert result.weights[1][0] == pytest.approx(0.0, abs=1e-5)


def test_apply_and_inverse() -> None:
    weights = [[1.0, 0.0], [0.0, 1.0]]
    bias = [0.0, 0.0]
    activations = [[1.0, 2.0], [-1.0, 0.5]]

    forward = AffineStitchingLayer.apply(activations, weights, bias)
    assert forward == activations

    inverse = AffineStitchingLayer.apply_inverse(activations, weights)
    assert inverse == activations


def test_train_from_crms() -> None:
    metadata = AnchorMetadata(
        total_count=2,
        semantic_prime_count=2,
        computational_gate_count=0,
        anchor_ids=["prime:a", "prime:b"],
    )
    source = ConceptResponseMatrix(
        model_identifier="source",
        layer_count=1,
        hidden_dim=2,
        anchor_metadata=metadata,
    )
    target = ConceptResponseMatrix(
        model_identifier="target",
        layer_count=1,
        hidden_dim=2,
        anchor_metadata=metadata,
    )

    source.activations = {
        0: {
            "prime:a": AnchorActivation("prime:a", 0, [1.0, 0.0]),
            "prime:b": AnchorActivation("prime:b", 0, [0.0, 1.0]),
        }
    }
    target.activations = {
        0: {
            "prime:a": AnchorActivation("prime:a", 0, [1.0, 0.0]),
            "prime:b": AnchorActivation("prime:b", 0, [0.0, 1.0]),
        }
    }

    config = Config(min_samples=1, max_iterations=3, weight_decay=0.0, use_momentum=False)
    result = AffineStitchingLayer.train_from_crms(source, target, layer=0, config=config)
    assert result is not None


class TestBackendAffineStitchingLayer:
    """Tests for the GPU-accelerated BackendAffineStitchingLayer."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    @pytest.fixture
    def identity_data(self, backend):
        """Create identity mapping test data."""
        return [
            AnchorPair(source_activation=[1.0, 0.0], target_activation=[1.0, 0.0]),
            AnchorPair(source_activation=[0.0, 1.0], target_activation=[0.0, 1.0]),
            AnchorPair(source_activation=[1.0, 1.0], target_activation=[1.0, 1.0]),
        ]

    @pytest.fixture
    def rotation_data(self, backend):
        """Create 90-degree rotation test data."""
        return [
            AnchorPair(
                source_activation=[1.0, 0.0],
                target_activation=[0.0, 1.0],
            ),
            AnchorPair(
                source_activation=[0.0, 1.0],
                target_activation=[-1.0, 0.0],
            ),
            AnchorPair(
                source_activation=[1.0, 1.0],
                target_activation=[-1.0, 1.0],
            ),
        ]

    def test_train_identity_mapping(self, backend, identity_data):
        """Test training on identity mapping produces identity matrix."""
        stitcher = BackendAffineStitchingLayer(backend)
        config = Config(
            min_samples=1,
            max_iterations=5,
            weight_decay=0.0,
            use_momentum=False,
            use_procrustes_warm_start=True,
        )
        result = stitcher.train(identity_data, config=config)
        assert result is not None
        assert result.forward_error == pytest.approx(0.0, abs=1e-5)
        assert result.backward_error == pytest.approx(0.0, abs=1e-5)
        assert result.weights[0][0] == pytest.approx(1.0, abs=1e-4)
        assert result.weights[1][1] == pytest.approx(1.0, abs=1e-4)
        assert result.weights[0][1] == pytest.approx(0.0, abs=1e-4)
        assert result.weights[1][0] == pytest.approx(0.0, abs=1e-4)

    def test_train_min_samples(self, backend):
        """Test that training fails when below min_samples."""
        data = [
            AnchorPair(source_activation=[1.0, 0.0], target_activation=[1.0, 0.0]),
            AnchorPair(source_activation=[0.0, 1.0], target_activation=[0.0, 1.0]),
        ]
        stitcher = BackendAffineStitchingLayer(backend)
        config = Config(min_samples=3, max_iterations=10)
        result = stitcher.train(data, config=config)
        assert result is None

    def test_apply_and_inverse(self, backend):
        """Test apply and apply_inverse are inverses of each other."""
        stitcher = BackendAffineStitchingLayer(backend)
        weights = backend.array([[1.0, 0.0], [0.0, 1.0]])
        bias = backend.array([0.0, 0.0])
        activations = backend.array([[1.0, 2.0], [-1.0, 0.5]])

        forward = stitcher.apply(activations, weights, bias)
        inverse = stitcher.apply_inverse(activations, weights)

        # Both should equal original for identity weights
        act_list = activations.tolist()
        forward_list = forward.tolist()
        inverse_list = inverse.tolist()
        for i in range(2):
            for j in range(2):
                expected = act_list[i][j]
                assert forward_list[i][j] == pytest.approx(expected, abs=1e-6)
                assert inverse_list[i][j] == pytest.approx(expected, abs=1e-6)

    def test_apply_with_rotation(self, backend):
        """Test apply correctly transforms with a rotation matrix."""
        stitcher = BackendAffineStitchingLayer(backend)
        # 90-degree rotation
        weights = backend.array([[0.0, -1.0], [1.0, 0.0]])
        bias = backend.array([0.0, 0.0])
        activations = backend.array([[1.0, 0.0]])

        result = stitcher.apply(activations, weights, bias)
        result_list = result.tolist()
        assert result_list[0][0] == pytest.approx(0.0, abs=1e-6)
        assert result_list[0][1] == pytest.approx(1.0, abs=1e-6)

    def test_train_rotation(self, backend, rotation_data):
        """Test training learns a rotation matrix."""
        stitcher = BackendAffineStitchingLayer(backend)
        config = Config(
            min_samples=1,
            max_iterations=50,
            learning_rate=0.1,
            weight_decay=0.0,
            use_momentum=False,
            use_procrustes_warm_start=True,
        )
        result = stitcher.train(rotation_data, config=config)
        assert result is not None
        # Errors should be low for a learnable transformation
        assert result.forward_error < 0.5
        assert result.backward_error < 0.5

    def test_results_match_pure_python(self, backend, identity_data):
        """Verify BackendAffineStitchingLayer matches AffineStitchingLayer."""
        config = Config(
            min_samples=1,
            max_iterations=5,
            weight_decay=0.0,
            use_momentum=False,
            use_procrustes_warm_start=True,
        )

        # Train with pure Python
        pure_result = AffineStitchingLayer.train(identity_data, config=config)

        # Train with Backend
        stitcher = BackendAffineStitchingLayer(backend)
        backend_result = stitcher.train(identity_data, config=config)

        assert pure_result is not None
        assert backend_result is not None

        # Compare weights
        for i in range(2):
            for j in range(2):
                assert pure_result.weights[i][j] == pytest.approx(
                    backend_result.weights[i][j], abs=1e-4
                )

        # Compare biases
        for i in range(2):
            assert pure_result.bias[i] == pytest.approx(backend_result.bias[i], abs=1e-4)

        # Compare errors
        assert pure_result.forward_error == pytest.approx(
            backend_result.forward_error, abs=1e-4
        )
        assert pure_result.backward_error == pytest.approx(
            backend_result.backward_error, abs=1e-4
        )


class TestGetAffineStitchingLayer:
    """Tests for the factory function."""

    def test_returns_pure_python_without_backend(self):
        """Test that factory returns AffineStitchingLayer without backend."""
        result = get_affine_stitching_layer()
        assert isinstance(result, AffineStitchingLayer)

    def test_returns_backend_with_backend(self):
        """Test that factory returns BackendAffineStitchingLayer with backend."""
        backend = get_default_backend()
        result = get_affine_stitching_layer(backend)
        assert isinstance(result, BackendAffineStitchingLayer)
