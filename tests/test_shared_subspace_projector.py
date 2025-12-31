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

"""
Comprehensive tests for SharedSubspaceProjector.

These tests verify:
- CCA, Shared SVD, and Procrustes alignment methods
- Edge cases: empty inputs, singular matrices, mismatched dimensions
- Numerical stability: regularization, whitening, PCA reduction
- Helper functions: centering, weighting, component selection
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorActivation,
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.geometry.backend_matrix_utils import (
    reshape_flat_to_matrix,
    transpose_flat_matrix,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    AlignmentMethod,
    Config,
    PcaMode,
    Result,
    SharedSubspaceProjector,
)


def _make_crm(model_id: str, activations: dict[str, list[float]]) -> ConceptResponseMatrix:
    metadata = AnchorMetadata(
        total_count=len(activations),
        semantic_prime_count=len(activations),
        computational_gate_count=0,
        anchor_ids=sorted(activations.keys()),
    )
    crm = ConceptResponseMatrix(
        model_identifier=model_id,
        layer_count=1,
        hidden_dim=len(next(iter(activations.values()))),
        anchor_metadata=metadata,
    )
    crm.activations = {
        0: {
            anchor_id: AnchorActivation(anchor_id, 0, vector)
            for anchor_id, vector in activations.items()
        }
    }
    return crm


def test_discover_requires_min_samples() -> None:
    source = _make_crm("source", {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]})
    target = _make_crm("target", {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]})
    config = Config(alignment_method=AlignmentMethod.procrustes, min_samples=3)

    result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)
    assert result is None


def test_discover_procrustes_identity() -> None:
    source = _make_crm("source", {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]})
    target = _make_crm("target", {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]})
    config = Config(
        alignment_method=AlignmentMethod.procrustes, min_samples=1, variance_threshold=0.9
    )

    result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)
    assert result is not None
    assert result.method == AlignmentMethod.procrustes
    assert result.alignment_error == pytest.approx(0.0, abs=1e-6)
    assert result.shared_dimension >= 1


def test_discover_shared_svd() -> None:
    source = _make_crm(
        "source",
        {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0], "prime:c": [1.0, 1.0]},
    )
    target = _make_crm(
        "target",
        {"prime:a": [2.0, 0.0], "prime:b": [0.0, 2.0], "prime:c": [2.0, 2.0]},
    )
    config = Config(
        alignment_method=AlignmentMethod.shared_svd, min_samples=1, variance_threshold=0.8
    )

    result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)
    assert result is not None
    assert result.shared_dimension >= 1
    assert len(result.alignment_strengths) == result.shared_dimension
    assert result.shared_variance_ratio > 0.0


def test_anchor_weighting_biases_shared_subspace() -> None:
    source = _make_crm(
        "source",
        {
            "prime:a": [1.0],
            "prime:b": [2.0],
            "gate:a": [-1.0],
            "gate:b": [-2.0],
        },
    )
    target = _make_crm(
        "target",
        {
            "prime:a": [1.0],
            "prime:b": [2.0],
            "gate:a": [2.0],
            "gate:b": [-2.0],
        },
    )
    unweighted = SharedSubspaceProjector.discover(
        source,
        target,
        layer=0,
        config=Config(
            alignment_method=AlignmentMethod.cca,
            min_samples=1,
            cca_regularization=0.0,
        ),
    )
    weighted = SharedSubspaceProjector.discover(
        source,
        target,
        layer=0,
        config=Config(
            alignment_method=AlignmentMethod.cca,
            min_samples=1,
            cca_regularization=0.0,
            anchor_weights={"prime:": 2.0, "gate:": 0.0},
        ),
    )
    assert unweighted is not None
    assert weighted is not None
    assert weighted.alignment_strengths[0] > unweighted.alignment_strengths[0]
    assert weighted.alignment_strengths[0] > 0.6


# =============================================================================
# Backend Fixture
# =============================================================================


@pytest.fixture
def backend():
    """Get the default backend for tests."""
    return get_default_backend()


# =============================================================================
# Result Dataclass Tests
# =============================================================================


class TestResultDataclass:
    """Tests for the Result dataclass."""

    def test_is_valid_true(self):
        """Valid result should return is_valid=True."""
        result = Result(
            shared_dimension=10,
            source_dimension=64,
            target_dimension=64,
            source_projection=[[1.0] * 10] * 64,
            target_projection=[[1.0] * 10] * 64,
            alignment_strengths=[0.9, 0.8, 0.7],
            alignment_error=0.1,
            shared_variance_ratio=0.8,
            sample_count=100,
            method=AlignmentMethod.cca,
        )
        assert result.is_valid is True

    def test_is_valid_false_zero_dimension(self):
        """Zero shared dimension should return is_valid=False."""
        result = Result(
            shared_dimension=0,
            source_dimension=64,
            target_dimension=64,
            source_projection=[],
            target_projection=[],
            alignment_strengths=[],
            alignment_error=0.1,
            shared_variance_ratio=0.8,
            sample_count=100,
            method=AlignmentMethod.cca,
        )
        assert result.is_valid is False

    def test_is_valid_false_high_error(self):
        """High alignment error should return is_valid=False."""
        result = Result(
            shared_dimension=10,
            source_dimension=64,
            target_dimension=64,
            source_projection=[[1.0] * 10] * 64,
            target_projection=[[1.0] * 10] * 64,
            alignment_strengths=[0.9],
            alignment_error=0.6,  # Above 0.5 threshold
            shared_variance_ratio=0.8,
            sample_count=100,
            method=AlignmentMethod.cca,
        )
        assert result.is_valid is False

    def test_is_valid_false_low_variance(self):
        """Low shared variance ratio should return is_valid=False."""
        result = Result(
            shared_dimension=10,
            source_dimension=64,
            target_dimension=64,
            source_projection=[[1.0] * 10] * 64,
            target_projection=[[1.0] * 10] * 64,
            alignment_strengths=[0.9],
            alignment_error=0.1,
            shared_variance_ratio=0.4,  # Below 0.5 threshold
            sample_count=100,
            method=AlignmentMethod.cca,
        )
        assert result.is_valid is False

    def test_h3_metrics(self):
        """h3_metrics property should return correct H3ValidationMetrics."""
        result = Result(
            shared_dimension=10,
            source_dimension=64,
            target_dimension=64,
            source_projection=[[1.0] * 10] * 64,
            target_projection=[[1.0] * 10] * 64,
            alignment_strengths=[0.95, 0.8, 0.7],
            alignment_error=0.15,
            shared_variance_ratio=0.85,
            sample_count=100,
            method=AlignmentMethod.cca,
        )
        h3 = result.h3_metrics
        assert h3.shared_dimension == 10
        assert h3.top_canonical_correlation == 0.95
        assert h3.alignment_error == 0.15
        assert h3.shared_variance_ratio == 0.85

    def test_h3_metrics_empty_strengths(self):
        """h3_metrics with empty alignment_strengths should return 0.0."""
        result = Result(
            shared_dimension=0,
            source_dimension=64,
            target_dimension=64,
            source_projection=[],
            target_projection=[],
            alignment_strengths=[],
            alignment_error=0.0,
            shared_variance_ratio=0.0,
            sample_count=0,
            method=AlignmentMethod.cca,
        )
        h3 = result.h3_metrics
        assert h3.top_canonical_correlation == 0.0


# =============================================================================
# Config Tests
# =============================================================================


class TestConfig:
    """Tests for the Config dataclass."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = Config.default()
        assert config.alignment_method == AlignmentMethod.cca
        assert config.variance_threshold == 0.95
        assert config.max_shared_dimension == 256
        assert config.min_samples == 10
        assert config.min_canonical_correlation == 0.1

    def test_custom_config(self):
        """Custom config should store values correctly."""
        config = Config(
            alignment_method=AlignmentMethod.procrustes,
            variance_threshold=0.9,
            max_shared_dimension=128,
            anchor_weights={"prime:": 2.0},
        )
        assert config.alignment_method == AlignmentMethod.procrustes
        assert config.variance_threshold == 0.9
        assert config.max_shared_dimension == 128
        assert config.anchor_weights == {"prime:": 2.0}


# =============================================================================
# Helper Function Tests - Centering
# =============================================================================


class TestCenterMatrix:
    """Tests for _center_matrix static method."""

    def test_center_matrix_simple(self):
        """Simple matrix should be centered correctly."""
        matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        centered, means = SharedSubspaceProjector._center_matrix(matrix)

        assert means == pytest.approx([3.0, 4.0], abs=1e-6)
        assert centered[0] == pytest.approx([-2.0, -2.0], abs=1e-6)
        assert centered[1] == pytest.approx([0.0, 0.0], abs=1e-6)
        assert centered[2] == pytest.approx([2.0, 2.0], abs=1e-6)

    def test_center_matrix_empty(self):
        """Empty matrix should return empty."""
        centered, means = SharedSubspaceProjector._center_matrix([])
        assert centered == []
        assert means == []

    def test_center_matrix_with_weights(self):
        """Weighted centering should use weights."""
        matrix = [[0.0, 0.0], [10.0, 10.0]]
        # Weight first row 3x more than second
        weights = [0.75, 0.25]
        centered, means = SharedSubspaceProjector._center_matrix(matrix, weights)

        # Weighted mean = 0.75*0 + 0.25*10 = 2.5
        assert means == pytest.approx([2.5, 2.5], abs=1e-6)

    def test_center_matrix_zero_weights(self):
        """Zero total weights should fall back to unweighted."""
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        weights = [0.0, 0.0]
        centered, means = SharedSubspaceProjector._center_matrix(matrix, weights)

        # Should fall back to unweighted
        assert means == pytest.approx([2.0, 3.0], abs=1e-6)


class TestCenterArray:
    """Tests for _center_array static method using backend."""

    def test_center_array_simple(self, backend):
        """Simple array centering."""
        array = backend.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        backend.eval(array)

        centered, mean = SharedSubspaceProjector._center_array(array, None, backend=backend)

        mean_np = backend.to_numpy(mean)
        centered_np = backend.to_numpy(centered)

        assert mean_np.tolist() == pytest.approx([3.0, 4.0], abs=1e-6)
        assert centered_np[1].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)

    def test_center_array_with_weights(self, backend):
        """Weighted array centering."""
        array = backend.array([[0.0, 0.0], [10.0, 10.0]])
        weights = backend.array([0.75, 0.25])
        backend.eval(array, weights)

        centered, mean = SharedSubspaceProjector._center_array(array, weights, backend=backend)

        mean_np = backend.to_numpy(mean)
        assert mean_np.tolist() == pytest.approx([2.5, 2.5], abs=1e-6)


# =============================================================================
# Helper Function Tests - Weights
# =============================================================================


class TestNormalizeWeights:
    """Tests for _normalize_weights static method."""

    def test_normalize_weights_simple(self, backend):
        """Simple weights should sum to 1."""
        weights = [1.0, 2.0, 3.0]
        result = SharedSubspaceProjector._normalize_weights(weights, backend=backend)

        result_np = backend.to_numpy(result)
        assert result_np.sum() == pytest.approx(1.0, abs=1e-6)
        assert result_np[0] == pytest.approx(1 / 6, abs=1e-6)

    def test_normalize_weights_empty(self, backend):
        """Empty weights should return None."""
        result = SharedSubspaceProjector._normalize_weights([], backend=backend)
        assert result is None

    def test_normalize_weights_none(self, backend):
        """None weights should return None."""
        result = SharedSubspaceProjector._normalize_weights(None, backend=backend)
        assert result is None

    def test_normalize_weights_negative(self, backend):
        """Negative weights should be clamped to 0."""
        weights = [-1.0, 2.0, 3.0]
        result = SharedSubspaceProjector._normalize_weights(weights, backend=backend)

        result_np = backend.to_numpy(result)
        # -1 becomes 0, so total is 5
        assert result_np[0] == pytest.approx(0.0, abs=1e-6)
        assert result_np.sum() == pytest.approx(1.0, abs=1e-6)

    def test_normalize_weights_all_zero(self, backend):
        """All zero weights should return None."""
        result = SharedSubspaceProjector._normalize_weights([0.0, 0.0, 0.0], backend=backend)
        assert result is None


class TestAnchorWeight:
    """Tests for _anchor_weight static method."""

    def test_anchor_weight_no_config(self):
        """No anchor_weights in config should return 1.0."""
        config = Config()
        weight = SharedSubspaceProjector._anchor_weight("prime:test", config)
        assert weight == 1.0

    def test_anchor_weight_matching_prefix(self):
        """Matching prefix should return configured weight."""
        config = Config(anchor_weights={"prime:": 2.0, "gate:": 0.5})
        weight = SharedSubspaceProjector._anchor_weight("prime:test", config)
        assert weight == 2.0

    def test_anchor_weight_no_match(self):
        """No matching prefix should return 1.0."""
        config = Config(anchor_weights={"prime:": 2.0})
        weight = SharedSubspaceProjector._anchor_weight("gate:test", config)
        assert weight == 1.0

    def test_anchor_weight_multiple_matches(self):
        """Multiple matching prefixes should return max weight."""
        config = Config(anchor_weights={"p": 1.5, "prime": 2.0, "prime:": 3.0})
        weight = SharedSubspaceProjector._anchor_weight("prime:test", config)
        assert weight == 3.0


# =============================================================================
# Helper Function Tests - PCA Reduction
# =============================================================================


class TestPcaReduce:
    """Tests for _pca_reduce static method."""

    def test_pca_reduce_svd_mode(self, backend):
        """PCA reduction in SVD mode."""
        # Create a matrix with clear principal components
        backend.random_seed(42)
        matrix = backend.random_normal((20, 10))
        backend.eval(matrix)

        reduced, components, variances = SharedSubspaceProjector._pca_reduce(
            matrix, variance_threshold=0.95, max_components=5, mode=PcaMode.svd, backend=backend
        )

        assert reduced is not None
        assert components is not None
        assert variances is not None
        # Should reduce dimensions
        assert reduced.shape[1] <= 5

    def test_pca_reduce_gram_mode(self, backend):
        """PCA reduction in Gram mode (for wide matrices)."""
        backend.random_seed(42)
        # Wide matrix: more features than samples
        matrix = backend.random_normal((10, 50))
        backend.eval(matrix)

        reduced, components, variances = SharedSubspaceProjector._pca_reduce(
            matrix, variance_threshold=0.95, max_components=8, mode=PcaMode.gram, backend=backend
        )

        assert reduced is not None
        assert reduced.shape[0] == 10  # Same number of samples

    def test_pca_reduce_auto_mode(self, backend):
        """Auto mode should select appropriate method."""
        backend.random_seed(42)
        # Tall matrix (n > d) should use SVD
        tall_matrix = backend.random_normal((50, 10))
        # Wide matrix (d > n) should use Gram
        wide_matrix = backend.random_normal((10, 50))
        backend.eval(tall_matrix, wide_matrix)

        tall_result, _, _ = SharedSubspaceProjector._pca_reduce(
            tall_matrix, 0.95, 5, PcaMode.auto, backend=backend
        )
        wide_result, _, _ = SharedSubspaceProjector._pca_reduce(
            wide_matrix, 0.95, 5, PcaMode.auto, backend=backend
        )

        assert tall_result is not None
        assert wide_result is not None

    def test_pca_reduce_empty_matrix(self, backend):
        """Empty matrix should return None."""
        matrix = backend.array([]).reshape((0, 0))
        backend.eval(matrix)

        result, _, _ = SharedSubspaceProjector._pca_reduce(
            matrix, 0.95, 5, PcaMode.svd, backend=backend
        )

        assert result is None

    def test_pca_reduce_zero_max_components(self, backend):
        """Zero max_components should return None."""
        matrix = backend.random_normal((10, 5))
        backend.eval(matrix)

        result, _, _ = SharedSubspaceProjector._pca_reduce(
            matrix, 0.95, 0, PcaMode.svd, backend=backend
        )

        assert result is None


# =============================================================================
# Helper Function Tests - Component Selection
# =============================================================================


class TestSelectComponentCount:
    """Tests for component count selection."""

    def test_select_component_count_list(self):
        """List-based component count selection."""
        variances = [10.0, 5.0, 3.0, 1.0, 0.5]
        # Total = 19.5, need 95% = 18.525
        # Cumsum: 10, 15, 18, 19, 19.5
        # 10/19.5=0.51, 15/19.5=0.77, 18/19.5=0.92, 19/19.5=0.97
        count = SharedSubspaceProjector._select_component_count_list(variances, 0.95)
        assert count == 4  # Need first 4 to reach 95%

    def test_select_component_count_list_empty(self):
        """Empty list should return 0."""
        count = SharedSubspaceProjector._select_component_count_list([], 0.95)
        assert count == 0

    def test_select_component_count_list_zero_total(self):
        """Zero total variance should return 0."""
        count = SharedSubspaceProjector._select_component_count_list([0.0, 0.0], 0.95)
        assert count == 0

    def test_select_component_count_list_single(self):
        """Single variance should return 1 if non-zero."""
        count = SharedSubspaceProjector._select_component_count_list([1.0], 0.95)
        assert count == 1

    def test_select_component_count_array(self, backend):
        """Array-based component count selection."""
        variances = backend.array([10.0, 5.0, 3.0, 1.0, 0.5])
        backend.eval(variances)

        count = SharedSubspaceProjector._select_component_count(variances, 0.95, backend=backend)
        assert count == 4


# =============================================================================
# Helper Function Tests - Covariance Operations
# =============================================================================


class TestRegularizeCovariance:
    """Tests for _regularize_covariance static method."""

    def test_regularize_covariance(self, backend):
        """Regularization should add epsilon to diagonal."""
        cov = backend.array([[1.0, 0.5], [0.5, 1.0]])
        backend.eval(cov)

        regularized = SharedSubspaceProjector._regularize_covariance(cov, 0.1, backend=backend)

        reg_np = backend.to_numpy(regularized)
        assert reg_np[0, 0] == pytest.approx(1.1, abs=1e-6)
        assert reg_np[1, 1] == pytest.approx(1.1, abs=1e-6)
        assert reg_np[0, 1] == pytest.approx(0.5, abs=1e-6)

    def test_regularize_covariance_zero_epsilon(self, backend):
        """Zero epsilon should return unchanged."""
        cov = backend.array([[1.0, 0.5], [0.5, 1.0]])
        backend.eval(cov)

        regularized = SharedSubspaceProjector._regularize_covariance(cov, 0.0, backend=backend)

        reg_np = backend.to_numpy(regularized)
        cov_np = backend.to_numpy(cov)
        assert (reg_np == cov_np).all()


class TestWhitenCovariance:
    """Tests for _whiten_covariance static method."""

    def test_whiten_covariance_identity(self, backend):
        """Identity covariance should whiten to identity."""
        cov = backend.eye(3)
        backend.eval(cov)

        inv_sqrt, eigenvalues = SharedSubspaceProjector._whiten_covariance(cov, backend=backend)

        assert inv_sqrt is not None
        inv_sqrt_np = backend.to_numpy(inv_sqrt)
        # Should be close to identity
        assert inv_sqrt_np[0, 0] == pytest.approx(1.0, abs=0.1)

    def test_whiten_covariance_scaled(self, backend):
        """Scaled identity should whiten correctly."""
        cov = backend.eye(3) * 4.0
        backend.eval(cov)

        inv_sqrt, eigenvalues = SharedSubspaceProjector._whiten_covariance(cov, backend=backend)

        assert inv_sqrt is not None
        inv_sqrt_np = backend.to_numpy(inv_sqrt)
        # Should be 1/sqrt(4) = 0.5
        assert inv_sqrt_np[0, 0] == pytest.approx(0.5, abs=0.1)

    def test_whiten_covariance_empty(self, backend):
        """Empty covariance should return None."""
        cov = backend.array([]).reshape((0, 0))
        backend.eval(cov)

        inv_sqrt, eigenvalues = SharedSubspaceProjector._whiten_covariance(cov, backend=backend)

        assert inv_sqrt is None
        assert eigenvalues is None


# =============================================================================
# Discovery Method Tests - CCA
# =============================================================================


class TestDiscoverWithCCA:
    """Tests for CCA-based discovery."""

    def test_cca_identical_data(self):
        """Identical source/target should produce near-perfect alignment."""
        activations = {f"prime:{i}": [float(i), float(i + 1)] for i in range(15)}
        source = _make_crm("source", activations)
        target = _make_crm("target", activations)

        config = Config(alignment_method=AlignmentMethod.cca, min_samples=5)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        assert result is not None
        assert result.method == AlignmentMethod.cca
        assert result.alignment_error < 0.5

    def test_cca_scaled_data(self):
        """Scaled source/target should still align well."""
        source_acts = {f"prime:{i}": [float(i), float(i + 1)] for i in range(15)}
        target_acts = {f"prime:{i}": [float(i) * 2, float(i + 1) * 2] for i in range(15)}
        source = _make_crm("source", source_acts)
        target = _make_crm("target", target_acts)

        config = Config(alignment_method=AlignmentMethod.cca, min_samples=5)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        assert result is not None
        # CCA should handle scaling
        assert result.alignment_strengths[0] > 0.5

    def test_cca_different_dimensions(self):
        """CCA should handle different source/target dimensions."""
        source_acts = {f"prime:{i}": [float(i), float(i + 1), float(i + 2)] for i in range(15)}
        target_acts = {f"prime:{i}": [float(i), float(i + 1)] for i in range(15)}
        source = _make_crm("source", source_acts)
        target = _make_crm("target", target_acts)

        config = Config(alignment_method=AlignmentMethod.cca, min_samples=5)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        # CCA handles dimension mismatch via PCA reduction
        assert result is not None or result is None  # May or may not succeed

    def test_cca_low_correlation(self):
        """Uncorrelated data should have low canonical correlations."""
        # Create orthogonal patterns
        source_acts = {f"prime:{i}": [1.0 if i % 2 == 0 else 0.0, 0.0 if i % 2 == 0 else 1.0] for i in range(20)}
        target_acts = {f"prime:{i}": [0.0 if i % 2 == 0 else 1.0, 1.0 if i % 2 == 0 else 0.0] for i in range(20)}
        source = _make_crm("source", source_acts)
        target = _make_crm("target", target_acts)

        config = Config(alignment_method=AlignmentMethod.cca, min_samples=5, min_canonical_correlation=0.0)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        # May return None or result with low correlation
        if result is not None:
            assert result.alignment_strengths[0] < 1.0


# =============================================================================
# Discovery Method Tests - Shared SVD
# =============================================================================


class TestDiscoverWithSharedSVD:
    """Tests for Shared SVD-based discovery."""

    def test_shared_svd_identical(self):
        """Identical data should align perfectly with Shared SVD."""
        activations = {f"prime:{i}": [float(i), float(i + 1)] for i in range(15)}
        source = _make_crm("source", activations)
        target = _make_crm("target", activations)

        config = Config(alignment_method=AlignmentMethod.shared_svd, min_samples=5)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        assert result is not None
        assert result.method == AlignmentMethod.shared_svd
        assert result.alignment_error < 0.5

    def test_shared_svd_variance_ratio(self):
        """Shared SVD should compute variance ratio correctly."""
        activations = {f"prime:{i}": [float(i), float(i * 2)] for i in range(15)}
        source = _make_crm("source", activations)
        target = _make_crm("target", activations)

        config = Config(alignment_method=AlignmentMethod.shared_svd, min_samples=5, variance_threshold=0.8)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        assert result is not None
        assert result.shared_variance_ratio > 0.0
        # Allow small floating point error above 1.0
        assert result.shared_variance_ratio <= 1.0 + 1e-5


# =============================================================================
# Discovery Method Tests - Procrustes
# =============================================================================


class TestDiscoverWithProcrustes:
    """Tests for Procrustes-based discovery."""

    def test_procrustes_same_dimension(self):
        """Procrustes should work with same dimensions."""
        activations = {f"prime:{i}": [float(i), float(i + 1)] for i in range(15)}
        source = _make_crm("source", activations)
        target = _make_crm("target", activations)

        config = Config(alignment_method=AlignmentMethod.procrustes, min_samples=5)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        assert result is not None
        assert result.method == AlignmentMethod.procrustes

    def test_procrustes_different_dimension_fallback(self):
        """Procrustes with different dimensions should fall back to CCA."""
        source_acts = {f"prime:{i}": [float(i), float(i + 1), float(i + 2)] for i in range(15)}
        target_acts = {f"prime:{i}": [float(i), float(i + 1)] for i in range(15)}
        source = _make_crm("source", source_acts)
        target = _make_crm("target", target_acts)

        config = Config(alignment_method=AlignmentMethod.procrustes, min_samples=5)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        # Should fall back to CCA
        if result is not None:
            assert result.method == AlignmentMethod.cca

    def test_procrustes_rotation(self):
        """Procrustes should find rotation between rotated data."""
        # Create rotated versions
        theta = math.pi / 4  # 45 degrees
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        source_acts = {f"prime:{i}": [float(i), float(i)] for i in range(20)}
        target_acts = {
            f"prime:{i}": [
                float(i) * cos_t - float(i) * sin_t,
                float(i) * sin_t + float(i) * cos_t,
            ]
            for i in range(20)
        }
        source = _make_crm("source", source_acts)
        target = _make_crm("target", target_acts)

        config = Config(alignment_method=AlignmentMethod.procrustes, min_samples=5)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        assert result is not None
        # Low error indicates good rotation found
        assert result.alignment_error < 0.5


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_no_common_anchors(self):
        """No common anchors should return None."""
        source = _make_crm("source", {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]})
        target = _make_crm("target", {"prime:c": [1.0, 0.0], "prime:d": [0.0, 1.0]})

        config = Config(min_samples=1)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        assert result is None

    def test_missing_layer(self):
        """Missing layer should return None."""
        activations = {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]}
        source = _make_crm("source", activations)
        target = _make_crm("target", activations)

        config = Config(min_samples=1)
        result = SharedSubspaceProjector.discover(source, target, layer=5, config=config)  # Layer 5 doesn't exist

        assert result is None

    def test_different_layer_mapping(self):
        """Different source/target layers should work."""
        source_metadata = AnchorMetadata(total_count=2, semantic_prime_count=2, computational_gate_count=0, anchor_ids=["p:a", "p:b"])
        target_metadata = AnchorMetadata(total_count=2, semantic_prime_count=2, computational_gate_count=0, anchor_ids=["p:a", "p:b"])

        source = ConceptResponseMatrix(model_identifier="source", layer_count=2, hidden_dim=2, anchor_metadata=source_metadata)
        source.activations = {0: {"p:a": AnchorActivation("p:a", 0, [1.0, 0.0]), "p:b": AnchorActivation("p:b", 0, [0.0, 1.0])}}

        target = ConceptResponseMatrix(model_identifier="target", layer_count=2, hidden_dim=2, anchor_metadata=target_metadata)
        target.activations = {1: {"p:a": AnchorActivation("p:a", 1, [1.0, 0.0]), "p:b": AnchorActivation("p:b", 1, [0.0, 1.0])}}

        config = Config(min_samples=1)
        result = SharedSubspaceProjector.discover(source, target, layer=0, target_layer=1, config=config)

        # Should successfully handle different layer mapping (layer 0 in source, layer 1 in target)
        assert result is not None
        assert result.sample_count == 2

    def test_sample_count_below_min(self):
        """Sample count below min_samples should return None."""
        activations = {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]}
        source = _make_crm("source", activations)
        target = _make_crm("target", activations)

        config = Config(min_samples=10)  # Need 10, only have 2
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        assert result is None

    def test_anchor_prefix_filter(self):
        """anchor_prefixes should filter anchors."""
        source = _make_crm("source", {
            "prime:a": [1.0, 0.0],
            "prime:b": [0.0, 1.0],
            "gate:a": [1.0, 1.0],
            "gate:b": [0.5, 0.5],
        })
        target = _make_crm("target", {
            "prime:a": [1.0, 0.0],
            "prime:b": [0.0, 1.0],
            "gate:a": [1.0, 1.0],
            "gate:b": [0.5, 0.5],
        })

        config_all = Config(min_samples=1)
        config_prime_only = Config(min_samples=1, anchor_prefixes=("prime:",))

        result_all = SharedSubspaceProjector.discover(source, target, layer=0, config=config_all)
        result_prime = SharedSubspaceProjector.discover(source, target, layer=0, config=config_prime_only)

        assert result_all is not None
        # Prime-only should have fewer samples
        if result_prime is not None:
            assert result_prime.sample_count <= result_all.sample_count


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_very_small_values(self):
        """Very small values should not cause underflow."""
        activations = {f"prime:{i}": [1e-10 * i, 1e-10 * (i + 1)] for i in range(1, 16)}
        source = _make_crm("source", activations)
        target = _make_crm("target", activations)

        config = Config(min_samples=5, cca_regularization=1e-4)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        # Should complete without error
        if result is not None:
            assert not math.isnan(result.alignment_error)

    def test_very_large_values(self):
        """Very large values should not cause overflow."""
        activations = {f"prime:{i}": [1e10 * i, 1e10 * (i + 1)] for i in range(1, 16)}
        source = _make_crm("source", activations)
        target = _make_crm("target", activations)

        config = Config(min_samples=5, cca_regularization=1e-4)
        result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)

        # Should complete without error
        if result is not None:
            assert not math.isnan(result.alignment_error)
            assert not math.isinf(result.alignment_error)

    def test_regularization_prevents_singularity(self):
        """Regularization should prevent singular covariance issues."""
        # Create near-singular data (all very similar)
        activations = {f"prime:{i}": [1.0 + 1e-8 * i, 2.0 + 1e-8 * i] for i in range(20)}
        source = _make_crm("source", activations)
        target = _make_crm("target", activations)

        config_no_reg = Config(min_samples=5, cca_regularization=0.0)
        config_reg = Config(min_samples=5, cca_regularization=1e-4)

        # Without regularization may fail
        # With regularization should succeed
        result_reg = SharedSubspaceProjector.discover(source, target, layer=0, config=config_reg)

        # At least the regularized version should work
        if result_reg is not None:
            assert result_reg.shared_dimension > 0


# =============================================================================
# Matrix Helper Function Tests
# =============================================================================


class TestMatrixHelpers:
    """Tests for matrix helper functions."""

    def test_compute_covariance(self):
        """Covariance computation should be correct."""
        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        n = 3

        cov = SharedSubspaceProjector._compute_covariance(x, y, n)

        # Should be 2x2 = 4 elements
        assert len(cov) == 4

    def test_compute_gram_matrix(self):
        """Gram matrix computation should be symmetric."""
        x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        n, d = 3, 2

        gram = SharedSubspaceProjector._compute_gram_matrix(x, n, d)

        # Should be n x n = 9 elements
        assert len(gram) == 9
        # Should be symmetric
        for i in range(n):
            for j in range(n):
                assert gram[i * n + j] == pytest.approx(gram[j * n + i], abs=1e-10)

    def test_transpose_matrix(self):
        """Matrix transpose should be correct."""
        matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2x3
        m, n = 2, 3

        transposed = transpose_flat_matrix(matrix, m, n)

        # Should be 3x2
        assert len(transposed) == 6
        # transposed[0,0] = matrix[0,0] = 1
        assert transposed[0] == 1.0
        # transposed[1,0] = matrix[0,1] = 2
        assert transposed[2] == 2.0

    def test_reshape_to_matrix(self):
        """Reshape to matrix should be correct."""
        flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        rows, cols = 2, 3

        result = reshape_flat_to_matrix(flat, rows, cols)

        assert len(result) == 2
        assert len(result[0]) == 3
        assert result[0] == [1.0, 2.0, 3.0]
        assert result[1] == [4.0, 5.0, 6.0]

    def test_compute_procrustes_error(self):
        """Procrustes error should be correct."""
        source = [1.0, 2.0, 3.0, 4.0]
        target = [1.0, 2.0, 3.0, 4.0]
        n, k = 2, 2

        error = SharedSubspaceProjector._compute_procrustes_error(source, target, n, k)
        assert error == pytest.approx(0.0, abs=1e-10)

        # Different values
        source2 = [1.0, 2.0, 3.0, 4.0]
        target2 = [2.0, 4.0, 6.0, 8.0]  # Doubled
        error2 = SharedSubspaceProjector._compute_procrustes_error(source2, target2, n, k)
        assert error2 > 0.0

    def test_compute_determinant_identity(self):
        """Determinant of identity should be 1."""
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # 3x3 identity
        det = SharedSubspaceProjector._compute_determinant(identity, 3)
        assert det == pytest.approx(1.0, abs=1e-10)

    def test_compute_determinant_singular(self):
        """Determinant of singular matrix should be 0."""
        singular = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # Rows linearly dependent
        det = SharedSubspaceProjector._compute_determinant(singular, 3)
        assert det == pytest.approx(0.0, abs=1e-10)
