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

"""Tests for embedding projector (cross-vocabulary embedding alignment)."""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.vocabulary.embedding_projector import (
    EmbeddingProjector,
    ProjectionResult,
    ProjectionStrategy,
)


def _div_eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


class TestProjectionStrategy:
    """Tests for ProjectionStrategy enum."""

    def test_truncate_value(self):
        assert ProjectionStrategy.TRUNCATE.value == "truncate"

    def test_pca_value(self):
        assert ProjectionStrategy.PCA.value == "pca"

    def test_procrustes_value(self):
        assert ProjectionStrategy.PROCRUSTES.value == "procrustes"

    def test_cca_value(self):
        assert ProjectionStrategy.CCA.value == "cca"

    def test_optimal_transport_value(self):
        assert ProjectionStrategy.OPTIMAL_TRANSPORT.value == "optimal_transport"

    def test_all_strategies_count(self):
        assert len(ProjectionStrategy) == 5

    def test_string_enum(self):
        # str(Enum) should work
        assert str(ProjectionStrategy.TRUNCATE) == "ProjectionStrategy.TRUNCATE"
        assert ProjectionStrategy.TRUNCATE == "truncate"


class TestProjectionResult:
    """Tests for ProjectionResult dataclass."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    @pytest.fixture
    def sample_result(self, backend):
        embeddings = backend.random_normal((100, 64))
        return ProjectionResult(
            projected_embeddings=embeddings,
            projection_matrix=None,
            reconstruction_error=0.1,
            alignment_score=0.95,
            strategy_used=ProjectionStrategy.TRUNCATE,
        )

    def test_required_fields(self, sample_result):
        assert sample_result.projected_embeddings is not None
        eps = _div_eps()
        assert abs(sample_result.reconstruction_error - 0.1) <= eps
        assert abs(sample_result.alignment_score - 0.95) <= eps
        assert sample_result.strategy_used == ProjectionStrategy.TRUNCATE

    def test_optional_projection_matrix(self, sample_result):
        assert sample_result.projection_matrix is None

    def test_default_metadata(self, sample_result):
        assert sample_result.metadata == {}

    def test_custom_metadata(self, backend):
        embeddings = backend.random_normal((100, 64))
        result = ProjectionResult(
            projected_embeddings=embeddings,
            projection_matrix=None,
            reconstruction_error=0.0,
            alignment_score=1.0,
            strategy_used=ProjectionStrategy.PCA,
            metadata={"n_components": 64, "explained_variance": 0.95},
        )
        assert result.metadata["n_components"] == 64
        assert abs(result.metadata["explained_variance"] - 0.95) <= _div_eps()

    def test_to_dict_contains_required_fields(self, sample_result):
        d = sample_result.to_dict()
        assert "reconstruction_error" in d
        assert "alignment_score" in d
        assert "strategy_used" in d
        assert "output_shape" in d
        assert "has_projection_matrix" in d

    def test_to_dict_values(self, sample_result):
        d = sample_result.to_dict()
        eps = _div_eps()
        assert abs(d["reconstruction_error"] - 0.1) <= eps
        assert abs(d["alignment_score"] - 0.95) <= eps
        assert d["strategy_used"] == "truncate"
        assert d["output_shape"] == [100, 64]
        assert d["has_projection_matrix"] is False

    def test_to_dict_includes_metadata(self, backend):
        embeddings = backend.random_normal((50, 32))
        result = ProjectionResult(
            projected_embeddings=embeddings,
            projection_matrix=None,
            reconstruction_error=0.0,
            alignment_score=1.0,
            strategy_used=ProjectionStrategy.CCA,
            metadata={"custom_key": "custom_value"},
        )
        d = result.to_dict()
        assert d["custom_key"] == "custom_value"


class TestEmbeddingProjectorInit:
    """Tests for EmbeddingProjector initialization."""

    def test_default_init(self):
        projector = EmbeddingProjector()
        assert projector.strategy == ProjectionStrategy.PROCRUSTES
        assert projector._backend is not None

    def test_custom_config(self):
        projector = EmbeddingProjector(strategy=ProjectionStrategy.PCA)
        assert projector.strategy == ProjectionStrategy.PCA

    def test_custom_backend(self):
        backend = get_default_backend()
        projector = EmbeddingProjector(backend=backend)
        assert projector._backend is backend


class TestEmbeddingProjectorTruncate:
    """Tests for TRUNCATE projection strategy."""

    @pytest.fixture
    def projector(self):
        return EmbeddingProjector(strategy=ProjectionStrategy.TRUNCATE)

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_same_dimension_no_change(self, projector, backend):
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.strategy_used == ProjectionStrategy.TRUNCATE
        assert result.projected_embeddings.shape == (100, 64)
        eps = _div_eps()
        assert result.reconstruction_error >= -eps
        assert result.alignment_score >= -1.0 - eps
        assert result.alignment_score <= 1.0 + eps

    def test_truncate_larger_source(self, projector, backend):
        source = backend.random_normal((100, 128))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        eps = _div_eps()
        assert result.reconstruction_error >= -eps
        assert result.alignment_score >= -1.0 - eps
        assert result.alignment_score <= 1.0 + eps

    def test_pad_smaller_source(self, projector, backend):
        source = backend.random_normal((100, 32))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        eps = _div_eps()
        assert result.reconstruction_error >= -eps
        assert result.alignment_score >= -1.0 - eps
        assert result.alignment_score <= 1.0 + eps


class TestEmbeddingProjectorPCA:
    """Tests for PCA projection strategy."""

    @pytest.fixture
    def projector(self):
        return EmbeddingProjector(strategy=ProjectionStrategy.PCA)

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_pca_returns_result(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.strategy_used == ProjectionStrategy.PCA
        assert result.projected_embeddings.shape == (100, 64)
        eps = _div_eps()
        assert result.alignment_score >= -1.0 - eps
        assert result.alignment_score <= 1.0 + eps

    def test_pca_with_dimension_reduction(self, backend):
        projector = EmbeddingProjector(strategy=ProjectionStrategy.PCA)

        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 32))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 32)
        assert result.metadata["n_components"] == 32

    def test_pca_explained_variance_in_metadata(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert "explained_variance_ratio" in result.metadata


class TestEmbeddingProjectorProcrustes:
    """Tests for PROCRUSTES projection strategy."""

    @pytest.fixture
    def projector(self):
        return EmbeddingProjector(strategy=ProjectionStrategy.PROCRUSTES)

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_procrustes_returns_result(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.strategy_used == ProjectionStrategy.PROCRUSTES
        assert result.projected_embeddings.shape == (100, 64)
        assert result.projection_matrix is not None

    def test_procrustes_with_shared_indices(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        shared_indices = (list(range(50)), list(range(50)))
        result = projector.project(source, target, shared_indices)

        assert result.strategy_used == ProjectionStrategy.PROCRUSTES
        assert "n_anchors" in result.metadata
        assert result.metadata["n_anchors"] == len(shared_indices[0])

    def test_procrustes_cross_dimension(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 128))
        result = projector.project(source, target)

        # Should pad/truncate to target dimension
        assert result.projected_embeddings.shape == (100, 128)

    def test_procrustes_identity_alignment(self, backend):
        projector = EmbeddingProjector(strategy=ProjectionStrategy.PROCRUSTES)

        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        result = projector.project(source, source)

        eps = _div_eps()
        assert result.reconstruction_error <= eps
        assert abs(result.alignment_score - 1.0) <= eps


class TestEmbeddingProjectorCCA:
    """Tests for CCA projection strategy."""

    @pytest.fixture
    def projector(self):
        return EmbeddingProjector(strategy=ProjectionStrategy.CCA)

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_cca_returns_result(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.strategy_used == ProjectionStrategy.CCA
        assert result.projected_embeddings.shape == (100, 64)

    def test_cca_with_shared_indices(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        shared_indices = (list(range(50)), list(range(50)))
        result = projector.project(source, target, shared_indices)

        assert result.strategy_used == ProjectionStrategy.CCA
        assert "n_components" in result.metadata

    def test_cca_canonical_correlations_in_metadata(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert "canonical_correlations" in result.metadata


class TestEmbeddingProjectorOptimalTransport:
    """Tests for OPTIMAL_TRANSPORT projection strategy."""

    @pytest.fixture
    def projector(self):
        return EmbeddingProjector(strategy=ProjectionStrategy.OPTIMAL_TRANSPORT)

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_optimal_transport_returns_result(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.strategy_used == ProjectionStrategy.OPTIMAL_TRANSPORT
        assert result.projected_embeddings.shape == (100, 64)

    def test_optimal_transport_metadata(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert "n_source_samples" in result.metadata
        assert "n_target_samples" in result.metadata
        assert "transport_cost" in result.metadata
        assert "shared_dim" in result.metadata


class TestEmbeddingProjectorAlignmentQuality:
    """Tests for compute_alignment_quality method."""

    @pytest.fixture
    def projector(self):
        return EmbeddingProjector()

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_quality_metrics_returned(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        projected = backend.random_normal((100, 64))

        metrics = projector.compute_alignment_quality(source, projected, target)

        assert "mse" in metrics
        assert "mean_cosine_similarity" in metrics
        assert "norm_preservation_ratio" in metrics
        assert "n_samples_evaluated" in metrics

    def test_quality_with_shared_indices(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        projected = backend.random_normal((100, 64))
        shared_indices = (list(range(50)), list(range(50)))

        metrics = projector.compute_alignment_quality(
            source, projected, target, shared_indices
        )

        assert metrics["n_samples_evaluated"] == 50

    def test_perfect_alignment_metrics(self, projector, backend):
        backend.random_seed(42)
        embeddings = backend.random_normal((100, 64))

        metrics = projector.compute_alignment_quality(embeddings, embeddings, embeddings)

        eps = _div_eps()
        assert metrics["mse"] <= eps
        assert abs(metrics["mean_cosine_similarity"] - 1.0) <= eps
        assert abs(metrics["norm_preservation_ratio"] - 1.0) <= eps


class TestProjectionStrategyDispatch:
    """Tests for strategy dispatch in project() method."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_unknown_strategy_raises(self, backend):
        projector = EmbeddingProjector(
            strategy="invalid"  # type: ignore[arg-type]
        )

        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))

        with pytest.raises(ValueError, match="Unknown projection strategy"):
            projector.project(source, target)

    def test_all_strategies_produce_results(self, backend):
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        target = backend.random_normal((50, 32))

        for strategy in ProjectionStrategy:
            projector = EmbeddingProjector(strategy=strategy)
            result = projector.project(source, target)

            assert result.strategy_used == strategy
            assert result.projected_embeddings.shape[0] == 50


class TestProjectionEdgeCases:
    """Tests for edge cases in embedding projection."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_small_vocab_size(self, backend):
        projector = EmbeddingProjector(strategy=ProjectionStrategy.PROCRUSTES)

        backend.random_seed(42)
        source = backend.random_normal((10, 64))  # Very small vocab
        target = backend.random_normal((10, 64))
        result = projector.project(source, target)

        # Should handle small vocab gracefully
        assert result.projected_embeddings.shape == (10, 64)
        assert result.metadata["n_anchors"] == 10

    def test_large_dimension_mismatch(self, backend):
        projector = EmbeddingProjector(strategy=ProjectionStrategy.TRUNCATE)

        backend.random_seed(42)
        source = backend.random_normal((50, 768))
        target = backend.random_normal((50, 4096))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (50, 4096)

    def test_different_vocab_sizes(self, backend):
        projector = EmbeddingProjector(strategy=ProjectionStrategy.PROCRUSTES)

        backend.random_seed(42)
        source = backend.random_normal((100, 64))  # 100 tokens
        target = backend.random_normal((200, 64))  # 200 tokens
        result = projector.project(source, target)

        # Should project source to target space (same dimension)
        assert result.projected_embeddings.shape == (100, 64)
