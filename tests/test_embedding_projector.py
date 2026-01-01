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
from modelcypher.core.domain.vocabulary.embedding_projector import (
    EmbeddingProjector,
    ProjectionConfig,
    ProjectionResult,
    ProjectionStrategy,
)


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


class TestProjectionConfig:
    """Tests for ProjectionConfig dataclass."""

    def test_defaults(self):
        config = ProjectionConfig()
        assert config.strategy == ProjectionStrategy.PROCRUSTES
        assert config.regularization is None
        assert config.n_components is None
        assert config.preserve_norms is True
        assert config.anchor_count == 1000

    def test_custom_strategy(self):
        config = ProjectionConfig(strategy=ProjectionStrategy.PCA)
        assert config.strategy == ProjectionStrategy.PCA

    def test_custom_regularization(self):
        config = ProjectionConfig(regularization=1e-4)
        assert config.regularization == 1e-4

    def test_custom_n_components(self):
        config = ProjectionConfig(n_components=128)
        assert config.n_components == 128

    def test_custom_preserve_norms(self):
        config = ProjectionConfig(preserve_norms=False)
        assert config.preserve_norms is False

    def test_custom_anchor_count(self):
        config = ProjectionConfig(anchor_count=500)
        assert config.anchor_count == 500


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
        assert sample_result.reconstruction_error == 0.1
        assert sample_result.alignment_score == 0.95
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
        assert result.metadata["explained_variance"] == 0.95

    def test_to_dict_contains_required_fields(self, sample_result):
        d = sample_result.to_dict()
        assert "reconstruction_error" in d
        assert "alignment_score" in d
        assert "strategy_used" in d
        assert "output_shape" in d
        assert "has_projection_matrix" in d

    def test_to_dict_values(self, sample_result):
        d = sample_result.to_dict()
        assert d["reconstruction_error"] == 0.1
        assert d["alignment_score"] == 0.95
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
        assert projector.config.strategy == ProjectionStrategy.PROCRUSTES
        assert projector._backend is not None

    def test_custom_config(self):
        config = ProjectionConfig(strategy=ProjectionStrategy.PCA)
        projector = EmbeddingProjector(config=config)
        assert projector.config.strategy == ProjectionStrategy.PCA

    def test_custom_backend(self):
        backend = get_default_backend()
        projector = EmbeddingProjector(backend=backend)
        assert projector._backend is backend


class TestEmbeddingProjectorTruncate:
    """Tests for TRUNCATE projection strategy."""

    @pytest.fixture
    def projector(self):
        config = ProjectionConfig(strategy=ProjectionStrategy.TRUNCATE)
        return EmbeddingProjector(config=config)

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_same_dimension_no_change(self, projector, backend):
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.strategy_used == ProjectionStrategy.TRUNCATE
        assert result.reconstruction_error == 0.0
        assert result.alignment_score == 1.0
        assert result.projected_embeddings.shape == (100, 64)

    def test_truncate_larger_source(self, projector, backend):
        source = backend.random_normal((100, 128))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        assert result.reconstruction_error > 0  # Lost some info
        assert result.alignment_score == 0.5  # 64/128

    def test_pad_smaller_source(self, projector, backend):
        source = backend.random_normal((100, 32))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        assert result.reconstruction_error == 0.0  # Padding adds zeros
        assert result.alignment_score == 0.5  # 32/64


class TestEmbeddingProjectorPCA:
    """Tests for PCA projection strategy."""

    @pytest.fixture
    def projector(self):
        config = ProjectionConfig(strategy=ProjectionStrategy.PCA)
        return EmbeddingProjector(config=config)

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
        assert 0.0 <= result.alignment_score <= 1.0

    def test_pca_with_dimension_reduction(self, backend):
        config = ProjectionConfig(strategy=ProjectionStrategy.PCA, n_components=32)
        projector = EmbeddingProjector(config=config)

        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 32))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 32)
        assert "n_components" in result.metadata

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
        config = ProjectionConfig(strategy=ProjectionStrategy.PROCRUSTES)
        return EmbeddingProjector(config=config)

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
        assert result.metadata["n_anchors"] <= 50

    def test_procrustes_cross_dimension(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 128))
        result = projector.project(source, target)

        # Should pad/truncate to target dimension
        assert result.projected_embeddings.shape == (100, 128)

    def test_procrustes_preserves_norms_when_configured(self, backend):
        config = ProjectionConfig(
            strategy=ProjectionStrategy.PROCRUSTES, preserve_norms=True
        )
        projector = EmbeddingProjector(config=config)

        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        # Check norms are approximately preserved
        source_norms = backend.norm(source, axis=1)
        projected_norms = backend.norm(result.projected_embeddings, axis=1)
        ratio = backend.mean(projected_norms / source_norms)
        assert abs(float(backend.to_numpy(ratio)) - 1.0) < 0.5


class TestEmbeddingProjectorCCA:
    """Tests for CCA projection strategy."""

    @pytest.fixture
    def projector(self):
        config = ProjectionConfig(strategy=ProjectionStrategy.CCA)
        return EmbeddingProjector(config=config)

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
        config = ProjectionConfig(
            strategy=ProjectionStrategy.OPTIMAL_TRANSPORT, anchor_count=50
        )
        return EmbeddingProjector(config=config)

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

        assert metrics["mse"] < 1e-6
        assert metrics["mean_cosine_similarity"] > 0.99
        assert abs(metrics["norm_preservation_ratio"] - 1.0) < 0.01


class TestProjectionStrategyDispatch:
    """Tests for strategy dispatch in project() method."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_unknown_strategy_raises(self, backend):
        projector = EmbeddingProjector()
        # Manually set an invalid strategy
        projector.config.strategy = "invalid"  # type: ignore

        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))

        with pytest.raises(ValueError, match="Unknown projection strategy"):
            projector.project(source, target)

    def test_all_strategies_produce_results(self, backend):
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        target = backend.random_normal((50, 32))

        for strategy in ProjectionStrategy:
            config = ProjectionConfig(strategy=strategy, anchor_count=20)
            projector = EmbeddingProjector(config=config)
            result = projector.project(source, target)

            assert result.strategy_used == strategy
            assert result.projected_embeddings.shape[0] == 50


class TestProjectionEdgeCases:
    """Tests for edge cases in embedding projection."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_small_vocab_size(self, backend):
        config = ProjectionConfig(
            strategy=ProjectionStrategy.PROCRUSTES, anchor_count=100
        )
        projector = EmbeddingProjector(config=config)

        backend.random_seed(42)
        source = backend.random_normal((10, 64))  # Very small vocab
        target = backend.random_normal((10, 64))
        result = projector.project(source, target)

        # Should handle small vocab gracefully
        assert result.projected_embeddings.shape == (10, 64)
        assert result.metadata["n_anchors"] <= 10

    def test_large_dimension_mismatch(self, backend):
        config = ProjectionConfig(strategy=ProjectionStrategy.TRUNCATE)
        projector = EmbeddingProjector(config=config)

        backend.random_seed(42)
        source = backend.random_normal((50, 768))
        target = backend.random_normal((50, 4096))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (50, 4096)

    def test_different_vocab_sizes(self, backend):
        config = ProjectionConfig(
            strategy=ProjectionStrategy.PROCRUSTES, anchor_count=20
        )
        projector = EmbeddingProjector(config=config)

        backend.random_seed(42)
        source = backend.random_normal((100, 64))  # 100 tokens
        target = backend.random_normal((200, 64))  # 200 tokens
        result = projector.project(source, target)

        # Should project source to target space (same dimension)
        assert result.projected_embeddings.shape == (100, 64)
