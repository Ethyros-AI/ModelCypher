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
)


def _div_eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


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
            metadata={"quality": "raw"},
        )

    def test_required_fields(self, sample_result):
        assert sample_result.projected_embeddings is not None
        eps = _div_eps()
        assert abs(sample_result.reconstruction_error - 0.1) <= eps

    def test_optional_projection_matrix(self, sample_result):
        assert sample_result.projection_matrix is None

    def test_default_metadata(self, sample_result):
        assert sample_result.metadata == {"quality": "raw"}

    def test_to_dict_contains_required_fields(self, sample_result):
        d = sample_result.to_dict()
        assert "reconstruction_error" in d
        assert "output_shape" in d
        assert "has_projection_matrix" in d

    def test_to_dict_values(self, sample_result):
        d = sample_result.to_dict()
        eps = _div_eps()
        assert abs(d["reconstruction_error"] - 0.1) <= eps
        assert d["output_shape"] == [100, 64]
        assert d["has_projection_matrix"] is False

    def test_to_dict_includes_metadata(self, backend):
        embeddings = backend.random_normal((50, 32))
        result = ProjectionResult(
            projected_embeddings=embeddings,
            projection_matrix=None,
            reconstruction_error=0.0,
            metadata={"custom_key": "custom_value"},
        )
        d = result.to_dict()
        assert d["custom_key"] == "custom_value"


class TestEmbeddingProjectorInit:
    """Tests for EmbeddingProjector initialization."""

    def test_default_init(self):
        projector = EmbeddingProjector()
        assert projector._backend is not None

    def test_custom_backend(self):
        backend = get_default_backend()
        projector = EmbeddingProjector(backend=backend)
        assert projector._backend is backend


class TestEmbeddingProjector:
    """Tests for deterministic projection behavior."""

    @pytest.fixture
    def projector(self):
        return EmbeddingProjector()

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_projection_returns_result(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        assert result.projection_matrix is not None

    def test_projection_with_shared_indices(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        shared_indices = (list(range(50)), list(range(50)))
        result = projector.project(source, target, shared_indices)

        assert result.metadata["n_anchors"] == len(shared_indices[0])

    def test_cross_dimension_projection(self, projector, backend):
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 128))
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 128)


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
