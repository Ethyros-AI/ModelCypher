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

"""Tests for the vocabulary module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.experimental.vocabulary.embedding_projector import (
    EmbeddingProjector,
    ProjectionResult,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def _div_eps(*values: float) -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


@pytest.fixture
def backend() -> "Backend":
    """Get the default backend."""
    return get_default_backend()


class TestProjectionResult:
    """Tests for ProjectionResult dataclass."""

    def test_to_dict_contains_required_fields(self, backend: "Backend") -> None:
        """to_dict should contain required summary fields."""
        embeddings = backend.random_normal((100, 64))
        backend.eval(embeddings)

        result = ProjectionResult(
            projected_embeddings=embeddings,
            projection_matrix=None,
            reconstruction_error=0.5,
            metadata={"test_key": "test_value"},
        )

        d = result.to_dict()

        eps = _div_eps(d["reconstruction_error"])
        assert abs(d["reconstruction_error"] - 0.5) < eps
        assert d["output_shape"] == [100, 64]
        assert d["has_projection_matrix"] is False
        assert d["test_key"] == "test_value"


class TestEmbeddingProjector:
    """Tests for deterministic projection behavior."""

    def test_projection_returns_result(self, backend: "Backend") -> None:
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(backend=backend)
        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        assert result.projection_matrix is not None

    def test_projection_with_shared_indices(self, backend: "Backend") -> None:
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(backend=backend)
        shared_indices = (list(range(50)), list(range(50)))
        result = projector.project(source, target, shared_indices)

        assert result.metadata["n_anchors"] == 50


class TestProjectionMetricsComputation:
    """Tests for projection metrics."""

    def test_compute_projection_metrics(self, backend: "Backend") -> None:
        """Should compute multiple projection metrics."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(backend=backend)
        result = projector.project(source, target)

        quality = projector.compute_projection_metrics(
            source, result.projected_embeddings, target
        )

        assert "mse" in quality
        assert "mean_cosine_similarity" in quality
        assert "norm_preservation_ratio" in quality
        assert "n_samples_evaluated" in quality
        assert quality["n_samples_evaluated"] == 100

    def test_projection_metrics_with_shared_indices(self, backend: "Backend") -> None:
        """Projection metrics should use shared indices when provided."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        projected = backend.random_normal((100, 64))
        backend.eval(source, target, projected)

        shared_indices = (list(range(30)), list(range(30)))

        projector = EmbeddingProjector(backend=backend)

        quality = projector.compute_projection_metrics(
            source, projected, target, shared_indices=shared_indices
        )

        assert quality["n_samples_evaluated"] == 30
