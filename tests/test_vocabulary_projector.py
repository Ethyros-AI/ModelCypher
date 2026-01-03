# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the vocabulary module.

Tests EmbeddingProjector projection strategies and CrossVocabMerger operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.vocabulary.embedding_projector import (
    EmbeddingProjector,
    ProjectionResult,
    ProjectionStrategy,
)
from modelcypher.core.domain.vocabulary.cross_vocab_merger import (
    AlignmentMethod,
    CrossVocabMerger,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@pytest.fixture
def backend() -> "Backend":
    """Get the default backend."""
    return get_default_backend()


def _div_eps(*values: float) -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


# =============================================================================
# EmbeddingProjector Tests
# =============================================================================


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
            alignment_score=0.85,
            strategy_used=ProjectionStrategy.TRUNCATE,
            metadata={"test_key": "test_value"},
        )

        d = result.to_dict()

        eps = _div_eps(d["reconstruction_error"], d["alignment_score"])
        assert abs(d["reconstruction_error"] - 0.5) < eps
        assert abs(d["alignment_score"] - 0.85) < eps
        assert d["strategy_used"] == "truncate"
        assert d["output_shape"] == [100, 64]
        assert d["has_projection_matrix"] is False
        assert d["test_key"] == "test_value"


class TestTruncateStrategy:
    """Tests for TRUNCATE projection strategy."""

    def test_same_dimensions_returns_unchanged(self, backend: "Backend") -> None:
        """Same dimensions should return source unchanged."""
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(
            strategy=ProjectionStrategy.TRUNCATE, backend=backend
        )

        result = projector.project(source, target)

        assert result.projected_embeddings.shape == source.shape
        assert result.reconstruction_error >= 0.0
        assert -1.0 <= result.alignment_score <= 1.0

    def test_larger_source_truncates(self, backend: "Backend") -> None:
        """Larger source dimension should be truncated."""
        source = backend.random_normal((100, 128))  # Larger
        target = backend.random_normal((100, 64))  # Smaller
        backend.eval(source, target)

        projector = EmbeddingProjector(
            strategy=ProjectionStrategy.TRUNCATE, backend=backend
        )

        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        assert result.reconstruction_error >= 0.0
        assert -1.0 <= result.alignment_score <= 1.0

    def test_smaller_source_pads(self, backend: "Backend") -> None:
        """Smaller source dimension should be zero-padded."""
        source = backend.random_normal((100, 32))  # Smaller
        target = backend.random_normal((100, 64))  # Larger
        backend.eval(source, target)

        projector = EmbeddingProjector(
            strategy=ProjectionStrategy.TRUNCATE, backend=backend
        )

        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        assert result.reconstruction_error >= 0.0
        assert -1.0 <= result.alignment_score <= 1.0


class TestPCAStrategy:
    """Tests for PCA projection strategy."""

    def test_pca_reduces_dimensions(self, backend: "Backend") -> None:
        """PCA should reduce to target dimension."""
        backend.random_seed(42)
        source = backend.random_normal((100, 128))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(strategy=ProjectionStrategy.PCA, backend=backend)

        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        assert -1.0 <= result.alignment_score <= 1.0

    def test_pca_with_n_components(self, backend: "Backend") -> None:
        """PCA with explicit n_components."""
        backend.random_seed(42)
        source = backend.random_normal((100, 128))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(strategy=ProjectionStrategy.PCA, backend=backend)

        result = projector.project(source, target)

        # Output is padded/truncated to target_dim
        assert result.projected_embeddings.shape == (100, 64)
        assert result.metadata["n_components"] == 64


class TestProcrustesStrategy:
    """Tests for Procrustes projection strategy."""

    def test_procrustes_produces_rotation(self, backend: "Backend") -> None:
        """Procrustes should produce an orthogonal rotation matrix."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(
            strategy=ProjectionStrategy.PROCRUSTES, backend=backend
        )

        result = projector.project(source, target)

        assert result.projection_matrix is not None
        assert result.projected_embeddings.shape == (100, 64)
        assert -1.0 <= result.alignment_score <= 1.0

    def test_procrustes_with_shared_indices(self, backend: "Backend") -> None:
        """Procrustes with explicit shared token indices."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        shared_indices = (list(range(50)), list(range(50)))

        projector = EmbeddingProjector(
            strategy=ProjectionStrategy.PROCRUSTES, backend=backend
        )

        result = projector.project(source, target, shared_token_indices=shared_indices)

        assert result.metadata["n_anchors"] == 50

    def test_procrustes_cross_dimension(self, backend: "Backend") -> None:
        """Procrustes with different dimensions pads smaller."""
        backend.random_seed(42)
        source = backend.random_normal((100, 48))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(
            strategy=ProjectionStrategy.PROCRUSTES, backend=backend
        )

        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)


class TestCCAStrategy:
    """Tests for CCA projection strategy."""

    def test_cca_produces_canonical_projection(self, backend: "Backend") -> None:
        """CCA should produce canonical correlation-based projection."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(strategy=ProjectionStrategy.CCA, backend=backend)

        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        assert "canonical_correlations" in result.metadata
        expected = min(source.shape[1], target.shape[1])
        assert len(result.metadata["canonical_correlations"]) == expected

    def test_cca_cross_dimension(self, backend: "Backend") -> None:
        """CCA with different dimensions."""
        backend.random_seed(42)
        source = backend.random_normal((100, 32))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(strategy=ProjectionStrategy.CCA, backend=backend)

        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)


class TestOptimalTransportStrategy:
    """Tests for Optimal Transport projection strategy."""

    def test_ot_produces_transport_plan(self, backend: "Backend") -> None:
        """Optimal transport should compute transport-based alignment."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(
            strategy=ProjectionStrategy.OPTIMAL_TRANSPORT, backend=backend
        )

        result = projector.project(source, target)

        assert result.projected_embeddings.shape == (100, 64)
        assert "transport_cost" in result.metadata
        assert result.metadata["n_source_samples"] == 100


class TestAlignmentQualityComputation:
    """Tests for alignment quality metrics."""

    def test_compute_alignment_quality(self, backend: "Backend") -> None:
        """Should compute multiple quality metrics."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        projector = EmbeddingProjector(
            strategy=ProjectionStrategy.PROCRUSTES, backend=backend
        )

        result = projector.project(source, target)

        quality = projector.compute_alignment_quality(
            source, result.projected_embeddings, target
        )

        assert "mse" in quality
        assert "mean_cosine_similarity" in quality
        assert "norm_preservation_ratio" in quality
        assert "n_samples_evaluated" in quality
        assert quality["n_samples_evaluated"] == 100

    def test_alignment_quality_with_shared_indices(self, backend: "Backend") -> None:
        """Alignment quality should use shared indices when provided."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        projected = backend.random_normal((100, 64))
        backend.eval(source, target, projected)

        shared_indices = (list(range(30)), list(range(30)))

        projector = EmbeddingProjector(backend=backend)

        quality = projector.compute_alignment_quality(
            source, projected, target, shared_indices=shared_indices
        )

        assert quality["n_samples_evaluated"] == 30


# =============================================================================
# CrossVocabMerger Tests
# =============================================================================

class TestCrossVocabMerger:
    """Tests for CrossVocabMerger."""

    def test_merge_same_dimensions(self, backend: "Backend") -> None:
        """Merge with same dimensions should work."""
        backend.random_seed(42)
        source = backend.random_normal((1000, 64))
        target = backend.random_normal((1000, 64))
        backend.eval(source, target)

        merger = CrossVocabMerger(
            projection_strategy=ProjectionStrategy.TRUNCATE, backend=backend
        )

        result = merger.merge(source, target)

        assert result.output_vocab_size == 1000
        assert result.output_hidden_dim == 64
        assert result.merged_embeddings.shape == (1000, 64)

    def test_merge_different_dimensions(self, backend: "Backend") -> None:
        """Merge with different dimensions should project."""
        backend.random_seed(42)
        source = backend.random_normal((1000, 48))
        target = backend.random_normal((1000, 64))
        backend.eval(source, target)

        merger = CrossVocabMerger(
            projection_strategy=ProjectionStrategy.PROCRUSTES, backend=backend
        )

        result = merger.merge(source, target)

        assert result.output_vocab_size == 1000
        assert result.output_hidden_dim == 64
        assert result.merged_embeddings.shape == (1000, 64)

    def test_merge_different_vocab_sizes(self, backend: "Backend") -> None:
        """Merge with different vocab sizes uses index alignment."""
        backend.random_seed(42)
        source = backend.random_normal((500, 64))
        target = backend.random_normal((1000, 64))
        backend.eval(source, target)

        merger = CrossVocabMerger(
            projection_strategy=ProjectionStrategy.TRUNCATE, backend=backend
        )

        result = merger.merge(source, target)

        # Output uses target vocab size
        assert result.output_vocab_size == 1000
        assert result.output_hidden_dim == 64
        assert result.alignment_method == AlignmentMethod.INDEX

    def test_merge_with_vocab_dicts(self, backend: "Backend") -> None:
        """Merge with vocab dicts uses string-based alignment."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        source_vocab = {f"token_{i}": i for i in range(100)}
        target_vocab = {f"token_{i}": i for i in range(100)}

        merger = CrossVocabMerger(
            projection_strategy=ProjectionStrategy.TRUNCATE, backend=backend
        )

        result = merger.merge(source, target, source_vocab, target_vocab)

        assert result.output_vocab_size == 100
        # Should find exact matches
        assert result.alignment_map.exact_matches == 100

    def test_merge_result_to_dict(self, backend: "Backend") -> None:
        """Merge result should serialize to dict."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        merger = CrossVocabMerger(
            projection_strategy=ProjectionStrategy.TRUNCATE, backend=backend
        )

        result = merger.merge(source, target)
        d = result.to_dict()

        assert "output_vocab_size" in d
        assert "output_hidden_dim" in d
        assert "alignment_summary" in d
        assert "projection_summary" in d
        assert "vocabulary_alignment" in d

    def test_analyze_merge_quality(self, backend: "Backend") -> None:
        """Should analyze merge quality."""
        backend.random_seed(42)
        source = backend.random_normal((100, 64))
        target = backend.random_normal((100, 64))
        backend.eval(source, target)

        merger = CrossVocabMerger(
            projection_strategy=ProjectionStrategy.PROCRUSTES, backend=backend
        )

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert "alignment_coverage" in quality
        assert "alignment_confidence" in quality
        assert "projection_alignment_score" in quality
        assert "alignment_score" in quality


class TestSpecialTokenHandling:
    """Tests for special token preservation."""

    def test_is_special_token(self, backend: "Backend") -> None:
        """Should recognize special token patterns."""
        merger = CrossVocabMerger(backend=backend)

        # Special tokens
        assert merger._is_special_token("<|endoftext|>") is True
        assert merger._is_special_token("<s>") is True
        assert merger._is_special_token("</s>") is True
        assert merger._is_special_token("<pad>") is True
        assert merger._is_special_token("[CLS]") is True
        assert merger._is_special_token("[SEP]") is True
        assert merger._is_special_token("<bos>") is True
        assert merger._is_special_token("<eos>") is True

        # Regular tokens
        assert merger._is_special_token("hello") is False
        assert merger._is_special_token("world") is False
        assert merger._is_special_token("the") is False
