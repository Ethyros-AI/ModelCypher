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

"""Extended tests for relative representation computation.

Tests critical APIs:
- compute_relative_representation(): Anchor-relative cosine similarities
- align_relative_representations(): Procrustes alignment in anchor space
- transfer_via_relative_space(): Full transfer pipeline
- RelativeRepresentation: Data class properties
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.relative_representation import (
    compute_relative_representation,
    align_relative_representations,
    transfer_via_relative_space,
    RelativeRepresentation,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestComputeRelativeRepresentation:
    """Tests for compute_relative_representation()."""

    def test_basic_computation(self, backend):
        """Basic relative representation should work."""
        hidden_states = backend.random_normal((16, 32))
        anchor_embeddings = backend.random_normal((8, 32))
        backend.eval(hidden_states, anchor_embeddings)

        rel_rep = compute_relative_representation(
            hidden_states, anchor_embeddings, backend=backend
        )

        assert rel_rep is not None
        assert backend.shape(rel_rep) == (16, 8)
        assert all_finite(rel_rep, backend)

    def test_output_shape(self, backend):
        """Output shape should be [n_samples, n_anchors]."""
        n_samples = 20
        n_anchors = 10
        d = 32

        hidden_states = backend.random_normal((n_samples, d))
        anchor_embeddings = backend.random_normal((n_anchors, d))
        backend.eval(hidden_states, anchor_embeddings)

        rel_rep = compute_relative_representation(
            hidden_states, anchor_embeddings, backend=backend
        )

        assert backend.shape(rel_rep) == (n_samples, n_anchors)

    def test_cosine_similarity_bounded(self, backend):
        """Cosine similarities should be in [-1, 1]."""
        hidden_states = backend.random_normal((16, 32))
        anchor_embeddings = backend.random_normal((8, 32))
        backend.eval(hidden_states, anchor_embeddings)

        rel_rep = compute_relative_representation(
            hidden_states, anchor_embeddings, backend=backend
        )

        min_val = backend.min(rel_rep)
        max_val = backend.max(rel_rep)
        backend.eval(min_val, max_val)

        eps = regularization_epsilon(backend, rel_rep)
        assert float(backend.to_scalar(min_val)) >= -1.0 - eps
        assert float(backend.to_scalar(max_val)) <= 1.0 + eps

    def test_single_sample(self, backend):
        """Single sample should work."""
        hidden_states = backend.random_normal((1, 32))
        anchor_embeddings = backend.random_normal((8, 32))
        backend.eval(hidden_states, anchor_embeddings)

        rel_rep = compute_relative_representation(
            hidden_states, anchor_embeddings, backend=backend
        )

        assert backend.shape(rel_rep) == (1, 8)

    def test_single_anchor(self, backend):
        """Single anchor should work."""
        hidden_states = backend.random_normal((16, 32))
        anchor_embeddings = backend.random_normal((1, 32))
        backend.eval(hidden_states, anchor_embeddings)

        rel_rep = compute_relative_representation(
            hidden_states, anchor_embeddings, backend=backend
        )

        assert backend.shape(rel_rep) == (16, 1)

    def test_self_similarity_high(self, backend):
        """Vector's similarity to itself should be 1.0."""
        anchor_embeddings = backend.random_normal((4, 32))
        backend.eval(anchor_embeddings)

        # Use anchors as hidden states - each should have high similarity to itself
        rel_rep = compute_relative_representation(
            anchor_embeddings, anchor_embeddings, backend=backend
        )

        # Diagonal should be 1.0 (self-similarity)
        diagonal = backend.diag(rel_rep)
        backend.eval(diagonal)

        mean_diag = backend.mean(diagonal)
        backend.eval(mean_diag)
        assert float(backend.to_scalar(mean_diag)) > 0.99


class TestAlignRelativeRepresentations:
    """Tests for align_relative_representations()."""

    def test_identical_input_identity_rotation(self, backend):
        """Identical inputs should produce near-identity rotation."""
        rel_rep = backend.random_normal((16, 8))
        backend.eval(rel_rep)

        R, error = align_relative_representations(rel_rep, rel_rep, backend=backend)

        # Should be near identity
        I = backend.eye(8)
        diff = backend.mean(backend.abs(R - I))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 0.1
        assert error < 0.1

    def test_rotation_shape(self, backend):
        """Rotation matrix should be [n_anchors, n_anchors]."""
        n_anchors = 8
        source_rel = backend.random_normal((16, n_anchors))
        target_rel = backend.random_normal((16, n_anchors))
        backend.eval(source_rel, target_rel)

        R, _ = align_relative_representations(source_rel, target_rel, backend=backend)

        assert backend.shape(R) == (n_anchors, n_anchors)

    def test_rotation_orthogonal(self, backend):
        """Rotation matrix should be orthogonal: R @ R^T = I."""
        source_rel = backend.random_normal((16, 8))
        target_rel = backend.random_normal((16, 8))
        backend.eval(source_rel, target_rel)

        R, _ = align_relative_representations(source_rel, target_rel, backend=backend)

        RRt = backend.matmul(R, backend.transpose(R))
        I = backend.eye(8)
        backend.eval(RRt)

        diff = backend.mean(backend.abs(RRt - I))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 0.1

    def test_error_non_negative(self, backend):
        """Alignment error should be non-negative."""
        source_rel = backend.random_normal((16, 8))
        target_rel = backend.random_normal((16, 8))
        backend.eval(source_rel, target_rel)

        _, error = align_relative_representations(source_rel, target_rel, backend=backend)

        assert error >= 0.0

    def test_single_sample_returns_identity(self, backend):
        """Single sample should return identity (degenerate case)."""
        source_rel = backend.random_normal((1, 8))
        target_rel = backend.random_normal((1, 8))
        backend.eval(source_rel, target_rel)

        R, error = align_relative_representations(source_rel, target_rel, backend=backend)

        # Should return identity for degenerate case
        I = backend.eye(8)
        diff = backend.mean(backend.abs(R - I))
        backend.eval(diff)
        eps = regularization_epsilon(backend, R)
        assert float(backend.to_scalar(diff)) < eps


class TestTransferViaRelativeSpace:
    """Tests for transfer_via_relative_space()."""

    def test_basic_transfer(self, backend):
        """Basic transfer should work."""
        source_hidden = backend.random_normal((16, 32))
        source_anchors = backend.random_normal((8, 32))
        target_anchors = backend.random_normal((8, 24))
        backend.eval(source_hidden, source_anchors, target_anchors)

        transferred = transfer_via_relative_space(
            source_hidden, source_anchors, target_anchors
        )

        assert transferred is not None
        assert backend.shape(transferred) == (16, 24)
        assert all_finite(transferred, backend)

    def test_transfer_shape(self, backend):
        """Transferred shape should match target dimension."""
        n_samples = 20
        d_source = 32
        d_target = 64
        n_anchors = 8

        source_hidden = backend.random_normal((n_samples, d_source))
        source_anchors = backend.random_normal((n_anchors, d_source))
        target_anchors = backend.random_normal((n_anchors, d_target))
        backend.eval(source_hidden, source_anchors, target_anchors)

        transferred = transfer_via_relative_space(
            source_hidden, source_anchors, target_anchors
        )

        assert backend.shape(transferred) == (n_samples, d_target)

    def test_same_anchors_preserves_structure(self, backend):
        """Same source/target anchors should preserve relative structure."""
        source_hidden = backend.random_normal((16, 32))
        anchors = backend.random_normal((8, 32))
        backend.eval(source_hidden, anchors)

        transferred = transfer_via_relative_space(
            source_hidden, anchors, anchors
        )

        # Shape should be preserved
        assert backend.shape(transferred) == (16, 32)


class TestRelativeRepresentationDataclass:
    """Tests for RelativeRepresentation dataclass."""

    def test_n_samples_property(self, backend):
        """n_samples property should return correct value."""
        similarities = backend.random_normal((20, 8))
        backend.eval(similarities)

        rel_rep = RelativeRepresentation(
            similarities=similarities,
            anchor_ids=tuple(f"anchor_{i}" for i in range(8)),
            hidden_dim=32,
        )

        assert rel_rep.n_samples == 20

    def test_n_anchors_property(self, backend):
        """n_anchors property should return correct value."""
        similarities = backend.random_normal((20, 8))
        backend.eval(similarities)

        rel_rep = RelativeRepresentation(
            similarities=similarities,
            anchor_ids=tuple(f"anchor_{i}" for i in range(8)),
            hidden_dim=32,
        )

        assert rel_rep.n_anchors == 8

    def test_frozen_dataclass(self, backend):
        """RelativeRepresentation should be frozen (immutable)."""
        similarities = backend.random_normal((20, 8))
        backend.eval(similarities)

        rel_rep = RelativeRepresentation(
            similarities=similarities,
            anchor_ids=tuple(f"anchor_{i}" for i in range(8)),
            hidden_dim=32,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            rel_rep.hidden_dim = 64


class TestRelativeRepresentationMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n_samples=st.integers(min_value=4, max_value=32),
        n_anchors=st.integers(min_value=2, max_value=16),
        d=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_relative_rep_shape(self, n_samples, n_anchors, d):
        """Relative representation shape should be [n_samples, n_anchors]."""
        backend = get_default_backend()
        hidden_states = backend.random_normal((n_samples, d))
        anchor_embeddings = backend.random_normal((n_anchors, d))
        backend.eval(hidden_states, anchor_embeddings)

        rel_rep = compute_relative_representation(
            hidden_states, anchor_embeddings, backend=backend
        )

        assert backend.shape(rel_rep) == (n_samples, n_anchors)

    @given(
        n_samples=st.integers(min_value=4, max_value=32),
        n_anchors=st.integers(min_value=2, max_value=16),
        d=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_relative_rep_finite(self, n_samples, n_anchors, d):
        """Relative representation should always be finite."""
        backend = get_default_backend()
        hidden_states = backend.random_normal((n_samples, d))
        anchor_embeddings = backend.random_normal((n_anchors, d))
        backend.eval(hidden_states, anchor_embeddings)

        rel_rep = compute_relative_representation(
            hidden_states, anchor_embeddings, backend=backend
        )

        assert all_finite(rel_rep, backend)

    @given(
        n_samples=st.integers(min_value=4, max_value=32),
        n_anchors=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=5, deadline=None)
    def test_rotation_orthogonal_property(self, n_samples, n_anchors):
        """Rotation matrix should be orthogonal."""
        backend = get_default_backend()
        source_rel = backend.random_normal((n_samples, n_anchors))
        target_rel = backend.random_normal((n_samples, n_anchors))
        backend.eval(source_rel, target_rel)

        R, _ = align_relative_representations(source_rel, target_rel, backend=backend)

        RRt = backend.matmul(R, backend.transpose(R))
        I = backend.eye(n_anchors)
        backend.eval(RRt)

        diff = backend.mean(backend.abs(RRt - I))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 0.15

    @given(
        n_samples=st.integers(min_value=4, max_value=32),
        n_anchors=st.integers(min_value=2, max_value=8),
        d_source=st.integers(min_value=8, max_value=32),
        d_target=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_transfer_output_shape(self, n_samples, n_anchors, d_source, d_target):
        """Transfer should produce correct output shape."""
        backend = get_default_backend()
        source_hidden = backend.random_normal((n_samples, d_source))
        source_anchors = backend.random_normal((n_anchors, d_source))
        target_anchors = backend.random_normal((n_anchors, d_target))
        backend.eval(source_hidden, source_anchors, target_anchors)

        transferred = transfer_via_relative_space(
            source_hidden, source_anchors, target_anchors
        )

        assert backend.shape(transferred) == (n_samples, d_target)

    @given(
        n_samples=st.integers(min_value=4, max_value=32),
        n_anchors=st.integers(min_value=2, max_value=8),
        d_source=st.integers(min_value=8, max_value=32),
        d_target=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=5, deadline=None)
    def test_transfer_finite(self, n_samples, n_anchors, d_source, d_target):
        """Transfer should always produce finite output."""
        backend = get_default_backend()
        source_hidden = backend.random_normal((n_samples, d_source))
        source_anchors = backend.random_normal((n_anchors, d_source))
        target_anchors = backend.random_normal((n_anchors, d_target))
        backend.eval(source_hidden, source_anchors, target_anchors)

        transferred = transfer_via_relative_space(
            source_hidden, source_anchors, target_anchors
        )

        assert all_finite(transferred, backend)
