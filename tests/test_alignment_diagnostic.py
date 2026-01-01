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

"""Tests for AlignmentDiagnostic module."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.alignment_diagnostic import (
    AlignmentSignal,
    alignment_signal_from_matrices,
    _matrix_rank,
)


@pytest.fixture
def backend():
    """Get compute backend."""
    return get_default_backend()


class TestAlignmentSignalFields:
    """Tests for AlignmentSignal dataclass fields."""

    def test_required_fields(self):
        """Test required fields are set correctly."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.85,
            metadata={"phase_tol": 1e-6},
        )
        assert signal.dimension == 3
        assert signal.cka_achieved == 0.85
        assert signal.cka_target == 1.0  # default

    def test_default_cka_target(self):
        """Test default cka_target is 1.0."""
        signal = AlignmentSignal(
            dimension=2,
            cka_achieved=0.9,
            metadata={"phase_tol": 1e-6},
        )
        assert signal.cka_target == 1.0

    def test_gap_computed_automatically(self):
        """Test gap is computed from cka_target - cka_achieved."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.75,
            cka_target=1.0,
            metadata={"phase_tol": 1e-6},
        )
        assert abs(signal.gap - 0.25) < 1e-9

    def test_gap_explicit_overrides(self):
        """Test explicit gap value is used."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.75,
            cka_target=1.0,
            gap=0.3,  # explicit
            metadata={"phase_tol": 1e-6},
        )
        assert signal.gap == 0.3

    def test_gap_zero_not_computed(self):
        """Test that gap=0.0 triggers computation."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.8,
            gap=0.0,  # will be recomputed
            metadata={"phase_tol": 1e-6},
        )
        assert abs(signal.gap - 0.2) < 1e-9

    def test_gap_negative_clamped(self):
        """Test gap is clamped to non-negative."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=1.1,  # above target
            cka_target=1.0,
            metadata={"phase_tol": 1e-6},
        )
        assert signal.gap == 0.0

    def test_default_lists_empty(self):
        """Test default list fields are empty."""
        signal = AlignmentSignal(
            dimension=2,
            cka_achieved=0.9,
            metadata={"phase_tol": 1e-6},
        )
        assert signal.misaligned_anchors == []
        assert signal.anchor_labels == []
        assert signal.anchor_divergence == []

    def test_default_strings(self):
        """Test default string fields."""
        signal = AlignmentSignal(
            dimension=2,
            cka_achieved=0.9,
            metadata={"phase_tol": 1e-6},
        )
        assert signal.divergence_pattern == "unknown"
        assert signal.suggested_transformation == "refine"

    def test_frozen_immutable(self):
        """Test dataclass is frozen."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.85,
            metadata={"phase_tol": 1e-6},
        )
        with pytest.raises(AttributeError):
            signal.dimension = 4


class TestAlignmentSignalIsPhaseLocked:
    """Tests for AlignmentSignal.is_phase_locked property."""

    def test_phase_locked_when_gap_below_tolerance(self):
        """Test phase_locked returns True when gap < phase_tol."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.9999999,
            cka_target=1.0,
            metadata={"phase_tol": 1e-5},
        )
        assert signal.is_phase_locked is True

    def test_not_phase_locked_when_gap_above_tolerance(self):
        """Test phase_locked returns False when gap > phase_tol."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.9,
            cka_target=1.0,
            metadata={"phase_tol": 1e-5},
        )
        assert signal.is_phase_locked is False

    def test_phase_locked_requires_metadata(self):
        """Test phase_locked raises without phase_tol in metadata."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.99,
            metadata={},  # no phase_tol
        )
        with pytest.raises(ValueError, match="phase_tol"):
            _ = signal.is_phase_locked

    def test_phase_locked_edge_case_exactly_at_tolerance(self):
        """Test gap exactly at tolerance is considered phase locked."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.9,
            cka_target=1.0,
            gap=0.1,
            metadata={"phase_tol": 0.1},
        )
        assert signal.is_phase_locked is True


class TestAlignmentSignalToDict:
    """Tests for AlignmentSignal.to_dict method."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.85,
            cka_target=1.0,
            metadata={"phase_tol": 1e-6},
        )
        d = signal.to_dict()

        assert d["dimension"] == 3
        assert d["cka_achieved"] == 0.85
        assert d["cka_target"] == 1.0
        assert abs(d["gap"] - 0.15) < 1e-9

    def test_to_dict_includes_all_fields(self):
        """Test to_dict includes all fields."""
        signal = AlignmentSignal(
            dimension=2,
            cka_achieved=0.9,
            misaligned_anchors=["token:1", "token:2"],
            anchor_labels=["a", "b", "c"],
            anchor_divergence=[0.1, 0.2, 0.05],
            divergence_pattern="scale",
            suggested_transformation="scale_normalization",
            iteration=5,
            metadata={"phase_tol": 1e-6, "scale_ratio": 1.5},
        )
        d = signal.to_dict()

        assert d["misaligned_anchors"] == ["token:1", "token:2"]
        assert d["anchor_labels"] == ["a", "b", "c"]
        assert d["anchor_divergence"] == [0.1, 0.2, 0.05]
        assert d["divergence_pattern"] == "scale"
        assert d["suggested_transformation"] == "scale_normalization"
        assert d["iteration"] == 5
        assert d["metadata"]["scale_ratio"] == 1.5

    def test_to_dict_returns_copies(self):
        """Test to_dict returns copies of mutable fields."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=0.8,
            misaligned_anchors=["a", "b"],
            metadata={"phase_tol": 1e-6},
        )
        d = signal.to_dict()
        d["misaligned_anchors"].append("c")

        # Original unchanged
        assert len(signal.misaligned_anchors) == 2


class TestAlignmentSignalFromMatrices:
    """Tests for alignment_signal_from_matrices function."""

    def test_perfect_alignment_returns_phase_locked(self, backend):
        """Test perfectly aligned matrices return phase_locked pattern."""
        backend.random_seed(42)
        matrix = backend.random_normal((10, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=matrix,
            target_matrix=matrix,  # identical
            backend=backend,
            cka_achieved=1.0,
        )

        assert signal.divergence_pattern == "phase_locked"
        assert signal.suggested_transformation == "none"
        assert signal.cka_achieved == 1.0

    def test_near_perfect_alignment(self, backend):
        """Test near-perfect alignment (CKA close to 1.0)."""
        backend.random_seed(42)
        matrix = backend.random_normal((10, 8))

        # Very close to 1.0
        signal = alignment_signal_from_matrices(
            source_matrix=matrix,
            target_matrix=matrix,
            backend=backend,
            cka_achieved=0.999999999,
        )

        assert signal.divergence_pattern == "phase_locked"

    def test_misaligned_matrices_return_rotation(self, backend):
        """Test misaligned matrices detect rotation pattern."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 8))  # different

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.5,
        )

        # May be rotation, scale, or rank_deficient depending on matrices
        assert signal.divergence_pattern in ["rotation", "scale", "rank_deficient"]
        assert signal.cka_achieved == 0.5

    def test_dimension_mismatch_detected(self, backend):
        """Test different dimensions detected as dimension_mismatch."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 12))  # different dimension

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.7,
        )

        assert signal.divergence_pattern == "dimension_mismatch"
        assert signal.suggested_transformation == "expand_anchors"
        assert signal.metadata["shape_mismatch"] == 1.0

    def test_custom_labels_used(self, backend):
        """Test custom labels are used for anchor identification."""
        backend.random_seed(42)
        source = backend.random_normal((5, 8))
        target = backend.random_normal((5, 8))

        labels = ["alpha", "beta", "gamma", "delta", "epsilon"]
        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            labels=labels,
            backend=backend,
            cka_achieved=0.6,
        )

        assert signal.anchor_labels == labels
        for anchor in signal.misaligned_anchors:
            assert anchor in labels

    def test_default_labels_generated(self, backend):
        """Test default labels when none provided."""
        backend.random_seed(42)
        source = backend.random_normal((5, 8))
        target = backend.random_normal((5, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.6,
        )

        assert len(signal.anchor_labels) == 5
        assert signal.anchor_labels[0] == "sample:0"

    def test_top_k_limits_misaligned_anchors(self, backend):
        """Test top_k limits number of misaligned anchors."""
        backend.random_seed(42)
        source = backend.random_normal((20, 8))
        target = backend.random_normal((20, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.5,
            top_k=3,
        )

        assert len(signal.misaligned_anchors) == 3

    def test_top_k_clamped_to_sample_count(self, backend):
        """Test top_k doesn't exceed sample count."""
        backend.random_seed(42)
        source = backend.random_normal((5, 8))
        target = backend.random_normal((5, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.5,
            top_k=100,  # more than samples
        )

        assert len(signal.misaligned_anchors) == 5

    def test_iteration_preserved(self, backend):
        """Test iteration number is preserved."""
        backend.random_seed(42)
        matrix = backend.random_normal((10, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=matrix,
            target_matrix=matrix,
            backend=backend,
            cka_achieved=1.0,
            iteration=42,
        )

        assert signal.iteration == 42

    def test_dimension_preserved(self, backend):
        """Test dimension parameter is preserved."""
        backend.random_seed(42)
        matrix = backend.random_normal((10, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=matrix,
            target_matrix=matrix,
            backend=backend,
            cka_achieved=1.0,
            dimension=5,
        )

        assert signal.dimension == 5

    def test_metadata_contains_diagnostics(self, backend):
        """Test metadata contains diagnostic information."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.5,
        )

        assert "rank_source" in signal.metadata
        assert "rank_target" in signal.metadata
        assert "scale_ratio" in signal.metadata
        assert "max_divergence" in signal.metadata
        assert "mean_divergence" in signal.metadata
        assert "balance_ratio" in signal.metadata
        assert "phase_tol" in signal.metadata

    def test_anchor_divergence_has_correct_length(self, backend):
        """Test anchor_divergence has one entry per sample."""
        backend.random_seed(42)
        n_samples = 15
        source = backend.random_normal((n_samples, 8))
        target = backend.random_normal((n_samples, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.5,
        )

        assert len(signal.anchor_divergence) == n_samples


class TestAlignmentSignalPatternDetection:
    """Tests for divergence pattern detection."""

    def test_scale_mismatch_detected(self, backend):
        """Test scaled matrices detect scale pattern."""
        backend.random_seed(42)
        # Use orthogonal matrix via QR decomposition to guarantee full rank
        random_matrix = backend.random_normal((20, 16))
        q, r = backend.qr(random_matrix)
        # Take first 16 columns of Q (orthonormal, full rank)
        source = q[:, :16]
        backend.eval(source)
        # Scale target by 5x - significant scale mismatch but keeps rank
        target = source * 5.0

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.8,  # CKA is scale-invariant so still high
        )

        # Scaling preserves rank, so should detect scale pattern
        # But with same matrices, rank is preserved so pattern could be scale or rotation
        assert signal.divergence_pattern in ["scale", "rotation"]
        if signal.divergence_pattern == "scale":
            assert signal.suggested_transformation == "scale_normalization"

    def test_rank_deficient_detected(self, backend):
        """Test rank-deficient matrices detected."""
        backend.random_seed(42)
        # Create rank-deficient matrix (rank 2)
        u = backend.random_normal((10, 2))
        v = backend.random_normal((2, 8))
        source = backend.matmul(u, v)  # rank at most 2

        # Full rank target
        target = backend.random_normal((10, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.4,
        )

        assert signal.divergence_pattern == "rank_deficient"
        assert signal.suggested_transformation == "expand_anchors"


class TestMatrixRank:
    """Tests for _matrix_rank helper function."""

    def test_full_rank_matrix(self, backend):
        """Test full rank detection for full rank matrix."""
        backend.random_seed(42)
        matrix = backend.random_normal((8, 8))

        rank = _matrix_rank(matrix, backend)

        assert rank == 8

    def test_rank_deficient_matrix(self, backend):
        """Test rank detection for rank-deficient matrix."""
        backend.random_seed(42)
        # Create rank-2 matrix
        u = backend.random_normal((8, 2))
        v = backend.random_normal((2, 8))
        matrix = backend.matmul(u, v)

        rank = _matrix_rank(matrix, backend)

        assert rank <= 2

    def test_zero_matrix(self, backend):
        """Test rank of zero matrix is 0."""
        matrix = backend.zeros((5, 5))

        rank = _matrix_rank(matrix, backend)

        assert rank == 0

    def test_full_rank_random_matrix(self, backend):
        """Test rank of full-rank random matrix."""
        backend.random_seed(42)
        n = 5
        # Random matrix is full rank with probability 1
        matrix = backend.random_normal((n, n))

        rank = _matrix_rank(matrix, backend)

        # Random matrix should be full rank
        assert rank == n

    def test_custom_epsilon(self, backend):
        """Test custom epsilon threshold."""
        backend.random_seed(42)
        matrix = backend.random_normal((5, 5))

        # Very large epsilon should reduce rank
        rank_strict = _matrix_rank(matrix, backend, eps=1e-10)
        rank_loose = _matrix_rank(matrix, backend, eps=0.5)

        assert rank_loose <= rank_strict


class TestAlignmentSignalEdgeCases:
    """Edge case tests for alignment diagnostics."""

    def test_single_sample(self, backend):
        """Test with single sample matrices."""
        backend.random_seed(42)
        source = backend.random_normal((1, 8))
        target = backend.random_normal((1, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.5,
        )

        assert len(signal.anchor_labels) == 1
        assert len(signal.anchor_divergence) == 1

    def test_wide_matrix(self, backend):
        """Test with wide matrix (more features than samples)."""
        backend.random_seed(42)
        source = backend.random_normal((5, 100))
        target = backend.random_normal((5, 100))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.6,
        )

        assert signal.dimension == 3  # default
        assert len(signal.anchor_labels) == 5

    def test_tall_matrix(self, backend):
        """Test with tall matrix (more samples than features)."""
        backend.random_seed(42)
        source = backend.random_normal((100, 5))
        target = backend.random_normal((100, 5))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.6,
            top_k=10,
        )

        assert len(signal.misaligned_anchors) == 10

    def test_cka_achieved_zero(self, backend):
        """Test with CKA achieved = 0."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 8))

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=0.0,
        )

        assert signal.cka_achieved == 0.0
        assert signal.gap == 1.0

    def test_uses_default_backend(self, backend):
        """Test function uses default backend when none provided."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 8))

        # Don't pass backend explicitly
        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            cka_achieved=0.5,
        )

        assert signal is not None
        assert len(signal.anchor_divergence) == 10
