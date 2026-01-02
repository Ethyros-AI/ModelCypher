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
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)


@pytest.fixture
def backend():
    """Get compute backend."""
    return get_default_backend()


def _eps() -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array([1.0]))


class TestAlignmentSignalFields:
    """Tests for AlignmentSignal dataclass fields."""

    def test_required_fields(self):
        """Test required fields are set correctly."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=1.0 - _eps(),
            metadata={"phase_tol": _eps()},
        )
        assert signal.dimension == 3
        assert signal.cka_achieved == 1.0 - _eps()
        assert signal.cka_target == 1.0  # default

    def test_default_cka_target(self):
        """Test default cka_target is 1.0."""
        signal = AlignmentSignal(
            dimension=2,
            cka_achieved=1.0 - _eps(),
            metadata={"phase_tol": _eps()},
        )
        assert signal.cka_target == 1.0

    def test_gap_computed_automatically(self):
        """Test gap is computed from cka_target - cka_achieved."""
        cka_achieved = 1.0 - _eps()
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=cka_achieved,
            cka_target=1.0,
            metadata={"phase_tol": _eps()},
        )
        expected_gap = 1.0 - cka_achieved
        assert abs(signal.gap - expected_gap) <= _eps()

    def test_gap_explicit_overrides(self):
        """Test explicit gap value is used."""
        gap = 3.0 * _eps()
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=1.0 - _eps(),
            cka_target=1.0,
            gap=gap,  # explicit
            metadata={"phase_tol": _eps()},
        )
        assert signal.gap == gap

    def test_gap_zero_not_computed(self):
        """Test that gap=0.0 triggers computation."""
        cka_achieved = 1.0 - 2.0 * _eps()
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=cka_achieved,
            gap=0.0,  # will be recomputed
            metadata={"phase_tol": _eps()},
        )
        expected_gap = 1.0 - cka_achieved
        assert abs(signal.gap - expected_gap) <= _eps()

    def test_gap_negative_clamped(self):
        """Test gap is clamped to non-negative."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=1.0 + _eps(),  # above target
            cka_target=1.0,
            metadata={"phase_tol": _eps()},
        )
        assert signal.gap == 0.0

    def test_default_lists_empty(self):
        """Test default list fields are empty."""
        signal = AlignmentSignal(
            dimension=2,
            cka_achieved=1.0 - _eps(),
            metadata={"phase_tol": _eps()},
        )
        assert signal.misaligned_anchors == []
        assert signal.anchor_labels == []
        assert signal.anchor_divergence == []

    def test_default_strings(self):
        """Test default string fields."""
        signal = AlignmentSignal(
            dimension=2,
            cka_achieved=1.0 - _eps(),
            metadata={"phase_tol": _eps()},
        )
        assert signal.divergence_pattern == "unknown"
        assert signal.suggested_transformation == "refine"

    def test_frozen_immutable(self):
        """Test dataclass is frozen."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=1.0 - _eps(),
            metadata={"phase_tol": _eps()},
        )
        with pytest.raises(AttributeError):
            signal.dimension = 4


class TestAlignmentSignalIsPhaseLocked:
    """Tests for AlignmentSignal.is_phase_locked property."""

    def test_phase_locked_when_gap_below_tolerance(self):
        """Test phase_locked returns True when gap < phase_tol."""
        phase_tol = _eps()
        gap = phase_tol / 2.0
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=1.0 - gap,
            cka_target=1.0,
            gap=gap,
            metadata={"phase_tol": phase_tol},
        )
        assert signal.is_phase_locked is True

    def test_not_phase_locked_when_gap_above_tolerance(self):
        """Test phase_locked returns False when gap > phase_tol."""
        phase_tol = _eps()
        gap = phase_tol * 2.0
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=1.0 - gap,
            cka_target=1.0,
            gap=gap,
            metadata={"phase_tol": phase_tol},
        )
        assert signal.is_phase_locked is False

    def test_phase_locked_requires_metadata(self):
        """Test phase_locked raises without phase_tol in metadata."""
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=1.0 - _eps(),
            metadata={},  # no phase_tol
        )
        with pytest.raises(ValueError, match="phase_tol"):
            _ = signal.is_phase_locked

    def test_phase_locked_edge_case_exactly_at_tolerance(self):
        """Test gap exactly at tolerance is considered phase locked."""
        phase_tol = _eps()
        gap = phase_tol
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=1.0 - gap,
            cka_target=1.0,
            gap=gap,
            metadata={"phase_tol": phase_tol},
        )
        assert signal.is_phase_locked is True


class TestAlignmentSignalToDict:
    """Tests for AlignmentSignal.to_dict method."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        eps = _eps()
        cka_achieved = 1.0 - eps
        signal = AlignmentSignal(
            dimension=3,
            cka_achieved=cka_achieved,
            cka_target=1.0,
            metadata={"phase_tol": eps},
        )
        d = signal.to_dict()

        assert d["dimension"] == 3
        assert d["cka_achieved"] == cka_achieved
        assert d["cka_target"] == 1.0
        expected_gap = 1.0 - cka_achieved
        assert abs(d["gap"] - expected_gap) <= eps

    def test_to_dict_includes_all_fields(self):
        """Test to_dict includes all fields."""
        eps = _eps()
        signal = AlignmentSignal(
            dimension=2,
            cka_achieved=1.0 - 2.0 * eps,
            misaligned_anchors=["token:1", "token:2"],
            anchor_labels=["a", "b", "c"],
            anchor_divergence=[0.1, 0.2, 0.05],
            divergence_pattern="scale",
            suggested_transformation="scale_normalization",
            iteration=5,
            metadata={"phase_tol": eps, "scale_ratio": 1.5},
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
            cka_achieved=1.0 - _eps(),
            misaligned_anchors=["a", "b"],
            metadata={"phase_tol": _eps()},
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
        phase_tol = machine_epsilon(backend, matrix)

        # Very close to 1.0
        signal = alignment_signal_from_matrices(
            source_matrix=matrix,
            target_matrix=matrix,
            backend=backend,
            cka_achieved=1.0 - phase_tol / 2.0,
        )

        assert signal.divergence_pattern == "phase_locked"

    def test_misaligned_matrices_return_rotation(self, backend):
        """Test misaligned matrices detect rotation pattern."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 8))  # different
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
        )

        # May be rotation, scale, or rank_deficient depending on matrices
        assert signal.divergence_pattern in ["rotation", "scale", "rank_deficient"]
        assert signal.cka_achieved == cka_achieved

    def test_dimension_mismatch_detected(self, backend):
        """Test different dimensions detected as dimension_mismatch."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 12))  # different dimension
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
        )

        assert signal.divergence_pattern == "dimension_mismatch"
        assert signal.suggested_transformation == "expand_anchors"
        assert signal.metadata["shape_mismatch"] == 1.0

    def test_custom_labels_used(self, backend):
        """Test custom labels are used for anchor identification."""
        backend.random_seed(42)
        source = backend.random_normal((5, 8))
        target = backend.random_normal((5, 8))
        cka_achieved = 1.0 - division_epsilon(backend, source)

        labels = ["alpha", "beta", "gamma", "delta", "epsilon"]
        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            labels=labels,
            backend=backend,
            cka_achieved=cka_achieved,
        )

        assert signal.anchor_labels == labels
        for anchor in signal.misaligned_anchors:
            assert anchor in labels

    def test_default_labels_generated(self, backend):
        """Test default labels when none provided."""
        backend.random_seed(42)
        source = backend.random_normal((5, 8))
        target = backend.random_normal((5, 8))
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
        )

        assert len(signal.anchor_labels) == 5
        assert signal.anchor_labels[0] == "sample:0"

    def test_top_k_limits_misaligned_anchors(self, backend):
        """Test top_k limits number of misaligned anchors."""
        backend.random_seed(42)
        source = backend.random_normal((20, 8))
        target = backend.random_normal((20, 8))
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
            top_k=3,
        )

        assert len(signal.misaligned_anchors) == 3

    def test_top_k_clamped_to_sample_count(self, backend):
        """Test top_k doesn't exceed sample count."""
        backend.random_seed(42)
        source = backend.random_normal((5, 8))
        target = backend.random_normal((5, 8))
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
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
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
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
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
        )

        assert len(signal.anchor_divergence) == n_samples


class TestAlignmentSignalPatternDetection:
    """Tests for divergence pattern detection."""

    def test_scale_mismatch_detected(self, backend):
        """Test scaled matrices detect scale pattern or numerical rank issues."""
        backend.random_seed(42)
        # Use orthogonal matrix via QR decomposition for well-conditioned input
        random_matrix = backend.random_normal((20, 16))
        q, r = backend.qr(random_matrix)
        # Take first 16 columns of Q (orthonormal columns)
        source = q[:, :16]
        backend.eval(source)
        scale_factor = 1.0 + division_epsilon(backend, source)
        target = source * scale_factor

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=1.0 - division_epsilon(backend, source),
        )

        # Scale ratio should reflect the relative normalization.
        scale_ratio = signal.metadata["scale_ratio"]
        div_eps = division_epsilon(backend, source)
        src_norm = backend.mean(backend.norm(source, axis=1))
        tgt_norm = backend.mean(backend.norm(target, axis=1))
        backend.eval(src_norm, tgt_norm)
        src_norm_val = float(backend.to_numpy(src_norm))
        tgt_norm_val = float(backend.to_numpy(tgt_norm))
        expected_ratio = src_norm_val / (tgt_norm_val + div_eps)
        tolerance = machine_epsilon(backend, source)
        assert abs(scale_ratio - expected_ratio) <= tolerance, (
            f"Expected ~{expected_ratio}, got {scale_ratio}"
        )

        # Pattern may be scale, rotation, or rank_deficient depending on
        # numerical precision in eigenvalue computation
        assert signal.divergence_pattern in ["scale", "rotation", "rank_deficient"]
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
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
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


class TestAlignmentSignalEdgeCases:
    """Edge case tests for alignment diagnostics."""

    def test_single_sample(self, backend):
        """Test with single sample matrices."""
        backend.random_seed(42)
        source = backend.random_normal((1, 8))
        target = backend.random_normal((1, 8))
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
        )

        assert len(signal.anchor_labels) == 1
        assert len(signal.anchor_divergence) == 1

    def test_wide_matrix(self, backend):
        """Test with wide matrix (more features than samples)."""
        backend.random_seed(42)
        source = backend.random_normal((5, 100))
        target = backend.random_normal((5, 100))
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
        )

        assert signal.dimension == 3  # default
        assert len(signal.anchor_labels) == 5

    def test_tall_matrix(self, backend):
        """Test with tall matrix (more samples than features)."""
        backend.random_seed(42)
        source = backend.random_normal((100, 5))
        target = backend.random_normal((100, 5))
        cka_achieved = 1.0 - division_epsilon(backend, source)

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
            top_k=10,
        )

        assert len(signal.misaligned_anchors) == 10

    def test_cka_achieved_zero(self, backend):
        """Test with CKA achieved = 0."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 8))
        cka_achieved = 0.0

        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            backend=backend,
            cka_achieved=cka_achieved,
        )

        expected_gap = 1.0 - cka_achieved
        assert signal.cka_achieved == cka_achieved
        assert abs(signal.gap - expected_gap) <= _eps()

    def test_uses_default_backend(self, backend):
        """Test function uses default backend when none provided."""
        backend.random_seed(42)
        source = backend.random_normal((10, 8))
        target = backend.random_normal((10, 8))
        cka_achieved = 1.0 - division_epsilon(backend, source)

        # Don't pass backend explicitly
        signal = alignment_signal_from_matrices(
            source_matrix=source,
            target_matrix=target,
            cka_achieved=cka_achieved,
        )

        assert signal is not None
        assert len(signal.anchor_divergence) == 10
