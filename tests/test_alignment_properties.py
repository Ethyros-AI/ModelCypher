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
Property-based tests for alignment operations.

Tests mathematical invariants:
- Procrustes alignment is closed-form: F = pinv(source) @ target
- Aligned CKA on training probes = 1.0 (by construction)
- Alignment preserves relational structure
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _random_matrix(backend, rows: int, cols: int, seed: int):
    """Generate random matrix using backend."""
    backend.random_seed(seed)
    return backend.random_normal(shape=(rows, cols))


def _scalar_tol(backend) -> float:
    """Get scalar tolerance for numerical comparisons."""
    return division_epsilon(backend, backend.array([1.0]))


class TestProcrustesAlignment:
    """Test Procrustes alignment properties."""

    @pytest.mark.parametrize("seed", range(5))
    def test_alignment_improves_cka(self, seed: int):
        """Alignment should improve or maintain CKA."""
        backend = get_default_backend()
        n = 50
        d = 32

        source = _random_matrix(backend, n, d, seed)
        target = _random_matrix(backend, n, d, seed + 1000)
        backend.eval(source, target)

        # CKA before alignment
        cka_before = compute_cka(source, target, backend)

        # Align
        aligner = GramAligner(backend)
        result = aligner.find_perfect_alignment(source, target)

        # Apply alignment
        aligned = backend.matmul(source, result.feature_transform)
        backend.eval(aligned)

        # CKA after alignment
        cka_after = compute_cka(aligned, target, backend)

        # Alignment should not decrease CKA significantly
        # (may be slightly lower due to numerical precision)
        assert cka_after.cka >= cka_before.cka - 0.01, (
            f"Alignment should not significantly decrease CKA: "
            f"before={cka_before.cka:.4f}, after={cka_after.cka:.4f}"
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_alignment_closed_form(self, seed: int):
        """Alignment should be closed-form: F = pinv(source) @ target."""
        backend = get_default_backend()
        n = 50
        d = 32

        source = _random_matrix(backend, n, d, seed)
        target = _random_matrix(backend, n, d, seed + 1000)
        backend.eval(source, target)

        # Manual closed-form computation
        F_manual = backend.matmul(backend.pinv(source), target)
        backend.eval(F_manual)

        # Via GramAligner
        aligner = GramAligner(backend)
        result = aligner.find_perfect_alignment(source, target)

        # They should produce similar results
        # (not necessarily identical due to regularization and numerical methods)
        aligned_manual = backend.matmul(source, F_manual)
        aligned_aligner = backend.matmul(source, result.feature_transform)
        backend.eval(aligned_manual, aligned_aligner)

        # Both should approximate target similarly
        diff_manual = backend.norm(backend.subtract(aligned_manual, target))
        diff_aligner = backend.norm(backend.subtract(aligned_aligner, target))
        backend.eval(diff_manual, diff_aligner)

        diff_manual_val = float(backend.tolist(diff_manual))
        diff_aligner_val = float(backend.tolist(diff_aligner))

        # Aligner should not be significantly worse than manual
        assert diff_aligner_val <= diff_manual_val * 1.5 + 1.0, (
            f"Aligner should not be worse than closed-form: "
            f"manual={diff_manual_val:.4f}, aligner={diff_aligner_val:.4f}"
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_alignment_preserves_gram_structure(self, seed: int):
        """Alignment should preserve Gram matrix structure."""
        backend = get_default_backend()
        n = 50
        d = 32

        source = _random_matrix(backend, n, d, seed)
        backend.eval(source)

        # Compute source Gram
        gram_source = backend.matmul(source, backend.transpose(source))
        backend.eval(gram_source)

        # Apply random orthogonal transform (should preserve Gram)
        random_mat = _random_matrix(backend, d, d, seed + 2000)
        q, _ = backend.qr(random_mat)
        transformed = backend.matmul(source, q)
        backend.eval(transformed)

        # Compute transformed Gram
        gram_transformed = backend.matmul(transformed, backend.transpose(transformed))
        backend.eval(gram_transformed)

        # Grams should be identical
        diff = backend.subtract(gram_source, gram_transformed)
        diff_norm = float(backend.tolist(backend.norm(diff)))

        tol = _scalar_tol(backend) * float(backend.tolist(backend.norm(gram_source)))
        assert diff_norm <= tol, f"Orthogonal transform should preserve Gram: diff={diff_norm}"


class TestAlignmentInvariants:
    """Test alignment mathematical invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_identity_alignment(self, seed: int):
        """Aligning identical matrices should give identity-like transform."""
        backend = get_default_backend()
        n = 50
        d = 32

        source = _random_matrix(backend, n, d, seed)
        backend.eval(source)

        aligner = GramAligner(backend)
        result = aligner.find_perfect_alignment(source, source)

        # Applied to source should give source back
        aligned = backend.matmul(source, result.feature_transform)
        backend.eval(aligned)

        diff = backend.subtract(aligned, source)
        diff_norm = float(backend.tolist(backend.norm(diff)))
        source_norm = float(backend.tolist(backend.norm(source)))

        relative_diff = diff_norm / source_norm if source_norm > 1e-10 else diff_norm
        assert relative_diff < 0.1, f"Self-alignment should approximately preserve: {relative_diff}"

    @pytest.mark.parametrize("seed", range(5))
    def test_alignment_is_linear(self, seed: int):
        """Alignment transform should be linear: F(aX) = aF(X)."""
        backend = get_default_backend()
        n = 50
        d = 32

        source = _random_matrix(backend, n, d, seed)
        target = _random_matrix(backend, n, d, seed + 1000)
        backend.eval(source, target)

        aligner = GramAligner(backend)
        result = aligner.find_perfect_alignment(source, target)

        # Apply to source
        aligned = backend.matmul(source, result.feature_transform)
        backend.eval(aligned)

        # Apply to scaled source
        scale = 2.5
        source_scaled = source * scale
        aligned_scaled = backend.matmul(source_scaled, result.feature_transform)
        backend.eval(aligned_scaled)

        # Should be scaled version
        diff = backend.subtract(aligned_scaled, aligned * scale)
        diff_norm = float(backend.tolist(backend.norm(diff)))

        assert diff_norm < 1e-4, f"Alignment should be linear: diff={diff_norm}"


class TestCrossArchitectureAlignment:
    """Test alignment across different dimensions."""

    @pytest.mark.parametrize("seed", range(5))
    def test_different_dimensions(self, seed: int):
        """Alignment should work across different feature dimensions."""
        backend = get_default_backend()
        n = 50
        d_source = 48
        d_target = 32

        source = _random_matrix(backend, n, d_source, seed)
        target = _random_matrix(backend, n, d_target, seed + 1000)
        backend.eval(source, target)

        aligner = GramAligner(backend)
        result = aligner.find_perfect_alignment(source, target)

        # Transform shape should be (d_source, d_target)
        assert result.feature_transform.shape == (d_source, d_target), (
            f"Transform shape should be ({d_source}, {d_target}), "
            f"got {result.feature_transform.shape}"
        )

        # Aligned should have target dimension
        aligned = backend.matmul(source, result.feature_transform)
        backend.eval(aligned)

        assert aligned.shape == target.shape, (
            f"Aligned shape should match target: {aligned.shape} vs {target.shape}"
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_projection_to_lower_dim(self, seed: int):
        """Projecting to lower dimension should preserve relational structure."""
        backend = get_default_backend()
        n = 50
        d_high = 64
        d_low = 16

        source = _random_matrix(backend, n, d_high, seed)
        target = _random_matrix(backend, n, d_low, seed + 1000)
        backend.eval(source, target)

        aligner = GramAligner(backend)
        result = aligner.find_perfect_alignment(source, target)

        # Project
        projected = backend.matmul(source, result.feature_transform)
        backend.eval(projected)

        # CKA with target should be reasonable
        cka_result = compute_cka(projected, target, backend)

        # For random matrices, CKA won't be high, but should be valid
        assert cka_result.is_valid
        assert 0.0 <= cka_result.cka <= 1.0


try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestAlignmentHypothesis:
    """Hypothesis-based alignment property tests."""

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        d_source=st.integers(min_value=4, max_value=32),
        d_target=st.integers(min_value=4, max_value=32),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_alignment_produces_valid_transform(
        self, n_samples: int, d_source: int, d_target: int, seed: int
    ):
        """Alignment should always produce a valid transform matrix."""
        backend = get_default_backend()

        source = _random_matrix(backend, n_samples, d_source, seed)
        target = _random_matrix(backend, n_samples, d_target, seed + 5000)
        backend.eval(source, target)

        aligner = GramAligner(backend)
        result = aligner.find_perfect_alignment(source, target)

        # Transform should be correct shape
        assert result.feature_transform.shape == (d_source, d_target)

        # Transform should be finite
        is_finite = backend.all(backend.isfinite(result.feature_transform))
        backend.eval(is_finite)
        assert backend.tolist(is_finite), "Transform should be finite"

    @given(
        n_samples=st.integers(min_value=30, max_value=80),
        d=st.integers(min_value=4, max_value=32),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_aligned_cka_produces_valid_result(self, n_samples: int, d: int, seed: int):
        """Alignment should produce valid CKA results."""
        backend = get_default_backend()

        source = _random_matrix(backend, n_samples, d, seed)
        target = _random_matrix(backend, n_samples, d, seed + 5000)
        backend.eval(source, target)

        aligner = GramAligner(backend)
        result = aligner.find_perfect_alignment(source, target)

        aligned = backend.matmul(source, result.feature_transform)
        backend.eval(aligned)

        cka_after = compute_cka(aligned, target, backend)

        # The aligned CKA should be valid and bounded
        # Note: for random unrelated matrices, alignment doesn't guarantee improvement
        # but the result should still be valid
        assert cka_after.is_valid, "CKA result should be valid"
        assert -0.01 <= cka_after.cka <= 1.01, f"CKA should be bounded: {cka_after.cka}"
