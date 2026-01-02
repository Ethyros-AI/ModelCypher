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

"""Property-based tests for SpectralAnalysis.

Tests mathematical invariants:
- Spectral norm ≥ 0 (non-negative)
- Spectral alignment ∈ [0, 1]
- Condition number ≥ 1 (by definition)
- Spectral ratio symmetry: alignment(a/b) = alignment(b/a)
- Frobenius norm ≥ 0
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.spectral_analysis import (
    SpectralConfig,
    compute_spectral_metrics,
)

# =============================================================================
# Spectral Norm Invariants
# =============================================================================


def _compute_metrics(source, target):
    return compute_spectral_metrics(source, target, config=SpectralConfig())


class TestSpectralNormInvariants:
    """Tests for spectral norm bounds."""

    @pytest.mark.parametrize("seed", range(5))
    def test_spectral_norm_non_negative(self, seed: int) -> None:
        """Spectral norm (max singular value) must be >= 0.

        Mathematical property: Singular values are non-negative by definition.
        """
        backend = get_default_backend()
        backend.random_seed(seed)

        source = backend.random_normal((10, 8))
        source = backend.astype(source, "float32")
        target = backend.random_normal((10, 8))
        target = backend.astype(target, "float32")
        backend.eval(source, target)

        metrics = _compute_metrics(source, target)

        assert metrics.source_spectral_norm >= 0
        assert metrics.target_spectral_norm >= 0

    @pytest.mark.parametrize("seed", range(5))
    def test_frobenius_norm_non_negative(self, seed: int) -> None:
        """Frobenius norm must be >= 0.

        Mathematical property: ||A||_F = sqrt(Σ|a_ij|²) ≥ 0.
        """
        backend = get_default_backend()
        backend.random_seed(seed)

        source = backend.random_normal((10, 8))
        source = backend.astype(source, "float32")
        target = backend.random_normal((10, 8))
        target = backend.astype(target, "float32")
        backend.eval(source, target)

        metrics = _compute_metrics(source, target)

        assert metrics.delta_frobenius >= 0


# =============================================================================
# Spectral Alignment Invariants
# =============================================================================


class TestSpectralAlignmentInvariants:
    """Tests for spectral alignment bounds."""

    @pytest.mark.parametrize("seed", range(5))
    def test_spectral_alignment_in_zero_one(self, seed: int) -> None:
        """Spectral alignment must be in [0, 1].

        Mathematical property: alignment = min(r, 1/r) where r > 0.
        """
        backend = get_default_backend()
        backend.random_seed(seed)

        source = backend.random_normal((10, 8))
        source = backend.astype(source, "float32")
        target = backend.random_normal((10, 8))
        target = backend.astype(target, "float32")
        backend.eval(source, target)

        metrics = _compute_metrics(source, target)

        assert 0.0 <= metrics.spectral_alignment <= 1.0

    def test_identical_matrices_have_unit_alignment(self) -> None:
        """Identical matrices should have spectral alignment = 1.

        Mathematical property: ratio = 1 → alignment = min(1, 1) = 1.
        """
        backend = get_default_backend()
        backend.random_seed(42)
        matrix = backend.random_normal((10, 8))
        matrix = backend.astype(matrix, "float32")
        backend.eval(matrix)

        metrics = _compute_metrics(matrix, matrix)
        eps = machine_epsilon(backend, matrix)

        assert abs(metrics.spectral_alignment - 1.0) <= eps
        assert abs(metrics.spectral_ratio - 1.0) <= eps

    @pytest.mark.parametrize("scale", [0.1, 0.5, 2.0, 10.0])
    def test_spectral_alignment_symmetric(self, scale: float) -> None:
        """Spectral alignment should be symmetric with respect to scaling.

        Mathematical property: alignment(a/b) = alignment(b/a).
        """
        backend = get_default_backend()
        backend.random_seed(42)
        base = backend.random_normal((10, 8))
        base = backend.astype(base, "float32")
        backend.eval(base)
        eps = machine_epsilon(backend, base)

        scaled = base * scale
        backend.eval(scaled)

        # Forward: base vs scaled
        metrics_fwd = _compute_metrics(base, scaled)
        # Reverse: scaled vs base
        metrics_rev = _compute_metrics(scaled, base)

        # Alignment should be the same either way
        assert abs(metrics_fwd.spectral_alignment - metrics_rev.spectral_alignment) <= eps


# =============================================================================
# Condition Number Invariants
# =============================================================================


class TestConditionNumberInvariants:
    """Tests for condition number bounds."""

    @pytest.mark.parametrize("seed", range(5))
    def test_condition_number_at_least_one(self, seed: int) -> None:
        """Condition number must be >= 1.

        Mathematical property: κ = σ_max / σ_min ≥ 1 since σ_max ≥ σ_min.
        """
        backend = get_default_backend()
        backend.random_seed(seed)

        source = backend.random_normal((10, 8))
        source = backend.astype(source, "float32")
        target = backend.random_normal((10, 8))
        target = backend.astype(target, "float32")
        backend.eval(source, target)

        metrics = _compute_metrics(source, target)

        assert metrics.condition_number >= 1.0

    def test_identity_has_condition_one(self) -> None:
        """Identity matrix should have condition number = 1.

        Mathematical property: All singular values of I are 1.
        """
        backend = get_default_backend()
        identity = backend.eye(10)
        identity = backend.astype(identity, "float32")
        backend.eval(identity)

        metrics = _compute_metrics(identity, identity)

        eps = machine_epsilon(backend, identity)
        assert abs(metrics.condition_number - 1.0) <= eps

    def test_ill_conditioned_detection(self) -> None:
        """Ill-conditioned matrix should be detected.

        Mathematical property: High ratio σ_max/σ_min indicates ill-conditioning.
        """
        backend = get_default_backend()
        backend.random_seed(42)

        # Create ill-conditioned matrix with controlled singular values
        u = backend.random_normal((10, 10))
        v = backend.random_normal((10, 10))
        u, _ = backend.qr(u)
        v, _ = backend.qr(v)
        s = backend.array([100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001])
        s = backend.diag(s)
        ill_matrix = backend.matmul(backend.matmul(u, s), v)
        backend.eval(ill_matrix)

        target = backend.astype(ill_matrix, "float32")
        source = backend.eye(10)
        source = backend.astype(source, "float32")
        backend.eval(source, target)

        metrics = _compute_metrics(source, target)

        expected_condition = 100.0 / 0.001
        assert metrics.condition_number >= expected_condition
        assert metrics.condition_number > 100


# =============================================================================
# 1D Vector Invariants
# =============================================================================


class Test1DVectorInvariants:
    """Tests for 1D vector (bias/layernorm) handling."""

    @pytest.mark.parametrize("seed", range(5))
    def test_1d_spectral_norm_non_negative(self, seed: int) -> None:
        """1D spectral norm (L2 norm) must be >= 0."""
        backend = get_default_backend()
        backend.random_seed(seed)

        source = backend.random_normal((10,))
        source = backend.astype(source, "float32")
        target = backend.random_normal((10,))
        target = backend.astype(target, "float32")
        backend.eval(source, target)

        metrics = _compute_metrics(source, target)

        assert metrics.source_spectral_norm >= 0
        assert metrics.target_spectral_norm >= 0

    def test_1d_condition_number_is_one(self) -> None:
        """1D vectors should have condition number = 1 (convention)."""
        backend = get_default_backend()
        backend.random_seed(42)

        source = backend.random_normal((10,))
        source = backend.astype(source, "float32")
        target = backend.random_normal((10,))
        target = backend.astype(target, "float32")
        backend.eval(source, target)

        metrics = _compute_metrics(source, target)

        assert metrics.condition_number == 1.0


# =============================================================================
# Spectral Ratio Invariants
# =============================================================================


class TestSpectralRatioInvariants:
    """Tests for spectral ratio properties."""

    @pytest.mark.parametrize("seed", range(5))
    def test_spectral_ratio_positive(self, seed: int) -> None:
        """Spectral ratio must be > 0.

        Mathematical property: Ratio of positive values is positive.
        """
        backend = get_default_backend()
        backend.random_seed(seed)

        source = backend.random_normal((10, 8))
        source = backend.astype(source, "float32")
        target = backend.random_normal((10, 8))
        target = backend.astype(target, "float32")
        backend.eval(source, target)

        metrics = _compute_metrics(source, target)

        assert metrics.spectral_ratio > 0

    def test_zero_delta_for_identical(self) -> None:
        """Identical matrices should have delta_frobenius = 0."""
        backend = get_default_backend()
        backend.random_seed(42)
        matrix = backend.random_normal((10, 8))
        matrix = backend.astype(matrix, "float32")
        backend.eval(matrix)

        metrics = _compute_metrics(matrix, matrix)

        eps = machine_epsilon(backend, matrix)
        assert abs(metrics.delta_frobenius) <= eps
