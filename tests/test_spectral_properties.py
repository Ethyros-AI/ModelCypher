"""Property-based tests for SpectralAnalysis.

Tests mathematical invariants:
- Spectral norm ≥ 0 (non-negative)
- Spectral confidence ∈ [0, 1]
- Condition number ≥ 1 (by definition)
- Spectral ratio symmetry: confidence(a/b) = confidence(b/a)
- Frobenius norm ≥ 0
"""

from __future__ import annotations

import numpy as np
import pytest

from modelcypher.core.domain.geometry.spectral_analysis import (
    compute_spectral_metrics,
    SpectralConfig,
    SpectralMetrics,
)


# =============================================================================
# Spectral Norm Invariants
# =============================================================================


class TestSpectralNormInvariants:
    """Tests for spectral norm bounds."""

    @pytest.mark.parametrize("seed", range(5))
    def test_spectral_norm_non_negative(self, seed: int) -> None:
        """Spectral norm (max singular value) must be >= 0.

        Mathematical property: Singular values are non-negative by definition.
        """
        rng = np.random.default_rng(seed)

        source = rng.standard_normal((10, 8)).astype(np.float32)
        target = rng.standard_normal((10, 8)).astype(np.float32)

        metrics = compute_spectral_metrics(source, target)

        assert metrics.source_spectral_norm >= 0
        assert metrics.target_spectral_norm >= 0

    @pytest.mark.parametrize("seed", range(5))
    def test_frobenius_norm_non_negative(self, seed: int) -> None:
        """Frobenius norm must be >= 0.

        Mathematical property: ||A||_F = sqrt(Σ|a_ij|²) ≥ 0.
        """
        rng = np.random.default_rng(seed)

        source = rng.standard_normal((10, 8)).astype(np.float32)
        target = rng.standard_normal((10, 8)).astype(np.float32)

        metrics = compute_spectral_metrics(source, target)

        assert metrics.delta_frobenius >= 0


# =============================================================================
# Spectral Confidence Invariants
# =============================================================================


class TestSpectralConfidenceInvariants:
    """Tests for spectral confidence bounds."""

    @pytest.mark.parametrize("seed", range(5))
    def test_spectral_confidence_in_zero_one(self, seed: int) -> None:
        """Spectral confidence must be in [0, 1].

        Mathematical property: confidence = min(r, 1/r) where r > 0.
        """
        rng = np.random.default_rng(seed)

        source = rng.standard_normal((10, 8)).astype(np.float32)
        target = rng.standard_normal((10, 8)).astype(np.float32)

        metrics = compute_spectral_metrics(source, target)

        assert 0.0 <= metrics.spectral_confidence <= 1.0

    def test_identical_matrices_have_unit_confidence(self) -> None:
        """Identical matrices should have spectral confidence = 1.

        Mathematical property: ratio = 1 → confidence = min(1, 1) = 1.
        """
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((10, 8)).astype(np.float32)

        metrics = compute_spectral_metrics(matrix, matrix)

        assert metrics.spectral_confidence == pytest.approx(1.0, abs=1e-6)
        assert metrics.spectral_ratio == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("scale", [0.1, 0.5, 2.0, 10.0])
    def test_spectral_confidence_symmetric(self, scale: float) -> None:
        """Spectral confidence should be symmetric with respect to scaling.

        Mathematical property: confidence(a/b) = confidence(b/a).
        """
        rng = np.random.default_rng(42)
        base = rng.standard_normal((10, 8)).astype(np.float32)
        scaled = base * scale

        # Forward: base vs scaled
        metrics_fwd = compute_spectral_metrics(base, scaled)
        # Reverse: scaled vs base
        metrics_rev = compute_spectral_metrics(scaled, base)

        # Confidence should be the same either way
        assert metrics_fwd.spectral_confidence == pytest.approx(
            metrics_rev.spectral_confidence, rel=1e-6
        )


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
        rng = np.random.default_rng(seed)

        source = rng.standard_normal((10, 8)).astype(np.float32)
        target = rng.standard_normal((10, 8)).astype(np.float32)

        metrics = compute_spectral_metrics(source, target)

        assert metrics.condition_number >= 1.0

    def test_identity_has_condition_one(self) -> None:
        """Identity matrix should have condition number = 1.

        Mathematical property: All singular values of I are 1.
        """
        identity = np.eye(10, dtype=np.float32)

        metrics = compute_spectral_metrics(identity, identity)

        assert metrics.condition_number == pytest.approx(1.0, abs=1e-6)

    def test_ill_conditioned_detection(self) -> None:
        """Ill-conditioned matrix should be detected.

        Mathematical property: High ratio σ_max/σ_min indicates ill-conditioning.
        """
        # Create ill-conditioned matrix
        rng = np.random.default_rng(42)
        U, _, Vt = np.linalg.svd(rng.standard_normal((10, 10)), full_matrices=False)
        # Singular values with high ratio
        s = np.array([100, 10, 1, 0.1, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001])
        ill_matrix = (U * s) @ Vt

        target = ill_matrix.astype(np.float32)
        source = np.eye(10, dtype=np.float32)

        metrics = compute_spectral_metrics(source, target)

        assert metrics.is_ill_conditioned is True
        assert metrics.condition_number > 100


# =============================================================================
# 1D Vector Invariants
# =============================================================================


class Test1DVectorInvariants:
    """Tests for 1D vector (bias/layernorm) handling."""

    @pytest.mark.parametrize("seed", range(5))
    def test_1d_spectral_norm_non_negative(self, seed: int) -> None:
        """1D spectral norm (L2 norm) must be >= 0."""
        rng = np.random.default_rng(seed)

        source = rng.standard_normal(10).astype(np.float32)
        target = rng.standard_normal(10).astype(np.float32)

        metrics = compute_spectral_metrics(source, target)

        assert metrics.source_spectral_norm >= 0
        assert metrics.target_spectral_norm >= 0

    def test_1d_condition_number_is_one(self) -> None:
        """1D vectors should have condition number = 1 (convention)."""
        rng = np.random.default_rng(42)

        source = rng.standard_normal(10).astype(np.float32)
        target = rng.standard_normal(10).astype(np.float32)

        metrics = compute_spectral_metrics(source, target)

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
        rng = np.random.default_rng(seed)

        source = rng.standard_normal((10, 8)).astype(np.float32)
        target = rng.standard_normal((10, 8)).astype(np.float32)

        metrics = compute_spectral_metrics(source, target)

        assert metrics.spectral_ratio > 0

    def test_zero_delta_for_identical(self) -> None:
        """Identical matrices should have delta_frobenius = 0."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((10, 8)).astype(np.float32)

        metrics = compute_spectral_metrics(matrix, matrix)

        assert metrics.delta_frobenius == pytest.approx(0.0, abs=1e-6)
