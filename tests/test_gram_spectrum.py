# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for gram_spectrum.py — compute_gram_spectrum, analyze_projection,
and compute_geometry_derived_scale."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_spectrum import (
    GramSpectrumResult,
    analyze_projection,
    compute_geometry_derived_scale,
    compute_gram_spectrum,
)


def _b():
    return get_default_backend()


def _rand(shape, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _gram_result(
    *,
    null_rank: int = 10,
    numeric_rank: int | None = None,
    condition_number: float = 10.0,
    spectral_gap: float = 0.5,
    d_features: int = 32,
) -> GramSpectrumResult:
    """Build a minimal GramSpectrumResult for testing compute_geometry_derived_scale."""
    nr = numeric_rank if numeric_rank is not None else d_features - null_rank
    return GramSpectrumResult(
        n_samples=16,
        d_features=d_features,
        eigenvalues=[1.0] * min(16, d_features),
        total_variance=16.0,
        max_eigenvalue=2.0,
        min_eigenvalue=0.01,
        condition_number=condition_number,
        numeric_rank=nr,
        null_rank=null_rank,
        intrinsic_dimension=4.0,
        energy_ratio_numeric_rank=0.9,
        energy_ratio_intrinsic_dim=0.5,
        spectral_gap=spectral_gap,
        rank_threshold=0.01,
    )


# ---------------------------------------------------------------------------
# GS1: Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_1d_raises(self) -> None:
        b = _b()
        arr = b.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        with pytest.raises(ValueError):
            compute_gram_spectrum(arr, backend=b)

    def test_empty_rows_raises(self) -> None:
        b = _b()
        arr = b.array(np.zeros((0, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            compute_gram_spectrum(arr, backend=b)


# ---------------------------------------------------------------------------
# GS2: Eigenvalue Invariants
# ---------------------------------------------------------------------------

class TestEigenvalueInvariants:
    def test_eigenvalues_nonneg(self) -> None:
        """G = A@A.T is PSD by construction → all eigenvalues ≥ 0 (within float tolerance)."""
        b = _b()
        result = compute_gram_spectrum(b.array(_rand((16, 8), seed=1)), backend=b)
        for v in result.eigenvalues:
            assert v >= -1e-5, f"Negative eigenvalue: {v}"

    def test_total_variance_equals_frobenius_sq(self) -> None:
        """total_variance = trace(G) = trace(A@A.T) = ||A||_F² (cyclic trace property)."""
        b = _b()
        A_np = _rand((16, 8), seed=2)
        frob_sq = float(np.sum(A_np ** 2))
        result = compute_gram_spectrum(b.array(A_np), backend=b)
        rel_err = abs(result.total_variance - frob_sq) / max(frob_sq, 1e-8)
        assert rel_err < 1e-3, (
            f"total_variance={result.total_variance:.6f} != ||A||_F²={frob_sq:.6f} "
            f"(rel_err={rel_err:.2e})"
        )


# ---------------------------------------------------------------------------
# GS3: Shape Metadata
# ---------------------------------------------------------------------------

class TestShapeMetadata:
    def test_n_samples_d_features_match_input(self) -> None:
        b = _b()
        n, d = 12, 6
        result = compute_gram_spectrum(b.array(_rand((n, d), seed=3)), backend=b)
        assert result.n_samples == n
        assert result.d_features == d

    def test_null_rank_nonneg(self) -> None:
        b = _b()
        result = compute_gram_spectrum(b.array(_rand((16, 8), seed=4)), backend=b)
        assert result.null_rank >= 0

    def test_numeric_plus_null_eq_d_features(self) -> None:
        """numeric_rank + null_rank == d_features (null_rank = max(0, d - numeric_rank))."""
        b = _b()
        result = compute_gram_spectrum(b.array(_rand((16, 8), seed=5)), backend=b)
        assert result.numeric_rank + result.null_rank == result.d_features


# ---------------------------------------------------------------------------
# GS4: Rank Detection
# ---------------------------------------------------------------------------

class TestRankDetection:
    def test_rank1_matrix_has_numeric_rank_1(self) -> None:
        """Rank-1 outer product → Gram matrix G = A@A.T is rank 1 → numeric_rank == 1."""
        b = _b()
        rng = np.random.default_rng(6)
        u = rng.standard_normal(16).astype(np.float32)
        v = rng.standard_normal(8).astype(np.float32)
        # Scale up to avoid numerical rank collapse at machine precision
        A_np = np.outer(u * 10.0, v * 10.0)  # [16, 8], rank 1
        result = compute_gram_spectrum(b.array(A_np), backend=b)
        assert result.numeric_rank == 1, (
            f"Expected numeric_rank=1 for rank-1 outer product, got {result.numeric_rank}"
        )


# ---------------------------------------------------------------------------
# GS5: Frozen Result
# ---------------------------------------------------------------------------

class TestResultIsFrozen:
    def test_mutating_field_raises(self) -> None:
        b = _b()
        result = compute_gram_spectrum(b.array(_rand((8, 4), seed=7)), backend=b)
        with pytest.raises(FrozenInstanceError):
            result.n_samples = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GS6: analyze_projection
# ---------------------------------------------------------------------------

class TestAnalyzeProjection:
    def test_preserved_fraction_in_0_1(self) -> None:
        """Output is a fraction: 0 ≤ preserved_fraction ≤ 1."""
        b = _b()
        A = _rand((16, 8), seed=10)
        dW = _rand((6, 8), seed=11)
        result = analyze_projection(dW, A, backend=b)
        assert 0.0 <= result.preserved_fraction <= 1.0 + 1e-5, (
            f"preserved_fraction={result.preserved_fraction} out of [0, 1]"
        )

    def test_projection_contracts_norm(self) -> None:
        """Projection can only shrink or preserve the output-space norm."""
        b = _b()
        A = _rand((16, 8), seed=12)
        dW = _rand((6, 8), seed=13)
        result = analyze_projection(dW, A, backend=b)
        assert result.projected_norm <= result.delta_norm + 1e-4, (
            f"projected_norm={result.projected_norm:.6f} > delta_norm={result.delta_norm:.6f}"
        )

    def test_row_space_delta_removed(self) -> None:
        """delta_W = C @ A (in row space of A) → projection removes all behavioral effect.

        Proof: delta_proj = delta_W - (delta_W @ A.T) @ G^+ @ A
        With delta_W = C @ A and G = A @ A.T:
          delta_W @ A.T = C @ G
          correction = C @ G @ G^+ @ A = C @ A = delta_W  (since G @ G^+ @ A = A)
          delta_proj = 0 → projected_norm = 0.
        """
        b = _b()
        rng = np.random.default_rng(14)
        n, in_dim, out_dim = 8, 4, 4
        A_np = rng.standard_normal((n, in_dim)).astype(np.float32)
        C_np = rng.standard_normal((out_dim, n)).astype(np.float32)
        dW_np = C_np @ A_np  # [out_dim, in_dim] — in row space of A
        result = analyze_projection(dW_np, A_np, backend=b)
        # After projection, behavioral output change must vanish
        assert result.projected_norm <= result.delta_norm * 0.01 + 1e-3, (
            f"Row-space delta not removed: projected_norm={result.projected_norm:.6f}, "
            f"delta_norm={result.delta_norm:.6f}"
        )

    def test_zero_delta_returns_zero_norms(self) -> None:
        """Zero weight delta → no behavioral change → all norms = 0."""
        b = _b()
        A = _rand((8, 4), seed=15)
        dW = np.zeros((4, 4), dtype=np.float32)
        result = analyze_projection(dW, A, backend=b)
        assert result.delta_norm == pytest.approx(0.0, abs=1e-6)
        assert result.projected_norm == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# GS7: compute_geometry_derived_scale
# ---------------------------------------------------------------------------

class TestGeometryDerivedScale:
    def test_scale_in_0_1(self) -> None:
        """Output is always in [0, 1] for any valid GramSpectrumResult."""
        scale = compute_geometry_derived_scale(_gram_result())
        assert 0.0 <= scale <= 1.0

    def test_null_rank_zero_gives_zero(self) -> None:
        """null_rank=0 → null_fraction=0 → product=0 (no capacity for transfer)."""
        result = _gram_result(null_rank=0, d_features=32, numeric_rank=32)
        assert compute_geometry_derived_scale(result) == pytest.approx(0.0)

    def test_huge_condition_number_reduces_scale(self) -> None:
        """Higher condition number means more precision lost → smaller scale."""
        good = compute_geometry_derived_scale(_gram_result(condition_number=10.0))
        bad = compute_geometry_derived_scale(_gram_result(condition_number=1e15))
        assert bad < good, (
            f"Expected kappa=1e15 to give smaller scale than kappa=10, "
            f"got {bad:.4f} vs {good:.4f}"
        )
