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

"""Tests for geometry_fingerprint module."""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.geometry_fingerprint import (
    AnchorSet,
    GeometricFingerprint,
    _mean_abs_diff,
)


@pytest.fixture
def backend():
    return get_default_backend()


def _eps(backend, *args):
    """Machine epsilon for float32."""
    return float(backend.finfo().eps)


class TestGramStatistics:
    """Tests for GeometricFingerprint.gram_statistics."""

    def test_identity_gram_has_zero_off_diagonal_mean(self, backend):
        """Identity matrix has all zeros off-diagonal."""
        n = 3
        # Identity: diag = 1, off-diag = 0
        gram = [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]
        mean, std, gram_hash = GeometricFingerprint.gram_statistics(gram, n)

        eps = _eps(backend)
        assert abs(mean) <= eps
        assert abs(std) <= eps
        assert len(gram_hash) == 64  # SHA256 hex length

    def test_constant_off_diagonal_has_zero_std(self, backend):
        """Gram with constant off-diagonal has std = 0."""
        n = 3
        # All off-diag = 0.5, diag = 1.0
        gram = [1.0 if i == j else 0.5 for i in range(n) for j in range(n)]
        mean, std, gram_hash = GeometricFingerprint.gram_statistics(gram, n)

        eps = _eps(backend)
        assert abs(mean - 0.5) <= eps
        assert abs(std) <= eps

    def test_gram_hash_is_deterministic(self, backend):
        """Same gram produces same hash."""
        n = 2
        gram = [1.0, 0.3, 0.3, 1.0]
        _, _, hash1 = GeometricFingerprint.gram_statistics(gram, n)
        _, _, hash2 = GeometricFingerprint.gram_statistics(gram, n)
        assert hash1 == hash2

    def test_different_grams_have_different_hashes(self, backend):
        """Different grams produce different hashes."""
        n = 2
        gram1 = [1.0, 0.3, 0.3, 1.0]
        gram2 = [1.0, 0.4, 0.4, 1.0]
        _, _, hash1 = GeometricFingerprint.gram_statistics(gram1, n)
        _, _, hash2 = GeometricFingerprint.gram_statistics(gram2, n)
        assert hash1 != hash2

    def test_invalid_gram_size_returns_zeros(self, backend):
        """Wrong gram size returns zeros and empty hash."""
        gram = [1.0, 0.5]  # Wrong size for n=3
        mean, std, gram_hash = GeometricFingerprint.gram_statistics(gram, 3)
        assert mean == 0.0
        assert std == 0.0
        assert gram_hash == ""

    def test_n_equals_1_returns_zeros(self, backend):
        """Single element gram returns zeros."""
        gram = [1.0]
        mean, std, gram_hash = GeometricFingerprint.gram_statistics(gram, 1)
        assert mean == 0.0
        assert std == 0.0
        assert gram_hash == ""


class TestEstimateSpectralRadius:
    """Tests for GeometricFingerprint.estimate_spectral_radius."""

    def test_identity_has_spectral_radius_one(self, backend):
        """Identity matrix has spectral radius = 1."""
        n = 3
        gram = [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]
        radius = GeometricFingerprint.estimate_spectral_radius(gram, n)

        # Power iteration converges to largest eigenvalue
        assert abs(radius - 1.0) <= 0.01  # Within 1%

    def test_scaled_identity_has_spectral_radius_scale(self, backend):
        """Scaled identity has spectral radius = scale."""
        n = 3
        scale = 5.0
        gram = [scale if i == j else 0.0 for i in range(n) for j in range(n)]
        radius = GeometricFingerprint.estimate_spectral_radius(gram, n)

        assert abs(radius - scale) <= 0.1  # Within 2%

    def test_invalid_gram_returns_zero(self, backend):
        """Invalid gram size returns 0."""
        gram = [1.0, 0.5]  # Wrong size
        radius = GeometricFingerprint.estimate_spectral_radius(gram, 3)
        assert radius == 0.0

    def test_zero_n_returns_zero(self, backend):
        """n=0 returns 0."""
        radius = GeometricFingerprint.estimate_spectral_radius([], 0)
        assert radius == 0.0


class TestEstimateConditionNumber:
    """Tests for GeometricFingerprint.estimate_condition_number."""

    def test_identity_has_condition_number_one(self, backend):
        """Identity matrix has condition number = 1."""
        n = 3
        gram = [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]
        cond = GeometricFingerprint.estimate_condition_number(gram, n)

        # Condition number = max_eig / min_eig = 1/1 = 1
        assert abs(cond - 1.0) <= 0.1

    def test_scaled_identity_has_condition_number_one(self, backend):
        """Scaled identity still has condition number = 1."""
        n = 3
        scale = 5.0
        gram = [scale if i == j else 0.0 for i in range(n) for j in range(n)]
        cond = GeometricFingerprint.estimate_condition_number(gram, n)

        assert abs(cond - 1.0) <= 0.1

    def test_ill_conditioned_has_large_condition_number(self, backend):
        """Ill-conditioned matrix has large condition number."""
        n = 3
        # Diagonal with very different eigenvalues
        gram = [0.0] * (n * n)
        gram[0] = 100.0  # Large eigenvalue
        gram[4] = 1.0  # Medium
        gram[8] = 0.01  # Small eigenvalue
        cond = GeometricFingerprint.estimate_condition_number(gram, n)

        # Should be around 100 / 0.01 = 10000
        assert cond >= 1000.0


class TestEstimateEffectiveDimensionality:
    """Tests for GeometricFingerprint.estimate_effective_dimensionality."""

    def test_identity_has_full_dimensionality(self, backend):
        """Identity matrix has effective dim = n."""
        n = 4
        gram = [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]
        dim = GeometricFingerprint.estimate_effective_dimensionality(gram, n)

        # Effective dimensionality = (sum_eig)^2 / sum(eig^2) = n^2 / n = n
        assert abs(dim - float(n)) <= 0.5

    def test_rank_one_has_dimensionality_one(self, backend):
        """Rank-1 matrix has effective dim = 1."""
        n = 4
        # All eigenvalues = 0 except one = n
        gram = [0.0] * (n * n)
        gram[0] = float(n)  # Single eigenvalue
        dim = GeometricFingerprint.estimate_effective_dimensionality(gram, n)

        # Effective dimensionality should be close to 1
        assert dim <= 2.0


class TestSymmetricEigenvalues:
    """Tests for GeometricFingerprint.symmetric_eigenvalues."""

    def test_identity_eigenvalues_are_ones(self, backend):
        """Identity matrix has all eigenvalues = 1."""
        n = 3
        gram = [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(gram, n)

        assert eigenvalues is not None
        assert len(eigenvalues) == n
        eps = _eps(backend) * 100  # Jacobi has lower precision
        for eig in eigenvalues:
            assert abs(eig - 1.0) <= eps

    def test_diagonal_eigenvalues_are_diagonal_elements(self, backend):
        """Diagonal matrix eigenvalues are the diagonal elements."""
        n = 3
        diag_vals = [3.0, 1.0, 2.0]
        gram = [diag_vals[i] if i == j else 0.0 for i in range(n) for j in range(n)]
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(gram, n)

        assert eigenvalues is not None
        assert len(eigenvalues) == n
        sorted_eig = sorted(eigenvalues)
        sorted_diag = sorted(diag_vals)
        eps = _eps(backend) * 100
        for i in range(n):
            assert abs(sorted_eig[i] - sorted_diag[i]) <= eps

    def test_single_element_returns_element(self, backend):
        """n=1 returns the single element."""
        gram = [5.0]
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(gram, 1)
        assert eigenvalues == [5.0]

    def test_invalid_gram_returns_none(self, backend):
        """Invalid gram size returns None."""
        gram = [1.0, 0.5]  # Wrong size
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues(gram, 3)
        assert eigenvalues is None

    def test_zero_n_returns_none(self, backend):
        """n=0 returns None."""
        eigenvalues = GeometricFingerprint.symmetric_eigenvalues([], 0)
        assert eigenvalues is None


class TestGeometricFingerprintDataclass:
    """Tests for GeometricFingerprint dataclass."""

    def test_placeholder_exists(self, backend):
        """Placeholder instance exists with expected fields."""
        placeholder = GeometricFingerprint.placeholder
        assert placeholder.gram_hash == "placeholder"
        assert placeholder.anchor_set == AnchorSet.hybrid
        assert placeholder.anchor_count == 131

    def test_anchor_set_enum_values(self, backend):
        """AnchorSet enum has expected values."""
        assert AnchorSet.semantic_primes.value == "semanticPrimes"
        assert AnchorSet.computational_gates.value == "computationalGates"
        assert AnchorSet.hybrid.value == "hybrid"
        assert AnchorSet.custom.value == "custom"


class TestMeanAbsDiff:
    """Tests for _mean_abs_diff helper function."""

    def test_identical_sequences_have_zero_diff(self, backend):
        """Identical sequences have mean diff = 0."""
        lhs = [1.0, 2.0, 3.0]
        rhs = [1.0, 2.0, 3.0]
        diff = _mean_abs_diff(lhs, rhs)
        assert diff == 0.0

    def test_shifted_sequence_has_nonzero_diff(self, backend):
        """Shifted sequence has predictable diff."""
        lhs = [1.0, 2.0, 3.0]
        rhs = [2.0, 3.0, 4.0]
        diff = _mean_abs_diff(lhs, rhs)
        assert diff == 1.0  # Each element differs by 1

    def test_empty_sequences_return_zero(self, backend):
        """Empty sequences return 0."""
        diff = _mean_abs_diff([], [])
        assert diff == 0.0

    def test_mismatched_lengths_uses_minimum(self, backend):
        """Mismatched lengths uses minimum length."""
        lhs = [1.0, 2.0, 3.0]
        rhs = [1.0, 2.0]
        diff = _mean_abs_diff(lhs, rhs)
        assert diff == 0.0  # First two elements match
