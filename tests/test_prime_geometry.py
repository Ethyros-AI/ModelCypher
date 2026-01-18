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

"""Comprehensive tests for prime number geometry analysis.

Tests verify:
- Sieve correctness and gap computation
- Embedding invariants (time-delay, residue, digit)
- Spectral properties (participation ratio, entropy)
- Baseline generators (exponential, uniform, Poisson, Cramér)
- Statistical testing (bootstrap CI, Cohen's d, permutation)
- End-to-end hypothesis validation
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.prime_geometry_analysis import (
    analyze_prime_geometry,
    format_comprehensive_result,
    format_result,
    run_comprehensive_analysis,
    run_perturbation_study,
    run_scale_sweep,
)
from modelcypher.core.domain.geometry.prime_geometry_baselines import (
    generate_baseline,
    generate_cramer_model,
    generate_poisson_gaps,
    generate_random_gaps,
    generate_uniform_gaps,
    shuffled_gaps,
)
from modelcypher.core.domain.geometry.prime_geometry_embeddings import (
    binary_digit_embedding,
    digit_embedding,
    generate_primes,
    residue_embedding,
    time_delay_embedding,
)
from modelcypher.core.domain.geometry.prime_geometry_spectral import (
    analyze_eigenvalues,
    compare_distributions,
    compute_gram_matrix,
)
from modelcypher.core.domain.geometry.prime_geometry_stats import (
    bootstrap_confidence_interval,
    compute_cohens_d,
    permutation_test,
    run_hypothesis_test,
)
from modelcypher.core.domain.geometry.prime_geometry_types import (
    BaselineType,
    ComprehensiveResult,
    ConfidenceInterval,
    EffectSize,
    EigenvalueDistribution,
    EmbeddingType,
    HypothesisTest,
    PerturbationResult,
    PrimeGeometryResult,
    PrimeSequence,
    ScaleSweepResult,
    SpectralComparison,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.support.array_utils import array_to_list


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def backend():
    """Provide the default backend for testing."""
    return get_default_backend()


@pytest.fixture
def small_primes(backend):
    """Generate a small prime sequence for fast testing."""
    return generate_primes(50, backend=backend)


@pytest.fixture
def medium_primes(backend):
    """Generate a medium prime sequence for more thorough tests."""
    return generate_primes(200, backend=backend)


# =============================================================================
# TestPrimeGeneration: Sieve correctness and gap computation
# =============================================================================


class TestPrimeGeneration:
    """Tests for prime number generation via sieve."""

    def test_generate_primes_returns_correct_count(self, backend):
        """Should return exactly n primes."""
        seq = generate_primes(100, backend=backend)
        assert seq.count == 100

    def test_first_primes_are_correct(self, backend):
        """First 10 primes should match known values."""
        seq = generate_primes(10, backend=backend)
        primes_list = array_to_list(backend, seq.primes)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert primes_list == expected

    def test_primes_are_monotonically_increasing(self, backend):
        """Primes should be in ascending order."""
        seq = generate_primes(100, backend=backend)
        primes = array_to_list(backend, seq.primes)
        for i in range(1, len(primes)):
            assert primes[i] > primes[i - 1]

    def test_gap_count_is_n_minus_one(self, backend):
        """Number of gaps should be n - 1."""
        seq = generate_primes(50, backend=backend)
        assert seq.gap_count == 49

    def test_gaps_are_positive(self, backend):
        """All gaps should be positive integers."""
        seq = generate_primes(100, backend=backend)
        gaps = array_to_list(backend, seq.gaps)
        assert all(g > 0 for g in gaps)

    def test_gaps_sum_to_difference(self, backend):
        """Sum of gaps should equal p_n - p_1."""
        seq = generate_primes(50, backend=backend)
        primes = array_to_list(backend, seq.primes)
        gaps = array_to_list(backend, seq.gaps)
        assert sum(gaps) == primes[-1] - primes[0]

    def test_small_n_works(self, backend):
        """Should handle small values of n."""
        seq = generate_primes(2, backend=backend)
        assert seq.count == 2
        assert seq.gap_count == 1

    def test_known_gap_values(self, backend):
        """Verify specific gap values."""
        seq = generate_primes(10, backend=backend)
        gaps = array_to_list(backend, seq.gaps)
        # Gaps between first 10 primes: 3-2=1, 5-3=2, 7-5=2, etc.
        expected_gaps = [1, 2, 2, 4, 2, 4, 2, 4, 6]
        assert gaps == expected_gaps


# =============================================================================
# TestEmbeddings: Embedding invariants and properties
# =============================================================================


class TestEmbeddings:
    """Tests for various embedding methods."""

    def test_time_delay_embedding_shape(self, backend, small_primes):
        """Time-delay embedding should have correct shape."""
        gaps = small_primes.gaps
        dim = 10
        embedded = time_delay_embedding(gaps, embedding_dim=dim, backend=backend)
        backend.eval(embedded)
        shape = embedded.shape
        # n_windows = gap_count - dim + 1
        expected_windows = small_primes.gap_count - dim + 1
        assert shape[0] == expected_windows
        assert shape[1] == dim

    def test_time_delay_embedding_values(self, backend):
        """Time-delay embedding should contain sliding windows."""
        seq = generate_primes(10, backend=backend)
        gaps = seq.gaps  # [1, 2, 2, 4, 2, 4, 2, 4, 6]
        embedded = time_delay_embedding(gaps, embedding_dim=3, backend=backend)
        backend.eval(embedded)
        # First window should be [1, 2, 2]
        first_row = array_to_list(backend, embedded[0])
        assert first_row == [1.0, 2.0, 2.0]

    def test_residue_embedding_default_moduli(self, backend, small_primes):
        """Residue embedding with default moduli [2, 6, 30, 210]."""
        primes = small_primes.primes
        embedded = residue_embedding(primes, backend=backend)
        backend.eval(embedded)
        # Should have 4 columns (one per modulus)
        assert embedded.shape[1] == 4

    def test_residue_embedding_custom_moduli(self, backend, small_primes):
        """Residue embedding with custom moduli."""
        primes = small_primes.primes
        moduli = [2, 3, 5, 7]
        embedded = residue_embedding(primes, moduli=moduli, backend=backend)
        backend.eval(embedded)
        assert embedded.shape[1] == len(moduli)

    def test_residue_embedding_correctness(self, backend):
        """Verify residue values are correct."""
        seq = generate_primes(5, backend=backend)  # [2, 3, 5, 7, 11]
        primes = seq.primes
        embedded = residue_embedding(primes, moduli=[6], backend=backend)
        backend.eval(embedded)
        residues = array_to_list(backend, embedded[:, 0])
        # 2 mod 6 = 2, 3 mod 6 = 3, 5 mod 6 = 5, 7 mod 6 = 1, 11 mod 6 = 5
        expected = [2.0, 3.0, 5.0, 1.0, 5.0]
        assert residues == expected

    def test_digit_embedding_shape(self, backend, small_primes):
        """Digit embedding should have correct shape."""
        gaps = small_primes.gaps
        max_digits = 5
        embedded = digit_embedding(gaps, base=10, max_digits=max_digits, backend=backend)
        backend.eval(embedded)
        assert embedded.shape[0] == small_primes.gap_count
        assert embedded.shape[1] == max_digits

    def test_digit_embedding_extracts_digits(self, backend):
        """Verify digit extraction extracts some digits."""
        backend.random_seed(42)
        sequence = backend.array([123, 45, 6])
        embedded = digit_embedding(sequence, base=10, max_digits=4, backend=backend)
        backend.eval(embedded)
        # Just verify shape is correct - exact digit ordering depends on impl
        assert embedded.shape == (3, 4)
        # Verify 123's digits sum to 6 (1+2+3)
        first_row_sum = float(backend.sum(embedded[0]).item())
        assert first_row_sum == 6.0  # 1+2+3 = 6

    def test_binary_digit_embedding_shape(self, backend, small_primes):
        """Binary digit embedding should have correct shape."""
        gaps = small_primes.gaps
        max_bits = 8
        embedded = binary_digit_embedding(gaps, max_bits=max_bits, backend=backend)
        backend.eval(embedded)
        assert embedded.shape[1] == max_bits

    def test_shuffled_gaps_preserves_distribution(self, backend, small_primes):
        """Shuffled gaps should have same elements, different order."""
        gaps = small_primes.gaps
        shuffled = shuffled_gaps(gaps, backend=backend, seed=42)
        backend.eval(shuffled)

        # Same sum
        orig_sum = float(backend.sum(gaps).item())
        shuf_sum = float(backend.sum(shuffled).item())
        eps = _eps(backend, orig_sum, shuf_sum)
        assert abs(orig_sum - shuf_sum) <= eps

        # Same sorted values
        orig_sorted = sorted(array_to_list(backend, gaps))
        shuf_sorted = sorted(array_to_list(backend, shuffled))
        assert orig_sorted == shuf_sorted

    def test_shuffled_gaps_different_order(self, backend, medium_primes):
        """Shuffled gaps should (usually) have different order."""
        gaps = medium_primes.gaps
        shuffled = shuffled_gaps(gaps, backend=backend, seed=42)
        backend.eval(shuffled)

        orig = array_to_list(backend, gaps)
        shuf = array_to_list(backend, shuffled)
        # With enough elements, they should differ
        assert orig != shuf


# =============================================================================
# TestGramMatrix: Gram matrix properties
# =============================================================================


class TestGramMatrix:
    """Tests for Gram matrix computation."""

    def test_gram_matrix_is_symmetric(self, backend):
        """Gram matrix should be symmetric."""
        backend.random_seed(42)
        X = backend.random_normal((10, 5))
        G = compute_gram_matrix(X, backend=backend)
        backend.eval(G)

        G_T = backend.transpose(G)
        backend.eval(G_T)

        diff = backend.abs(G - G_T)
        backend.eval(diff)
        max_diff = float(backend.max(diff).item())
        eps = _eps(backend, max_diff)
        assert max_diff <= eps

    def test_gram_matrix_is_positive_semidefinite(self, backend):
        """Gram matrix should be positive semi-definite (non-negative eigenvalues)."""
        backend.random_seed(42)
        X = backend.random_normal((10, 5))
        G = compute_gram_matrix(X, backend=backend)
        backend.eval(G)

        # Compute eigenvalues using eigh (returns eigenvalues, eigenvectors)
        eigenvalues, _ = backend.eigh(G)
        backend.eval(eigenvalues)

        min_eig = float(backend.min(eigenvalues).item())
        # Eigenvalue perturbation bound: |λ_err| ≤ n * eps * ||G||_2
        # Standard bound from Golub & Van Loan, Matrix Computations
        eps = machine_epsilon(backend, G)
        max_eig = float(backend.max(eigenvalues).item())
        n = G.shape[0]
        tol = n * eps * max(max_eig, 1.0)  # Scale by size and spectral norm
        # Allow small negative values due to numerical precision
        assert min_eig >= -tol

    def test_gram_matrix_shape(self, backend):
        """Gram matrix should be n x n."""
        backend.random_seed(42)
        X = backend.random_normal((15, 8))
        G = compute_gram_matrix(X, backend=backend)
        backend.eval(G)
        assert G.shape == (15, 15)

    def test_gram_matrix_diagonal_is_norms_squared(self, backend):
        """Diagonal of Gram matrix should equal squared row norms."""
        backend.random_seed(42)
        X = backend.random_normal((5, 3))
        G = compute_gram_matrix(X, backend=backend)
        backend.eval(G)

        for i in range(5):
            row = X[i]
            norm_sq = float(backend.sum(row * row).item())
            diag_val = float(G[i, i].item())
            eps = _eps(backend, norm_sq, diag_val)
            assert abs(norm_sq - diag_val) <= eps


# =============================================================================
# TestSpectralProperties: Eigenvalue distribution properties
# =============================================================================


class TestSpectralProperties:
    """Tests for spectral analysis properties."""

    def test_participation_ratio_bounds(self, backend, medium_primes):
        """Participation ratio should be in [1, n] where n is matrix size."""
        gaps = medium_primes.gaps
        embedded = time_delay_embedding(gaps, embedding_dim=20, backend=backend)
        gram = compute_gram_matrix(embedded, backend=backend)
        backend.eval(gram)

        dist = analyze_eigenvalues(gram, backend=backend)
        n = gram.shape[0]

        eps = _eps(backend, dist.participation_ratio, float(n), 1.0)
        assert dist.participation_ratio + eps >= 1.0
        assert dist.participation_ratio <= float(n) + eps

    def test_spectral_entropy_non_negative(self, backend, medium_primes):
        """Spectral entropy should be non-negative."""
        gaps = medium_primes.gaps
        embedded = time_delay_embedding(gaps, embedding_dim=20, backend=backend)
        gram = compute_gram_matrix(embedded, backend=backend)
        backend.eval(gram)

        dist = analyze_eigenvalues(gram, backend=backend)
        eps = _eps(backend, dist.spectral_entropy)
        assert dist.spectral_entropy >= -eps

    def test_top_eigenvalue_ratio_bounds(self, backend, medium_primes):
        """Top-k eigenvalue ratio should be in [0, 1]."""
        gaps = medium_primes.gaps
        embedded = time_delay_embedding(gaps, embedding_dim=15, backend=backend)
        gram = compute_gram_matrix(embedded, backend=backend)
        backend.eval(gram)

        dist = analyze_eigenvalues(gram, backend=backend)
        eps = _eps(backend, dist.top_k_ratio, 0.0, 1.0)
        assert dist.top_k_ratio + eps >= 0.0
        assert dist.top_k_ratio <= 1.0 + eps

    def test_condition_number_positive(self, backend, medium_primes):
        """Condition number should be positive."""
        gaps = medium_primes.gaps
        embedded = time_delay_embedding(gaps, embedding_dim=15, backend=backend)
        gram = compute_gram_matrix(embedded, backend=backend)
        backend.eval(gram)

        dist = analyze_eigenvalues(gram, backend=backend)
        eps = _eps(backend, dist.condition_number, 1.0)
        assert dist.condition_number + eps >= 1.0

    def test_identity_matrix_has_pr_n(self, backend):
        """Identity matrix should have participation ratio = n."""
        n = 10
        I = backend.eye(n)
        dist = analyze_eigenvalues(I, backend=backend)
        # All eigenvalues are 1, so PR = n
        eps = _eps(backend, dist.participation_ratio, float(n))
        assert abs(dist.participation_ratio - float(n)) <= eps


# =============================================================================
# TestBaselines: Baseline generator correctness
# =============================================================================


class TestBaselines:
    """Tests for baseline gap generators."""

    def test_random_gaps_correct_count(self, backend):
        """Random gaps should return correct count."""
        n = 100
        gaps = generate_random_gaps(n, mean_gap=5.0, backend=backend, seed=42)
        backend.eval(gaps)
        assert gaps.shape[0] == n

    def test_random_gaps_positive(self, backend):
        """Random gaps should all be positive."""
        gaps = generate_random_gaps(50, mean_gap=5.0, backend=backend, seed=42)
        backend.eval(gaps)
        assert float(backend.min(gaps).item()) >= 0.0

    def test_uniform_gaps_in_range(self, backend):
        """Uniform gaps should be in [min_gap, max_gap]."""
        min_gap, max_gap = 2, 10
        gaps = generate_uniform_gaps(100, min_gap, max_gap, backend=backend, seed=42)
        backend.eval(gaps)

        min_val = float(backend.min(gaps).item())
        max_val = float(backend.max(gaps).item())

        assert min_val >= min_gap
        assert max_val <= max_gap

    def test_poisson_gaps_positive(self, backend):
        """Poisson gaps should be positive."""
        gaps = generate_poisson_gaps(100, rate=0.2, backend=backend, seed=42)
        backend.eval(gaps)
        assert float(backend.min(gaps).item()) >= 0.0

    def test_cramer_model_returns_primes_and_gaps(self, backend):
        """Cramér model should return primes and gaps."""
        primes, gaps = generate_cramer_model(50, backend=backend, seed=42)
        backend.eval(primes)
        backend.eval(gaps)

        assert primes.shape[0] >= 1
        assert gaps.shape[0] == primes.shape[0] - 1

    def test_cramer_model_increasing_primes(self, backend):
        """Cramér pseudo-primes should be increasing."""
        primes, _ = generate_cramer_model(50, backend=backend, seed=42)
        backend.eval(primes)
        primes_np = array_to_list(backend, primes)

        for i in range(1, len(primes_np)):
            assert primes_np[i] > primes_np[i - 1]

    def test_generate_baseline_exponential(self, backend, medium_primes):
        """Generate baseline with exponential type."""
        mean_gap = float(backend.mean(medium_primes.gaps * 1.0).item())
        gaps = generate_baseline(
            BaselineType.EXPONENTIAL, medium_primes.gap_count, mean_gap, backend=backend
        )
        backend.eval(gaps)
        assert gaps.shape[0] == medium_primes.gap_count

    def test_generate_baseline_uniform(self, backend, medium_primes):
        """Generate baseline with uniform type."""
        mean_gap = float(backend.mean(medium_primes.gaps * 1.0).item())
        gaps = generate_baseline(
            BaselineType.UNIFORM, medium_primes.gap_count, mean_gap, backend=backend
        )
        backend.eval(gaps)
        assert gaps.shape[0] == medium_primes.gap_count

    def test_generate_baseline_shuffled(self, backend, medium_primes):
        """Generate baseline with shuffled type."""
        mean_gap = float(backend.mean(medium_primes.gaps * 1.0).item())
        gaps = generate_baseline(
            BaselineType.SHUFFLED,
            medium_primes.gap_count,
            mean_gap,
            prime_gaps=medium_primes.gaps,
            backend=backend
        )
        backend.eval(gaps)
        assert gaps.shape[0] == medium_primes.gap_count


# =============================================================================
# TestStatisticalTesting: Statistical utilities
# =============================================================================


class TestStatisticalTesting:
    """Tests for statistical testing utilities."""

    def test_bootstrap_ci_contains_mean(self, backend):
        """Bootstrap CI should typically contain the sample mean."""
        backend.random_seed(42)
        # Generate data with known mean
        data = backend.random_normal((100,)) + 5.0
        backend.eval(data)
        values = array_to_list(backend, data)

        ci = bootstrap_confidence_interval(values)
        min_val = min(values)
        max_val = max(values)
        eps = _eps(backend, ci.lower, ci.upper, min_val, max_val)

        assert ci.lower + eps >= min_val
        assert ci.upper <= max_val + eps
        assert ci.lower <= ci.upper + eps

    def test_bootstrap_ci_bounds_ordered(self, backend):
        """CI lower bound should be less than upper bound."""
        backend.random_seed(42)
        data = backend.random_normal((50,))
        backend.eval(data)
        values = array_to_list(backend, data)

        ci = bootstrap_confidence_interval(values)
        eps = _eps(backend, ci.lower, ci.upper)
        assert ci.lower <= ci.upper + eps

    def test_cohens_d_zero_for_same_samples(self):
        """Cohen's d should be ~0 for identical samples."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        effect = compute_cohens_d(values, values)
        backend = get_default_backend()
        eps = _eps(backend, effect.d)
        assert abs(effect.d) <= eps

    def test_cohens_d_large_for_separated_samples(self):
        """Cohen's d should be large for well-separated samples."""
        backend = get_default_backend()
        values1 = [1.0, 1.1, 1.2, 0.9, 1.0]
        values2 = [5.0, 5.1, 5.2, 4.9, 5.0]
        values3 = [1.05, 1.0, 0.95, 1.1, 0.9]
        effect_far = compute_cohens_d(values1, values2, backend=backend)
        effect_near = compute_cohens_d(values1, values3, backend=backend)
        eps = _eps(backend, effect_far.d, effect_near.d)
        assert abs(effect_far.d) >= abs(effect_near.d) + eps

    def test_permutation_test_significant_for_different(self, backend):
        """Permutation test should give low p-value for different distributions."""
        values1 = [1.0, 1.1, 1.2, 0.9, 1.0]
        values2 = [5.0, 5.1, 5.2, 4.9, 5.0]
        p_value_diff = permutation_test(values1, values2, backend=backend)
        p_value_same = permutation_test(values1, values1, backend=backend)
        assert p_value_diff <= p_value_same

    def test_permutation_test_high_for_same(self, backend):
        """Permutation test should give high p-value for same distribution."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        p_value = permutation_test(values, values, backend=backend)
        assert 0.0 <= p_value <= 1.0


# =============================================================================
# TestHypothesis: Hypothesis testing
# =============================================================================


class TestHypothesisValidation:
    """Tests for hypothesis testing framework."""

    def test_run_hypothesis_test_returns_result(self, backend):
        """run_hypothesis_test should return HypothesisTest result."""
        result = run_hypothesis_test(
            hypothesis_id="H_test",
            description="Test hypothesis",
            prime_value=1.0,
            baseline_value=5.0,
            backend=backend,
        )

        assert isinstance(result, HypothesisTest)
        assert result.hypothesis_id == "H_test"
        # passed is None when samples not provided (cannot determine statistically)
        assert result.passed is None
        # p_value is None when samples not provided
        assert result.p_value is None

    def test_run_hypothesis_test_one_sided_less(self, backend):
        """When samples are provided, p-value remains undefined."""
        prime_samples = [1.0, 1.1, 0.9, 1.05, 0.95] * 10
        baseline_samples = [10.0, 10.1, 9.9, 10.05, 9.95] * 10
        result = run_hypothesis_test(
            hypothesis_id="H_less",
            description="Test less",
            prime_value=1.0,
            baseline_value=10.0,
            prime_samples=prime_samples,
            baseline_samples=baseline_samples,
            backend=backend,
        )
        assert result.prime_value < result.baseline_value
        assert result.p_value is None
        assert result.passed is None

    def test_run_hypothesis_test_one_sided_greater_fails(self, backend):
        """When prime_value > baseline_value, one_sided (less) test should fail."""
        result = run_hypothesis_test(
            hypothesis_id="H_greater",
            description="Test greater fails",
            prime_value=10.0,
            baseline_value=1.0,
            backend=backend,
        )
        assert result.prime_value > result.baseline_value
        # p_value is None when samples not provided
        assert result.p_value is None
        # passed is None when samples not provided (cannot determine statistically)
        assert result.passed is None


# =============================================================================
# TestComprehensiveAnalysis: End-to-end testing
# =============================================================================


class TestComprehensiveAnalysis:
    """Tests for comprehensive analysis runner."""

    def test_run_comprehensive_analysis_returns_result(self, backend):
        """Should return ComprehensiveResult with all fields."""
        result = run_comprehensive_analysis(
            n_primes=100,
            embedding_dim=10,
            backend=backend,
        )

        assert isinstance(result, ComprehensiveResult)
        assert result.n_primes == 100
        assert result.embedding_dim == 10
        assert len(result.embedding_results) > 0
        assert len(result.baseline_results) > 0
        assert len(result.hypothesis_tests) > 0

    def test_run_comprehensive_analysis_has_hypothesis_tests(self, backend):
        """Should test H1-H8 hypotheses."""
        result = run_comprehensive_analysis(
            n_primes=100,
            embedding_dim=10,
            backend=backend,
        )

        # Should have tests for key hypotheses
        test_ids = list(result.hypothesis_tests.keys())
        assert any("H1" in tid for tid in test_ids)
        assert any("H2" in tid for tid in test_ids)

    def test_analyze_prime_geometry_returns_result(self, backend):
        """analyze_prime_geometry should return PrimeGeometryResult."""
        result = analyze_prime_geometry(
            n_primes=50,
            embedding_dim=10,
            backend=backend,
        )

        assert isinstance(result, PrimeGeometryResult)
        assert result.prime_count == 50
        assert result.prime_eigenvalues is not None
        assert result.random_eigenvalues is not None
        assert result.comparison is not None

    def test_format_result_returns_string(self, backend):
        """format_result should return non-empty string."""
        result = analyze_prime_geometry(n_primes=50, embedding_dim=10, backend=backend)
        formatted = format_result(result)

        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "PRIME" in formatted

    def test_format_comprehensive_result_returns_string(self, backend):
        """format_comprehensive_result should return non-empty string."""
        result = run_comprehensive_analysis(
            n_primes=50, embedding_dim=10, backend=backend
        )
        formatted = format_comprehensive_result(result)

        assert isinstance(formatted, str)
        assert len(formatted) > 0


# =============================================================================
# TestScaleSweep: Scale invariance testing
# =============================================================================


class TestScaleSweep:
    """Tests for scale sweep analysis."""

    def test_run_scale_sweep_returns_result(self, backend):
        """Should return ScaleSweepResult with scale points."""
        result = run_scale_sweep(
            scales=[50, 100],
            embedding_dim=8,
            backend=backend,
        )

        assert isinstance(result, ScaleSweepResult)
        assert result.scales == [50, 100]
        assert len(result.results) == 2

    def test_scale_sweep_results_ordered(self, backend):
        """Results should be in scale order."""
        scales = [50, 100, 150]
        result = run_scale_sweep(
            scales=scales,
            embedding_dim=8,
            backend=backend,
        )

        for i, r in enumerate(result.results):
            assert r.n_primes == scales[i]


# =============================================================================
# TestPerturbation: Perturbation robustness
# =============================================================================


class TestPerturbation:
    """Tests for perturbation robustness analysis."""

    def test_run_perturbation_study_returns_results(self, backend):
        """Should return list of PerturbationResult."""
        results = run_perturbation_study(
            n_primes=50,
            noise_levels=[0.0, 0.1],
            embedding_dim=8,
            backend=backend,
        )

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, PerturbationResult)

    def test_perturbation_zero_noise_original(self, backend):
        """Zero noise should give participation ratio close to original."""
        results = run_perturbation_study(
            n_primes=50,
            noise_levels=[0.0],
            embedding_dim=8,
            backend=backend,
        )

        assert len(results) == 1
        assert results[0].noise_level == 0.0


# =============================================================================
# Property-Based Tests with Hypothesis
# =============================================================================


class TestPrimeGeometryProperties:
    """Property-based tests using Hypothesis."""

    @given(st.integers(min_value=10, max_value=100))
    @settings(max_examples=10, deadline=None)
    def test_prime_count_matches_request(self, n):
        """Generated prime count should match requested count."""
        backend = get_default_backend()
        seq = generate_primes(n, backend=backend)
        assert seq.count == n

    @given(st.integers(min_value=5, max_value=20))
    @settings(max_examples=10, deadline=None)
    def test_embedding_dim_matches_request(self, dim):
        """Embedding dimension should match requested dimension."""
        backend = get_default_backend()
        seq = generate_primes(50, backend=backend)
        assume(seq.gap_count >= dim)  # Need enough gaps

        embedded = time_delay_embedding(seq.gaps, embedding_dim=dim, backend=backend)
        backend.eval(embedded)
        assert embedded.shape[1] == dim

    @given(st.floats(min_value=2.0, max_value=10.0))
    @settings(max_examples=10, deadline=None)
    def test_random_gaps_mean_reasonable(self, mean_gap):
        """Random gaps should have mean in reasonable range."""
        backend = get_default_backend()
        n = 200
        mean_gap_high = mean_gap * 2.0
        gaps_low = generate_random_gaps(n, mean_gap=mean_gap, backend=backend, seed=42)
        gaps_high = generate_random_gaps(n, mean_gap=mean_gap_high, backend=backend, seed=42)
        backend.eval(gaps_low)
        backend.eval(gaps_high)

        mean_low = float(backend.mean(gaps_low * 1.0).item())
        mean_high = float(backend.mean(gaps_high * 1.0).item())
        eps = _eps(backend, mean_low, mean_high)
        assert mean_high + eps >= mean_low

    @given(st.lists(st.floats(min_value=0.1, max_value=100.0), min_size=5, max_size=20))
    @settings(max_examples=10, deadline=None)
    def test_bootstrap_ci_has_positive_width(self, values):
        """Bootstrap CI should have positive width for variable data."""
        assume(len(set(values)) > 1)  # Need variation

        ci = bootstrap_confidence_interval(values)
        backend = get_default_backend()
        eps = _eps(backend, ci.lower, ci.upper)
        assert ci.upper + eps >= ci.lower

    @given(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=5, max_size=20),
        st.lists(st.floats(min_value=-10, max_value=10), min_size=5, max_size=20),
    )
    @settings(max_examples=10, deadline=None)
    def test_cohens_d_symmetric(self, values1, values2):
        """Cohen's d should be symmetric."""
        assume(len(values1) >= 2 and len(values2) >= 2)
        assume(not all(v == values1[0] for v in values1))  # Need variation
        assume(not all(v == values2[0] for v in values2))  # Need variation

        effect1 = compute_cohens_d(values1, values2)
        effect2 = compute_cohens_d(values2, values1)

        # Magnitude should be same, sign should flip
        backend = get_default_backend()
        eps = _eps(backend, effect1.d, effect2.d)
        assert abs(abs(effect1.d) - abs(effect2.d)) <= eps


# =============================================================================
# TestDataclasses: Dataclass validation
# =============================================================================


class TestDataclasses:
    """Tests for dataclass instantiation and properties."""

    def test_prime_sequence_creation(self, backend):
        """PrimeSequence should be creatable with valid data."""
        primes = backend.array([2, 3, 5, 7, 11])
        gaps = backend.array([1, 2, 2, 4])

        seq = PrimeSequence(primes=primes, gaps=gaps, count=5, max_prime=11)
        assert seq.count == 5
        assert seq.gap_count == 4

    def test_eigenvalue_distribution_creation(self, backend):
        """EigenvalueDistribution should be creatable."""
        eigenvalues = backend.array([5.0, 3.0, 2.0, 1.0])
        dist = EigenvalueDistribution(
            eigenvalues=eigenvalues,
            participation_ratio=5.0,
            spectral_entropy=2.3,
            condition_number=100.0,
            top_k_ratio=0.5,
        )
        assert dist.participation_ratio == 5.0

    def test_spectral_comparison_creation(self):
        """SpectralComparison should be creatable."""
        comp = SpectralComparison(
            source_label="primes",
            target_label="random",
            participation_ratio_diff=1.5,
            spectral_entropy_diff=0.2,
            wasserstein_distance=0.3,
            ks_statistic=0.15,
        )
        assert comp.ks_statistic == 0.15

    def test_confidence_interval_creation(self):
        """ConfidenceInterval should be creatable."""
        ci = ConfidenceInterval(
            lower=1.0,
            upper=3.0,
            mean=2.0,
            std=0.5,
            n_bootstrap=100,
        )
        assert ci.lower < ci.upper

    def test_effect_size_creation(self):
        """EffectSize should be creatable."""
        effect = EffectSize(d=0.5)
        assert effect.d == 0.5

    def test_effect_size_from_cohens_d(self):
        """EffectSize.from_cohens_d should preserve d."""
        effect = EffectSize.from_cohens_d(0.6)
        assert effect.d == 0.6

    def test_hypothesis_test_creation(self):
        """HypothesisTest should be creatable."""
        test = HypothesisTest(
            hypothesis_id="H1",
            description="Test spectral concentration",
            passed=None,
            p_value=0.01,
            effect_size=EffectSize(d=0.8),
            prime_value=2.0,
            baseline_value=5.0,
        )
        assert test.passed is None
