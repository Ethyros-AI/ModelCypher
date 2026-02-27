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

"""Tests for core.domain.statistics — two-group hypothesis testing and curve fitting."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.statistics import (
    binomial_degradation_is_significant,
    cohens_d_bootstrap_ci,
    cohens_d_two_groups,
    confidence_intervals_overlap,
    fit_exponential,
    fit_inverse,
    fit_linear,
    levene_test_statistic,
    mean_trajectory,
    permutation_test_p_value,
    safe_mean,
    safe_std,
    spearman_correlation,
)

# ---------------------------------------------------------------------------
# confidence_intervals_overlap / binomial_degradation_is_significant
# ---------------------------------------------------------------------------


class TestBinomialDegradationSignificance:
    def test_confidence_intervals_overlap_true(self):
        assert confidence_intervals_overlap((0.6, 0.9), (0.8, 0.95)) is True

    def test_confidence_intervals_overlap_false(self):
        assert confidence_intervals_overlap((0.1, 0.2), (0.3, 0.4)) is False

    def test_confidence_intervals_overlap_invalid_interval(self):
        with pytest.raises(ValueError, match="interval_a"):
            confidence_intervals_overlap((0.4, 0.3), (0.1, 0.2))

    def test_degradation_not_significant_for_24_to_21_of_25(self):
        significant, current_ci, baseline_ci = binomial_degradation_is_significant(
            baseline_n_correct=24,
            current_n_correct=21,
            n_total=25,
            alpha=1.0 / 25.0,
        )
        assert significant is False
        assert confidence_intervals_overlap(current_ci, baseline_ci) is True

    def test_degradation_significant_for_24_to_10_of_25(self):
        significant, current_ci, baseline_ci = binomial_degradation_is_significant(
            baseline_n_correct=24,
            current_n_correct=10,
            n_total=25,
            alpha=1.0 / 25.0,
        )
        assert significant is True
        assert confidence_intervals_overlap(current_ci, baseline_ci) is False


# ---------------------------------------------------------------------------
# safe_mean / safe_std
# ---------------------------------------------------------------------------


class TestSafeMean:
    def test_normal_values(self):
        assert safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_nan_filtering(self):
        assert safe_mean([1.0, float("nan"), 3.0]) == pytest.approx(2.0)

    def test_all_nan(self):
        assert math.isnan(safe_mean([float("nan"), float("nan")]))

    def test_empty(self):
        assert math.isnan(safe_mean([]))


class TestSafeStd:
    def test_known_std(self):
        # Population std of [1, 3] = 1.0
        assert safe_std([1.0, 3.0]) == pytest.approx(1.0)

    def test_single_value(self):
        assert safe_std([5.0]) == 0.0

    def test_nan_filtering(self):
        result = safe_std([1.0, float("nan"), 3.0])
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mean_trajectory
# ---------------------------------------------------------------------------


class TestMeanTrajectory:
    def test_element_wise(self):
        t1 = [1.0, 2.0, 3.0]
        t2 = [3.0, 4.0, 5.0]
        result = mean_trajectory([t1, t2])
        assert result == pytest.approx([2.0, 3.0, 4.0])

    def test_ragged(self):
        t1 = [1.0, 2.0]
        t2 = [3.0, 4.0, 5.0]
        result = mean_trajectory([t1, t2])
        assert len(result) == 3
        assert result[0] == pytest.approx(2.0)
        assert result[2] == pytest.approx(5.0)

    def test_empty(self):
        assert mean_trajectory([]) == []
        assert mean_trajectory([[]]) == []


# ---------------------------------------------------------------------------
# cohens_d_two_groups
# ---------------------------------------------------------------------------


class TestCohensD:
    def test_known_effect(self):
        # Two groups with means 0 and 1, unit variance → d ≈ 1.0
        g1 = [0.0] * 50
        g2 = [1.0] * 50
        # pooled_var = 0 here, so use groups with actual variance
        g1 = [float(i) / 50.0 for i in range(50)]  # mean ≈ 0.49
        g2 = [float(i) / 50.0 + 2.0 for i in range(50)]  # mean ≈ 2.49
        d = cohens_d_two_groups(g1, g2)
        assert d < 0  # g1 mean < g2 mean
        assert abs(d) > 1.0  # large effect

    def test_identical_groups(self):
        g = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cohens_d_two_groups(g, g) == pytest.approx(0.0)

    def test_zero_variance(self):
        g1 = [5.0, 5.0, 5.0]
        g2 = [5.0, 5.0, 5.0]
        assert cohens_d_two_groups(g1, g2) == 0.0

    def test_too_few_samples(self):
        assert cohens_d_two_groups([1.0], [2.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# cohens_d_bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        g1 = [float(i) for i in range(20)]
        g2 = [float(i) + 5.0 for i in range(20)]
        d = cohens_d_two_groups(g1, g2)
        lo, hi = cohens_d_bootstrap_ci(g1, g2, seed=42)
        assert lo <= d <= hi

    def test_deterministic_with_seed(self):
        g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        g2 = [3.0, 4.0, 5.0, 6.0, 7.0]
        ci1 = cohens_d_bootstrap_ci(g1, g2, seed=123)
        ci2 = cohens_d_bootstrap_ci(g1, g2, seed=123)
        assert ci1 == ci2

    def test_too_few_samples(self):
        assert cohens_d_bootstrap_ci([1.0], [2.0]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# permutation_test_p_value
# ---------------------------------------------------------------------------


class TestPermutationTest:
    def test_identical_groups(self):
        g = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = permutation_test_p_value(g, g.copy(), seed=42)
        assert p > 0.5  # no difference → high p

    def test_separated_groups(self):
        g1 = [0.0, 0.1, 0.2, 0.3, 0.4]
        g2 = [100.0, 100.1, 100.2, 100.3, 100.4]
        p = permutation_test_p_value(g1, g2, seed=42)
        assert p < 0.05  # clearly different

    def test_empty_group(self):
        assert permutation_test_p_value([], [1.0, 2.0]) == 1.0


# ---------------------------------------------------------------------------
# levene_test_statistic
# ---------------------------------------------------------------------------


class TestLevene:
    def test_equal_variance(self):
        g1 = [float(i) for i in range(20)]
        g2 = [float(i) + 100.0 for i in range(20)]
        _, p = levene_test_statistic([g1, g2])
        assert p > 0.3  # same variance, different means

    def test_different_variance(self):
        g1 = [float(i) * 0.1 for i in range(30)]  # low variance
        g2 = [float(i) * 10.0 for i in range(30)]  # high variance
        F, p = levene_test_statistic([g1, g2])
        assert F > 1.0
        assert p < 0.05

    def test_single_group(self):
        assert levene_test_statistic([[1.0, 2.0]]) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# spearman_correlation
# ---------------------------------------------------------------------------


class TestSpearman:
    def test_perfect_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert spearman_correlation(x, y) == pytest.approx(1.0)

    def test_anticorrelation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        assert spearman_correlation(x, y) == pytest.approx(-1.0)

    def test_too_few_points(self):
        assert spearman_correlation([1.0, 2.0], [3.0, 4.0]) == 0.0


# ---------------------------------------------------------------------------
# fit_linear
# ---------------------------------------------------------------------------


class TestFitLinear:
    def test_perfect_line(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0, 3.0, 5.0, 7.0, 9.0]  # y = 2x + 1
        alpha, beta, r2 = fit_linear(x, y)
        assert alpha == pytest.approx(2.0)
        assert beta == pytest.approx(1.0)
        assert r2 == pytest.approx(1.0)

    def test_constant_x(self):
        x = [1.0, 1.0, 1.0]
        y = [2.0, 3.0, 4.0]
        alpha, beta, r2 = fit_linear(x, y)
        assert alpha == 0.0
        assert r2 == 0.0

    def test_too_few(self):
        assert fit_linear([1.0], [2.0]) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# fit_exponential
# ---------------------------------------------------------------------------


class TestFitExponential:
    def test_known_decay(self):
        # y = 10 * exp(-0.5 * x)
        x = [float(i) for i in range(10)]
        y = [10.0 * math.exp(-0.5 * xi) for xi in x]
        a, b, r2 = fit_exponential(x, y)
        assert a == pytest.approx(10.0, rel=0.01)
        assert b == pytest.approx(0.5, rel=0.01)
        assert r2 > 0.99

    def test_no_positive_y(self):
        assert fit_exponential([1.0, 2.0], [0.0, -1.0]) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# fit_inverse
# ---------------------------------------------------------------------------


class TestFitInverse:
    def test_known_inverse(self):
        # y = 6/x + 2
        x = [1.0, 2.0, 3.0, 6.0]
        y = [8.0, 5.0, 4.0, 3.0]
        alpha, beta, r2 = fit_inverse(x, y)
        assert alpha == pytest.approx(6.0, rel=0.01)
        assert beta == pytest.approx(2.0, rel=0.01)
        assert r2 > 0.99

    def test_zero_x_filtered(self):
        # Should not crash with x=0
        x = [0.0, 1.0, 2.0]
        y = [999.0, 3.0, 1.5]
        alpha, beta, r2 = fit_inverse(x, y)
        # Only 2 points used, still returns something
        assert isinstance(alpha, float)
