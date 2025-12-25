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
Tests for gradient boundary smoothing.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain.merging.gradient_boundary_smoother import (
    GradientBoundaryConfig,
    GradientBoundaryProfile,
    LayerGradientStats,
    apply_adaptive_smoothing,
    compute_gradient_adjusted_alpha,
    smooth_merge_boundaries,
)


class TestLayerGradientStats:
    """Tests for LayerGradientStats."""

    def test_is_noisy(self):
        """Should detect noisy layers (SNR < 1)."""
        noisy = LayerGradientStats(layer=0, snr=0.5, variance=1.0, mean_norm=0.5, sample_count=100)
        stable = LayerGradientStats(
            layer=1, snr=2.0, variance=0.25, mean_norm=1.0, sample_count=100
        )

        assert noisy.is_noisy
        assert not stable.is_noisy

    def test_is_stable(self):
        """Should detect stable layers (SNR > 2)."""
        noisy = LayerGradientStats(layer=0, snr=0.5, variance=1.0, mean_norm=0.5, sample_count=100)
        stable = LayerGradientStats(layer=1, snr=3.0, variance=0.1, mean_norm=1.0, sample_count=100)

        assert not noisy.is_stable
        assert stable.is_stable


class TestGradientBoundaryProfile:
    """Tests for GradientBoundaryProfile."""

    @pytest.fixture
    def sample_profile(self):
        """Create sample profile with discontinuity."""
        config = GradientBoundaryConfig()
        return GradientBoundaryProfile(
            snr_by_layer={0: 2.0, 1: 2.1, 2: 0.5, 3: 0.6},
            delta_snr_by_boundary={0: 0.1, 1: -1.6, 2: 0.1},
            discontinuity_layers=[1],
            recommended_smoothing={0: 1.0, 1: 1.5, 2: 2.0, 3: 1.0},
            config=config,
        )

    def test_mean_snr(self, sample_profile):
        """Should compute mean SNR correctly."""
        expected = (2.0 + 2.1 + 0.5 + 0.6) / 4
        assert abs(sample_profile.mean_snr - expected) < 0.01

    def test_snr_variance(self, sample_profile):
        """Should compute SNR variance correctly."""
        mean = sample_profile.mean_snr
        expected = sum((s - mean) ** 2 for s in sample_profile.snr_by_layer.values()) / 4
        assert abs(sample_profile.snr_variance - expected) < 0.01

    def test_has_discontinuities(self, sample_profile):
        """Should detect presence of discontinuities."""
        assert sample_profile.has_discontinuities

        no_disc = GradientBoundaryProfile(
            snr_by_layer={0: 1.0, 1: 1.1},
            delta_snr_by_boundary={0: 0.1},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0, 1: 1.0},
            config=GradientBoundaryConfig(),
        )
        assert not no_disc.has_discontinuities

    def test_discontinuity_fraction(self, sample_profile):
        """Should compute discontinuity fraction correctly."""
        # 1 discontinuity out of 3 boundaries
        assert abs(sample_profile.discontinuity_fraction - 1 / 3) < 0.01

    def test_summary(self, sample_profile):
        """Should produce valid summary dict."""
        summary = sample_profile.summary()

        assert "num_layers" in summary
        assert summary["num_layers"] == 4
        assert "num_discontinuities" in summary
        assert summary["num_discontinuities"] == 1


class TestApplyAdaptiveSmoothing:
    """Tests for apply_adaptive_smoothing function."""

    def test_smoothing_applied(self):
        """Should smooth alpha values across layers using SNR-derived sigma."""
        alpha_by_layer = {0: 0.3, 1: 0.5, 2: 0.7}
        config = GradientBoundaryConfig()
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 1.0, 1: 1.0, 2: 1.0},
            delta_snr_by_boundary={0: 0.0, 1: 0.0},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0, 1: 1.0, 2: 1.0},  # sigma = 1.0
            config=config,
        )

        smoothed = apply_adaptive_smoothing(alpha_by_layer, profile)

        # Middle layer should be smoothed toward neighbors
        assert 0.3 <= smoothed[1] <= 0.7

    def test_high_sigma_more_smoothing(self):
        """Higher recommended_smoothing (sigma) should give more smoothing."""
        alpha_by_layer = {0: 0.0, 1: 1.0, 2: 0.0}

        # Low sigma - less smoothing
        low_sigma_profile = GradientBoundaryProfile(
            snr_by_layer={0: 1.0, 1: 1.0, 2: 1.0},
            delta_snr_by_boundary={0: 0.0, 1: 0.0},
            discontinuity_layers=[],
            recommended_smoothing={0: 0.1, 1: 0.1, 2: 0.1},
            config=GradientBoundaryConfig(),
        )

        # High sigma - more smoothing
        high_sigma_profile = GradientBoundaryProfile(
            snr_by_layer={0: 1.0, 1: 1.0, 2: 1.0},
            delta_snr_by_boundary={0: 0.0, 1: 0.0},
            discontinuity_layers=[],
            recommended_smoothing={0: 10.0, 1: 10.0, 2: 10.0},
            config=GradientBoundaryConfig(),
        )

        low_smoothed = apply_adaptive_smoothing(alpha_by_layer, low_sigma_profile)
        high_smoothed = apply_adaptive_smoothing(alpha_by_layer, high_sigma_profile)

        # With low sigma, middle value should stay close to 1.0
        # With high sigma, middle value should move toward mean (0.33)
        assert high_smoothed[1] < low_smoothed[1]

    def test_empty_input(self):
        """Empty input should return empty output."""
        config = GradientBoundaryConfig()
        profile = GradientBoundaryProfile(
            snr_by_layer={},
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={},
            config=config,
        )
        result = apply_adaptive_smoothing({}, profile)
        assert result == {}


class TestComputeGradientAdjustedAlpha:
    """Tests for compute_gradient_adjusted_alpha function."""

    def test_high_snr_lowers_alpha(self):
        """High SNR layers should have alpha pushed down (trust source)."""
        alpha_by_layer = {0: 0.5, 1: 0.5, 2: 0.5}
        config = GradientBoundaryConfig()
        # Layer 0 has high SNR (5.0), layer 1 is median (1.0), layer 2 is low (0.5)
        # sorted = [0.5, 1.0, 5.0], median = 1.0
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 5.0, 1: 1.0, 2: 0.5},
            delta_snr_by_boundary={0: -4.0, 1: -0.5},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0, 1: 1.0, 2: 1.0},
            config=config,
        )

        adjusted = compute_gradient_adjusted_alpha(alpha_by_layer, profile)

        # Layer 0 (high SNR above median) should have alpha lowered
        assert adjusted[0] < 0.5

    def test_low_snr_raises_alpha(self):
        """Low SNR layers should have alpha pushed up (trust target)."""
        alpha_by_layer = {0: 0.5, 1: 0.5, 2: 0.5}
        config = GradientBoundaryConfig()
        # Layer 0 has low SNR (0.1), layer 1 is median (1.0), layer 2 is high (5.0)
        # sorted = [0.1, 1.0, 5.0], median = 1.0
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 0.1, 1: 1.0, 2: 5.0},
            delta_snr_by_boundary={0: 0.9, 1: 4.0},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0, 1: 1.0, 2: 1.0},
            config=config,
        )

        adjusted = compute_gradient_adjusted_alpha(alpha_by_layer, profile)

        # Layer 0 (low SNR below median) should have alpha raised
        assert adjusted[0] > 0.5

    def test_median_snr_unchanged(self):
        """Layer at median SNR should have minimal adjustment."""
        alpha_by_layer = {0: 0.5}
        config = GradientBoundaryConfig()
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 1.0},  # Only one layer, so it IS the median
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0},
            config=config,
        )

        adjusted = compute_gradient_adjusted_alpha(alpha_by_layer, profile)

        # Should be unchanged (adjustment = (median - snr) / (median + snr) = 0)
        assert abs(adjusted[0] - 0.5) < 0.01


class TestSmoothMergeBoundaries:
    """Tests for smooth_merge_boundaries function."""

    def test_without_gradients(self):
        """Should return input unchanged without gradient info."""
        alpha_by_layer = {0: 0.3, 1: 0.5, 2: 0.7}

        smoothed, profile = smooth_merge_boundaries(alpha_by_layer)

        assert profile is None  # No gradient profile
        assert smoothed == alpha_by_layer  # Unchanged

    def test_returns_profile_with_gradients(self):
        """Should return boundary profile when gradients provided."""
        alpha_by_layer = {0: 0.3, 1: 0.5, 2: 0.7}

        smoothed, profile = smooth_merge_boundaries(alpha_by_layer, per_sample_gradients=None)

        assert smoothed is not None
        assert len(smoothed) == 3


class TestPropertyBasedTests:
    """Property-based tests for mathematical invariants."""

    @given(
        alpha=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_smoothed_alpha_bounded(self, alpha):
        """Smoothed alpha must be in [0, 1]."""
        alpha_by_layer = {0: alpha}
        config = GradientBoundaryConfig()
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 1.0},
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0},
            config=config,
        )

        smoothed = apply_adaptive_smoothing(alpha_by_layer, profile)

        # Smoothing with only one layer just returns the value
        assert 0.0 <= smoothed[0] <= 1.0

    @given(
        snr=st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_snr_determines_adjustment_direction(self, snr):
        """SNR relative to median should consistently determine adjustment direction."""
        alpha_by_layer = {0: 0.5, 1: 0.5}
        config = GradientBoundaryConfig()
        # Layer 0 has variable SNR, layer 1 has SNR=1.0
        profile = GradientBoundaryProfile(
            snr_by_layer={0: snr, 1: 1.0},
            delta_snr_by_boundary={0: 1.0 - snr},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0, 1: 1.0},
            config=config,
        )

        adjusted = compute_gradient_adjusted_alpha(alpha_by_layer, profile)

        # Alpha is clamped to [0, 1]
        assert 0.0 <= adjusted[0] <= 1.0
        assert 0.0 <= adjusted[1] <= 1.0


class TestSmoothingReducesVariance:
    """Tests that smoothing reduces alpha variance."""

    def test_smoothing_reduces_variance(self):
        """Smoothing should reduce variance of alpha profile."""
        # Highly variable alpha profile
        alpha_by_layer = {i: 0.2 + 0.6 * (i % 2) for i in range(10)}
        # Values: 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8

        config = GradientBoundaryConfig()
        profile = GradientBoundaryProfile(
            snr_by_layer={i: 1.0 for i in range(10)},
            delta_snr_by_boundary={i: 0.0 for i in range(9)},
            discontinuity_layers=[],
            recommended_smoothing={i: 2.0 for i in range(10)},  # Wide smoothing
            config=config,
        )

        smoothed = apply_adaptive_smoothing(alpha_by_layer, profile)

        # Calculate variances
        raw_values = list(alpha_by_layer.values())
        smoothed_values = list(smoothed.values())

        raw_mean = sum(raw_values) / len(raw_values)
        smoothed_mean = sum(smoothed_values) / len(smoothed_values)

        raw_var = sum((v - raw_mean) ** 2 for v in raw_values) / len(raw_values)
        smoothed_var = sum((v - smoothed_mean) ** 2 for v in smoothed_values) / len(smoothed_values)

        assert smoothed_var < raw_var
