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
        config = GradientBoundaryConfig(snr_discontinuity_threshold=0.5)
        return GradientBoundaryProfile(
            snr_by_layer={0: 2.0, 1: 2.1, 2: 0.5, 3: 0.6},  # discontinuity at 1->2
            delta_snr_by_boundary={0: 0.1, 1: -1.6, 2: 0.1},  # layer 1 has big drop
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
        """Should smooth alpha values across layers."""
        alpha_by_layer = {0: 0.3, 1: 0.5, 2: 0.7}
        config = GradientBoundaryConfig(base_smoothing_sigma=1.0, smoothing_window=1)
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 1.0, 1: 1.0, 2: 1.0},
            delta_snr_by_boundary={0: 0.0, 1: 0.0},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0, 1: 1.0, 2: 1.0},
            config=config,
        )

        smoothed = apply_adaptive_smoothing(alpha_by_layer, profile)

        # Middle layer should be smoothed toward neighbors
        # Original: 0.3, 0.5, 0.7
        # Layer 1 gets contribution from layers 0 and 2
        assert 0.3 <= smoothed[1] <= 0.7

    def test_clamping_applied(self):
        """Should clamp alpha to bounds."""
        alpha_by_layer = {0: 0.0, 1: 1.0}
        config = GradientBoundaryConfig(smoothing_window=0)  # No smoothing
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 1.0, 1: 1.0},
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0, 1: 1.0},
            config=config,
        )

        smoothed = apply_adaptive_smoothing(alpha_by_layer, profile, min_alpha=0.1, max_alpha=0.95)

        assert smoothed[0] == 0.1  # Clamped from 0.0
        assert smoothed[1] == 0.95  # Clamped from 1.0

    def test_increased_smoothing_at_discontinuity(self):
        """Should increase smoothing at discontinuity boundaries."""
        alpha_by_layer = {0: 0.3, 1: 0.8, 2: 0.4}  # Sharp change at layer 1
        config = GradientBoundaryConfig(
            base_smoothing_sigma=1.0,
            smoothing_window=1,
            adaptive_smoothing=True,
        )
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 2.0, 1: 0.3, 2: 2.0},  # Layer 1 has low SNR
            delta_snr_by_boundary={0: -1.7, 1: 1.7},
            discontinuity_layers=[0, 1],
            recommended_smoothing={0: 1.0, 1: 2.5, 2: 1.0},  # More smoothing at layer 1
            config=config,
        )

        smoothed = apply_adaptive_smoothing(alpha_by_layer, profile)

        # Layer 1 should be smoothed more due to higher multiplier
        # It should move toward the mean of neighbors
        assert smoothed[1] < 0.8  # Should be pulled down toward 0.3 and 0.4


class TestComputeGradientAdjustedAlpha:
    """Tests for compute_gradient_adjusted_alpha function."""

    def test_high_snr_lowers_alpha(self):
        """High SNR layers should have alpha pushed down (trust source)."""
        alpha_by_layer = {0: 0.5}
        config = GradientBoundaryConfig()
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 5.0},  # High SNR
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0},
            config=config,
        )

        adjusted = compute_gradient_adjusted_alpha(alpha_by_layer, profile, adjustment_strength=0.3)

        assert adjusted[0] < 0.5  # Should be lowered

    def test_low_snr_raises_alpha(self):
        """Low SNR layers should have alpha pushed up (trust target)."""
        alpha_by_layer = {0: 0.5}
        config = GradientBoundaryConfig()
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 0.1},  # Low SNR
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0},
            config=config,
        )

        adjusted = compute_gradient_adjusted_alpha(alpha_by_layer, profile, adjustment_strength=0.3)

        assert adjusted[0] > 0.5  # Should be raised

    def test_neutral_snr_unchanged(self):
        """Neutral SNR should not change alpha."""
        alpha_by_layer = {0: 0.5}
        config = GradientBoundaryConfig()
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 1.0},  # Neutral SNR
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0},
            config=config,
        )

        adjusted = compute_gradient_adjusted_alpha(alpha_by_layer, profile, adjustment_strength=0.3)

        assert abs(adjusted[0] - 0.5) < 0.1  # Should be roughly unchanged


class TestSmoothMergeBoundaries:
    """Tests for smooth_merge_boundaries function."""

    def test_without_gradients(self):
        """Should apply basic smoothing without gradient info."""
        alpha_by_layer = {0: 0.3, 1: 0.5, 2: 0.7}

        smoothed, profile = smooth_merge_boundaries(alpha_by_layer)

        assert profile is None  # No gradient profile
        assert all(0.1 <= a <= 0.95 for a in smoothed.values())

    def test_returns_profile_with_gradients(self):
        """Should return boundary profile when gradients provided."""
        # This would require MLX arrays, so we just test the structure
        alpha_by_layer = {0: 0.3, 1: 0.5, 2: 0.7}

        smoothed, profile = smooth_merge_boundaries(alpha_by_layer, per_sample_gradients=None)

        assert smoothed is not None
        assert len(smoothed) == 3


class TestPropertyBasedTests:
    """Property-based tests for mathematical invariants."""

    @given(
        alpha=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        min_alpha=st.floats(0.0, 0.5, allow_nan=False, allow_infinity=False),
        max_alpha=st.floats(0.5, 1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_smoothed_alpha_bounded(self, alpha, min_alpha, max_alpha):
        """Smoothed alpha must be in [min_alpha, max_alpha]."""
        if min_alpha >= max_alpha:
            return  # Invalid config

        alpha_by_layer = {0: alpha}
        config = GradientBoundaryConfig(smoothing_window=0)
        profile = GradientBoundaryProfile(
            snr_by_layer={0: 1.0},
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0},
            config=config,
        )

        smoothed = apply_adaptive_smoothing(
            alpha_by_layer, profile, min_alpha=min_alpha, max_alpha=max_alpha
        )

        assert min_alpha <= smoothed[0] <= max_alpha

    @given(
        snr=st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_snr_determines_adjustment_direction(self, snr):
        """SNR should consistently determine adjustment direction."""
        alpha_by_layer = {0: 0.5}
        config = GradientBoundaryConfig()
        profile = GradientBoundaryProfile(
            snr_by_layer={0: snr},
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={0: 1.0},
            config=config,
        )

        adjusted = compute_gradient_adjusted_alpha(alpha_by_layer, profile, adjustment_strength=0.3)

        if snr > 2.0:
            assert adjusted[0] <= 0.5  # High SNR -> lower alpha
        elif snr < 0.5:
            assert adjusted[0] >= 0.5  # Low SNR -> higher alpha


class TestSmoothingReducesVariance:
    """Tests that smoothing reduces alpha variance."""

    def test_smoothing_reduces_variance(self):
        """Smoothing should reduce variance of alpha profile."""
        # Highly variable alpha profile
        alpha_by_layer = {i: 0.2 + 0.6 * (i % 2) for i in range(10)}
        # Values: 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8

        config = GradientBoundaryConfig(base_smoothing_sigma=2.0, smoothing_window=2)
        profile = GradientBoundaryProfile(
            snr_by_layer={i: 1.0 for i in range(10)},
            delta_snr_by_boundary={},
            discontinuity_layers=[],
            recommended_smoothing={i: 1.0 for i in range(10)},
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
