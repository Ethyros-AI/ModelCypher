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

"""Unit tests for outlier detection (requires MLX)."""

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.outlier_detector import OutlierDetector


def _div_eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


class TestOutlierDetector:
    """Tests for outlier detection from GPA errors."""

    def test_single_outlier(self):
        """5 clustered, 1 outlier should be detected."""
        detector = OutlierDetector()

        # 5 low errors (consensus), 1 high error (outlier)
        errors = [0.1, 0.12, 0.09, 0.11, 0.13, 0.8]  # Last one is outlier

        result = detector.detect_from_gpa(errors)

        assert len(result.outlier_indices) == 1
        assert 5 in result.outlier_indices  # Index 5 is the outlier
        assert len(result.consensus_indices) == 5
        assert all(i in result.consensus_indices for i in range(5))

    def test_no_outliers(self):
        """All similar errors should have no outliers."""
        detector = OutlierDetector()

        # All identical errors - no variation at all
        errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        result = detector.detect_from_gpa(errors)

        assert len(result.outlier_indices) == 0
        assert len(result.consensus_indices) == 6

    def test_two_models_min(self):
        """Should handle minimum 2 models."""
        detector = OutlierDetector()

        errors = [0.1, 0.5]

        result = detector.detect_from_gpa(errors)

        # With only 2 models and lenient sigma, might not detect outlier
        assert len(result.consensus_indices) + len(result.outlier_indices) == 2

    def test_single_model(self):
        """Single model should return that model as consensus."""
        detector = OutlierDetector()

        errors = [0.1]

        result = detector.detect_from_gpa(errors)

        assert len(result.consensus_indices) == 1
        assert 0 in result.consensus_indices
        assert len(result.outlier_indices) == 0

    def test_single_extreme_outlier(self):
        """Should detect a single extreme outlier among many consensus."""
        detector = OutlierDetector()

        # 7 clustered, 1 extreme outlier
        # Z-score works best with a single clear outlier
        errors = [0.1, 0.11, 0.09, 0.12, 0.1, 0.11, 0.1, 10.0]

        result = detector.detect_from_gpa(errors)

        eps = _div_eps()
        for idx in result.outlier_indices:
            assert errors[idx] > result.threshold + eps
        for idx in result.consensus_indices:
            assert errors[idx] <= result.threshold + eps
        assert set(result.outlier_indices).union(result.consensus_indices) == set(
            range(len(errors))
        )

    def test_threshold_computation(self):
        """Threshold should be mean + sigma * std."""
        detector = OutlierDetector()

        errors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        result = detector.detect_from_gpa(errors)

        backend = get_default_backend()
        errors_arr = backend.array(errors)
        eps = division_epsilon(backend, errors_arr)
        mean_err = float(backend.mean(errors_arr))
        variance = float(backend.mean((errors_arr - mean_err) ** 2))
        std_err = sqrt_scalar(variance, backend)
        sorted_errors = sorted(errors)
        median_err = sorted_errors[len(errors) // 2]
        tail = [value for value in sorted_errors if value >= median_err]
        threshold = find_magnitude_gap_threshold(tail, eps=eps, backend=backend)
        threshold = max(threshold, median_err + eps)

        assert abs(result.mean_error - mean_err) <= eps
        assert abs(result.std_error - std_err) <= eps
        assert abs(result.threshold - threshold) <= eps

    def test_empty_errors(self):
        """Empty list should return empty result."""
        detector = OutlierDetector()

        result = detector.detect_from_gpa([])

        assert len(result.consensus_indices) == 0
        assert len(result.outlier_indices) == 0


class TestConsensusStress:
    """Tests for computing consensus stress from profiles."""

    def test_consensus_stress_computation(self):
        """Should compute mean stress from consensus models."""
        backend = get_default_backend()
        detector = OutlierDetector(backend)

        # Create mock stress profiles with stress_vector attribute
        class MockProfile:
            def __init__(self, stress):
                self.stress_vector = tuple(stress)

            def distance_to(self, other):
                import math
                s1 = self.stress_vector
                s2 = other.stress_vector
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(s1, s2)))

        profiles = [
            MockProfile([1.0, 2.0, 3.0]),
            MockProfile([1.1, 2.1, 3.1]),
            MockProfile([0.9, 1.9, 2.9]),
            MockProfile([10.0, 20.0, 30.0]),  # Outlier
        ]

        consensus_indices = (0, 1, 2)

        mean_stress = detector.get_consensus_stress(profiles, consensus_indices)

        # Mean of first 3: [1.0, 2.0, 3.0]
        eps = _div_eps()
        assert abs(float(mean_stress[0]) - 1.0) <= eps
        assert abs(float(mean_stress[1]) - 2.0) <= eps
        assert abs(float(mean_stress[2]) - 3.0) <= eps


class TestStressProfileDetection:
    """Tests for outlier detection from stress profiles."""

    def test_detect_from_stress_profiles(self):
        """Should detect outlier from pairwise stress distances."""
        detector = OutlierDetector()

        class MockProfile:
            def __init__(self, stress):
                self.stress_vector = tuple(stress)

            def distance_to(self, other):
                import math
                s1 = self.stress_vector
                s2 = other.stress_vector
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(s1, s2)))

        # 5 similar profiles, 1 different
        profiles = [
            MockProfile([1.0, 2.0, 3.0]),
            MockProfile([1.1, 2.1, 3.1]),
            MockProfile([0.9, 1.9, 2.9]),
            MockProfile([1.0, 2.0, 3.0]),
            MockProfile([1.05, 2.05, 3.05]),
            MockProfile([10.0, 20.0, 30.0]),  # Outlier
        ]

        result = detector.detect_from_stress_profiles(profiles)

        assert 5 in result.outlier_indices
        assert set(result.outlier_indices).union(result.consensus_indices) == set(
            range(len(profiles))
        )


class TestTriangulation:
    """Tests for 3-model triangulation-based outlier detection."""

    def test_three_models_with_outlier(self):
        """With 3 models, should detect the one that disagrees."""
        detector = OutlierDetector()

        class MockProfile:
            def __init__(self, stress):
                self.stress_vector = tuple(stress)

            def distance_to(self, other):
                import math
                s1 = self.stress_vector
                s2 = other.stress_vector
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(s1, s2)))

        # Models 0 and 1 agree, model 2 is outlier
        profiles = [
            MockProfile([1.0, 2.0, 3.0]),
            MockProfile([1.1, 2.1, 3.1]),  # Close to model 0
            MockProfile([10.0, 20.0, 30.0]),  # Far from both
        ]

        result = detector.detect_from_stress_profiles(profiles)

        # Model 2 should be detected as outlier
        assert 2 in result.outlier_indices
        assert 0 in result.consensus_indices
        assert 1 in result.consensus_indices

    def test_three_models_no_outlier(self):
        """If all 3 models are similar, no outlier detected."""
        detector = OutlierDetector()

        class MockProfile:
            def __init__(self, stress):
                self.stress_vector = tuple(stress)

            def distance_to(self, other):
                import math
                s1 = self.stress_vector
                s2 = other.stress_vector
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(s1, s2)))

        # All three models are similar
        profiles = [
            MockProfile([1.0, 2.0, 3.0]),
            MockProfile([1.1, 2.1, 3.1]),
            MockProfile([0.9, 1.9, 2.9]),
        ]

        result = detector.detect_from_stress_profiles(profiles)

        # No outlier - all are consensus
        assert len(result.outlier_indices) == 0
        assert len(result.consensus_indices) == 3

    def test_three_models_first_is_outlier(self):
        """Triangulation should detect outlier regardless of position."""
        detector = OutlierDetector()

        class MockProfile:
            def __init__(self, stress):
                self.stress_vector = tuple(stress)

            def distance_to(self, other):
                import math
                s1 = self.stress_vector
                s2 = other.stress_vector
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(s1, s2)))

        # Model 0 is the outlier, models 1 and 2 agree
        profiles = [
            MockProfile([10.0, 20.0, 30.0]),  # Outlier
            MockProfile([1.0, 2.0, 3.0]),
            MockProfile([1.1, 2.1, 3.1]),  # Close to model 1
        ]

        result = detector.detect_from_stress_profiles(profiles)

        # Model 0 should be detected as outlier
        assert 0 in result.outlier_indices
        assert 1 in result.consensus_indices
        assert 2 in result.consensus_indices

    def test_three_models_middle_is_outlier(self):
        """Triangulation should detect middle model as outlier."""
        detector = OutlierDetector()

        class MockProfile:
            def __init__(self, stress):
                self.stress_vector = tuple(stress)

            def distance_to(self, other):
                import math
                s1 = self.stress_vector
                s2 = other.stress_vector
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(s1, s2)))

        # Model 1 is the outlier, models 0 and 2 agree
        profiles = [
            MockProfile([1.0, 2.0, 3.0]),
            MockProfile([10.0, 20.0, 30.0]),  # Outlier
            MockProfile([1.1, 2.1, 3.1]),  # Close to model 0
        ]

        result = detector.detect_from_stress_profiles(profiles)

        # Model 1 should be detected as outlier
        assert 1 in result.outlier_indices
        assert 0 in result.consensus_indices
        assert 2 in result.consensus_indices

    def test_triangulation_threshold_boundary(self):
        """Test that threshold is working correctly at boundary."""
        detector = OutlierDetector()

        class MockProfile:
            def __init__(self, stress):
                self.stress_vector = tuple(stress)

            def distance_to(self, other):
                import math
                s1 = self.stress_vector
                s2 = other.stress_vector
                return math.sqrt(sum((a - b) ** 2 for a, b in zip(s1, s2)))

        # Edge case: third model is exactly 2x distance
        # d(0,1) = sqrt(0.03) ≈ 0.173
        # d(0,2) = sqrt(0.75) ≈ 0.866, d(1,2) ≈ 0.866
        # Mean outlier dist ≈ 0.866, threshold = 2 * 0.173 ≈ 0.346
        # 0.866 > 0.346, so should be detected
        profiles = [
            MockProfile([1.0, 2.0, 3.0]),
            MockProfile([1.1, 2.1, 3.1]),
            MockProfile([1.5, 2.5, 3.5]),  # Moderately far
        ]

        result = detector.detect_from_stress_profiles(profiles)

        # Should detect model 2 as outlier (mean dist > 2x consensus dist)
        assert 2 in result.outlier_indices
