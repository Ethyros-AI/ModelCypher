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

"""Tests for metaphor invariance analysis."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.metaphor_invariance import (
    BatchInvarianceResult,
    MetaphorInvarianceAnalyzer,
    MetaphorInvarianceResult,
    PlatonicMetaphorValidator,
    batch_result_to_dict,
    invariance_result_to_dict,
)
from modelcypher.core.domain.geometry.metaphor_trajectory import (
    MetaphorTrajectory,
    MetaphorTrajectoryPoint,
)


def make_trajectory(
    metaphor_id: str,
    model_id: str,
    cka_values: list[float],
) -> MetaphorTrajectory:
    """Create a trajectory with given CKA values."""
    points = [
        MetaphorTrajectoryPoint(
            layer_index=i,
            cka_source_target=cka,
            cosine_similarity=cka * 0.9,
            source_centroid_norm=1.0,
            target_centroid_norm=1.0,
        )
        for i, cka in enumerate(cka_values)
    ]
    return MetaphorTrajectory(
        metaphor_id=metaphor_id,
        metaphor_name=metaphor_id.upper(),
        source_domain="SOURCE",
        target_domain="TARGET",
        model_id=model_id,
        points=tuple(points),
    )


class TestMetaphorInvarianceAnalyzer:
    """Tests for MetaphorInvarianceAnalyzer."""

    def test_identical_trajectories_have_high_cka(self):
        """Identical trajectories should have CKA near 1.0."""
        backend = get_default_backend()
        analyzer = MetaphorInvarianceAnalyzer(backend)

        cka_values = [0.1, 0.3, 0.6, 0.9, 0.7]
        traj_a = make_trajectory("test", "model_a", cka_values)
        traj_b = make_trajectory("test", "model_b", cka_values)

        result = analyzer.compare_metaphor_geometry(traj_a, traj_b)

        assert result.trajectory_cka > 0.99
        assert result.convergence_layer_delta_normalized == 0.0
        assert result.peak_cka_delta == 0.0

    def test_different_trajectories_have_lower_cka(self):
        """Different trajectories should have measurable differences."""
        backend = get_default_backend()
        analyzer = MetaphorInvarianceAnalyzer(backend)

        # One trajectory peaks early, the other peaks late
        traj_a = make_trajectory("test", "model_a", [0.9, 0.7, 0.3, 0.2, 0.1])  # Early peak
        traj_b = make_trajectory("test", "model_b", [0.1, 0.2, 0.5, 0.8, 0.9])  # Late peak

        result = analyzer.compare_metaphor_geometry(traj_a, traj_b)

        # Convergence layers should be different
        assert result.convergence_layer_a == 0  # Peak at layer 0
        assert result.convergence_layer_b == 4  # Peak at layer 4
        # Peak CKA delta should be small (both peak at ~0.9)
        assert result.peak_cka_delta < 0.05
        # But convergence layer delta should be large
        assert result.convergence_layer_delta_normalized > 0.5

    def test_different_metaphor_ids_raise_error(self):
        """Should raise error when comparing different metaphors."""
        backend = get_default_backend()
        analyzer = MetaphorInvarianceAnalyzer(backend)

        traj_a = make_trajectory("metaphor_a", "model", [0.5])
        traj_b = make_trajectory("metaphor_b", "model", [0.5])

        with pytest.raises(ValueError, match="different metaphors"):
            analyzer.compare_metaphor_geometry(traj_a, traj_b)

    def test_different_length_trajectories(self):
        """Should handle trajectories of different lengths via interpolation."""
        backend = get_default_backend()
        analyzer = MetaphorInvarianceAnalyzer(backend)

        # 5-layer model
        traj_a = make_trajectory("test", "model_a", [0.1, 0.3, 0.5, 0.7, 0.9])
        # 10-layer model with same pattern
        traj_b = make_trajectory(
            "test", "model_b",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
        )

        result = analyzer.compare_metaphor_geometry(traj_a, traj_b)

        # Same pattern should have high CKA even with different lengths
        assert result.trajectory_cka > 0.8

    def test_convergence_layer_delta(self):
        """Should compute normalized convergence layer delta."""
        backend = get_default_backend()
        analyzer = MetaphorInvarianceAnalyzer(backend)

        # Model A peaks at layer 3
        traj_a = make_trajectory("test", "model_a", [0.1, 0.3, 0.5, 0.9, 0.7])
        # Model B peaks at layer 1
        traj_b = make_trajectory("test", "model_b", [0.1, 0.9, 0.5, 0.3, 0.2])

        result = analyzer.compare_metaphor_geometry(traj_a, traj_b)

        assert result.convergence_layer_a == 3
        assert result.convergence_layer_b == 1
        # Delta = |3 - 1| / 5 = 0.4
        assert abs(result.convergence_layer_delta_normalized - 0.4) < 0.01


class TestBatchInvariance:
    """Tests for batch invariance testing."""

    def test_batch_invariance(self):
        """Should compute aggregate statistics across model pairs."""
        backend = get_default_backend()
        analyzer = MetaphorInvarianceAnalyzer(backend)

        # Two metaphors, three models
        trajectories = {
            "model_a": [
                make_trajectory("meta1", "model_a", [0.1, 0.5, 0.9, 0.7]),
                make_trajectory("meta2", "model_a", [0.2, 0.6, 0.8, 0.5]),
            ],
            "model_b": [
                make_trajectory("meta1", "model_b", [0.1, 0.5, 0.9, 0.7]),  # Same as A
                make_trajectory("meta2", "model_b", [0.2, 0.6, 0.8, 0.5]),  # Same as A
            ],
            "model_c": [
                make_trajectory("meta1", "model_c", [0.5, 0.3, 0.2, 0.1]),  # Different
                make_trajectory("meta2", "model_c", [0.8, 0.6, 0.4, 0.2]),  # Different
            ],
        }

        result = analyzer.batch_invariance_test(trajectories)

        # 3 models -> 3 pairs (A-B, A-C, B-C) x 2 metaphors = 6 results
        assert len(result.results) == 6

        # A-B pairs should have high CKA, A-C and B-C lower
        assert result.mean_trajectory_cka > 0.5  # Mix of high and low

        # Per-metaphor breakdown
        assert "meta1" in result.per_metaphor_trajectory_cka
        assert "meta2" in result.per_metaphor_trajectory_cka


class TestPlatonicMetaphorValidator:
    """Tests for Platonic hypothesis validation."""

    def test_validate_cross_architecture(self):
        """Should return aggregate statistics for Platonic hypothesis testing."""
        backend = get_default_backend()
        validator = PlatonicMetaphorValidator(backend)

        # Similar trajectories across models (supporting Platonic hypothesis)
        trajectories = {
            "qwen": [
                make_trajectory("time_money", "qwen", [0.1, 0.4, 0.8, 0.6]),
            ],
            "llama": [
                make_trajectory("time_money", "llama", [0.12, 0.38, 0.82, 0.58]),
            ],
        }

        result = validator.validate_cross_architecture(trajectories)

        assert "mean_trajectory_cka" in result
        assert "std_trajectory_cka" in result
        assert result["model_count"] == 2
        assert result["metaphor_count"] == 1
        assert result["pair_count"] == 1

        # Similar trajectories should have high CKA
        assert result["mean_trajectory_cka"] > 0.95


class TestSerialization:
    """Tests for result serialization."""

    def test_invariance_result_to_dict(self):
        """Should serialize invariance result to dict."""
        result = MetaphorInvarianceResult(
            metaphor_id="test",
            metaphor_name="TEST",
            model_a="model_a",
            model_b="model_b",
            trajectory_cka=0.85,
            convergence_layer_a=5,
            convergence_layer_b=6,
            convergence_layer_delta_normalized=0.1,
            peak_cka_a=0.9,
            peak_cka_b=0.88,
            peak_cka_delta=0.02,
        )

        d = invariance_result_to_dict(result)

        assert d["metaphor_id"] == "test"
        assert d["trajectory_cka"] == 0.85
        assert d["convergence_layer_delta_normalized"] == 0.1

    def test_batch_result_to_dict(self):
        """Should serialize batch result to dict."""
        result = BatchInvarianceResult(
            results=[],
            mean_trajectory_cka=0.8,
            std_trajectory_cka=0.1,
            mean_convergence_delta=0.05,
            mean_peak_cka_delta=0.02,
            per_metaphor_trajectory_cka={"test": 0.9},
            per_family_trajectory_cka={},
        )

        d = batch_result_to_dict(result)

        assert d["mean_trajectory_cka"] == 0.8
        assert d["std_trajectory_cka"] == 0.1
        assert d["per_metaphor_trajectory_cka"]["test"] == 0.9
