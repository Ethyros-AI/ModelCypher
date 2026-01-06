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

"""Tests for metaphor geometry modules (invariants, invariance, trajectory).

Tests cover:
- Metaphor probe generation (metaphor_invariants)
- Trajectory convergence profiling (metaphor_trajectory)
- Invariance analysis (metaphor_invariance)
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock

from modelcypher.core.domain.geometry.metaphor_invariants import (
    MetaphorDomain,
    MetaphorProbe,
    generate_time_probes,
    generate_all_metaphor_probes,
    generate_cross_cultural_pairs,
)
from modelcypher.core.domain.geometry.metaphor_trajectory import (
    MetaphorTrajectory,
    MetaphorTrajectoryPoint,
    compute_convergence_profile,
    ConvergenceProfile,
    _compute_spearman_correlation,
)
from modelcypher.core.domain.geometry.metaphor_invariance import (
    MetaphorInvarianceAnalyzer,
    MetaphorInvarianceResult,
)


# =============================================================================
# Metaphor Invariants (Probe Generation)
# =============================================================================


class TestMetaphorInvariants:
    """Tests for metaphor probe generation."""

    def test_generate_time_probes(self):
        """generate_time_probes produces probes for TIME domain."""
        probes = generate_time_probes()
        assert len(probes) > 0
        assert all(isinstance(p, MetaphorProbe) for p in probes)
        assert all(p.domain == MetaphorDomain.TIME for p in probes)

    def test_generate_all_probes_filters_domains(self):
        """generate_all_metaphor_probes filters by domain."""
        domains = {MetaphorDomain.TIME}
        probes = generate_all_metaphor_probes(domains=domains)
        assert all(p.domain == MetaphorDomain.TIME for p in probes)
        
        # Should not contain other domains
        assert not any(p.domain == MetaphorDomain.EMOTION for p in probes)

    def test_cross_cultural_pairs(self):
        """generate_cross_cultural_pairs produces pairs."""
        pairs = generate_cross_cultural_pairs()
        assert len(pairs) > 0
        
        pair = pairs[0]
        assert pair.concept is not None
        assert pair.probe_a is not None
        assert pair.probe_b is not None


# =============================================================================
# Metaphor Trajectory
# =============================================================================


class TestMetaphorTrajectory:
    """Tests for MetaphorTrajectory and profiling."""

    def test_trajectory_properties(self):
        """Trajectory properties (peak_cka, convergence_layer) work."""
        points = [
            MetaphorTrajectoryPoint(0, 0.1, 0.1, 1.0, 1.0),
            MetaphorTrajectoryPoint(1, 0.5, 0.5, 1.0, 1.0),
            MetaphorTrajectoryPoint(2, 0.9, 0.9, 1.0, 1.0), # Peak
            MetaphorTrajectoryPoint(3, 0.8, 0.8, 1.0, 1.0),
        ]
        
        traj = MetaphorTrajectory(
            metaphor_id="test", 
            metaphor_name="Test Metaphor", 
            source_domain="source", 
            target_domain="target", 
            model_id="model_a", 
            points=tuple(points)
        )
        
        assert traj.layer_count == 4
        assert traj.peak_cka == 0.9
        assert traj.convergence_layer == 2
        assert traj.cka_at_layer(1) == 0.5
        assert traj.cka_at_layer(99) is None

    def test_spearman_correlation(self):
        """_compute_spearman_correlation works correctly."""
        # Perfectly correlated
        x = [1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0]
        assert _compute_spearman_correlation(x, y) == 1.0
        
        # Perfectly anti-correlated
        y_rev = [30.0, 20.0, 10.0]
        assert _compute_spearman_correlation(x, y_rev) == -1.0

    def test_convergence_profile(self):
        """compute_convergence_profile calculates metrics."""
        points = [
            MetaphorTrajectoryPoint(0, 0.2, 0.2, 1.0, 1.0),
            MetaphorTrajectoryPoint(1, 0.5, 0.5, 1.0, 1.0),
            MetaphorTrajectoryPoint(2, 0.8, 0.8, 1.0, 1.0),
        ]
        traj = MetaphorTrajectory(
            metaphor_id="test", 
            metaphor_name="Test", 
            source_domain="source", 
            target_domain="target", 
            model_id="model_a", 
            points=tuple(points)
        )
        
        profile = compute_convergence_profile(traj)
        
        assert isinstance(profile, ConvergenceProfile)
        assert profile.peak_cka == 0.8
        assert profile.trajectory_monotonicity > 0.9  # Should be high/1.0
        assert profile.early_layer_cka == 0.2
        assert profile.late_layer_cka == 0.8


# =============================================================================
# Metaphor Invariance Analyzer
# =============================================================================


class TestMetaphorInvarianceAnalyzer:
    """Tests for MetaphorInvarianceAnalyzer."""

    def test_compare_metaphor_geometry(self):
        """compare_metaphor_geometry computes trajectory CKA."""
        analyzer = MetaphorInvarianceAnalyzer()
        
        # Identical trajectories should have CKA ~ 1.0
        points_a = [
            MetaphorTrajectoryPoint(0, 0.1, 0.1, 1.0, 1.0),
            MetaphorTrajectoryPoint(1, 0.5, 0.5, 1.0, 1.0),
        ]
        traj_a = MetaphorTrajectory(
            metaphor_id="m1", 
            metaphor_name="Metaphor 1", 
            source_domain="source", 
            target_domain="target", 
            model_id="model_a", 
            points=tuple(points_a)
        )
        traj_b = MetaphorTrajectory(
            metaphor_id="m1", 
            metaphor_name="Metaphor 1", 
            source_domain="source", 
            target_domain="target", 
            model_id="model_b", 
            points=tuple(points_a)
        )
        
        result = analyzer.compare_metaphor_geometry(traj_a, traj_b)
        
        assert isinstance(result, MetaphorInvarianceResult)
        assert result.metaphor_id == "m1"
        assert result.model_a == "model_a"
        assert result.model_b == "model_b"
        assert abs(result.trajectory_cka - 1.0) < 1e-5
        assert result.peak_cka_delta == 0.0

    def test_compare_metaphor_diff_lengths(self):
        """compare_metaphor_geometry handles different lengths."""
        analyzer = MetaphorInvarianceAnalyzer()
        
        points_a = [
            MetaphorTrajectoryPoint(0, 0.1, 0.1, 1.0, 1.0),
            MetaphorTrajectoryPoint(1, 0.9, 0.9, 1.0, 1.0),
        ]
        # Model B has more layers, but similar shape (interpolated)
        points_b = [
            MetaphorTrajectoryPoint(0, 0.1, 0.1, 1.0, 1.0),
            MetaphorTrajectoryPoint(1, 0.5, 0.5, 1.0, 1.0),
            MetaphorTrajectoryPoint(2, 0.9, 0.9, 1.0, 1.0),
        ]
        
        traj_a = MetaphorTrajectory(
            metaphor_id="m1", 
            metaphor_name="M1", 
            source_domain="source", 
            target_domain="target", 
            model_id="A", 
            points=tuple(points_a)
        )
        traj_b = MetaphorTrajectory(
            metaphor_id="m1", 
            metaphor_name="M1", 
            source_domain="source", 
            target_domain="target", 
            model_id="B", 
            points=tuple(points_b)
        )
        
        result = analyzer.compare_metaphor_geometry(traj_a, traj_b)
        
        # Should be high similarity
        assert result.trajectory_cka > 0.9
        
    def test_batch_invariance_test(self):
        """batch_invariance_test aggregates results."""
        analyzer = MetaphorInvarianceAnalyzer()
        
        # Use points with variance so CKA is defined (centering yields non-zero)
        points = [
            MetaphorTrajectoryPoint(0, 0.1, 0.1, 1.0, 1.0),
            MetaphorTrajectoryPoint(1, 0.9, 0.9, 1.0, 1.0),
        ]
        traj_a = MetaphorTrajectory(
            metaphor_id="m1", 
            metaphor_name="M1", 
            source_domain="source", 
            target_domain="target", 
            model_id="A", 
            points=tuple(points)
        )
        traj_b = MetaphorTrajectory(
            metaphor_id="m1", 
            metaphor_name="M1", 
            source_domain="source", 
            target_domain="target", 
            model_id="B", 
            points=tuple(points)
        )
        
        trajectories = {
            "A": [traj_a],
            "B": [traj_b],
        }
        
        batch_result = analyzer.batch_invariance_test(trajectories)
        
        assert len(batch_result.results) == 1
        assert batch_result.mean_trajectory_cka > 0.9
