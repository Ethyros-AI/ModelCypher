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

"""Tests for metaphor trajectory analysis."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
    CMTFamily,
    CMTMapping,
    ConceptualMetaphorInventory,
)
from modelcypher.core.domain.geometry.metaphor_trajectory import (
    ConvergenceProfile,
    MetaphorTrajectory,
    MetaphorTrajectoryCollector,
    MetaphorTrajectoryPoint,
    compute_convergence_profile,
    convergence_profile_to_dict,
    trajectory_to_dict,
)


class TestCMTAtlas:
    """Tests for Conceptual Metaphor Theory atlas."""

    def test_all_mappings_exist(self):
        """All 8 CMT mappings should be available."""
        assert len(ConceptualMetaphorInventory.ALL_MAPPINGS) == 8

    def test_mapping_structure(self):
        """Each mapping should have required fields."""
        for mapping in ConceptualMetaphorInventory.ALL_MAPPINGS:
            assert mapping.id
            assert mapping.name
            assert mapping.source_domain
            assert mapping.target_domain
            assert len(mapping.source_exemplars) > 0
            assert len(mapping.target_exemplars) > 0
            assert len(mapping.bridging_expressions) > 0

    def test_get_by_id(self):
        """Should find mapping by ID."""
        mapping = ConceptualMetaphorInventory.get_by_id("cmt_time_is_money")
        assert mapping is not None
        assert mapping.name == "TIME IS MONEY"

    def test_get_by_name(self):
        """Should find mapping by name."""
        mapping = ConceptualMetaphorInventory.get_by_name("TIME IS MONEY")
        assert mapping is not None
        assert mapping.id == "cmt_time_is_money"

    def test_get_by_name_case_insensitive(self):
        """Name lookup should be case-insensitive."""
        mapping = ConceptualMetaphorInventory.get_by_name("time is money")
        assert mapping is not None
        assert mapping.id == "cmt_time_is_money"

    def test_mappings_by_family(self):
        """Should filter by family."""
        time_mappings = ConceptualMetaphorInventory.mappings_by_family(
            CMTFamily.TIME_AS_RESOURCE
        )
        assert len(time_mappings) == 1
        assert time_mappings[0].name == "TIME IS MONEY"


class TestMetaphorTrajectory:
    """Tests for MetaphorTrajectory dataclass."""

    def test_empty_trajectory(self):
        """Empty trajectory should have sensible defaults."""
        traj = MetaphorTrajectory(
            metaphor_id="test",
            metaphor_name="TEST",
            source_domain="A",
            target_domain="B",
            model_id="model",
            points=[],
        )
        assert traj.convergence_layer == -1
        assert traj.peak_cka == 0.0
        assert traj.layer_count == 0

    def test_single_point_trajectory(self):
        """Single point trajectory should work."""
        point = MetaphorTrajectoryPoint(
            layer_index=5,
            cka_source_target=0.8,
            cosine_similarity=0.7,
            source_centroid_norm=1.0,
            target_centroid_norm=1.0,
        )
        traj = MetaphorTrajectory(
            metaphor_id="test",
            metaphor_name="TEST",
            source_domain="A",
            target_domain="B",
            model_id="model",
            points=[point],
        )
        assert traj.convergence_layer == 5
        assert traj.peak_cka == 0.8
        assert traj.layer_count == 1

    def test_convergence_layer_is_max_cka(self):
        """Convergence layer should be where CKA peaks."""
        points = [
            MetaphorTrajectoryPoint(
                layer_index=0, cka_source_target=0.2, cosine_similarity=0.1,
                source_centroid_norm=1.0, target_centroid_norm=1.0
            ),
            MetaphorTrajectoryPoint(
                layer_index=1, cka_source_target=0.5, cosine_similarity=0.3,
                source_centroid_norm=1.0, target_centroid_norm=1.0
            ),
            MetaphorTrajectoryPoint(
                layer_index=2, cka_source_target=0.9, cosine_similarity=0.8,
                source_centroid_norm=1.0, target_centroid_norm=1.0
            ),
            MetaphorTrajectoryPoint(
                layer_index=3, cka_source_target=0.7, cosine_similarity=0.6,
                source_centroid_norm=1.0, target_centroid_norm=1.0
            ),
        ]
        traj = MetaphorTrajectory(
            metaphor_id="test",
            metaphor_name="TEST",
            source_domain="A",
            target_domain="B",
            model_id="model",
            points=points,
        )
        assert traj.convergence_layer == 2
        assert traj.peak_cka == 0.9

    def test_cka_at_layer(self):
        """Should retrieve CKA at specific layer."""
        points = [
            MetaphorTrajectoryPoint(
                layer_index=0, cka_source_target=0.2, cosine_similarity=0.1,
                source_centroid_norm=1.0, target_centroid_norm=1.0
            ),
            MetaphorTrajectoryPoint(
                layer_index=5, cka_source_target=0.8, cosine_similarity=0.7,
                source_centroid_norm=1.0, target_centroid_norm=1.0
            ),
        ]
        traj = MetaphorTrajectory(
            metaphor_id="test",
            metaphor_name="TEST",
            source_domain="A",
            target_domain="B",
            model_id="model",
            points=points,
        )
        assert traj.cka_at_layer(0) == 0.2
        assert traj.cka_at_layer(5) == 0.8
        assert traj.cka_at_layer(99) is None


class TestConvergenceProfile:
    """Tests for convergence profile computation."""

    def test_empty_trajectory_profile(self):
        """Empty trajectory should produce zero profile."""
        traj = MetaphorTrajectory(
            metaphor_id="test",
            metaphor_name="TEST",
            source_domain="A",
            target_domain="B",
            model_id="model",
            points=[],
        )
        profile = compute_convergence_profile(traj)
        assert profile.convergence_layer == -1
        assert profile.peak_cka == 0.0
        assert profile.layer_count == 0

    def test_monotonic_increasing_trajectory(self):
        """Monotonically increasing trajectory should have positive monotonicity."""
        points = [
            MetaphorTrajectoryPoint(
                layer_index=i, cka_source_target=i * 0.1 + 0.1, cosine_similarity=0.5,
                source_centroid_norm=1.0, target_centroid_norm=1.0
            )
            for i in range(10)
        ]
        traj = MetaphorTrajectory(
            metaphor_id="test",
            metaphor_name="TEST",
            source_domain="A",
            target_domain="B",
            model_id="model",
            points=points,
        )
        profile = compute_convergence_profile(traj)
        assert profile.trajectory_monotonicity > 0.9  # Should be ~1.0
        assert profile.convergence_layer == 9  # Last layer has max CKA
        assert profile.late_layer_cka > profile.early_layer_cka

    def test_layer_region_means(self):
        """Should compute early/mid/late layer means correctly."""
        # 8 layers: early=0-1, mid=2-5, late=6-7
        cka_values = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        points = [
            MetaphorTrajectoryPoint(
                layer_index=i, cka_source_target=cka_values[i], cosine_similarity=0.5,
                source_centroid_norm=1.0, target_centroid_norm=1.0
            )
            for i in range(8)
        ]
        traj = MetaphorTrajectory(
            metaphor_id="test",
            metaphor_name="TEST",
            source_domain="A",
            target_domain="B",
            model_id="model",
            points=points,
        )
        profile = compute_convergence_profile(traj)
        # Early = first 25% = 2 layers -> mean(0.1, 0.2) = 0.15
        assert abs(profile.early_layer_cka - 0.15) < 0.01
        # Late = last 25% = 2 layers -> mean(0.8, 0.9) = 0.85
        assert abs(profile.late_layer_cka - 0.85) < 0.01


class TestMetaphorTrajectoryCollector:
    """Tests for trajectory collection."""

    def test_collect_from_activations(self):
        """Should compute CKA between source and target activations."""
        backend = get_default_backend()
        collector = MetaphorTrajectoryCollector(backend)

        # Create synthetic activations
        backend.random_seed(42)
        n_samples = 20
        hidden_dim = 64

        # Layer activations: dict[layer_idx, (source_acts, target_acts)]
        layer_activations = {}
        for layer in range(4):
            # Source and target activations get more similar at layer 2
            source = backend.random_normal((n_samples, hidden_dim))
            if layer == 2:
                # At layer 2, target is very similar to source
                noise = backend.random_normal((n_samples, hidden_dim))
                target = source + noise * 0.1
            else:
                # Other layers: target is random
                target = backend.random_normal((n_samples, hidden_dim))
            layer_activations[layer] = (source, target)

        mapping = ConceptualMetaphorInventory.TIME_IS_MONEY
        traj = collector.collect_from_activations(mapping, "test_model", layer_activations)

        assert traj.metaphor_id == "cmt_time_is_money"
        assert traj.model_id == "test_model"
        assert len(traj.points) == 4
        # Layer 2 should have highest CKA since source ≈ target
        assert traj.convergence_layer == 2
        assert traj.peak_cka > 0.9


class TestSerialization:
    """Tests for trajectory/profile serialization."""

    def test_trajectory_to_dict(self):
        """Should serialize trajectory to dict."""
        points = [
            MetaphorTrajectoryPoint(
                layer_index=0, cka_source_target=0.5, cosine_similarity=0.4,
                source_centroid_norm=1.0, target_centroid_norm=1.1
            ),
        ]
        traj = MetaphorTrajectory(
            metaphor_id="test",
            metaphor_name="TEST",
            source_domain="A",
            target_domain="B",
            model_id="model",
            points=points,
        )
        d = trajectory_to_dict(traj)
        assert d["metaphor_id"] == "test"
        assert d["convergence_layer"] == 0
        assert d["peak_cka"] == 0.5
        assert len(d["points"]) == 1

    def test_convergence_profile_to_dict(self):
        """Should serialize profile to dict."""
        profile = ConvergenceProfile(
            convergence_layer=5,
            peak_cka=0.9,
            early_layer_cka=0.3,
            mid_layer_cka=0.6,
            late_layer_cka=0.8,
            trajectory_monotonicity=0.95,
            layer_count=10,
        )
        d = convergence_profile_to_dict(profile)
        assert d["convergence_layer"] == 5
        assert d["peak_cka"] == 0.9
        assert d["trajectory_monotonicity"] == 0.95
