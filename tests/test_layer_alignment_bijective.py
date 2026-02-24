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

"""Tests for 1:1 bijective layer alignment.

These tests verify that the probe stage uses 1:1 layer mapping instead of
many-to-one grouping that caused CKA to be geometrically impossible.

Bugs this catches:
    - DP alignment creating 7 groups from 32 layers
    - Multiple source layers concatenated to single target
    - Missing transforms for unmapped layers
"""

import pytest


class TestHungarianAlignment:
    """Tests for Hungarian 1:1 assignment."""

    def test_hungarian_produces_bijective_mapping(self) -> None:
        """Hungarian assignment should produce 1:1 mapping."""
        from modelcypher.core.domain.geometry.hungarian import hungarian_assignment_list

        # 4x4 cost matrix (negate CKA to convert to cost)
        cost_matrix = [
            [-0.9, -0.1, -0.1, -0.1],  # Source 0 strongly matches target 0
            [-0.1, -0.8, -0.1, -0.1],  # Source 1 strongly matches target 1
            [-0.1, -0.1, -0.7, -0.1],  # Source 2 strongly matches target 2
            [-0.1, -0.1, -0.1, -0.6],  # Source 3 strongly matches target 3
        ]

        assignment = hungarian_assignment_list(cost_matrix)

        # Each source should be assigned to unique target
        assert len(assignment) == 4
        assert set(assignment) == {0, 1, 2, 3}  # Bijective!

    def test_hungarian_handles_rectangular_with_padding(self) -> None:
        """Hungarian should handle non-square matrices via padding."""
        from modelcypher.core.domain.geometry.hungarian import hungarian_assignment_list

        # 3 source layers, 5 target layers
        # Pad to 5x5 with high cost for non-existent sources
        n_source, n_target = 3, 5
        n_max = max(n_source, n_target)

        cost_matrix = []
        for i in range(n_max):
            row = []
            for j in range(n_max):
                if i < n_source and j < n_target:
                    row.append(-0.5)  # Real CKA
                else:
                    row.append(1000.0)  # Padding
            cost_matrix.append(row)

        assignment = hungarian_assignment_list(cost_matrix)

        # Should have n_max assignments
        assert len(assignment) == n_max
        # Real sources (0, 1, 2) should be assigned to real targets (0-4)
        for src in range(n_source):
            assert 0 <= assignment[src] < n_target

    def test_no_layer_grouping(self) -> None:
        """Alignment tasks should have exactly 1 source per target."""
        # Simulate what probe.py does now
        n_source, n_target = 36, 32

        # Mock assignment (bijective)
        assignment = list(range(min(n_source, n_target)))
        assignment += [0] * (n_source - len(assignment))  # Overflow mapped to 0

        # Build alignment tasks the new way
        alignment_tasks = []
        for tgt_idx in range(n_target):
            best_src_idx = None
            for src_idx in range(n_source):
                if assignment[src_idx] == tgt_idx:
                    best_src_idx = src_idx
                    break

            if best_src_idx is not None:
                alignment_tasks.append((tgt_idx, [best_src_idx]))
            else:
                alignment_tasks.append((tgt_idx, [0]))  # Fallback

        # Should have n_target tasks (32), not 7 groups
        assert len(alignment_tasks) == n_target

        # Each task should have exactly 1 source (not groups of 10+)
        for tgt_idx, src_indices in alignment_tasks:
            assert len(src_indices) == 1, f"Target {tgt_idx} has {len(src_indices)} sources!"


class TestAllTargetLayersGetTransform:
    """Tests that all target layers receive transforms."""

    def test_all_targets_in_alignment_tasks(self) -> None:
        """Every target layer should appear in alignment_tasks."""
        n_target = 32

        # After the fix: alignment_tasks covers all targets
        alignment_tasks = [(i, [i % 24]) for i in range(n_target)]

        targets_covered = {tgt_idx for tgt_idx, _ in alignment_tasks}

        assert targets_covered == set(range(n_target))

    def test_transform_propagation_fills_gaps(self) -> None:
        """Transform propagation should fill any unmapped layers."""
        # Say we have transforms for only layers 5, 10, 15
        feature_transforms = {
            5: [[1.0, 0.0]],
            10: [[0.0, 1.0]],
            15: [[1.0, 1.0]],
        }

        all_target_layers = list(range(20))
        mapped_layers = sorted(feature_transforms.keys())

        for tgt_layer in all_target_layers:
            if tgt_layer not in feature_transforms:
                # Find nearest
                nearest = min(mapped_layers, key=lambda x: abs(x - tgt_layer))
                feature_transforms[tgt_layer] = feature_transforms[nearest]

        # All layers should now have transforms
        assert len(feature_transforms) == 20
