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

"""Tests for attention collapse detection.

All thresholds derived from IEEE 754 machine precision.
No magic numbers.
"""
from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.attention_collapse import (
    AttentionCollapseResult,
    CollapseProfile,
    LayerCollapseResult,
    compute_attention_collapse,
    compute_collapse_profile,
    summarize_layer_collapse,
)


class TestComputeAttentionCollapse:
    """Tests for compute_attention_collapse on synthetic matrices."""

    def test_rank1_matrix_detected(self) -> None:
        """A rank-1 attention matrix (all rows identical) should be detected."""
        row = [0.7, 0.2, 0.1, 0.0]
        matrix = [row[:] for _ in range(4)]
        result = compute_attention_collapse(matrix, "float32")
        assert result.is_rank1
        # σ₂ should be ~0 for rank-1 matrix
        assert result.rank1_ratio < math.sqrt(math.ldexp(1.0, -23))

    def test_full_rank_matrix_not_collapsed(self) -> None:
        """Identity-like attention should NOT be rank-1."""
        matrix = [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
        result = compute_attention_collapse(matrix, "float32")
        assert not result.is_rank1
        assert result.rank1_ratio > 0.1

    def test_column_mass_sums_to_one(self) -> None:
        """Column mass distribution should sum to 1.0."""
        matrix = [
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.2, 0.2, 0.6],
        ]
        result = compute_attention_collapse(matrix, "float32")
        total = sum(result.column_mass)
        assert abs(total - 1.0) < 1e-6

    def test_gradient_suppression_near_zero_for_rank1(self) -> None:
        """Gradient suppression = σ₂ / √(2T). For rank-1, σ₂ ≈ 0."""
        row = [1.0, 0.0, 0.0, 0.0]
        matrix = [row[:] for _ in range(4)]
        result = compute_attention_collapse(matrix, "float32")
        assert result.gradient_suppression < 1e-6

    def test_singular_values_descending(self) -> None:
        """Singular values should be in descending order."""
        matrix = [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.5, 0.2, 0.2],
            [0.2, 0.2, 0.4, 0.2],
            [0.3, 0.1, 0.1, 0.5],
        ]
        result = compute_attention_collapse(matrix, "float32")
        for i in range(len(result.singular_values) - 1):
            assert result.singular_values[i] >= result.singular_values[i + 1]

    def test_effective_rank_1_for_rank1_matrix(self) -> None:
        """Effective rank should be ~1.0 for a rank-1 matrix."""
        row = [0.5, 0.3, 0.2]
        matrix = [row[:] for _ in range(3)]
        result = compute_attention_collapse(matrix, "float32")
        assert abs(result.effective_rank - 1.0) < 0.01

    def test_effective_rank_higher_for_full_rank(self) -> None:
        """Effective rank should be > 1 for a full-rank matrix."""
        # Identity-like: effective rank should approach dimension
        matrix = [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
        result = compute_attention_collapse(matrix, "float32")
        assert result.effective_rank > 1.5

    def test_bfloat16_uses_correct_epsilon(self) -> None:
        """bfloat16 has 7-bit mantissa, different from float16's 10-bit."""
        # Create a matrix that's rank-1 under bf16 but not under f32
        # bf16 sqrt(eps) ≈ 0.0884, f32 sqrt(eps) ≈ 3.45e-4
        # A matrix with σ₂/σ₁ ≈ 0.05 should be rank-1 under bf16 but not f32
        import numpy as np
        # Construct: rank-1 + small perturbation
        base_row = np.array([0.7, 0.2, 0.1])
        A = np.outer(np.ones(3), base_row)
        A[1, 0] += 0.03  # small perturbation giving σ₂/σ₁ ~ 0.03-0.05
        A = A / A.sum(axis=1, keepdims=True)  # re-normalize rows
        matrix = A.tolist()

        result_bf16 = compute_attention_collapse(matrix, "bfloat16")
        result_f32 = compute_attention_collapse(matrix, "float32")
        # bf16 is less precise, so more likely to call it rank-1
        assert result_bf16.is_rank1
        assert not result_f32.is_rank1

    def test_to_dict_camel_case(self) -> None:
        """to_dict should use camelCase keys."""
        matrix = [[0.5, 0.5], [0.3, 0.7]]
        result = compute_attention_collapse(matrix, "float32")
        d = result.to_dict()
        assert "singularValues" in d
        assert "rank1Ratio" in d
        assert "isRank1" in d
        assert "columnMass" in d
        assert "gradientSuppression" in d
        assert "effectiveRank" in d
        assert "position0Mass" in d

    def test_position_0_mass_from_column_mass(self) -> None:
        """position_0_mass equals column_mass[0]."""
        # Concentrated on column 0
        row = [0.9, 0.05, 0.05]
        matrix = [row[:] for _ in range(3)]
        result = compute_attention_collapse(matrix, "float32")
        assert abs(result.position_0_mass - result.column_mass[0]) < 1e-10
        assert result.position_0_mass > 0.5

    def test_position_0_mass_low_when_sink_elsewhere(self) -> None:
        """position_0_mass is low when attention concentrates on another column."""
        # Concentrated on column 2
        matrix = [
            [0.05, 0.05, 0.9],
            [0.05, 0.05, 0.9],
            [0.05, 0.05, 0.9],
        ]
        result = compute_attention_collapse(matrix, "float32")
        assert result.position_0_mass < 0.1


class TestSummarizeLayerCollapse:
    """Tests for summarize_layer_collapse."""

    def test_all_heads_collapsed(self) -> None:
        """If all heads are rank-1, layer is collapsed."""
        heads = [
            AttentionCollapseResult(
                singular_values=[1.0, 0.0, 0.0],
                rank1_ratio=0.0,
                is_rank1=True,
                column_mass=[1.0, 0.0, 0.0],
                gradient_suppression=0.0,
                effective_rank=1.0,
                position_0_mass=1.0,
            )
            for _ in range(4)
        ]
        result = summarize_layer_collapse(heads, layer_idx=5)
        assert result.is_collapsed
        assert result.collapsed_head_count == 4
        assert result.active_head_count == 0
        assert result.layer_idx == 5

    def test_mixed_heads(self) -> None:
        """Some collapsed, some active."""
        collapsed = AttentionCollapseResult(
            singular_values=[1.0, 0.0],
            rank1_ratio=0.0,
            is_rank1=True,
            column_mass=[1.0, 0.0],
            gradient_suppression=0.0,
            effective_rank=1.0,
            position_0_mass=1.0,
        )
        active = AttentionCollapseResult(
            singular_values=[0.7, 0.3],
            rank1_ratio=0.43,
            is_rank1=False,
            column_mass=[0.6, 0.4],
            gradient_suppression=0.12,
            effective_rank=1.8,
            position_0_mass=0.6,
        )
        result = summarize_layer_collapse([collapsed, active, collapsed], layer_idx=3)
        assert not result.is_collapsed
        assert result.collapsed_head_count == 2
        assert result.active_head_count == 1

    def test_no_heads_collapsed(self) -> None:
        """All heads active."""
        active = AttentionCollapseResult(
            singular_values=[0.5, 0.3, 0.2],
            rank1_ratio=0.6,
            is_rank1=False,
            column_mass=[0.4, 0.3, 0.3],
            gradient_suppression=0.1,
            effective_rank=2.5,
            position_0_mass=0.4,
        )
        result = summarize_layer_collapse([active, active], layer_idx=0)
        assert not result.is_collapsed
        assert result.collapsed_head_count == 0
        assert result.active_head_count == 2

    def test_layer_result_to_dict(self) -> None:
        """LayerCollapseResult.to_dict uses camelCase."""
        result = LayerCollapseResult(
            layer_idx=2,
            is_collapsed=False,
            collapsed_head_count=1,
            active_head_count=3,
            max_effective_rank=3.2,
            mean_gradient_suppression=0.05,
        )
        d = result.to_dict()
        assert d["layerIdx"] == 2
        assert "collapsedHeadCount" in d


class TestCollapseProfile:
    """Tests for compute_collapse_profile."""

    def test_collapse_onset_detection(self) -> None:
        """Onset should be the first layer with ANY collapsed head."""
        layers = [
            LayerCollapseResult(
                layer_idx=0, is_collapsed=False,
                collapsed_head_count=0, active_head_count=4,
                max_effective_rank=3.5, mean_gradient_suppression=0.1,
            ),
            LayerCollapseResult(
                layer_idx=2, is_collapsed=False,
                collapsed_head_count=1, active_head_count=3,
                max_effective_rank=3.0, mean_gradient_suppression=0.08,
            ),
            LayerCollapseResult(
                layer_idx=5, is_collapsed=True,
                collapsed_head_count=4, active_head_count=0,
                max_effective_rank=1.0, mean_gradient_suppression=0.001,
            ),
        ]
        profile = compute_collapse_profile(layers)
        assert profile.total_layers == 3
        assert profile.attention_layers == 3
        assert profile.collapsed_layer_count == 1
        assert profile.collapse_onset_layer == 2  # first layer with collapsed heads

    def test_no_collapse(self) -> None:
        """No collapsed heads anywhere."""
        layers = [
            LayerCollapseResult(
                layer_idx=i, is_collapsed=False,
                collapsed_head_count=0, active_head_count=4,
                max_effective_rank=3.0, mean_gradient_suppression=0.1,
            )
            for i in range(3)
        ]
        profile = compute_collapse_profile(layers)
        assert profile.collapsed_layer_count == 0
        assert profile.collapse_onset_layer is None

    def test_profile_to_dict(self) -> None:
        """CollapseProfile.to_dict uses camelCase and nests layer results."""
        layers = [
            LayerCollapseResult(
                layer_idx=0, is_collapsed=False,
                collapsed_head_count=0, active_head_count=2,
                max_effective_rank=2.0, mean_gradient_suppression=0.05,
            ),
        ]
        profile = compute_collapse_profile(layers)
        d = profile.to_dict()
        assert "totalLayers" in d
        assert "collapseOnsetLayer" in d
        assert "layerResults" in d
        assert len(d["layerResults"]) == 1
        assert "layerIdx" in d["layerResults"][0]
