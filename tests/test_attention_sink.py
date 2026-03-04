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

"""Tests for attention sink analysis."""

import math

import pytest

from modelcypher.core.domain.geometry.attention_sink import (
    compute_active_sinks,
    compute_sink_scores,
    summarize_layer_sinks,
)


class TestComputeSinkScores:
    """Tests for compute_sink_scores on synthetic causal attention matrices."""

    def test_uniform_causal_attention(self):
        """Uniform causal: A[u][i] = 1/(u+1) for i <= u.

        Token 0 is attended by all T tokens, each giving 1/(u+1).
        Analytically: s_0 = (1/T) * sum_{u=0}^{T-1} 1/(u+1) = H_T / T
        where H_T is the T-th harmonic number.
        """
        T = 4
        A = [[0.0] * T for _ in range(T)]
        for u in range(T):
            weight = 1.0 / (u + 1)
            for i in range(u + 1):
                A[u][i] = weight

        result = compute_sink_scores(A)

        # s_0 = (1/4) * (1/1 + 1/2 + 1/3 + 1/4) = (1/4) * (25/12) = 25/48
        harmonic_4 = 1.0 + 0.5 + 1.0 / 3 + 0.25
        expected_s0 = harmonic_4 / T
        assert abs(result.token_sinks[0].sink_score - expected_s0) < 1e-10

    def test_concentrated_first_token(self):
        """When all attention goes to token 0, it has highest sink score."""
        T = 4
        A = [[0.0] * T for _ in range(T)]
        for u in range(T):
            A[u][0] = 1.0  # all attention to position 0

        result = compute_sink_scores(A)
        assert result.max_sink_position == 0
        assert abs(result.max_sink_score - 1.0) < 1e-10

        # All other positions should have sink_score 0 (except self-attn row)
        for i in range(1, T):
            assert result.token_sinks[i].sink_score < 1e-10

    def test_self_attention_only(self):
        """Identity-like: each token only attends to itself."""
        T = 4
        A = [[0.0] * T for _ in range(T)]
        for u in range(T):
            A[u][u] = 1.0

        result = compute_sink_scores(A)

        # s_i = (1/(T-i)) * A[i][i] = 1/(T-i) for each i
        for i in range(T):
            expected = 1.0 / (T - i)
            assert abs(result.token_sinks[i].sink_score - expected) < 1e-10

        # Last token always has highest sink score (1/(T-T+1) = 1/1 = 1.0)
        assert result.max_sink_position == T - 1

    def test_consistency_error_near_zero(self):
        """Consistency error |s_i*(T-i) - col_sum| should be near zero."""
        T = 5
        # Create a realistic causal attention pattern
        A = [[0.0] * T for _ in range(T)]
        for u in range(T):
            weight = 1.0 / (u + 1)
            for i in range(u + 1):
                A[u][i] = weight

        result = compute_sink_scores(A)

        for ts in result.token_sinks:
            assert ts.consistency_error < 1e-14, (
                f"Consistency error at position {ts.position}: {ts.consistency_error}"
            )

    def test_lapeigval_identity_error_near_zero(self):
        """LapEigval identity residual should be near zero."""
        T = 5
        A = [[0.0] * T for _ in range(T)]
        for u in range(T):
            weight = 1.0 / (u + 1)
            for i in range(u + 1):
                A[u][i] = weight

        result = compute_sink_scores(A)
        for ts in result.token_sinks:
            assert ts.lap_eigval_identity_error < 1e-14, (
                f"LapEigval identity error at position {ts.position}: "
                f"{ts.lap_eigval_identity_error}"
            )

    def test_sink_scores_sum_property(self):
        """For row-stochastic causal matrix, sum of s_i * (T-i) = T."""
        T = 5
        A = [[0.0] * T for _ in range(T)]
        for u in range(T):
            # Random-ish but valid row-stochastic causal pattern
            weights = [1.0 / (abs(u - i) + 1) for i in range(u + 1)]
            total = sum(weights)
            for i in range(u + 1):
                A[u][i] = weights[i] / total

        result = compute_sink_scores(A)

        # sum_{i=0}^{T-1} s_i * (T-i) = sum_{i} sum_{u>=i} A[u][i]
        # = sum_u sum_{i<=u} A[u][i] = sum_u 1.0 = T (row-stochastic)
        weighted_sum = sum(
            ts.sink_score * (T - ts.position) for ts in result.token_sinks
        )
        assert abs(weighted_sum - T) < 1e-10

    def test_empty_matrix(self):
        """Empty matrix returns empty result."""
        result = compute_sink_scores([])
        assert result.token_sinks == []
        assert result.max_sink_score == 0.0

    def test_to_dict_camel_case(self):
        """Verify camelCase serialization."""
        A = [[1.0]]
        result = compute_sink_scores(A)
        d = result.to_dict()
        assert "headIdx" in d
        assert "maxSinkPosition" in d
        assert "maxSinkScore" in d
        td = d["tokenSinks"][0]
        assert "sinkScore" in td
        assert "selfAttention" in td
        assert "consistencyError" in td
        assert "lapEigvalIdentityError" in td


class TestComputeActiveSinks:
    """Tests for active sink scores (sink * value norm)."""

    def test_value_norm_weighting(self):
        """Active score = sink_score * ||V_i||."""
        T = 3
        A = [[0.0] * T for _ in range(T)]
        for u in range(T):
            A[u][0] = 1.0  # all attention to token 0

        head_sink = compute_sink_scores(A)
        value_norms = [2.0, 5.0, 0.5]

        active = compute_active_sinks(head_sink, value_norms)

        # Token 0 has sink_score=1.0, V norm=2.0 → active=2.0
        # Token 1 has sink_score=0.0, V norm=5.0 → active=0.0
        assert abs(active.active_scores[0] - 2.0) < 1e-10
        assert abs(active.active_scores[1] - 0.0) < 1e-10

    def test_zero_value_norm_suppresses(self):
        """Zero value norm suppresses active score regardless of sink score."""
        T = 3
        # Spread attention so multiple tokens have nonzero sink scores
        A = [[0.0] * T for _ in range(T)]
        A[0][0] = 1.0
        A[1][0] = 0.6
        A[1][1] = 0.4
        A[2][0] = 0.5
        A[2][1] = 0.3
        A[2][2] = 0.2

        head_sink = compute_sink_scores(A)
        # Token 0 has highest sink but zero V norm → suppressed
        assert head_sink.max_sink_position == 0
        value_norms = [0.0, 1.0, 1.0]

        active = compute_active_sinks(head_sink, value_norms)
        assert abs(active.active_scores[0]) < 1e-10
        assert active.max_active_position != 0

    def test_active_differs_from_raw(self):
        """Active scores differ from raw sink scores when V norms vary."""
        T = 3
        A = [[0.0] * T for _ in range(T)]
        # Token 0 gets most attention
        A[0][0] = 1.0
        A[1][0] = 0.8
        A[1][1] = 0.2
        A[2][0] = 0.7
        A[2][1] = 0.2
        A[2][2] = 0.1

        head_sink = compute_sink_scores(A)

        # Without value weighting, token 0 has highest sink
        assert head_sink.max_sink_position == 0

        # But if token 0 has tiny V norm and token 2 has huge V norm...
        value_norms = [0.01, 1.0, 100.0]
        active = compute_active_sinks(head_sink, value_norms)

        # Token 2's active score should dominate despite low sink score
        assert active.max_active_position == 2

    def test_to_dict_camel_case(self):
        """Verify camelCase serialization."""
        A = [[1.0]]
        head_sink = compute_sink_scores(A)
        active = compute_active_sinks(head_sink, [1.0])
        d = active.to_dict()
        assert "headIdx" in d
        assert "activeScores" in d
        assert "maxActivePosition" in d
        assert "maxActiveScore" in d


class TestSummarizeLayerSinks:
    """Tests for layer-level aggregation."""

    def test_dominant_position(self):
        """Dominant sink position is the most common max_sink across heads."""
        T = 3
        # Head 0: sink at position 0
        A0 = [[0.0] * T for _ in range(T)]
        for u in range(T):
            A0[u][0] = 1.0
        # Head 1: sink at position 0
        A1 = [[0.0] * T for _ in range(T)]
        for u in range(T):
            A1[u][0] = 1.0
        # Head 2: sink at position 2
        A2 = [[0.0] * T for _ in range(T)]
        for u in range(T):
            A2[u][u] = 1.0

        heads = [
            compute_sink_scores(A0, head_idx=0),
            compute_sink_scores(A1, head_idx=1),
            compute_sink_scores(A2, head_idx=2),
        ]

        layer = summarize_layer_sinks(heads, layer_idx=5)
        # 2 out of 3 heads have max at position 0
        assert layer.dominant_sink_position == 0
        assert layer.layer_idx == 5

    def test_mean_max_sink(self):
        """Mean max sink score is average of per-head max scores."""
        A1 = [[1.0]]
        A2 = [[1.0]]
        heads = [
            compute_sink_scores(A1, head_idx=0),
            compute_sink_scores(A2, head_idx=1),
        ]
        layer = summarize_layer_sinks(heads)
        assert abs(layer.mean_max_sink_score - 1.0) < 1e-10

    def test_to_dict_camel_case(self):
        """Verify camelCase serialization."""
        A = [[1.0]]
        heads = [compute_sink_scores(A)]
        layer = summarize_layer_sinks(heads, layer_idx=3)
        d = layer.to_dict()
        assert "layerIdx" in d
        assert "headResults" in d
        assert "meanMaxSinkScore" in d
        assert "dominantSinkPosition" in d
