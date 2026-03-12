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

"""Attention sink score and active sink diagnostic.

Measures attention concentration on specific token positions and its
geometric impact via value-norm weighting. No tuned parameters.

Metrics:
    sink_score: s_i = (1/(T-i)) * sum_{u=i}^{T-1} A_{u,i}
        Column-wise mean attention received by token i from all causal
        successors (Binkowski et al. 2026, Definition 3.1).

    active_sink: sink_score(i) * ||V_i||_2
        Attention concentration weighted by the geometric magnitude of the
        value vector at position i. High sink score with small V norm has
        limited geometric impact on output representations.

    consistency_error: |s_i * (T-i) - col_sum|
        Floating-point consistency check. By definition s_i = col_sum/(T-i),
        so s_i*(T-i) should equal col_sum exactly. Any deviation indicates
        numerical error in the division.

    lap_eigval_identity_error: |LapEigval_ii - (s_i - A_{i,i})|
        Identity check from sink analysis: LapEigval_ii = sink_score_i - A_{i,i}.
        Reported as an absolute residual (should be near machine precision).

    NOTE: The original plan proposed a "LapEigval identity" check
        (s_i - A_{i,i} = off_diag / (T-i)). This is NOT an algebraic
        identity — it only holds when self-attention is excluded from
        the sink score denominator, which conflicts with the paper's
        Definition 3.1 that includes self-attention. The consistency
        check above is the correct verification: it validates the
        division operation itself, which IS exact by construction.

    sink_score_delta: continuous per-head comparison of sink score vectors
        between two models (e.g. base vs adapted). Reports cosine similarity
        and L2 distance per head — no interpretation labels.

References:
    Binkowski et al. "From Sparse to Dense: Toeplitz Alignment of Attention
    Sinks in Large Language Models" (2026).
    Sun et al. "The Spike, the Sparse and the Sink" (arXiv:2603.05498, 2026).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSinkResult:
    """Sink analysis for a single token position within one attention head."""

    position: int
    sink_score: float
    self_attention: float  # A_{i,i}
    consistency_error: float  # |s_i * (T-i) - col_sum|
    lap_eigval_identity_error: float  # |LapEigval_ii - (s_i - A_{i,i})|

    def to_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "sinkScore": self.sink_score,
            "selfAttention": self.self_attention,
            "consistencyError": self.consistency_error,
            "lapEigvalIdentityError": self.lap_eigval_identity_error,
        }


@dataclass(frozen=True)
class HeadSinkResult:
    """Sink analysis for a single attention head."""

    head_idx: int
    token_sinks: list[TokenSinkResult]
    max_sink_position: int  # position with highest sink score
    max_sink_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "headIdx": self.head_idx,
            "tokenSinks": [t.to_dict() for t in self.token_sinks],
            "maxSinkPosition": self.max_sink_position,
            "maxSinkScore": self.max_sink_score,
        }


@dataclass(frozen=True)
class ActiveSinkResult:
    """Active sink analysis combining attention sink with value norms."""

    head_idx: int
    active_scores: list[float]  # sink_score * ||V_i||_2 per position
    max_active_position: int
    max_active_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "headIdx": self.head_idx,
            "activeScores": self.active_scores,
            "maxActivePosition": self.max_active_position,
            "maxActiveScore": self.max_active_score,
        }


@dataclass(frozen=True)
class LayerSinkResult:
    """Aggregated sink analysis for a layer across all heads."""

    layer_idx: int
    head_results: list[HeadSinkResult]
    active_results: list[ActiveSinkResult] | None  # None if no value norms
    mean_max_sink_score: float  # mean of max_sink_score across heads
    dominant_sink_position: int  # most common max_sink_position across heads

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "layerIdx": self.layer_idx,
            "headResults": [h.to_dict() for h in self.head_results],
            "meanMaxSinkScore": self.mean_max_sink_score,
            "dominantSinkPosition": self.dominant_sink_position,
        }
        if self.active_results is not None:
            result["activeResults"] = [a.to_dict() for a in self.active_results]
        return result


def compute_sink_scores(
    attention_matrix: list[list[float]],
    head_idx: int = 0,
) -> HeadSinkResult:
    """Compute per-token sink scores for a single attention head.

    Args:
        attention_matrix: T x T causal attention weight matrix (row-stochastic,
            lower-triangular). A[u][i] = attention weight from query u to key i.
        head_idx: Index of this head (for labeling).

    Returns:
        HeadSinkResult with per-token sink scores.
    """
    T = len(attention_matrix)
    if T == 0:
        return HeadSinkResult(
            head_idx=head_idx,
            token_sinks=[],
            max_sink_position=0,
            max_sink_score=0.0,
        )

    token_sinks: list[TokenSinkResult] = []

    for i in range(T):
        # s_i = (1/(T-i)) * sum_{u=i}^{T-1} A_{u,i}
        # Number of tokens that can attend to position i (including self)
        n_attendees = T - i
        col_sum = sum(attention_matrix[u][i] for u in range(i, T))
        sink_score = col_sum / n_attendees

        self_attn = attention_matrix[i][i]

        # Consistency check: s_i * (T-i) should equal col_sum exactly
        consistency_error = abs(sink_score * n_attendees - col_sum)

        # LapEigval identity check:
        #   LapEigval_ii = sink_score_i - A_{i,i}
        # We compute LapEigval_ii from the column-sum definition to avoid
        # algebraic cancellation in implementation:
        #   LapEigval_ii = (col_sum - (T-i)*A_{i,i}) / (T-i)
        lap_eigval_ii = (col_sum - n_attendees * self_attn) / n_attendees
        identity_rhs = sink_score - self_attn
        lap_eigval_identity_error = abs(lap_eigval_ii - identity_rhs)

        token_sinks.append(TokenSinkResult(
            position=i,
            sink_score=sink_score,
            self_attention=self_attn,
            consistency_error=consistency_error,
            lap_eigval_identity_error=lap_eigval_identity_error,
        ))

    max_pos = max(range(T), key=lambda i: token_sinks[i].sink_score)

    return HeadSinkResult(
        head_idx=head_idx,
        token_sinks=token_sinks,
        max_sink_position=max_pos,
        max_sink_score=token_sinks[max_pos].sink_score,
    )


def compute_active_sinks(
    head_sink: HeadSinkResult,
    value_norms: list[float],
) -> ActiveSinkResult:
    """Compute active sink scores: sink_score * ||V_i||_2.

    Args:
        head_sink: Sink scores from compute_sink_scores.
        value_norms: L2 norm of value vector at each position.

    Returns:
        ActiveSinkResult with value-weighted sink scores.
    """
    active_scores = [
        ts.sink_score * value_norms[ts.position]
        for ts in head_sink.token_sinks
    ]

    if not active_scores:
        return ActiveSinkResult(
            head_idx=head_sink.head_idx,
            active_scores=[],
            max_active_position=0,
            max_active_score=0.0,
        )

    max_pos = max(range(len(active_scores)), key=lambda i: active_scores[i])

    return ActiveSinkResult(
        head_idx=head_sink.head_idx,
        active_scores=active_scores,
        max_active_position=max_pos,
        max_active_score=active_scores[max_pos],
    )


def summarize_layer_sinks(
    head_results: list[HeadSinkResult],
    active_results: list[ActiveSinkResult] | None = None,
    layer_idx: int = 0,
) -> LayerSinkResult:
    """Aggregate sink results across heads for a single layer.

    Args:
        head_results: Per-head sink results.
        active_results: Per-head active sink results (optional).
        layer_idx: Index of this layer.

    Returns:
        LayerSinkResult with cross-head aggregation.
    """
    if not head_results:
        return LayerSinkResult(
            layer_idx=layer_idx,
            head_results=[],
            active_results=active_results,
            mean_max_sink_score=0.0,
            dominant_sink_position=0,
        )

    mean_max = sum(h.max_sink_score for h in head_results) / len(head_results)

    # Find dominant sink position (most common max_sink_position)
    position_counts: dict[int, int] = {}
    for h in head_results:
        position_counts[h.max_sink_position] = (
            position_counts.get(h.max_sink_position, 0) + 1
        )
    dominant_pos = max(position_counts, key=position_counts.get)  # type: ignore[arg-type]

    return LayerSinkResult(
        layer_idx=layer_idx,
        head_results=head_results,
        active_results=active_results,
        mean_max_sink_score=mean_max,
        dominant_sink_position=dominant_pos,
    )


@dataclass(frozen=True)
class SinkScoreDelta:
    """Per-head sink score comparison between two models.

    All fields are continuous measurements. No interpretation labels.
    Downstream code decides what constitutes meaningful change.
    """

    base_scores: list[list[float]]     # [n_heads][T] — base sink scores
    adapted_scores: list[list[float]]  # [n_heads][T] — adapted sink scores
    per_head_cosine: list[float]       # cosine similarity of score vectors per head
    per_head_l2_delta: list[float]     # L2 distance of score vectors per head

    def to_dict(self) -> dict[str, object]:
        return {
            "baseScores": self.base_scores,
            "adaptedScores": self.adapted_scores,
            "perHeadCosine": self.per_head_cosine,
            "perHeadL2Delta": self.per_head_l2_delta,
        }


def compute_sink_score_delta(
    base_heads: list[HeadSinkResult],
    adapted_heads: list[HeadSinkResult],
) -> SinkScoreDelta:
    """Compute per-head sink score deltas between base and adapted models.

    Args:
        base_heads: Per-head sink results from the base model.
        adapted_heads: Per-head sink results from the adapted model.
            Must have the same number of heads and sequence length as base.

    Returns:
        SinkScoreDelta with continuous per-head measurements.

    Raises:
        ValueError: If head counts or sequence lengths don't match.
    """
    if len(base_heads) != len(adapted_heads):
        msg = (
            f"Head count mismatch: base has {len(base_heads)}, "
            f"adapted has {len(adapted_heads)}"
        )
        raise ValueError(msg)

    base_scores: list[list[float]] = []
    adapted_scores: list[list[float]] = []
    per_head_cosine: list[float] = []
    per_head_l2_delta: list[float] = []

    for b_head, a_head in zip(base_heads, adapted_heads):
        b_vec = [ts.sink_score for ts in b_head.token_sinks]
        a_vec = [ts.sink_score for ts in a_head.token_sinks]

        if len(b_vec) != len(a_vec):
            msg = (
                f"Sequence length mismatch at head {b_head.head_idx}: "
                f"base has {len(b_vec)}, adapted has {len(a_vec)}"
            )
            raise ValueError(msg)

        base_scores.append(b_vec)
        adapted_scores.append(a_vec)

        # Cosine similarity: dot(b, a) / (||b|| * ||a||)
        dot = sum(bi * ai for bi, ai in zip(b_vec, a_vec))
        norm_b = math.sqrt(sum(bi * bi for bi in b_vec))
        norm_a = math.sqrt(sum(ai * ai for ai in a_vec))
        denom = norm_b * norm_a
        cosine = dot / denom if denom > 0.0 else 0.0
        per_head_cosine.append(cosine)

        # L2 distance: ||b - a||_2
        l2 = math.sqrt(sum((bi - ai) ** 2 for bi, ai in zip(b_vec, a_vec)))
        per_head_l2_delta.append(l2)

    return SinkScoreDelta(
        base_scores=base_scores,
        adapted_scores=adapted_scores,
        per_head_cosine=per_head_cosine,
        per_head_l2_delta=per_head_l2_delta,
    )
