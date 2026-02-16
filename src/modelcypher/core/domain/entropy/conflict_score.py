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
Conflict Score: Distinguishing Adapter Specialization from Fighting the Prior.

Notes
-----
Entropy differential (ΔH) alone cannot distinguish between:
- Specialization: Adapter concentrates logit mass on domain-specific tokens
- Prior tension: Adapter shifts logit geometry toward tokens outside the base model frontier

Conflict Score = meanKL × (1 - baseFrontierRate)

The base frontier is derived from the largest logit gap, not a fixed top-K.
When frontier rate is high, the adapter stays within the base logit geometry.

Ported from ConflictScore.swift (342 lines).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.entropy.conflict_score")


# =============================================================================
# Conflict Score Result
# =============================================================================


@dataclass(frozen=True)
class ConflictScoreResult:
    """Result of conflict score computation.

    Attributes
    ----------
    mean_kl : float
        Mean KL divergence D_KL(adapter || base). Higher values indicate more divergence.
    base_frontier_rate : float
        Fraction of sampled tokens inside the base logit frontier [0, 1].
    conflict_score : float
        KL × (1 - frontier_rate). Higher values indicate more prior tension.
    """

    mean_kl: float
    base_frontier_rate: float
    conflict_score: float


# =============================================================================
# Conflict Score Calculator
# =============================================================================


class ConflictScoreCalculator:
    """Calculates conflict score between base and adapted model logits."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize calculator."""
        self._backend = backend or get_default_backend()

    def compute(
        self,
        base_logits: "Array",
        adapted_logits: "Array",
        sampled_token: int,
    ) -> ConflictScoreResult:
        """Compute conflict metrics for a single token prediction.

        Args:
            base_logits: Logits from base model [vocab_size] or [batch, seq, vocab].
            adapted_logits: Logits from adapter-augmented model.
            sampled_token: The token ID that was actually sampled.

        Returns:
            ConflictScoreResult with raw measurements.
        """
        # Flatten to 1D
        base_flat = self._flatten_to_vocab(base_logits)
        adapted_flat = self._flatten_to_vocab(adapted_logits)

        # Compute KL divergence: D_KL(adapted || base)
        kl = self._compute_kl_divergence(adapted_flat, base_flat)

        # Check if sampled token was in base model's frontier.
        in_frontier = self._is_in_frontier(base_flat, sampled_token)
        frontier_rate = 1.0 if in_frontier else 0.0

        # Conflict score = KL × (1 - frontier_rate)
        conflict = kl * (1.0 - frontier_rate)

        return ConflictScoreResult(
            mean_kl=kl,
            base_frontier_rate=frontier_rate,
            conflict_score=conflict,
        )

    def compute_window(
        self,
        base_logits_sequence: "list[Array]",
        adapted_logits_sequence: "list[Array]",
        sampled_tokens: list[int],
    ) -> ConflictScoreResult:
        """Compute conflict metrics over a window of tokens.

        Args:
            base_logits_sequence: Array of logit tensors from base model.
            adapted_logits_sequence: Array of logit tensors from adapted model.
            sampled_tokens: Array of sampled token IDs.

        Returns:
            Aggregated ConflictScoreResult with raw measurements.
        """
        if (
            len(base_logits_sequence) != len(adapted_logits_sequence)
            or len(base_logits_sequence) != len(sampled_tokens)
            or len(base_logits_sequence) == 0
        ):
            return ConflictScoreResult(
                mean_kl=0.0,
                base_frontier_rate=1.0,
                conflict_score=0.0,
            )

        total_kl = 0.0
        frontier_count = 0

        for i in range(len(base_logits_sequence)):
            base_flat = self._flatten_to_vocab(base_logits_sequence[i])
            adapted_flat = self._flatten_to_vocab(adapted_logits_sequence[i])

            total_kl += self._compute_kl_divergence(adapted_flat, base_flat)

            if self._is_in_frontier(base_flat, sampled_tokens[i]):
                frontier_count += 1

        mean_kl = total_kl / len(base_logits_sequence)
        frontier_rate = frontier_count / len(sampled_tokens)
        conflict = mean_kl * (1.0 - frontier_rate)

        return ConflictScoreResult(
            mean_kl=mean_kl,
            base_frontier_rate=frontier_rate,
            conflict_score=conflict,
        )

    def _flatten_to_vocab(self, logits: "Array") -> "Array":
        """Flatten logits to 1D vocab vector."""
        if logits.ndim == 3:
            # [batch, seq, vocab] -> last token
            return logits[0, -1, :]
        elif logits.ndim == 2:
            # [batch, vocab] -> first batch
            return logits[0, :]
        return logits

    def _compute_kl_divergence(self, p_logits: "Array", q_logits: "Array") -> float:
        """
        Compute KL divergence D_KL(p || q) from logits.

        Uses numerically stable softmax computation.
        """
        b = self._backend
        # Stable softmax
        p_max = b.max(p_logits)
        q_max = b.max(q_logits)

        p_shifted = p_logits - p_max
        q_shifted = q_logits - q_max

        p_exp = b.exp(p_shifted)
        q_exp = b.exp(q_shifted)

        p_sum = b.sum(p_exp)
        q_sum = b.sum(q_exp)

        p_weights = p_exp / p_sum
        q_weights = q_exp / q_sum

        # KL = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
        # Use separate epsilon for each simplex vector to handle different scales
        eps_p = division_epsilon(b, p_weights)
        eps_q = division_epsilon(b, q_weights)
        p_log_w = b.log(p_weights + eps_p)
        q_log_w = b.log(q_weights + eps_q)

        kl = b.sum(p_weights * (p_log_w - q_log_w))

        # Evaluate and extract scalar
        kl_f32 = b.astype(kl, "float32")
        b.eval(kl_f32)
        return max(0.0, float(b.to_scalar(kl_f32)))

    def _is_in_frontier(self, logits: "Array", token_id: int) -> bool:
        """Check if token_id is inside the base logit frontier."""
        b = self._backend
        logits = self._flatten_to_vocab(logits)
        vocab_size = logits.shape[0]
        if vocab_size <= 1:
            return True

        token_logit = logits[token_id]
        rank_sum = b.sum(b.astype(logits > token_logit, "float32"))
        b.eval(rank_sum)
        token_rank = int(b.to_scalar(rank_sum))
        frontier_size = self._frontier_size(logits)
        return token_rank < frontier_size

    def _frontier_size(self, logits: "Array") -> int:
        """Compute frontier size from the largest logit gap."""
        b = self._backend
        vocab_size = logits.shape[0]
        if vocab_size <= 1:
            return 1

        sorted_logits = -b.sort(-logits)
        gaps = sorted_logits[:-1] - sorted_logits[1:]
        eps = division_epsilon(b, logits)
        max_gap_arr = b.max(gaps)
        argmax_arr = b.argmax(gaps)
        b.eval(max_gap_arr, argmax_arr)
        max_gap = float(b.to_scalar(max_gap_arr))
        if max_gap <= eps:
            return vocab_size
        return int(b.to_scalar(argmax_arr)) + 1


# =============================================================================
# Conflict Analysis
# =============================================================================


@dataclass(frozen=True)
class ConflictAnalysis:
    """Aggregated conflict metrics over a generation trace.

    Attributes
    ----------
    mean_kl : float
        Mean KL divergence across tokens. Higher = more divergent.
    base_frontier_rate : float
        Fraction of tokens in base logit frontier [0, 1]. Higher = more agreement.
    conflict_score : float
        Combined signal: KL × (1 - frontier_rate).
    token_count : int
        Number of tokens analyzed.
    """

    mean_kl: float
    base_frontier_rate: float
    conflict_score: float
    token_count: int

    @staticmethod
    def compute(
        kl_divergences: list[float | None],
        base_frontier_hit: list[bool | None],
    ) -> "ConflictAnalysis" | None:
        """Compute ConflictAnalysis from token-level metrics.

        Parameters
        ----------
            kl_divergences: Per-token D_KL(p_adapter || p_base) values.
            base_frontier_hit: Per-token frontier hit flags.

        Returns:
            ConflictAnalysis with raw measurements if enough data, else None.
        """
        kl_sum = 0.0
        token_count = 0
        frontier_count = 0

        for kl, hit in zip(kl_divergences, base_frontier_hit):
            if kl is None or hit is None:
                continue
            token_count += 1
            kl_sum += float(kl)
            if hit:
                frontier_count += 1

        if token_count == 0:
            return None

        mean_kl = kl_sum / token_count
        frontier_rate = frontier_count / token_count
        conflict_score = mean_kl * (1.0 - frontier_rate)

        return ConflictAnalysis(
            mean_kl=mean_kl,
            base_frontier_rate=frontier_rate,
            conflict_score=conflict_score,
            token_count=token_count,
        )
