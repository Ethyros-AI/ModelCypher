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
- Specialization (good): Adapter narrows distribution to domain-specific tokens
- Fighting prior (bad): Adapter pushes toward tokens the base model rejected

Conflict Score = meanKL × (1 - baseApprovalRate)

When baseApprovalRate is high (sampled tokens in base top-K), the adapter is
refining within the base model's comfort zone. When low, the adapter is fighting.

Ported from ConflictScore.swift (342 lines).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.entropy.conflict_score")


# =============================================================================
# Conflict Score Result
# =============================================================================


@dataclass(frozen=True)
class ConflictScoreResult:
    """
    Result of conflict score computation.

    Returns raw geometric measurements. The conflict_score IS the conflict state -
    no need for CARVING/MILD_TENSION/FIGHTING categories that destroy information.

    Attributes:
        mean_kl: Mean KL divergence between adapted and base distributions.
        base_approval_rate: Fraction of tokens in base model's top-K [0, 1].
        conflict_score: KL × (1 - approval_rate). High = fighting prior.
    """

    mean_kl: float
    """Mean KL divergence D_KL(adapter || base). Higher = more divergent."""

    base_approval_rate: float
    """Fraction of sampled tokens in base model's top-K [0, 1]. Higher = more agreement."""

    conflict_score: float
    """KL × (1 - approval_rate). The measurement IS the conflict state."""

    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if conflict_score exceeds a given threshold.

        This replaces the removed is_conflicting field. Callers must
        explicitly provide thresholds - no arbitrary defaults.
        """
        return self.conflict_score > threshold


# =============================================================================
# Conflict Score Calculator
# =============================================================================


class ConflictScoreCalculator:
    """
    Calculates conflict score between base and adapted model logits.

    Returns raw measurements. The conflict_score IS the conflict state.

    Usage:
        calculator = ConflictScoreCalculator(top_k=10)
        result = calculator.compute(
            base_logits=base_model_logits,
            adapted_logits=adapter_logits,
            sampled_token=token_id,
        )
        # Use raw conflict_score or check against explicit threshold
        if result.exceeds_threshold(0.3):  # Caller decides threshold
            # Adapter may be fighting the prior - potential safety concern
    """

    def __init__(
        self,
        top_k: int = 10,
        epsilon: float = 1e-10,
        backend: "Backend | None" = None,
    ) -> None:
        """
        Initialize calculator.

        Args:
            top_k: Number of top tokens to consider for base approval.
            epsilon: Numerical stability epsilon.
            backend: Compute backend (defaults to MLX on macOS).
        """
        self.top_k = top_k
        self.epsilon = epsilon
        self._backend = backend or get_default_backend()

    def compute(
        self,
        base_logits: "Array",
        adapted_logits: "Array",
        sampled_token: int,
    ) -> ConflictScoreResult:
        """
        Compute conflict metrics for a single token prediction.

        Returns raw measurements. The conflict_score IS the conflict state.
        Use result.exceeds_threshold(t) to check against a specific threshold.

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

        # Check if sampled token was in base model's top-K
        was_approved = self._is_in_top_k(base_flat, sampled_token, self.top_k)
        approval_rate = 1.0 if was_approved else 0.0

        # Conflict score = KL × (1 - approval)
        conflict = kl * (1.0 - approval_rate)

        return ConflictScoreResult(
            mean_kl=kl,
            base_approval_rate=approval_rate,
            conflict_score=conflict,
        )

    def compute_window(
        self,
        base_logits_sequence: "list[Array]",
        adapted_logits_sequence: "list[Array]",
        sampled_tokens: list[int],
    ) -> ConflictScoreResult:
        """
        Compute conflict metrics over a window of tokens.

        Returns raw measurements. The conflict_score IS the conflict state.
        Use result.exceeds_threshold(t) to check against a specific threshold.

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
                base_approval_rate=1.0,
                conflict_score=0.0,
            )

        total_kl = 0.0
        approved_count = 0

        for i in range(len(base_logits_sequence)):
            base_flat = self._flatten_to_vocab(base_logits_sequence[i])
            adapted_flat = self._flatten_to_vocab(adapted_logits_sequence[i])

            total_kl += self._compute_kl_divergence(adapted_flat, base_flat)

            if self._is_in_top_k(base_flat, sampled_tokens[i], self.top_k):
                approved_count += 1

        mean_kl = total_kl / len(base_logits_sequence)
        approval_rate = approved_count / len(sampled_tokens)
        conflict = mean_kl * (1.0 - approval_rate)

        return ConflictScoreResult(
            mean_kl=mean_kl,
            base_approval_rate=approval_rate,
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

        p_probs = p_exp / p_sum
        q_probs = q_exp / q_sum

        # KL = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
        eps = b.array(self.epsilon)
        p_log_probs = b.log(p_probs + eps)
        q_log_probs = b.log(q_probs + eps)

        kl = b.sum(p_probs * (p_log_probs - q_log_probs))

        # Evaluate and extract scalar
        kl_f32 = b.astype(kl, "float32")
        b.eval(kl_f32)

        kl_np = b.to_numpy(kl_f32)
        return max(0.0, float(kl_np.item()))

    def _is_in_top_k(self, logits: "Array", token_id: int, k: int) -> bool:
        """Check if token_id is in the top-K of logits."""
        if k <= 0:
            return False

        vocab_size = logits.shape[0]
        kk = min(k, vocab_size)
        if kk <= 0:
            return False

        b = self._backend
        # Use argpartition for O(n) complexity
        neg_logits = -logits
        top_k_indices = b.argpartition(neg_logits, kth=kk - 1)[:kk]
        b.eval(top_k_indices)

        indices = b.to_numpy(top_k_indices).tolist()
        return token_id in indices


# =============================================================================
# Conflict Analysis
# =============================================================================


@dataclass(frozen=True)
class ConflictAnalysis:
    """
    Aggregated conflict metrics over a generation trace.

    Returns raw geometric measurements. The conflict_score IS the conflict state -
    no need for CARVING/MILD_TENSION/FIGHTING categories that destroy information.
    """

    mean_kl: float
    """Mean KL divergence across tokens. Higher = more divergent."""

    base_approval_rate: float
    """Fraction of tokens in base model's top-K [0, 1]. Higher = more agreement."""

    conflict_score: float
    """KL × (1 - approval_rate). The measurement IS the conflict state."""

    token_count: int
    """Number of tokens analyzed."""

    @staticmethod
    def compute(
        kl_divergences: list[float | None],
        base_approved_top_k: list[bool | None],
    ) -> "ConflictAnalysis" | None:
        """
        Compute ConflictAnalysis from token-level metrics.

        Returns raw measurements. The conflict_score IS the conflict state.

        Args:
            kl_divergences: Per-token D_KL(p_adapter || p_base) values.
            base_approved_top_k: Per-token approval flags (sampled in base top-K).

        Returns:
            ConflictAnalysis with raw measurements if enough data, else None.
        """
        kl_sum = 0.0
        token_count = 0
        approved_count = 0

        for kl, approved in zip(kl_divergences, base_approved_top_k):
            if kl is None or approved is None:
                continue
            token_count += 1
            kl_sum += float(kl)
            if approved:
                approved_count += 1

        if token_count == 0:
            return None

        mean_kl = kl_sum / token_count
        approval_rate = approved_count / token_count
        conflict_score = mean_kl * (1.0 - approval_rate)

        return ConflictAnalysis(
            mean_kl=mean_kl,
            base_approval_rate=approval_rate,
            conflict_score=conflict_score,
            token_count=token_count,
        )

    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if conflict_score exceeds a given threshold.

        Callers must explicitly provide thresholds - no arbitrary defaults.
        """
        return self.conflict_score > threshold
