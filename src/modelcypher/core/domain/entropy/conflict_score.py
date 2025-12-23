"""
Conflict Score: Distinguishing Adapter Specialization from Fighting the Prior.

Problem: Entropy differential (ΔH) alone cannot distinguish between:
- Specialization (good): Adapter narrows distribution to domain-specific tokens
- Fighting prior (bad): Adapter pushes toward tokens the base model rejected

Solution: Conflict Score = meanKL × (1 - baseApprovalRate)

When baseApprovalRate is high (sampled tokens in base top-K), the adapter is
refining within the base model's comfort zone. When low, the adapter is fighting.

Ported from ConflictScore.swift (342 lines).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger("modelcypher.entropy.conflict_score")


# =============================================================================
# Conflict Score Result
# =============================================================================


@dataclass(frozen=True)
class ConflictScoreResult:
    """
    Result of conflict score computation.
    
    Attributes:
        mean_kl: Mean KL divergence between adapted and base distributions.
        base_approval_rate: Fraction of tokens in base model's top-K [0, 1].
        conflict_score: KL × (1 - approval_rate). High = fighting prior.
        is_conflicting: Whether conflict_score exceeds threshold.
    """
    mean_kl: float
    base_approval_rate: float
    conflict_score: float
    is_conflicting: bool
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.base_approval_rate > 0.8:
            if self.mean_kl < 0.5:
                return "Normal operation: adapter refining within base model's preferences"
            return "Aggressive specialization: high divergence but within approved tokens"
        elif self.base_approval_rate > 0.5:
            return "Moderate drift: adapter occasionally choosing tokens base model deprioritized"
        return "Significant conflict: adapter systematically choosing tokens base model rejected"


# =============================================================================
# Conflict Score Calculator
# =============================================================================


class ConflictScoreCalculator:
    """
    Calculates conflict score between base and adapted model logits.
    
    Usage:
        calculator = ConflictScoreCalculator(top_k=10)
        result = calculator.compute(
            base_logits=base_model_logits,
            adapted_logits=adapter_logits,
            sampled_token=token_id,
        )
        if result.is_conflicting:
            # Adapter is fighting the prior - potential safety concern
    """
    
    def __init__(self, top_k: int = 10, epsilon: float = 1e-10):
        """
        Initialize calculator.
        
        Args:
            top_k: Number of top tokens to consider for base approval.
            epsilon: Numerical stability epsilon.
        """
        self.top_k = top_k
        self.epsilon = epsilon
    
    def compute(
        self,
        base_logits: mx.array,
        adapted_logits: mx.array,
        sampled_token: int,
        conflict_threshold: float = 0.3,
    ) -> ConflictScoreResult:
        """
        Compute conflict metrics for a single token prediction.
        
        Args:
            base_logits: Logits from base model [vocab_size] or [batch, seq, vocab].
            adapted_logits: Logits from adapter-augmented model.
            sampled_token: The token ID that was actually sampled.
            conflict_threshold: Threshold above which conflict is flagged.
        
        Returns:
            ConflictScoreResult with all metrics.
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
            is_conflicting=conflict > conflict_threshold,
        )
    
    def compute_window(
        self,
        base_logits_sequence: List[mx.array],
        adapted_logits_sequence: List[mx.array],
        sampled_tokens: List[int],
        conflict_threshold: float = 0.3,
    ) -> ConflictScoreResult:
        """
        Compute conflict metrics over a window of tokens.
        
        Args:
            base_logits_sequence: Array of logit tensors from base model.
            adapted_logits_sequence: Array of logit tensors from adapted model.
            sampled_tokens: Array of sampled token IDs.
            conflict_threshold: Threshold above which conflict is flagged.
        
        Returns:
            Aggregated ConflictScoreResult.
        """
        if (len(base_logits_sequence) != len(adapted_logits_sequence) or
            len(base_logits_sequence) != len(sampled_tokens) or
            len(base_logits_sequence) == 0):
            return ConflictScoreResult(
                mean_kl=0.0,
                base_approval_rate=1.0,
                conflict_score=0.0,
                is_conflicting=False,
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
            is_conflicting=conflict > conflict_threshold,
        )
    
    def _flatten_to_vocab(self, logits: mx.array) -> mx.array:
        """Flatten logits to 1D vocab vector."""
        if logits.ndim == 3:
            # [batch, seq, vocab] -> last token
            return logits[0, -1, :]
        elif logits.ndim == 2:
            # [batch, vocab] -> first batch
            return logits[0, :]
        return logits
    
    def _compute_kl_divergence(self, p_logits: mx.array, q_logits: mx.array) -> float:
        """
        Compute KL divergence D_KL(p || q) from logits.
        
        Uses numerically stable softmax computation.
        """
        # Stable softmax
        p_max = mx.max(p_logits)
        q_max = mx.max(q_logits)
        
        p_shifted = p_logits - p_max
        q_shifted = q_logits - q_max
        
        p_exp = mx.exp(p_shifted)
        q_exp = mx.exp(q_shifted)
        
        p_sum = mx.sum(p_exp)
        q_sum = mx.sum(q_exp)
        
        p_probs = p_exp / p_sum
        q_probs = q_exp / q_sum
        
        # KL = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
        eps = mx.array(self.epsilon)
        p_log_probs = mx.log(p_probs + eps)
        q_log_probs = mx.log(q_probs + eps)
        
        kl = mx.sum(p_probs * (p_log_probs - q_log_probs))
        
        # Evaluate and extract scalar
        kl_f32 = kl.astype(mx.float32)
        mx.eval(kl_f32)
        
        return max(0.0, float(kl_f32.item()))
    
    def _is_in_top_k(self, logits: mx.array, token_id: int, k: int) -> bool:
        """Check if token_id is in the top-K of logits."""
        if k <= 0:
            return False
        
        vocab_size = logits.shape[0]
        kk = min(k, vocab_size)
        if kk <= 0:
            return False
        
        # Use argpartition for O(n) complexity
        neg_logits = -logits
        top_k_indices = mx.argpartition(neg_logits, kth=kk - 1)[:kk]
        mx.eval(top_k_indices)
        
        indices = top_k_indices.tolist()
        return token_id in indices


# =============================================================================
# Conflict Level and Analysis
# =============================================================================


class ConflictLevel(str, Enum):
    """Coarse interpretation of adapter vs base disagreement."""
    CARVING = "carving"           # Adapter specializes within base's top-K (high agreement)
    MILD_TENSION = "mild_tension" # Adapter sometimes overrides base (moderate agreement)
    FIGHTING = "fighting"         # Adapter frequently contradicts base (low agreement)


@dataclass(frozen=True)
class ConflictAnalysis:
    """
    Aggregated conflict metrics over a generation trace.
    
    Computes overall conflict level from per-token KL divergences
    and base model top-K approval rates.
    """
    mean_kl: float
    base_approval_rate: float
    conflict_score: float
    level: ConflictLevel
    interpretation: str
    
    @staticmethod
    def compute(
        kl_divergences: List[Optional[float]],
        base_approved_top_k: List[Optional[bool]],
    ) -> Optional["ConflictAnalysis"]:
        """
        Compute ConflictAnalysis from token-level metrics.
        
        Args:
            kl_divergences: Per-token D_KL(p_adapter || p_base) values.
            base_approved_top_k: Per-token approval flags (sampled in base top-K).
        
        Returns:
            ConflictAnalysis if enough data, else None.
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
        
        # Determine level
        if approval_rate >= 0.95 and conflict_score < 0.5:
            level = ConflictLevel.CARVING
            interpretation = (
                "Adapter is carving: sampled tokens largely remain within the base model's "
                "top-K; divergence reflects specialization, not contradiction."
            )
        elif approval_rate >= 0.70 and conflict_score < 2.0:
            level = ConflictLevel.MILD_TENSION
            interpretation = (
                "Adapter shows mild tension: sampled tokens sometimes fall outside the base "
                "model's top-K; monitor for drift or mismatched persona."
            )
        else:
            level = ConflictLevel.FIGHTING
            interpretation = (
                "Adapter is fighting: sampled tokens frequently fall outside the base model's "
                "top-K and divergence is high; investigate for misalignment or backdoor behavior."
            )
        
        return ConflictAnalysis(
            mean_kl=mean_kl,
            base_approval_rate=approval_rate,
            conflict_score=conflict_score,
            level=level,
            interpretation=interpretation,
        )

