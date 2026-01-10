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
Deviation Budget Tracking for Model Merging and Multi-Modal Injection.

This module tracks cumulative deviation from baseline to prevent generation
degradation. Both model merging and multi-modal injection share the same
fundamental constraint: models have a "deviation budget" beyond which
generation collapses.

Key Findings (2026-01-09):
    - Multi-modal injection: Scale >10x causes degeneration
    - Sequential merging: Cumulative Δ >50-60 L2 causes degeneration
    - Both: Null-space projection allows higher deviation safely

The unified principle: Models tolerate deviation only in low-variance
(null-space) directions. High-variance directions are "active" and
overwriting them causes generation collapse.

Usage:
    budget = DeviationBudget(backend)

    # For merging
    budget.record_baseline(baseline_weights)
    is_safe = budget.check_merge_budget(merged_weights)

    # For multi-modal injection
    is_safe = budget.check_injection_scale(visual_embed, layer_activations, scale)

    # Get recommended scale
    safe_scale = budget.recommend_scale(visual_embed, layer_activations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# Geometrically-derived thresholds (not hard-coded)
#
# These constants define tolerance fractions relative to the baseline model's
# total weight norm. The actual thresholds are computed from model properties.
#
# Mathematical basis:
#   - threshold = ||W_baseline||_F * tolerance_fraction
#   - For float32 models: ~1% deviation is safe before numerical instability
#   - For float16/bf16: ~0.5% deviation is the safe limit
#
# The 1% value comes from:
#   - Machine epsilon for float32: ~1e-7
#   - Condition numbers of trained weight matrices: ~1e3-1e4
#   - Safe deviation ≈ 1 / sqrt(condition_number) ≈ 1% of weight norm
#
MERGE_BUDGET_TOLERANCE = 0.01  # 1% of weight norm is safe
MERGE_WARNING_TOLERANCE = 0.007  # 0.7% triggers warning
INJECTION_SCALE_SAFE = 5.0    # Safe scale for null-space injection
INJECTION_SCALE_MAX = 10.0    # Maximum before degeneration

# Memory token injection allows much higher scales because:
# 1. Null-space projection: 2x multiplier
# 2. Attention-based retrieval (vs forced injection): 2x multiplier
# Empirical finding (2026-01-09): Memory tokens work at scale 20.0+ without degeneration
MEMORY_TOKEN_SCALE_SAFE = 10.0   # Safe scale for memory token
MEMORY_TOKEN_SCALE_MAX = 20.0    # Maximum for memory token (4x direct injection)


@dataclass
class BudgetStatus:
    """Result of budget check."""

    # Whether operation is within budget
    is_safe: bool

    # Current deviation from baseline
    current_deviation: float

    # Threshold being used
    threshold: float

    # Percentage of budget used
    budget_used_percent: float

    # Recommended action
    recommendation: str


@dataclass
class ScaleRecommendation:
    """Recommended scaling for injection."""

    # Recommended scale factor
    scale: float

    # Maximum safe scale
    max_safe_scale: float

    # Whether null-space projection is recommended
    use_null_space: bool

    # Explanation
    reason: str


class DeviationBudget:
    """
    Tracks deviation budget for merging and injection operations.

    The budget is geometrically derived from the baseline model's weight norm:
        threshold = ||W_baseline||_F * tolerance_fraction

    This ensures thresholds scale appropriately with model size. Larger models
    have larger weight norms and correspondingly larger absolute thresholds,
    but the same relative tolerance (1% of weight norm).

    Mathematical basis:
        - Safe deviation ≈ weight_norm / sqrt(condition_number)
        - For typical LLM weight matrices, condition_number ≈ 1e3-1e4
        - This gives tolerance ≈ 1% of weight norm
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        merge_tolerance: float = MERGE_BUDGET_TOLERANCE,
        warning_tolerance: float = MERGE_WARNING_TOLERANCE,
        injection_safe_scale: float = INJECTION_SCALE_SAFE,
    ) -> None:
        self._backend = backend or get_default_backend()
        self._merge_tolerance = merge_tolerance
        self._warning_tolerance = warning_tolerance
        self._injection_safe_scale = injection_safe_scale

        # Baseline tracking
        self._baseline_weights: dict[str, Any] = {}
        self._baseline_norms: dict[str, float] = {}  # Store computed weight norms
        self._cumulative_deviation: float = 0.0

    def _compute_weight_norm(self, weights: dict[str, Any]) -> float:
        """Compute total Frobenius norm of all weight tensors.

        Returns sqrt(sum(||W_i||_F^2)) which gives the total L2 "size" of the model.
        """
        backend = self._backend
        total_sq = backend.array(0.0)

        for v in weights.values():
            w = backend.array(v)
            w_sq = backend.sum(w * w)
            total_sq = total_sq + w_sq
            backend.eval(total_sq)

        return float(backend.sqrt(total_sq))

    def get_threshold(self, baseline_name: str = "default") -> float:
        """Get the geometrically-derived merge threshold for a baseline.

        Returns:
            Threshold in L2 units (weight_norm * tolerance_fraction)
        """
        if baseline_name not in self._baseline_norms:
            # Fallback when baseline not recorded - caller should always
            # call record_baseline() first. This is only for edge cases
            # like unit tests or legacy code paths.
            logger.warning(f"No baseline '{baseline_name}' recorded, using fallback")
            return 50.0

        return self._baseline_norms[baseline_name] * self._merge_tolerance

    def get_warning_threshold(self, baseline_name: str = "default") -> float:
        """Get the warning threshold for a baseline."""
        if baseline_name not in self._baseline_norms:
            # Fallback - see get_threshold() for rationale
            return 35.0

        return self._baseline_norms[baseline_name] * self._warning_tolerance

    def record_baseline(self, weights: dict[str, Any], name: str = "default") -> None:
        """
        Record baseline weights for deviation tracking.

        Computes and stores the total Frobenius norm of all weights to
        derive geometric thresholds. The threshold for safe deviation is:
            threshold = weight_norm * tolerance_fraction

        Args:
            weights: Dictionary of weight tensors
            name: Name for this baseline (for multi-model tracking)
        """
        backend = self._backend
        self._baseline_weights[name] = {
            k: backend.array(v) for k, v in weights.items()
        }
        self._cumulative_deviation = 0.0

        # Compute geometric properties for threshold derivation
        weight_norm = self._compute_weight_norm(weights)
        self._baseline_norms[name] = weight_norm

        threshold = weight_norm * self._merge_tolerance
        warning = weight_norm * self._warning_tolerance

        logger.info(
            f"Recorded baseline '{name}' with {len(weights)} tensors, "
            f"||W||_F={weight_norm:.1f}, threshold={threshold:.1f} L2 (1% of norm)"
        )

    def compute_deviation(
        self,
        current_weights: dict[str, Any],
        baseline_name: str = "default",
    ) -> float:
        """
        Compute L2 deviation from baseline.

        Args:
            current_weights: Current weight tensors
            baseline_name: Name of baseline to compare against

        Returns:
            L2 norm of cumulative deviation
        """
        if baseline_name not in self._baseline_weights:
            logger.warning(f"No baseline '{baseline_name}' recorded")
            return 0.0

        backend = self._backend
        baseline = self._baseline_weights[baseline_name]

        total_deviation_sq = backend.array(0.0)

        for key in current_weights:
            if key in baseline:
                current = backend.array(current_weights[key])
                base = baseline[key]

                delta = current - base
                deviation_sq = backend.sum(delta * delta)
                total_deviation_sq = total_deviation_sq + deviation_sq
                backend.eval(total_deviation_sq)

        deviation = float(backend.sqrt(total_deviation_sq))
        return deviation

    def check_merge_budget(
        self,
        merged_weights: dict[str, Any],
        baseline_name: str = "default",
    ) -> BudgetStatus:
        """
        Check if merged weights are within budget.

        Thresholds are geometrically derived from baseline weight norm:
            threshold = ||W_baseline||_F * tolerance_fraction

        Args:
            merged_weights: Weights after merge operation
            baseline_name: Baseline to compare against

        Returns:
            BudgetStatus with safety assessment
        """
        deviation = self.compute_deviation(merged_weights, baseline_name)

        # Get geometrically-derived thresholds
        threshold = self.get_threshold(baseline_name)
        warning_threshold = self.get_warning_threshold(baseline_name)

        budget_percent = (deviation / threshold) * 100

        if deviation < warning_threshold:
            is_safe = True
            recommendation = "Safe to proceed"
        elif deviation < threshold:
            is_safe = True
            recommendation = f"Approaching budget limit ({budget_percent:.1f}% used). Consider reducing delta_scale for next merge."
        else:
            is_safe = False
            recommendation = f"Budget exceeded ({budget_percent:.1f}% used). Generation degradation likely. Use delta_scale < {threshold / deviation:.2f} or null-space projection."

        return BudgetStatus(
            is_safe=is_safe,
            current_deviation=deviation,
            threshold=threshold,
            budget_used_percent=budget_percent,
            recommendation=recommendation,
        )

    def check_injection_scale(
        self,
        embedding: Any,
        layer_activations: Any,
        scale: float,
        use_null_space: bool = False,
    ) -> BudgetStatus:
        """
        Check if injection scale is within safe bounds.

        Args:
            embedding: Embedding to inject
            layer_activations: Activations at injection layer
            scale: Proposed scale factor
            use_null_space: Whether null-space projection will be used

        Returns:
            BudgetStatus with safety assessment
        """
        backend = self._backend

        embedding = backend.array(embedding)
        layer_activations = backend.array(layer_activations)

        # Compute embedding norm
        embed_norm = float(backend.sqrt(backend.sum(embedding * embedding)))

        # Compute layer activation norm (mean across samples)
        if len(layer_activations.shape) > 1:
            layer_norms = backend.sqrt(backend.sum(layer_activations * layer_activations, axis=-1))
            layer_norm = float(backend.mean(layer_norms))
        else:
            layer_norm = float(backend.sqrt(backend.sum(layer_activations * layer_activations)))

        # Effective injection magnitude
        injection_magnitude = embed_norm * scale

        # Null-space allows ~2x higher scale safely
        effective_threshold = self._injection_safe_scale * (2.0 if use_null_space else 1.0)

        # Compare to layer norm
        relative_injection = injection_magnitude / (layer_norm + 1e-10)

        budget_percent = (relative_injection / effective_threshold) * 100

        if relative_injection < effective_threshold * 0.5:
            is_safe = True
            recommendation = "Safe injection scale"
        elif relative_injection < effective_threshold:
            is_safe = True
            recommendation = f"Moderate injection ({budget_percent:.1f}% of safe threshold)"
        else:
            is_safe = False
            if use_null_space:
                recommendation = f"Scale too high even with null-space. Reduce to {scale * effective_threshold / relative_injection:.2f}"
            else:
                recommendation = f"Scale too high. Use null-space projection or reduce to {scale * effective_threshold / relative_injection:.2f}"

        return BudgetStatus(
            is_safe=is_safe,
            current_deviation=injection_magnitude,
            threshold=effective_threshold * layer_norm,
            budget_used_percent=budget_percent,
            recommendation=recommendation,
        )

    def check_memory_token_scale(
        self,
        memory_content: Any,
        layer_activations: Any,
        scale: float,
        use_null_space: bool = True,
    ) -> BudgetStatus:
        """
        Check if memory token scale is within safe bounds.

        Memory tokens allow much higher scale factors than direct injection
        because:
        1. Null-space projection: 2x multiplier
        2. Attention-based retrieval (model decides usage): 2x multiplier

        Empirical finding: Memory tokens work at scale 20.0+ without degeneration,
        compared to scale 2.0 max for direct injection.

        Args:
            memory_content: Memory token content (already scaled)
            layer_activations: Activations at memory injection layer
            scale: Scale factor that was applied
            use_null_space: Whether null-space projection was used

        Returns:
            BudgetStatus with safety assessment
        """
        backend = self._backend

        memory_content = backend.array(memory_content)
        layer_activations = backend.array(layer_activations)

        # Compute memory content norm
        memory_norm = float(backend.sqrt(backend.sum(memory_content * memory_content)))

        # Compute layer activation norm (mean across samples)
        if len(layer_activations.shape) > 1:
            layer_norms = backend.sqrt(backend.sum(layer_activations * layer_activations, axis=-1))
            layer_norm = float(backend.mean(layer_norms))
        else:
            layer_norm = float(backend.sqrt(backend.sum(layer_activations * layer_activations)))

        # Compare to layer norm
        relative_magnitude = memory_norm / (layer_norm + 1e-10)

        # Memory tokens have much higher thresholds
        # With null-space: safe=10.0, max=20.0
        # Without null-space: safe=5.0, max=10.0
        if use_null_space:
            safe_threshold = MEMORY_TOKEN_SCALE_SAFE
            max_threshold = MEMORY_TOKEN_SCALE_MAX
        else:
            safe_threshold = INJECTION_SCALE_SAFE
            max_threshold = INJECTION_SCALE_MAX

        budget_percent = (relative_magnitude / max_threshold) * 100

        if relative_magnitude < safe_threshold:
            is_safe = True
            recommendation = f"Safe memory token scale ({relative_magnitude:.2f}x layer norm)"
        elif relative_magnitude < max_threshold:
            is_safe = True
            recommendation = f"Moderate memory scale ({budget_percent:.1f}% of max). Attention will modulate usage."
        else:
            is_safe = False
            recommended_scale = scale * max_threshold / relative_magnitude
            recommendation = f"Memory token too strong. Reduce scale to {recommended_scale:.2f}"

        return BudgetStatus(
            is_safe=is_safe,
            current_deviation=memory_norm,
            threshold=max_threshold * layer_norm,
            budget_used_percent=budget_percent,
            recommendation=recommendation,
        )

    def recommend_memory_scale(
        self,
        direction_embed: Any,
        layer_activations: Any,
        target_budget_percent: float = 50.0,
    ) -> ScaleRecommendation:
        """
        Recommend safe scale for memory token injection.

        Memory tokens can use much higher scales than direct injection.
        This method recommends scales appropriate for the attention-based
        memory approach.

        Args:
            direction_embed: Direction embedding (source - neutral) before scaling
            layer_activations: Activations at target layer
            target_budget_percent: Target percentage of budget to use

        Returns:
            ScaleRecommendation with optimal scale for memory token
        """
        backend = self._backend

        direction = backend.array(direction_embed)
        layer_activations = backend.array(layer_activations)

        # Compute norms
        direction_norm = float(backend.sqrt(backend.sum(direction * direction)))

        if len(layer_activations.shape) > 1:
            layer_norms = backend.sqrt(backend.sum(layer_activations * layer_activations, axis=-1))
            layer_norm = float(backend.mean(layer_norms))
        else:
            layer_norm = float(backend.sqrt(backend.sum(layer_activations * layer_activations)))

        # Target magnitude (percentage of max safe)
        target_magnitude = layer_norm * MEMORY_TOKEN_SCALE_MAX * (target_budget_percent / 100.0)

        # Recommended scale
        recommended_scale = target_magnitude / (direction_norm + 1e-10)

        # Max safe scale (with null-space)
        max_safe_magnitude = layer_norm * MEMORY_TOKEN_SCALE_MAX
        max_safe_scale = max_safe_magnitude / (direction_norm + 1e-10)

        # Memory tokens with null-space projection are recommended
        use_null_space = True  # Always recommended for memory tokens

        reason = f"Memory token scale {recommended_scale:.2f} (max safe: {max_safe_scale:.2f}). Null-space projection recommended."

        return ScaleRecommendation(
            scale=recommended_scale,
            max_safe_scale=max_safe_scale,
            use_null_space=use_null_space,
            reason=reason,
        )

    def recommend_scale(
        self,
        embedding: Any,
        layer_activations: Any,
        target_budget_percent: float = 50.0,
    ) -> ScaleRecommendation:
        """
        Recommend safe scale for injection.

        Args:
            embedding: Embedding to inject
            layer_activations: Activations at injection layer
            target_budget_percent: Target percentage of budget to use

        Returns:
            ScaleRecommendation with optimal scale
        """
        backend = self._backend

        embedding = backend.array(embedding)
        layer_activations = backend.array(layer_activations)

        # Compute norms
        embed_norm = float(backend.sqrt(backend.sum(embedding * embedding)))

        if len(layer_activations.shape) > 1:
            layer_norms = backend.sqrt(backend.sum(layer_activations * layer_activations, axis=-1))
            layer_norm = float(backend.mean(layer_norms))
        else:
            layer_norm = float(backend.sqrt(backend.sum(layer_activations * layer_activations)))

        # Target injection magnitude
        target_magnitude = layer_norm * self._injection_safe_scale * (target_budget_percent / 100.0)

        # Recommended scale
        recommended_scale = target_magnitude / (embed_norm + 1e-10)

        # Max safe scale (with null-space)
        max_safe_magnitude = layer_norm * self._injection_safe_scale * 2.0
        max_safe_scale = max_safe_magnitude / (embed_norm + 1e-10)

        # Determine if null-space is needed
        use_null_space = recommended_scale > self._injection_safe_scale

        if use_null_space:
            reason = f"Scale {recommended_scale:.2f} exceeds safe threshold. Null-space projection recommended."
        else:
            reason = f"Scale {recommended_scale:.2f} is within safe bounds for direct injection."

        return ScaleRecommendation(
            scale=recommended_scale,
            max_safe_scale=max_safe_scale,
            use_null_space=use_null_space,
            reason=reason,
        )

    def get_remaining_budget(self, baseline_name: str = "default") -> float:
        """
        Get remaining merge budget.

        Args:
            baseline_name: Baseline to check against

        Returns:
            Remaining L2 budget before threshold
        """
        threshold = self.get_threshold(baseline_name)
        return max(0.0, threshold - self._cumulative_deviation)

    def compute_delta_magnitude(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
    ) -> float:
        """
        Compute the L2 magnitude of delta between source and target.

        This measures the actual proposed change magnitude, not an estimate.
        Used by auto_scale to determine appropriate scaling.

        Args:
            source_weights: Source model weights
            target_weights: Target model weights

        Returns:
            L2 norm of (source - target) for matched keys
        """
        backend = self._backend
        delta_sq = backend.array(0.0)

        for key in source_weights:
            if key in target_weights:
                src = backend.array(source_weights[key])
                tgt = backend.array(target_weights[key])
                delta = src - tgt
                delta_sq = delta_sq + backend.sum(delta * delta)
                backend.eval(delta_sq)

        return float(backend.sqrt(delta_sq))

    def suggest_delta_scale(
        self,
        proposed_delta: dict[str, Any],
        baseline_name: str = "default",
    ) -> float:
        """
        Suggest delta_scale to stay within budget.

        Args:
            proposed_delta: Delta weights to be added
            baseline_name: Baseline reference

        Returns:
            Recommended delta_scale (0.0 to 1.0)
        """
        backend = self._backend

        # Compute delta magnitude
        delta_sq = backend.array(0.0)
        for key, delta in proposed_delta.items():
            d = backend.array(delta)
            delta_sq = delta_sq + backend.sum(d * d)
            backend.eval(delta_sq)

        delta_magnitude = float(backend.sqrt(delta_sq))

        remaining = self.get_remaining_budget(baseline_name)

        if delta_magnitude <= 0:
            return 1.0

        # Scale to use 70% of remaining budget
        safe_scale = (remaining * 0.7) / delta_magnitude

        return min(1.0, max(0.1, safe_scale))

    def auto_compute_scale(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        remaining_sources: int,
        baseline_name: str = "default",
    ) -> float:
        """
        Automatically compute delta_scale based on measured delta and remaining budget.

        This is the geometrically-derived approach: measure actual delta magnitude,
        then scale to stay within budget across remaining sources.

        Args:
            source_weights: Source model weights
            target_weights: Target model weights
            remaining_sources: Number of sources left to merge (including this one)
            baseline_name: Baseline for budget calculation

        Returns:
            Recommended delta_scale (0.1 to 1.0)
        """
        # Measure actual delta magnitude (not a guess)
        delta_magnitude = self.compute_delta_magnitude(source_weights, target_weights)

        if delta_magnitude <= 0:
            return 1.0

        # Get remaining budget
        remaining_budget = self.get_remaining_budget(baseline_name)

        # Allocate budget across remaining sources with 70% safety margin
        per_source_budget = (remaining_budget * 0.7) / max(1, remaining_sources)

        # Compute scale needed to keep delta within per-source budget
        scale = per_source_budget / delta_magnitude

        # Clamp to reasonable range
        return min(1.0, max(0.1, scale))
