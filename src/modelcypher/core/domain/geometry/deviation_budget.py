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


# Empirically derived thresholds from experiments (2026-01-09)
MERGE_BUDGET_THRESHOLD = 50.0  # L2 norm cumulative delta
MERGE_BUDGET_WARNING = 35.0   # Warning threshold
INJECTION_SCALE_SAFE = 5.0    # Safe scale for null-space injection
INJECTION_SCALE_MAX = 10.0    # Maximum before degeneration


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

    The budget is based on the empirical finding that models collapse
    when cumulative deviation exceeds ~50-60 L2 norm from baseline.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        merge_threshold: float = MERGE_BUDGET_THRESHOLD,
        injection_safe_scale: float = INJECTION_SCALE_SAFE,
    ) -> None:
        self._backend = backend or get_default_backend()
        self._merge_threshold = merge_threshold
        self._injection_safe_scale = injection_safe_scale

        # Baseline tracking
        self._baseline_weights: dict[str, Any] = {}
        self._cumulative_deviation: float = 0.0

    def record_baseline(self, weights: dict[str, Any], name: str = "default") -> None:
        """
        Record baseline weights for deviation tracking.

        Args:
            weights: Dictionary of weight tensors
            name: Name for this baseline (for multi-model tracking)
        """
        backend = self._backend
        self._baseline_weights[name] = {
            k: backend.array(v) for k, v in weights.items()
        }
        self._cumulative_deviation = 0.0
        logger.info(f"Recorded baseline '{name}' with {len(weights)} tensors")

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

        Args:
            merged_weights: Weights after merge operation
            baseline_name: Baseline to compare against

        Returns:
            BudgetStatus with safety assessment
        """
        deviation = self.compute_deviation(merged_weights, baseline_name)
        budget_percent = (deviation / self._merge_threshold) * 100

        if deviation < MERGE_BUDGET_WARNING:
            is_safe = True
            recommendation = "Safe to proceed"
        elif deviation < self._merge_threshold:
            is_safe = True
            recommendation = f"Approaching budget limit ({budget_percent:.1f}% used). Consider reducing delta_scale for next merge."
        else:
            is_safe = False
            recommendation = f"Budget exceeded ({budget_percent:.1f}% used). Generation degradation likely. Use delta_scale < {self._merge_threshold / deviation:.2f} or null-space projection."

        return BudgetStatus(
            is_safe=is_safe,
            current_deviation=deviation,
            threshold=self._merge_threshold,
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
        return max(0.0, self._merge_threshold - self._cumulative_deviation)

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
