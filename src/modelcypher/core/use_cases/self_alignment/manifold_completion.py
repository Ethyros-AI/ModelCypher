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

"""Manifold completion tracking and criteria.

A model's manifold is "complete" when:
1. SVD ratios locked to fundamental constants (within machine precision)
2. Complexity-dimension law validated at <1% error
3. Constant alignment saturated across ALL layers
4. No improving directions exist at ANY scale (geometric saturation)

This module tracks layer-by-layer completion and determines when
the full manifold has reached its geometric potential.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.fundamental_constants import (
    FundamentalConstant,
    find_constant_match,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
    from modelcypher.core.domain.geometry.manifold_entropy import (
        ManifoldEntropyResult,
        SVDSignatureResult,
    )


class CompletionLevel(Enum):
    """Level of geometric completion for a layer or manifold."""

    INCOMPLETE = "incomplete"  # Still improving
    PARTIAL = "partial"  # Some constants aligned, still room
    SATURATED = "saturated"  # No more improvement possible at this layer
    COMPLETE = "complete"  # Geometrically complete (within machine precision)


@dataclass
class LayerCompletion:
    """Completion status for a single layer."""

    layer_idx: int

    # How many SVD ratio matches fall within machine-precision error
    n_aligned_ratios: int = 0

    # How many could theoretically align (based on matrix rank)
    n_possible_ratios: int = 0

    # Best alignment error seen (lower is better)
    best_alignment_quality: float = float("inf")

    # Current alignment error (lower is better)
    current_alignment_quality: float = float("inf")

    # Rounds since improvement
    rounds_without_improvement: int = 0

    # Completion level
    level: CompletionLevel = CompletionLevel.INCOMPLETE

    # Which constants are represented
    constants_present: Set[FundamentalConstant] = field(default_factory=set)

    # Entropy at this layer
    current_entropy: float = float("inf")
    best_entropy: float = float("inf")


@dataclass
class ManifoldCompletion:
    """Completion status for the full manifold."""

    # Per-layer completion
    layer_completions: Dict[int, LayerCompletion] = field(default_factory=dict)

    # Overall completion metrics
    total_aligned_ratios: int = 0
    total_possible_ratios: int = 0

    # Percentage of layers at each level
    n_incomplete: int = 0
    n_partial: int = 0
    n_saturated: int = 0
    n_complete: int = 0

    # Global completion level
    level: CompletionLevel = CompletionLevel.INCOMPLETE

    # Constants represented across all layers
    constants_coverage: Dict[FundamentalConstant, int] = field(default_factory=dict)

    # Complexity-dimension law status
    complexity_law_error: float = float("inf")  # % error from e/π slope
    complexity_law_validated: bool = False  # Deprecated: derived from machine epsilon


class ManifoldCompletionTracker:
    """Track progress toward manifold completion.

    Monitors layer-by-layer geometric completion and determines
    when the model has "filled its space" - reached geometric saturation.

    Usage:
        tracker = ManifoldCompletionTracker(backend)

        for round in range(max_rounds):
            entropy_result = measure_entropy(...)
            completion = tracker.update(entropy_result)

            if tracker.is_complete():
                break

            # Print progress
            print(tracker.get_progress_report())
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        saturation_patience: int = 10,  # Rounds without improvement = saturated
    ) -> None:
        """Initialize completion tracker.

        Args:
            backend: Computational backend
            saturation_patience: Rounds without improvement before saturated
        """
        self._backend = backend or get_default_backend()
        self._saturation_patience = saturation_patience
        ref = self._backend.array([1.0], dtype="float32")
        eps = machine_epsilon(self._backend, ref)
        self._percent_epsilon = sqrt_scalar(eps, self._backend) * 100.0

        self._completion = ManifoldCompletion()
        self._round_idx = 0

    @property
    def completion(self) -> ManifoldCompletion:
        """Get current completion status."""
        return self._completion

    def update(
        self,
        entropy_result: "ManifoldEntropyResult",
        svd_results: Optional[Dict[int, "SVDSignatureResult"]] = None,
    ) -> ManifoldCompletion:
        """Update completion status based on new measurements.

        Args:
            entropy_result: Latest manifold entropy measurement
            svd_results: Optional per-layer SVD signatures

        Returns:
            Updated ManifoldCompletion
        """
        self._round_idx += 1

        # Update layer completions
        for layer_idx, layer_entropy_result in entropy_result.layer_entropies.items():
            if layer_idx not in self._completion.layer_completions:
                self._completion.layer_completions[layer_idx] = LayerCompletion(
                    layer_idx=layer_idx
                )

            layer_comp = self._completion.layer_completions[layer_idx]

            # Extract entropy value from LayerEntropyResult
            layer_entropy = layer_entropy_result.spectral_entropy

            # Update entropy
            if layer_entropy < layer_comp.best_entropy:
                layer_comp.best_entropy = layer_entropy
                layer_comp.rounds_without_improvement = 0
            else:
                layer_comp.rounds_without_improvement += 1

            layer_comp.current_entropy = layer_entropy

        # Update alignment from SVD signatures
        if svd_results:
            self._update_from_svd(svd_results)
        elif entropy_result.svd_signature:
            # Use global SVD signature
            self._update_alignment_from_global(entropy_result)

        # Update global metrics
        self._update_global_metrics(entropy_result)

        # Determine completion levels
        self._determine_completion_levels()

        return self._completion

    def _update_from_svd(self, svd_results: Dict[int, "SVDSignatureResult"]) -> None:
        """Update completion from per-layer SVD signatures."""
        for layer_idx, svd in svd_results.items():
            if layer_idx not in self._completion.layer_completions:
                self._completion.layer_completions[layer_idx] = LayerCompletion(
                    layer_idx=layer_idx
                )

            layer_comp = self._completion.layer_completions[layer_idx]

            # Count aligned ratios
            # SVDSignatureResult.matches is List[Tuple[int, int, ConstantMatch]]
            n_aligned = 0
            constants_present: Set[FundamentalConstant] = set()

            for i, j, match in svd.matches:
                if match.error_percent <= self._percent_epsilon:
                    n_aligned += 1
                constants_present.add(match.constant)

            layer_comp.n_aligned_ratios = n_aligned
            layer_comp.constants_present = constants_present
            layer_comp.current_alignment_quality = svd.mean_error

            if layer_comp.current_alignment_quality < layer_comp.best_alignment_quality:
                layer_comp.best_alignment_quality = layer_comp.current_alignment_quality

    def _update_alignment_from_global(
        self, entropy_result: "ManifoldEntropyResult"
    ) -> None:
        """Update alignment from global SVD signature."""
        svd = entropy_result.svd_signature
        if svd is None:
            return

        # Distribute signature quality across layers
        for layer_idx, layer_comp in self._completion.layer_completions.items():
            # Use raw mean error as alignment metric
            layer_comp.current_alignment_quality = svd.mean_error

            if layer_comp.current_alignment_quality < layer_comp.best_alignment_quality:
                layer_comp.best_alignment_quality = layer_comp.current_alignment_quality

        # Track constants coverage
        # SVDSignatureResult.matches is List[Tuple[int, int, ConstantMatch]]
        for i, j, match in svd.matches:
            const = match.constant
            if const not in self._completion.constants_coverage:
                self._completion.constants_coverage[const] = 0
            self._completion.constants_coverage[const] += 1

    def _update_global_metrics(self, entropy_result: "ManifoldEntropyResult") -> None:
        """Update global completion metrics."""
        # Total aligned ratios
        total_aligned = sum(
            lc.n_aligned_ratios
            for lc in self._completion.layer_completions.values()
        )
        self._completion.total_aligned_ratios = total_aligned

        # Complexity-dimension law
        if entropy_result.complexity_law:
            law = entropy_result.complexity_law
            if law.slope_error is not None:
                self._completion.complexity_law_error = law.slope_error
                self._completion.complexity_law_validated = (
                    law.slope_error <= self._percent_epsilon
                )

    def _determine_completion_levels(self) -> None:
        """Determine completion level for each layer and globally."""
        n_incomplete = 0
        n_partial = 0
        n_saturated = 0
        n_complete = 0

        for layer_idx, layer_comp in self._completion.layer_completions.items():
            # Determine layer level
            if layer_comp.rounds_without_improvement >= self._saturation_patience:
                # Saturated: stuck with no improvement
                if layer_comp.current_alignment_quality <= self._percent_epsilon:
                    layer_comp.level = CompletionLevel.COMPLETE
                    n_complete += 1
                else:
                    layer_comp.level = CompletionLevel.SATURATED
                    n_saturated += 1
            elif layer_comp.n_aligned_ratios > 0:
                layer_comp.level = CompletionLevel.PARTIAL
                n_partial += 1
            else:
                layer_comp.level = CompletionLevel.INCOMPLETE
                n_incomplete += 1

        self._completion.n_incomplete = n_incomplete
        self._completion.n_partial = n_partial
        self._completion.n_saturated = n_saturated
        self._completion.n_complete = n_complete

        # Determine global level
        n_layers = len(self._completion.layer_completions)
        if n_layers == 0:
            self._completion.level = CompletionLevel.INCOMPLETE
        elif n_complete == n_layers:
            self._completion.level = CompletionLevel.COMPLETE
        elif n_saturated + n_complete == n_layers:
            self._completion.level = CompletionLevel.SATURATED
        elif n_partial + n_saturated + n_complete > 0:
            self._completion.level = CompletionLevel.PARTIAL
        else:
            self._completion.level = CompletionLevel.INCOMPLETE

    def is_complete(self) -> bool:
        """Check if manifold is geometrically complete."""
        return self._completion.level == CompletionLevel.COMPLETE

    def is_saturated(self) -> bool:
        """Check if manifold is saturated (no more improvement possible)."""
        return self._completion.level in {
            CompletionLevel.SATURATED,
            CompletionLevel.COMPLETE,
        }

    def get_incomplete_layers(self) -> List[int]:
        """Get list of layers that are still incomplete."""
        return [
            layer_idx
            for layer_idx, lc in self._completion.layer_completions.items()
            if lc.level in {CompletionLevel.INCOMPLETE, CompletionLevel.PARTIAL}
        ]

    def get_progress_percentage(self) -> float:
        """Get overall completion percentage."""
        n_layers = len(self._completion.layer_completions)
        if n_layers == 0:
            return 0.0

        # Progress = fraction of layers at or beyond saturation
        progressed = sum(
            1
            for lc in self._completion.layer_completions.values()
            if lc.level in {CompletionLevel.SATURATED, CompletionLevel.COMPLETE}
        )
        return progressed / n_layers * 100

    def get_progress_report(self) -> str:
        """Get human-readable progress report."""
        comp = self._completion
        n_layers = len(comp.layer_completions)

        lines = [
            f"Manifold Completion: {self.get_progress_percentage():.1f}%",
            f"Level: {comp.level.value}",
            f"Layers: {comp.n_complete} complete, {comp.n_saturated} saturated, "
            f"{comp.n_partial} partial, {comp.n_incomplete} incomplete (of {n_layers})",
        ]

        if comp.complexity_law_validated:
            lines.append(f"Complexity law: VALIDATED (error={comp.complexity_law_error:.2f}%)")
        else:
            lines.append(f"Complexity law: not validated (error={comp.complexity_law_error:.2f}%)")

        # Constants coverage
        if comp.constants_coverage:
            const_str = ", ".join(
                f"{c.value}:{count}"
                for c, count in sorted(
                    comp.constants_coverage.items(),
                    key=lambda x: -x[1]
                )[:5]
            )
            lines.append(f"Constants: {const_str}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset tracker for a new run."""
        self._completion = ManifoldCompletion()
        self._round_idx = 0


__all__ = [
    "CompletionLevel",
    "LayerCompletion",
    "ManifoldCompletion",
    "ManifoldCompletionTracker",
]
