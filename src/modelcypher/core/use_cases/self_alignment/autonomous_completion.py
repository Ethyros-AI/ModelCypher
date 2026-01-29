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

"""Autonomous manifold completion - let a model fill its geometric space.

This is the main entry point for running self-alignment until the model's
manifold is geometrically complete. It coordinates:

1. Multi-scale perturbation to escape local minima
2. Layer-wise completion tracking
3. Full-manifold convergence criteria
4. Adaptive round scheduling

The loop continues until:
- All layers are geometrically saturated
- No improving directions exist at any scale
- The complexity-dimension law is validated

No external supervision. The geometry IS the teacher.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_entropy import ManifoldEntropy

from .convergence_detector import ConvergenceDetector
from .direction_generator import DirectionGenerator, DirectionStrategy
from .manifold_completion import (
    CompletionLevel,
    ManifoldCompletion,
    ManifoldCompletionTracker,
)
from .multi_scale_perturbation import MultiScalePerturbation
from modelcypher.core.domain.geometry.geodesic_null_space import GeodesicNullSpaceFilter

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


logger = logging.getLogger(__name__)


@dataclass
class AutonomousRunResult:
    """Result of an autonomous manifold completion run."""

    # Completion status
    completed: bool
    completion_level: CompletionLevel
    completion_percentage: float

    # Entropy metrics
    initial_entropy: float
    final_entropy: float
    entropy_reduction: float
    entropy_reduction_percent: float

    # Alignment metrics (complexity law r_squared)
    initial_alignment: float
    final_alignment: float
    alignment_improvement: float

    # Round statistics
    total_rounds: int
    effective_rounds: int  # Rounds that actually improved entropy
    scale_cycles: int

    # Timing
    start_time: str
    end_time: str
    duration_seconds: float

    # Completion report
    completion_report: str

    # Per-round history
    round_history: List[Dict[str, Any]] = field(default_factory=list)


class AutonomousCompletion:
    """Run self-alignment autonomously until manifold is complete.

    This is the "let it run until it's done" mode. The model will:
    1. Start with medium-scale perturbations
    2. Try finer scales when improving
    3. Try larger scales when stuck (escape local minima)
    4. Track layer-by-layer completion
    5. Stop when geometrically saturated at all layers and scales

    Usage:
        completer = AutonomousCompletion(backend)

        result = completer.run(
            get_weights=model.get_weights,
            set_weights=model.set_weights,
            get_activations=model.get_activations,
            layer_indices=[4, 8, 12, 16, 20, 24],
            probes=probes,
        )

        if result.completed:
            print("Manifold geometrically complete!")
        else:
            print(f"Reached saturation at {result.completion_percentage:.1f}%")
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        n_directions_per_round: int = 10,
        max_scale_cycles: int = 5,
        saturation_patience: int = 15,
        checkpoint_interval: int = 50,
    ) -> None:
        """Initialize autonomous completion.

        Args:
            backend: Computational backend
            n_directions_per_round: Candidate directions per round
            max_scale_cycles: Maximum complete cycles through all scales
            saturation_patience: Rounds without improvement = saturated
            checkpoint_interval: Rounds between progress reports
        """
        self._backend = backend or get_default_backend()
        self._n_directions = n_directions_per_round
        self._max_scale_cycles = max_scale_cycles
        self._saturation_patience = saturation_patience
        self._checkpoint_interval = checkpoint_interval

        # Initialize components
        self._entropy = ManifoldEntropy(self._backend)
        self._generator = DirectionGenerator(self._backend)
        self._null_space = GeodesicNullSpaceFilter(self._backend)
        self._multi_scale = MultiScalePerturbation(
            self._backend,
            n_scales=7,
            patience_per_scale=5,
        )
        self._completion_tracker = ManifoldCompletionTracker(
            self._backend,
            saturation_patience=saturation_patience,
        )

    def run(
        self,
        get_weights: Callable[[int], "Array"],
        set_weights: Callable[[int, "Array"], None],
        get_activations: Callable[[List[str]], Dict[int, "Array"]],
        layer_indices: List[int],
        probes: List[str],
        max_rounds: int = 1000,
        strategies: Optional[List[DirectionStrategy]] = None,
        dry_run: bool = False,
        checkpoint_callback: Optional[Callable[[int, "ManifoldCompletion"], None]] = None,
    ) -> AutonomousRunResult:
        """Run autonomous manifold completion.

        Args:
            get_weights: Function to get weights for a layer index
            set_weights: Function to set weights for a layer index
            get_activations: Function to get activations for probes
            layer_indices: Which layers to align
            probes: Text probes to use for activation collection
            max_rounds: Maximum rounds (safety limit)
            strategies: Direction generation strategies
            dry_run: If True, evaluate but don't apply changes
            checkpoint_callback: Called every checkpoint_interval rounds

        Returns:
            AutonomousRunResult with completion status and metrics
        """
        b = self._backend
        start_time = datetime.now()

        if strategies is None:
            strategies = [
                DirectionStrategy.CONSTANT_ALIGNED,
                DirectionStrategy.SPECTRAL_COMPRESS,
                DirectionStrategy.SVD_GAP,
                DirectionStrategy.RANDOM,
            ]

        # Reset trackers
        self._multi_scale.reset()
        self._completion_tracker.reset()

        round_history: List[Dict[str, Any]] = []
        effective_rounds = 0

        # Initial measurement
        logger.info("Computing initial manifold state...")
        layer_activations = get_activations(probes)
        initial_result = self._entropy.compute_from_activations(layer_activations)
        initial_entropy = initial_result.total_entropy
        initial_alignment = (
            initial_result.complexity_law.r_squared
            if initial_result.complexity_law is not None
            else 0.0
        )

        logger.info(f"Initial entropy: {initial_entropy:.4f}")
        logger.info(f"Initial alignment (r_squared): {initial_alignment:.4f}")
        logger.info(f"Scale schedule: {[f'{s:.2e}' for s in self._multi_scale.scales]}")
        logger.info("=" * 70)
        logger.info("STARTING AUTONOMOUS MANIFOLD COMPLETION")
        logger.info("=" * 70)

        # Update completion tracker with initial state
        self._completion_tracker.update(initial_result)

        # Main loop
        for round_idx in range(1, max_rounds + 1):
            round_start = time.time()

            # Get current scale (multi-scale handles escape logic)
            current_scale = self._multi_scale.get_current_scale()

            # Measure current state
            layer_activations = get_activations(probes)
            current_result = self._entropy.compute_from_activations(layer_activations)
            entropy_before = current_result.total_entropy

            # Focus on incomplete layers
            incomplete_layers = self._completion_tracker.get_incomplete_layers()
            target_layers = incomplete_layers if incomplete_layers else layer_indices

            # Generate and evaluate directions
            best_direction = None
            best_delta = 0.0
            best_layer_idx = -1
            best_strategy = None

            for layer_idx in target_layers:
                weights = get_weights(layer_idx)
                b.eval(weights)

                # Generate directions at current scale
                directions = self._generator.generate(
                    weights,
                    n_directions=self._n_directions,
                    strategies=strategies,
                    scale=current_scale,
                )

                activations = layer_activations.get(layer_idx)
                if activations is None:
                    continue

                for direction in directions:
                    # Project to null space
                    null_result = self._null_space.filter_delta(
                        direction.direction,
                        activations,
                        delta_space="weights",
                    )

                    projected = (
                        null_result.filtered_delta
                        if null_result.filtering_applied
                        else direction.direction
                    )

                    if not dry_run:
                        # Evaluate
                        new_weights = weights + projected
                        b.eval(new_weights)
                        set_weights(layer_idx, new_weights)

                        new_activations = get_activations(probes)
                        new_result = self._entropy.compute_from_activations(new_activations)
                        new_entropy = new_result.total_entropy

                        # Restore
                        set_weights(layer_idx, weights)

                        delta = entropy_before - new_entropy
                        if delta > best_delta:
                            best_delta = delta
                            best_direction = projected
                            best_layer_idx = layer_idx
                            best_strategy = direction.strategy

            # Apply best direction if it improves
            entropy_after = entropy_before
            direction_applied = False

            if best_direction is not None and best_delta > 0 and not dry_run:
                weights = get_weights(best_layer_idx)
                new_weights = weights + best_direction
                b.eval(new_weights)
                set_weights(best_layer_idx, new_weights)
                direction_applied = True
                effective_rounds += 1

                # Measure after applying
                new_activations = get_activations(probes)
                new_result = self._entropy.compute_from_activations(new_activations)
                entropy_after = new_result.total_entropy

            # Update multi-scale state
            self._multi_scale.update(best_delta, entropy_after)

            # Update completion tracker
            final_activations = get_activations(probes)
            final_result = self._entropy.compute_from_activations(final_activations)
            self._completion_tracker.update(final_result)

            # Record round
            round_time = time.time() - round_start
            round_record = {
                "round": round_idx,
                "scale": current_scale,
                "entropy_before": entropy_before,
                "entropy_after": entropy_after,
                "entropy_delta": best_delta,
                "direction_applied": direction_applied,
                "strategy": best_strategy.value if best_strategy else None,
                "layer": best_layer_idx if direction_applied else None,
                "completion_pct": self._completion_tracker.get_progress_percentage(),
                "time_seconds": round_time,
            }
            round_history.append(round_record)

            # Log progress
            if direction_applied:
                logger.info(
                    f"Round {round_idx}: Δ={best_delta:.4e} (scale={current_scale:.2e}, "
                    f"layer={best_layer_idx}, {best_strategy.value if best_strategy else 'N/A'})"
                )
            else:
                if round_idx % 10 == 0:
                    logger.info(
                        f"Round {round_idx}: No improvement "
                        f"({self._multi_scale.get_status()})"
                    )

            # Checkpoint
            if round_idx % self._checkpoint_interval == 0:
                logger.info("-" * 50)
                logger.info(f"CHECKPOINT at round {round_idx}")
                logger.info(self._completion_tracker.get_progress_report())
                logger.info(f"Effective rounds: {effective_rounds}/{round_idx}")
                logger.info("-" * 50)

                if checkpoint_callback:
                    checkpoint_callback(round_idx, self._completion_tracker.completion)

            # Check termination conditions
            if self._completion_tracker.is_saturated():
                logger.info(f"\nManifold SATURATED after {round_idx} rounds")
                break

            if not self._multi_scale.should_continue(self._max_scale_cycles):
                logger.info(
                    f"\nExhausted {self._max_scale_cycles} scale cycles after {round_idx} rounds"
                )
                break

        # Final measurement
        end_time = datetime.now()
        final_activations = get_activations(probes)
        final_result = self._entropy.compute_from_activations(final_activations)
        final_entropy = final_result.total_entropy
        final_alignment = (
            final_result.complexity_law.r_squared
            if final_result.complexity_law is not None
            else 0.0
        )

        entropy_reduction = initial_entropy - final_entropy
        entropy_reduction_pct = (
            (entropy_reduction / initial_entropy * 100) if initial_entropy > 0 else 0
        )

        completion = self._completion_tracker.completion
        completion_report = self._completion_tracker.get_progress_report()

        # Log final results
        logger.info("\n" + "=" * 70)
        logger.info("AUTONOMOUS COMPLETION FINISHED")
        logger.info("=" * 70)
        logger.info(f"Initial entropy:    {initial_entropy:.4f}")
        logger.info(f"Final entropy:      {final_entropy:.4f}")
        logger.info(f"Reduction:          {entropy_reduction:.4f} ({entropy_reduction_pct:.2f}%)")
        logger.info(f"Initial alignment (r_squared):  {initial_alignment:.4f}")
        logger.info(f"Final alignment (r_squared):    {final_alignment:.4f}")
        logger.info(f"Effective rounds:   {effective_rounds}/{len(round_history)}")
        logger.info(f"Scale cycles:       {self._multi_scale.state.cycle_count}")
        logger.info("-" * 40)
        logger.info(completion_report)

        return AutonomousRunResult(
            completed=self._completion_tracker.is_complete(),
            completion_level=completion.level,
            completion_percentage=self._completion_tracker.get_progress_percentage(),
            initial_entropy=initial_entropy,
            final_entropy=final_entropy,
            entropy_reduction=entropy_reduction,
            entropy_reduction_percent=entropy_reduction_pct,
            initial_alignment=initial_alignment,
            final_alignment=final_alignment,
            alignment_improvement=final_alignment - initial_alignment,
            total_rounds=len(round_history),
            effective_rounds=effective_rounds,
            scale_cycles=self._multi_scale.state.cycle_count,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            completion_report=completion_report,
            round_history=round_history,
        )


__all__ = [
    "AutonomousCompletion",
    "AutonomousRunResult",
]
