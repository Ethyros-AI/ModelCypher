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
manifold entropy is minimized. It coordinates:

1. Geometry-derived perturbations from spectral structure
2. Entropy-based convergence tracking
3. Round scheduling

The loop continues until:
- No improving directions exist (within numerical resolution)
- Entropy has stabilized

No external supervision. The geometry IS the teacher.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.geodesic_null_space import GeodesicNullSpaceFilter
from modelcypher.core.domain.geometry.gram_spectrum import (
    compute_geometry_derived_scale,
    compute_gram_spectrum,
)
from modelcypher.core.domain.geometry.manifold_entropy import ManifoldEntropy
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)

from .direction_generator import DirectionGenerator, DirectionStrategy

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


logger = logging.getLogger(__name__)


@dataclass
class AutonomousRunResult:
    """Result of an autonomous manifold completion run."""

    # Completion status
    saturated: bool  # No more entropy improvement possible

    # Entropy metrics
    initial_entropy: float
    final_entropy: float
    entropy_reduction: float
    entropy_reduction_percent: float

    # Round statistics
    total_rounds: int
    effective_rounds: int  # Rounds that actually improved entropy

    # Timing
    start_time: str
    end_time: str
    duration_seconds: float

    # Per-round history
    round_history: List[Dict[str, Any]] = field(default_factory=list)


class AutonomousCompletion:
    """Run self-alignment autonomously until entropy is minimized.

    This is the "let it run until it's done" mode. The model will:
    1. Generate geometry-derived perturbations from spectral structure
    2. Project deltas into null space for safety
    3. Track entropy reduction
    4. Stop when no improvement is possible

    Usage:
        completer = AutonomousCompletion(backend)

        result = completer.run(
            get_weights=model.get_weights,
            set_weights=model.set_weights,
            get_activations=model.get_activations,
            layer_indices=[4, 8, 12, 16, 20, 24],
            probes=probes,
        )

        if result.saturated:
            print("Manifold entropy minimized!")
        print(f"Reduced entropy by {result.entropy_reduction_percent:.1f}%")
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        checkpoint_interval: int = 50,
    ) -> None:
        """Initialize autonomous completion.

        Args:
            backend: Computational backend
            checkpoint_interval: Rounds between progress reports
        """
        self._backend = backend or get_default_backend()
        self._checkpoint_interval = checkpoint_interval
        ref = self._backend.array([1.0], dtype="float32")
        eps = machine_epsilon(self._backend, ref)
        self._sqrt_eps = sqrt_scalar(eps, self._backend)

        # Initialize components
        self._entropy = ManifoldEntropy(self._backend)
        self._generator = DirectionGenerator(self._backend)
        self._null_space = GeodesicNullSpaceFilter(self._backend)

    def run(
        self,
        get_weights: Callable[[int], "Array"],
        set_weights: Callable[[int, "Array"], None],
        get_activations: Callable[[List[str]], Dict[int, "Array"]],
        layer_indices: List[int],
        probes: List[str],
        max_rounds: Optional[int] = None,
        strategies: Optional[List[DirectionStrategy]] = None,
        dry_run: bool = False,
        checkpoint_callback: Optional[Callable[[int, float], None]] = None,
    ) -> AutonomousRunResult:
        """Run autonomous manifold completion.

        Args:
            get_weights: Function to get weights for a layer index
            set_weights: Function to set weights for a layer index
            get_activations: Function to get activations for probes
            layer_indices: Which layers to align
            probes: Text probes to use for activation collection
            max_rounds: Optional maximum rounds (safety override)
            strategies: Direction generation strategies
            dry_run: If True, evaluate but don't apply changes
            checkpoint_callback: Called every checkpoint_interval rounds with (round, entropy)

        Returns:
            AutonomousRunResult with completion status and metrics
        """
        b = self._backend
        start_time = datetime.now()

        if strategies is None:
            strategies = [
                DirectionStrategy.SPECTRAL_COMPRESS,
                DirectionStrategy.SVD_GAP,
                DirectionStrategy.RANDOM,
            ]

        round_history: List[Dict[str, Any]] = []
        effective_rounds = 0
        rounds_without_improvement = 0

        # Initial measurement
        logger.info("Computing initial manifold state...")
        layer_activations = get_activations(probes)
        initial_result = self._entropy.compute_from_activations(layer_activations)
        initial_entropy = initial_result.total_entropy
        best_entropy = initial_entropy

        logger.info(f"Initial entropy: {initial_entropy:.4f}")
        logger.info("=" * 70)
        logger.info("STARTING AUTONOMOUS MANIFOLD COMPLETION")
        logger.info("=" * 70)

        # Main loop
        round_idx = 0
        saturated = False

        while True:
            round_idx += 1
            if max_rounds is not None and round_idx > max_rounds:
                logger.info(f"\nReached max_rounds={max_rounds}")
                break
            round_start = time.time()

            # Measure current state
            layer_activations = get_activations(probes)
            current_result = self._entropy.compute_from_activations(layer_activations)
            entropy_before = current_result.total_entropy
            entropy_eps = self._sqrt_eps
            if layer_activations:
                sample_act = next(iter(layer_activations.values()))
                entropy_eps = sqrt_scalar(machine_epsilon(b, sample_act), b)
            entropy_tol = entropy_eps * max(1.0, abs(entropy_before))

            # Evaluate all requested layers (geometry decides)
            target_layers = layer_indices

            # Compute geometry-derived scale per layer
            layer_scales: Dict[int, float] = {}
            for layer_idx in target_layers:
                activations = layer_activations.get(layer_idx)
                if activations is None:
                    continue
                spectrum = compute_gram_spectrum(activations, backend=b)
                eps = machine_epsilon(b, activations)
                layer_scales[layer_idx] = compute_geometry_derived_scale(
                    spectrum,
                    epsilon=eps,
                )

            # Generate and evaluate directions
            best_direction = None
            best_delta = 0.0
            best_layer_idx = -1
            best_strategy = None

            for layer_idx in target_layers:
                if layer_idx not in layer_scales:
                    continue
                weights = get_weights(layer_idx)
                b.eval(weights)

                directions = self._generator.generate(
                    weights,
                    strategies=strategies,
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
                    scale_factor = layer_scales.get(layer_idx, 0.0)
                    if scale_factor <= 0.0:
                        continue
                    projected = projected * scale_factor

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

            if best_direction is not None and best_delta > entropy_tol and not dry_run:
                weights = get_weights(best_layer_idx)
                new_weights = weights + best_direction
                b.eval(new_weights)
                set_weights(best_layer_idx, new_weights)
                direction_applied = True
                effective_rounds += 1
                rounds_without_improvement = 0

                # Measure after applying
                new_activations = get_activations(probes)
                new_result = self._entropy.compute_from_activations(new_activations)
                entropy_after = new_result.total_entropy

                if entropy_after < best_entropy:
                    best_entropy = entropy_after
            else:
                rounds_without_improvement += 1

            # Record round
            round_time = time.time() - round_start
            round_record = {
                "round": round_idx,
                "entropy_before": entropy_before,
                "entropy_after": entropy_after,
                "entropy_delta": best_delta,
                "direction_applied": direction_applied,
                "strategy": best_strategy.value if best_strategy else None,
                "layer": best_layer_idx if direction_applied else None,
                "time_seconds": round_time,
            }
            round_history.append(round_record)

            # Log progress
            if direction_applied:
                logger.info(
                    f"Round {round_idx}: Δ={best_delta:.4e} "
                    f"(layer={best_layer_idx}, {best_strategy.value if best_strategy else 'N/A'})"
                )
            else:
                if round_idx % 10 == 0:
                    logger.info(
                        f"Round {round_idx}: No improvement"
                    )

            # Checkpoint
            if round_idx % self._checkpoint_interval == 0:
                logger.info("-" * 50)
                logger.info(f"CHECKPOINT at round {round_idx}")
                logger.info(f"Current entropy: {entropy_after:.4f}")
                logger.info(f"Best entropy: {best_entropy:.4f}")
                logger.info(f"Effective rounds: {effective_rounds}/{round_idx}")
                logger.info("-" * 50)

                if checkpoint_callback:
                    checkpoint_callback(round_idx, entropy_after)

            # Check termination conditions
            if best_delta <= entropy_tol:
                logger.info(
                    f"\nNo improvement beyond numerical resolution after {round_idx} rounds"
                )
                saturated = True
                break

        # Final measurement
        end_time = datetime.now()
        final_activations = get_activations(probes)
        final_result = self._entropy.compute_from_activations(final_activations)
        final_entropy = final_result.total_entropy

        entropy_reduction = initial_entropy - final_entropy
        entropy_reduction_pct = (
            (entropy_reduction / initial_entropy * 100) if initial_entropy > 0 else 0
        )

        # Log final results
        logger.info("\n" + "=" * 70)
        logger.info("AUTONOMOUS COMPLETION FINISHED")
        logger.info("=" * 70)
        logger.info(f"Initial entropy:    {initial_entropy:.4f}")
        logger.info(f"Final entropy:      {final_entropy:.4f}")
        logger.info(f"Reduction:          {entropy_reduction:.4f} ({entropy_reduction_pct:.2f}%)")
        logger.info(f"Effective rounds:   {effective_rounds}/{len(round_history)}")

        return AutonomousRunResult(
            saturated=saturated,
            initial_entropy=initial_entropy,
            final_entropy=final_entropy,
            entropy_reduction=entropy_reduction,
            entropy_reduction_percent=entropy_reduction_pct,
            total_rounds=len(round_history),
            effective_rounds=effective_rounds,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            round_history=round_history,
        )


__all__ = [
    "AutonomousCompletion",
    "AutonomousRunResult",
]
