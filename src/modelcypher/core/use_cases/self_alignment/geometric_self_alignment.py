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

"""Geometric Self-Alignment Orchestrator.

The main loop that lets a model self-play and modify its weights to reduce
entropy across the full manifold, using pure geometric measurements.

Algorithm:
1. MEASURE current manifold entropy
2. GENERATE candidate weight perturbations
3. EVALUATE each direction by entropy reduction
4. APPLY best direction (if safe, via null-space projection)
5. CHECK convergence
6. REPEAT until converged

No external supervision. The geometry IS the teacher.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

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

from .convergence_detector import ConvergenceDetector, ConvergenceResult
from .direction_generator import DirectionGenerator, DirectionResult, DirectionStrategy

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class AlignmentRoundResult:
    """Result of a single alignment round."""

    round_idx: int
    entropy_before: float
    entropy_after: float
    entropy_delta: float  # Positive = improvement
    best_direction: Optional[DirectionResult]
    direction_applied: bool
    n_directions_evaluated: int


@dataclass
class AlignmentResult:
    """Result of the full self-alignment process."""

    converged: bool
    convergence_reason: str
    n_rounds: int
    initial_entropy: float
    final_entropy: float
    entropy_reduction: float  # Positive = improvement
    alignment_quality_initial: float  # complexity law r_squared
    alignment_quality_final: float  # complexity law r_squared

    # Per-round history
    round_history: List[AlignmentRoundResult] = field(default_factory=list)

    # Final convergence metrics
    convergence: Optional[ConvergenceResult] = None


class GeometricSelfAlignment:
    """Orchestrate geometric self-alignment of a model.

    This is the main entry point for self-alignment. It coordinates:
    - ManifoldEntropy: Measure the current state
    - DirectionGenerator: Generate candidate perturbations
    - GeodesicNullSpaceFilter: Project directions to safe space
    - ConvergenceDetector: Detect when we're done

    Usage:
        aligner = GeometricSelfAlignment(backend)

        # Define how to get weights and activations from your model
        def get_weights(layer_idx):
            return model.layers[layer_idx].mlp.weight

        def set_weights(layer_idx, weights):
            model.layers[layer_idx].mlp.weight = weights

        def get_activations(probes):
            # Run probes through model, return per-layer activations
            return {0: acts_0, 1: acts_1, ...}

        result = aligner.run(
            get_weights=get_weights,
            set_weights=set_weights,
            get_activations=get_activations,
            layer_indices=[4, 8, 12],
            probes=["The sky is blue.", "Fire is hot.", ...],
            max_rounds=50,
        )
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize the self-alignment orchestrator.

        Args:
            backend: Computational backend
        """
        self._backend = backend or get_default_backend()
        self._entropy = ManifoldEntropy(self._backend)
        self._generator = DirectionGenerator(self._backend)
        self._null_space = GeodesicNullSpaceFilter(self._backend)
        self._convergence = ConvergenceDetector(self._backend)
        ref = self._backend.array([1.0], dtype="float32")
        eps = machine_epsilon(self._backend, ref)
        self._sqrt_eps = sqrt_scalar(eps, self._backend)

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
    ) -> AlignmentResult:
        """Run the self-alignment loop.

        Args:
            get_weights: Function to get weights for a layer index
            set_weights: Function to set weights for a layer index
            get_activations: Function to get activations for probes
            layer_indices: Which layers to align
            probes: Text probes to use for activation collection
            max_rounds: Optional maximum rounds (safety override)
            strategies: Direction generation strategies (default: all)
            dry_run: If True, evaluate but don't apply changes

        Returns:
            AlignmentResult with convergence info and history
        """
        b = self._backend

        if strategies is None:
            strategies = [
                DirectionStrategy.SPECTRAL_COMPRESS,
                DirectionStrategy.SVD_GAP,
                DirectionStrategy.RANDOM,
            ]

        self._convergence.reset()
        round_history: List[AlignmentRoundResult] = []

        # Initial measurement
        logger.info("Computing initial manifold entropy...")
        layer_activations = get_activations(probes)
        initial_result = self._entropy.compute_from_activations(layer_activations)
        initial_entropy = initial_result.total_entropy
        initial_alignment = 0.0  # Reserved for future geometric metrics

        logger.info(f"Initial entropy: {initial_entropy:.4f}")
        logger.info(f"Initial alignment (r_squared): {initial_alignment:.4f}")

        self._convergence.update(initial_result, 0)

        # Main loop
        round_idx = 0
        while True:
            round_idx += 1
            if max_rounds is not None and round_idx > max_rounds:
                logger.info(f"\nReached max_rounds={max_rounds}")
                break
            logger.info(f"\n--- Round {round_idx} ---")

            # Measure current entropy
            layer_activations = get_activations(probes)
            current_result = self._entropy.compute_from_activations(layer_activations)
            entropy_before = current_result.total_entropy
            entropy_eps = self._sqrt_eps
            if layer_activations:
                sample_act = next(iter(layer_activations.values()))
                entropy_eps = sqrt_scalar(machine_epsilon(b, sample_act), b)
            entropy_tol = entropy_eps * max(1.0, abs(entropy_before))

            best_direction: Optional[DirectionResult] = None
            best_delta = 0.0
            best_layer_idx = -1
            n_evaluated = 0

            # Geometry-derived scale per layer
            layer_scales: Dict[int, float] = {}
            for layer_idx in layer_indices:
                activations = layer_activations.get(layer_idx)
                if activations is None:
                    continue
                spectrum = compute_gram_spectrum(activations, backend=b)
                eps = machine_epsilon(b, activations)
                layer_scales[layer_idx] = compute_geometry_derived_scale(
                    spectrum,
                    epsilon=eps,
                )

            # Evaluate directions for each layer
            for layer_idx in layer_indices:
                if layer_idx not in layer_scales:
                    continue
                weights = get_weights(layer_idx)
                b.eval(weights)

                # Generate candidate directions
                directions = self._generator.generate(
                    weights,
                    strategies=strategies,
                )

                for direction in directions:
                    n_evaluated += 1

                    # Project to null space for safety
                    activations = layer_activations.get(layer_idx)
                    if activations is None:
                        continue

                    null_result = self._null_space.filter_delta(
                        direction.direction,
                        activations,
                        delta_space="weights",
                    )

                    if not null_result.filtering_applied:
                        projected_direction = direction.direction
                    else:
                        projected_direction = null_result.filtered_delta
                    scale_factor = layer_scales.get(layer_idx, 0.0)
                    if scale_factor <= 0.0:
                        continue
                    projected_direction = projected_direction * scale_factor
                    proj_norm = b.sqrt(b.sum(projected_direction * projected_direction))
                    b.eval(proj_norm)
                    proj_norm_val = float(b.to_scalar(proj_norm))

                    # Evaluate entropy after applying direction
                    new_weights = weights + projected_direction
                    b.eval(new_weights)

                    # Temporarily apply direction
                    set_weights(layer_idx, new_weights)

                    # Measure new entropy
                    new_activations = get_activations(probes)
                    new_result = self._entropy.compute_from_activations(new_activations)
                    new_entropy = new_result.total_entropy

                    # Restore original weights
                    set_weights(layer_idx, weights)

                    # Check if this direction improves entropy
                    delta = entropy_before - new_entropy  # Positive = improvement
                    if delta > best_delta:
                        best_delta = delta
                        best_direction = DirectionResult(
                            direction=projected_direction,
                            strategy=direction.strategy,
                            scale=proj_norm_val,
                            target_ratio_indices=direction.target_ratio_indices,
                            expected_entropy_reduction=delta,
                        )
                        best_layer_idx = layer_idx

            # Apply best direction if it improves entropy
            direction_applied = False
            entropy_after = entropy_before

            if best_direction is not None and best_delta > entropy_tol and not dry_run:
                weights = get_weights(best_layer_idx)
                new_weights = weights + best_direction.direction
                b.eval(new_weights)
                set_weights(best_layer_idx, new_weights)
                direction_applied = True

                # Measure final entropy after applying
                new_activations = get_activations(probes)
                new_result = self._entropy.compute_from_activations(new_activations)
                entropy_after = new_result.total_entropy

                logger.info(
                    f"Applied {best_direction.strategy.value} direction to layer {best_layer_idx}"
                )
                logger.info(f"Entropy: {entropy_before:.4f} -> {entropy_after:.4f} (Δ={best_delta:.4f})")
            else:
                logger.info("No improving direction found")
                if best_delta <= entropy_tol:
                    logger.info("No improvement beyond numerical resolution")

            # Record round result
            round_result = AlignmentRoundResult(
                round_idx=round_idx,
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                entropy_delta=entropy_before - entropy_after,
                best_direction=best_direction,
                direction_applied=direction_applied,
                n_directions_evaluated=n_evaluated,
            )
            round_history.append(round_result)

            # Update convergence detector
            final_activations = get_activations(probes)
            final_result = self._entropy.compute_from_activations(final_activations)
            self._convergence.update(final_result, round_idx)

            # Check convergence
            conv_result = self._convergence.check_convergence()
            if conv_result.is_converged:
                logger.info(f"\nConverged: {conv_result.reason}")
                break

        # Final measurement
        final_activations = get_activations(probes)
        final_result = self._entropy.compute_from_activations(final_activations)
        final_entropy = final_result.total_entropy
        final_alignment = 0.0  # Reserved for future geometric metrics

        logger.info("\n" + "=" * 60)
        logger.info("ALIGNMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Initial entropy: {initial_entropy:.4f}")
        logger.info(f"Final entropy: {final_entropy:.4f}")
        logger.info(f"Reduction: {initial_entropy - final_entropy:.4f}")
        logger.info(f"Initial alignment (r_squared): {initial_alignment:.4f}")
        logger.info(f"Final alignment (r_squared): {final_alignment:.4f}")

        return AlignmentResult(
            converged=self._convergence.is_converged(),
            convergence_reason=self._convergence.check_convergence().reason,
            n_rounds=len(round_history),
            initial_entropy=initial_entropy,
            final_entropy=final_entropy,
            entropy_reduction=initial_entropy - final_entropy,
            alignment_quality_initial=initial_alignment,
            alignment_quality_final=final_alignment,
            round_history=round_history,
            convergence=self._convergence.check_convergence(),
        )

    def run_single_round(
        self,
        get_weights: Callable[[int], "Array"],
        set_weights: Callable[[int, "Array"], None],
        get_activations: Callable[[List[str]], Dict[int, "Array"]],
        layer_indices: List[int],
        probes: List[str],
        strategies: Optional[List[DirectionStrategy]] = None,
    ) -> Tuple[float, bool]:
        """Run a single alignment round.

        Useful for debugging or manual control.

        Returns:
            Tuple of (entropy_delta, direction_applied)
        """
        b = self._backend

        if strategies is None:
            strategies = [DirectionStrategy.CONSTANT_ALIGNED]

        # Measure current entropy
        layer_activations = get_activations(probes)
        current_result = self._entropy.compute_from_activations(layer_activations)
        entropy_before = current_result.total_entropy

        best_delta = 0.0
        best_direction = None
        best_layer_idx = -1

        # Geometry-derived scale per layer
        layer_scales: Dict[int, float] = {}
        for layer_idx in layer_indices:
            activations = layer_activations.get(layer_idx)
            if activations is None:
                continue
            spectrum = compute_gram_spectrum(activations, backend=b)
            eps = machine_epsilon(b, activations)
            layer_scales[layer_idx] = compute_geometry_derived_scale(
                spectrum,
                epsilon=eps,
            )

        for layer_idx in layer_indices:
            if layer_idx not in layer_scales:
                continue
            weights = get_weights(layer_idx)
            b.eval(weights)

            directions = self._generator.generate(
                weights,
                strategies=strategies,
            )

            for direction in directions:
                activations = layer_activations.get(layer_idx)
                if activations is None:
                    continue

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
                set_weights(layer_idx, new_weights)
                new_acts = get_activations(probes)
                new_result = self._entropy.compute_from_activations(new_acts)
                set_weights(layer_idx, weights)

                delta = entropy_before - new_result.total_entropy
                if delta > best_delta:
                    best_delta = delta
                    best_direction = projected
                    best_layer_idx = layer_idx

        # Apply best
        entropy_eps = self._sqrt_eps
        if layer_activations:
            sample_act = next(iter(layer_activations.values()))
            entropy_eps = sqrt_scalar(machine_epsilon(b, sample_act), b)
        entropy_tol = entropy_eps * max(1.0, abs(entropy_before))
        if best_direction is not None and best_delta > entropy_tol:
            weights = get_weights(best_layer_idx)
            set_weights(best_layer_idx, weights + best_direction)
            return best_delta, True

        return 0.0, False


__all__ = [
    "GeometricSelfAlignment",
    "AlignmentResult",
    "AlignmentRoundResult",
]
