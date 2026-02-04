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

"""Composable multi-layer compressor with error tracking.

Handles compression of multiple layers with:
- Per-layer error tracking
- Cumulative error propagation analysis
- Safe combination discovery
- Held-out validation

Key insight from experiments: Individual layers at 100% may fail when combined.
Errors compound through layers, especially when compression is applied to
non-contiguous layer sets.

References:
    - Compression Investigation Findings (docs/compression_investigation_findings.md)
    - Experiment 1: Error propagation analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Any

from modelcypher.core.domain._backend import get_default_backend
from .rmt_compressor import (
    RMTAwareCompressor,
    CompressionResult,
)
from .geodesic_analyzer import (
    GeodesicLayerAnalyzer,
    GeodesicLayerProfile,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class LayerCompressionState:
    """State of compression for a single layer.

    Attributes:
        layer_idx: Index of the layer in the model.
        T: Linear transform [d_out, d_in] that approximates MLP.
        compression_result: Full compression result with diagnostics.
        geodesic_profile: Optional geodesic analysis of the layer.
        calibration_accuracy: Top-1 accuracy on calibration data.
        held_out_accuracy: Top-1 accuracy on held-out data (if evaluated).
        accumulated_error: Estimated error accumulated from prior layers.
    """

    layer_idx: int
    T: "Array"
    compression_result: CompressionResult
    geodesic_profile: GeodesicLayerProfile | None = None
    calibration_accuracy: float = 0.0
    held_out_accuracy: float | None = None
    accumulated_error: float = 0.0


@dataclass
class CompositionResult:
    """Result of composing multiple compressed layers.

    Attributes:
        layer_states: List of per-layer compression states.
        layer_indices: Indices of compressed layers.
        overall_accuracy: Combined accuracy on held-out data.
        cumulative_error: Total accumulated reconstruction error.
        is_safe: Whether this composition maintains target accuracy.
        compression_ratio: Fraction of model that is compressed.
    """

    layer_states: list[LayerCompressionState]
    layer_indices: list[int]
    overall_accuracy: float
    cumulative_error: float
    is_safe: bool
    compression_ratio: float


class ComposableLayerCompressor:
    """Orchestrates multi-layer compression with error tracking.

    The algorithm:
    1. Profile each layer with GeodesicLayerAnalyzer
    2. Compress each layer with RMTAwareCompressor
    3. Track error propagation through layers
    4. Find safe combinations that maintain accuracy

    Key findings from experiments:
    - Contiguous layers compress better than non-contiguous
    - Error compounds multiplicatively through layers
    - Some "gate" layers (like layer 6) cannot be compressed
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._compressor = RMTAwareCompressor(backend=self._backend)
        self._analyzer = GeodesicLayerAnalyzer(backend=self._backend)

    def compress_layer(
        self,
        X: "Array",
        Y: "Array",
        layer_idx: int,
        analyze_geodesic: bool = True,
    ) -> LayerCompressionState:
        """Compress a single layer with optional geodesic analysis.

        Args:
            X: MLP input activations [n_samples, d_in].
            Y: MLP output activations [n_samples, d_out].
            layer_idx: Index of this layer in the model.
            analyze_geodesic: Whether to run geodesic analysis.

        Returns:
            LayerCompressionState with compression result and diagnostics.
        """
        b = self._backend

        X = b.array(X)
        Y = b.array(Y)
        b.eval(X, Y)

        logger.info(f"COMPOSABLE: Compressing layer {layer_idx}")

        # Step 1: RMT compression
        compression_result = self._compressor.compress_layer(X, Y)

        # Step 2: Optional geodesic analysis
        geodesic_profile = None
        if analyze_geodesic:
            try:
                geodesic_profile = self._analyzer.analyze(X)
            except Exception as e:
                logger.warning(f"Geodesic analysis failed for layer {layer_idx}: {e}")

        # Step 3: Compute calibration accuracy
        T = compression_result.T
        T_T = b.transpose(T)
        Y_pred = b.matmul(X, T_T)
        b.eval(Y_pred)

        n_samples = int(X.shape[0])
        correct = 0

        for i in range(n_samples):
            y_true = Y[i, :]
            y_pred = Y_pred[i, :]

            true_argmax = b.argmax(y_true)
            pred_argmax = b.argmax(y_pred)
            b.eval(true_argmax, pred_argmax)

            if int(b.to_scalar(true_argmax)) == int(b.to_scalar(pred_argmax)):
                correct += 1

        calibration_accuracy = correct / n_samples if n_samples > 0 else 0.0

        logger.info(
            f"COMPOSABLE: Layer {layer_idx} calibration_accuracy={calibration_accuracy:.1%}, "
            f"signal_rank={compression_result.signal_rank}"
        )

        return LayerCompressionState(
            layer_idx=layer_idx,
            T=T,
            compression_result=compression_result,
            geodesic_profile=geodesic_profile,
            calibration_accuracy=calibration_accuracy,
            held_out_accuracy=None,
            accumulated_error=compression_result.reconstruction_error,
        )

    def compress_sequence(
        self,
        layer_data: list[tuple["Array", "Array", int]],
        analyze_geodesic: bool = True,
    ) -> list[LayerCompressionState]:
        """Compress a sequence of layers.

        Args:
            layer_data: List of (X, Y, layer_idx) tuples.
            analyze_geodesic: Whether to run geodesic analysis.

        Returns:
            List of LayerCompressionState for each layer.
        """
        states = []

        for X, Y, layer_idx in layer_data:
            state = self.compress_layer(X, Y, layer_idx, analyze_geodesic)
            states.append(state)

        return states

    def evaluate_composition(
        self,
        states: list[LayerCompressionState],
        evaluate_fn: Callable[[list[int], list["Array"]], float],
    ) -> CompositionResult:
        """Evaluate a composition of compressed layers.

        Args:
            states: List of LayerCompressionState to evaluate together.
            evaluate_fn: Function that takes (layer_indices, T_matrices) and
                        returns accuracy on held-out data.

        Returns:
            CompositionResult with overall accuracy and safety assessment.
        """
        layer_indices = [s.layer_idx for s in states]
        T_matrices = [s.T for s in states]

        # Compute cumulative error (multiply reconstruction errors)
        cumulative_error = 1.0
        for s in states:
            cumulative_error *= (1.0 - s.compression_result.reconstruction_error)
        cumulative_error = 1.0 - cumulative_error  # Convert to error fraction

        # Evaluate with provided function
        accuracy = evaluate_fn(layer_indices, T_matrices)

        # Update held-out accuracy in states
        for s in states:
            s.held_out_accuracy = accuracy

        # Determine if this composition is safe (100% accuracy)
        is_safe = accuracy >= 1.0 - 1e-6

        # Compute compression ratio (simplified: assume equal layer contribution)
        # In practice, this should be: sum(layer_params) / total_model_params
        compression_ratio = len(states) / 36  # Assuming 36-layer model

        logger.info(
            f"COMPOSABLE: Composition {layer_indices} -> accuracy={accuracy:.1%}, "
            f"cumulative_error={cumulative_error:.4f}, is_safe={is_safe}"
        )

        return CompositionResult(
            layer_states=states,
            layer_indices=layer_indices,
            overall_accuracy=accuracy,
            cumulative_error=cumulative_error,
            is_safe=is_safe,
            compression_ratio=compression_ratio,
        )

    def find_safe_combinations(
        self,
        all_states: list[LayerCompressionState],
        evaluate_fn: Callable[[list[int], list["Array"]], float],
        max_layers: int = 10,
        require_contiguous: bool = False,
        accuracy_threshold: float = 1.0,
    ) -> list[CompositionResult]:
        """Find safe layer combinations that maintain target accuracy.

        Uses greedy search: start with best individual layers, add more
        while maintaining accuracy.

        Args:
            all_states: All compressed layer states to consider.
            evaluate_fn: Evaluation function for combinations.
            max_layers: Maximum layers to include in a combination.
            require_contiguous: Only consider contiguous layer ranges.
            accuracy_threshold: Minimum accuracy to be considered "safe".

        Returns:
            List of safe CompositionResults, sorted by compression ratio.
        """
        safe_combinations = []

        # Sort by calibration accuracy (best first)
        sorted_states = sorted(
            all_states,
            key=lambda s: s.calibration_accuracy,
            reverse=True
        )

        if require_contiguous:
            # Test contiguous ranges
            safe_combinations = self._find_contiguous_safe(
                all_states, evaluate_fn, accuracy_threshold
            )
        else:
            # Greedy search
            safe_combinations = self._greedy_search(
                sorted_states, evaluate_fn, max_layers, accuracy_threshold
            )

        # Sort by compression ratio (most compression first)
        safe_combinations.sort(key=lambda c: c.compression_ratio, reverse=True)

        logger.info(
            f"COMPOSABLE: Found {len(safe_combinations)} safe combinations "
            f"(threshold={accuracy_threshold:.1%})"
        )

        return safe_combinations

    def _find_contiguous_safe(
        self,
        all_states: list[LayerCompressionState],
        evaluate_fn: Callable[[list[int], list["Array"]], float],
        accuracy_threshold: float,
    ) -> list[CompositionResult]:
        """Find safe contiguous layer ranges."""
        safe_combinations = []

        # Sort by layer index
        sorted_states = sorted(all_states, key=lambda s: s.layer_idx)
        n = len(sorted_states)

        # Try all contiguous ranges
        for start in range(n):
            for end in range(start + 1, n + 1):
                subset = sorted_states[start:end]
                result = self.evaluate_composition(subset, evaluate_fn)

                if result.overall_accuracy >= accuracy_threshold:
                    safe_combinations.append(result)

        return safe_combinations

    def _greedy_search(
        self,
        sorted_states: list[LayerCompressionState],
        evaluate_fn: Callable[[list[int], list["Array"]], float],
        max_layers: int,
        accuracy_threshold: float,
    ) -> list[CompositionResult]:
        """Greedy search for safe combinations."""
        safe_combinations = []

        # Start with each individual layer
        for state in sorted_states:
            result = self.evaluate_composition([state], evaluate_fn)
            if result.overall_accuracy >= accuracy_threshold:
                safe_combinations.append(result)

        # Greedy expansion: add layers one at a time
        current_set = []
        remaining = list(sorted_states)

        while len(current_set) < max_layers and remaining:
            best_addition = None
            best_result = None
            best_accuracy = -1

            for candidate in remaining:
                test_set = current_set + [candidate]
                result = self.evaluate_composition(test_set, evaluate_fn)

                if result.overall_accuracy >= accuracy_threshold:
                    if result.overall_accuracy > best_accuracy:
                        best_accuracy = result.overall_accuracy
                        best_addition = candidate
                        best_result = result

            if best_addition is not None:
                current_set.append(best_addition)
                remaining.remove(best_addition)
                safe_combinations.append(best_result)
            else:
                break  # No safe addition found

        return safe_combinations

    def profile_all_layers(
        self,
        layer_data: list[tuple["Array", "Array", int]],
    ) -> dict[int, GeodesicLayerProfile]:
        """Profile all layers without compression.

        Useful for predicting which layers are compressible before
        committing to compression.

        Args:
            layer_data: List of (X, Y, layer_idx) tuples.

        Returns:
            Dict mapping layer_idx to GeodesicLayerProfile.
        """
        profiles = {}

        for X, Y, layer_idx in layer_data:
            b = self._backend
            X = b.array(X)
            b.eval(X)

            try:
                profile = self._analyzer.analyze(X)
                profiles[layer_idx] = profile
                logger.info(
                    f"COMPOSABLE: Layer {layer_idx} profile: "
                    f"geodesic_rank={profile.geodesic_rank}, "
                    f"rmt_signal_rank={profile.rmt_signal_rank}, "
                    f"compressibility={profile.compressibility_score:.3f}"
                )
            except Exception as e:
                logger.warning(f"Failed to profile layer {layer_idx}: {e}")

        return profiles

    def recommend_layers(
        self,
        profiles: dict[int, GeodesicLayerProfile],
        min_compressibility: float,
    ) -> list[int]:
        """Recommend layers for compression based on profiles.

        Args:
            profiles: Dict mapping layer_idx to GeodesicLayerProfile.
            min_compressibility: Minimum compressibility score (required).
                Score range is [0, 1] where 1 = highly compressible.
                No default - caller must choose threshold for their use case.

        Returns:
            List of layer indices recommended for compression.
        """
        recommended = []

        for layer_idx, profile in profiles.items():
            if profile.compressibility_score >= min_compressibility:
                recommended.append(layer_idx)

        recommended.sort()

        logger.info(
            f"COMPOSABLE: Recommended {len(recommended)} layers for compression: {recommended}"
        )

        return recommended


def compress_model_layers(
    layer_data: list[tuple["Array", "Array", int]],
    evaluate_fn: Callable[[list[int], list["Array"]], float],
    backend: "Backend | None" = None,
    accuracy_threshold: float = 1.0,
) -> CompositionResult | None:
    """Convenience function to find the best safe layer combination.

    Args:
        layer_data: List of (X, Y, layer_idx) tuples.
        evaluate_fn: Function to evaluate combinations.
        backend: Backend to use (uses default if None).
        accuracy_threshold: Minimum accuracy for safety.

    Returns:
        Best CompositionResult, or None if no safe combination found.
    """
    b = backend or get_default_backend()
    compressor = ComposableLayerCompressor(backend=b)

    # Compress all layers
    states = compressor.compress_sequence(layer_data, analyze_geodesic=False)

    # Find safe combinations
    safe = compressor.find_safe_combinations(
        states,
        evaluate_fn,
        accuracy_threshold=accuracy_threshold,
    )

    if safe:
        return safe[0]  # Return the one with highest compression ratio
    return None
