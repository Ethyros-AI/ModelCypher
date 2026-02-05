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

"""Cognitive pivot detection via L2 distance spikes (STARS method).

Implements cognitive pivot detection from Lee et al. 2026 "Training-Free
Cognitive Steering in Large Language Models through State-Space Analysis"
(arXiv:2601.22484).

Key insight: Cognitive pivots - moments where the model makes a decision or
shifts reasoning strategy - are detectable as L2 distance spikes in hidden
states between consecutive tokens. These spikes indicate trajectory
discontinuities where the model's representation makes a sharp turn.

The STARS method identifies these pivots without training by:
1. Computing L2 distance between consecutive hidden states at each layer
2. Detecting spikes using adaptive threshold (mean + k*std)
3. Classifying pivots based on cross-layer spike patterns

This is TRAJECTORY analysis - we look at the full generation path, not just
the answer token. The signal is in WHERE the model turns, not WHERE it ends up.

Reference: Lee et al. (2026) arXiv:2601.22484
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class CognitivePivot:
    """A detected cognitive pivot in the generation trajectory.

    Attributes:
        token_idx: Index of the token where the pivot occurred.
        token_str: String representation of the token (if available).
        l2_distance: Mean L2 distance across layers at this position.
        l2_distances_by_layer: L2 distance at each layer.
        spike_score: How many standard deviations above mean (z-score).
        layers_with_spike: Which layers showed spikes at this position.
    """

    token_idx: int
    token_str: str
    l2_distance: float
    l2_distances_by_layer: tuple[float, ...]
    spike_score: float
    layers_with_spike: tuple[int, ...]


@dataclass(frozen=True)
class TrajectoryPoint:
    """A single point in the generation trajectory.

    Attributes:
        token_idx: Index of this token in the sequence.
        token_str: String representation of the token.
        hidden_states: Hidden states at each layer [num_layers, hidden_dim].
        l2_from_previous: L2 distance from previous token at each layer.
        is_pivot: Whether this point is a cognitive pivot.
    """

    token_idx: int
    token_str: str
    hidden_states: tuple[tuple[float, ...], ...]  # Nested tuple for immutability
    l2_from_previous: tuple[float, ...]
    is_pivot: bool


@dataclass(frozen=True)
class ReasoningTrajectory:
    """Complete reasoning trajectory with cognitive pivots.

    Attributes:
        prompt: The input prompt.
        generated_tokens: List of generated token strings.
        points: TrajectoryPoints for each generated token.
        pivots: Detected cognitive pivots.
        is_correct: Whether the final answer was correct (if known).
        mean_l2_distance: Mean L2 distance across all tokens and layers.
        std_l2_distance: Std dev of L2 distances.
    """

    prompt: str
    generated_tokens: tuple[str, ...]
    points: tuple[TrajectoryPoint, ...]
    pivots: tuple[CognitivePivot, ...]
    is_correct: bool | None
    mean_l2_distance: float
    std_l2_distance: float


class CognitivePivotDetector:
    """Detects cognitive pivots via L2 distance spikes in hidden states.

    Following Lee et al. 2026, this detects moments where the model makes
    decisions or shifts reasoning strategy by looking for L2 spikes between
    consecutive token hidden states.
    """

    def __init__(
        self,
        backend: Backend | None = None,
        spike_threshold_k: float = 2.0,
        min_layers_for_pivot: int = 1,
    ) -> None:
        """Initialize the detector.

        Args:
            backend: Backend for tensor operations. If None, uses default.
            spike_threshold_k: Number of standard deviations above mean to
                consider a spike. Default 2.0 (top ~5% of distances).
            min_layers_for_pivot: Minimum number of layers that must show
                a spike to classify as a cognitive pivot. Default 1.
        """
        self._backend = backend or get_default_backend()
        self._spike_threshold_k = spike_threshold_k
        self._min_layers_for_pivot = min_layers_for_pivot

    def compute_l2_distances(
        self,
        hidden_states: dict[int, Array],
        previous_hidden_states: dict[int, Array] | None,
    ) -> dict[int, float]:
        """Compute L2 distances from previous hidden states at each layer.

        Args:
            hidden_states: Current hidden states per layer [hidden_dim].
            previous_hidden_states: Previous token hidden states per layer.

        Returns:
            Dict mapping layer index to L2 distance.
        """
        b = self._backend

        if previous_hidden_states is None:
            # First token has no previous - return zeros
            return {layer: 0.0 for layer in hidden_states}

        distances = {}
        for layer_idx in hidden_states:
            if layer_idx not in previous_hidden_states:
                distances[layer_idx] = 0.0
                continue

            current = hidden_states[layer_idx]
            previous = previous_hidden_states[layer_idx]

            diff = current - previous
            l2 = b.sqrt(b.sum(diff * diff))
            b.eval(l2)
            distances[layer_idx] = float(b.to_scalar(l2))

        return distances

    def detect_spikes(
        self,
        all_l2_distances: list[dict[int, float]],
    ) -> list[tuple[int, float, tuple[int, ...]]]:
        """Detect L2 spikes across a trajectory.

        Args:
            all_l2_distances: List of per-layer L2 distances for each token.

        Returns:
            List of (token_idx, spike_score, layers_with_spike) for detected spikes.
        """
        if len(all_l2_distances) < 2:
            return []

        # Get all layer indices
        layer_indices = sorted(all_l2_distances[0].keys())
        if not layer_indices:
            return []

        # Compute per-layer statistics
        layer_stats: dict[int, tuple[float, float]] = {}  # layer -> (mean, std)
        for layer_idx in layer_indices:
            layer_distances = [d[layer_idx] for d in all_l2_distances if layer_idx in d]
            if len(layer_distances) < 2:
                layer_stats[layer_idx] = (0.0, 1.0)
                continue

            mean_dist = sum(layer_distances) / len(layer_distances)
            variance = sum((d - mean_dist) ** 2 for d in layer_distances) / len(layer_distances)
            std_dist = variance ** 0.5 if variance > 0 else 1e-8
            layer_stats[layer_idx] = (mean_dist, std_dist)

        # Detect spikes at each token position
        spikes = []
        for token_idx, distances in enumerate(all_l2_distances):
            if token_idx == 0:
                continue  # Skip first token

            layers_with_spike = []
            spike_scores = []

            for layer_idx in layer_indices:
                if layer_idx not in distances:
                    continue

                mean, std = layer_stats[layer_idx]
                if std < 1e-10:
                    continue

                z_score = (distances[layer_idx] - mean) / std
                if z_score > self._spike_threshold_k:
                    layers_with_spike.append(layer_idx)
                    spike_scores.append(z_score)

            if len(layers_with_spike) >= self._min_layers_for_pivot:
                mean_spike_score = sum(spike_scores) / len(spike_scores) if spike_scores else 0.0
                spikes.append((token_idx, mean_spike_score, tuple(layers_with_spike)))

        return spikes

    def analyze_trajectory(
        self,
        layer_hidden_states_sequence: list[dict[int, Array]],
        tokens: list[str],
        prompt: str = "",
        is_correct: bool | None = None,
    ) -> ReasoningTrajectory:
        """Analyze a complete generation trajectory for cognitive pivots.

        Args:
            layer_hidden_states_sequence: List of per-layer hidden states for
                each generated token. Each entry is dict[layer_idx, hidden_state].
            tokens: List of generated token strings.
            prompt: The input prompt.
            is_correct: Whether the final answer was correct (if known).

        Returns:
            ReasoningTrajectory with detected cognitive pivots.
        """
        b = self._backend

        if not layer_hidden_states_sequence:
            return ReasoningTrajectory(
                prompt=prompt,
                generated_tokens=tuple(tokens),
                points=(),
                pivots=(),
                is_correct=is_correct,
                mean_l2_distance=0.0,
                std_l2_distance=0.0,
            )

        # Compute L2 distances for each token
        all_l2_distances: list[dict[int, float]] = []
        previous_states: dict[int, Array] | None = None

        for states in layer_hidden_states_sequence:
            distances = self.compute_l2_distances(states, previous_states)
            all_l2_distances.append(distances)
            previous_states = states

        # Detect spikes
        spikes = self.detect_spikes(all_l2_distances)
        spike_indices = {s[0] for s in spikes}

        # Build trajectory points
        points = []
        for idx, (states, distances) in enumerate(zip(layer_hidden_states_sequence, all_l2_distances)):
            token_str = tokens[idx] if idx < len(tokens) else ""

            # Convert hidden states to nested tuples
            layer_indices = sorted(states.keys())
            hidden_states_tuple = tuple(
                tuple(b.tolist(states[layer_idx])) for layer_idx in layer_indices
            )
            l2_tuple = tuple(distances[layer_idx] for layer_idx in layer_indices)

            points.append(TrajectoryPoint(
                token_idx=idx,
                token_str=token_str,
                hidden_states=hidden_states_tuple,
                l2_from_previous=l2_tuple,
                is_pivot=idx in spike_indices,
            ))

        # Build cognitive pivot objects
        pivots = []
        for token_idx, spike_score, layers_with_spike in spikes:
            distances = all_l2_distances[token_idx]
            layer_indices = sorted(distances.keys())
            l2_by_layer = tuple(distances[layer_idx] for layer_idx in layer_indices)
            mean_l2 = sum(l2_by_layer) / len(l2_by_layer) if l2_by_layer else 0.0

            pivots.append(CognitivePivot(
                token_idx=token_idx,
                token_str=tokens[token_idx] if token_idx < len(tokens) else "",
                l2_distance=mean_l2,
                l2_distances_by_layer=l2_by_layer,
                spike_score=spike_score,
                layers_with_spike=layers_with_spike,
            ))

        # Compute overall statistics
        all_distances = [
            d for distances in all_l2_distances for d in distances.values()
        ]
        if all_distances:
            mean_l2 = sum(all_distances) / len(all_distances)
            variance = sum((d - mean_l2) ** 2 for d in all_distances) / len(all_distances)
            std_l2 = variance ** 0.5
        else:
            mean_l2 = 0.0
            std_l2 = 0.0

        return ReasoningTrajectory(
            prompt=prompt,
            generated_tokens=tuple(tokens),
            points=tuple(points),
            pivots=tuple(pivots),
            is_correct=is_correct,
            mean_l2_distance=mean_l2,
            std_l2_distance=std_l2,
        )


def detect_cognitive_pivots(
    layer_hidden_states_sequence: list[dict[int, Array]],
    tokens: list[str],
    backend: Backend | None = None,
    spike_threshold_k: float = 2.0,
) -> list[CognitivePivot]:
    """Convenience function to detect cognitive pivots in a trajectory.

    Args:
        layer_hidden_states_sequence: List of per-layer hidden states for
            each generated token.
        tokens: List of generated token strings.
        backend: Backend for tensor operations.
        spike_threshold_k: Spike detection threshold (std devs above mean).

    Returns:
        List of detected CognitivePivot objects.
    """
    detector = CognitivePivotDetector(backend=backend, spike_threshold_k=spike_threshold_k)
    trajectory = detector.analyze_trajectory(layer_hidden_states_sequence, tokens)
    return list(trajectory.pivots)


__all__ = [
    "CognitivePivot",
    "CognitivePivotDetector",
    "ReasoningTrajectory",
    "TrajectoryPoint",
    "detect_cognitive_pivots",
]
