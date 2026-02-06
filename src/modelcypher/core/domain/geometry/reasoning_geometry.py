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

"""Unified reasoning geometry analysis combining three validated signals.

Composes three independently validated measurements of reasoning geometry
into a single coherent analysis:

1. **Linear probes** (Zhang et al. 2025, Marks & Tegmark 2024): Correctness
   is linearly encoded in hidden states. AUROC=0.686 at layer 8 on GSM8K
   with LFM2.5-1.2B. Signal peaks in middle layers, not output layer.

2. **Cognitive pivots** (Lee et al. 2026, STARS method): Trajectory
   discontinuities (L2 spikes) indicate decision points. Incorrect answers
   have 42% more pivots (d=-0.90).

3. **β₁ topology** (persistent homology): Reasoning creates topological
   loops (β₁ > 0) in the token manifold. Δβ₁ > 0 (loops persist to exit)
   predicts correct reasoning; Δβ₁ < 0 (loops collapse) predicts incorrect.

This module delegates to existing implementations and adds no new math.
All thresholds come from the referenced papers or dtype-derived constants.

EXPERIMENTAL: Not promoted to CLI. Needs cross-model validation.

References:
- Zhang et al. (2025) arXiv:2504.05419
- Marks & Tegmark (2024) arXiv:2310.06824
- Lee et al. (2026) arXiv:2601.22484
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cognitive_pivots import (
    CognitivePivot,
    CognitivePivotDetector,
)
from modelcypher.core.domain.geometry.topological_fingerprint import (
    TopologicalFingerprint,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class TopologySignal:
    """Per-layer Betti numbers and Δβ₁ trajectory.

    β₁ measures topological loops in the token manifold at each layer.
    Reasoning prompts create loops (circular dependencies between tokens);
    narrative prompts do not.

    Δβ₁ = mean(β₁ late layers) - mean(β₁ early layers).
    Positive Δβ₁ means loops persist through exit layers (correct reasoning).
    Negative Δβ₁ means loops collapse before exit (incorrect reasoning).

    Attributes:
        betti_by_layer: Betti numbers at each layer. Keys are dimension (0, 1, 2).
        beta1_by_layer: β₁ value at each layer for easy access.
        delta_beta1: Late β₁ minus early β₁.
        mean_beta1: Mean β₁ across all layers.
        persistence_entropy_by_layer: Persistence entropy at each layer.
        layer_indices: Which layers were analyzed.
    """

    betti_by_layer: tuple[dict[int, int], ...]
    beta1_by_layer: tuple[int, ...]
    delta_beta1: float
    mean_beta1: float
    persistence_entropy_by_layer: tuple[float, ...]
    layer_indices: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "betti_by_layer": [dict(b) for b in self.betti_by_layer],
            "beta1_by_layer": list(self.beta1_by_layer),
            "delta_beta1": self.delta_beta1,
            "mean_beta1": self.mean_beta1,
            "persistence_entropy_by_layer": list(self.persistence_entropy_by_layer),
            "layer_indices": list(self.layer_indices),
        }


@dataclass(frozen=True)
class ProbeSignal:
    """Per-layer linear probe results for correctness prediction.

    Trained on hidden states from a batch of correct/incorrect trajectories.
    AUROC > 0.65 indicates the model encodes correctness at that layer.

    Attributes:
        auroc_by_layer: AUROC at each layer.
        best_layer: Layer with highest AUROC.
        best_auroc: Highest AUROC across layers.
        separation_by_layer: Fisher separation score at each layer.
        layer_indices: Which layers were probed.
    """

    auroc_by_layer: tuple[float, ...]
    best_layer: int
    best_auroc: float
    separation_by_layer: tuple[float, ...]
    layer_indices: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "auroc_by_layer": list(self.auroc_by_layer),
            "best_layer": self.best_layer,
            "best_auroc": self.best_auroc,
            "separation_by_layer": list(self.separation_by_layer),
            "layer_indices": list(self.layer_indices),
        }


@dataclass(frozen=True)
class PivotSignal:
    """Cognitive pivot statistics from trajectory analysis.

    Pivots are L2 distance spikes between consecutive token hidden states.
    More pivots correlate with incorrect reasoning (d=-0.90).

    Attributes:
        pivot_count: Number of detected cognitive pivots.
        mean_l2_distance: Mean L2 distance across all tokens and layers.
        std_l2_distance: Standard deviation of L2 distances.
        pivots: The detected pivot objects.
    """

    pivot_count: int
    mean_l2_distance: float
    std_l2_distance: float
    pivots: tuple[CognitivePivot, ...]

    def as_dict(self) -> dict:
        return {
            "pivot_count": self.pivot_count,
            "mean_l2_distance": self.mean_l2_distance,
            "std_l2_distance": self.std_l2_distance,
        }


@dataclass(frozen=True)
class ReasoningGeometryResult:
    """Combined reasoning geometry measurement from all three signals.

    Attributes:
        topology: β₁ topology signal (Δβ₁ trajectory).
        probe: Linear probe signal (per-layer AUROC). None if no labels.
        pivots: Cognitive pivot signal (spike count and statistics).
        n_layers: Number of model layers analyzed.
        n_tokens: Number of tokens in the trajectory.
    """

    topology: TopologySignal
    probe: ProbeSignal | None
    pivots: PivotSignal
    n_layers: int
    n_tokens: int

    def as_dict(self) -> dict:
        result = {
            "topology": self.topology.as_dict(),
            "pivots": self.pivots.as_dict(),
            "n_layers": self.n_layers,
            "n_tokens": self.n_tokens,
        }
        if self.probe is not None:
            result["probe"] = self.probe.as_dict()
        return result


class ReasoningGeometryAnalyzer:
    """Unified reasoning geometry analysis combining three validated signals.

    Computes topology (β₁), cognitive pivots, and optionally linear probes
    from generation trajectory hidden states.

    Example:
        >>> analyzer = ReasoningGeometryAnalyzer()
        >>> result = analyzer.analyze_trajectory(hidden_states_seq, tokens)
        >>> print(f"Δβ₁: {result.topology.delta_beta1}")
        >>> print(f"Pivots: {result.pivots.pivot_count}")
    """

    def __init__(
        self,
        backend: Backend | None = None,
        spike_threshold_k: float = 2.0,
        max_tda_points: int = 50,
    ) -> None:
        """Initialize the analyzer.

        Args:
            backend: Backend for tensor operations. If None, uses default.
            spike_threshold_k: Threshold for pivot detection (std devs above
                mean). Default 2.0 per Lee et al. 2026 (arXiv:2601.22484,
                Section 3.2). This is a parameter, not a constant - callers
                should tune based on their model and task.
            max_tda_points: Maximum points for PCA reduction before TDA.
                Standard practice for high-dimensional persistent homology.
        """
        self._backend = backend or get_default_backend()
        self._pivot_detector = CognitivePivotDetector(
            backend=self._backend,
            spike_threshold_k=spike_threshold_k,
        )
        self._max_tda_points = max_tda_points

    def analyze_trajectory(
        self,
        layer_hidden_states_sequence: list[dict[int, Array]],
        tokens: list[str],
    ) -> ReasoningGeometryResult:
        """Analyze a single generation trajectory for reasoning geometry.

        Computes topology (β₁) and cognitive pivots from the trajectory.
        Linear probes require labeled batches; use analyze_batch() for that.

        Args:
            layer_hidden_states_sequence: List of per-layer hidden states for
                each generated token. Each entry maps layer_idx -> Array[hidden_dim].
            tokens: List of generated token strings.

        Returns:
            ReasoningGeometryResult with topology and pivot signals.
            Probe signal will be None (needs batch with labels).
        """
        if not layer_hidden_states_sequence:
            return self._empty_result()

        # Get layer indices from first token's hidden states
        layer_indices = sorted(layer_hidden_states_sequence[0].keys())
        n_layers = len(layer_indices)
        n_tokens = len(layer_hidden_states_sequence)

        # Compute pivot signal
        pivot_signal = self._compute_pivot_signal(
            layer_hidden_states_sequence, tokens
        )

        # Compute topology signal
        topology_signal = self._compute_topology_signal(
            layer_hidden_states_sequence, layer_indices
        )

        return ReasoningGeometryResult(
            topology=topology_signal,
            probe=None,
            pivots=pivot_signal,
            n_layers=n_layers,
            n_tokens=n_tokens,
        )

    def analyze_batch(
        self,
        trajectories: list[
            tuple[list[dict[int, Array]], list[str], bool | None]
        ],
    ) -> tuple[list[ReasoningGeometryResult], ProbeSignal | None]:
        """Analyze multiple trajectories and train per-layer probes if labeled.

        Args:
            trajectories: List of (hidden_states_sequence, tokens, is_correct)
                tuples. is_correct=None means unlabeled.

        Returns:
            Tuple of (list of per-trajectory results, probe signal or None).
            The probe signal is computed across all labeled trajectories.
        """
        if not trajectories:
            return [], None

        # Analyze each trajectory individually for topology + pivots
        results = []
        for hidden_states_seq, tokens, _ in trajectories:
            result = self.analyze_trajectory(hidden_states_seq, tokens)
            results.append(result)

        # Train per-layer probes if we have labeled data
        correct_trajectories = [
            (hs, tok) for hs, tok, label in trajectories if label is True
        ]
        incorrect_trajectories = [
            (hs, tok) for hs, tok, label in trajectories if label is False
        ]

        probe_signal = None
        if correct_trajectories and incorrect_trajectories:
            probe_signal = self._compute_probe_signal(
                correct_trajectories, incorrect_trajectories
            )

            # Update results with probe signal
            results = [
                ReasoningGeometryResult(
                    topology=r.topology,
                    probe=probe_signal,
                    pivots=r.pivots,
                    n_layers=r.n_layers,
                    n_tokens=r.n_tokens,
                )
                for r in results
            ]

        return results, probe_signal

    def _compute_pivot_signal(
        self,
        layer_hidden_states_sequence: list[dict[int, Array]],
        tokens: list[str],
    ) -> PivotSignal:
        """Compute cognitive pivot signal from trajectory."""
        trajectory = self._pivot_detector.analyze_trajectory(
            layer_hidden_states_sequence, tokens
        )

        return PivotSignal(
            pivot_count=len(trajectory.pivots),
            mean_l2_distance=trajectory.mean_l2_distance,
            std_l2_distance=trajectory.std_l2_distance,
            pivots=trajectory.pivots,
        )

    def _compute_topology_signal(
        self,
        layer_hidden_states_sequence: list[dict[int, Array]],
        layer_indices: list[int],
    ) -> TopologySignal:
        """Compute per-layer topology (β₁) from trajectory hidden states.

        At each layer, the token hidden states form a point cloud.
        We compute persistent homology on this cloud to get Betti numbers.
        """
        b = self._backend
        n_tokens = len(layer_hidden_states_sequence)

        if n_tokens < 2 or not layer_indices:
            return TopologySignal(
                betti_by_layer=(),
                beta1_by_layer=(),
                delta_beta1=0.0,
                mean_beta1=0.0,
                persistence_entropy_by_layer=(),
                layer_indices=tuple(layer_indices),
            )

        betti_by_layer = []
        beta1_by_layer = []
        entropy_by_layer = []

        for layer_idx in layer_indices:
            # Collect hidden states at this layer across all tokens
            points_arrays = []
            for token_states in layer_hidden_states_sequence:
                if layer_idx in token_states:
                    arr = token_states[layer_idx]
                    if not hasattr(arr, "shape"):
                        arr = b.array(arr)
                    b.eval(arr)
                    points_arrays.append(arr)

            if len(points_arrays) < 2:
                betti_by_layer.append({0: len(points_arrays), 1: 0})
                beta1_by_layer.append(0)
                entropy_by_layer.append(0.0)
                continue

            # Stack into [n_tokens, hidden_dim]
            point_cloud = b.stack(points_arrays, axis=0)
            b.eval(point_cloud)

            # PCA reduction if hidden_dim > max_tda_points
            # Standard TDA practice for high-dimensional data
            n_points = int(b.shape(point_cloud)[0])
            hidden_dim = int(b.shape(point_cloud)[1])
            target_dim = min(self._max_tda_points, n_points - 1, hidden_dim)

            if hidden_dim > target_dim and target_dim > 0:
                point_cloud = self._pca_reduce(point_cloud, target_dim)

            # Convert to Python lists for TopologicalFingerprint
            b.eval(point_cloud)
            points_list = b.tolist(point_cloud)

            # Compute persistent homology
            fingerprint = TopologicalFingerprint.compute(points_list)

            betti = fingerprint.betti_numbers
            betti_by_layer.append(dict(betti))
            beta1_by_layer.append(betti.get(1, 0))
            entropy_by_layer.append(fingerprint.summary.persistence_entropy)

        # Compute Δβ₁: late minus early
        n_layers = len(layer_indices)
        third = max(1, n_layers // 3)
        early_beta1 = beta1_by_layer[:third]
        late_beta1 = beta1_by_layer[-third:]

        mean_early = sum(early_beta1) / len(early_beta1) if early_beta1 else 0.0
        mean_late = sum(late_beta1) / len(late_beta1) if late_beta1 else 0.0
        delta_beta1 = mean_late - mean_early

        all_beta1 = beta1_by_layer
        mean_beta1 = sum(all_beta1) / len(all_beta1) if all_beta1 else 0.0

        return TopologySignal(
            betti_by_layer=tuple(betti_by_layer),
            beta1_by_layer=tuple(beta1_by_layer),
            delta_beta1=delta_beta1,
            mean_beta1=mean_beta1,
            persistence_entropy_by_layer=tuple(entropy_by_layer),
            layer_indices=tuple(layer_indices),
        )

    def _compute_probe_signal(
        self,
        correct_trajectories: list[tuple[list[dict[int, Array]], list[str]]],
        incorrect_trajectories: list[tuple[list[dict[int, Array]], list[str]]],
    ) -> ProbeSignal:
        """Train per-layer linear probes on labeled trajectory batches.

        For each layer, collects the mean hidden state across all tokens
        in each trajectory, then trains a difference-in-means probe.
        """
        from modelcypher.core.domain.geometry.linear_probe import (
            CorrectnessProbe,
            compute_difference_in_means,
        )

        b = self._backend

        # Determine layer indices from first trajectory
        first_seq = correct_trajectories[0][0]
        if not first_seq:
            return ProbeSignal(
                auroc_by_layer=(),
                best_layer=-1,
                best_auroc=0.5,
                separation_by_layer=(),
                layer_indices=(),
            )
        layer_indices = sorted(first_seq[0].keys())

        auroc_by_layer = []
        separation_by_layer = []

        for layer_idx in layer_indices:
            # Collect mean hidden state per trajectory at this layer
            correct_states = self._collect_layer_means(
                correct_trajectories, layer_idx
            )
            incorrect_states = self._collect_layer_means(
                incorrect_trajectories, layer_idx
            )

            if not correct_states or not incorrect_states:
                auroc_by_layer.append(0.5)
                separation_by_layer.append(0.0)
                continue

            # Split into train/test (80/20)
            n_correct = len(correct_states)
            n_incorrect = len(incorrect_states)
            n_train_c = max(1, int(n_correct * 0.8))
            n_train_i = max(1, int(n_incorrect * 0.8))

            train_correct = correct_states[:n_train_c]
            test_correct = correct_states[n_train_c:] or correct_states[-1:]
            train_incorrect = incorrect_states[:n_train_i]
            test_incorrect = incorrect_states[n_train_i:] or incorrect_states[-1:]

            # Train probe
            probe = CorrectnessProbe(backend=b, method="difference_in_means")
            probe.fit(train_correct, train_incorrect)

            # Evaluate
            auroc, _ = probe.evaluate(test_correct, test_incorrect)
            auroc_by_layer.append(auroc)

            # Fisher separation
            _, separation = compute_difference_in_means(
                correct_states, incorrect_states, backend=b
            )
            separation_by_layer.append(separation)

        # Find best layer
        if auroc_by_layer:
            best_idx = max(range(len(auroc_by_layer)), key=lambda i: auroc_by_layer[i])
            best_layer = layer_indices[best_idx]
            best_auroc = auroc_by_layer[best_idx]
        else:
            best_layer = -1
            best_auroc = 0.5

        return ProbeSignal(
            auroc_by_layer=tuple(auroc_by_layer),
            best_layer=best_layer,
            best_auroc=best_auroc,
            separation_by_layer=tuple(separation_by_layer),
            layer_indices=tuple(layer_indices),
        )

    def _collect_layer_means(
        self,
        trajectories: list[tuple[list[dict[int, Array]], list[str]]],
        layer_idx: int,
    ) -> list[Array]:
        """Collect mean hidden state at a layer across tokens in each trajectory."""
        b = self._backend
        means = []

        for hidden_states_seq, _ in trajectories:
            layer_states = []
            for token_states in hidden_states_seq:
                if layer_idx in token_states:
                    arr = token_states[layer_idx]
                    if not hasattr(arr, "shape"):
                        arr = b.array(arr)
                    layer_states.append(arr)

            if layer_states:
                stacked = b.stack(layer_states, axis=0)
                mean_state = b.mean(stacked, axis=0)
                b.eval(mean_state)
                means.append(mean_state)

        return means

    def _pca_reduce(self, data: Array, target_dim: int) -> Array:
        """Reduce dimensionality via PCA (center, SVD, project).

        Args:
            data: Array of shape [n_points, n_features].
            target_dim: Number of dimensions to keep.

        Returns:
            Projected array of shape [n_points, target_dim].
        """
        b = self._backend

        # Center the data
        mean = b.mean(data, axis=0)
        b.eval(mean)
        centered = data - mean
        b.eval(centered)

        # SVD
        U, S, Vt = b.svd(centered, full_matrices=False)
        b.eval(U, S, Vt)

        # Project onto top-k components
        # centered @ V[:, :target_dim] = U[:, :target_dim] * S[:target_dim]
        U_k = U[:, :target_dim]
        S_k = S[:target_dim]
        projected = U_k * S_k
        b.eval(projected)

        return projected

    def _empty_result(self) -> ReasoningGeometryResult:
        """Return an empty result for degenerate inputs."""
        return ReasoningGeometryResult(
            topology=TopologySignal(
                betti_by_layer=(),
                beta1_by_layer=(),
                delta_beta1=0.0,
                mean_beta1=0.0,
                persistence_entropy_by_layer=(),
                layer_indices=(),
            ),
            probe=None,
            pivots=PivotSignal(
                pivot_count=0,
                mean_l2_distance=0.0,
                std_l2_distance=0.0,
                pivots=(),
            ),
            n_layers=0,
            n_tokens=0,
        )


def analyze_reasoning_geometry(
    layer_hidden_states_sequence: list[dict[int, Array]],
    tokens: list[str],
    backend: Backend | None = None,
) -> ReasoningGeometryResult:
    """Convenience function to analyze reasoning geometry of a single trajectory.

    Args:
        layer_hidden_states_sequence: List of per-layer hidden states for
            each generated token.
        tokens: List of generated token strings.
        backend: Backend for tensor operations.

    Returns:
        ReasoningGeometryResult with topology and pivot signals.
    """
    analyzer = ReasoningGeometryAnalyzer(backend=backend)
    return analyzer.analyze_trajectory(layer_hidden_states_sequence, tokens)


__all__ = [
    "PivotSignal",
    "ProbeSignal",
    "ReasoningGeometryAnalyzer",
    "ReasoningGeometryResult",
    "TopologySignal",
    "analyze_reasoning_geometry",
]
