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
Reasoning Flow Geometry (Zhou et al., ICLR 2026)

Implements the geometric framework from "The Geometry of Reasoning: Flowing
Logics in Representation Space" for analyzing reasoning trajectories.

Key insight: LLM reasoning forms smooth flows in embedding space. Logical
statements act as local controllers governing the velocity of these flows.

Order-0 (Positions): Embeddings cluster by surface-level semantics (topic, language)
Order-1 (Velocities): Trajectories with same logic structure align across semantics
Order-2 (Menger Curvature): Logic signal intensifies beyond surface semantics

Uses the Backend protocol for all tensor operations - no numpy.

Reference: Zhou et al. (2025) arXiv:2510.09782
GitHub: https://github.com/MasterZhou1/Reasoning-Flow
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

Array = TypeVar("Array")


@dataclass(frozen=True)
class FlowMetrics:
    """Geometric metrics for a reasoning flow trajectory."""

    # Order-0: Position statistics
    arc_length: float  # Total path length in embedding space
    mean_speed: float  # Average step size

    # Order-1: Velocity statistics
    velocities: Any  # Backend array [T-1, hidden_dim]
    mean_velocity_norm: float  # Average velocity magnitude
    velocity_variance: float  # Variance in velocity magnitudes

    # Order-2: Curvature statistics
    curvatures: Any  # Backend array [T-2] Menger curvatures
    mean_curvature: float  # Average curvature
    max_curvature: float  # Peak curvature (sharpest turn)
    curvature_integral: float  # Total absolute curvature (bending energy)

    # Derived metrics
    smoothness: float  # 1 / (1 + mean_curvature) - higher = smoother
    directness: float  # straight_line_dist / arc_length - higher = more direct


class ReasoningFlowAnalyzer:
    """Analyzer for reasoning flow geometry using backend operations."""

    def __init__(self, backend: Backend) -> None:
        self._b = backend
        self._eps = backend.finfo().eps

    def compute_velocities(self, positions: Array) -> Array:
        """
        Compute first-order differences (velocities) from position trajectory.

        Args:
            positions: [T, hidden_dim] array of positions

        Returns:
            [T-1, hidden_dim] array of velocity vectors
        """
        T = int(self._b.shape(positions)[0])
        if T < 2:
            hidden_dim = int(self._b.shape(positions)[1]) if T > 0 else 1
            return self._b.zeros((0, hidden_dim))
        return positions[1:] - positions[:-1]

    def compute_menger_curvature(self, positions: Array) -> Array:
        """
        Compute Menger curvature per interior step for a position trajectory.

        Menger curvature κ for triplet (p_{i-1}, p_i, p_{i+1}) is:
            κ = 4A / (abc)
        where:
            a = ||p_i - p_{i-1}||
            b = ||p_{i+1} - p_i||
            c = ||p_{i+1} - p_{i-1}||
            A = triangle area via Heron's formula

        This is the discrete analog of continuous curvature and measures how
        sharply the trajectory bends at each point.

        Args:
            positions: [T, hidden_dim] array of positions

        Returns:
            [T-2] array of curvature values at interior points

        Reference: Zhou et al. (2025) use this as order-2 geometric descriptor
        """
        T = int(self._b.shape(positions)[0])
        if T < 3:
            return self._b.zeros((0,))

        # Three consecutive points
        p_prev = positions[:-2]  # p_{i-1}
        p_curr = positions[1:-1]  # p_i
        p_next = positions[2:]  # p_{i+1}

        # Edge lengths
        a = self._b.norm(p_curr - p_prev, axis=1)
        b = self._b.norm(p_next - p_curr, axis=1)
        c = self._b.norm(p_next - p_prev, axis=1)

        # Heron's formula for triangle area
        s = 0.5 * (a + b + c)
        # Numerical safety: clamp inside sqrt >= 0
        area_sq = self._b.maximum(s * (s - a) * (s - b) * (s - c), self._b.zeros_like(s))
        area = self._b.sqrt(area_sq)

        # Menger curvature: κ = 4A / (abc)
        denom = (a * b * c) + self._eps
        kappa = 4.0 * area / denom

        # Clean up numerical issues
        kappa = self._b.where(
            self._b.isfinite(kappa),
            kappa,
            self._b.zeros_like(kappa)
        )
        return kappa

    def compute_arc_length(self, positions: Array) -> float:
        """
        Compute total arc length (path length) of trajectory.

        Args:
            positions: [T, hidden_dim] array of positions

        Returns:
            Total arc length (sum of step distances)
        """
        T = int(self._b.shape(positions)[0])
        if T < 2:
            return 0.0
        steps = self._b.norm(positions[1:] - positions[:-1], axis=1)
        self._b.eval(steps)
        return float(self._b.sum(steps))

    def analyze_flow(self, positions: Array) -> FlowMetrics:
        """
        Compute full geometric analysis of a reasoning flow trajectory.

        Args:
            positions: [T, hidden_dim] array of token embeddings forming the trajectory

        Returns:
            FlowMetrics containing all order-0, order-1, order-2 statistics
        """
        T = int(self._b.shape(positions)[0])

        # Order-0: Position statistics
        arc_length = self.compute_arc_length(positions)
        mean_speed = arc_length / max(T - 1, 1)

        # Order-1: Velocity statistics
        velocities = self.compute_velocities(positions)
        if int(self._b.shape(velocities)[0]) > 0:
            velocity_norms = self._b.norm(velocities, axis=1)
            self._b.eval(velocity_norms)
            mean_velocity_norm = float(self._b.mean(velocity_norms))
            velocity_variance = float(self._b.var(velocity_norms))
        else:
            velocity_norms = self._b.zeros((1,))
            mean_velocity_norm = 0.0
            velocity_variance = 0.0

        # Order-2: Curvature statistics
        curvatures = self.compute_menger_curvature(positions)
        self._b.eval(curvatures)
        if int(self._b.shape(curvatures)[0]) > 0:
            mean_curvature = float(self._b.mean(curvatures))
            max_curvature = float(self._b.max(curvatures))
            curvature_integral = float(self._b.sum(self._b.abs(curvatures)))
        else:
            mean_curvature = 0.0
            max_curvature = 0.0
            curvature_integral = 0.0

        # Derived metrics
        smoothness = 1.0 / (1.0 + mean_curvature)

        # Directness: straight-line distance / arc length
        if arc_length > self._eps and T >= 2:
            straight_dist = float(self._b.norm(positions[-1] - positions[0]))
            directness = straight_dist / arc_length
        else:
            directness = 1.0

        return FlowMetrics(
            arc_length=arc_length,
            mean_speed=mean_speed,
            velocities=velocities,
            mean_velocity_norm=mean_velocity_norm,
            velocity_variance=velocity_variance,
            curvatures=curvatures,
            mean_curvature=mean_curvature,
            max_curvature=max_curvature,
            curvature_integral=curvature_integral,
            smoothness=smoothness,
            directness=directness,
        )

    def cosine_similarity_sequence(
        self,
        seq_a: Array,
        seq_b: Array,
    ) -> Array:
        """
        Compute per-step cosine similarity between two aligned sequences.

        Args:
            seq_a: [T, D] first sequence
            seq_b: [T, D] second sequence (must have same length)

        Returns:
            [T] array of cosine similarities
        """
        if self._b.shape(seq_a)[0] != self._b.shape(seq_b)[0]:
            raise ValueError(
                f"Sequence length mismatch: {self._b.shape(seq_a)[0]} vs {self._b.shape(seq_b)[0]}"
            )

        norm_a = self._b.norm(seq_a, axis=1, keepdims=True) + self._eps
        norm_b = self._b.norm(seq_b, axis=1, keepdims=True) + self._eps
        dot_products = self._b.sum(seq_a * seq_b, axis=1)
        cos_sim = dot_products / (self._b.squeeze(norm_a) * self._b.squeeze(norm_b))
        return cos_sim

    def flow_similarity(
        self,
        flow_a: Array,
        flow_b: Array,
        order: int = 1,
    ) -> float:
        """
        Compute similarity between two reasoning flows at a given order.

        Order 0: Compare positions (semantic similarity)
        Order 1: Compare velocities (logic structure similarity)
        Order 2: Compare curvatures (higher-order logic similarity)

        Args:
            flow_a: [T_a, hidden_dim] first flow trajectory
            flow_b: [T_b, hidden_dim] second flow trajectory
            order: 0=positions, 1=velocities, 2=curvatures

        Returns:
            Mean cosine similarity (for order 0,1) or Pearson correlation (for order 2)
        """
        # Get sequences at specified order
        if order == 0:
            seq_a, seq_b = flow_a, flow_b
        elif order == 1:
            seq_a = self.compute_velocities(flow_a)
            seq_b = self.compute_velocities(flow_b)
        elif order == 2:
            seq_a = self._b.reshape(self.compute_menger_curvature(flow_a), (-1, 1))
            seq_b = self._b.reshape(self.compute_menger_curvature(flow_b), (-1, 1))
        else:
            raise ValueError(f"Order must be 0, 1, or 2, got {order}")

        len_a = int(self._b.shape(seq_a)[0])
        len_b = int(self._b.shape(seq_b)[0])

        if len_a == 0 or len_b == 0:
            return 0.0

        # Truncate to same length
        min_len = min(len_a, len_b)
        seq_a = seq_a[:min_len]
        seq_b = seq_b[:min_len]

        if order == 2:
            # For curvatures, use Pearson correlation (centered)
            a = self._b.reshape(seq_a, (-1,))
            b = self._b.reshape(seq_b, (-1,))
            a_centered = a - self._b.mean(a)
            b_centered = b - self._b.mean(b)
            num = float(self._b.sum(a_centered * b_centered))
            den = float(self._b.norm(a_centered) + self._eps) * float(self._b.norm(b_centered) + self._eps)
            return num / den
        else:
            # For positions/velocities, use mean cosine similarity
            cos_sims = self.cosine_similarity_sequence(seq_a, seq_b)
            self._b.eval(cos_sims)
            return float(self._b.mean(cos_sims))


@dataclass(frozen=True)
class LayerFlowProfile:
    """Flow metrics computed at each layer of the model."""

    layer_idx: int
    metrics: FlowMetrics


def analyze_multilayer_flow(
    backend: Backend,
    layer_positions: dict[int, Array],
) -> list[LayerFlowProfile]:
    """
    Analyze reasoning flow at each layer of the model.

    Args:
        backend: Backend for tensor operations
        layer_positions: Dict mapping layer_idx -> [T, hidden_dim] positions

    Returns:
        List of LayerFlowProfile, one per layer, sorted by layer index
    """
    analyzer = ReasoningFlowAnalyzer(backend)
    profiles = []
    for layer_idx in sorted(layer_positions.keys()):
        positions = layer_positions[layer_idx]
        metrics = analyzer.analyze_flow(positions)
        profiles.append(LayerFlowProfile(layer_idx=layer_idx, metrics=metrics))
    return profiles


@dataclass(frozen=True)
class TokenCurvatureProfile:
    """Token-level curvature analysis (Zhou et al. methodology).

    This measures WHERE in the prompt the reasoning trajectory bends most,
    aggregated across all layers. High curvature at a token position indicates
    a "reasoning turn" - where the model's representation changes direction.
    """

    # Per-token curvature (averaged across layers)
    token_curvatures: list[float]  # [T-2] curvature at each interior token
    token_indices: list[int]  # Token indices (1 to T-2, since curvature needs triplets)

    # Peak analysis
    peak_token_idx: int  # Token with highest curvature
    peak_curvature: float  # Curvature value at peak

    # Statistics
    mean_curvature: float
    std_curvature: float
    total_tokens: int


def analyze_token_curvature(
    backend: Backend,
    layer_positions: dict[int, Array],
) -> TokenCurvatureProfile:
    """
    Compute token-level curvature profile (Zhou et al. methodology).

    This aggregates curvature across all layers to show WHERE in the prompt
    the reasoning trajectory bends most. Unlike layer-level analysis which
    shows architectural properties, token-level analysis shows prompt-specific
    reasoning dynamics.

    Args:
        backend: Backend for tensor operations
        layer_positions: Dict mapping layer_idx -> [T, hidden_dim] positions

    Returns:
        TokenCurvatureProfile with per-token curvature values
    """
    analyzer = ReasoningFlowAnalyzer(backend)

    # Get number of tokens from first layer
    first_layer = min(layer_positions.keys())
    T = int(backend.shape(layer_positions[first_layer])[0])

    if T < 3:
        return TokenCurvatureProfile(
            token_curvatures=[],
            token_indices=[],
            peak_token_idx=-1,
            peak_curvature=0.0,
            mean_curvature=0.0,
            std_curvature=0.0,
            total_tokens=T,
        )

    # Accumulate curvature across layers for each token position
    # Curvature array has length T-2 (needs triplets)
    n_curvature_points = T - 2
    curvature_sum = backend.zeros((n_curvature_points,))
    n_layers = 0

    for layer_idx in sorted(layer_positions.keys()):
        positions = layer_positions[layer_idx]
        curvatures = analyzer.compute_menger_curvature(positions)
        backend.eval(curvatures)

        if int(backend.shape(curvatures)[0]) == n_curvature_points:
            curvature_sum = curvature_sum + curvatures
            n_layers += 1

    if n_layers == 0:
        return TokenCurvatureProfile(
            token_curvatures=[],
            token_indices=[],
            peak_token_idx=-1,
            peak_curvature=0.0,
            mean_curvature=0.0,
            std_curvature=0.0,
            total_tokens=T,
        )

    # Average across layers
    avg_curvatures = curvature_sum / float(n_layers)
    backend.eval(avg_curvatures)

    # Convert to list
    token_curvatures = [float(avg_curvatures[i]) for i in range(n_curvature_points)]
    # Token indices: curvature[i] corresponds to token i+1 (center of triplet i, i+1, i+2)
    token_indices = list(range(1, T - 1))

    # Find peak
    peak_idx = token_curvatures.index(max(token_curvatures))
    peak_token_idx = token_indices[peak_idx]
    peak_curvature = token_curvatures[peak_idx]

    # Statistics
    mean_curv = sum(token_curvatures) / len(token_curvatures)
    variance = sum((c - mean_curv) ** 2 for c in token_curvatures) / len(token_curvatures)
    std_curv = variance ** 0.5

    return TokenCurvatureProfile(
        token_curvatures=token_curvatures,
        token_indices=token_indices,
        peak_token_idx=peak_token_idx,
        peak_curvature=peak_curvature,
        mean_curvature=mean_curv,
        std_curvature=std_curv,
        total_tokens=T,
    )


__all__ = [
    "FlowMetrics",
    "LayerFlowProfile",
    "ReasoningFlowAnalyzer",
    "TokenCurvatureProfile",
    "analyze_multilayer_flow",
    "analyze_token_curvature",
]
