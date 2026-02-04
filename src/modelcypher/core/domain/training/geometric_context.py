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

"""Geometric context injection for self-aware training.

This module provides the model with access to its own geometric state
during training. The context is injected as a natural language prefix
that the model can learn to interpret and use.

All values are relational (ratios/comparisons) - no arbitrary units:
- loop_persistence: ΔH (current - baseline) entropy change
- expansion_ratio: peak_norm / final_norm
- highway_depth: highway_layer / n_layers
- exit_convergence: mean_norm / dev_norm at exit
- has_reasoning_loops: spectral entropy proxy for β₁ > 0

The format uses [GEOMETRY]...[/GEOMETRY] markers that work with any
tokenizer without special tokens.

This module contains ONLY pure geometric analysis using the Backend protocol.
Framework-specific model inference code lives in adapters/training/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class GeometricContext:
    """All values are ratios or relative comparisons. No arbitrary units.

    Attributes:
        loop_persistence: ΔH = H_exit - H_highway (entropy change).
            Positive = healthy (loops preserved), negative = collapsed.
        expansion_ratio: peak_norm / final_norm.
            >1 = intermediate expansion, <1 = monotonic compression.
        highway_depth: highway_layer / n_layers.
            Where in the model the compression bottleneck occurs.
        exit_convergence: mean_norm / dev_norm at exit.
            High = consistent outputs, low = varied outputs.
        has_reasoning_loops: True if H_exit > H_highway.
            Spectral proxy for topological β₁ > 0.
    """

    loop_persistence: float
    expansion_ratio: float
    highway_depth: float
    exit_convergence: float
    has_reasoning_loops: bool

    @classmethod
    def from_metrics(
        cls,
        h_highway_entropy: float,
        h_exit_entropy: float,
        base_delta_entropy: float,
        layer_norms: list[float],
        highway_layer: int,
        n_layers: int,
        exit_mean_norm: float,
        exit_dev_norm: float,
    ) -> "GeometricContext":
        """Compute geometric context from pre-computed metrics.

        This is a pure function. The actual model inference to compute
        these metrics lives in adapters.

        Args:
            h_highway_entropy: Spectral entropy at highway layer.
            h_exit_entropy: Spectral entropy at exit layer.
            base_delta_entropy: Base model's ΔH for comparison.
            layer_norms: List of norms at each layer.
            highway_layer: Highway layer index.
            n_layers: Total number of layers.
            exit_mean_norm: Mean norm at exit layer.
            exit_dev_norm: Standard deviation of norms at exit layer.

        Returns:
            GeometricContext with all relational metrics.
        """
        # Loop persistence: ΔH relative to base
        current_delta = h_exit_entropy - h_highway_entropy
        loop_persistence = current_delta - base_delta_entropy

        # Expansion ratio: peak_norm / final_norm
        peak_norm = max(layer_norms) if layer_norms else 1.0
        final_norm = layer_norms[-1] if layer_norms else 1.0
        expansion_ratio = peak_norm / max(final_norm, 1e-8)

        # Highway depth: fraction of total layers
        highway_depth = highway_layer / max(n_layers, 1)

        # Exit convergence: mean_norm / dev_norm at exit
        exit_convergence = exit_mean_norm / max(exit_dev_norm, 1e-8)

        # Has reasoning loops: spectral proxy (entropy grows after highway)
        has_reasoning_loops = current_delta > 0

        return cls(
            loop_persistence=loop_persistence,
            expansion_ratio=expansion_ratio,
            highway_depth=highway_depth,
            exit_convergence=exit_convergence,
            has_reasoning_loops=has_reasoning_loops,
        )

    @classmethod
    def empty(cls) -> "GeometricContext":
        """Return an empty/default context for degenerate cases."""
        return cls(
            loop_persistence=0.0,
            expansion_ratio=1.0,
            highway_depth=0.0,
            exit_convergence=1.0,
            has_reasoning_loops=False,
        )

    def format(self) -> str:
        """Format as prefix. Values are raw floats, model learns interpretation.

        Uses [GEOMETRY]...[/GEOMETRY] markers that work with any tokenizer
        without requiring special tokens.
        """
        reasoning = "yes" if self.has_reasoning_loops else "no"
        return f"""[GEOMETRY]
loop_persistence: {self.loop_persistence:+.2f}
expansion_ratio: {self.expansion_ratio:.2f}
highway_depth: {self.highway_depth:.2f}
convergence: {self.exit_convergence:.2f}
reasoning_loops: {reasoning}
[/GEOMETRY]

"""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging/saving."""
        return {
            "loop_persistence": self.loop_persistence,
            "expansion_ratio": self.expansion_ratio,
            "highway_depth": self.highway_depth,
            "exit_convergence": self.exit_convergence,
            "has_reasoning_loops": self.has_reasoning_loops,
        }


def compute_hidden_norm(
    hidden: "Array",
    backend: "Backend",
) -> float:
    """Compute L2 norm of hidden states (mean across batch/seq).

    Args:
        hidden: Hidden states tensor [batch, seq, hidden_dim] or [n, hidden_dim].
        backend: Backend for tensor operations.

    Returns:
        Mean L2 norm across all positions.
    """
    b = backend

    # Flatten to [n_samples, hidden_dim]
    if len(hidden.shape) == 3:
        hidden = b.reshape(hidden, (-1, int(hidden.shape[-1])))

    # Compute per-row norms
    norm_sq = b.sum(hidden * hidden, axis=-1)
    b.eval(norm_sq)

    # Square root per element
    n_samples = int(norm_sq.shape[0])
    if n_samples == 0:
        return 0.0

    total_norm = 0.0
    for i in range(n_samples):
        val = float(b.to_scalar(norm_sq[i]))
        total_norm += sqrt_scalar(val, b)

    return total_norm / n_samples


def compute_exit_convergence(
    hidden: "Array",
    backend: "Backend",
) -> tuple[float, float]:
    """Compute mean and deviation of norms at exit layer.

    Args:
        hidden: Hidden states tensor [batch, seq, hidden_dim] or [n, hidden_dim].
        backend: Backend for tensor operations.

    Returns:
        Tuple of (mean_norm, dev_norm).
        High mean/dev ratio = consistent outputs.
    """
    b = backend

    # Flatten to [n_samples, hidden_dim]
    if len(hidden.shape) == 3:
        hidden = b.reshape(hidden, (-1, int(hidden.shape[-1])))

    # Compute per-row norms
    norm_sq = b.sum(hidden * hidden, axis=-1)
    b.eval(norm_sq)

    n_samples = int(norm_sq.shape[0])
    if n_samples == 0:
        return 0.0, 1.0

    # Compute norms (sqrt of squared norms)
    norms = []
    for i in range(n_samples):
        val = float(b.to_scalar(norm_sq[i]))
        norms.append(sqrt_scalar(val, b))

    # Mean
    mean_norm = sum(norms) / n_samples

    # Standard deviation: sqrt(E[(X - μ)²])
    variance = sum((n - mean_norm) ** 2 for n in norms) / n_samples
    dev_norm = sqrt_scalar(variance, b)

    return mean_norm, dev_norm


__all__ = [
    "GeometricContext",
    "compute_hidden_norm",
    "compute_exit_convergence",
]
