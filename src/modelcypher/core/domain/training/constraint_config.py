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

"""Constraint configuration for constrained geometric training.

Primal-dual optimization with three constraints:
- Invariance: same logic → similar hidden states (C_inv ≤ ε_inv)
- Separation: different logic → different hidden states (C_sep ≥ m_sep)
- Geometric expansion: target layers should increase effective rank
  (C_geo <= 0 where C_geo is target-rank shortfall)

All thresholds are derived from base-model measurements.
Lagrange multipliers are updated via projected dual ascent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConstraintConfig:
    """Configuration for constrained geometric training.

    All thresholds are derived from base model geometry measurements.
    Multipliers are initialized to 1.0 and adapted via dual ascent.
    """

    # Invariance constraint: C_inv ≤ ε_inv
    epsilon_inv: float  # max allowed hidden-state distance for same-logic pairs
    # Separation constraint: C_sep ≥ m_sep
    margin_sep: float  # min required hidden-state distance for different-logic pairs
    # Geometric expansion constraint: C_geo ≤ ε_tail.
    # C_geo is defined as mean target-rank shortfall across target layers.
    # For strict expansion objective, ε_tail is 0.
    epsilon_tail: float

    # Target layers for hidden state collection and geodesic guardrail
    target_layers: list[int] = field(default_factory=list)

    # Baseline effective rank per target layer.
    baseline_entropy: dict[int, float] = field(default_factory=dict)
    # Per-layer effective-rank spread from baseline measurements.
    # Used to derive per-layer target uplift without cross-layer leakage.
    baseline_entropy_std: dict[int, float] = field(default_factory=dict)
    # Target effective rank per target layer (derived from baseline distribution).
    target_entropy: dict[int, float] = field(default_factory=dict)

    # Pair index information (set during dataset preparation)
    # Each entry: (anchor_idx, partner_idx) within a batch
    # These are populated per-batch by the paired batch iterator

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon_inv": self.epsilon_inv,
            "margin_sep": self.margin_sep,
            "epsilon_tail": self.epsilon_tail,
            "target_layers": self.target_layers,
            "baseline_entropy": self.baseline_entropy,
            "baseline_entropy_std": self.baseline_entropy_std,
            "target_entropy": self.target_entropy,
        }


@dataclass
class ConstraintState:
    """Mutable state for primal-dual optimization.

    Lagrange multipliers are updated after each gradient step.
    Constraint values are logged per epoch for diagnostics.

    Use ``frozen`` to lock specific multipliers at their initial value
    (for ablation experiments that disable constraint groups).
    """

    mu_inv: float = 1.0
    mu_sep: float = 1.0
    mu_geo: float = 1.0

    # Frozen multipliers — names in this set are not updated by dual_update().
    # Example: frozenset({"mu_inv", "mu_sep"}) freezes invariance + separation.
    frozen: frozenset[str] = field(default_factory=frozenset)

    # Running constraint values (updated each step, logged per epoch)
    last_C_inv: float = 0.0
    last_C_sep: float = 0.0
    last_C_geo: float = 0.0
    last_ce_loss: float = 0.0

    def dual_update(self, C_inv: float, C_sep: float, C_geo: float,
                    config: ConstraintConfig, alpha_dual: float) -> None:
        """Update Lagrange multipliers via projected dual ascent.

        μ_new = max(0, μ + α * constraint_violation)

        Multipliers listed in ``self.frozen`` are skipped.

        Args:
            C_inv: Current invariance constraint value.
            C_sep: Current separation constraint value.
            C_geo: Current geodesic tail constraint value.
            config: Constraint thresholds.
            alpha_dual: Dual step size (typically = primal LR).
        """
        if "mu_inv" not in self.frozen:
            self.mu_inv = max(0.0, self.mu_inv + alpha_dual * (C_inv - config.epsilon_inv))
        if "mu_sep" not in self.frozen:
            self.mu_sep = max(0.0, self.mu_sep + alpha_dual * (config.margin_sep - C_sep))
        if "mu_geo" not in self.frozen:
            self.mu_geo = max(0.0, self.mu_geo + alpha_dual * (C_geo - config.epsilon_tail))

        self.last_C_inv = C_inv
        self.last_C_sep = C_sep
        self.last_C_geo = C_geo

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu_inv": self.mu_inv,
            "mu_sep": self.mu_sep,
            "mu_geo": self.mu_geo,
            "frozen": sorted(self.frozen),
            "C_inv": self.last_C_inv,
            "C_sep": self.last_C_sep,
            "C_geo": self.last_C_geo,
            "ce_loss": self.last_ce_loss,
        }


def _mean_std(vals: list[float]) -> tuple[float, float]:
    """Compute mean and sample standard deviation."""
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return mean, variance ** 0.5


def derive_constraint_thresholds(
    base_inv_distances: list[float],
    base_sep_distances: list[float],
    base_layer_entropies: dict[int, float],
    base_layer_entropy_stds: dict[int, float] | None = None,
) -> ConstraintConfig:
    """Derive constraint thresholds from base model measurements.

    All thresholds are distribution-derived (mean ± std), not heuristic.
    Requires at least 3 distance measurements per constraint for reliable
    statistics. Raises ValueError if measurements are insufficient.

    Args:
        base_inv_distances: Hidden-state distances between invariance pairs on base model.
        base_sep_distances: Hidden-state distances between counterfactual pairs on base model.
        base_layer_entropies: Per-layer effective rank on base model.
        base_layer_entropy_stds: Optional per-layer effective-rank spread from
            repeated baseline measurements. If omitted, falls back to a single
            cross-layer spread value.

    Returns:
        ConstraintConfig with geometry-derived thresholds.

    Raises:
        ValueError: If insufficient measurements for reliable threshold derivation.
    """
    if len(base_inv_distances) < 3:
        raise ValueError(
            f"Need ≥3 invariance distance measurements, got {len(base_inv_distances)}. "
            "Ensure the dataset has enough invariance pairs (same logic_id, different template_id)."
        )
    if len(base_sep_distances) < 3:
        raise ValueError(
            f"Need ≥3 separation distance measurements, got {len(base_sep_distances)}. "
            "Ensure the dataset has enough counterfactual pairs (same template_id, different logic_id)."
        )
    if not base_layer_entropies:
        raise ValueError("No layer entropies provided for geodesic guardrail.")

    # Invariance: ε_inv = mean - 1σ (lower envelope of base model distances).
    # The adapter should compress same-logic hidden states to below this.
    mean_inv, std_inv = _mean_std(base_inv_distances)
    epsilon_inv = max(0.0, mean_inv - std_inv)

    # Separation: m_sep = mean + 1σ (improvement target).
    # Counterfactual pairs should be pushed farther apart than baseline.
    mean_sep, std_sep = _mean_std(base_sep_distances)
    margin_sep = max(0.0, mean_sep + std_sep)

    # Geometric expansion target:
    # Prefer per-layer baseline spread (from repeated measurements) so each
    # layer gets a local uplift target. This avoids cross-layer variance
    # leakage where one volatile layer over-inflates all targets.
    # Fallback: cross-layer spread if per-layer spreads are unavailable.
    layer_vals = list(base_layer_entropies.values())
    _, std_layers = _mean_std(layer_vals)
    if base_layer_entropy_stds is None:
        base_layer_entropy_stds = {}
    target_entropy = {
        layer_idx: base_erank + max(0.0, base_layer_entropy_stds.get(layer_idx, std_layers))
        for layer_idx, base_erank in base_layer_entropies.items()
    }
    # C_geo is already a non-negative shortfall; strict target is zero shortfall.
    epsilon_tail = 0.0

    target_layers = sorted(base_layer_entropies.keys())

    config = ConstraintConfig(
        epsilon_inv=epsilon_inv,
        margin_sep=margin_sep,
        epsilon_tail=epsilon_tail,
        target_layers=target_layers,
        baseline_entropy=dict(base_layer_entropies),
        baseline_entropy_std=dict(base_layer_entropy_stds),
        target_entropy=target_entropy,
    )

    logger.info(
        "Constraint thresholds: ε_inv=%.4f (mean=%.4f, σ=%.4f), "
        "m_sep=%.4f (mean=%.4f, σ=%.4f), C_geo target uplift mean=+%.4f (ε_tail=%.1f), "
        "layers=%s",
        epsilon_inv, mean_inv, std_inv,
        margin_sep, mean_sep, std_sep,
        (
            sum(
                target_entropy[layer] - base_layer_entropies[layer]
                for layer in target_layers
            ) / max(1, len(target_layers))
        ),
        epsilon_tail,
        target_layers,
    )

    return config


__all__ = [
    "ConstraintConfig",
    "ConstraintState",
    "derive_constraint_thresholds",
]
