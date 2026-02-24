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

"""Curiosity policy based on Expected Free Energy (Active Inference).

This module implements probe ranking using Friston's Expected Free Energy (EFE)
framework from Active Inference theory (Friston et al. 2017).

Expected Free Energy decomposes into:
    G(π, τ) = D[Q(o_τ|π) || P(o_τ)] + E_Q[H[P(o_τ|s_τ)]]
            = risk term           + ambiguity term

In ModelCypher's geometric terms:
    - Risk = (1 - capacity_fraction)² - deviation from target (full capacity)
    - Ambiguity = eigenscore - manifold sparsity (observation uncertainty)
    - Epistemic value = eigenscore × capacity_fraction - the "curiosity score"

The epistemic value is the product form because:
    - Zero capacity => zero value (can't encode anything)
    - Zero eigenscore => zero value (nothing to learn)
    - Maximum at high eigenscore AND high capacity

Exploration temperature is geometry-derived:
    T = mean_eigenscore / sqrt(eps)

When T >> 1: explore uniformly (sparse manifold)
When T << 1: exploit best candidate (dense manifold)
When T ~ 1: balanced

All thresholds derive from machine precision (sqrt(eps)), not heuristics.

References:
    - Friston et al. 2017 "Active Inference: A Process Theory"
    - activeinference.github.io/papers/process_theory.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


class CuriosityAction(str, Enum):
    """Actions the curiosity daemon can take."""

    PROBE = "probe"  # Execute a probe at selected coordinates
    CONSOLIDATE = "consolidate"  # Trigger consolidation before probing
    WAIT = "wait"  # Insufficient capacity or candidates
    COMPLETE = "complete"  # Manifold is uniformly dense


@dataclass(frozen=True)
class ProbeCandidate:
    """A candidate region for curiosity-driven probing.

    All values are raw measurements - no interpretation.

    Attributes
    ----------
    coordinates : tuple[float, ...]
        Manifold coordinates (activation space location).
    eigenscore : float
        Manifold sparsity at this location [0, 1].
        Higher = more uncertain = more information gain potential.
    capacity_fraction : float
        Null-space available [0, 1].
        Higher = more room to encode knowledge.
    epistemic_value : float
        Curiosity score = eigenscore × capacity_fraction.
        Product form ensures probe is BOTH informative AND encodable.
    efe_score : float
        Expected Free Energy = risk + ambiguity.
        Lower EFE = better probe (we minimize free energy).
    layer_id : int
        Which layer this candidate belongs to (-1 = model-level).
    neighbor_density : float
        k-NN density estimate at this point [0, 1].
        Higher = denser region.
    intrinsic_dimension : float
        Local intrinsic dimension estimate.
    """

    coordinates: tuple[float, ...]
    eigenscore: float
    capacity_fraction: float
    epistemic_value: float
    efe_score: float
    layer_id: int
    neighbor_density: float
    intrinsic_dimension: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "coordinates_len": len(self.coordinates),
            "eigenscore": self.eigenscore,
            "capacity_fraction": self.capacity_fraction,
            "epistemic_value": self.epistemic_value,
            "efe_score": self.efe_score,
            "layer_id": self.layer_id,
            "neighbor_density": self.neighbor_density,
            "intrinsic_dimension": self.intrinsic_dimension,
        }


@dataclass(frozen=True)
class CuriosityState:
    """Current state of the curiosity evaluation.

    All values are raw measurements - no interpretation.

    Attributes
    ----------
    n_candidates : int
        Number of probe candidates evaluated.
    top_candidate : ProbeCandidate | None
        Best candidate (highest epistemic value).
    mean_eigenscore : float
        Mean sparsity across all candidates.
    mean_capacity : float
        Mean capacity across all layers.
    exploration_temperature : float
        Softmax temperature for action selection.
        Derived from geometry: mean_eigenscore / sqrt_eps.
    sqrt_eps : float
        Machine precision threshold.
    """

    n_candidates: int
    top_candidate: ProbeCandidate | None
    mean_eigenscore: float
    mean_capacity: float
    exploration_temperature: float
    sqrt_eps: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "n_candidates": self.n_candidates,
            "top_candidate": (
                self.top_candidate.to_dict() if self.top_candidate else None
            ),
            "mean_eigenscore": self.mean_eigenscore,
            "mean_capacity": self.mean_capacity,
            "exploration_temperature": self.exploration_temperature,
            "sqrt_eps": self.sqrt_eps,
        }


@runtime_checkable
class CuriosityPolicy(Protocol):
    """Protocol for curiosity-driven action selection.

    Implementations must rank probe candidates and select actions
    using only geometry-derived quantities (no hardcoded thresholds).
    """

    def rank_candidates(
        self,
        candidates: list[ProbeCandidate],
    ) -> list[ProbeCandidate]:
        """Rank candidates by epistemic value (descending)."""
        ...

    def select_action(
        self,
        state: CuriosityState,
    ) -> tuple[CuriosityAction, ProbeCandidate | None]:
        """Select next action based on current state."""
        ...

    def compute_exploration_temperature(
        self,
        mean_eigenscore: float,
        sqrt_eps: float,
    ) -> float:
        """Compute softmax temperature from manifold state."""
        ...


def compute_epistemic_value(
    eigenscore: float,
    capacity_fraction: float,
) -> float:
    """Compute epistemic value = eigenscore × capacity_fraction.

    Mathematical basis:
        - Eigenscore measures information gain potential (manifold sparsity)
        - Capacity measures ability to encode (available null-space dimensions)
        - Product ensures we probe regions that are BOTH informative AND encodable

    This is the exploration term from Active Inference:
        epistemic_value ≈ H[Q(s|o)] - H[Q(s|o,π)]

    Approximated as eigenscore (current uncertainty) weighted by capacity
    (ability to reduce that uncertainty via encoding).

    Parameters
    ----------
    eigenscore : float
        Manifold sparsity [0, 1].
    capacity_fraction : float
        Available null-space [0, 1].

    Returns
    -------
    float
        Epistemic value in [0, 1].
    """
    return eigenscore * capacity_fraction


def compute_efe(
    eigenscore: float,
    capacity_fraction: float,
) -> float:
    """Compute Expected Free Energy for a probe action.

    From Friston et al. 2017:
        G = D[Q(o|π) || P(o)] + E_Q[H[P(o|s)]]
          = risk              + ambiguity

    In ModelCypher terms:
        risk = (1 - capacity_fraction)²
             Deviation from preferred state (full capacity = 1.0)

        ambiguity = eigenscore
             Uncertainty about observations given state

    Lower EFE = better action. But we want to maximize curiosity.

    The product form (epistemic_value) emerges because:
        - Zero capacity => zero value (can't encode anything)
        - Zero eigenscore => zero value (nothing to learn)
        - Maximum at high eigenscore AND high capacity

    Parameters
    ----------
    eigenscore : float
        Manifold sparsity [0, 1].
    capacity_fraction : float
        Available null-space [0, 1].

    Returns
    -------
    float
        Expected Free Energy (lower = better probe).
    """
    # Risk: deviation from target (full capacity = 1.0)
    risk = (1.0 - capacity_fraction) ** 2

    # Ambiguity: eigenscore measures observation uncertainty
    ambiguity = eigenscore

    # EFE = risk + ambiguity (lower is better)
    return risk + ambiguity


class EFECuriosityPolicy:
    """Curiosity policy based on Expected Free Energy.

    Selects probe actions to maximize epistemic value while respecting
    capacity constraints. All thresholds derived from machine precision.

    The policy ranks candidates by epistemic_value = eigenscore × capacity,
    then selects actions based on:
        - COMPLETE: mean_eigenscore <= sqrt_eps (manifold dense)
        - WAIT: mean_capacity <= sqrt_eps (no encoding capacity)
        - CONSOLIDATE: geometric conditions met
        - PROBE: select top candidate

    Parameters
    ----------
    backend : Backend, optional
        Compute backend. Defaults to system-selected backend.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize the EFE curiosity policy."""
        self._backend = backend or get_default_backend()

        # Derive precision thresholds from machine epsilon
        ref = self._backend.array([1.0])
        eps = machine_epsilon(self._backend, ref)
        self._sqrt_eps = math.sqrt(float(eps))

    @property
    def sqrt_eps(self) -> float:
        """Machine precision threshold."""
        return self._sqrt_eps

    def rank_candidates(
        self,
        candidates: list[ProbeCandidate],
    ) -> list[ProbeCandidate]:
        """Rank candidates by epistemic value (descending).

        Parameters
        ----------
        candidates : list[ProbeCandidate]
            Probe candidates to rank.

        Returns
        -------
        list[ProbeCandidate]
            Candidates sorted by epistemic value (highest first).
        """
        return sorted(candidates, key=lambda c: c.epistemic_value, reverse=True)

    def select_action(
        self,
        state: CuriosityState,
    ) -> tuple[CuriosityAction, ProbeCandidate | None]:
        """Select action based on EFE principles.

        Decision logic (derived from geometry, not heuristics):

        1. COMPLETE: mean_eigenscore <= sqrt_eps
           Manifold is uniformly dense (below precision floor)

        2. WAIT: mean_capacity <= sqrt_eps
           No capacity to encode - wait for consolidation

        3. CONSOLIDATE: geometric conditions met
           (delegated to BackgroundConsolidationMonitor)

        4. PROBE: Select top candidate by epistemic value

        Parameters
        ----------
        state : CuriosityState
            Current curiosity evaluation state.

        Returns
        -------
        tuple[CuriosityAction, ProbeCandidate | None]
            Selected action and candidate (if PROBE).
        """
        # Check completion: manifold uniformly dense
        if state.mean_eigenscore <= state.sqrt_eps:
            return CuriosityAction.COMPLETE, None

        # Check capacity: can we encode?
        if state.mean_capacity <= state.sqrt_eps:
            return CuriosityAction.WAIT, None

        # No candidates available
        if state.top_candidate is None:
            return CuriosityAction.WAIT, None

        # Check if consolidation should trigger first
        if self._should_consolidate(state):
            return CuriosityAction.CONSOLIDATE, None

        # Select probe candidate (top by epistemic value)
        return CuriosityAction.PROBE, state.top_candidate

    def _should_consolidate(self, state: CuriosityState) -> bool:
        """Check consolidation trigger conditions.

        From BackgroundConsolidation:
            should_consolidate = (
                event_count >= MIN_EVENTS AND
                mean_eigenscore > 2 * sqrt(eps) AND
                mean_capacity > sqrt(eps)
            )

        MIN_EVENTS check is delegated to BackgroundConsolidationMonitor.
        Here we check only the geometric conditions.

        Parameters
        ----------
        state : CuriosityState
            Current curiosity state.

        Returns
        -------
        bool
            True if consolidation should trigger.
        """
        eigenscore_threshold = 2 * state.sqrt_eps

        return (
            state.mean_eigenscore > eigenscore_threshold
            and state.mean_capacity > state.sqrt_eps
        )

    def compute_exploration_temperature(
        self,
        mean_eigenscore: float,
        sqrt_eps: float,
    ) -> float:
        """Derive exploration temperature from manifold state.

        Mathematical basis:
            - High mean eigenscore => manifold is sparse => EXPLORE more (high T)
            - Low mean eigenscore => manifold is dense => EXPLOIT more (low T)

        Temperature scaling:
            T = mean_eigenscore / sqrt_eps

        Why sqrt_eps?
            - sqrt(machine_epsilon) is the precision floor
            - When mean_eigenscore ~ sqrt_eps, we're at numerical precision
            - T should be ~1 at this boundary (balanced exploration/exploitation)

        When T >> 1: softmax is nearly uniform (exploration)
        When T << 1: softmax is concentrated on best (exploitation)
        When T ~ 1: balanced

        This is NOT a hardcoded threshold - it's derived from:
        1. The manifold state (mean_eigenscore)
        2. Machine precision (sqrt_eps)

        Parameters
        ----------
        mean_eigenscore : float
            Mean manifold sparsity across candidates.
        sqrt_eps : float
            Machine precision threshold.

        Returns
        -------
        float
            Exploration temperature >= sqrt_eps.
        """
        if mean_eigenscore <= sqrt_eps:
            # Manifold is dense (below precision floor) => pure exploitation
            return sqrt_eps  # Minimum temperature

        # Temperature scales with sparsity
        return mean_eigenscore / sqrt_eps

    def create_candidate(
        self,
        coordinates: tuple[float, ...],
        eigenscore: float,
        capacity_fraction: float,
        layer_id: int = -1,
        neighbor_density: float = 0.0,
        intrinsic_dimension: float = 0.0,
    ) -> ProbeCandidate:
        """Create a probe candidate with computed EFE scores.

        Parameters
        ----------
        coordinates : tuple[float, ...]
            Manifold coordinates.
        eigenscore : float
            Manifold sparsity [0, 1].
        capacity_fraction : float
            Available null-space [0, 1].
        layer_id : int
            Layer index (-1 for model-level).
        neighbor_density : float
            Local k-NN density estimate.
        intrinsic_dimension : float
            Local intrinsic dimension estimate.

        Returns
        -------
        ProbeCandidate
            Candidate with computed epistemic value and EFE.
        """
        epistemic = compute_epistemic_value(eigenscore, capacity_fraction)
        efe = compute_efe(eigenscore, capacity_fraction)

        return ProbeCandidate(
            coordinates=coordinates,
            eigenscore=eigenscore,
            capacity_fraction=capacity_fraction,
            epistemic_value=epistemic,
            efe_score=efe,
            layer_id=layer_id,
            neighbor_density=neighbor_density,
            intrinsic_dimension=intrinsic_dimension,
        )

    def create_state(
        self,
        candidates: list[ProbeCandidate],
        mean_capacity: float | None = None,
    ) -> CuriosityState:
        """Create curiosity state from candidates.

        Parameters
        ----------
        candidates : list[ProbeCandidate]
            Evaluated probe candidates.
        mean_capacity : float, optional
            Override mean capacity (e.g., from NullSpaceTracker).

        Returns
        -------
        CuriosityState
            Current curiosity evaluation state.
        """
        if not candidates:
            return CuriosityState(
                n_candidates=0,
                top_candidate=None,
                mean_eigenscore=0.0,
                mean_capacity=mean_capacity or 0.0,
                exploration_temperature=self._sqrt_eps,
                sqrt_eps=self._sqrt_eps,
            )

        ranked = self.rank_candidates(candidates)
        mean_eigen = sum(c.eigenscore for c in candidates) / len(candidates)
        mean_cap = (
            mean_capacity
            if mean_capacity is not None
            else sum(c.capacity_fraction for c in candidates) / len(candidates)
        )
        temp = self.compute_exploration_temperature(mean_eigen, self._sqrt_eps)

        return CuriosityState(
            n_candidates=len(candidates),
            top_candidate=ranked[0] if ranked else None,
            mean_eigenscore=mean_eigen,
            mean_capacity=mean_cap,
            exploration_temperature=temp,
            sqrt_eps=self._sqrt_eps,
        )


def create_efe_policy(backend: "Backend | None" = None) -> EFECuriosityPolicy:
    """Create an EFE-based curiosity policy.

    Parameters
    ----------
    backend : Backend, optional
        Compute backend.

    Returns
    -------
    EFECuriosityPolicy
        Configured policy instance.
    """
    return EFECuriosityPolicy(backend=backend)
