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

"""Curiosity Daemon for active manifold exploration.

This module implements the Curiosity Daemon - an async service that autonomously
explores the activation manifold using Expected Free Energy (Active Inference)
and geometry-based acquisition functions.

The daemon implements the following state machine:
    IDLE → SELECTING → ACQUIRING → EXECUTING → MEASURING → [CONSOLIDATING]

Convergence is detected when coverage_rate > 0 and coverage_rate < sqrt(eps) - geometry-based,
not time-based.

Research basis:
    - Friston et al. 2017 "Active Inference: A Process Theory"
    - Sener & Savarese 2018 "Active Learning: A Core-Set Approach"
    - Huang et al. "Active Manifold Exploration"

All thresholds derived from sqrt(eps) or geometric invariants.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.curiosity_policy import (
    CuriosityAction,
    CuriosityState,
    EFECuriosityPolicy,
    ProbeCandidate,
)
from modelcypher.core.domain.geometry.acquisition_composite import CompositeAcquisition
from modelcypher.core.domain.geometry.acquisition_protocols import AcquisitionResult
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.core.use_cases.consolidation_service import (
        ConsolidationService,
        ConsolidationStats,
    )
    from modelcypher.ports.backend import Array, Backend


logger = logging.getLogger(__name__)


class DaemonState(str, Enum):
    """State machine states for the curiosity daemon."""

    IDLE = "idle"  # Waiting for candidates or events
    SELECTING = "selecting"  # Ranking candidates via EFE policy
    ACQUIRING = "acquiring"  # Computing acquisition scores
    EXECUTING = "executing"  # Running probes through model
    MEASURING = "measuring"  # Computing coverage metrics
    CONSOLIDATING = "consolidating"  # Triggering ManifoldCompletion
    CONVERGED = "converged"  # Exploration complete
    STOPPED = "stopped"  # Daemon stopped


@dataclass
class CoverageMetrics:
    """Coverage metrics for convergence detection.

    Attributes
    ----------
    coverage_radius : float
        Current k-center coverage radius.
    previous_radius : float
        Coverage radius from previous iteration.
    coverage_rate : float
        Rate of coverage improvement (should decrease toward zero).
    mean_local_id : float
        Mean local intrinsic dimension.
    sparse_fraction : float
        Fraction of corpus with local ID above modal.
    n_corpus : int
        Current corpus size.
    n_probes_executed : int
        Total probes executed.
    """

    coverage_radius: float = float("inf")
    previous_radius: float = float("inf")
    coverage_rate: float = 1.0
    mean_local_id: float = 0.0
    sparse_fraction: float = 1.0
    n_corpus: int = 0
    n_probes_executed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "coverage_radius": self.coverage_radius,
            "previous_radius": self.previous_radius,
            "coverage_rate": self.coverage_rate,
            "mean_local_id": self.mean_local_id,
            "sparse_fraction": self.sparse_fraction,
            "n_corpus": self.n_corpus,
            "n_probes_executed": self.n_probes_executed,
        }


@dataclass
class DaemonStatus:
    """Status of the curiosity daemon."""

    state: DaemonState = DaemonState.STOPPED
    is_running: bool = False
    last_state_change: str = ""
    iterations_completed: int = 0
    probes_executed: int = 0
    consolidations_triggered: int = 0
    current_metrics: CoverageMetrics | None = None
    current_curiosity_state: CuriosityState | None = None
    convergence_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "state": self.state.value,
            "is_running": self.is_running,
            "last_state_change": self.last_state_change,
            "iterations_completed": self.iterations_completed,
            "probes_executed": self.probes_executed,
            "consolidations_triggered": self.consolidations_triggered,
            "current_metrics": (
                self.current_metrics.to_dict() if self.current_metrics else None
            ),
            "current_curiosity_state": (
                self.current_curiosity_state.to_dict()
                if self.current_curiosity_state
                else None
            ),
            "convergence_reason": self.convergence_reason,
        }


@dataclass
class DaemonConfig:
    """Configuration for the curiosity daemon.

    Attributes
    ----------
    max_iterations : int
        Maximum exploration iterations (0 = no limit).
    batch_size : int
        Number of probes to select per iteration.
    check_interval : float
        Seconds between state machine ticks.
    min_candidates : int
        Minimum candidates to start exploration.
    enable_auto_consolidate : bool
        Whether to auto-trigger consolidation.
    on_probe_complete : Callable, optional
        Callback when a probe completes.
    on_consolidation : Callable, optional
        Callback when consolidation completes.
    """

    max_iterations: int = 0  # 0 = no limit
    batch_size: int = 10
    check_interval: float = 1.0
    min_candidates: int = 5
    enable_auto_consolidate: bool = True
    on_probe_complete: Callable[[ProbeCandidate, Any], None] | None = None
    on_consolidation: Callable[["ConsolidationStats"], None] | None = None


# Type alias for probe executor
ProbeExecutor = Callable[[ProbeCandidate], "Array"]


class CuriosityDaemon:
    """Async daemon for curiosity-driven manifold exploration.

    Implements Active Inference (Expected Free Energy) for probe selection
    with geometry-based convergence detection.

    The daemon orchestrates:
    1. EFE policy for probe ranking
    2. Composite acquisition for probe selection
    3. Probe execution through the model
    4. Coverage measurement and convergence detection
    5. Optional consolidation triggering

    Convergence: coverage_rate > 0 and coverage_rate < sqrt(eps) - derived from machine precision.

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension.
    backend : Backend, optional
        Compute backend. Defaults to MLX.
    config : DaemonConfig, optional
        Daemon configuration.
    consolidation_service : ConsolidationService, optional
        Service for triggering consolidation.
    """

    def __init__(
        self,
        hidden_dim: int,
        backend: "Backend | None" = None,
        config: DaemonConfig | None = None,
        consolidation_service: "ConsolidationService | None" = None,
    ) -> None:
        """Initialize the curiosity daemon."""
        self._backend = backend or get_default_backend()
        self._hidden_dim = hidden_dim
        self._config = config or DaemonConfig()
        self._consolidation_service = consolidation_service

        # Derive precision thresholds
        ref = self._backend.array([1.0])
        eps = machine_epsilon(self._backend, ref)
        self._sqrt_eps = math.sqrt(float(eps))

        # EFE policy and acquisition
        self._policy = EFECuriosityPolicy(backend=self._backend)
        self._acquisition = CompositeAcquisition(backend=self._backend)

        # State
        self._state = DaemonState.STOPPED
        self._status = DaemonStatus()
        self._running = False
        self._daemon_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # Corpus and candidates
        self._corpus: list["Array"] = []
        self._candidates: list[ProbeCandidate] = []
        self._metrics = CoverageMetrics()

        # Probe executor (set by user)
        self._probe_executor: ProbeExecutor | None = None

    @property
    def state(self) -> DaemonState:
        """Current daemon state."""
        return self._state

    @property
    def sqrt_eps(self) -> float:
        """Machine precision threshold."""
        return self._sqrt_eps

    def set_probe_executor(self, executor: ProbeExecutor) -> None:
        """Set the function that executes probes.

        The executor takes a ProbeCandidate and returns the resulting
        activation vector from the model.

        Parameters
        ----------
        executor : ProbeExecutor
            Function that executes probes.
        """
        self._probe_executor = executor

    def add_candidate(self, candidate: ProbeCandidate) -> None:
        """Add a candidate for exploration.

        Called externally (e.g., from sparsity events during inference).

        Parameters
        ----------
        candidate : ProbeCandidate
            Candidate to add.
        """
        self._candidates.append(candidate)

    def add_candidates(self, candidates: list[ProbeCandidate]) -> None:
        """Add multiple candidates for exploration.

        Parameters
        ----------
        candidates : list[ProbeCandidate]
            Candidates to add.
        """
        self._candidates.extend(candidates)

    def add_to_corpus(self, activation: "Array") -> None:
        """Add an activation to the corpus.

        Parameters
        ----------
        activation : Array
            Activation vector to add to corpus.
        """
        activation = self._backend.array(activation)
        self._backend.eval(activation)
        self._corpus.append(activation)
        self._metrics.n_corpus = len(self._corpus)

    def get_status(self) -> DaemonStatus:
        """Get current daemon status."""
        return self._status

    def get_metrics(self) -> CoverageMetrics:
        """Get current coverage metrics."""
        return self._metrics

    def _transition_state(self, new_state: DaemonState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._status.state = new_state
        self._status.last_state_change = datetime.utcnow().isoformat()
        logger.debug("Daemon state: %s → %s", old_state.value, new_state.value)

    async def _check_convergence(self) -> bool:
        """Check if exploration has converged.

        Convergence condition: coverage_rate > 0 and coverage_rate < sqrt(eps)
        This is geometry-derived from machine precision.

        Returns
        -------
        bool
            True if converged.
        """
        # No corpus = not converged
        if len(self._corpus) == 0:
            return False

        # Compute coverage metrics
        if len(self._corpus) >= 2:
            corpus_arr = self._backend.stack(self._corpus, axis=0)
            self._backend.eval(corpus_arr)

            # Get acquisition result for coverage metrics
            result = self._acquisition.score(corpus_arr, corpus_arr)
            self._metrics.coverage_radius = result.coverage_radius
            self._metrics.mean_local_id = result.mean_local_id
            self._metrics.sparse_fraction = result.sparse_fraction

        # Compute coverage rate (improvement since last iteration)
        # Positive rate = radius shrinking (good), negative = radius growing (bad)
        if self._metrics.previous_radius > self._sqrt_eps:
            radius_change = self._metrics.previous_radius - self._metrics.coverage_radius
            self._metrics.coverage_rate = radius_change / self._metrics.previous_radius
        else:
            self._metrics.coverage_rate = 0.0

        # Update previous radius
        self._metrics.previous_radius = self._metrics.coverage_radius

        # Convergence requires POSITIVE improvement that is small
        # - Negative rate means radius is growing (regression, NOT converged)
        # - Zero rate means no improvement (NOT converged)
        # - Small positive rate means shrinking slowly (converged)
        if self._metrics.coverage_rate > 0.0 and self._metrics.coverage_rate < self._sqrt_eps:
            self._status.convergence_reason = (
                f"Coverage rate {self._metrics.coverage_rate:.2e} < sqrt(eps) {self._sqrt_eps:.2e} (converging)"
            )
            return True

        # Check for stagnation: radius not improving at all
        if self._metrics.coverage_rate <= 0.0:
            # NOT converged - exploration is stuck or regressing
            # This will be handled by the exploration loop - don't terminate early
            logger.debug(
                "Coverage rate %.2e <= 0 (stagnant/regressing), continuing exploration",
                self._metrics.coverage_rate,
            )

        # Also check sparse fraction - if manifold is uniformly dense
        if self._metrics.sparse_fraction < self._sqrt_eps:
            self._status.convergence_reason = (
                f"Sparse fraction {self._metrics.sparse_fraction:.2e} < sqrt(eps) (uniformly dense)"
            )
            return True

        return False

    async def _select_candidates(self) -> list[ProbeCandidate]:
        """Select candidates using EFE policy and acquisition.

        Returns
        -------
        list[ProbeCandidate]
            Selected candidates for probing.
        """
        if len(self._candidates) < self._config.min_candidates:
            return []

        # Rank by EFE policy
        ranked = self._policy.rank_candidates(self._candidates)

        # Build corpus array
        if len(self._corpus) >= 2:
            corpus_arr = self._backend.stack(self._corpus, axis=0)
            self._backend.eval(corpus_arr)

            # Build candidate activations (using coordinates)
            candidate_coords = [
                self._backend.array(c.coordinates) for c in ranked
            ]
            if candidate_coords:
                candidates_arr = self._backend.stack(candidate_coords, axis=0)
                self._backend.eval(candidates_arr)

                # Select batch using composite acquisition
                selected_indices = self._acquisition.select_batch(
                    candidates_arr,
                    corpus_arr,
                    self._config.batch_size,
                )

                return [ranked[i] for i in selected_indices]

        # If no corpus, just take top by EFE score
        return ranked[: self._config.batch_size]

    async def _execute_probes(
        self,
        candidates: list[ProbeCandidate],
    ) -> list["Array"]:
        """Execute probes through the model.

        Parameters
        ----------
        candidates : list[ProbeCandidate]
            Candidates to probe.

        Returns
        -------
        list[Array]
            Resulting activations from probes.
        """
        if self._probe_executor is None:
            logger.warning("No probe executor set - skipping execution")
            return []

        results: list["Array"] = []
        for candidate in candidates:
            try:
                result = self._probe_executor(candidate)
                result = self._backend.array(result)
                self._backend.eval(result)
                results.append(result)

                self._status.probes_executed += 1
                self._metrics.n_probes_executed += 1

                if self._config.on_probe_complete:
                    self._config.on_probe_complete(candidate, result)

            except Exception as exc:
                logger.error("Probe execution failed: %s", exc)

        return results

    async def _maybe_consolidate(self) -> bool:
        """Check and trigger consolidation if needed.

        Returns
        -------
        bool
            True if consolidation was triggered.
        """
        if not self._config.enable_auto_consolidate:
            return False

        if self._consolidation_service is None:
            return False

        # Check EFE policy for consolidation decision
        # Don't override mean_capacity - let policy compute from candidates' capacity_fraction
        # sparse_fraction is density-related, not capacity. The candidates have actual
        # capacity information derived from null space analysis.
        curiosity_state = self._policy.create_state(
            self._candidates,
            # mean_capacity computed from candidates by default
        )
        self._status.current_curiosity_state = curiosity_state

        action, _ = self._policy.select_action(curiosity_state)

        if action == CuriosityAction.CONSOLIDATE:
            self._transition_state(DaemonState.CONSOLIDATING)

            try:
                # This would trigger the consolidation service
                # For now, just log
                logger.info("Consolidation triggered by curiosity policy")
                self._status.consolidations_triggered += 1

                # if self._config.on_consolidation:
                #     stats = await self._consolidation_service.consolidate_async()
                #     self._config.on_consolidation(stats)

                return True
            finally:
                self._transition_state(DaemonState.IDLE)

        return False

    async def _run_iteration(self) -> bool:
        """Run one exploration iteration.

        Returns
        -------
        bool
            True if iteration completed, False if should stop.
        """
        # SELECTING
        self._transition_state(DaemonState.SELECTING)
        selected = await self._select_candidates()

        if not selected:
            self._transition_state(DaemonState.IDLE)
            return True  # Continue but nothing to do

        # Remove selected from candidates
        selected_set = set(id(c) for c in selected)
        self._candidates = [c for c in self._candidates if id(c) not in selected_set]

        # ACQUIRING (scores already computed in selection)
        self._transition_state(DaemonState.ACQUIRING)

        # EXECUTING
        self._transition_state(DaemonState.EXECUTING)
        results = await self._execute_probes(selected)

        # Add results to corpus
        for result in results:
            self.add_to_corpus(result)

        # MEASURING
        self._transition_state(DaemonState.MEASURING)
        if await self._check_convergence():
            self._transition_state(DaemonState.CONVERGED)
            return False  # Stop

        # Maybe consolidate
        await self._maybe_consolidate()

        # Back to IDLE
        self._transition_state(DaemonState.IDLE)
        self._status.iterations_completed += 1

        # Check max iterations
        if (
            self._config.max_iterations > 0
            and self._status.iterations_completed >= self._config.max_iterations
        ):
            self._status.convergence_reason = (
                f"Max iterations ({self._config.max_iterations}) reached"
            )
            self._transition_state(DaemonState.CONVERGED)
            return False

        return True

    async def _daemon_loop(self) -> None:
        """Main daemon loop."""
        self._transition_state(DaemonState.IDLE)

        while self._running:
            try:
                async with self._lock:
                    should_continue = await self._run_iteration()

                if not should_continue:
                    break

                await asyncio.sleep(self._config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Daemon loop error: %s", exc)
                await asyncio.sleep(self._config.check_interval)

        self._transition_state(DaemonState.STOPPED)

    def start(self) -> None:
        """Start the curiosity daemon."""
        if self._running:
            return

        self._running = True
        self._status.is_running = True
        self._daemon_task = asyncio.create_task(self._daemon_loop())
        logger.info("Curiosity daemon started")

    async def stop(self) -> None:
        """Stop the curiosity daemon."""
        self._running = False
        self._status.is_running = False

        if self._daemon_task:
            self._daemon_task.cancel()
            try:
                await self._daemon_task
            except asyncio.CancelledError:
                pass
            self._daemon_task = None

        self._transition_state(DaemonState.STOPPED)
        logger.info("Curiosity daemon stopped")

    async def run_until_converged(self) -> DaemonStatus:
        """Run daemon until convergence (blocking).

        Returns
        -------
        DaemonStatus
            Final status after convergence.
        """
        self.start()

        try:
            while self._running and self._state != DaemonState.CONVERGED:
                await asyncio.sleep(self._config.check_interval)
        finally:
            await self.stop()

        return self._status


def create_curiosity_daemon(
    hidden_dim: int,
    backend: "Backend | None" = None,
    max_iterations: int = 0,
    batch_size: int = 10,
    check_interval: float = 1.0,
    enable_auto_consolidate: bool = True,
    consolidation_service: "ConsolidationService | None" = None,
) -> CuriosityDaemon:
    """Create a curiosity daemon.

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension.
    backend : Backend, optional
        Compute backend.
    max_iterations : int
        Maximum iterations (0 = no limit).
    batch_size : int
        Probes per iteration.
    check_interval : float
        Seconds between state machine ticks.
    enable_auto_consolidate : bool
        Whether to auto-trigger consolidation.
    consolidation_service : ConsolidationService, optional
        Service for consolidation.

    Returns
    -------
    CuriosityDaemon
        Configured daemon instance.
    """
    config = DaemonConfig(
        max_iterations=max_iterations,
        batch_size=batch_size,
        check_interval=check_interval,
        enable_auto_consolidate=enable_auto_consolidate,
    )
    return CuriosityDaemon(
        hidden_dim=hidden_dim,
        backend=backend,
        config=config,
        consolidation_service=consolidation_service,
    )
