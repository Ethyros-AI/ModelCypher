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

"""Background consolidation with geometry-based triggers.

Automatic consolidation when geometric conditions warrant - NOT time-based.

The Geometric Trigger Condition:
    should_consolidate = (
        not lock.is_locked() AND                      # System idle
        len(sparsity_queue) >= MIN_EVENTS AND         # Accumulated surprise
        mean(event.eigenscore) > 2 * sqrt(eps) AND    # Meaningful sparsity
        mean(capacity_fraction) > sqrt(eps)           # Room in model
    )

Threshold Derivations:
    sqrt(eps): Machine precision - values below are indistinguishable from noise
    2 * sqrt(eps): Standard numerical analysis margin above noise floor
    MIN_EVENTS: max(20, hidden_dim/32) from k-NN density estimation (Facco et al. 2017)

Why hidden_dim/32? Observed compression ratio from hidden_dim to intrinsic
manifold dimension. Need ~30 samples for stable TwoNN estimation.
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


from modelcypher.core.use_cases.entropy_learning_bridge import SparsityEvent

if TYPE_CHECKING:
    from modelcypher.core.use_cases.consolidation_service import (
        ConsolidationService,
        ConsolidationStats,
    )
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


class ConsolidationTriggerReason(str, Enum):
    """Reason why consolidation was triggered."""

    GEOMETRIC = "geometric"  # Geometric conditions met
    MANUAL = "manual"  # User requested
    CAPACITY = "capacity"  # Null space nearly full
    QUEUE_FULL = "queue_full"  # Sparsity queue at limit


@dataclass
class GeometricConditions:
    """Current geometric conditions for consolidation decision.

    All values are raw measurements - no interpretation.
    """

    # Event statistics
    event_count: int = 0
    min_events_required: int = 20

    # Mean EigenScore across queued events
    mean_eigenscore: float = 0.0
    eigenscore_threshold: float = 0.0  # 2 * sqrt(eps)

    # Mean capacity fraction across layers
    mean_capacity_fraction: float = 0.0
    capacity_threshold: float = 0.0  # sqrt(eps)

    # System state
    system_idle: bool = False

    # Derived
    sqrt_eps: float = 0.0
    should_consolidate: bool = False
    trigger_reason: ConsolidationTriggerReason | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "event_count": self.event_count,
            "min_events_required": self.min_events_required,
            "mean_eigenscore": self.mean_eigenscore,
            "eigenscore_threshold": self.eigenscore_threshold,
            "mean_capacity_fraction": self.mean_capacity_fraction,
            "capacity_threshold": self.capacity_threshold,
            "system_idle": self.system_idle,
            "sqrt_eps": self.sqrt_eps,
            "should_consolidate": self.should_consolidate,
            "trigger_reason": self.trigger_reason.value if self.trigger_reason else None,
        }


@dataclass
class MonitorStatus:
    """Status of the background consolidation monitor."""

    is_running: bool = False
    is_consolidating: bool = False
    last_check_time: str = ""
    last_consolidation_time: str | None = None
    consolidation_count: int = 0
    total_events_processed: int = 0
    current_conditions: GeometricConditions | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "is_running": self.is_running,
            "is_consolidating": self.is_consolidating,
            "last_check_time": self.last_check_time,
            "last_consolidation_time": self.last_consolidation_time,
            "consolidation_count": self.consolidation_count,
            "total_events_processed": self.total_events_processed,
            "current_conditions": (
                self.current_conditions.to_dict()
                if self.current_conditions
                else None
            ),
        }


@dataclass
class MonitorConfig:
    """Configuration for the background consolidation monitor.

    All values except check_interval are derived from machine precision
    or geometric requirements.
    """

    # How often to check conditions (seconds) - only time-based parameter
    check_interval: float = 30.0

    # Maximum queue size before forced consolidation
    max_queue_size: int = 1000

    # Enable/disable automatic consolidation
    enabled: bool = True

    # Callback when consolidation completes
    on_consolidation: Callable[["ConsolidationStats"], None] | None = None


class BackgroundConsolidationMonitor:
    """Background monitor that triggers consolidation on geometric conditions.

    Unlike time-based schedulers, this monitor uses geometric signals:
    - Accumulated sparsity events (surprise)
    - Mean EigenScore (manifold sparsity)
    - Mean capacity fraction (null space availability)
    - System lock status (inference not running)

    All thresholds are derived from machine precision (sqrt(eps)).
    """

    def __init__(
        self,
        consolidation_service: "ConsolidationService",
        hidden_dim: int,
        backend: "Backend",
        config: MonitorConfig | None = None,
    ) -> None:
        """Initialize the background consolidation monitor.

        Args:
            consolidation_service: Service to run consolidation.
            hidden_dim: Model hidden dimension (for MIN_EVENTS calculation).
            backend: Compute backend.
            config: Monitor configuration.
        """
        self._backend = backend
        self._service = consolidation_service
        self._hidden_dim = hidden_dim
        self._config = config or MonitorConfig()

        # Compute sqrt(eps) for thresholds
        ref_array = self._backend.array([1.0])
        eps = self._backend.finfo(ref_array.dtype).eps
        self._sqrt_eps = math.sqrt(float(eps))

        # Derive MIN_EVENTS from geometry
        # k-NN density estimation needs ~30 samples for stable results
        # hidden_dim/32 is the observed intrinsic dimension compression ratio
        self._min_events = max(20, hidden_dim // 32)

        # State
        self._sparsity_queue: list[SparsityEvent] = []
        self._capacity_fractions: dict[int, float] = {}
        self._status = MonitorStatus()
        self._monitor_task: asyncio.Task | None = None
        self._running = False

        # Lock for inference coordination
        self._lock = asyncio.Lock()

    def add_sparsity_event(self, event: SparsityEvent) -> None:
        """Add a sparsity event to the queue.

        Called during inference when WARN action is triggered.

        Args:
            event: Sparsity event from entropy learning bridge.
        """
        self._sparsity_queue.append(event)

        # Check for queue overflow
        if len(self._sparsity_queue) > self._config.max_queue_size:
            logger.warning(
                "Sparsity queue overflow (%d > %d), forcing consolidation",
                len(self._sparsity_queue),
                self._config.max_queue_size,
            )
            # Don't block - just mark for next check
            self._status.current_conditions = self._evaluate_conditions()
            if self._status.current_conditions:
                self._status.current_conditions.trigger_reason = (
                    ConsolidationTriggerReason.QUEUE_FULL
                )

    def update_capacity(self, layer_id: int, capacity_fraction: float) -> None:
        """Update capacity fraction for a layer.

        Called after null-space analysis.

        Args:
            layer_id: Layer index.
            capacity_fraction: Available capacity [0, 1].
        """
        self._capacity_fractions[layer_id] = capacity_fraction

    def get_queue_size(self) -> int:
        """Get current sparsity queue size."""
        return len(self._sparsity_queue)

    def get_status(self) -> MonitorStatus:
        """Get current monitor status."""
        return self._status

    def _evaluate_conditions(self) -> GeometricConditions:
        """Evaluate geometric conditions for consolidation.

        Returns:
            GeometricConditions with all measurements and decision.
        """
        conditions = GeometricConditions(
            event_count=len(self._sparsity_queue),
            min_events_required=self._min_events,
            sqrt_eps=self._sqrt_eps,
            eigenscore_threshold=2 * self._sqrt_eps,
            capacity_threshold=self._sqrt_eps,
        )

        # Check event count
        if conditions.event_count < conditions.min_events_required:
            conditions.should_consolidate = False
            return conditions

        # Compute mean EigenScore
        if self._sparsity_queue:
            eigenscores = [e.eigenscore for e in self._sparsity_queue]
            conditions.mean_eigenscore = sum(eigenscores) / len(eigenscores)

        # Compute mean capacity fraction
        if self._capacity_fractions:
            conditions.mean_capacity_fraction = sum(
                self._capacity_fractions.values()
            ) / len(self._capacity_fractions)

        # Check system idle (not consolidating)
        conditions.system_idle = not self._status.is_consolidating

        # Evaluate trigger conditions
        # All derived from sqrt(eps) - machine precision
        eigenscore_met = conditions.mean_eigenscore > conditions.eigenscore_threshold
        capacity_met = conditions.mean_capacity_fraction > conditions.capacity_threshold
        count_met = conditions.event_count >= conditions.min_events_required

        if conditions.system_idle and count_met and eigenscore_met and capacity_met:
            conditions.should_consolidate = True
            conditions.trigger_reason = ConsolidationTriggerReason.GEOMETRIC
        elif conditions.event_count >= self._config.max_queue_size:
            conditions.should_consolidate = True
            conditions.trigger_reason = ConsolidationTriggerReason.QUEUE_FULL

        return conditions

    async def _run_consolidation(self) -> "ConsolidationStats | None":
        """Run consolidation on queued events.

        Returns:
            ConsolidationStats if successful, None otherwise.
        """
        from modelcypher.core.use_cases.consolidation_service import (
            ConsolidationConfig,
        )
        from modelcypher.core.use_cases.entropy_learning_bridge import (
            EntropyLearningBridge,
        )

        if not self._sparsity_queue:
            return None

        async with self._lock:
            self._status.is_consolidating = True

            try:
                # Create bridge with queued events
                bridge = EntropyLearningBridge(hidden_dim=self._hidden_dim)
                for event in self._sparsity_queue:
                    bridge._sparsity_queue.append(event)

                # Run consolidation
                config = ConsolidationConfig(
                    max_probes=min(100, len(self._sparsity_queue)),
                    max_completion_steps=50,
                    clear_queue_after=True,
                )
                stats = self._service.consolidate_from_bridge(bridge, config)

                # Update status
                events_processed = len(self._sparsity_queue)
                self._sparsity_queue.clear()
                self._status.consolidation_count += 1
                self._status.total_events_processed += events_processed
                self._status.last_consolidation_time = datetime.utcnow().isoformat()

                # Callback if configured
                if self._config.on_consolidation:
                    self._config.on_consolidation(stats)

                logger.info(
                    "Background consolidation complete: %d events processed",
                    events_processed,
                )

                return stats

            except Exception as exc:
                logger.error("Background consolidation failed: %s", exc)
                return None

            finally:
                self._status.is_consolidating = False

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Update status
                self._status.last_check_time = datetime.utcnow().isoformat()

                # Evaluate conditions
                conditions = self._evaluate_conditions()
                self._status.current_conditions = conditions

                # Trigger consolidation if conditions met
                if conditions.should_consolidate and self._config.enabled:
                    logger.info(
                        "Geometric conditions met: %s (events=%d, eigenscore=%.4f, capacity=%.4f)",
                        conditions.trigger_reason,
                        conditions.event_count,
                        conditions.mean_eigenscore,
                        conditions.mean_capacity_fraction,
                    )
                    await self._run_consolidation()

                # Wait for next check
                await asyncio.sleep(self._config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Monitor loop error: %s", exc)
                await asyncio.sleep(self._config.check_interval)

    def start(self) -> None:
        """Start the background monitor."""
        if self._running:
            return

        self._running = True
        self._status.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Background consolidation monitor started")

    async def stop(self) -> None:
        """Stop the background monitor."""
        self._running = False
        self._status.is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Background consolidation monitor stopped")

    async def trigger_manual(self) -> "ConsolidationStats | None":
        """Manually trigger consolidation.

        Returns:
            ConsolidationStats if successful, None otherwise.
        """
        logger.info("Manual consolidation triggered")
        conditions = self._evaluate_conditions()
        conditions.trigger_reason = ConsolidationTriggerReason.MANUAL
        self._status.current_conditions = conditions
        return await self._run_consolidation()

    def get_conditions(self) -> GeometricConditions:
        """Get current geometric conditions without triggering.

        Returns:
            Current GeometricConditions.
        """
        return self._evaluate_conditions()


def create_background_monitor(
    consolidation_service: "ConsolidationService",
    hidden_dim: int,
    check_interval: float = 30.0,
    max_queue_size: int = 1000,
    enabled: bool = True,
) -> BackgroundConsolidationMonitor:
    """Create a background consolidation monitor.

    Args:
        consolidation_service: Service to run consolidation.
        hidden_dim: Model hidden dimension.
        check_interval: Seconds between condition checks.
        max_queue_size: Maximum sparsity events before forced consolidation.
        enabled: Whether to auto-consolidate.

    Returns:
        Configured BackgroundConsolidationMonitor.
    """
    config = MonitorConfig(
        check_interval=check_interval,
        max_queue_size=max_queue_size,
        enabled=enabled,
    )
    return BackgroundConsolidationMonitor(
        consolidation_service=consolidation_service,
        hidden_dim=hidden_dim,
        config=config,
    )
