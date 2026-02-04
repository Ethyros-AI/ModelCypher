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


from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from modelcypher.core.domain.agents.agent_trace import InferenceMetrics

# --- Dual Path Types ---


@dataclass
class SecurityScanMetrics:
    min_kl: float
    max_kl: float
    mean_kl: float
    entropy_variance: float
    conflict_rate: float  # % of tokens with high KL


# --- Comparison Types ---


class ComparisonTimeouts:
    def __init__(self, idle_sec: float, absolute_sec: float):
        self.idle_sec = idle_sec
        self.absolute_sec = absolute_sec


@dataclass
class ComparisonResult:
    checkpoint_path: str
    response: str
    metrics: "InferenceMetrics | None"


class EventType(Enum):
    PREFETCH_STARTED = "prefetch_started"
    PREFETCH_FINISHED = "prefetch_finished"
    PREFETCH_FAILED = "prefetch_failed"
    CHECKPOINT_STARTED = "checkpoint_started"
    TOKEN = "token"
    CHECKPOINT_FINISHED = "checkpoint_finished"
    CHECKPOINT_FAILED = "checkpoint_failed"


@dataclass
class ComparisonEvent:
    type: EventType
    index: int
    path: str | None = None
    text: str | None = None
    result: ComparisonResult | None = None
    error: str | None = None


# --- Adapter Pool Types ---


class AdapterPreloadPriority(Enum):
    """Priority levels for adapter preloading.

    Higher priority adapters are less likely to be evicted from the pool.
    """

    NORMAL = 0
    HIGH = 1
    CRITICAL = 2

    def __lt__(self, other: "AdapterPreloadPriority") -> bool:
        """Allow comparison for priority-based eviction."""
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclass
class AdapterPoolEntry:
    """Entry in the adapter pool."""

    id: uuid.UUID
    path: str
    priority: AdapterPreloadPriority
    estimated_memory_bytes: int
    last_accessed_at: float = field(default_factory=time.time)


@dataclass
class AdapterSwapResult:
    """Result of an adapter swap operation."""

    previous_adapter_id: uuid.UUID | None
    new_adapter_id: uuid.UUID | None
    swap_duration_ms: float
    was_cache_hit: bool


class AdapterPoolError(Exception):
    """Errors raised by adapter pool operations."""


@dataclass
class MemoryStats:
    """System memory statistics for adapter pool capacity management."""

    available_bytes: int
    total_bytes: int

    @property
    def available_ratio(self) -> float:
        return self.available_bytes / self.total_bytes if self.total_bytes else 0.0


class MemoryManaging(Protocol):
    """Protocol for memory management implementations."""

    async def memory_stats(self) -> MemoryStats: ...
