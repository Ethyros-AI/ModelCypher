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
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import uuid
import time

# --- Dual Path Types ---

@dataclass
class SecurityScanMetrics:
    min_kl: float
    max_kl: float
    mean_kl: float
    entropy_variance: float
    conflict_rate: float # % of tokens with high KL
    
@dataclass
class DualPathGeneratorConfiguration:
    base_model_path: str
    adapter_path: str | None = None
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    # Safety thresholds
    max_kl_threshold: float = 2.5
    burst_length_limit: int = 5
    accumulated_kl_limit: float = 50.0

# --- Comparison Types ---

class ComparisonTimeouts:
    def __init__(self, idle_sec: float = 1800, absolute_sec: float = 7200):
        self.idle_sec = idle_sec
        self.absolute_sec = absolute_sec
        
@dataclass
class ComparisonResult:
    checkpoint_path: str
    response: str
    metrics: Any # InferenceMetrics type placeholder

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

class MemoryPressure(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class AdapterPoolConfiguration:
    max_pooled_normal: int = 4
    max_pooled_warning: int = 2
    max_pooled_critical: int = 1
    target_swap_ms: float = 100.0

@dataclass
class AdapterPoolEntry:
    id: uuid.UUID
    path: str
    priority: Any # AdapterPreloadPriority to be defined in protocol or separate enum
    estimated_memory_bytes: int
    last_accessed_at: float = field(default_factory=time.time)

@dataclass
class AdapterSwapResult:
    previous_adapter_id: uuid.UUID | None
    new_adapter_id: uuid.UUID | None
    swap_duration_ms: float
    was_cache_hit: bool

class AdapterPoolError(Exception):
    pass
