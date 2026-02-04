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

"""Memory monitoring service using Backend protocol.

No framework imports here - uses Backend for GPU memory stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


class MemoryPressure(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MemoryStats:
    total_gb: float
    available_gb: float
    used_gb: float
    pressure: MemoryPressure
    gpu_peak_gb: float
    gpu_active_gb: float


class MemoryService:
    """Service to monitor system and GPU memory usage.

    Uses Backend protocol for GPU-specific memory queries.
    """

    _instance = None

    def __new__(cls, backend: "Backend | None" = None):
        if cls._instance is None:
            cls._instance = super(MemoryService, cls).__new__(cls)
            cls._instance._backend = None
        return cls._instance

    def __init__(self, backend: "Backend | None" = None):
        if backend is not None:
            self._backend = backend
        elif self._backend is None:
            from modelcypher.core.domain._backend import get_default_backend
            self._backend = get_default_backend()

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        available_gb = vm.available / (1024**3)
        used_gb = vm.used / (1024**3)

        gpu_peak = self._backend.get_peak_memory_gb()
        gpu_active = self._backend.get_active_memory_gb()

        pressure = MemoryPressure.NORMAL
        if available_gb < 2.0:
            pressure = MemoryPressure.CRITICAL
        elif available_gb < 4.0:
            pressure = MemoryPressure.WARNING

        return MemoryStats(
            total_gb=round(total_gb, 2),
            available_gb=round(available_gb, 2),
            used_gb=round(used_gb, 2),
            pressure=pressure,
            gpu_peak_gb=round(gpu_peak, 2),
            gpu_active_gb=round(gpu_active, 2),
        )

    def clear_cache(self):
        """Force GPU cache cleanup."""
        self._backend.clear_cache()


# Backwards compatibility alias
MLXMemoryService = MemoryService
