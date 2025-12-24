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

import mlx.core as mx
import psutil
from dataclasses import dataclass
from enum import Enum

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
    mlx_peak_gb: float
    mlx_active_gb: float

class MLXMemoryService:
    """
    Service to monitor system and MLX-specific memory usage.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLXMemoryService, cls).__new__(cls)
        return cls._instance

    def get_memory_stats(self) -> MemoryStats:
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        available_gb = vm.available / (1024**3)
        used_gb = vm.used / (1024**3)
        
        mlx_peak = mx.metal.get_peak_memory() / (1024**3)
        mlx_active = mx.metal.get_active_memory() / (1024**3)
        
        pressure = MemoryPressure.NORMAL
        # Simple heuristics for pressure
        if available_gb < 2.0: # Less than 2GB free
            pressure = MemoryPressure.CRITICAL
        elif available_gb < 4.0:
            pressure = MemoryPressure.WARNING
            
        return MemoryStats(
            total_gb=round(total_gb, 2),
            available_gb=round(available_gb, 2),
            used_gb=round(used_gb, 2),
            pressure=pressure,
            mlx_peak_gb=round(mlx_peak, 2),
            mlx_active_gb=round(mlx_active, 2)
        )
    
    def clear_cache(self):
        """Force MLX cache cleanup."""
        mx.metal.clear_cache()
