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
