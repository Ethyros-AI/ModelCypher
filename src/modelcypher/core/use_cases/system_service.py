from __future__ import annotations

import platform
from pathlib import Path

from modelcypher.adapters.filesystem_storage import FileSystemStore


class SystemService:
    def __init__(self, store: FileSystemStore | None = None) -> None:
        self.store = store or FileSystemStore()

    def status(self) -> dict:
        return self.readiness()

    def readiness(self) -> dict:
        metal_available = self._mlx_available()
        system_memory = self._system_memory_bytes()
        unified_gb = int(system_memory / (1024**3)) if system_memory else 0
        mlx_version = self._mlx_version()
        
        # Disk usage check
        disk_total, disk_used, disk_free = self._disk_usage(self.store.paths.base)
        disk_free_gb = int(disk_free / (1024**3))
        
        # Scoring logic
        score = 0
        score += 40 if metal_available else 0
        score += 20 if unified_gb >= 16 else (10 if unified_gb >= 8 else 0)
        score += 20 if disk_free_gb >= 50 else (10 if disk_free_gb >= 20 else 0)
        score += 20 if mlx_version != "unavailable" else 0
        
        # Cap score at 100
        readiness_score = min(score, 100)

        return {
            "machineName": platform.node(),
            "unifiedMemoryGB": unified_gb,
            "mlxVersion": mlx_version,
            "readinessScore": readiness_score,
            "scoreBreakdown": {
                "totalScore": readiness_score,
                "datasetScore": 100, # Placeholder until dataset service integration
                "memoryFitScore": 100 if unified_gb >= 16 else 50,
                "systemPressureScore": 100, # Placeholder
                "mlxHealthScore": 100 if mlx_version != "unavailable" else 0,
                "storageScore": 100 if disk_free_gb > 100 else 50,
                "preflightScore": readiness_score,
                "band": "excellent" if readiness_score >= 90 else ("good" if readiness_score >= 70 else "warning"),
            },
            "resources": {
                "gpuMemoryBytes": system_memory // 2 if system_memory else 0,
                "systemMemoryBytes": system_memory,
                "diskFreeBytes": disk_free,
            },
            "metalAvailable": metal_available,
            "blockers": [] if metal_available else ["MLX/Metal not available"],
        }
    
    def _disk_usage(self, path: Path) -> tuple[int, int, int]:
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            return total, used, free
        except Exception:
            return 0, 0, 0

    def probe(self, target: str) -> dict:
        metal_available = self._mlx_available()
        system_memory = self._system_memory_bytes()
        gpu_memory = system_memory // 2 if system_memory else 0
        data = {
            "target": target,
            "metal": {"available": metal_available, "deviceName": platform.machine()},
            "mlx": {"version": self._mlx_version(), "gpuAvailable": metal_available},
            "memory": {"systemBytes": system_memory, "gpuBytes": gpu_memory},
        }
        if target == "metal":
            return {"target": target, "metal": data["metal"]}
        if target == "mlx":
            return {"target": target, "mlx": data["mlx"]}
        if target == "memory":
            return {"target": target, "memory": data["memory"]}
        return data

    @staticmethod
    def _mlx_available() -> bool:
        try:
            import mlx.core  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _mlx_version() -> str:
        try:
            import mlx

            return getattr(mlx, "__version__", "unknown")
        except Exception:
            return "unavailable"

    @staticmethod
    def _system_memory_bytes() -> int:
        try:
            import os

            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            return 0
