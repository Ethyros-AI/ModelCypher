from __future__ import annotations

import platform


class SystemService:
    def status(self) -> dict:
        metal_available = self._mlx_available()
        system_memory = self._system_memory_bytes()
        gpu_memory = system_memory // 2 if system_memory else 0
        return {
            "metalAvailable": metal_available,
            "gpuMemoryBytes": gpu_memory,
            "systemMemoryBytes": system_memory,
            "activeJobs": 0,
            "thermalState": "nominal",
        }

    def readiness(self) -> dict:
        system_memory = self._system_memory_bytes()
        unified_gb = int(system_memory / (1024**3)) if system_memory else 0
        mlx_version = self._mlx_version()
        readiness_score = 95 if self._mlx_available() else 40
        return {
            "machineName": platform.node(),
            "unifiedMemoryGB": unified_gb,
            "mlxVersion": mlx_version,
            "readinessScore": readiness_score,
            "scoreBreakdown": {
                "totalScore": readiness_score,
                "datasetScore": 100,
                "memoryFitScore": 90 if unified_gb else 0,
                "systemPressureScore": 100,
                "mlxHealthScore": 100 if mlx_version != "unavailable" else 0,
                "storageScore": 85,
                "preflightScore": readiness_score,
                "band": "excellent" if readiness_score >= 90 else "warning",
            },
            "blockers": [],
        }

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
