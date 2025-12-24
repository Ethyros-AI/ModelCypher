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

import platform
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modelcypher.ports import DatasetStore, ModelStore


class _StorePaths(Protocol):
    """Protocol for store paths needed by SystemService."""

    base: Path


@runtime_checkable
class SystemServiceStore(Protocol):
    """Protocol for the store required by SystemService.

    Requires a paths attribute with a base path for disk usage checks.
    """

    paths: _StorePaths


class SystemService:
    def __init__(
        self,
        model_store: "ModelStore",
        dataset_store: "DatasetStore",
    ) -> None:
        """Initialize SystemService with required dependencies.

        Args:
            model_store: Model store port implementation (REQUIRED).
            dataset_store: Dataset store port implementation (REQUIRED).
        """
        self._model_store = model_store
        self._dataset_store = dataset_store

    def status(self) -> dict:
        return self.readiness()

    def readiness(self) -> dict:
        metal_available = self._mlx_available()
        system_memory = self._system_memory_bytes()
        unified_gb = int(system_memory / (1024**3)) if system_memory else 0
        mlx_version = self._mlx_version()

        # Disk usage check
        disk_total, disk_used, disk_free = self._disk_usage(self._model_store.paths.base)
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
                "datasetScore": 100,  # Placeholder until dataset service integration
                "memoryFitScore": 100 if unified_gb >= 16 else 50,
                "systemPressureScore": 100,  # Placeholder
                "mlxHealthScore": 100 if mlx_version != "unavailable" else 0,
                "storageScore": 100 if disk_free_gb > 100 else 50,
                "preflightScore": readiness_score,
                "band": "excellent"
                if readiness_score >= 90
                else ("good" if readiness_score >= 70 else "warning"),
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
