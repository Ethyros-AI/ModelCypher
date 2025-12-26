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
    from modelcypher.ports import ModelStore


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
    ) -> None:
        """Initialize SystemService with required dependencies.

        Args:
            model_store: Model store port implementation (REQUIRED).
        """
        self._model_store = model_store

    def status(self) -> dict:
        return self.readiness()

    def readiness(self) -> dict:
        metal_available = self._mlx_available()
        cuda_available = self._cuda_available()
        jax_available = self._jax_available()
        system_memory = self._system_memory_bytes()
        memory_gb = int(system_memory / (1024**3)) if system_memory else 0
        mlx_version = self._mlx_version()
        cuda_version = self._cuda_version()
        jax_version = self._jax_version()
        preferred_backend = self._preferred_backend(
            metal_available=metal_available,
            cuda_available=cuda_available,
            jax_available=jax_available,
        )

        # Disk usage check
        disk_total, disk_used, disk_free = self._disk_usage(self._model_store.paths.base)
        disk_free_gb = int(disk_free / (1024**3))

        # Scoring logic
        score = 0
        has_gpu_backend = metal_available or cuda_available or jax_available
        score += 40 if has_gpu_backend else 0
        score += 20 if memory_gb >= 16 else (10 if memory_gb >= 8 else 0)
        score += 20 if disk_free_gb >= 50 else (10 if disk_free_gb >= 20 else 0)
        if preferred_backend == "mlx":
            score += 20 if mlx_version != "unavailable" else 0
        elif preferred_backend == "cuda":
            score += 20 if cuda_version != "unavailable" else 0
        elif preferred_backend == "jax":
            score += 20 if jax_version != "unavailable" else 0

        # Cap score at 100
        readiness_score = min(score, 100)

        return {
            "machineName": platform.node(),
            "unifiedMemoryGB": memory_gb,
            "mlxVersion": mlx_version,
            "cudaVersion": cuda_version,
            "jaxVersion": jax_version,
            "preferredBackend": preferred_backend,
            "readinessScore": readiness_score,
            "scoreBreakdown": {
                "totalScore": readiness_score,
                "datasetScore": 100,  # Placeholder until dataset service integration
                "memoryFitScore": 100 if memory_gb >= 16 else 50,
                "systemPressureScore": 100,  # Placeholder
                "mlxHealthScore": 100 if mlx_version != "unavailable" else 0,
                "cudaHealthScore": 100 if cuda_version != "unavailable" else 0,
                "jaxHealthScore": 100 if jax_version != "unavailable" else 0,
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
            "cudaAvailable": cuda_available,
            "jaxAvailable": jax_available,
            "blockers": [] if has_gpu_backend else ["No GPU backend available"],
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
        cuda_available = self._cuda_available()
        jax_available = self._jax_available()
        data = {
            "target": target,
            "metal": {"available": metal_available, "deviceName": platform.machine()},
            "mlx": {"version": self._mlx_version(), "gpuAvailable": metal_available},
            "cuda": {
                "available": cuda_available,
                "version": self._cuda_version(),
                "deviceName": self._cuda_device_name(),
                "flashAttentionAvailable": self._cuda_flash_attention_available(),
                "flashAttentionEnabled": self._cuda_flash_attention_enabled(),
            },
            "jax": {
                "available": jax_available,
                "version": self._jax_version(),
                "defaultBackend": self._jax_default_backend(),
                "devicePlatforms": self._jax_device_platforms(),
            },
            "memory": {"systemBytes": system_memory, "gpuBytes": gpu_memory},
        }
        if target == "metal":
            return {"target": target, "metal": data["metal"]}
        if target == "mlx":
            return {"target": target, "mlx": data["mlx"]}
        if target == "cuda":
            return {"target": target, "cuda": data["cuda"]}
        if target == "jax":
            return {"target": target, "jax": data["jax"]}
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
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _cuda_version() -> str:
        try:
            import torch

            version = torch.version.cuda
            return version if version is not None else "unknown"
        except Exception:
            return "unavailable"

    @staticmethod
    def _cuda_device_name() -> str | None:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return torch.cuda.get_device_name(0)
        except Exception:
            return None

    @staticmethod
    def _cuda_flash_attention_available() -> bool:
        try:
            import torch

            return torch.backends.cuda.is_flash_attention_available()
        except Exception:
            return False

    @staticmethod
    def _cuda_flash_attention_enabled() -> bool:
        try:
            import torch

            return torch.backends.cuda.can_use_flash_attention()
        except Exception:
            return False

    @staticmethod
    def _jax_available() -> bool:
        try:
            import jax

            return bool(jax.devices())
        except Exception:
            return False

    @staticmethod
    def _jax_version() -> str:
        try:
            import jax

            return getattr(jax, "__version__", "unknown")
        except Exception:
            return "unavailable"

    @staticmethod
    def _jax_default_backend() -> str:
        try:
            import jax

            return jax.default_backend()
        except Exception:
            return "unavailable"

    @staticmethod
    def _jax_device_platforms() -> list[str]:
        try:
            import jax

            return sorted({device.platform for device in jax.devices()})
        except Exception:
            return []

    @staticmethod
    def _preferred_backend(
        metal_available: bool,
        cuda_available: bool,
        jax_available: bool,
    ) -> str:
        import os

        env_backend = os.environ.get("MC_BACKEND", "").lower()
        if not env_backend:
            env_backend = os.environ.get("MODELCYPHER_BACKEND", "").lower()
        if env_backend in ("mlx", "cuda", "jax"):
            return env_backend
        if metal_available:
            return "mlx"
        if cuda_available:
            return "cuda"
        if jax_available:
            return "jax"
        return "cpu"

    @staticmethod
    def _system_memory_bytes() -> int:
        try:
            import os

            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            return 0
