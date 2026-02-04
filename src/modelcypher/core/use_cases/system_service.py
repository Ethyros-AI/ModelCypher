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

"""System status and readiness service."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from modelcypher.backends import BackendDescriptor


class _StorePaths(Protocol):
    base: Path


class _ModelStore(Protocol):
    paths: _StorePaths


class SystemService:
    def __init__(self, model_store: "_ModelStore") -> None:
        self._model_store = model_store

    def status(self) -> dict:
        return self.readiness()

    def readiness(self) -> dict:
        from modelcypher.backends import detect_default_backend_type, probe_backends

        probes = probe_backends(explicit=False)
        preferred_backend = detect_default_backend_type()
        preferred_probe = next(
            (probe for probe in probes if probe.key == preferred_backend),
            None,
        )
        has_backend = any(probe.available for probe in probes)
        system_memory = self._system_memory_bytes()
        memory_gb = int(system_memory / (1024**3)) if system_memory else 0
        backend_versions = {
            probe.key: probe.system_info.get("version")
            for probe in probes
        }

        disk_total, disk_used, disk_free = self._disk_usage(self._model_store.paths.base)
        disk_free_gb = int(disk_free / (1024**3))

        score = 0
        score += 40 if has_backend else 0
        score += 20 if memory_gb >= 16 else (10 if memory_gb >= 8 else 0)
        score += 20 if disk_free_gb >= 50 else (10 if disk_free_gb >= 20 else 0)
        if preferred_probe and preferred_probe.available:
            score += 20

        readiness_score = min(score, 100)

        backend_health = {
            probe.key: 100 if probe.available else 0
            for probe in probes
        }

        return {
            "machineName": platform.node(),
            "preferredBackend": preferred_backend,
            "readinessScore": readiness_score,
            "scoreBreakdown": {
                "totalScore": readiness_score,
                "datasetScore": 100,
                "memoryFitScore": 100 if memory_gb >= 16 else 50,
                "systemPressureScore": 100,
                "backendHealth": backend_health,
                "storageScore": 100 if disk_free_gb > 100 else 50,
                "preflightScore": readiness_score,
            },
            "resources": {
                "gpuMemoryBytes": system_memory // 2 if system_memory else 0,
                "systemMemoryBytes": system_memory,
                "diskFreeBytes": disk_free,
            },
            "backends": [self._probe_payload(probe) for probe in probes],
            "backendVersions": backend_versions,
            "blockers": [] if has_backend else ["No backend available"],
        }

    def _disk_usage(self, path: Path) -> tuple[int, int, int]:
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            return total, used, free
        except Exception:
            return 0, 0, 0

    def probe(self, target: str) -> dict:
        from modelcypher.backends import probe_backends

        probes = probe_backends(explicit=True)
        system_memory = self._system_memory_bytes()
        gpu_memory = system_memory // 2 if system_memory else 0
        memory_payload = {"systemBytes": system_memory, "gpuBytes": gpu_memory}
        backend_payloads = [self._probe_payload(probe) for probe in probes]

        if target == "memory":
            return {"target": target, "memory": memory_payload}
        for probe in probes:
            if target == probe.key:
                return {
                    "target": target,
                    "backend": self._probe_payload(probe),
                    "memory": memory_payload,
                }
        if target in ("backends", "all"):
            return {"target": target, "backends": backend_payloads, "memory": memory_payload}
        return {"target": target, "backends": backend_payloads, "memory": memory_payload}

    @staticmethod
    def _probe_payload(probe: "BackendDescriptor") -> dict:
        return {
            "key": probe.key,
            "displayName": probe.display_name,
            "available": probe.available,
            "error": probe.error,
            "systemInfo": probe.system_info,
        }

    @staticmethod
    def _system_memory_bytes() -> int:
        try:
            import os
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            return 0
