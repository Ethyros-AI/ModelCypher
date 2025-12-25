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

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

from modelcypher.core.domain.storage_usage import DiskStats, StorageSnapshot, StorageUsage
from modelcypher.utils.paths import ensure_dir, expand_path

if TYPE_CHECKING:
    from modelcypher.ports.storage import DatasetStore, JobStore, ModelStore


BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class _CachedSnapshot:
    snapshot: StorageSnapshot
    expires_at: float


class _DiskUsage(Protocol):
    total: int
    free: int


class StorageService:
    def __init__(
        self,
        model_store: "ModelStore",
        job_store: "JobStore",
        dataset_store: "DatasetStore",
        base_dir: Path,
        logs_dir: Path,
        disk_usage_provider: Callable[[str], _DiskUsage] | None = None,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        self._model_store = model_store
        self._job_store = job_store
        self._dataset_store = dataset_store
        self._disk_usage_provider = disk_usage_provider or shutil.disk_usage
        self._cache_ttl_seconds = max(0.0, cache_ttl_seconds)
        self._cached_snapshot: _CachedSnapshot | None = None

        self._base_dir = base_dir
        self._logs_dir = logs_dir
        self._caches_dir = self._base_dir / "caches"
        self._exports_dir = self._base_dir / "exports"
        self._hf_cache_dir = self._resolve_hf_cache_dir()

    def storage_usage(self) -> StorageUsage:
        return self._compute_snapshot().usage

    def compute_snapshot(self) -> StorageSnapshot:
        return self._compute_snapshot()

    def invalidate_cache(self) -> None:
        self._cached_snapshot = None

    def cleanup(self, targets: list[str]) -> list[str]:
        normalized = {target.lower().strip() for target in targets if target}
        cleared: list[str] = []

        if "caches" in normalized:
            self._clear_directory_contents(self._caches_dir)
            self._clear_directory_contents(self._hf_cache_dir)
            ensure_dir(self._caches_dir)
            ensure_dir(self._hf_cache_dir)
            cleared.append("caches")

        if not cleared:
            raise ValueError("No valid cleanup targets selected.")

        self.invalidate_cache()
        return cleared

    def _compute_snapshot(self) -> StorageSnapshot:
        now = time.time()
        cached = self._cached_snapshot
        if cached and cached.expires_at > now:
            return cached.snapshot

        disk = self._disk_usage_provider(str(self._base_dir))
        disk_stats = DiskStats(total_bytes=disk.total, free_bytes=disk.free)

        models_bytes = self._sum_paths(
            Path(model.path) for model in self._model_store.list_models() if model.path
        )
        checkpoints_bytes = self._sum_paths(
            Path(checkpoint.file_path) for checkpoint in self._job_store.list_checkpoints()
        )

        dataset_bytes = 0
        for dataset in self._dataset_store.list_datasets():
            path = Path(dataset.path)
            if path.exists():
                dataset_bytes += self._path_size(path)
            else:
                dataset_bytes += int(dataset.size_bytes)

        other_bytes = dataset_bytes
        other_bytes += self._path_size(self._caches_dir)
        other_bytes += self._path_size(self._hf_cache_dir)
        other_bytes += self._path_size(self._exports_dir)
        other_bytes += self._path_size(self._logs_dir)

        usage = StorageUsage(
            total_gb=float(disk_stats.total_bytes) / BYTES_PER_GB,
            models_gb=float(models_bytes) / BYTES_PER_GB,
            checkpoints_gb=float(checkpoints_bytes) / BYTES_PER_GB,
            other_gb=float(other_bytes) / BYTES_PER_GB,
        )
        snapshot = StorageSnapshot(usage=usage, disk=disk_stats)
        self._cached_snapshot = _CachedSnapshot(
            snapshot=snapshot, expires_at=now + self._cache_ttl_seconds
        )
        return snapshot

    @staticmethod
    def _resolve_hf_cache_dir() -> Path:
        base = os.environ.get("HF_HOME", "~/.cache/huggingface")
        return expand_path(base)

    def _sum_paths(self, paths: Iterable[Path]) -> int:
        total = 0
        seen: set[str] = set()
        for path in paths:
            resolved = expand_path(path)
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            total += self._path_size(resolved)
        return total

    def _path_size(self, path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        if not path.is_dir():
            return 0
        return self._directory_size(path)

    def _directory_size(self, path: Path) -> int:
        total = 0
        try:
            # os.scandir avoids building full file lists and is faster for large trees.
            with os.scandir(path) as it:
                for entry in it:
                    if entry.name.startswith("."):
                        continue
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            total += self._directory_size(Path(entry.path))
                    except OSError:
                        continue
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return 0
        return total

    def _clear_directory_contents(self, path: Path) -> None:
        if not path.exists() or not path.is_dir():
            return
        for entry in path.iterdir():
            try:
                if entry.is_symlink() or entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
            except FileNotFoundError:
                continue
