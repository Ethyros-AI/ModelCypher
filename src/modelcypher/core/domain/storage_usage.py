from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiskStats:
    total_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class StorageUsage:
    total_gb: float
    models_gb: float
    checkpoints_gb: float
    other_gb: float


@dataclass(frozen=True)
class StorageSnapshot:
    usage: StorageUsage
    disk: DiskStats
