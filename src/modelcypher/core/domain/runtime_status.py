from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimeOwner(str, Enum):
    """Exclusive runtime owners for GPU-heavy workflows."""

    TRAINING = "training"
    INFERENCE = "inference"
    EXPORT = "export"


@dataclass(frozen=True)
class RuntimeMemoryStatus:
    """GPU memory snapshot for a runtime owner."""

    active_gpu_memory_gb: float | None = None
    peak_gpu_memory_gb: float | None = None

    def to_dict(self) -> dict[str, float]:
        payload: dict[str, float] = {}
        if self.active_gpu_memory_gb is not None:
            payload["active_gpu_memory_gb"] = self.active_gpu_memory_gb
        if self.peak_gpu_memory_gb is not None:
            payload["peak_gpu_memory_gb"] = self.peak_gpu_memory_gb
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RuntimeMemoryStatus | None":
        if not payload:
            return None
        return cls(
            active_gpu_memory_gb=(
                float(payload["active_gpu_memory_gb"])
                if payload.get("active_gpu_memory_gb") is not None
                else None
            ),
            peak_gpu_memory_gb=(
                float(payload["peak_gpu_memory_gb"])
                if payload.get("peak_gpu_memory_gb") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class RuntimeStatus:
    """Cross-process runtime state for train/infer/export ownership."""

    owner: str
    job_id: str
    phase: str
    started_at: str
    updated_at: str
    eta_seconds: float | None = None
    throughput_tokens_per_second: float | None = None
    memory: RuntimeMemoryStatus | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "owner": self.owner,
            "job_id": self.job_id,
            "phase": self.phase,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }
        if self.eta_seconds is not None:
            payload["eta_seconds"] = self.eta_seconds
        if self.throughput_tokens_per_second is not None:
            payload["throughput_tokens_per_second"] = self.throughput_tokens_per_second
        if self.memory is not None:
            payload["memory"] = self.memory.to_dict()
        if self.details:
            payload["details"] = dict(self.details)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RuntimeStatus | None":
        if not payload:
            return None
        return cls(
            owner=str(payload["owner"]),
            job_id=str(payload["job_id"]),
            phase=str(payload["phase"]),
            started_at=str(payload["started_at"]),
            updated_at=str(payload["updated_at"]),
            eta_seconds=(
                float(payload["eta_seconds"])
                if payload.get("eta_seconds") is not None
                else None
            ),
            throughput_tokens_per_second=(
                float(payload["throughput_tokens_per_second"])
                if payload.get("throughput_tokens_per_second") is not None
                else None
            ),
            memory=RuntimeMemoryStatus.from_dict(payload.get("memory")),
            details=dict(payload.get("details") or {}),
        )
