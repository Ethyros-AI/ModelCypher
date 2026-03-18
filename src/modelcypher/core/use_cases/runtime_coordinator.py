from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from modelcypher.core.domain.runtime_status import (
    RuntimeMemoryStatus,
    RuntimeOwner,
    RuntimeStatus,
)
from modelcypher.utils.locks import FileLock, FileLockError
from modelcypher.utils.paths import get_modelcypher_home


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuntimeBusyError(RuntimeError):
    """Raised when another GPU-heavy workflow already owns the runtime."""

    def __init__(self, status: RuntimeStatus | None = None):
        self.status = status
        if status is None:
            message = "Another GPU-heavy workflow is active."
        else:
            message = (
                f"{status.owner} is active"
                f" (phase={status.phase}, job_id={status.job_id})."
            )
        super().__init__(message)


@dataclass(frozen=True)
class RuntimeClaim:
    owner: RuntimeOwner
    job_id: str
    phase: str
    details: dict[str, Any]


class RuntimeCoordinator:
    """Cross-process runtime ownership and status publication."""

    def __init__(self, base_path: Path | None = None) -> None:
        runtime_dir = (base_path or get_modelcypher_home()) / "runtime"
        self._lock = FileLock(runtime_dir / "workload.lock")
        self._state_path = runtime_dir / "status.json"
        self._claim: RuntimeClaim | None = None
        self._started_at: str | None = None

    def status(self) -> RuntimeStatus | None:
        if self._claim is None and not self._lock.is_locked():
            return None
        return self._read_state()

    def claim(
        self,
        *,
        owner: RuntimeOwner,
        job_id: str,
        phase: str,
        details: dict[str, Any] | None = None,
    ) -> RuntimeStatus:
        try:
            self._lock.acquire()
        except FileLockError as exc:
            raise RuntimeBusyError(self.status()) from exc

        now = _utc_now()
        self._claim = RuntimeClaim(
            owner=owner,
            job_id=job_id,
            phase=phase,
            details=dict(details or {}),
        )
        self._started_at = now
        return self.update(phase=phase)

    def update(
        self,
        *,
        phase: str | None = None,
        eta_seconds: float | None = None,
        throughput_tokens_per_second: float | None = None,
        memory: RuntimeMemoryStatus | None = None,
        details: dict[str, Any] | None = None,
    ) -> RuntimeStatus:
        if self._claim is None or self._started_at is None:
            raise RuntimeError("RuntimeCoordinator.update() requires an active claim")

        next_details = dict(self._claim.details)
        if details:
            next_details.update(details)
        next_claim = RuntimeClaim(
            owner=self._claim.owner,
            job_id=self._claim.job_id,
            phase=phase or self._claim.phase,
            details=next_details,
        )
        self._claim = next_claim

        status = RuntimeStatus(
            owner=next_claim.owner.value,
            job_id=next_claim.job_id,
            phase=next_claim.phase,
            started_at=self._started_at,
            updated_at=_utc_now(),
            eta_seconds=eta_seconds,
            throughput_tokens_per_second=throughput_tokens_per_second,
            memory=memory,
            details=next_details,
        )
        self._write_state(status)
        return status

    def release(self) -> None:
        self._claim = None
        self._started_at = None
        try:
            if self._state_path.exists():
                self._state_path.unlink()
        finally:
            self._lock.release()

    @contextmanager
    def session(
        self,
        *,
        owner: RuntimeOwner,
        job_id: str,
        phase: str,
        details: dict[str, Any] | None = None,
    ) -> Iterator["RuntimeCoordinator"]:
        self.claim(owner=owner, job_id=job_id, phase=phase, details=details)
        try:
            yield self
        finally:
            self.release()

    def _read_state(self) -> RuntimeStatus | None:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        pid = payload.get("details", {}).get("pid")
        if isinstance(pid, int) and not self._pid_exists(pid) and self._claim is None:
            return None
        return RuntimeStatus.from_dict(payload)

    def _write_state(self, status: RuntimeStatus) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(status.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self._state_path)

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
