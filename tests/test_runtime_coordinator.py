from __future__ import annotations

import os
from pathlib import Path

import pytest

from modelcypher.core.domain.runtime_status import RuntimeMemoryStatus, RuntimeOwner
from modelcypher.core.use_cases.runtime_coordinator import (
    RuntimeBusyError,
    RuntimeCoordinator,
)


def test_runtime_coordinator_blocks_competing_claims(tmp_path: Path) -> None:
    coordinator_a = RuntimeCoordinator(base_path=tmp_path)
    coordinator_b = RuntimeCoordinator(base_path=tmp_path)

    with coordinator_a.session(
        owner=RuntimeOwner.TRAINING,
        job_id="train-1",
        phase="train",
        details={"pid": os.getpid()},
    ):
        status = coordinator_b.status()
        assert status is not None
        assert status.owner == "training"
        assert status.phase == "train"
        with pytest.raises(RuntimeBusyError):
            coordinator_b.claim(
                owner=RuntimeOwner.EXPORT,
                job_id="export-1",
                phase="export",
            )

    assert coordinator_b.status() is None


def test_runtime_coordinator_updates_phase_and_memory(tmp_path: Path) -> None:
    coordinator = RuntimeCoordinator(base_path=tmp_path)
    with coordinator.session(
        owner=RuntimeOwner.EXPORT,
        job_id="export-1",
        phase="starting",
        details={"pid": os.getpid()},
    ):
        updated = coordinator.update(
            phase="quantize",
            eta_seconds=12.5,
            throughput_tokens_per_second=64.0,
            memory=RuntimeMemoryStatus(
                active_gpu_memory_gb=3.0,
                peak_gpu_memory_gb=5.5,
            ),
        )
        assert updated.phase == "quantize"
        assert updated.eta_seconds == 12.5
        assert updated.memory is not None
        assert updated.memory.peak_gpu_memory_gb == 5.5
