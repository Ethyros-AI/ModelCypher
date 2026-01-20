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

"""Tests for idle training scheduler (thermal/memory job management)."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from modelcypher.core.domain.training.idle_training_scheduler import (
    IdleTrainingScheduler,
    JobFilter,
    JobID,
    JobStatus,
    JobSummary,
    ManagedJob,
    MemoryManager,
    MemoryPressure,
    MemoryStats,
    PauseReason,
    ProcessInfoThermalProvider,
    SchedulerPolicy,
    SchedulerStatus,
)


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_running_value(self):
        assert JobStatus.RUNNING.value == "running"

    def test_paused_value(self):
        assert JobStatus.PAUSED.value == "paused"

    def test_completed_value(self):
        assert JobStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert JobStatus.FAILED.value == "failed"


class TestMemoryPressure:
    """Tests for MemoryPressure enum."""

    def test_normal_value(self):
        assert MemoryPressure.NORMAL.value == "normal"

    def test_warning_value(self):
        assert MemoryPressure.WARNING.value == "warning"

    def test_critical_value(self):
        assert MemoryPressure.CRITICAL.value == "critical"


class TestPauseReason:
    """Tests for PauseReason enum."""

    def test_thermal_value(self):
        assert PauseReason.THERMAL.value == "thermal"

    def test_memory_value(self):
        assert PauseReason.MEMORY.value == "memory"


class TestJobSummary:
    """Tests for JobSummary dataclass."""

    def test_required_fields(self):
        summary = JobSummary(id=JobID("job-123"), status=JobStatus.RUNNING)
        assert summary.id == "job-123"
        assert summary.status == JobStatus.RUNNING


class TestJobFilter:
    """Tests for JobFilter dataclass."""

    def test_default_status_none(self):
        filter = JobFilter()
        assert filter.status is None

    def test_with_status(self):
        filter = JobFilter(status=JobStatus.PAUSED)
        assert filter.status == JobStatus.PAUSED


class TestSchedulerPolicy:
    """Tests for SchedulerPolicy dataclass."""

    def test_defaults(self):
        policy = SchedulerPolicy()
        assert policy.enabled is False
        assert policy.min_idle_seconds == 60.0
        assert policy.max_thermal_state_raw == 1
        assert policy.evaluation_interval == 30.0
        assert policy.cooldown_duration == 120.0

    def test_custom_values(self):
        policy = SchedulerPolicy(
            enabled=True,
            min_idle_seconds=120.0,
            max_thermal_state_raw=2,
            evaluation_interval=60.0,
            cooldown_duration=300.0,
        )
        assert policy.enabled is True
        assert policy.min_idle_seconds == 120.0


class TestManagedJob:
    """Tests for ManagedJob dataclass."""

    def test_required_fields(self):
        managed = ManagedJob(reason=PauseReason.THERMAL, paused_at=1000.0)
        assert managed.reason == PauseReason.THERMAL
        assert managed.paused_at == 1000.0


class TestSchedulerStatus:
    """Tests for SchedulerStatus dataclass."""

    def test_required_fields(self):
        status = SchedulerStatus(
            thermal_raw=2,
            memory_critical=True,
            timestamp=1234567890.0,
        )
        assert status.thermal_raw == 2
        assert status.memory_critical is True
        assert status.timestamp == 1234567890.0


class TestProcessInfoThermalProvider:
    """Tests for ProcessInfoThermalProvider class."""

    def test_returns_nominal_state(self):
        provider = ProcessInfoThermalProvider()
        state = provider.current_thermal_state_raw()
        # Default implementation returns 0 (nominal)
        assert state == 0


class TestMemoryManager:
    """Tests for MemoryManager class."""

    @pytest.mark.asyncio
    async def test_shared_returns_same_instance(self):
        manager1 = MemoryManager.shared()
        manager2 = MemoryManager.shared()
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_memory_stats_returns_normal(self):
        manager = MemoryManager.shared()
        stats = await manager.memory_stats()
        assert stats.pressure == MemoryPressure.NORMAL


class TestIdleTrainingSchedulerInit:
    """Tests for IdleTrainingScheduler initialization."""

    def test_default_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            scheduler = IdleTrainingScheduler(state_file_path=state_file)

            assert scheduler.policy.enabled is False
            assert scheduler.managed_jobs == {}
            assert scheduler.cooldown_start is None
            assert scheduler.training_service is None

    def test_configure_sets_training_service(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            scheduler = IdleTrainingScheduler(state_file_path=state_file)

            mock_service = mock.Mock()
            scheduler.configure(mock_service)

            assert scheduler.training_service is mock_service


class TestIdleTrainingSchedulerPolicy:
    """Tests for policy management."""

    @pytest.mark.asyncio
    async def test_set_policy_updates_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            scheduler = IdleTrainingScheduler(state_file_path=state_file)

            new_policy = SchedulerPolicy(enabled=True, min_idle_seconds=90.0)
            await scheduler.set_policy(new_policy)

            assert scheduler.policy.enabled is True
            assert scheduler.policy.min_idle_seconds == 90.0


class TestIdleTrainingSchedulerPersistence:
    """Tests for state persistence (load/save JSON)."""

    def test_saves_state_on_policy_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            scheduler = IdleTrainingScheduler(state_file_path=str(state_file))

            # Change policy
            scheduler.policy = SchedulerPolicy(enabled=True)
            scheduler._mark_state_dirty()
            scheduler._persist_state_if_needed()

            assert state_file.exists()
            with open(state_file) as f:
                data = json.load(f)
            assert data["policy"]["enabled"] is True

    def test_loads_state_on_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"

            # Pre-populate state file
            state_data = {
                "policy": {
                    "enabled": True,
                    "min_idle_seconds": 120.0,
                    "max_thermal_state_raw": 2,
                    "evaluation_interval": 60.0,
                    "cooldown_duration": 240.0,
                },
                "managed_jobs": {},
                "cooldown_start": None,
                "last_idle_transition": None,
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f)

            # Create scheduler - should load state
            scheduler = IdleTrainingScheduler(state_file_path=str(state_file))

            assert scheduler.policy.enabled is True
            assert scheduler.policy.min_idle_seconds == 120.0

    def test_loads_managed_jobs_from_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"

            # Pre-populate state file with managed jobs
            state_data = {
                "policy": {"enabled": False},
                "managed_jobs": {
                    "job-1": {"reason": "thermal", "paused_at": 1000.0},
                    "job-2": {"reason": "memory", "paused_at": 2000.0},
                },
                "cooldown_start": 3000.0,
                "last_idle_transition": None,
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f)

            scheduler = IdleTrainingScheduler(state_file_path=str(state_file))

            assert len(scheduler.managed_jobs) == 2
            assert "job-1" in scheduler.managed_jobs
            assert scheduler.managed_jobs["job-1"].reason == PauseReason.THERMAL
            assert scheduler.cooldown_start == 3000.0


class TestIdleTrainingSchedulerThermalProvider:
    """Tests for thermal provider integration."""

    def test_set_thermal_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            scheduler = IdleTrainingScheduler(state_file_path=state_file)

            mock_provider = mock.Mock()
            mock_provider.current_thermal_state_raw.return_value = 3
            scheduler.set_thermal_provider(mock_provider)

            assert scheduler.thermal_provider is mock_provider


class TestIdleTrainingSchedulerMemory:
    """Tests for memory manager integration."""

    def test_set_memory_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            scheduler = IdleTrainingScheduler(state_file_path=state_file)

            mock_manager = mock.Mock()
            scheduler.set_memory_manager(mock_manager)

            assert scheduler.memory_manager is mock_manager

    @pytest.mark.asyncio
    async def test_memory_pressure_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            scheduler = IdleTrainingScheduler(state_file_path=state_file)

            mock_manager = mock.AsyncMock()
            mock_manager.memory_stats.return_value = MemoryStats(MemoryPressure.CRITICAL)
            scheduler.set_memory_manager(mock_manager)

            # First call - should query manager
            is_critical = await scheduler._is_memory_pressure_critical()
            assert is_critical is True
            assert mock_manager.memory_stats.call_count == 1

            # Second call within cache duration - should use cached value
            is_critical = await scheduler._is_memory_pressure_critical()
            assert is_critical is True
            # Should still be 1 call due to caching
            assert mock_manager.memory_stats.call_count == 1


class TestIdleTrainingSchedulerEvaluate:
    """Tests for evaluate() method with mocked training service."""

    @pytest.mark.asyncio
    async def test_evaluate_without_training_service_no_op(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            scheduler = IdleTrainingScheduler(state_file_path=state_file)

            # Don't configure training service
            await scheduler.evaluate()  # Should not raise

    @pytest.mark.asyncio
    async def test_evaluate_pauses_on_high_thermal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")

            # Mock thermal provider to return high thermal state
            mock_thermal = mock.Mock()
            mock_thermal.current_thermal_state_raw.return_value = 3  # > max (1)

            scheduler = IdleTrainingScheduler(
                thermal_provider=mock_thermal,
                state_file_path=state_file,
            )

            # Configure training service with running job
            mock_service = mock.AsyncMock()
            mock_service.list_jobs.side_effect = [
                [JobSummary(id=JobID("job-1"), status=JobStatus.RUNNING)],
                [],  # paused jobs
            ]
            scheduler.configure(mock_service)

            # Enable policy
            scheduler.policy = SchedulerPolicy(enabled=True, max_thermal_state_raw=1)

            await scheduler.evaluate()

            # Should have called pause_job
            mock_service.pause_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_pauses_on_memory_critical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")

            # Mock memory manager to return critical
            mock_memory = mock.AsyncMock()
            mock_memory.memory_stats.return_value = MemoryStats(MemoryPressure.CRITICAL)

            scheduler = IdleTrainingScheduler(state_file_path=state_file)
            scheduler.set_memory_manager(mock_memory)

            # Configure training service
            mock_service = mock.AsyncMock()
            mock_service.list_jobs.side_effect = [
                [JobSummary(id=JobID("job-1"), status=JobStatus.RUNNING)],
                [],  # paused jobs
            ]
            scheduler.configure(mock_service)

            # Enable policy
            scheduler.policy = SchedulerPolicy(enabled=True)

            await scheduler.evaluate()

            # Should have called pause_job
            mock_service.pause_job.assert_called_once()


class TestIdleTrainingSchedulerMonitoring:
    """Tests for start/stop monitoring."""

    @pytest.mark.asyncio
    async def test_stop_monitoring_cancels_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            scheduler = IdleTrainingScheduler(state_file_path=state_file)

            # Start monitoring
            scheduler.start_monitoring()
            assert scheduler.monitor_task is not None

            # Stop monitoring
            await scheduler.stop_monitoring()
            assert scheduler.monitor_task is None

    @pytest.mark.asyncio
    async def test_start_monitoring_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            scheduler = IdleTrainingScheduler(state_file_path=state_file)

            scheduler.start_monitoring()
            task1 = scheduler.monitor_task

            scheduler.start_monitoring()  # Call again
            task2 = scheduler.monitor_task

            # Should be same task
            assert task1 is task2

            # Clean up
            task1.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task1
