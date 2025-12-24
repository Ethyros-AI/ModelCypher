"""
Idle Training Scheduler.

Intelligent training scheduler with thermal and memory pressure management.
Automatically pauses/resumes training based on system conditions.

Ported 1:1 from the reference Swift implementation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol, Any
from uuid import UUID

logger = logging.getLogger("IdleTrainingScheduler")

# =============================================================================
# Protocols
# =============================================================================

class ThermalStateProviding(Protocol):
    def current_thermal_state_raw(self) -> int:
        ...

class ProcessInfoThermalProvider(ThermalStateProviding):
    def current_thermal_state_raw(self) -> int:
        # Python doesn't have standard lib access to macOS thermal state easily without objc
        # Returning 0 (nominal) for default implementation
        return 0

class JobID(str):
    pass

class JobStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class JobSummary:
    id: JobID
    status: JobStatus

@dataclass
class JobFilter:
    status: JobStatus | None = None

class TrainingService(Protocol):
    async def list_jobs(self, filter: JobFilter | None = None) -> list[JobSummary]: ...
    async def pause_job(self, job_id: JobID) -> None: ...
    async def resume_job(self, job_id: JobID) -> None: ...

class MemoryPressure(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class MemoryStats:
    pressure: MemoryPressure

class MemoryManaging(Protocol):
    async def memory_stats(self) -> MemoryStats: ...

class MemoryManager(MemoryManaging):
    _shared = None
    
    @classmethod
    def shared(cls) -> "MemoryManager":
        if not cls._shared:
            cls._shared = cls()
        return cls._shared

    async def memory_stats(self) -> MemoryStats:
        return MemoryStats(MemoryPressure.NORMAL)

# =============================================================================
# Idle Training Scheduler
# =============================================================================

@dataclass
class SchedulerPolicy:
    enabled: bool = False
    min_idle_seconds: float = 60.0
    max_thermal_state_raw: int = 1
    evaluation_interval: float = 30.0
    cooldown_duration: float = 120.0

class PauseReason(str, Enum):
    THERMAL = "thermal"
    MEMORY = "memory"

@dataclass
class ManagedJob:
    reason: PauseReason
    paused_at: float # timestamp

@dataclass
class SchedulerStatus:
    thermal_raw: int
    memory_critical: bool
    timestamp: float

@dataclass
class PersistedState:
    policy: SchedulerPolicy
    managed_jobs: dict[str, Any] # JobID -> ManagedJob dict
    cooldown_start: float | None
    last_idle_transition: float | None


class IdleTrainingScheduler:
    
    def __init__(
        self,
        thermal_provider: ThermalStateProviding = ProcessInfoThermalProvider(),
        state_file_path: str | None = None
    ):
        self.thermal_provider = thermal_provider
        self.state_file_path = state_file_path or "idle_scheduler_state.json"
        
        self.policy = SchedulerPolicy()
        self.managed_jobs: dict[JobID, ManagedJob] = {}
        self.cooldown_start: float | None = None
        self.last_idle_transition: float | None = None
        
        self.training_service: TrainingService | None = None
        self.memory_manager: MemoryManaging = MemoryManager.shared()
        self.monitor_task: asyncio.Task | None = None
        
        self.is_evaluating = False
        self.state_dirty = False
        self.cached_memory_pressure: tuple[bool, float] | None = None
        self.memory_cache_valid_duration = 10.0
        
        self._load_state()

    def configure(self, training_service: TrainingService):
        self.training_service = training_service

    def set_thermal_provider(self, provider: ThermalStateProviding):
        self.thermal_provider = provider

    def set_memory_manager(self, manager: MemoryManaging):
        self.memory_manager = manager

    async def set_policy(self, policy: SchedulerPolicy):
        self.policy = policy
        self._mark_state_dirty()
        await self.evaluate()

    # MARK: - Monitoring Control

    def start_monitoring(self):
        if self.monitor_task and not self.monitor_task.done():
            return
        self.monitor_task = asyncio.create_task(self._run_monitor_loop())

    async def stop_monitoring(self):
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
        self._persist_state_if_needed()

    async def _run_monitor_loop(self):
        logger.info("Starting idle scheduler monitor loop")
        try:
            while True:
                await self.evaluate()
                await asyncio.sleep(self.policy.evaluation_interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._persist_state_if_needed()

    # MARK: - Evaluation Logic

    async def evaluate(self):
        if self.is_evaluating:
            logger.debug("Skipping evaluate() re-entry")
            return
        
        self.is_evaluating = True
        try:
            if not self.training_service:
                logger.debug("Training service not configured")
                return

            now = datetime.now().timestamp()
            thermal_raw = max(0, min(3, self.thermal_provider.current_thermal_state_raw()))
            memory_critical = await self._is_memory_pressure_critical()

            try:
                running_jobs = await self.training_service.list_jobs(JobFilter(JobStatus.RUNNING))
                paused_jobs = await self.training_service.list_jobs(JobFilter(JobStatus.PAUSED))
            except Exception as e:
                logger.error(f"Failed to fetch jobs: {e}")
                self._set_cooldown_start(None)
                self._set_last_idle_transition(None)
                return

            self._cleanup_managed_jobs(paused_jobs=paused_jobs)
            
            if not self.managed_jobs:
                self._set_cooldown_start(None)

            if not running_jobs:
                if self.last_idle_transition is None:
                    self._set_last_idle_transition(now)
            else:
                self._set_last_idle_transition(None)

            if not self.policy.enabled:
                status = SchedulerStatus(thermal_raw, memory_critical, now)
                await self._resume_managed_jobs_if_possible(paused_jobs, running_jobs, status, force=True)
                return

            should_pause_thermal = thermal_raw > self.policy.max_thermal_state_raw
            should_pause_memory = memory_critical

            if should_pause_thermal or should_pause_memory:
                reason = PauseReason.THERMAL if should_pause_thermal else PauseReason.MEMORY
                self._set_cooldown_start(None)
                await self._pause_jobs_if_needed(running_jobs, reason, thermal_raw, now)
                return

            if self.cooldown_start is None and self.managed_jobs:
                self._set_cooldown_start(now)

            status = SchedulerStatus(thermal_raw, memory_critical, now)
            await self._resume_managed_jobs_if_possible(paused_jobs, running_jobs, status, force=False)
            
        finally:
            self.is_evaluating = False
            self._persist_state_if_needed()

    # MARK: - Job Management

    def _cleanup_managed_jobs(self, paused_jobs: list[JobSummary]):
        paused_ids = {job.id for job in paused_jobs}
        to_remove = []
        for job_id in self.managed_jobs:
            if job_id not in paused_ids:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.managed_jobs[job_id]
        
        if to_remove:
            self._mark_state_dirty()

    async def _pause_jobs_if_needed(
        self, 
        running_jobs: list[JobSummary], 
        reason: PauseReason, 
        thermal_raw: int, 
        now: float
    ):
        if not self.training_service: return
        
        for job in running_jobs:
            if job.id not in self.managed_jobs:
                try:
                    await self.training_service.pause_job(job.id)
                    self.managed_jobs[job.id] = ManagedJob(reason, now)
                    self._mark_state_dirty()
                    logger.info(f"Paused job {job.id} due to {reason.value} (thermal: {thermal_raw})")
                except Exception as e:
                    logger.error(f"Unable to pause job {job.id}: {e}")
        
        self._set_last_idle_transition(now)

    async def _resume_managed_jobs_if_possible(
        self,
        paused_jobs: list[JobSummary],
        running_jobs: list[JobSummary],
        status: SchedulerStatus,
        force: bool
    ):
        if not self.training_service: return
        if not self.managed_jobs: return

        paused_ids = {job.id for job in paused_jobs}
        managed_ids = list(self.managed_jobs.keys())
        
        # Check foreign running jobs (jobs running that we didn't pause)
        # In Swift: let foreignRunning = runningJobs.filter { self.managedJobs[$0.id] == nil }
        foreign_running = [j for j in running_jobs if j.id not in self.managed_jobs]

        if not force and foreign_running:
            return # Don't resume if user manually started other jobs

        if not force and (status.thermal_raw > self.policy.max_thermal_state_raw or status.memory_critical):
            return

        if not force and self.cooldown_start is not None:
            elapsed = status.timestamp - self.cooldown_start
            if elapsed < self.policy.cooldown_duration:
                return

        idle_duration = (status.timestamp - self.last_idle_transition) if self.last_idle_transition else 0.0
        resumed_jobs = []

        for job_id in managed_ids:
            if job_id not in paused_ids:
                resumed_jobs.append(job_id)
                continue

            managed = self.managed_jobs[job_id]
            
            if not force:
                if idle_duration < self.policy.min_idle_seconds: continue
                if (status.timestamp - managed.paused_at) < self.policy.min_idle_seconds: continue

            try:
                await self.training_service.resume_job(job_id)
                resumed_jobs.append(job_id)
                logger.info(f"Resumed job {job_id}")
            except Exception as e:
                logger.error(f"Failed to resume job {job_id}: {e}")

        for job_id in resumed_jobs:
            if job_id in self.managed_jobs:
                del self.managed_jobs[job_id]
        
        if resumed_jobs:
            self._mark_state_dirty()
        
        if not self.managed_jobs:
            self._set_cooldown_start(None)

    # MARK: - Memory Pressure

    async def _is_memory_pressure_critical(self) -> bool:
        now = datetime.now().timestamp()
        if self.cached_memory_pressure:
            is_crit, ts = self.cached_memory_pressure
            if (now - ts) < self.memory_cache_valid_duration:
                return is_crit
        
        stats = await self.memory_manager.memory_stats()
        is_critical = (stats.pressure == MemoryPressure.CRITICAL)
        self.cached_memory_pressure = (is_critical, now)
        return is_critical

    # MARK: - Persistence

    def _set_cooldown_start(self, timestamp: float | None):
        if self.cooldown_start != timestamp:
            self.cooldown_start = timestamp
            self._mark_state_dirty()

    def _set_last_idle_transition(self, timestamp: float | None):
        if self.last_idle_transition != timestamp:
            self.last_idle_transition = timestamp
            self._mark_state_dirty()

    def _mark_state_dirty(self):
        self.state_dirty = True

    def _persist_state_if_needed(self):
        if not self.state_dirty: return
        try:
            state = {
                "policy": asdict(self.policy),
                "managed_jobs": {str(k): asdict(v) for k, v in self.managed_jobs.items()},
                "cooldown_start": self.cooldown_start,
                "last_idle_transition": self.last_idle_transition
            }
            with open(self.state_file_path, 'w') as f:
                json.dump(state, f, indent=2)
            self.state_dirty = False
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    def _load_state(self):
        if not os.path.exists(self.state_file_path):
            return
        
        try:
            with open(self.state_file_path, 'r') as f:
                data = json.load(f)
            
            p_data = data.get("policy", {})
            self.policy = SchedulerPolicy(**p_data)
            
            mj_data = data.get("managed_jobs", {})
            self.managed_jobs = {}
            for k, v in mj_data.items():
                reason = PauseReason(v.get("reason", "thermal"))
                paused_at = v.get("paused_at", 0.0)
                self.managed_jobs[JobID(k)] = ManagedJob(reason, paused_at)
                
            self.cooldown_start = data.get("cooldown_start")
            self.last_idle_transition = data.get("last_idle_transition")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

