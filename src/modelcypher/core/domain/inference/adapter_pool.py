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

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Protocol

# Setup Logging
logger = logging.getLogger("modelcypher.adapter_pool")


@dataclass
class MemoryStats:
    available_bytes: int
    total_bytes: int

    @property
    def available_ratio(self) -> float:
        return self.available_bytes / self.total_bytes if self.total_bytes else 0.0


class MemoryManaging(Protocol):
    async def memory_stats(self) -> MemoryStats: ...


class SystemMemoryManager(MemoryManaging):
    """Real memory manager that reads actual system memory stats."""

    async def memory_stats(self) -> MemoryStats:
        """Get real system memory statistics."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            total = mem.total
            available = mem.available
        except ImportError:
            # Fallback to platform-specific methods
            total, available = self._get_memory_fallback()

        if total <= 0 or available <= 0:
            raise RuntimeError(
                "Unable to measure system memory. Install psutil or provide a MemoryManaging "
                "implementation with explicit measurements."
            )

        return MemoryStats(available, total)

    def _get_memory_fallback(self) -> tuple[int, int]:
        """Platform-specific memory detection without psutil."""
        import platform
        import subprocess

        system = platform.system()

        if system == "Darwin":  # macOS
            try:
                # Use vm_stat for macOS
                result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return self._parse_macos_vm_stat(result.stdout)
            except Exception:
                pass

            # Fallback: sysctl for total memory
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    total = int(result.stdout.strip())
                    return total, 0
            except Exception:
                pass

        elif system == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                return self._parse_linux_meminfo(meminfo)
            except Exception:
                pass

        # Ultimate fallback: unknown
        return 0, 0

    def _parse_macos_vm_stat(self, output: str) -> tuple[int, int]:
        """Parse macOS vm_stat output."""
        import subprocess

        # Get page size
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.pagesize"], capture_output=True, text=True, timeout=5
            )
            page_size = int(result.stdout.strip()) if result.returncode == 0 else 4096
        except Exception:
            page_size = 4096

        # Get total memory
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            total = int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception:
            total = 0

        # Parse vm_stat for free + inactive pages
        free_pages = 0
        inactive_pages = 0
        for line in output.split("\n"):
            if "Pages free:" in line:
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages inactive:" in line:
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))

        available = (free_pages + inactive_pages) * page_size
        return total, available

    def _parse_linux_meminfo(self, meminfo: str) -> tuple[int, int]:
        """Parse Linux /proc/meminfo."""
        total = 0
        available = 0

        for line in meminfo.split("\n"):
            if line.startswith("MemTotal:"):
                # Value is in kB
                total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024

        return total, available


class AdapterPreloadPriority(Enum):
    NORMAL = 0
    HIGH = 1
    CRITICAL = 2

    # Allow comparison
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclass
class AdapterPoolEntry:
    id: uuid.UUID
    path: str
    priority: AdapterPreloadPriority
    estimated_memory_bytes: int
    last_accessed_at: float = field(default_factory=time.time)


@dataclass
class AdapterSwapResult:
    previous_adapter_id: uuid.UUID | None
    new_adapter_id: uuid.UUID | None
    swap_duration_ms: float
    was_cache_hit: bool


from modelcypher.core.domain.inference.types import AdapterPoolError


class MLXAdapterPool:
    """
    Multi-LoRA hot-swap pool for instant adapter switching.
    Ported from MLXAdapterPool.swift.
    """

    def __init__(
        self,
        memory_manager: MemoryManaging | None = None,
    ):
        if memory_manager is None:
            memory_manager = SystemMemoryManager()
        self.memory_manager = memory_manager

        # State (protected by lock in async methods)
        self.pool: dict[uuid.UUID, AdapterPoolEntry] = {}
        self.usage_order: list[uuid.UUID] = []
        self.current_active_id: uuid.UUID | None = None

        self.current_model_id: str | None = None
        self.registered_models: dict[str, dict] = {}  # Dict of callbacks

        self._lock = asyncio.Lock()

    async def register_model(
        self,
        model_id: str,
        load_adapter: Callable[[str], Awaitable[None]],
        unload_adapter: Callable[[], Awaitable[None]],
    ):
        async with self._lock:
            self.registered_models[model_id] = {"load": load_adapter, "unload": unload_adapter}
            logger.debug(f"Registered model context: {model_id}")

    async def unregister_model(self, model_id: str):
        async with self._lock:
            if self.current_model_id == model_id:
                self.current_active_id = None
                self.current_model_id = None
            if model_id in self.registered_models:
                del self.registered_models[model_id]
                logger.debug(f"Unregistered model context: {model_id}")

    async def preload(self, adapter_id: uuid.UUID, path: str, priority: AdapterPreloadPriority):
        async with self._lock:
            if adapter_id in self.pool:
                # Update existing
                entry = self.pool[adapter_id]
                entry.priority = priority
                entry.last_accessed_at = time.time()
                self._touch_lru(adapter_id)
                logger.debug(f"Adapter {adapter_id} updated priority to {priority}")
                return

            # Estimate memory
            mem_bytes = self._estimate_adapter_memory(path)
            await self._ensure_capacity(mem_bytes, priority)

            entry = AdapterPoolEntry(
                id=adapter_id, path=path, priority=priority, estimated_memory_bytes=mem_bytes
            )
            self.pool[adapter_id] = entry
            self.usage_order.append(adapter_id)

            logger.info(f"Preloaded adapter {adapter_id} from {os.path.basename(path)}")

    async def evict(self, adapter_id: uuid.UUID):
        # Internal helper, assumes lock held or called from locked context?
        # Actually evict is public in Swift. Let's lock.
        # But if called from _ensure_capacity (which locks), we need re-entrant lock or separation.
        # Python asyncio.Lock is NOT re-entrant.
        # We will separate public/private methods.
        async with self._lock:
            await self._evict_impl(adapter_id)

    async def _evict_impl(self, adapter_id: uuid.UUID):
        if adapter_id not in self.pool:
            return

        self.pool.pop(adapter_id)
        if adapter_id in self.usage_order:
            self.usage_order.remove(adapter_id)

        # If active, unload
        if self.current_active_id == adapter_id:
            if self.current_model_id and self.current_model_id in self.registered_models:
                handlers = self.registered_models[self.current_model_id]
                await handlers["unload"]()
            self.current_active_id = None

        logger.debug(f"Evicted adapter {adapter_id}")

    async def swap(self, to_adapter_id: uuid.UUID | None, model_id: str) -> AdapterSwapResult:
        async with self._lock:
            swap_start = time.time()
            previous_id = self.current_active_id

            if model_id not in self.registered_models:
                raise AdapterPoolError(f"Model {model_id} not registered")

            handlers = self.registered_models[model_id]
            self.current_model_id = model_id

            # Case 1: Return to base
            if to_adapter_id is None:
                if previous_id is not None:
                    await handlers["unload"]()
                self.current_active_id = None

                duration = (time.time() - swap_start) * 1000
                return AdapterSwapResult(previous_id, None, duration, True)

            # Case 2: Swap to pooled
            target_id = to_adapter_id
            if target_id not in self.pool:
                raise AdapterPoolError(f"Adapter {target_id} not in pool")

            entry = self.pool[target_id]

            if previous_id != target_id and previous_id is not None:
                await handlers["unload"]()

            # Load target
            try:
                await handlers["load"](entry.path)
            except Exception as e:
                raise AdapterPoolError(f"Load failed: {e}")

            self.current_active_id = target_id
            self._touch_lru(target_id)
            entry.last_accessed_at = time.time()

            duration = (time.time() - swap_start) * 1000
            logger.info(f"Swapped to {target_id} in {duration:.1f}ms")

            return AdapterSwapResult(previous_id, target_id, duration, True)

    def _touch_lru(self, uid: uuid.UUID):
        if uid in self.usage_order:
            self.usage_order.remove(uid)
        self.usage_order.append(uid)

    async def _ensure_capacity(self, required_bytes: int, priority: AdapterPreloadPriority):
        stats = await self.memory_manager.memory_stats()
        available_bytes = stats.available_bytes
        if required_bytes > available_bytes:
            raise AdapterPoolError(
                f"Adapter requires {required_bytes} bytes but only {available_bytes} bytes available"
            )

        while self._current_pool_bytes() + required_bytes > available_bytes:
            victim = self._select_eviction_victim(sparing=self.current_active_id, priority=priority)
            if not victim:
                raise AdapterPoolError(
                    f"Pool uses {self._current_pool_bytes()} bytes with only "
                    f"{available_bytes} bytes available"
                )
            await self._evict_impl(victim)

    def _select_eviction_victim(
        self, sparing: uuid.UUID | None, priority: AdapterPreloadPriority
    ) -> uuid.UUID | None:
        candidates = [uid for uid in self.usage_order if uid != sparing]

        # 1. Lower priority
        for uid in candidates:
            if self.pool[uid].priority < priority:
                return uid

        # 2. LRU
        if candidates:
            return candidates[0]

        return None

    def _current_pool_bytes(self) -> int:
        return sum(entry.estimated_memory_bytes for entry in self.pool.values())

    def _estimate_adapter_memory(self, path: str) -> int:
        # Simple recursive size
        total = 0
        try:
            if os.path.isfile(path):
                total = os.path.getsize(path)
            else:
                for root, _, files in os.walk(path):
                    for f in files:
                        fp = os.path.join(root, f)
                        total += os.path.getsize(fp)
        except Exception:
            total = 0
        if total <= 0:
            raise AdapterPoolError(f"Adapter size could not be measured for {path}")
        return total
