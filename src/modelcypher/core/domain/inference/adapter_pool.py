
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable, Protocol
import asyncio
import time
import uuid
import os
from enum import Enum
import logging

# Setup Logging
logger = logging.getLogger("modelcypher.adapter_pool")

class MemoryPressure(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class MemoryStats:
    pressure: MemoryPressure
    available_bytes: int
    total_bytes: int

class MemoryManaging(Protocol):
    async def memory_stats(self) -> MemoryStats: ...

class MockMemoryManager(MemoryManaging):
    async def memory_stats(self) -> MemoryStats:
        # Mock implementation always returning NORMAL
        return MemoryStats(MemoryPressure.NORMAL, 16 * 1024**3, 32 * 1024**3)

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
    previous_adapter_id: Optional[uuid.UUID]
    new_adapter_id: Optional[uuid.UUID]
    swap_duration_ms: float
    was_cache_hit: bool

@dataclass
class AdapterPoolConfiguration:
    max_pooled_normal: int = 4
    max_pooled_warning: int = 2
    max_pooled_critical: int = 1
    target_swap_ms: float = 100.0

from modelcypher.core.domain.inference.types import AdapterPoolError


class MLXAdapterPool:
    """
    Multi-LoRA hot-swap pool for instant adapter switching.
    Ported from MLXAdapterPool.swift.
    """
    
    def __init__(
        self, 
        config: AdapterPoolConfiguration = AdapterPoolConfiguration(), 
        memory_manager: MemoryManaging = MockMemoryManager()
    ):
        self.config = config
        self.memory_manager = memory_manager
        
        # State (protected by lock in async methods)
        self.pool: Dict[uuid.UUID, AdapterPoolEntry] = {}
        self.usage_order: List[uuid.UUID] = []
        self.current_active_id: Optional[uuid.UUID] = None
        
        self.current_model_id: Optional[str] = None
        self.registered_models: Dict[str, dict] = {} # Dict of callbacks
        
        self._lock = asyncio.Lock()
        
    async def register_model(
        self, 
        model_id: str,
        load_adapter: Callable[[str], Awaitable[None]],
        unload_adapter: Callable[[], Awaitable[None]]
    ):
        async with self._lock:
            self.registered_models[model_id] = {
                "load": load_adapter,
                "unload": unload_adapter
            }
            logger.debug(f"Registered model context: {model_id}")

    async def unregister_model(self, model_id: str):
        async with self._lock:
            if self.current_model_id == model_id:
                self.current_active_id = None
                self.current_model_id = None
            if model_id in self.registered_models:
                del self.registered_models[model_id]
                logger.debug(f"Unregistered model context: {model_id}")

    async def preload(
        self,
        adapter_id: uuid.UUID,
        path: str,
        priority: AdapterPreloadPriority
    ):
        async with self._lock:
            if adapter_id in self.pool:
                # Update existing
                entry = self.pool[adapter_id]
                entry.priority = priority
                entry.last_accessed_at = time.time()
                self._touch_lru(adapter_id)
                logger.debug(f"Adapter {adapter_id} updated priority to {priority}")
                return

            await self._ensure_capacity(priority)
            
            # Estimate memory
            mem_bytes = self._estimate_adapter_memory(path)
            
            entry = AdapterPoolEntry(
                id=adapter_id,
                path=path,
                priority=priority,
                estimated_memory_bytes=mem_bytes
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
            
        entry = self.pool.pop(adapter_id)
        if adapter_id in self.usage_order:
            self.usage_order.remove(adapter_id)
            
        # If active, unload
        if self.current_active_id == adapter_id:
            if self.current_model_id and self.current_model_id in self.registered_models:
                handlers = self.registered_models[self.current_model_id]
                await handlers["unload"]()
            self.current_active_id = None
            
        logger.debug(f"Evicted adapter {adapter_id}")

    async def swap(
        self,
        to_adapter_id: Optional[uuid.UUID],
        model_id: str
    ) -> AdapterSwapResult:
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

    async def _ensure_capacity(self, priority: AdapterPreloadPriority):
        stats = await self.memory_manager.memory_stats()
        max_cap = self._max_capacity_for_pressure(stats.pressure)
        
        while len(self.pool) >= max_cap:
            victim = self._select_eviction_victim(sparing=self.current_active_id, priority=priority)
            if not victim:
                raise AdapterPoolError(f"Pool at capacity {max_cap} and no evictable victim found")
            await self._evict_impl(victim)

    def _max_capacity_for_pressure(self, pressure: MemoryPressure) -> int:
        if pressure == MemoryPressure.NORMAL: return self.config.max_pooled_normal
        if pressure == MemoryPressure.WARNING: return self.config.max_pooled_warning
        return self.config.max_pooled_critical

    def _select_eviction_victim(
        self, 
        sparing: Optional[uuid.UUID],
        priority: AdapterPreloadPriority
    ) -> Optional[uuid.UUID]:
        candidates = [uid for uid in self.usage_order if uid != sparing]
        
        # 1. Lower priority
        for uid in candidates:
            if self.pool[uid].priority < priority:
                return uid
                
        # 2. LRU
        if candidates:
            return candidates[0]
            
        return None

    def _estimate_adapter_memory(self, path: str) -> int:
        # Simple recursive size
        total = 0
        try:
            for root, dirs, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    total += os.path.getsize(fp)
        except:
            pass
        return total if total > 0 else 50_000_000
