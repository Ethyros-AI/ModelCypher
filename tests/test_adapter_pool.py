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

"""Tests for MLX adapter pool infrastructure."""

from __future__ import annotations

import uuid

import pytest

from modelcypher.core.domain.inference.adapter_pool import (
    AdapterPoolEntry,
    AdapterPreloadPriority,
    AdapterSwapResult,
    MemoryStats,
    MLXAdapterPool,
    SystemMemoryManager,
)
from modelcypher.core.domain.inference.types import AdapterPoolError


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_memory_stats_creation(self):
        """Test creating memory stats."""
        stats = MemoryStats(
            available_bytes=8_000_000_000,
            total_bytes=16_000_000_000,
        )

        assert stats.available_bytes == 8_000_000_000
        assert stats.total_bytes == 16_000_000_000
        assert stats.available_ratio == 0.5

    def test_memory_stats_immutable_fields(self):
        """Test that memory stats fields are accessible."""
        stats = MemoryStats(
            available_bytes=4_000_000_000,
            total_bytes=16_000_000_000,
        )

        assert stats.available_bytes == 4_000_000_000


class TestAdapterPreloadPriority:
    """Tests for AdapterPreloadPriority enum."""

    def test_priority_values(self):
        """Test priority enum numeric values."""
        assert AdapterPreloadPriority.NORMAL.value == 0
        assert AdapterPreloadPriority.HIGH.value == 1
        assert AdapterPreloadPriority.CRITICAL.value == 2

    def test_priority_comparison(self):
        """Test priority comparison operators."""
        assert AdapterPreloadPriority.NORMAL < AdapterPreloadPriority.HIGH
        assert AdapterPreloadPriority.HIGH < AdapterPreloadPriority.CRITICAL
        assert not (AdapterPreloadPriority.CRITICAL < AdapterPreloadPriority.NORMAL)

    def test_priority_not_less_than_self(self):
        """Test priority is not less than itself."""
        assert not (AdapterPreloadPriority.NORMAL < AdapterPreloadPriority.NORMAL)


class TestAdapterPoolEntry:
    """Tests for AdapterPoolEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a pool entry."""
        uid = uuid.uuid4()
        entry = AdapterPoolEntry(
            id=uid,
            path="/path/to/adapter",
            priority=AdapterPreloadPriority.HIGH,
            estimated_memory_bytes=100_000_000,
        )

        assert entry.id == uid
        assert entry.path == "/path/to/adapter"
        assert entry.priority == AdapterPreloadPriority.HIGH
        assert entry.estimated_memory_bytes == 100_000_000
        assert entry.last_accessed_at > 0  # Timestamp set by default

    def test_entry_last_accessed_default(self):
        """Test that last_accessed_at is set to current time."""
        import time

        before = time.time()
        entry = AdapterPoolEntry(
            id=uuid.uuid4(),
            path="/path",
            priority=AdapterPreloadPriority.NORMAL,
            estimated_memory_bytes=50_000_000,
        )
        after = time.time()

        assert before <= entry.last_accessed_at <= after


class TestAdapterSwapResult:
    """Tests for AdapterSwapResult dataclass."""

    def test_swap_result_creation(self):
        """Test creating a swap result."""
        prev_id = uuid.uuid4()
        new_id = uuid.uuid4()
        result = AdapterSwapResult(
            previous_adapter_id=prev_id,
            new_adapter_id=new_id,
            swap_duration_ms=50.5,
            was_cache_hit=True,
        )

        assert result.previous_adapter_id == prev_id
        assert result.new_adapter_id == new_id
        assert result.swap_duration_ms == 50.5
        assert result.was_cache_hit is True

    def test_swap_result_with_none_ids(self):
        """Test swap result with None adapter IDs."""
        result = AdapterSwapResult(
            previous_adapter_id=None,
            new_adapter_id=None,
            swap_duration_ms=10.0,
            was_cache_hit=True,
        )

        assert result.previous_adapter_id is None
        assert result.new_adapter_id is None


class TestSystemMemoryManager:
    """Tests for SystemMemoryManager."""

    @pytest.mark.asyncio
    async def test_memory_stats_returns_valid_structure(self):
        """Test that memory_stats returns valid MemoryStats."""
        manager = SystemMemoryManager()
        try:
            stats = await manager.memory_stats()
        except RuntimeError as exc:
            pytest.skip(str(exc))

        assert isinstance(stats, MemoryStats)
        # We can't assert exact values, but total should be non-negative
        assert stats.total_bytes >= 0
        if stats.total_bytes > 0:
            assert 0.0 <= stats.available_ratio <= 1.0

    def test_parse_linux_meminfo(self):
        """Test parsing Linux /proc/meminfo format."""
        manager = SystemMemoryManager()
        meminfo = """MemTotal:       16384000 kB
MemFree:         1234567 kB
MemAvailable:    8000000 kB
Buffers:          123456 kB
Cached:          4000000 kB
"""
        total, available = manager._parse_linux_meminfo(meminfo)

        assert total == 16384000 * 1024
        assert available == 8000000 * 1024

    def test_parse_linux_meminfo_missing_available(self):
        """Test parsing meminfo without MemAvailable."""
        manager = SystemMemoryManager()
        meminfo = """MemTotal:       16384000 kB
MemFree:         1234567 kB
"""
        total, available = manager._parse_linux_meminfo(meminfo)

        assert total == 16384000 * 1024
        assert available == 0  # Not found

class MockMemoryManager:
    """Mock memory manager for testing."""

    def __init__(
        self, available_bytes: int = 8_000_000_000, total_bytes: int = 16_000_000_000
    ):
        self.available_bytes = available_bytes
        self.total_bytes = total_bytes
        self.call_count = 0

    async def memory_stats(self) -> MemoryStats:
        self.call_count += 1
        return MemoryStats(
            available_bytes=self.available_bytes,
            total_bytes=self.total_bytes,
        )

    def set_available_bytes(self, available_bytes: int):
        self.available_bytes = available_bytes


class TestMLXAdapterPool:
    """Tests for MLXAdapterPool."""

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory manager."""
        return MockMemoryManager()

    @pytest.fixture
    def adapter_bytes(self):
        """Default adapter size for tests."""
        return 100

    @pytest.fixture
    def pool(self, mock_memory, adapter_bytes, monkeypatch):
        """Create adapter pool with mock memory."""
        mock_memory.set_available_bytes(adapter_bytes * 4)
        pool = MLXAdapterPool(memory_manager=mock_memory)
        monkeypatch.setattr(pool, "_estimate_adapter_memory", lambda path: adapter_bytes)
        return pool

    @pytest.mark.asyncio
    async def test_pool_initialization(self, pool):
        """Test pool starts empty."""
        assert len(pool.pool) == 0
        assert len(pool.usage_order) == 0
        assert pool.current_active_id is None

    @pytest.mark.asyncio
    async def test_preload_adds_adapter(self, pool):
        """Test preloading an adapter adds it to pool."""
        adapter_id = uuid.uuid4()
        await pool.preload(adapter_id, "/path/to/adapter", AdapterPreloadPriority.NORMAL)

        assert adapter_id in pool.pool
        assert adapter_id in pool.usage_order
        assert pool.pool[adapter_id].priority == AdapterPreloadPriority.NORMAL

    @pytest.mark.asyncio
    async def test_preload_updates_existing_priority(self, pool):
        """Test preloading existing adapter updates priority."""
        adapter_id = uuid.uuid4()
        await pool.preload(adapter_id, "/path", AdapterPreloadPriority.NORMAL)
        await pool.preload(adapter_id, "/path", AdapterPreloadPriority.HIGH)

        assert pool.pool[adapter_id].priority == AdapterPreloadPriority.HIGH
        assert len(pool.pool) == 1  # No duplicate

    @pytest.mark.asyncio
    async def test_evict_removes_adapter(self, pool):
        """Test evicting an adapter removes it from pool."""
        adapter_id = uuid.uuid4()
        await pool.preload(adapter_id, "/path", AdapterPreloadPriority.NORMAL)
        await pool.evict(adapter_id)

        assert adapter_id not in pool.pool
        assert adapter_id not in pool.usage_order

    @pytest.mark.asyncio
    async def test_evict_nonexistent_is_safe(self, pool):
        """Test evicting nonexistent adapter doesn't raise."""
        nonexistent = uuid.uuid4()
        await pool.evict(nonexistent)  # Should not raise

    @pytest.mark.asyncio
    async def test_lru_ordering(self, pool):
        """Test LRU ordering is maintained."""
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        id3 = uuid.uuid4()

        await pool.preload(id1, "/path1", AdapterPreloadPriority.NORMAL)
        await pool.preload(id2, "/path2", AdapterPreloadPriority.NORMAL)
        await pool.preload(id3, "/path3", AdapterPreloadPriority.NORMAL)

        # Order should be [id1, id2, id3]
        assert pool.usage_order == [id1, id2, id3]

        # Touch id1 (via preload update)
        await pool.preload(id1, "/path1", AdapterPreloadPriority.NORMAL)

        # Order should now be [id2, id3, id1]
        assert pool.usage_order == [id2, id3, id1]

    @pytest.mark.asyncio
    async def test_capacity_limit_available_bytes(self, pool, mock_memory, adapter_bytes):
        """Test capacity is limited by available bytes."""
        max_capacity = mock_memory.available_bytes // adapter_bytes

        for i in range(max_capacity):
            await pool.preload(uuid.uuid4(), f"/path{i}", AdapterPreloadPriority.NORMAL)

        assert len(pool.pool) == max_capacity

    @pytest.mark.asyncio
    async def test_capacity_evicts_lru_when_full(self, pool, mock_memory):
        """Test oldest adapter is evicted when capacity is reached."""
        max_capacity = mock_memory.available_bytes // pool._estimate_adapter_memory("/path")

        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        id3 = uuid.uuid4()
        id4 = uuid.uuid4()
        id5 = uuid.uuid4()

        # Fill to capacity
        preload_ids = [id1, id2, id3, id4]
        for uid in preload_ids[:max_capacity]:
            await pool.preload(uid, f"/path{uid}", AdapterPreloadPriority.NORMAL)

        # Add one more - should evict id1 (oldest)
        await pool.preload(id5, "/path5", AdapterPreloadPriority.NORMAL)

        assert id1 not in pool.pool
        assert id5 in pool.pool
        assert len(pool.pool) == max_capacity

    @pytest.mark.asyncio
    async def test_high_priority_evicts_lower(self, pool, mock_memory):
        """Test high priority adapter evicts lower priority first."""
        max_capacity = mock_memory.available_bytes // pool._estimate_adapter_memory("/path")

        low_id = uuid.uuid4()
        high_id = uuid.uuid4()

        # Fill with low priority
        for i in range(max_capacity):
            await pool.preload(uuid.uuid4(), f"/path{i}", AdapterPreloadPriority.NORMAL)

        # Preload one at start with NORMAL priority
        await pool.evict(pool.usage_order[0])  # Make room
        await pool.preload(low_id, "/low", AdapterPreloadPriority.NORMAL)

        # Add HIGH priority - should evict a NORMAL one first
        await pool.preload(high_id, "/high", AdapterPreloadPriority.HIGH)

        assert high_id in pool.pool

    @pytest.mark.asyncio
    async def test_register_and_unregister_model(self, pool):
        """Test model registration."""
        calls = []

        async def mock_load(path):
            calls.append(("load", path))

        async def mock_unload():
            calls.append(("unload",))

        await pool.register_model("model1", mock_load, mock_unload)
        assert "model1" in pool.registered_models

        await pool.unregister_model("model1")
        assert "model1" not in pool.registered_models

    @pytest.mark.asyncio
    async def test_swap_to_adapter(self, pool):
        """Test swapping to a pooled adapter."""
        load_calls = []
        unload_calls = []

        async def mock_load(path):
            load_calls.append(path)

        async def mock_unload():
            unload_calls.append(True)

        await pool.register_model("model1", mock_load, mock_unload)

        adapter_id = uuid.uuid4()
        await pool.preload(adapter_id, "/adapter/path", AdapterPreloadPriority.NORMAL)

        result = await pool.swap(adapter_id, "model1")

        assert result.new_adapter_id == adapter_id
        assert result.was_cache_hit is True
        assert result.swap_duration_ms > 0
        assert len(load_calls) == 1
        assert load_calls[0] == "/adapter/path"
        assert pool.current_active_id == adapter_id

    @pytest.mark.asyncio
    async def test_swap_to_none_unloads(self, pool):
        """Test swapping to None unloads current adapter."""
        unload_calls = []

        async def mock_load(path):
            return None

        async def mock_unload():
            unload_calls.append(True)

        await pool.register_model("model1", mock_load, mock_unload)

        adapter_id = uuid.uuid4()
        await pool.preload(adapter_id, "/path", AdapterPreloadPriority.NORMAL)
        await pool.swap(adapter_id, "model1")

        # Now swap to None
        result = await pool.swap(None, "model1")

        assert result.previous_adapter_id == adapter_id
        assert result.new_adapter_id is None
        assert len(unload_calls) == 1
        assert pool.current_active_id is None

    @pytest.mark.asyncio
    async def test_swap_unregistered_model_raises(self, pool):
        """Test swapping with unregistered model raises error."""
        adapter_id = uuid.uuid4()
        await pool.preload(adapter_id, "/path", AdapterPreloadPriority.NORMAL)

        with pytest.raises(AdapterPoolError, match="not registered"):
            await pool.swap(adapter_id, "unregistered_model")

    @pytest.mark.asyncio
    async def test_swap_unpooled_adapter_raises(self, pool):
        """Test swapping to unpooled adapter raises error."""
        async def mock_load(path):
            return None

        async def mock_unload():
            return None

        await pool.register_model("model1", mock_load, mock_unload)

        unpooled_id = uuid.uuid4()
        with pytest.raises(AdapterPoolError, match="not in pool"):
            await pool.swap(unpooled_id, "model1")

    @pytest.mark.asyncio
    async def test_current_pool_bytes(self, pool, adapter_bytes):
        """Test current pool byte count."""
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        await pool.preload(id1, "/path1", AdapterPreloadPriority.NORMAL)
        await pool.preload(id2, "/path2", AdapterPreloadPriority.NORMAL)

        assert pool._current_pool_bytes() == adapter_bytes * 2
