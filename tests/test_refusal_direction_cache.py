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

"""Tests for RefusalDirectionCache module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from modelcypher.core.domain.geometry.refusal_direction_cache import (
    RefusalDirectionCache,
)
from modelcypher.core.domain.geometry.refusal_direction_detector import (
    RefusalDirection,
)


@pytest.fixture
def cache(tmp_path):
    """Create cache with temp directory."""
    return RefusalDirectionCache(cache_directory=tmp_path / "refusal_cache")


@pytest.fixture
def sample_direction():
    """Create sample RefusalDirection for testing."""
    return RefusalDirection(
        direction=[0.1, 0.2, 0.3, 0.4, 0.5],
        layer_index=12,
        hidden_size=5,
        strength=0.85,
        explained_variance=0.42,
        model_id="test-model",
        computed_at=datetime(2025, 1, 15, 10, 30, 0),
    )


class TestRefusalDirectionCacheInit:
    """Tests for RefusalDirectionCache initialization."""

    def test_default_cache_directory(self):
        """Test default cache directory is set."""
        cache = RefusalDirectionCache()
        assert "ModelCypher" in str(cache.cache_directory)
        assert "refusal_directions" in str(cache.cache_directory)

    def test_custom_cache_directory(self, tmp_path):
        """Test custom cache directory is used."""
        custom_dir = tmp_path / "custom_cache"
        cache = RefusalDirectionCache(cache_directory=custom_dir)
        assert cache.cache_directory == custom_dir

    def test_empty_memory_cache_on_init(self, cache):
        """Test memory cache starts empty."""
        assert len(cache._memory_cache) == 0


class TestRefusalDirectionCacheShared:
    """Tests for singleton pattern."""

    def test_shared_returns_same_instance(self):
        """Test shared() returns singleton."""
        # Reset singleton for test
        RefusalDirectionCache._shared_instance = None

        instance1 = RefusalDirectionCache.shared()
        instance2 = RefusalDirectionCache.shared()

        assert instance1 is instance2

    def test_shared_creates_instance_if_none(self):
        """Test shared() creates instance when none exists."""
        RefusalDirectionCache._shared_instance = None

        instance = RefusalDirectionCache.shared()

        assert instance is not None
        assert isinstance(instance, RefusalDirectionCache)

        # Clean up
        RefusalDirectionCache._shared_instance = None


class TestRefusalDirectionCacheSave:
    """Tests for save method."""

    def test_save_creates_directory(self, cache, sample_direction, tmp_path):
        """Test save creates cache directory."""
        model_path = tmp_path / "models" / "test-model"

        assert not cache.cache_directory.exists()
        cache.save(sample_direction, model_path)
        assert cache.cache_directory.exists()

    def test_save_writes_to_disk(self, cache, sample_direction, tmp_path):
        """Test save writes JSON file to disk."""
        model_path = tmp_path / "models" / "test-model"

        cache.save(sample_direction, model_path)

        # Find the cache file
        cache_files = list(cache.cache_directory.glob("*.json"))
        assert len(cache_files) == 1

    def test_save_updates_memory_cache(self, cache, sample_direction, tmp_path):
        """Test save updates memory cache."""
        model_path = tmp_path / "models" / "test-model"

        cache.save(sample_direction, model_path)

        assert len(cache._memory_cache) == 1

    def test_save_writes_correct_json_structure(self, cache, sample_direction, tmp_path):
        """Test saved JSON has correct structure."""
        model_path = tmp_path / "models" / "test-model"

        cache.save(sample_direction, model_path)

        cache_files = list(cache.cache_directory.glob("*.json"))
        payload = json.loads(cache_files[0].read_text())

        assert "direction" in payload
        assert "layerIndex" in payload
        assert "hiddenSize" in payload
        assert "strength" in payload
        assert "explainedVariance" in payload
        assert "modelID" in payload
        assert "computedAt" in payload

    def test_save_preserves_direction_values(self, cache, sample_direction, tmp_path):
        """Test saved direction values match original."""
        model_path = tmp_path / "models" / "test-model"

        cache.save(sample_direction, model_path)

        cache_files = list(cache.cache_directory.glob("*.json"))
        payload = json.loads(cache_files[0].read_text())

        assert payload["direction"] == sample_direction.direction
        assert payload["layerIndex"] == sample_direction.layer_index
        assert payload["strength"] == sample_direction.strength


class TestRefusalDirectionCacheLoad:
    """Tests for load method."""

    def test_load_returns_none_for_missing(self, cache, tmp_path):
        """Test load returns None for non-existent cache."""
        model_path = tmp_path / "models" / "nonexistent"

        result = cache.load(model_path)

        assert result is None

    def test_load_from_disk(self, cache, sample_direction, tmp_path):
        """Test load retrieves from disk."""
        model_path = tmp_path / "models" / "test-model"
        cache.save(sample_direction, model_path)

        # Clear memory cache to force disk load
        cache._memory_cache.clear()

        result = cache.load(model_path)

        assert result is not None
        assert result.layer_index == sample_direction.layer_index
        assert result.strength == sample_direction.strength

    def test_load_from_memory_cache(self, cache, sample_direction, tmp_path):
        """Test load retrieves from memory cache."""
        model_path = tmp_path / "models" / "test-model"
        cache.save(sample_direction, model_path)

        # Delete disk file to confirm memory cache is used
        for f in cache.cache_directory.glob("*.json"):
            f.unlink()

        result = cache.load(model_path)

        assert result is not None
        assert result.layer_index == sample_direction.layer_index

    def test_load_updates_memory_cache_from_disk(self, cache, sample_direction, tmp_path):
        """Test load updates memory cache when loading from disk."""
        model_path = tmp_path / "models" / "test-model"
        cache.save(sample_direction, model_path)
        cache._memory_cache.clear()

        assert len(cache._memory_cache) == 0

        cache.load(model_path)

        assert len(cache._memory_cache) == 1

    def test_load_returns_none_on_model_id_mismatch(self, cache, sample_direction, tmp_path):
        """Test load returns None if model_id doesn't match path."""
        model_path = tmp_path / "models" / "test-model"
        cache.save(sample_direction, model_path)
        cache._memory_cache.clear()

        # Create direction with different model_id
        wrong_direction = RefusalDirection(
            direction=sample_direction.direction,
            layer_index=sample_direction.layer_index,
            hidden_size=sample_direction.hidden_size,
            strength=sample_direction.strength,
            explained_variance=sample_direction.explained_variance,
            model_id="wrong-model",  # Different
            computed_at=sample_direction.computed_at,
        )

        # Overwrite cache file with wrong model_id
        cache_key = cache._cache_key(model_path)
        cache_file = cache._cache_file(cache_key)
        payload = cache._direction_to_dict(wrong_direction)
        cache_file.write_text(json.dumps(payload))

        result = cache.load(model_path)

        assert result is None


class TestRefusalDirectionCacheInvalidate:
    """Tests for invalidate method."""

    def test_invalidate_removes_from_memory(self, cache, sample_direction, tmp_path):
        """Test invalidate removes from memory cache."""
        model_path = tmp_path / "models" / "test-model"
        cache.save(sample_direction, model_path)

        assert len(cache._memory_cache) == 1

        cache.invalidate(model_path)

        assert len(cache._memory_cache) == 0

    def test_invalidate_removes_from_disk(self, cache, sample_direction, tmp_path):
        """Test invalidate removes disk file."""
        model_path = tmp_path / "models" / "test-model"
        cache.save(sample_direction, model_path)

        cache_files = list(cache.cache_directory.glob("*.json"))
        assert len(cache_files) == 1

        cache.invalidate(model_path)

        cache_files = list(cache.cache_directory.glob("*.json"))
        assert len(cache_files) == 0

    def test_invalidate_handles_missing(self, cache, tmp_path):
        """Test invalidate handles non-existent cache gracefully."""
        model_path = tmp_path / "models" / "nonexistent"

        # Should not raise
        cache.invalidate(model_path)


class TestRefusalDirectionCacheClearAll:
    """Tests for clear_all method."""

    def test_clear_all_empties_memory_cache(self, cache, sample_direction, tmp_path):
        """Test clear_all empties memory cache."""
        model_path = tmp_path / "models" / "test-model"
        cache.save(sample_direction, model_path)

        cache.clear_all()

        assert len(cache._memory_cache) == 0

    def test_clear_all_removes_all_disk_files(self, cache, tmp_path):
        """Test clear_all removes all disk cache files."""
        # Save multiple directions
        for i in range(3):
            direction = RefusalDirection(
                direction=[0.1 * i],
                layer_index=i,
                hidden_size=1,
                strength=0.5,
                explained_variance=0.3,
                model_id=f"model-{i}",
                computed_at=datetime.now(),
            )
            cache.save(direction, tmp_path / "models" / f"model-{i}")

        cache_files = list(cache.cache_directory.glob("*.json"))
        assert len(cache_files) == 3

        cache.clear_all()

        cache_files = list(cache.cache_directory.glob("*.json"))
        assert len(cache_files) == 0

    def test_clear_all_handles_missing_directory(self, cache):
        """Test clear_all handles non-existent directory."""
        # Directory doesn't exist yet
        assert not cache.cache_directory.exists()

        # Should not raise
        cache.clear_all()


class TestRefusalDirectionCacheListCached:
    """Tests for list_cached method."""

    def test_list_cached_empty_when_no_directory(self, cache):
        """Test list_cached returns empty for non-existent directory."""
        result = cache.list_cached()

        assert result == []

    def test_list_cached_returns_all_entries(self, cache, tmp_path):
        """Test list_cached returns all cached entries."""
        for i in range(3):
            direction = RefusalDirection(
                direction=[0.1 * (i + 1)],  # Non-zero values
                layer_index=i + 1,  # Non-zero to avoid falsy serialization issue
                hidden_size=1,
                strength=0.5,
                explained_variance=0.3,
                model_id=f"model-{i}",
                computed_at=datetime(2025, 1, 15 + i, 10, 0, 0),
            )
            cache.save(direction, tmp_path / "models" / f"model-{i}")

        result = cache.list_cached()

        assert len(result) == 3

    def test_list_cached_sorted_by_date_descending(self, cache, tmp_path):
        """Test list_cached returns newest first."""
        dates = [
            datetime(2025, 1, 10),
            datetime(2025, 1, 20),
            datetime(2025, 1, 15),
        ]

        for i, dt in enumerate(dates):
            direction = RefusalDirection(
                direction=[0.1],
                layer_index=5,  # Non-zero to avoid falsy serialization issue
                hidden_size=1,
                strength=0.5,
                explained_variance=0.3,
                model_id=f"model-{i}",
                computed_at=dt,
            )
            cache.save(direction, tmp_path / "models" / f"model-{i}")

        result = cache.list_cached()

        # Should be sorted newest to oldest
        result_dates = [item[1] for item in result]
        assert result_dates[0] == datetime(2025, 1, 20)
        assert result_dates[1] == datetime(2025, 1, 15)
        assert result_dates[2] == datetime(2025, 1, 10)

    def test_list_cached_returns_model_ids(self, cache, sample_direction, tmp_path):
        """Test list_cached includes model IDs."""
        model_path = tmp_path / "models" / "test-model"
        cache.save(sample_direction, model_path)

        result = cache.list_cached()

        assert len(result) == 1
        assert result[0][0] == "test-model"


class TestRefusalDirectionCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_cache_key_includes_model_name(self, cache, tmp_path):
        """Test cache key includes model name."""
        model_path = tmp_path / "models" / "my-model"

        key = cache._cache_key(model_path)

        assert "my-model" in key

    def test_cache_key_deterministic(self, cache, tmp_path):
        """Test same path produces same key."""
        model_path = tmp_path / "models" / "test-model"

        key1 = cache._cache_key(model_path)
        key2 = cache._cache_key(model_path)

        assert key1 == key2

    def test_cache_key_different_for_different_paths(self, cache, tmp_path):
        """Test different paths produce different keys."""
        path1 = tmp_path / "models" / "model-a"
        path2 = tmp_path / "models" / "model-b"

        key1 = cache._cache_key(path1)
        key2 = cache._cache_key(path2)

        assert key1 != key2

    def test_stable_hash_deterministic(self):
        """Test stable hash is deterministic."""
        value = "test-string"

        hash1 = RefusalDirectionCache._stable_hash(value)
        hash2 = RefusalDirectionCache._stable_hash(value)

        assert hash1 == hash2

    def test_stable_hash_different_for_different_values(self):
        """Test different values produce different hashes."""
        hash1 = RefusalDirectionCache._stable_hash("string-a")
        hash2 = RefusalDirectionCache._stable_hash("string-b")

        assert hash1 != hash2


class TestRefusalDirectionSerialization:
    """Tests for direction serialization/deserialization."""

    def test_direction_to_dict(self, sample_direction):
        """Test direction serialization."""
        result = RefusalDirectionCache._direction_to_dict(sample_direction)

        assert result["direction"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result["layerIndex"] == 12
        assert result["hiddenSize"] == 5
        assert result["strength"] == 0.85
        assert result["explainedVariance"] == 0.42
        assert result["modelID"] == "test-model"
        assert "2025-01-15" in result["computedAt"]

    def test_direction_from_dict(self):
        """Test direction deserialization."""
        payload = {
            "direction": [1.0, 2.0, 3.0],
            "layerIndex": 8,
            "hiddenSize": 3,
            "strength": 0.9,
            "explainedVariance": 0.5,
            "modelID": "restored-model",
            "computedAt": "2025-06-01T12:00:00",
        }

        result = RefusalDirectionCache._direction_from_dict(payload)

        assert result.direction == [1.0, 2.0, 3.0]
        assert result.layer_index == 8
        assert result.hidden_size == 3
        assert result.strength == 0.9
        assert result.explained_variance == 0.5
        assert result.model_id == "restored-model"
        assert result.computed_at == datetime(2025, 6, 1, 12, 0, 0)

    def test_direction_roundtrip(self, sample_direction):
        """Test serialization roundtrip preserves data."""
        payload = RefusalDirectionCache._direction_to_dict(sample_direction)
        restored = RefusalDirectionCache._direction_from_dict(payload)

        assert restored.direction == sample_direction.direction
        assert restored.layer_index == sample_direction.layer_index
        assert restored.hidden_size == sample_direction.hidden_size
        assert restored.strength == sample_direction.strength
        assert restored.explained_variance == sample_direction.explained_variance
        assert restored.model_id == sample_direction.model_id
        assert restored.computed_at == sample_direction.computed_at

    def test_direction_from_dict_with_snake_case_keys(self):
        """Test deserialization handles snake_case keys."""
        payload = {
            "direction": [1.0],
            "layer_index": 5,
            "hidden_size": 1,
            "strength": 0.7,
            "explained_variance": 0.3,
            "model_id": "snake-model",
            "computed_at": "2025-03-15T08:00:00",
        }

        result = RefusalDirectionCache._direction_from_dict(payload)

        assert result.layer_index == 5
        assert result.hidden_size == 1
        assert result.model_id == "snake-model"

    def test_direction_from_dict_missing_computed_at(self):
        """Test deserialization handles missing computed_at."""
        payload = {
            "direction": [1.0],
            "layerIndex": 5,
            "hiddenSize": 1,
            "strength": 0.7,
            "explainedVariance": 0.3,
            "modelID": "no-date-model",
            # No computedAt
        }

        result = RefusalDirectionCache._direction_from_dict(payload)

        # Should use current time (just verify it's a datetime)
        assert isinstance(result.computed_at, datetime)
