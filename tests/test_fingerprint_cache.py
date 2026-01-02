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

"""Tests for fingerprint cache (model activation fingerprint caching)."""

import json
from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.fingerprint_cache import (
    CachedFingerprints,
    ModelFingerprintCache,
    make_config_hash,
)
from modelcypher.core.domain.geometry.invariant_layer_mapper import (
    ActivatedDimension,
    ActivationFingerprint,
    ModelFingerprints,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


class TestCachedFingerprints:
    """Tests for CachedFingerprints dataclass."""

    def test_fields(self):
        cached = CachedFingerprints(
            model_id="test-model",
            layer_count=32,
            fingerprint_count=10,
            fingerprints_data=(
                ("prime_1", "text 1", ((0, ((0, 0.5), (1, 0.3))),)),
                ("prime_2", "text 2", ((0, ((2, 0.8),)),)),
            ),
        )
        assert cached.model_id == "test-model"
        assert cached.layer_count == 32
        assert cached.fingerprint_count == 10
        assert len(cached.fingerprints_data) == 2

    def test_frozen(self):
        cached = CachedFingerprints(
            model_id="test",
            layer_count=10,
            fingerprint_count=1,
            fingerprints_data=(),
        )
        with pytest.raises(AttributeError):
            cached.model_id = "other"

    def test_hashable(self):
        cached = CachedFingerprints(
            model_id="test",
            layer_count=10,
            fingerprint_count=1,
            fingerprints_data=(("p1", "t1", ((0, ((1, 0.5),)),)),),
        )
        h = hash(cached)
        assert isinstance(h, int)

    def test_usable_in_set(self):
        c1 = CachedFingerprints(
            model_id="model1",
            layer_count=10,
            fingerprint_count=1,
            fingerprints_data=(),
        )
        c2 = CachedFingerprints(
            model_id="model2",
            layer_count=20,
            fingerprint_count=2,
            fingerprints_data=(),
        )
        s = {c1, c2}
        assert len(s) == 2


class TestModelFingerprintCacheInit:
    """Tests for ModelFingerprintCache initialization."""

    def test_init_default_directory(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path / "fingerprints")
        assert cache._cache is not None

    def test_init_custom_directory(self, tmp_path):
        custom_dir = tmp_path / "my_fingerprints"
        cache = ModelFingerprintCache(cache_directory=custom_dir)
        assert cache._cache is not None

    def test_shared_singleton(self):
        # Reset singleton for testing
        ModelFingerprintCache._shared_instance = None
        instance1 = ModelFingerprintCache.shared()
        instance2 = ModelFingerprintCache.shared()
        assert instance1 is instance2
        # Reset after test
        ModelFingerprintCache._shared_instance = None


class TestModelFingerprintCacheSaveLoad:
    """Tests for ModelFingerprintCache save/load operations."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a cache with temp directory."""
        return ModelFingerprintCache(cache_directory=tmp_path / "fingerprints")

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create a fake model directory with config.json."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        config = model_path / "config.json"
        config.write_text('{"model_type": "llama"}')
        return model_path

    @pytest.fixture
    def sample_fingerprints(self):
        """Create sample ModelFingerprints."""
        fps = [
            ActivationFingerprint(
                prime_id="being",
                prime_text="existence",
                activated_dimensions={
                    0: [
                        ActivatedDimension(index=10, activation=0.85),
                        ActivatedDimension(index=20, activation=0.72),
                    ],
                    1: [
                        ActivatedDimension(index=5, activation=0.65),
                    ],
                },
            ),
            ActivationFingerprint(
                prime_id="negation",
                prime_text="not",
                activated_dimensions={
                    0: [
                        ActivatedDimension(index=15, activation=0.91),
                    ],
                },
            ),
        ]
        return ModelFingerprints(
            model_id="test-model",
            layer_count=32,
            fingerprints=fps,
        )

    def test_save_and_load(self, cache, model_dir, sample_fingerprints):
        config_hash = "abc12345"

        # Save
        cache.save(str(model_dir), config_hash, sample_fingerprints)

        # Load
        loaded = cache.load(str(model_dir), config_hash)
        assert loaded is not None
        assert loaded.model_id == sample_fingerprints.model_id
        assert loaded.layer_count == sample_fingerprints.layer_count
        assert len(loaded.fingerprints) == len(sample_fingerprints.fingerprints)

    def test_load_nonexistent_path(self, cache):
        result = cache.load("/nonexistent/path", "config123")
        assert result is None

    def test_load_uncached(self, cache, model_dir):
        result = cache.load(str(model_dir), "never_cached")
        assert result is None

    def test_save_nonexistent_path(self, cache, sample_fingerprints):
        # Should not raise, just log warning
        cache.save("/nonexistent/path", "config123", sample_fingerprints)

    def test_load_preserves_fingerprint_data(self, cache, model_dir, sample_fingerprints):
        config_hash = "test123"
        cache.save(str(model_dir), config_hash, sample_fingerprints)

        loaded = cache.load(str(model_dir), config_hash)
        assert loaded is not None

        # Check first fingerprint
        orig_fp = sample_fingerprints.fingerprints[0]
        loaded_fp = loaded.fingerprints[0]
        assert loaded_fp.prime_id == orig_fp.prime_id
        assert loaded_fp.prime_text == orig_fp.prime_text

        # Check activated dimensions
        assert 0 in loaded_fp.activated_dimensions
        loaded_dims = loaded_fp.activated_dimensions[0]
        assert len(loaded_dims) == 2
        assert loaded_dims[0].index == 10
        assert abs(loaded_dims[0].activation - 0.85) <= _eps(
            loaded_dims[0].activation, 0.85
        )


class TestModelFingerprintCacheInvalidation:
    """Tests for ModelFingerprintCache invalidation."""

    @pytest.fixture
    def cache(self, tmp_path):
        return ModelFingerprintCache(cache_directory=tmp_path / "fingerprints")

    @pytest.fixture
    def model_dir(self, tmp_path):
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        config = model_path / "config.json"
        config.write_text('{"model_type": "llama"}')
        return model_path

    @pytest.fixture
    def sample_fingerprints(self):
        fps = [
            ActivationFingerprint(
                prime_id="test",
                prime_text="test text",
                activated_dimensions={0: [ActivatedDimension(index=1, activation=0.5)]},
            ),
        ]
        return ModelFingerprints(model_id="test", layer_count=10, fingerprints=fps)

    def test_invalidate_model(self, cache, model_dir, sample_fingerprints):
        config_hash = "hash123"
        cache.save(str(model_dir), config_hash, sample_fingerprints)

        # Verify it's cached
        assert cache.load(str(model_dir), config_hash) is not None

        # Invalidate
        cache.invalidate_model(str(model_dir))

        # Should be gone
        assert cache.load(str(model_dir), config_hash) is None

    def test_clear_all(self, cache, model_dir, sample_fingerprints):
        # Cache multiple
        cache.save(str(model_dir), "hash1", sample_fingerprints)
        cache.save(str(model_dir), "hash2", sample_fingerprints)

        # Clear all
        cache.clear_all()

        # Both should be gone
        assert cache.load(str(model_dir), "hash1") is None
        assert cache.load(str(model_dir), "hash2") is None


class TestModelFingerprintCacheMtime:
    """Tests for modification time handling."""

    @pytest.fixture
    def cache(self, tmp_path):
        return ModelFingerprintCache(cache_directory=tmp_path / "fingerprints")

    def test_cache_invalidates_on_mtime_change(self, cache, tmp_path):
        import os
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = model_dir / "config.json"
        config.write_text('{"v": 1}')

        fps = ModelFingerprints(
            model_id="test",
            layer_count=10,
            fingerprints=[
                ActivationFingerprint(
                    prime_id="p1",
                    prime_text="t1",
                    activated_dimensions={0: [ActivatedDimension(index=1, activation=0.5)]},
                )
            ],
        )

        # Save with current mtime
        cache.save(str(model_dir), "hash1", fps)
        assert cache.load(str(model_dir), "hash1") is not None

        # Modify mtime by setting it to a future time
        current_mtime = config.stat().st_mtime
        future_mtime = current_mtime + 100  # 100 seconds in the future
        os.utime(config, (future_mtime, future_mtime))

        # Cache should miss due to different mtime
        assert cache.load(str(model_dir), "hash1") is None

    def test_fallback_to_dir_mtime_without_config(self, cache, tmp_path):
        model_dir = tmp_path / "model_no_config"
        model_dir.mkdir()
        # No config.json

        fps = ModelFingerprints(
            model_id="test",
            layer_count=5,
            fingerprints=[
                ActivationFingerprint(
                    prime_id="p1",
                    prime_text="t1",
                    activated_dimensions={},
                )
            ],
        )

        # Should use directory mtime as fallback
        cache.save(str(model_dir), "hash", fps)
        loaded = cache.load(str(model_dir), "hash")
        assert loaded is not None


class TestModelFingerprintCacheSerialization:
    """Tests for serialization/deserialization methods."""

    def test_serialize_roundtrip(self):
        cached = CachedFingerprints(
            model_id="test-model",
            layer_count=32,
            fingerprint_count=2,
            fingerprints_data=(
                ("prime_1", "text 1", ((0, ((0, 0.5), (1, 0.3))), (1, ((5, 0.8),)))),
                ("prime_2", "text 2", ((2, ((10, 0.9),)),)),
            ),
        )

        serialized = ModelFingerprintCache._serialize(cached)
        deserialized = ModelFingerprintCache._deserialize(serialized)

        assert deserialized.model_id == cached.model_id
        assert deserialized.layer_count == cached.layer_count
        assert deserialized.fingerprint_count == cached.fingerprint_count
        assert len(deserialized.fingerprints_data) == len(cached.fingerprints_data)

    def test_serialize_produces_json_compatible(self):
        cached = CachedFingerprints(
            model_id="test",
            layer_count=10,
            fingerprint_count=1,
            fingerprints_data=(("p1", "t1", ((0, ((1, 0.5),)),)),),
        )

        serialized = ModelFingerprintCache._serialize(cached)
        # Should be JSON-serializable
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)

    def test_deserialize_handles_v1_format(self):
        # V1 format didn't have prime_text
        v1_data = {
            "model_id": "old-model",
            "layer_count": 16,
            "fingerprints": [
                {
                    "prime_id": "being",
                    # No prime_text
                    "layers": [
                        {"layer": 0, "dims": [[5, 0.7], [10, 0.3]]},
                    ],
                }
            ],
        }

        deserialized = ModelFingerprintCache._deserialize(v1_data)
        assert deserialized.model_id == "old-model"
        assert len(deserialized.fingerprints_data) == 1
        # prime_text should default to empty string
        prime_id, prime_text, _ = deserialized.fingerprints_data[0]
        assert prime_id == "being"
        assert prime_text == ""


class TestMakeConfigHash:
    """Tests for make_config_hash function."""

    def test_basic_hash(self):
        h = make_config_hash("sequenceInvariants")
        assert isinstance(h, str)
        assert len(h) == 8

    def test_deterministic(self):
        h1 = make_config_hash("sequenceInvariants")
        h2 = make_config_hash("sequenceInvariants")
        assert h1 == h2

    def test_different_scopes_different_hashes(self):
        h1 = make_config_hash("sequenceInvariants")
        h2 = make_config_hash("philosophicalInvariants")
        assert h1 != h2

    def test_case_insensitive_scope(self):
        h1 = make_config_hash("SequenceInvariants")
        h2 = make_config_hash("sequenceinvariants")
        assert h1 == h2

    def test_with_families(self):
        h1 = make_config_hash("test", families=["fibonacci", "primes"])
        h2 = make_config_hash("test", families=["primes", "fibonacci"])
        # Order shouldn't matter (sorted internally)
        assert h1 == h2

    def test_different_families_different_hashes(self):
        h1 = make_config_hash("test", families=["fibonacci"])
        h2 = make_config_hash("test", families=["primes"])
        assert h1 != h2

    def test_with_atlas_sources(self):
        h1 = make_config_hash("test", atlas_sources=["metaphor", "emotion"])
        h2 = make_config_hash("test", atlas_sources=["emotion", "metaphor"])
        assert h1 == h2

    def test_with_atlas_domains(self):
        h1 = make_config_hash("test", atlas_domains=["safety", "refusal"])
        h2 = make_config_hash("test", atlas_domains=["refusal", "safety"])
        assert h1 == h2

    def test_all_params(self):
        h = make_config_hash(
            "sequenceInvariants",
            families=["fib", "prime"],
            atlas_sources=["metaphor"],
            atlas_domains=["safety"],
        )
        assert isinstance(h, str)
        assert len(h) == 8

    def test_none_families_vs_empty(self):
        h1 = make_config_hash("test", families=None)
        h2 = make_config_hash("test", families=[])
        # Implementation sorts empty list to [], and None stays None,
        # but content_hash may treat them equivalently in JSON
        # This test documents actual behavior
        assert h1 == h2  # Both produce same hash


class TestConversionMethods:
    """Tests for fingerprint conversion methods."""

    @pytest.fixture
    def sample_fingerprints(self):
        fps = [
            ActivationFingerprint(
                prime_id="being",
                prime_text="existence",
                activated_dimensions={
                    0: [
                        ActivatedDimension(index=10, activation=0.85),
                        ActivatedDimension(index=20, activation=0.72),
                    ],
                    5: [
                        ActivatedDimension(index=100, activation=0.91),
                    ],
                },
            ),
            ActivationFingerprint(
                prime_id="negation",
                prime_text="not",
                activated_dimensions={},
            ),
        ]
        return ModelFingerprints(
            model_id="test-model",
            layer_count=32,
            fingerprints=fps,
        )

    def test_from_model_fingerprints(self, tmp_path, sample_fingerprints):
        cache = ModelFingerprintCache(cache_directory=tmp_path)
        cached = cache._from_model_fingerprints(sample_fingerprints)

        assert cached.model_id == "test-model"
        assert cached.layer_count == 32
        assert cached.fingerprint_count == 2
        assert len(cached.fingerprints_data) == 2

    def test_to_model_fingerprints(self, tmp_path):
        cached = CachedFingerprints(
            model_id="test",
            layer_count=16,
            fingerprint_count=1,
            fingerprints_data=(
                ("prime_1", "text 1", ((0, ((5, 0.75), (10, 0.3))),)),
            ),
        )

        cache = ModelFingerprintCache(cache_directory=tmp_path)
        result = cache._to_model_fingerprints(cached)

        assert result.model_id == "test"
        assert result.layer_count == 16
        assert len(result.fingerprints) == 1
        fp = result.fingerprints[0]
        assert fp.prime_id == "prime_1"
        assert fp.prime_text == "text 1"
        assert 0 in fp.activated_dimensions
        dims = fp.activated_dimensions[0]
        assert len(dims) == 2
        assert dims[0].index == 5
        assert dims[0].activation == 0.75

    def test_roundtrip_conversion(self, tmp_path, sample_fingerprints):
        cache = ModelFingerprintCache(cache_directory=tmp_path)

        cached = cache._from_model_fingerprints(sample_fingerprints)
        restored = cache._to_model_fingerprints(cached)

        assert restored.model_id == sample_fingerprints.model_id
        assert restored.layer_count == sample_fingerprints.layer_count
        assert len(restored.fingerprints) == len(sample_fingerprints.fingerprints)

        for orig, rest in zip(sample_fingerprints.fingerprints, restored.fingerprints):
            assert rest.prime_id == orig.prime_id
            assert rest.prime_text == orig.prime_text
            assert set(rest.activated_dimensions.keys()) == set(
                orig.activated_dimensions.keys()
            )


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_make_cache_key_format(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path)
        model_path = Path("/some/model/path")
        config_hash = "abc12345"
        mtime = 1234567890.0

        key = cache._make_cache_key(model_path, config_hash, mtime)

        # Key format: {path_hash}_{config_hash}_{mtime}
        parts = key.split("_")
        assert len(parts) == 3
        assert parts[1] == config_hash
        assert parts[2] == "1234567890"

    def test_different_paths_different_keys(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path)

        key1 = cache._make_cache_key(Path("/path/model1"), "hash", 1000.0)
        key2 = cache._make_cache_key(Path("/path/model2"), "hash", 1000.0)

        assert key1 != key2

    def test_different_configs_different_keys(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path)
        path = Path("/model")

        key1 = cache._make_cache_key(path, "hash1", 1000.0)
        key2 = cache._make_cache_key(path, "hash2", 1000.0)

        assert key1 != key2

    def test_different_mtimes_different_keys(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path)
        path = Path("/model")

        key1 = cache._make_cache_key(path, "hash", 1000.0)
        key2 = cache._make_cache_key(path, "hash", 2000.0)

        assert key1 != key2
