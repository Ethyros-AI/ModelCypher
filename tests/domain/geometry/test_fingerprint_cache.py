# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for fingerprint_cache module.

Covers CachedFingerprints dataclass, ModelFingerprintCache operations,
and make_config_hash consistency.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.fingerprint_cache import (
    CachedFingerprints,
    ModelFingerprintCache,
    make_config_hash,
)


# ---------------------------------------------------------------------------
# CachedFingerprints dataclass
# ---------------------------------------------------------------------------


class TestCachedFingerprints:
    """Tests for the CachedFingerprints frozen dataclass."""

    def test_instantiation_with_all_fields(self):
        fp = CachedFingerprints(
            model_id="test-model",
            layer_count=12,
            fingerprint_count=3,
            fingerprints_data=(
                ("p2", "two", ((0, ((10, 0.5), (20, 0.3))),)),
                ("p3", "three", ((0, ((5, 0.9),)),)),
            ),
        )
        assert fp.model_id == "test-model"
        assert fp.layer_count == 12
        assert fp.fingerprint_count == 3
        assert len(fp.fingerprints_data) == 2

    def test_field_access(self):
        fp = CachedFingerprints(
            model_id="abc",
            layer_count=4,
            fingerprint_count=0,
            fingerprints_data=(),
        )
        assert fp.model_id == "abc"
        assert fp.layer_count == 4
        assert fp.fingerprint_count == 0
        assert fp.fingerprints_data == ()

    def test_frozen(self):
        fp = CachedFingerprints(
            model_id="frozen-test",
            layer_count=1,
            fingerprint_count=0,
            fingerprints_data=(),
        )
        with pytest.raises(AttributeError):
            fp.model_id = "changed"  # type: ignore[misc]

    def test_equality(self):
        kwargs = dict(
            model_id="eq",
            layer_count=2,
            fingerprint_count=1,
            fingerprints_data=(("p2", "two", ((0, ((7, 0.1),)),)),),
        )
        a = CachedFingerprints(**kwargs)
        b = CachedFingerprints(**kwargs)
        assert a == b

    def test_hashable(self):
        fp = CachedFingerprints(
            model_id="hash",
            layer_count=1,
            fingerprint_count=0,
            fingerprints_data=(),
        )
        # Frozen dataclass should be hashable
        assert isinstance(hash(fp), int)


# ---------------------------------------------------------------------------
# ModelFingerprintCache
# ---------------------------------------------------------------------------


class TestModelFingerprintCache:
    """Tests for ModelFingerprintCache initialization and basic operations."""

    def test_constructor_with_tmp_path(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path / "fp_cache")
        assert cache is not None

    def test_shared_returns_same_instance(self, monkeypatch):
        monkeypatch.setattr(ModelFingerprintCache, "_shared_instance", None)
        a = ModelFingerprintCache.shared()
        b = ModelFingerprintCache.shared()
        assert a is b

    def test_shared_singleton_reset(self, monkeypatch):
        """Resetting the singleton should produce a new instance."""
        monkeypatch.setattr(ModelFingerprintCache, "_shared_instance", None)
        first = ModelFingerprintCache.shared()

        monkeypatch.setattr(ModelFingerprintCache, "_shared_instance", None)
        second = ModelFingerprintCache.shared()

        # Different objects after reset
        assert first is not second

    def test_load_empty_cache_returns_none(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path / "empty_fp")
        result = cache.load(model_path=str(tmp_path), config_hash="abc12345")
        assert result is None

    def test_load_nonexistent_model_path_returns_none(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path / "no_model_fp")
        result = cache.load(
            model_path=str(tmp_path / "does_not_exist"),
            config_hash="abc12345",
        )
        assert result is None

    def test_clear_all_no_error(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path / "clear_fp")
        # Should not raise even on empty cache
        cache.clear_all()

    def test_invalidate_model_no_error(self, tmp_path):
        cache = ModelFingerprintCache(cache_directory=tmp_path / "inv_fp")
        # Invalidating a path that has no cached entries should not raise
        cache.invalidate_model(str(tmp_path / "some_model"))

    def test_invalidate_model_with_existing_dir(self, tmp_path):
        model_dir = tmp_path / "model_exists"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        cache = ModelFingerprintCache(cache_directory=tmp_path / "inv2_fp")
        cache.invalidate_model(str(model_dir))
        # Should not raise


# ---------------------------------------------------------------------------
# make_config_hash
# ---------------------------------------------------------------------------


class TestMakeConfigHash:
    """Tests for the make_config_hash function."""

    def test_consistent_hash(self):
        h1 = make_config_hash(invariant_scope="sequenceInvariants")
        h2 = make_config_hash(invariant_scope="sequenceInvariants")
        assert h1 == h2

    def test_different_scope_different_hash(self):
        h1 = make_config_hash(invariant_scope="sequenceInvariants")
        h2 = make_config_hash(invariant_scope="geometricInvariants")
        assert h1 != h2

    def test_case_insensitive_scope(self):
        h1 = make_config_hash(invariant_scope="SequenceInvariants")
        h2 = make_config_hash(invariant_scope="sequenceinvariants")
        assert h1 == h2

    def test_returns_8_char_string(self):
        h = make_config_hash(invariant_scope="test")
        assert isinstance(h, str)
        assert len(h) == 8

    def test_with_families(self):
        h1 = make_config_hash(
            invariant_scope="test",
            families=["fibonacci", "primes"],
        )
        h2 = make_config_hash(
            invariant_scope="test",
            families=["primes", "fibonacci"],
        )
        # Order should not matter (families are sorted internally)
        assert h1 == h2

    def test_families_vs_no_families(self):
        h1 = make_config_hash(invariant_scope="test", families=["primes"])
        h2 = make_config_hash(invariant_scope="test")
        assert h1 != h2

    def test_with_atlas_sources(self):
        h1 = make_config_hash(
            invariant_scope="test",
            atlas_sources=["oeis", "custom"],
        )
        h2 = make_config_hash(
            invariant_scope="test",
            atlas_sources=["custom", "oeis"],
        )
        assert h1 == h2

    def test_atlas_sources_vs_no_sources(self):
        h1 = make_config_hash(
            invariant_scope="test",
            atlas_sources=["oeis"],
        )
        h2 = make_config_hash(invariant_scope="test")
        assert h1 != h2

    def test_with_atlas_domains(self):
        h1 = make_config_hash(
            invariant_scope="test",
            atlas_domains=["math", "logic"],
        )
        h2 = make_config_hash(
            invariant_scope="test",
            atlas_domains=["logic", "math"],
        )
        assert h1 == h2

    def test_atlas_domains_vs_no_domains(self):
        h1 = make_config_hash(
            invariant_scope="test",
            atlas_domains=["math"],
        )
        h2 = make_config_hash(invariant_scope="test")
        assert h1 != h2

    def test_with_probe_texts(self):
        h1 = make_config_hash(
            invariant_scope="test",
            probe_texts={"p1": "hello", "p2": "world"},
        )
        h2 = make_config_hash(
            invariant_scope="test",
            probe_texts={"p1": "hello", "p2": "world"},
        )
        assert h1 == h2

    def test_different_probe_texts(self):
        h1 = make_config_hash(
            invariant_scope="test",
            probe_texts={"p1": "hello"},
        )
        h2 = make_config_hash(
            invariant_scope="test",
            probe_texts={"p1": "goodbye"},
        )
        assert h1 != h2

    def test_all_optional_params_combined(self):
        h1 = make_config_hash(
            invariant_scope="test",
            families=["primes"],
            atlas_sources=["oeis"],
            atlas_domains=["math"],
            probe_texts={"p1": "text"},
        )
        h2 = make_config_hash(
            invariant_scope="test",
            families=["primes"],
            atlas_sources=["oeis"],
            atlas_domains=["math"],
            probe_texts={"p1": "text"},
        )
        assert h1 == h2
        assert len(h1) == 8
