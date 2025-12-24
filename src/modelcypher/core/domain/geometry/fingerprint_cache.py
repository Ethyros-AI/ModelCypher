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

"""Cache for model activation fingerprints.

Caches extracted fingerprints from MLX model inference to avoid
expensive repeated computations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path


from modelcypher.core.domain.cache import CacheConfig, TwoLevelCache, content_hash
from modelcypher.core.domain.geometry.invariant_layer_mapper import (
    ModelFingerprints,
    ActivationFingerprint,
    ActivatedDimension,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedFingerprints:
    """Cached model fingerprints data."""

    model_id: str
    layer_count: int
    fingerprint_count: int
    fingerprints_data: tuple[tuple[str, tuple[tuple[int, tuple[tuple[int, float], ...]], ...]], ...]
    """Nested tuple structure for hashability: (prime_id, ((layer_idx, ((dim, act), ...)), ...))"""


class ModelFingerprintCache:
    """
    Two-level cache for model activation fingerprints.

    Caches fingerprints extracted from MLX model inference.
    Cache key includes model path, modification time, and config hash.

    Cache is stored in ~/Library/Caches/ModelCypher/fingerprints/
    """

    CACHE_VERSION = 1
    _shared_instance: "ModelFingerprintCache" | None = None

    @classmethod
    def shared(cls) -> "ModelFingerprintCache":
        """Get the shared singleton instance."""
        if cls._shared_instance is None:
            cls._shared_instance = ModelFingerprintCache()
        return cls._shared_instance

    def __init__(self, cache_directory: Path | None = None) -> None:
        """
        Initialize the cache.

        Args:
            cache_directory: Override default cache directory
        """
        base = cache_directory or (
            Path.home() / "Library" / "Caches" / "ModelCypher" / "fingerprints"
        )

        config = CacheConfig(
            memory_limit=10,  # Keep only 10 models in memory (fingerprints are large)
            disk_ttl_seconds=30 * 24 * 60 * 60,  # 30 days (model fingerprints rarely change)
            cache_version=self.CACHE_VERSION,
        )

        self._cache: TwoLevelCache[CachedFingerprints] = TwoLevelCache(
            cache_directory=base,
            serializer=self._serialize,
            deserializer=self._deserialize,
            config=config,
        )

    def load(
        self,
        model_path: str,
        config_hash: str,
    ) -> ModelFingerprints | None:
        """
        Load cached fingerprints for a model.

        Args:
            model_path: Path to the model directory
            config_hash: Hash of the fingerprinting config

        Returns:
            Cached ModelFingerprints or None if not found/invalid
        """
        path = Path(model_path).expanduser().resolve()

        # Check if model has been modified since cache
        mtime = self._get_model_mtime(path)
        if mtime is None:
            logger.debug("Model path does not exist: %s", path)
            return None

        cache_key = self._make_cache_key(path, config_hash, mtime)
        cached = self._cache.get(cache_key)

        if cached is None:
            return None

        # Convert cached data back to ModelFingerprints
        return self._to_model_fingerprints(cached)

    def save(
        self,
        model_path: str,
        config_hash: str,
        fingerprints: ModelFingerprints,
    ) -> None:
        """
        Cache fingerprints for a model.

        Args:
            model_path: Path to the model directory
            config_hash: Hash of the fingerprinting config
            fingerprints: Fingerprints to cache
        """
        path = Path(model_path).expanduser().resolve()
        mtime = self._get_model_mtime(path)

        if mtime is None:
            logger.warning("Cannot cache fingerprints - model path does not exist: %s", path)
            return

        cache_key = self._make_cache_key(path, config_hash, mtime)
        cached = self._from_model_fingerprints(fingerprints)
        self._cache.set(cache_key, cached)
        logger.info("Cached fingerprints for %s (%d probes)", path.name, len(fingerprints.fingerprints))

    def invalidate_model(self, model_path: str) -> None:
        """
        Invalidate all cached fingerprints for a model.

        This removes all cache entries for the given model path,
        regardless of config hash or modification time.

        Args:
            model_path: Path to the model directory
        """
        path = Path(model_path).expanduser().resolve()
        prefix = hashlib.sha256(str(path).encode()).hexdigest()[:8]

        # List all cached keys and remove those matching the model
        for key in self._cache.list_cached():
            if key.startswith(prefix):
                self._cache.invalidate(key)

        logger.info("Invalidated fingerprint cache for %s", path.name)

    def clear_all(self) -> None:
        """Clear all cached fingerprints."""
        self._cache.clear_all()
        logger.info("Cleared all fingerprint caches")

    def _get_model_mtime(self, path: Path) -> float | None:
        """Get model modification time from config.json."""
        config_path = path / "config.json"
        if config_path.exists():
            return config_path.stat().st_mtime
        # Fall back to model path if config.json doesn't exist
        if path.exists():
            return path.stat().st_mtime
        return None

    def _make_cache_key(self, path: Path, config_hash: str, mtime: float) -> str:
        """Create cache key from model path, config hash, and mtime."""
        # Include model path hash, config hash, and mtime
        path_hash = hashlib.sha256(str(path).encode()).hexdigest()[:8]
        mtime_str = f"{mtime:.0f}"
        return f"{path_hash}_{config_hash}_{mtime_str}"

    def _from_model_fingerprints(self, fingerprints: ModelFingerprints) -> CachedFingerprints:
        """Convert ModelFingerprints to cacheable format."""
        fp_data = []
        for fp in fingerprints.fingerprints:
            layer_data = []
            for layer_idx, dims in fp.activated_dimensions.items():
                dim_data = tuple((d.dimension, d.activation) for d in dims)
                layer_data.append((layer_idx, dim_data))
            fp_data.append((fp.prime_id, tuple(sorted(layer_data))))

        return CachedFingerprints(
            model_id=fingerprints.model_id,
            layer_count=fingerprints.layer_count,
            fingerprint_count=len(fingerprints.fingerprints),
            fingerprints_data=tuple(fp_data),
        )

    def _to_model_fingerprints(self, cached: CachedFingerprints) -> ModelFingerprints:
        """Convert cached data back to ModelFingerprints."""
        fingerprints = []
        for prime_id, layer_data in cached.fingerprints_data:
            activated_dimensions = {}
            for layer_idx, dim_data in layer_data:
                dims = [
                    ActivatedDimension(dimension=dim, activation=act)
                    for dim, act in dim_data
                ]
                activated_dimensions[layer_idx] = dims

            fingerprints.append(
                ActivationFingerprint(
                    prime_id=prime_id,
                    activated_dimensions=activated_dimensions,
                )
            )

        return ModelFingerprints(
            model_id=cached.model_id,
            layer_count=cached.layer_count,
            fingerprints=fingerprints,
        )

    @staticmethod
    def _serialize(cached: CachedFingerprints) -> dict:
        """Serialize CachedFingerprints to dict."""
        return {
            "model_id": cached.model_id,
            "layer_count": cached.layer_count,
            "fingerprint_count": cached.fingerprint_count,
            "fingerprints": [
                {
                    "prime_id": prime_id,
                    "layers": [
                        {
                            "layer": layer_idx,
                            "dims": [[dim, act] for dim, act in dim_data],
                        }
                        for layer_idx, dim_data in layer_data
                    ],
                }
                for prime_id, layer_data in cached.fingerprints_data
            ],
        }

    @staticmethod
    def _deserialize(data: dict) -> CachedFingerprints:
        """Deserialize dict to CachedFingerprints."""
        fp_data = []
        for fp_dict in data.get("fingerprints", []):
            prime_id = fp_dict["prime_id"]
            layer_data = []
            for layer_dict in fp_dict.get("layers", []):
                layer_idx = layer_dict["layer"]
                dim_data = tuple((d[0], d[1]) for d in layer_dict.get("dims", []))
                layer_data.append((layer_idx, dim_data))
            fp_data.append((prime_id, tuple(layer_data)))

        return CachedFingerprints(
            model_id=data["model_id"],
            layer_count=data["layer_count"],
            fingerprint_count=data.get("fingerprint_count", 0),
            fingerprints_data=tuple(fp_data),
        )


def make_config_hash(
    invariant_scope: str,
    families: list[str] | None = None,
    atlas_sources: list[str] | None = None,
    atlas_domains: list[str] | None = None,
) -> str:
    """
    Create a hash of fingerprinting config for cache key.

    Args:
        invariant_scope: Scope string (e.g., "sequenceInvariants")
        families: Optional list of sequence families
        atlas_sources: Optional list of atlas sources
        atlas_domains: Optional list of atlas domains

    Returns:
        8-character hash string
    """
    data = {
        "scope": invariant_scope.lower(),
        "families": sorted(families) if families else None,
        "sources": sorted(atlas_sources) if atlas_sources else None,
        "domains": sorted(atlas_domains) if atlas_domains else None,
    }
    return content_hash(data)[:8]
