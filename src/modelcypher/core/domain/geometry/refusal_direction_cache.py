from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from modelcypher.core.domain.refusal_direction_detector import RefusalDirection

logger = logging.getLogger(__name__)


class RefusalDirectionCache:
    _shared_instance: "RefusalDirectionCache" | None = None

    @classmethod
    def shared(cls) -> "RefusalDirectionCache":
        if cls._shared_instance is None:
            cls._shared_instance = RefusalDirectionCache()
        return cls._shared_instance

    def __init__(self, cache_directory: Optional[Path] = None) -> None:
        base = cache_directory or (Path.home() / "Library" / "Caches" / "ModelCypher" / "refusal_directions")
        self.cache_directory = base
        self._memory_cache: dict[str, RefusalDirection] = {}

    def load(self, model_path: str | Path) -> Optional[RefusalDirection]:
        cache_key = self._cache_key(model_path)

        if cache_key in self._memory_cache:
            logger.debug("RefusalDirection cache hit (memory): %s", cache_key)
            return self._memory_cache[cache_key]

        cache_file = self._cache_file(cache_key)
        if not cache_file.exists():
            logger.debug("RefusalDirection cache miss: %s", cache_key)
            return None

        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            direction = self._direction_from_dict(payload)
            if direction.model_id != Path(model_path).name:
                logger.warning(
                    "RefusalDirection cache mismatch: expected %s, got %s",
                    Path(model_path).name,
                    direction.model_id,
                )
                return None
            self._memory_cache[cache_key] = direction
            logger.debug("RefusalDirection cache hit (disk): %s", cache_key)
            return direction
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load cached RefusalDirection: %s", exc)
            return None

    def save(self, direction: RefusalDirection, model_path: str | Path) -> None:
        cache_key = self._cache_key(model_path)
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        payload = self._direction_to_dict(direction)
        cache_file = self._cache_file(cache_key)
        cache_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        self._memory_cache[cache_key] = direction
        logger.info("Cached RefusalDirection for %s at %s", Path(model_path).name, cache_file)

    def invalidate(self, model_path: str | Path) -> None:
        cache_key = self._cache_key(model_path)
        self._memory_cache.pop(cache_key, None)
        cache_file = self._cache_file(cache_key)
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Invalidated RefusalDirection cache for %s", cache_key)

    def clear_all(self) -> None:
        self._memory_cache.clear()
        if self.cache_directory.exists():
            for file_path in self.cache_directory.glob("*.json"):
                file_path.unlink()
        logger.info("Cleared all RefusalDirection caches")

    def list_cached(self) -> list[tuple[str, datetime]]:
        if not self.cache_directory.exists():
            return []

        results: list[tuple[str, datetime]] = []
        for file_path in self.cache_directory.glob("*.json"):
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                direction = self._direction_from_dict(payload)
                results.append((direction.model_id, direction.computed_at))
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to read cached direction %s: %s", file_path.name, exc)

        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def _cache_key(self, model_path: str | Path) -> str:
        path = Path(model_path)
        return f"{path.name}_{self._stable_hash(str(path))}"

    @staticmethod
    def _stable_hash(value: str) -> int:
        hash_value = 5381
        for char in value.encode("utf-8"):
            hash_value = ((hash_value << 5) + hash_value) + char
        return hash_value

    def _cache_file(self, cache_key: str) -> Path:
        return self.cache_directory / f"{cache_key}.json"

    @staticmethod
    def _direction_to_dict(direction: RefusalDirection) -> dict:
        return {
            "direction": list(direction.direction),
            "layerIndex": direction.layer_index,
            "hiddenSize": direction.hidden_size,
            "strength": direction.strength,
            "explainedVariance": direction.explained_variance,
            "modelID": direction.model_id,
            "computedAt": direction.computed_at.isoformat(),
        }

    @staticmethod
    def _direction_from_dict(payload: dict) -> RefusalDirection:
        computed_value = payload.get("computedAt") or payload.get("computed_at")
        computed_at = datetime.fromisoformat(computed_value) if computed_value else datetime.utcnow()
        return RefusalDirection(
            direction=list(payload["direction"]),
            layer_index=int(payload.get("layerIndex") or payload.get("layer_index")),
            hidden_size=int(payload.get("hiddenSize") or payload.get("hidden_size")),
            strength=float(payload["strength"]),
            explained_variance=float(payload.get("explainedVariance") or payload.get("explained_variance")),
            model_id=payload.get("modelID") or payload.get("model_id"),
            computed_at=computed_at,
        )
