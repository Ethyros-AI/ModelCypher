from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from modelcypher import __version__
from modelcypher.core.domain.model_search import (
    MemoryFitStatus,
    ModelSearchError,
    ModelSearchFilters,
    ModelSearchLibraryFilter,
    ModelSearchPage,
    ModelSearchQuantization,
    ModelSearchResult,
    ModelSearchSortOption,
)
from modelcypher.ports.model_search import ModelSearchService


class HfModelSearchAdapter(ModelSearchService):
    _API_BASE = "https://huggingface.co"
    _MODELS_API = f"{_API_BASE}/api/models"
    _CACHE_TTL_SECONDS = 300.0

    def __init__(self, user_agent: str | None = None) -> None:
        self._user_agent = user_agent or f"ModelCypher/{__version__}"
        self._cache: dict[str, _CachedPage] = {}
        self._rate_limited_until: float | None = None

    def search_models(self, filters: ModelSearchFilters, cursor: str | None = None) -> ModelSearchPage:
        now = time.time()
        if self._rate_limited_until and now < self._rate_limited_until:
            raise ModelSearchError.rate_limited(self._rate_limited_until - now)

        cache_key = self._cache_key(filters, cursor)
        cached = self._cache.get(cache_key)
        if cached and cached.expires_at > now:
            return cached.page

        url = self._build_search_url(filters, cursor)
        if not url:
            raise ModelSearchError.search_failed("Failed to build search URL")

        request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": self._user_agent})
        token = self._hugging_face_token()
        if token:
            request.add_header("Authorization", f"Bearer {token}")

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.getcode()
                headers = response.headers
                payload = response.read()
        except urllib.error.HTTPError as exc:
            self._handle_http_error(exc)
        except urllib.error.URLError as exc:
            raise ModelSearchError.network_error(str(exc.reason)) from exc

        if status == 401:
            raise ModelSearchError.authentication_required()
        if status is None or status < 200 or status >= 300:
            raise ModelSearchError.search_failed(f"HTTP {status}")

        try:
            raw_models = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelSearchError.invalid_response(f"Failed to parse response: {exc}") from exc

        if not isinstance(raw_models, list):
            raise ModelSearchError.invalid_response("Expected model list response")

        available_gb = self._available_memory_gb()
        models: list[ModelSearchResult] = []
        for raw in raw_models:
            if not isinstance(raw, dict):
                continue
            result = self._parse_model(raw)
            if available_gb and result.estimated_size_gb is not None:
                fit_status = self._calculate_fit_status(result.estimated_size_gb, available_gb)
                result = ModelSearchResult(
                    id=result.id,
                    downloads=result.downloads,
                    likes=result.likes,
                    tags=result.tags,
                    author=result.author,
                    pipeline_tag=result.pipeline_tag,
                    last_modified=result.last_modified,
                    is_private=result.is_private,
                    is_gated=result.is_gated,
                    memory_fit_status=fit_status,
                )
            models.append(result)

        next_cursor = self._extract_cursor(headers.get("Link"))
        page = ModelSearchPage(models=models, next_cursor=next_cursor)
        self._cache[cache_key] = _CachedPage(page=page, expires_at=now + self._CACHE_TTL_SECONDS)
        self._prune_cache()
        return page

    def clear_cache(self) -> None:
        self._cache.clear()
        self._rate_limited_until = None

    def _build_search_url(self, filters: ModelSearchFilters, cursor: str | None) -> str | None:
        query: list[tuple[str, str]] = [
            ("pipeline_tag", "text-generation"),
            ("full", "true"),
            ("limit", str(filters.limit)),
        ]

        if filters.sort_by == ModelSearchSortOption.downloads:
            query.extend([("sort", "downloads"), ("direction", "-1")])
        elif filters.sort_by == ModelSearchSortOption.likes:
            query.extend([("sort", "likes"), ("direction", "-1")])
        elif filters.sort_by == ModelSearchSortOption.last_modified:
            query.extend([("sort", "lastModified"), ("direction", "-1")])
        elif filters.sort_by == ModelSearchSortOption.trending:
            query.extend([("sort", "trendingScore"), ("direction", "-1")])

        if filters.query:
            query.append(("search", filters.query))
        if filters.author:
            query.append(("author", filters.author))

        if filters.library != ModelSearchLibraryFilter.any:
            query.append(("library", filters.library.value))

        if filters.quantization == ModelSearchQuantization.four_bit:
            query.append(("quantized", "4bit"))
        elif filters.quantization == ModelSearchQuantization.eight_bit:
            query.append(("quantized", "8bit"))

        if cursor:
            query.append(("cursor", cursor))

        return f"{self._MODELS_API}?{urllib.parse.urlencode(query)}"

    def _parse_model(self, raw: dict[str, Any]) -> ModelSearchResult:
        model_id = raw.get("modelId") or raw.get("id") or ""
        tags = raw.get("tags") if isinstance(raw.get("tags"), list) else []
        return ModelSearchResult(
            id=str(model_id),
            downloads=int(raw.get("downloads") or 0),
            likes=int(raw.get("likes") or 0),
            tags=[str(tag) for tag in tags],
            author=str(raw.get("author")) if raw.get("author") else None,
            pipeline_tag=str(raw.get("pipeline_tag")) if raw.get("pipeline_tag") else None,
            last_modified=self._parse_datetime(raw.get("lastModified")),
            is_private=bool(raw.get("private") or False),
            is_gated=bool(raw.get("gated") or False),
        )

    def _handle_http_error(self, exc: urllib.error.HTTPError) -> None:
        status = exc.code
        if status in (429, 503):
            retry_after = self._parse_retry_after(exc.headers.get("Retry-After"))
            self._rate_limited_until = time.time() + (retry_after or 60)
            raise ModelSearchError.rate_limited(retry_after)
        if status == 401:
            raise ModelSearchError.authentication_required()
        raise ModelSearchError.search_failed(f"HTTP {status}")

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            return None
        try:
            if value.endswith("Z"):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_retry_after(header: str | None) -> float | None:
        if not header:
            return None
        try:
            return float(header)
        except ValueError:
            pass
        try:
            parsed = parsedate_to_datetime(header)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(tz=timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_cursor(link_header: str | None) -> str | None:
        if not link_header:
            return None
        for part in link_header.split(","):
            if 'rel="next"' not in part:
                continue
            start = part.find("<")
            end = part.find(">")
            if start == -1 or end == -1 or end <= start:
                continue
            url = part[start + 1 : end]
            parsed = urllib.parse.urlparse(url)
            query = urllib.parse.parse_qs(parsed.query)
            cursor_values = query.get("cursor")
            if cursor_values:
                return cursor_values[0]
        return None

    @staticmethod
    def _calculate_fit_status(size_gb: float, available_gb: float) -> MemoryFitStatus:
        training_required = size_gb * 2.5
        inference_required = size_gb * 1.5
        buffer = available_gb * 0.85
        if training_required <= buffer:
            return MemoryFitStatus.fits
        if inference_required <= buffer:
            return MemoryFitStatus.tight
        return MemoryFitStatus.too_big

    @staticmethod
    def _available_memory_gb() -> float:
        bytes_available = _available_memory_bytes()
        if bytes_available <= 0:
            return 0.0
        return bytes_available / (1024**3)

    @staticmethod
    def _hugging_face_token() -> str | None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            return None
        token = token.strip()
        return token if token else None

    def _cache_key(self, filters: ModelSearchFilters, cursor: str | None) -> str:
        parts = [
            filters.query or "",
            filters.library.value,
            filters.sort_by.value,
            str(filters.limit),
            filters.author or "",
            filters.quantization.value if filters.quantization else "",
            cursor or "",
        ]
        return "|".join(parts)

    def _prune_cache(self) -> None:
        now = time.time()
        self._cache = {key: value for key, value in self._cache.items() if value.expires_at > now}
        if len(self._cache) > 50:
            oldest = sorted(self._cache.items(), key=lambda item: item[1].expires_at)
            for key, _value in oldest[: len(self._cache) - 40]:
                self._cache.pop(key, None)


@dataclass(frozen=True)
class _CachedPage:
    page: ModelSearchPage
    expires_at: float


def _available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (ValueError, AttributeError, OSError):
        return 0
