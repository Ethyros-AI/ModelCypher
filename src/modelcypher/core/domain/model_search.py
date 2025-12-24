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

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


_PARAMETER_TAG_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb]")


class ModelSearchLibraryFilter(str, Enum):
    mlx = "mlx"
    safetensors = "safetensors"
    pytorch = "pytorch"
    any = "any"


class ModelSearchQuantization(str, Enum):
    four_bit = "4bit"
    eight_bit = "8bit"
    any = "any"


class ModelSearchSortOption(str, Enum):
    downloads = "downloads"
    likes = "likes"
    last_modified = "lastModified"
    trending = "trending"


class MemoryFitStatus(str, Enum):
    fits = "fits"
    tight = "tight"
    too_big = "tooBig"
    unknown = "unknown"


@dataclass(frozen=True)
class ModelSearchFilters:
    query: str | None = None
    architecture: str | None = None
    max_size_gb: float | None = None
    author: str | None = None
    library: ModelSearchLibraryFilter = ModelSearchLibraryFilter.mlx
    quantization: ModelSearchQuantization | None = None
    sort_by: ModelSearchSortOption = ModelSearchSortOption.downloads
    limit: int = 20

    def __post_init__(self) -> None:
        normalized = max(1, min(self.limit, 100))
        object.__setattr__(self, "limit", normalized)


@dataclass(frozen=True)
class ModelSearchResult:
    id: str
    downloads: int
    likes: int
    tags: list[str]
    author: str | None
    pipeline_tag: str | None
    last_modified: datetime | None
    is_private: bool
    is_gated: bool
    memory_fit_status: MemoryFitStatus | None = None

    @property
    def model_id(self) -> str:
        return self.id

    @property
    def display_name(self) -> str:
        return self.id.split("/")[-1] if self.id else self.id

    @property
    def estimated_size_gb(self) -> float | None:
        for tag in self.tags:
            size = _parse_parameter_size(tag)
            if size is not None:
                return size * 2.0
        return None

    @property
    def is_recommended(self) -> bool:
        return "mlx" in self.tags and ("4bit" in self.tags or "8bit" in self.tags or "quantized" in self.tags)


@dataclass(frozen=True)
class ModelSearchPage:
    models: list[ModelSearchResult]
    next_cursor: str | None

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None


class ModelSearchError(Exception):
    def __init__(self, kind: str, detail: str, retry_after: float | None = None) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.retry_after = retry_after

    @classmethod
    def network_error(cls, message: str) -> ModelSearchError:
        return cls("network", message)

    @classmethod
    def invalid_response(cls, message: str) -> ModelSearchError:
        return cls("invalid_response", message)

    @classmethod
    def rate_limited(cls, retry_after: float | None) -> ModelSearchError:
        return cls("rate_limited", "Rate limited", retry_after=retry_after)

    @classmethod
    def authentication_required(cls) -> ModelSearchError:
        return cls("authentication_required", "Authentication required")

    @classmethod
    def search_failed(cls, message: str) -> ModelSearchError:
        return cls("search_failed", message)

    def __str__(self) -> str:
        if self.kind == "network":
            return f"Network error: {self.detail}"
        if self.kind == "invalid_response":
            return f"Invalid response: {self.detail}"
        if self.kind == "rate_limited":
            if self.retry_after is not None:
                return f"Rate limited. Retry after {int(self.retry_after)} seconds."
            return "Rate limited. Please try again later."
        if self.kind == "authentication_required":
            return "Authentication required. Set HF_TOKEN environment variable."
        if self.kind == "search_failed":
            return f"Search failed: {self.detail}"
        return self.detail


def _parse_parameter_size(tag: str) -> float | None:
    match = _PARAMETER_TAG_RE.search(tag)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
