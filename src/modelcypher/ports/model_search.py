from __future__ import annotations

from typing import Protocol

from modelcypher.core.domain.model_search import ModelSearchFilters, ModelSearchPage


class ModelSearchService(Protocol):
    def search_models(self, filters: ModelSearchFilters, cursor: str | None = None) -> ModelSearchPage: ...
    def clear_cache(self) -> None: ...
