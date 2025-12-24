from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.model_search import ModelSearchService as ModelSearchPort

from modelcypher.core.domain.model_search import ModelSearchFilters, ModelSearchPage


class ModelSearchService:
    def __init__(self, adapter: "ModelSearchPort") -> None:
        """Initialize ModelSearchService with required dependencies.

        Args:
            adapter: Model search port implementation (REQUIRED).
        """
        self._adapter = adapter

    def search(self, filters: ModelSearchFilters, cursor: str | None = None) -> ModelSearchPage:
        return self._adapter.search_models(filters, cursor)

    def clear_cache(self) -> None:
        self._adapter.clear_cache()
