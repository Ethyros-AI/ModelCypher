from __future__ import annotations

from modelcypher.adapters.hf_model_search import HfModelSearchAdapter
from modelcypher.core.domain.model_search import ModelSearchFilters, ModelSearchPage
from modelcypher.ports.model_search import ModelSearchService as ModelSearchPort


class ModelSearchService:
    def __init__(self, adapter: ModelSearchPort | None = None) -> None:
        self._adapter = adapter or HfModelSearchAdapter()

    def search(self, filters: ModelSearchFilters, cursor: str | None = None) -> ModelSearchPage:
        return self._adapter.search_models(filters, cursor)

    def clear_cache(self) -> None:
        self._adapter.clear_cache()
