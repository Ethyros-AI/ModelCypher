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
