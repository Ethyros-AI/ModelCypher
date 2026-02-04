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

from modelcypher.adapters.backend_embedding_provider import get_embedding_provider
from modelcypher.ports.embedding import EmbeddingProvider


def make_default_embedder() -> EmbeddingProvider | None:
    """Get the default embedding provider (backend-based)."""
    try:
        return get_embedding_provider()
    except Exception:
        return None


class EmbeddingDefaults:
    """Default embedding configuration."""

    EMBEDDING_API_URL_ENV = "MC_EMBEDDING_API_URL"

    @classmethod
    def resolved_source(cls) -> tuple[str, str | None]:
        """Return embedding source and optional URL."""
        return ("backend", None)

    @classmethod
    def make_default_embedder(cls) -> EmbeddingProvider | None:
        """Get default embedder."""
        return make_default_embedder()
