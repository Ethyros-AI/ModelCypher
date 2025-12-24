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

import os
from urllib.parse import urlparse

from modelcypher.adapters.embedding_http import HTTPEmbeddingConfig, HTTPEmbeddingProvider
from modelcypher.adapters.embedding_mlx import MLXEmbeddingConfig, MLXEmbeddingProvider, MLXEmbeddingError
from modelcypher.ports.embedding import EmbeddingProvider


class EmbeddingDefaults:
    EMBEDDING_API_URL_ENV = "TC_EMBEDDING_API_URL"

    @staticmethod
    def resolved_source(environment: dict[str, str] | None = None) -> tuple[str, str | None]:
        env = environment or os.environ
        raw_url = (env.get(EmbeddingDefaults.EMBEDDING_API_URL_ENV) or "").strip()
        if not raw_url:
            return ("mlx", None)
        parsed = urlparse(raw_url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ("mlx", None)
        return ("http", raw_url)

    @staticmethod
    def make_default_embedder(environment: dict[str, str] | None = None) -> EmbeddingProvider | None:
        source, value = EmbeddingDefaults.resolved_source(environment)
        if source == "http" and value:
            return HTTPEmbeddingProvider(HTTPEmbeddingConfig(base_url=value))
        try:
            return MLXEmbeddingProvider(MLXEmbeddingConfig())
        except MLXEmbeddingError:
            return None
