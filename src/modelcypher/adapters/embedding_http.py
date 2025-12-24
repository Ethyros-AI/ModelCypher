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

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from modelcypher.ports.embedding import EmbeddingProvider


class HTTPEmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True)
class HTTPEmbeddingConfig:
    base_url: str
    model_name: str = "nomic-embed-code"
    dimension: int = 3584
    timeout_seconds: int = 60


class HTTPEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: HTTPEmbeddingConfig) -> None:
        self._config = config

    @property
    def dimension(self) -> int:
        return self._config.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        endpoint = self._normalize_base_url(self._config.base_url) + "/v1/embeddings"
        payload = {"input": texts, "model": self._config.model_name}
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as response:
                status = response.getcode()
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise HTTPEmbeddingError(f"Embeddings API error: {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise HTTPEmbeddingError(f"Embeddings API unavailable: {exc.reason}") from exc

        if status != 200:
            raise HTTPEmbeddingError(f"Embeddings API error: {status}")

        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPEmbeddingError("Embeddings API returned invalid JSON") from exc

        data_list = body.get("data")
        if not isinstance(data_list, list):
            raise HTTPEmbeddingError("Embeddings API response missing data array")

        embeddings: list[list[float]] = []
        for item in data_list:
            if not isinstance(item, dict):
                raise HTTPEmbeddingError("Embeddings API response item invalid")
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise HTTPEmbeddingError("Embeddings API response missing embedding list")
            embeddings.append([float(value) for value in vec])
        return embeddings

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        trimmed = base_url.rstrip("/")
        if trimmed.startswith("http://") or trimmed.startswith("https://"):
            return trimmed
        return "http://" + trimmed
