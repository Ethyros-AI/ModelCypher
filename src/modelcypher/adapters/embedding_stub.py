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

"""Deterministic embedding provider for test/CI environments.

Encodes text as normalized byte-frequency vectors (256-dim). This preserves
character-level structure without introducing heuristic similarity scores.
"""

from __future__ import annotations

from modelcypher.ports.embedding import EmbeddingProvider


class ByteFrequencyEmbeddingProvider(EmbeddingProvider):
    """Embeds text as normalized byte-frequency vectors."""

    _dimension = 256

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            if all(ord(ch) < 256 for ch in text):
                data = text.encode("latin-1", errors="replace")
            else:
                data = text.encode("utf-8", errors="replace")
            counts = [0.0] * self._dimension
            for byte in data:
                counts[byte] += 1.0
            total = float(len(data)) if data else 1.0
            embeddings.append([value / total for value in counts])
        return embeddings
