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

"""Factory for multimodal embedding extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.multimodal import MultiModalEmbeddingPort


def get_multimodal_embedding_extractor() -> "MultiModalEmbeddingPort":
    """Get the multimodal embedding extractor.

    Returns:
        MultiModalEmbeddingExtractor instance.
    """
    from modelcypher.adapters.multimodal_embedding_extractor import MultiModalEmbeddingExtractor

    return MultiModalEmbeddingExtractor()


__all__ = ["get_multimodal_embedding_extractor"]
