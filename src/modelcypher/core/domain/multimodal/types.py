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

"""Multi-modal shared types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


class ModalityType(Enum):
    """Supported modality types."""

    TEXT = "text"  # LLM text embeddings
    VISION = "vision"  # CLIP-style vision encoder
    AUDIO = "audio"  # Whisper-style audio encoder


@dataclass(frozen=True)
class ModalityEmbeddings:
    """Embeddings from a single modality.

    Attributes:
        modality: The type of modality these embeddings come from.
        embeddings: Shape [n_concepts, hidden_dim] embedding matrix.
        concepts: List of concept strings that were embedded.
        hidden_dim: Dimensionality of the embedding space.
        model_name: Name/path of the model used.
    """

    modality: ModalityType
    embeddings: "Backend.Array"  # type: ignore
    concepts: tuple[str, ...]
    hidden_dim: int
    model_name: str


__all__ = ["ModalityType", "ModalityEmbeddings"]
