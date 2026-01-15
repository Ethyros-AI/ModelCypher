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

from typing import Protocol, runtime_checkable

from modelcypher.core.domain.multimodal.types import ModalityEmbeddings


@runtime_checkable
class MultiModalEmbeddingPort(Protocol):
    """Port for extracting modality embeddings."""

    def extract_llm(
        self,
        model_path: str,
        concepts: list[str],
        highway_layers: tuple[int, int, int] = (7, 8, 9),
    ) -> ModalityEmbeddings: ...

    def extract_clip(
        self,
        concepts: list[str],
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> ModalityEmbeddings: ...

    def extract_whisper(
        self,
        concepts: list[str],
        model_name: str = "openai/whisper-base",
    ) -> ModalityEmbeddings: ...


__all__ = ["MultiModalEmbeddingPort"]
