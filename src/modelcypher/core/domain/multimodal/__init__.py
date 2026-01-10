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

"""Multi-modal domain components.

Provides embedding extraction and alignment for cross-modal knowledge transfer.

Key insight: All modalities encode the same conceptual shapes. CLIP, Whisper,
and LLMs all discover the same high-dimensional geometry because they all
represent knowledge itself. The difference is coordinate system, not content.

Components:
    - MultiModalEmbeddingExtractor: Extract embeddings from LLM, CLIP, Whisper
    - MultiModalChannelAdapter: Bridge encoders with multi-channel merge pipeline
    - OfframpProjection: Bidirectional projection for modality access
"""

from modelcypher.core.domain.multimodal.embedding_extractor import (
    ModalityType,
    MultiModalEmbeddingExtractor,
    ModalityEmbeddings,
)
from modelcypher.core.domain.multimodal.channel_adapter import (
    MultiModalChannelAdapter,
    MultiModalOfframpResult,
    OfframpProjection,
)

__all__ = [
    "ModalityType",
    "MultiModalEmbeddingExtractor",
    "ModalityEmbeddings",
    "MultiModalChannelAdapter",
    "MultiModalOfframpResult",
    "OfframpProjection",
]
