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

"""Multi-modal embedding extractor using Backend protocol.

Extracts embeddings from text, vision, and audio encoders for cross-modal
alignment and transfer workflows. NO framework imports - uses Backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

from modelcypher.core.domain.multimodal.types import ModalityEmbeddings, ModalityType

logger = logging.getLogger(__name__)


class MultiModalEmbeddingExtractor:
    """Extract embeddings from different modality models.

    This class provides a unified interface for extracting embeddings from:
    - LLMs (text modality) - uses semantic highway layers
    - CLIP (vision modality) - uses text encoder for concept descriptions
    - Whisper (audio modality) - uses decoder embeddings for audio concepts

    Example:
        >>> extractor = MultiModalEmbeddingExtractor()
        >>> concepts = ["a red ball", "a blue sky", "music playing"]
        >>> lfm2_embeds = extractor.extract_llm(model_path, concepts)
        >>> clip_embeds = extractor.extract_clip(concepts)
        >>> whisper_embeds = extractor.extract_whisper(concepts)
    """

    def __init__(self, backend: "Backend | None" = None):
        """Initialize the extractor.

        Args:
            backend: Optional backend instance. If None, uses default.
        """
        if backend is None:
            from modelcypher.core.domain._backend import get_default_backend

            backend = get_default_backend()
        self._backend = backend

    def extract_llm(
        self,
        model_path: str,
        concepts: list[str],
        highway_layers: tuple[int, int, int] = (7, 8, 9),
    ) -> ModalityEmbeddings:
        """Extract embeddings from an LLM's semantic highway.

        Args:
            model_path: Path to the model directory.
            concepts: List of concept strings to embed.
            highway_layers: Tuple of layer indices to average across.

        Returns:
            ModalityEmbeddings with shape [n_concepts, hidden_dim].
        """
        logger.info(f"Loading LLM from {model_path}")
        model, tokenizer = self._backend.load_model(model_path)

        all_embeds = []
        for concept in concepts:
            tokens = self._backend.encode_tokens(tokenizer, concept)
            input_ids = self._backend.array([tokens])

            # Get base model
            base = getattr(model, "model", model)
            if hasattr(base, "embed_tokens"):
                hidden = base.embed_tokens(input_ids)
                highway_states = []

                if hasattr(base, "layers"):
                    for i, layer in enumerate(base.layers):
                        result = layer(hidden)
                        hidden = result[0] if isinstance(result, tuple) else result
                        if i in highway_layers:
                            highway_states.append(hidden)

                if highway_states:
                    highway_avg = self._backend.mean(
                        self._backend.stack(highway_states, axis=0), axis=0
                    )
                else:
                    highway_avg = hidden

                self._backend.eval(highway_avg)
                pooled = self._backend.mean(highway_avg, axis=1)
                all_embeds.append(pooled)

        embeddings = self._backend.concatenate(all_embeds, axis=0)
        self._backend.eval(embeddings)

        return ModalityEmbeddings(
            modality=ModalityType.TEXT,
            embeddings=embeddings,
            concepts=tuple(concepts),
            hidden_dim=int(self._backend.shape(embeddings)[1]),
            model_name=model_path,
        )

    def extract_clip(
        self,
        concepts: list[str],
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> ModalityEmbeddings:
        """Extract embeddings from CLIP's text encoder.

        Note: CLIP uses transformers. The embeddings are converted to backend
        arrays after extraction.

        Args:
            concepts: List of concept strings to embed.
            model_name: HuggingFace model identifier for CLIP.

        Returns:
            ModalityEmbeddings with shape [n_concepts, 512].
        """
        # CLIP extraction requires transformers - import locally
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers required for CLIP extraction. "
                "Install with: pip install transformers"
            ) from exc

        logger.info(f"Loading CLIP from {model_name}")
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        inputs = processor(text=concepts, return_tensors="pt", padding=True)

        # Extract features (transformers handles device placement)
        outputs = model.get_text_features(**inputs)

        # Convert to backend array via list (avoids numpy)
        outputs_list = outputs.detach().cpu().tolist()
        embeddings = self._backend.array(outputs_list)

        return ModalityEmbeddings(
            modality=ModalityType.VISION,
            embeddings=embeddings,
            concepts=tuple(concepts),
            hidden_dim=int(self._backend.shape(embeddings)[1]),
            model_name=model_name,
        )

    def extract_whisper(
        self,
        concepts: list[str],
        model_name: str = "openai/whisper-base",
    ) -> ModalityEmbeddings:
        """Extract embeddings from Whisper's decoder.

        Note: This uses text tokenization as a proxy for audio concepts. In
        production, actual audio embeddings should be extracted.

        Args:
            concepts: List of concept strings to embed.
            model_name: HuggingFace model identifier for Whisper.

        Returns:
            ModalityEmbeddings with shape [n_concepts, 512].
        """
        # Whisper extraction requires transformers - import locally
        try:
            from transformers import WhisperModel, WhisperProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers required for Whisper extraction. "
                "Install with: pip install transformers"
            ) from exc

        logger.info(f"Loading Whisper from {model_name}")
        model = WhisperModel.from_pretrained(model_name)
        processor = WhisperProcessor.from_pretrained(model_name)

        tokenizer = processor.tokenizer
        all_embeds = []

        for concept in concepts:
            tokens = tokenizer(concept, return_tensors="pt").input_ids
            embed_layer = model.decoder.embed_tokens
            embeds = embed_layer(tokens)
            pooled = embeds.mean(dim=1)
            all_embeds.append(pooled.detach().cpu().tolist()[0])

        embeddings = self._backend.array(all_embeds)

        return ModalityEmbeddings(
            modality=ModalityType.AUDIO,
            embeddings=embeddings,
            concepts=tuple(concepts),
            hidden_dim=int(self._backend.shape(embeddings)[1]),
            model_name=model_name,
        )
