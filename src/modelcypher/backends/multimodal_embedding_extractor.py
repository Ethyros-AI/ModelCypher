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

"""Multi-modal embedding extractor.

Extracts embeddings from text, vision, and audio encoders for cross-modal
alignment and transfer workflows.
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

    def _as_backend_array(self, array: Any) -> "Backend.Array":  # type: ignore
        """Convert array-like inputs to backend arrays without forced list copies."""
        try:
            return self._backend.array(array)
        except Exception:
            if hasattr(array, "tolist"):
                return self._backend.array(array.tolist())
            raise

    def extract_llm(
        self,
        model_path: str,
        concepts: list[str],
        highway_layers: tuple[int, int, int] = (7, 8, 9),
    ) -> ModalityEmbeddings:
        """Extract embeddings from an LLM's semantic highway.

        Args:
            model_path: Path to the MLX model directory.
            concepts: List of concept strings to embed.
            highway_layers: Tuple of layer indices to average across.

        Returns:
            ModalityEmbeddings with shape [n_concepts, hidden_dim].
        """
        from mlx_lm import load

        import mlx.core as mx

        logger.info(f"Loading LLM from {model_path}")
        model, tokenizer = load(model_path)

        all_embeds = []
        for concept in concepts:
            tokens = mx.array([tokenizer.encode(concept)])

            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                hidden = model.model.embed_tokens(tokens)
                highway_states = []

                if hasattr(model.model, "layers"):
                    for i, layer in enumerate(model.model.layers):
                        hidden = layer(hidden)
                        if i in highway_layers:
                            highway_states.append(hidden)

                if highway_states:
                    highway_avg = mx.mean(mx.stack(highway_states, axis=0), axis=0)
                else:
                    highway_avg = hidden

                mx.eval(highway_avg)
                pooled = mx.mean(highway_avg, axis=1)
                all_embeds.append(pooled)

        embeddings = mx.concatenate(all_embeds, axis=0)
        mx.eval(embeddings)

        # Convert to backend array
        embeddings_backend = self._as_backend_array(embeddings)

        return ModalityEmbeddings(
            modality=ModalityType.TEXT,
            embeddings=embeddings_backend,
            concepts=tuple(concepts),
            hidden_dim=int(embeddings.shape[1]),
            model_name=model_path,
        )

    def extract_clip(
        self,
        concepts: list[str],
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> ModalityEmbeddings:
        """Extract embeddings from CLIP's text encoder.

        Args:
            concepts: List of concept strings to embed.
            model_name: HuggingFace model identifier for CLIP.

        Returns:
            ModalityEmbeddings with shape [n_concepts, 512].
        """
        from transformers import CLIPModel, CLIPProcessor
        import torch

        logger.info(f"Loading CLIP from {model_name}")
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        inputs = processor(text=concepts, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        # Convert to backend array
        outputs_np = outputs.detach().cpu().numpy()
        embeddings_backend = self._as_backend_array(outputs_np)

        return ModalityEmbeddings(
            modality=ModalityType.VISION,
            embeddings=embeddings_backend,
            concepts=tuple(concepts),
            hidden_dim=int(outputs.shape[1]),
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
        from transformers import WhisperModel, WhisperProcessor
        import torch

        logger.info(f"Loading Whisper from {model_name}")
        model = WhisperModel.from_pretrained(model_name)
        processor = WhisperProcessor.from_pretrained(model_name)

        tokenizer = processor.tokenizer
        all_embeds = []

        for concept in concepts:
            tokens = tokenizer(concept, return_tensors="pt").input_ids

            with torch.no_grad():
                embed_layer = model.decoder.embed_tokens
                embeds = embed_layer(tokens)
                pooled = embeds.mean(dim=1)
                all_embeds.append(pooled)

        embeddings = torch.cat(all_embeds, dim=0)

        # Convert to backend array
        embeddings_np = embeddings.detach().cpu().numpy()
        embeddings_backend = self._as_backend_array(embeddings_np)

        return ModalityEmbeddings(
            modality=ModalityType.AUDIO,
            embeddings=embeddings_backend,
            concepts=tuple(concepts),
            hidden_dim=int(embeddings.shape[1]),
            model_name=model_name,
        )
