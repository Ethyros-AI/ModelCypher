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

"""Model architecture port for hexagonal architecture.

This port abstracts model architecture access, eliminating string-based
heuristics for component detection (e.g., "lm_head", "self_attn.q_proj").

Different model families use different naming conventions:
- LLaMA: model.model.embed_tokens, model.model.layers, self_attn.q_proj
- GPT-2: model.wte, model.h, attn.c_attn (fused QKV)
- BERT: embeddings.word_embeddings, encoder.layer, attention.self.query
- LFM: model.embed_tokens, model.layers (similar to LLaMA but flat)

This protocol provides a unified interface regardless of naming.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    pass


@runtime_checkable
class ModelArchitecturePort(Protocol):
    """Normalized access to model components regardless of naming convention.

    Implementations wrap specific model families (LLaMA, GPT-2, BERT, etc.)
    and provide consistent access to common components.
    """

    @property
    def is_causal(self) -> bool:
        """True for decoder-only/causal LMs, False for encoder/bidirectional.

        Used to determine appropriate pooling strategy:
        - Causal models: last-token pooling (optimized for next-token prediction)
        - Bidirectional: mean pooling (all positions see full context)
        """
        ...

    @property
    def model_type(self) -> str:
        """The model_type string from config.json (e.g., 'llama', 'qwen2', 'gpt2')."""
        ...

    @property
    def embed_module(self) -> Any:
        """Token embedding module.

        Returns the module that converts token IDs to embeddings:
        - LLaMA: model.model.embed_tokens
        - GPT-2: model.wte
        - BERT: model.embeddings.word_embeddings
        """
        ...

    @property
    def layers(self) -> Sequence[Any]:
        """Transformer layers as a sequence.

        Returns the list/ModuleList of transformer blocks:
        - LLaMA: model.model.layers
        - GPT-2: model.h
        - BERT: model.encoder.layer
        """
        ...

    @property
    def norm(self) -> Any | None:
        """Final layer normalization, or None if not present.

        - LLaMA: model.model.norm
        - GPT-2: model.ln_f
        - BERT: None (uses layer norm within each block)
        """
        ...

    @property
    def output_projection(self) -> Any | None:
        """LM head / output projection layer, or None if not present.

        - LLaMA: model.lm_head
        - GPT-2: model.lm_head (or tied to embeddings)
        - BERT: model.cls.predictions.decoder (for MLM)
        """
        ...

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        ...

    def output_projection_key(self) -> str | None:
        """Weight key for output projection (e.g., 'lm_head.weight').

        Returns None if model has no output projection.
        """
        ...

    def layer_attention_keys(self, layer_idx: int) -> list[str]:
        """Weight keys for attention projections in this layer.

        Returns keys like:
        - LLaMA: ['model.layers.0.self_attn.q_proj.weight', ...]
        - GPT-2: ['h.0.attn.c_attn.weight'] (fused QKV)
        """
        ...

    def layer_mlp_keys(self, layer_idx: int) -> list[str]:
        """Weight keys for MLP/FFN in this layer.

        Returns keys like:
        - LLaMA: ['model.layers.0.mlp.gate_proj.weight', ...]
        - GPT-2: ['h.0.mlp.c_fc.weight', 'h.0.mlp.c_proj.weight']
        """
        ...


__all__ = ["ModelArchitecturePort"]
