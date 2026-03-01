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

"""Model architecture implementations.

Provides concrete implementations of ModelArchitecturePort for different
model families, eliminating string-based heuristics for component detection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from modelcypher.ports.model_architecture import LayerAccessorPort, ModelArchitecturePort

logger = logging.getLogger(__name__)

# Map model_type from config.json to architecture family
ARCHITECTURE_FAMILIES: dict[str, str] = {
    # LLaMA-family (nested model.model or flat model with embed_tokens)
    "llama": "llama",
    "qwen": "llama",
    "qwen2": "llama",
    "qwen3": "llama",
    "qwen3_5": "llama",
    "qwen3_5_text": "llama",
    "mistral": "llama",
    "deepseek": "llama",
    "deepseek_v2": "llama",
    "deepseek_v3": "llama",
    "internlm": "llama",
    "internlm2": "llama",
    "baichuan": "llama",
    "yi": "llama",
    "codellama": "llama",
    # LFM (Liquid Foundation Models) - similar to LLaMA but may differ
    "lfm": "lfm",
    "lfm2": "lfm",
    # GPT-2 family
    "gpt2": "gpt2",
    "gpt_neox": "gpt_neox",
    "gpt_neo": "gpt_neox",
    # Falcon
    "falcon": "falcon",
    # BERT family (encoder-only)
    "bert": "bert",
    "roberta": "bert",
    "distilbert": "bert",
    # T5 family (encoder-decoder)
    "t5": "t5",
    "mt5": "t5",
    # Phi family
    "phi": "phi",
    "phi3": "phi",
    "phi4": "phi",
    # Gemma family
    "gemma": "gemma",
    "gemma2": "gemma",
    # StableLM
    "stablelm": "llama",
    # Mamba (SSM, no attention)
    "mamba": "mamba",
}

# Causal (decoder-only) model types
CAUSAL_MODEL_TYPES: frozenset[str] = frozenset({
    "llama", "lfm", "lfm2", "gpt2", "gpt_neox", "falcon", "phi", "phi3", "phi4",
    "gemma", "gemma2", "qwen", "qwen2", "qwen3", "qwen3_5", "qwen3_5_text", "mistral", "deepseek",
    "deepseek_v2", "deepseek_v3", "internlm", "internlm2", "baichuan", "yi",
    "codellama", "stablelm", "mamba", "gpt_neo",
})


# ==============================================================================
# Layer Accessor Implementations
# ==============================================================================


class LlamaLayerAccessor:
    """Layer accessor for LLaMA-family models.

    Component names:
    - input_layernorm → pre-attention norm
    - self_attn → attention
    - post_attention_layernorm → post-attention norm
    - mlp → feed-forward network
    """

    def __init__(self, layer: Any, layer_idx: int) -> None:
        self._layer = layer
        self._layer_idx = layer_idx

    @property
    def layer_idx(self) -> int:
        return self._layer_idx

    @property
    def input_norm(self) -> Any | None:
        return getattr(self._layer, "input_layernorm", None)

    @property
    def attention(self) -> Any | None:
        return getattr(self._layer, "self_attn", None)

    @property
    def post_attn_norm(self) -> Any | None:
        return getattr(self._layer, "post_attention_layernorm", None)

    @property
    def mlp(self) -> Any | None:
        return getattr(self._layer, "mlp", None)

    @property
    def is_ssm_layer(self) -> bool:
        return False


class LFMLayerAccessor:
    """Layer accessor for LFM (Liquid Foundation Models).

    LFM uses a hybrid architecture with attention and SSM blocks:
    - Attention layers: operator_norm, self_attn, ffn_norm, feed_forward
    - SSM/Conv layers: operator_norm, conv, ffn_norm, feed_forward

    Component names:
    - operator_norm → pre-attention/conv norm
    - self_attn or conv → attention or SSM
    - ffn_norm → post-attention norm
    - feed_forward → MLP
    """

    def __init__(self, layer: Any, layer_idx: int) -> None:
        self._layer = layer
        self._layer_idx = layer_idx

    @property
    def layer_idx(self) -> int:
        return self._layer_idx

    @property
    def input_norm(self) -> Any | None:
        # LFM uses operator_norm, fallback to input_layernorm
        return getattr(self._layer, "operator_norm", None) or getattr(
            self._layer, "input_layernorm", None
        )

    @property
    def attention(self) -> Any | None:
        # LFM hybrid: try self_attn first, then conv for SSM layers
        attn = getattr(self._layer, "self_attn", None)
        if attn is not None:
            return attn
        return getattr(self._layer, "conv", None)

    @property
    def post_attn_norm(self) -> Any | None:
        # LFM uses ffn_norm, fallback to post_attention_layernorm
        return getattr(self._layer, "ffn_norm", None) or getattr(
            self._layer, "post_attention_layernorm", None
        )

    @property
    def mlp(self) -> Any | None:
        # LFM uses feed_forward, fallback to mlp
        return getattr(self._layer, "feed_forward", None) or getattr(
            self._layer, "mlp", None
        )

    @property
    def is_ssm_layer(self) -> bool:
        """True if this layer uses conv/SSM instead of attention."""
        return hasattr(self._layer, "conv") and not hasattr(self._layer, "self_attn")


class GPT2LayerAccessor:
    """Layer accessor for GPT-2 family models.

    Component names:
    - ln_1 → pre-attention norm
    - attn → attention
    - ln_2 → post-attention norm
    - mlp → feed-forward network
    """

    def __init__(self, layer: Any, layer_idx: int) -> None:
        self._layer = layer
        self._layer_idx = layer_idx

    @property
    def layer_idx(self) -> int:
        return self._layer_idx

    @property
    def input_norm(self) -> Any | None:
        return getattr(self._layer, "ln_1", None)

    @property
    def attention(self) -> Any | None:
        return getattr(self._layer, "attn", None)

    @property
    def post_attn_norm(self) -> Any | None:
        return getattr(self._layer, "ln_2", None)

    @property
    def mlp(self) -> Any | None:
        return getattr(self._layer, "mlp", None)

    @property
    def is_ssm_layer(self) -> bool:
        return False


class GPTNeoXLayerAccessor:
    """Layer accessor for GPT-NeoX family models.

    Component names:
    - input_layernorm → pre-attention norm
    - attention → attention
    - post_attention_layernorm → post-attention norm
    - mlp → feed-forward network
    """

    def __init__(self, layer: Any, layer_idx: int) -> None:
        self._layer = layer
        self._layer_idx = layer_idx

    @property
    def layer_idx(self) -> int:
        return self._layer_idx

    @property
    def input_norm(self) -> Any | None:
        return getattr(self._layer, "input_layernorm", None)

    @property
    def attention(self) -> Any | None:
        return getattr(self._layer, "attention", None)

    @property
    def post_attn_norm(self) -> Any | None:
        return getattr(self._layer, "post_attention_layernorm", None)

    @property
    def mlp(self) -> Any | None:
        return getattr(self._layer, "mlp", None)

    @property
    def is_ssm_layer(self) -> bool:
        return False


class BERTLayerAccessor:
    """Layer accessor for BERT family models.

    BERT uses a different structure:
    - attention.self → attention
    - attention.output → attention output projection + norm
    - intermediate → MLP up-projection
    - output → MLP down-projection + norm
    """

    def __init__(self, layer: Any, layer_idx: int) -> None:
        self._layer = layer
        self._layer_idx = layer_idx

    @property
    def layer_idx(self) -> int:
        return self._layer_idx

    @property
    def input_norm(self) -> Any | None:
        # BERT doesn't have pre-attention norm in the same sense
        return None

    @property
    def attention(self) -> Any | None:
        if hasattr(self._layer, "attention"):
            return self._layer.attention.self
        return None

    @property
    def post_attn_norm(self) -> Any | None:
        # BERT uses layer norm in attention.output
        if hasattr(self._layer, "attention") and hasattr(self._layer.attention, "output"):
            return getattr(self._layer.attention.output, "LayerNorm", None)
        return None

    @property
    def mlp(self) -> Any | None:
        # BERT splits MLP into intermediate and output
        if hasattr(self._layer, "intermediate"):
            return self._layer.intermediate
        return None

    @property
    def is_ssm_layer(self) -> bool:
        return False


# ==============================================================================
# Architecture Implementations
# ==============================================================================


class LlamaArchitecture:
    """Architecture wrapper for LLaMA-family models.

    Handles models with structure:
    - model.model.embed_tokens (nested CausalLM wrapper)
    - model.model.layers
    - model.model.norm
    - model.lm_head

    Or flat structure:
    - model.embed_tokens
    - model.layers
    - model.norm
    """

    def __init__(self, model: Any, model_type: str = "llama") -> None:
        self._model = model
        self._model_type = model_type
        self._resolve_structure()

    def _resolve_structure(self) -> None:
        """Resolve whether model uses nested or flat structure."""
        # Nested: model.model.embed_tokens (HuggingFace CausalLM wrapper)
        if hasattr(self._model, "model") and hasattr(self._model.model, "embed_tokens"):
            self._base = self._model.model
            self._has_lm_head = hasattr(self._model, "lm_head")
        # Flat: model.embed_tokens
        elif hasattr(self._model, "embed_tokens") and hasattr(self._model, "layers"):
            self._base = self._model
            self._has_lm_head = hasattr(self._model, "lm_head")
        else:
            raise ValueError(
                f"Could not resolve LLaMA-family architecture for model type {self._model_type}. "
                f"Expected model.model.embed_tokens or model.embed_tokens."
            )

    @property
    def is_causal(self) -> bool:
        return True

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def embed_module(self) -> Any:
        return self._base.embed_tokens

    @property
    def layers(self) -> Sequence[Any]:
        return self._base.layers

    @property
    def norm(self) -> Any | None:
        return getattr(self._base, "norm", None)

    @property
    def output_projection(self) -> Any | None:
        if self._has_lm_head:
            return self._model.lm_head
        return None

    @property
    def num_layers(self) -> int:
        return len(self._base.layers)

    @property
    def is_moe(self) -> bool:
        if not self._base.layers:
            return False
        mlp = getattr(self._base.layers[0], "mlp", None)
        return bool(mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"))

    @property
    def num_experts(self) -> int | None:
        if not self.is_moe or not self._base.layers:
            return None
        mlp = getattr(self._base.layers[0], "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is None:
            return None
        try:
            return int(len(experts))
        except Exception:
            return None

    def output_projection_key(self) -> str | None:
        if self._has_lm_head:
            return "lm_head.weight"
        return None

    def router_gate_key(self, layer_idx: int) -> str | None:
        if not self.is_moe:
            return None
        return f"model.layers.{layer_idx}.mlp.gate.weight"

    def layer_attention_keys(self, layer_idx: int) -> list[str]:
        prefix = f"model.layers.{layer_idx}.self_attn"
        return [
            f"{prefix}.q_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
            f"{prefix}.o_proj.weight",
        ]

    def layer_mlp_keys(self, layer_idx: int) -> list[str]:
        prefix = f"model.layers.{layer_idx}.mlp"
        return [
            f"{prefix}.gate_proj.weight",
            f"{prefix}.up_proj.weight",
            f"{prefix}.down_proj.weight",
        ]

    def layer_expert_keys(self, layer_idx: int, expert_idx: int) -> list[str]:
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        return [
            f"{prefix}.gate_proj.weight",
            f"{prefix}.up_proj.weight",
            f"{prefix}.down_proj.weight",
        ]

    def layer_shared_expert_keys(self, layer_idx: int) -> list[str]:
        prefix = f"model.layers.{layer_idx}.mlp.shared_expert"
        return [
            f"{prefix}.gate_proj.weight",
            f"{prefix}.up_proj.weight",
            f"{prefix}.down_proj.weight",
        ]

    def layer_accessor(self, layer_idx: int) -> "LayerAccessorPort":
        """Get normalized accessor for layer components."""
        return LlamaLayerAccessor(self._base.layers[layer_idx], layer_idx)


class LFMArchitecture:
    """Architecture wrapper for LFM (Liquid Foundation Models).

    LFM uses a hybrid architecture with both attention and SSM blocks.
    Similar to LLaMA but may have different layer naming.
    """

    def __init__(self, model: Any, model_type: str = "lfm2") -> None:
        self._model = model
        self._model_type = model_type
        self._resolve_structure()

    def _resolve_structure(self) -> None:
        """Resolve LFM model structure."""
        # LFM typically uses flat structure: model.embed_tokens
        if hasattr(self._model, "model") and hasattr(self._model.model, "embed_tokens"):
            self._base = self._model.model
            self._has_lm_head = hasattr(self._model, "lm_head")
        elif hasattr(self._model, "embed_tokens") and hasattr(self._model, "layers"):
            self._base = self._model
            self._has_lm_head = hasattr(self._model, "lm_head")
        else:
            raise ValueError(
                f"Could not resolve LFM architecture for model type {self._model_type}. "
                f"Expected model.embed_tokens or model.model.embed_tokens."
            )

    @property
    def is_causal(self) -> bool:
        return True

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def embed_module(self) -> Any:
        return self._base.embed_tokens

    @property
    def layers(self) -> Sequence[Any]:
        return self._base.layers

    @property
    def norm(self) -> Any | None:
        return getattr(self._base, "norm", None)

    @property
    def output_projection(self) -> Any | None:
        if self._has_lm_head:
            return self._model.lm_head
        return None

    @property
    def num_layers(self) -> int:
        return len(self._base.layers)

    @property
    def is_moe(self) -> bool:
        if not self._base.layers:
            return False
        mlp = getattr(self._base.layers[0], "mlp", None) or getattr(
            self._base.layers[0], "feed_forward", None,
        )
        return bool(mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"))

    @property
    def num_experts(self) -> int | None:
        if not self.is_moe or not self._base.layers:
            return None
        mlp = getattr(self._base.layers[0], "mlp", None) or getattr(
            self._base.layers[0], "feed_forward", None,
        )
        experts = getattr(mlp, "experts", None)
        if experts is None:
            return None
        try:
            return int(len(experts))
        except Exception:
            return None

    def output_projection_key(self) -> str | None:
        if self._has_lm_head:
            return "lm_head.weight"
        return None

    def router_gate_key(self, layer_idx: int) -> str | None:
        if not self.is_moe:
            return None
        return f"model.layers.{layer_idx}.mlp.gate.weight"

    def layer_attention_keys(self, layer_idx: int) -> list[str]:
        # LFM may use different attention naming
        prefix = f"model.layers.{layer_idx}.self_attn"
        return [
            f"{prefix}.q_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
            f"{prefix}.o_proj.weight",
        ]

    def layer_mlp_keys(self, layer_idx: int) -> list[str]:
        # LFM uses feed_forward with w1/w2/w3 naming (LLaMA-internal style)
        # w1 = gate_proj (gating for SwiGLU)
        # w3 = up_proj (up projection)
        # w2 = down_proj (down projection)
        prefix = f"model.layers.{layer_idx}.feed_forward"
        return [
            f"{prefix}.w1.weight",  # gate_proj
            f"{prefix}.w3.weight",  # up_proj
            f"{prefix}.w2.weight",  # down_proj
        ]

    def layer_expert_keys(self, layer_idx: int, expert_idx: int) -> list[str]:
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        return [
            f"{prefix}.gate_proj.weight",
            f"{prefix}.up_proj.weight",
            f"{prefix}.down_proj.weight",
        ]

    def layer_shared_expert_keys(self, layer_idx: int) -> list[str]:
        prefix = f"model.layers.{layer_idx}.mlp.shared_expert"
        return [
            f"{prefix}.gate_proj.weight",
            f"{prefix}.up_proj.weight",
            f"{prefix}.down_proj.weight",
        ]

    def layer_accessor(self, layer_idx: int) -> "LayerAccessorPort":
        """Get normalized accessor for layer components."""
        return LFMLayerAccessor(self._base.layers[layer_idx], layer_idx)


class GPT2Architecture:
    """Architecture wrapper for GPT-2 family models.

    GPT-2 uses:
    - model.wte (word token embeddings)
    - model.wpe (positional embeddings)
    - model.h (transformer blocks)
    - model.ln_f (final layer norm)
    - lm_head (output projection, may be tied to wte)
    """

    def __init__(self, model: Any, model_type: str = "gpt2") -> None:
        self._model = model
        self._model_type = model_type
        self._resolve_structure()

    def _resolve_structure(self) -> None:
        """Resolve GPT-2 model structure."""
        # GPT-2 from HuggingFace: model.transformer.wte, model.lm_head
        if hasattr(self._model, "transformer") and hasattr(self._model.transformer, "wte"):
            self._base = self._model.transformer
            self._has_lm_head = hasattr(self._model, "lm_head")
        # Flat structure: model.wte
        elif hasattr(self._model, "wte") and hasattr(self._model, "h"):
            self._base = self._model
            self._has_lm_head = hasattr(self._model, "lm_head")
        else:
            raise ValueError(
                f"Could not resolve GPT-2 architecture for model type {self._model_type}. "
                f"Expected model.transformer.wte or model.wte."
            )

    @property
    def is_causal(self) -> bool:
        return True

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def embed_module(self) -> Any:
        return self._base.wte

    @property
    def layers(self) -> Sequence[Any]:
        return self._base.h

    @property
    def norm(self) -> Any | None:
        return getattr(self._base, "ln_f", None)

    @property
    def output_projection(self) -> Any | None:
        if self._has_lm_head:
            return self._model.lm_head
        return None

    @property
    def num_layers(self) -> int:
        return len(self._base.h)

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def num_experts(self) -> int | None:
        return None

    def output_projection_key(self) -> str | None:
        if self._has_lm_head:
            return "lm_head.weight"
        return None

    def router_gate_key(self, layer_idx: int) -> str | None:
        del layer_idx
        return None

    def layer_attention_keys(self, layer_idx: int) -> list[str]:
        # GPT-2 uses fused QKV (c_attn) and output (c_proj)
        prefix = f"h.{layer_idx}.attn"
        return [
            f"{prefix}.c_attn.weight",  # Fused Q, K, V
            f"{prefix}.c_proj.weight",  # Output projection
        ]

    def layer_mlp_keys(self, layer_idx: int) -> list[str]:
        prefix = f"h.{layer_idx}.mlp"
        return [
            f"{prefix}.c_fc.weight",
            f"{prefix}.c_proj.weight",
        ]

    def layer_expert_keys(self, layer_idx: int, expert_idx: int) -> list[str]:
        del layer_idx, expert_idx
        return []

    def layer_shared_expert_keys(self, layer_idx: int) -> list[str]:
        del layer_idx
        return []

    def layer_accessor(self, layer_idx: int) -> "LayerAccessorPort":
        """Get normalized accessor for layer components."""
        return GPT2LayerAccessor(self._base.h[layer_idx], layer_idx)


class GPTNeoXArchitecture:
    """Architecture wrapper for GPT-NeoX family models.

    Similar to GPT-2 but with different naming.
    """

    def __init__(self, model: Any, model_type: str = "gpt_neox") -> None:
        self._model = model
        self._model_type = model_type
        self._resolve_structure()

    def _resolve_structure(self) -> None:
        """Resolve GPT-NeoX model structure."""
        if hasattr(self._model, "gpt_neox"):
            self._base = self._model.gpt_neox
            self._has_lm_head = hasattr(self._model, "embed_out")
        elif hasattr(self._model, "embed_in") and hasattr(self._model, "layers"):
            self._base = self._model
            self._has_lm_head = hasattr(self._model, "embed_out")
        else:
            raise ValueError(
                f"Could not resolve GPT-NeoX architecture for model type {self._model_type}."
            )

    @property
    def is_causal(self) -> bool:
        return True

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def embed_module(self) -> Any:
        return self._base.embed_in

    @property
    def layers(self) -> Sequence[Any]:
        return self._base.layers

    @property
    def norm(self) -> Any | None:
        return getattr(self._base, "final_layer_norm", None)

    @property
    def output_projection(self) -> Any | None:
        if self._has_lm_head:
            return self._model.embed_out
        return None

    @property
    def num_layers(self) -> int:
        return len(self._base.layers)

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def num_experts(self) -> int | None:
        return None

    def output_projection_key(self) -> str | None:
        if self._has_lm_head:
            return "embed_out.weight"
        return None

    def router_gate_key(self, layer_idx: int) -> str | None:
        del layer_idx
        return None

    def layer_attention_keys(self, layer_idx: int) -> list[str]:
        prefix = f"gpt_neox.layers.{layer_idx}.attention"
        return [
            f"{prefix}.query_key_value.weight",  # Fused QKV
            f"{prefix}.dense.weight",
        ]

    def layer_mlp_keys(self, layer_idx: int) -> list[str]:
        prefix = f"gpt_neox.layers.{layer_idx}.mlp"
        return [
            f"{prefix}.dense_h_to_4h.weight",
            f"{prefix}.dense_4h_to_h.weight",
        ]

    def layer_expert_keys(self, layer_idx: int, expert_idx: int) -> list[str]:
        del layer_idx, expert_idx
        return []

    def layer_shared_expert_keys(self, layer_idx: int) -> list[str]:
        del layer_idx
        return []

    def layer_accessor(self, layer_idx: int) -> "LayerAccessorPort":
        """Get normalized accessor for layer components."""
        return GPTNeoXLayerAccessor(self._base.layers[layer_idx], layer_idx)


class BERTArchitecture:
    """Architecture wrapper for BERT-family models (encoder-only, bidirectional)."""

    def __init__(self, model: Any, model_type: str = "bert") -> None:
        self._model = model
        self._model_type = model_type
        self._resolve_structure()

    def _resolve_structure(self) -> None:
        """Resolve BERT model structure."""
        if hasattr(self._model, "bert"):
            self._base = self._model.bert
        elif hasattr(self._model, "embeddings") and hasattr(self._model, "encoder"):
            self._base = self._model
        else:
            raise ValueError(
                f"Could not resolve BERT architecture for model type {self._model_type}."
            )

    @property
    def is_causal(self) -> bool:
        return False  # BERT is bidirectional

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def embed_module(self) -> Any:
        return self._base.embeddings.word_embeddings

    @property
    def layers(self) -> Sequence[Any]:
        return self._base.encoder.layer

    @property
    def norm(self) -> Any | None:
        # BERT uses layer norm within blocks, no final norm
        return None

    @property
    def output_projection(self) -> Any | None:
        # BERT for MLM: cls.predictions.decoder
        if hasattr(self._model, "cls") and hasattr(self._model.cls, "predictions"):
            return self._model.cls.predictions.decoder
        return None

    @property
    def num_layers(self) -> int:
        return len(self._base.encoder.layer)

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def num_experts(self) -> int | None:
        return None

    def output_projection_key(self) -> str | None:
        if self.output_projection is not None:
            return "cls.predictions.decoder.weight"
        return None

    def router_gate_key(self, layer_idx: int) -> str | None:
        del layer_idx
        return None

    def layer_attention_keys(self, layer_idx: int) -> list[str]:
        prefix = f"bert.encoder.layer.{layer_idx}.attention.self"
        return [
            f"{prefix}.query.weight",
            f"{prefix}.key.weight",
            f"{prefix}.value.weight",
            f"bert.encoder.layer.{layer_idx}.attention.output.dense.weight",
        ]

    def layer_mlp_keys(self, layer_idx: int) -> list[str]:
        prefix = f"bert.encoder.layer.{layer_idx}"
        return [
            f"{prefix}.intermediate.dense.weight",
            f"{prefix}.output.dense.weight",
        ]

    def layer_expert_keys(self, layer_idx: int, expert_idx: int) -> list[str]:
        del layer_idx, expert_idx
        return []

    def layer_shared_expert_keys(self, layer_idx: int) -> list[str]:
        del layer_idx
        return []

    def layer_accessor(self, layer_idx: int) -> "LayerAccessorPort":
        """Get normalized accessor for layer components."""
        return BERTLayerAccessor(self._base.encoder.layer[layer_idx], layer_idx)


def load_config(model_path: str | Path) -> dict:
    """Load config.json from model directory."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        logger.warning("No config.json found at %s", config_path)
        return {}
    with open(config_path) as f:
        return json.load(f)


def get_model_architecture(
    model: Any,
    config: dict | None = None,
    model_path: str | Path | None = None,
) -> "ModelArchitecturePort":
    """Create architecture wrapper from model and config.

    Args:
        model: The loaded model instance
        config: Model config dict (from config.json). If None, loaded from model_path.
        model_path: Path to model directory (used to load config if not provided)

    Returns:
        ModelArchitecturePort implementation for this model family

    Raises:
        ValueError: If model architecture cannot be determined or is unsupported
    """
    if config is None and model_path is not None:
        config = load_config(model_path)
    if config is None:
        config = {}

    model_type = config.get("model_type", "").lower()

    # Try to infer from architectures list if model_type is missing
    if not model_type:
        architectures = config.get("architectures", [])
        if architectures:
            arch_name = architectures[0].lower()
            if "llama" in arch_name:
                model_type = "llama"
            elif "lfm" in arch_name:
                model_type = "lfm2"
            elif "qwen" in arch_name:
                model_type = "qwen2"
            elif "gpt2" in arch_name:
                model_type = "gpt2"
            elif "bert" in arch_name:
                model_type = "bert"

    arch_family = ARCHITECTURE_FAMILIES.get(model_type)

    if arch_family == "llama":
        return LlamaArchitecture(model, model_type)
    elif arch_family == "lfm":
        return LFMArchitecture(model, model_type)
    elif arch_family == "gpt2":
        return GPT2Architecture(model, model_type)
    elif arch_family == "gpt_neox":
        return GPTNeoXArchitecture(model, model_type)
    elif arch_family == "bert":
        return BERTArchitecture(model, model_type)
    elif arch_family in ("falcon", "phi", "gemma", "t5", "mamba"):
        # Fall back to LLaMA-style for similar architectures
        logger.warning(
            "Using LLaMA-style wrapper for %s (arch_family=%s). "
            "May need dedicated implementation.",
            model_type, arch_family
        )
        return LlamaArchitecture(model, model_type)
    else:
        # Try to auto-detect from model structure
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            logger.warning(
                "Unknown model_type '%s', using LLaMA-style wrapper based on structure.",
                model_type
            )
            return LlamaArchitecture(model, model_type or "unknown")
        elif hasattr(model, "embed_tokens") and hasattr(model, "layers"):
            logger.warning(
                "Unknown model_type '%s', using LFM-style wrapper based on structure.",
                model_type
            )
            return LFMArchitecture(model, model_type or "unknown")
        elif hasattr(model, "wte") or (hasattr(model, "transformer") and hasattr(model.transformer, "wte")):
            logger.warning(
                "Unknown model_type '%s', using GPT-2-style wrapper based on structure.",
                model_type
            )
            return GPT2Architecture(model, model_type or "unknown")
        else:
            raise ValueError(
                f"Unknown model architecture: model_type='{model_type}'. "
                f"Could not auto-detect from model structure. "
                f"Supported families: {sorted(set(ARCHITECTURE_FAMILIES.values()))}"
            )


def is_causal_model(config: dict) -> bool:
    """Check if model is causal (decoder-only) from config.

    Args:
        config: Model config dict from config.json

    Returns:
        True if model is causal, False if bidirectional/encoder
    """
    model_type = config.get("model_type", "").lower()
    return model_type in CAUSAL_MODEL_TYPES


def get_output_projection_key(config: dict, weights: dict[str, Any]) -> str | None:
    """Find the output projection (lm_head) weight key.

    Uses architecture-aware patterns instead of string heuristics.

    Args:
        config: Model config dict from config.json
        weights: Weight dictionary from safetensors

    Returns:
        Weight key for output projection, or None if not found
    """
    model_type = config.get("model_type", "").lower()
    arch_family = ARCHITECTURE_FAMILIES.get(model_type, "llama")

    # Architecture-specific output projection key patterns
    patterns = {
        "llama": ["lm_head.weight"],
        "lfm": ["lm_head.weight"],
        "gpt2": ["lm_head.weight"],
        "gpt_neox": ["embed_out.weight"],
        "bert": ["cls.predictions.decoder.weight", "cls.predictions.bias"],
        "falcon": ["lm_head.weight"],
        "phi": ["lm_head.weight"],
        "gemma": ["lm_head.weight"],
    }

    for pattern in patterns.get(arch_family, patterns["llama"]):
        if pattern in weights:
            return pattern

    # Fallback: search for common patterns
    for key in weights:
        key_lower = key.lower()
        if "lm_head" in key_lower and "weight" in key_lower:
            return key
        if key_lower == "output.weight":
            return key

    return None


def get_attention_key_pattern(config: dict, layer_idx: int) -> list[str]:
    """Get expected attention weight key patterns for a layer.

    Args:
        config: Model config dict from config.json
        layer_idx: Layer index

    Returns:
        List of expected attention key patterns (any of which indicates attention)
    """
    model_type = config.get("model_type", "").lower()
    arch_family = ARCHITECTURE_FAMILIES.get(model_type, "llama")

    # Architecture-specific attention patterns
    patterns = {
        "llama": [
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        ],
        "lfm": [
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.attn.q_proj.weight",
        ],
        "gpt2": [
            f"h.{layer_idx}.attn.c_attn.weight",
            f"transformer.h.{layer_idx}.attn.c_attn.weight",
        ],
        "gpt_neox": [
            f"gpt_neox.layers.{layer_idx}.attention.query_key_value.weight",
        ],
        "bert": [
            f"bert.encoder.layer.{layer_idx}.attention.self.query.weight",
            f"encoder.layer.{layer_idx}.attention.self.query.weight",
        ],
    }

    return patterns.get(arch_family, patterns["llama"])


def is_attention_key(key: str, config: dict, layer_idx: int) -> bool:
    """Check if a weight key is an attention projection for the given layer.

    Args:
        key: Weight key to check
        config: Model config dict
        layer_idx: Expected layer index

    Returns:
        True if key is an attention projection for this layer
    """
    model_type = config.get("model_type", "").lower()
    arch_family = ARCHITECTURE_FAMILIES.get(model_type, "llama")

    # Check if layer index matches
    key_lower = key.lower()

    # Architecture-specific attention key detection
    if arch_family in ("llama", "lfm"):
        return (
            f"layers.{layer_idx}." in key
            and ("self_attn." in key or "attn." in key)
            and "q_proj" in key
        )
    elif arch_family == "gpt2":
        return f"h.{layer_idx}.attn.c_attn" in key or f"h.{layer_idx}.attn.c_proj" in key
    elif arch_family == "gpt_neox":
        return f"layers.{layer_idx}.attention." in key
    elif arch_family == "bert":
        return f"layer.{layer_idx}.attention." in key and "query" in key_lower

    # Default fallback for LLaMA-like
    return f"layers.{layer_idx}." in key and "self_attn" in key and "q_proj" in key


# ==============================================================================
# Factory Implementation (for ports.model_architecture_factory)
# ==============================================================================


class AdapterFactory:
    """Adapter implementation of ModelArchitectureFactoryPort.

    Wraps the module-level functions as instance methods to satisfy
    the factory protocol interface.
    """

    def load_config(self, model_path: str | Path) -> dict:
        """Load config.json from model directory."""
        return load_config(model_path)

    def get_architecture(
        self,
        model: Any,
        config: dict | None = None,
        model_path: str | Path | None = None,
    ) -> "ModelArchitecturePort":
        """Create architecture wrapper from model and config."""
        return get_model_architecture(model, config, model_path)

    def is_causal_model(self, config: dict) -> bool:
        """Check if model is causal (decoder-only) from config."""
        return is_causal_model(config)

    def get_output_projection_key(
        self, config: dict, weights: dict[str, Any]
    ) -> str | None:
        """Find the output projection (lm_head) weight key."""
        return get_output_projection_key(config, weights)

    def get_attention_key_pattern(self, config: dict, layer_idx: int) -> list[str]:
        """Get expected attention weight key patterns for a layer."""
        return get_attention_key_pattern(config, layer_idx)

    def is_attention_key(self, key: str, config: dict, layer_idx: int) -> bool:
        """Check if a weight key is an attention projection for the given layer."""
        return is_attention_key(key, config, layer_idx)


__all__ = [
    "ARCHITECTURE_FAMILIES",
    "CAUSAL_MODEL_TYPES",
    # Factory for ports
    "AdapterFactory",
    # Architecture implementations
    "BERTArchitecture",
    "GPT2Architecture",
    "GPTNeoXArchitecture",
    "LFMArchitecture",
    "LlamaArchitecture",
    # Layer accessor implementations
    "BERTLayerAccessor",
    "GPT2LayerAccessor",
    "GPTNeoXLayerAccessor",
    "LFMLayerAccessor",
    "LlamaLayerAccessor",
    # Helper functions
    "get_attention_key_pattern",
    "get_model_architecture",
    "get_output_projection_key",
    "is_attention_key",
    "is_causal_model",
    "load_config",
]
