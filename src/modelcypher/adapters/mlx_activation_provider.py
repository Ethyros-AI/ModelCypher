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

"""
MLX Activation Provider - Collects activations from MLX models.

This is an ADAPTER in hexagonal architecture. It implements the ActivationProvider
protocol for the MLX backend (macOS Metal GPU).

Usage:
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    provider = MLXActivationProvider()
    hidden_acts = provider.collect_hidden_activations(model, tokenizer, text)

    # With architecture-aware pooling (recommended)
    provider = MLXActivationProvider(model_path="/path/to/model")
    hidden_acts = provider.collect_hidden_activations(model, tokenizer, text)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array
    from modelcypher.ports.model_architecture import ModelArchitecturePort

logger = logging.getLogger(__name__)

PoolingStrategy = Literal["auto", "last", "mean", "max"]


class MLXActivationProvider:
    """
    MLX implementation of ActivationProvider protocol.

    Collects activations from mlx_lm models, keeping all tensors on Metal GPU.

    Args:
        model_path: Optional path to model directory (for loading config.json)
        config: Optional model config dict (alternative to model_path)
        pooling: Default pooling strategy for activation collection
            - "auto": Use last-token for causal models, mean for bidirectional
            - "last": Always use last token (best for causal LMs)
            - "mean": Always use mean pooling (best for encoders)
            - "max": Always use max pooling
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: dict | None = None,
        pooling: PoolingStrategy = "auto",
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._config = config
        self._default_pooling = pooling
        self._architecture_cache: dict[int, "ModelArchitecturePort"] = {}

    def _get_architecture(self, model: Any) -> "ModelArchitecturePort":
        """Get or create architecture wrapper for model.

        Uses cached wrapper if model instance matches, otherwise creates new one.
        Falls back to auto-detection if no config was provided.
        """
        from modelcypher.adapters.model_architecture import get_model_architecture

        model_id = id(model)
        if model_id in self._architecture_cache:
            return self._architecture_cache[model_id]

        # Load config if not provided
        config = self._config
        if config is None and self._model_path is not None:
            from modelcypher.adapters.model_architecture import load_config
            config = load_config(self._model_path)

        arch = get_model_architecture(model, config=config, model_path=self._model_path)
        self._architecture_cache[model_id] = arch
        return arch

    def _resolve_pooling(self, model: Any, pooling: PoolingStrategy | None = None) -> str:
        """Resolve pooling strategy, using architecture-aware auto-detection."""
        strategy = pooling or self._default_pooling

        if strategy == "auto":
            try:
                arch = self._get_architecture(model)
                return "last" if arch.is_causal else "mean"
            except (ValueError, AttributeError):
                # Fall back to mean if architecture detection fails
                logger.debug("Architecture detection failed, using mean pooling")
                return "mean"

        return strategy

    def _pool_activation(
        self,
        hidden: Any,  # mx.array with shape [batch, seq_len, hidden_dim]
        pooling: str,
    ) -> Any:
        """Apply pooling strategy to hidden state.

        Args:
            hidden: Activation tensor with shape [batch, seq_len, hidden_dim]
            pooling: "last", "mean", or "max"

        Returns:
            Pooled tensor with shape [hidden_dim]
        """
        import mlx.core as mx

        if pooling == "last":
            # Last token pooling - best for causal models
            return hidden[0, -1, :]
        elif pooling == "max":
            # Max pooling
            return mx.max(hidden[0], axis=0)
        else:
            # Mean pooling (default) - best for bidirectional
            return mx.mean(hidden, axis=(0, 1))

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
        pooling: PoolingStrategy | None = None,
    ) -> dict[int, "Array"]:
        """
        Collect per-layer hidden state activations for a text input.

        Runs the text through the model and extracts the final hidden state
        at each layer, pooled according to the specified strategy.

        Args:
            model: MLX model instance
            tokenizer: Tokenizer for encoding text
            text: Input text to process
            token_ids: Pre-computed token IDs (optional)
            pooling: Pooling strategy (default: instance default or "auto")

        Returns:
            Dict mapping layer index to pooled activation (MLX arrays on Metal GPU)
        """
        import mlx.core as mx

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
        input_ids = mx.array([token_ids])

        activations: dict[int, "Array"] = {}
        pool_strategy = self._resolve_pooling(model, pooling)

        try:
            # Try fast path with forward_with_hidden_states first
            if hasattr(model, "forward_with_hidden_states"):
                _, hidden_states = model.forward_with_hidden_states(input_ids)
                for layer_idx, hidden in enumerate(hidden_states):
                    pooled = self._pool_activation(hidden, pool_strategy)
                    mx.eval(pooled)
                    activations[layer_idx] = pooled
                return activations

            # Use architecture-aware layer iteration
            try:
                arch = self._get_architecture(model)
                h = arch.embed_module(input_ids)

                for layer_idx, layer in enumerate(arch.layers):
                    # Layer may return single tensor or (tensor, cache) tuple
                    result = layer(h)
                    if isinstance(result, tuple):
                        h = result[0]
                    else:
                        h = result
                    pooled = self._pool_activation(h, pool_strategy)
                    mx.eval(pooled)
                    activations[layer_idx] = pooled

            except ValueError:
                # Architecture detection failed, try raw model call
                output = model(input_ids)
                mx.eval(output)
                pooled = self._pool_activation(output, pool_strategy)
                mx.eval(pooled)
                activations[0] = pooled

        except Exception as e:
            logger.warning("Activation collection failed for text '%s...': %s", text[:30], e)

        if not activations:
            logger.debug("No activations collected for text: %s", text[:50])

        return activations

    def collect_embedding_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
        pooling: PoolingStrategy | None = None,
    ) -> "Array":
        """
        Collect post-embedding activation for a text input.

        This captures the OUTPUT of embed_tokens (before layer 0 input_layernorm).
        Shape: [hidden_dim] (pooled over sequence length).

        Used for GramAlign at the 1D→2D interface (token IDs → embedding space).
        Linear alignment is exact on probes; geodesic CKA is the overlap diagnostic
        at the embedding dimension.

        Returns MLX array directly (stays on Metal GPU).
        """
        import mlx.core as mx

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
        input_ids = mx.array([token_ids])

        pool_strategy = self._resolve_pooling(model, pooling)

        # Use architecture-aware embedding access
        try:
            arch = self._get_architecture(model)
            h = arch.embed_module(input_ids)
        except ValueError as e:
            raise RuntimeError(
                f"Cannot find embedding layer. Model architecture not supported: {e}"
            ) from e

        if h is None:
            raise RuntimeError(
                "No embedding output collected. embed_module() returned None."
            )

        # h: [batch=1, seq_len, hidden_dim]
        pooled = self._pool_activation(h, pool_strategy)
        mx.eval(pooled)
        return pooled

    def collect_gate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
        pooling: PoolingStrategy | None = None,
    ) -> dict[int, "Array"]:
        """
        Collect per-layer PRE-SiLU gate_proj activations for a text input.

        Captures the gate_proj output before SiLU is applied. Use this
        pre-nonlinearity space for gate_proj/up_proj alignment.

        Shape: [intermediate_dim] (e.g., 2560 for SmolLM, 4864 for Qwen)

        Returns MLX arrays directly (stays on Metal GPU).
        """
        import mlx.core as mx

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
        input_ids = mx.array([token_ids])

        activations: dict[int, "Array"] = {}
        pool_strategy = self._resolve_pooling(model, pooling)

        try:
            arch = self._get_architecture(model)
            h = arch.embed_module(input_ids)

            for layer_idx in range(arch.num_layers):
                accessor = arch.layer_accessor(layer_idx)

                # Apply input layer norm
                input_norm = accessor.input_norm
                h_norm = input_norm(h) if input_norm is not None else h

                # Apply attention (or SSM for hybrid models)
                attn = accessor.attention
                if attn is not None:
                    attn_out = attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = mx.zeros_like(h)

                # Add residual
                h = h + attn_out

                # Post-attention norm
                post_norm = accessor.post_attn_norm
                h_post = post_norm(h) if post_norm is not None else h

                # Extract PRE-SiLU gate_proj output from MLP
                ff_module = accessor.mlp
                if ff_module is not None:
                    if hasattr(ff_module, "gate_proj"):
                        # Standard SwiGLU/SiLU architecture
                        gate = ff_module.gate_proj(h_post)
                        mx.eval(gate)
                        pooled = self._pool_activation(gate, pool_strategy)
                        mx.eval(pooled)
                        activations[layer_idx] = pooled
                    elif hasattr(ff_module, "w1"):
                        # LFM2/Mamba-style (w1=gate)
                        gate = ff_module.w1(h_post)
                        mx.eval(gate)
                        pooled = self._pool_activation(gate, pool_strategy)
                        mx.eval(pooled)
                        activations[layer_idx] = pooled

                # Complete the layer forward for next iteration
                if ff_module is not None:
                    mlp_out = ff_module(h_post)
                    h = h + mlp_out
                else:
                    # Fall back to full layer call
                    layer = arch.layers[layer_idx]
                    result = layer(h)
                    if isinstance(result, tuple):
                        h = result[0]
                    else:
                        h = result

        except ValueError:
            logger.debug("Model structure not compatible with gate activation collection")
        except Exception as e:
            logger.warning("Gate activation collection failed: %s", e)

        return activations

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
        pooling: PoolingStrategy | None = None,
    ) -> dict[int, "Array"]:
        """
        Collect per-layer MLP intermediate activations for a text input.

        Captures the activation INSIDE the MLP (after gate_proj * up_proj, before down_proj).
        This is the post-nonlinearity space: SiLU(gate) * up.

        Use this for down_proj input stitching. For gate_proj/up_proj stitching,
        use collect_gate_activations() to get the pre-SiLU space.

        Shape: [intermediate_dim] (e.g., 2560 for SmolLM, 4864 for Qwen)

        Returns MLX arrays directly (stays on Metal GPU).
        """
        import mlx.core as mx
        from mlx import nn

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
        input_ids = mx.array([token_ids])

        activations: dict[int, "Array"] = {}
        pool_strategy = self._resolve_pooling(model, pooling)

        try:
            arch = self._get_architecture(model)
            h = arch.embed_module(input_ids)

            for layer_idx in range(arch.num_layers):
                accessor = arch.layer_accessor(layer_idx)

                # Apply input layer norm
                input_norm = accessor.input_norm
                h_norm = input_norm(h) if input_norm is not None else h

                # Apply attention (or SSM for hybrid models)
                attn = accessor.attention
                if attn is not None:
                    attn_out = attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = mx.zeros_like(h)

                # Add residual
                h = h + attn_out

                # Post-attention norm
                post_norm = accessor.post_attn_norm
                h_post = post_norm(h) if post_norm is not None else h

                # Extract MLP intermediate activation
                ff_module = accessor.mlp
                if ff_module is not None:
                    if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                        # Standard SwiGLU/SiLU architecture (LLaMA, Qwen, Mistral)
                        up = ff_module.up_proj(h_post)
                        gate = ff_module.gate_proj(h_post)
                        intermediate = nn.silu(gate) * up
                        mx.eval(intermediate)
                        pooled = self._pool_activation(intermediate, pool_strategy)
                        mx.eval(pooled)
                        activations[layer_idx] = pooled
                    elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                        # LFM2/Mamba-style SwiGLU (w1=gate, w3=up, w2=down)
                        gate = ff_module.w1(h_post)
                        up = ff_module.w3(h_post)
                        intermediate = nn.silu(gate) * up
                        mx.eval(intermediate)
                        pooled = self._pool_activation(intermediate, pool_strategy)
                        mx.eval(pooled)
                        activations[layer_idx] = pooled
                    elif hasattr(ff_module, "fc1") and hasattr(ff_module, "fc2"):
                        # GPT-style MLP (fc1 -> activation -> fc2)
                        intermediate = ff_module.fc1(h_post)
                        mx.eval(intermediate)
                        pooled = self._pool_activation(intermediate, pool_strategy)
                        mx.eval(pooled)
                        activations[layer_idx] = pooled
                    else:
                        logger.debug("Layer %d: Unknown MLP structure", layer_idx)

                # Complete the layer forward for next iteration
                if ff_module is not None:
                    mlp_out = ff_module(h_post)
                    h = h + mlp_out
                else:
                    layer = arch.layers[layer_idx]
                    result = layer(h)
                    if isinstance(result, tuple):
                        h = result[0]
                    else:
                        h = result

        except ValueError:
            logger.debug("Model structure not compatible with intermediate activation collection")
        except Exception as e:
            logger.warning("Intermediate activation collection failed: %s", e)

        return activations

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
        pooling: PoolingStrategy | None = None,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"], dict[int, "Array"]]:
        """
        Collect per-layer attention Q, K, and V activations for a text input.

        Returns THREE dicts for finest-level granular alignment:
        1. Q activations: [num_heads * head_dim] - for q_proj/o_proj stitching
        2. K activations: [num_kv_heads * head_dim] - for k_proj stitching
        3. V activations: [num_kv_heads * head_dim] - for v_proj stitching

        Each component gets its own alignment transform; geodesic CKA reports
        overlap per component.

        Returns tuple of (q_activations, k_activations, v_activations).
        """
        import mlx.core as mx

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
        input_ids = mx.array([token_ids])

        q_activations: dict[int, "Array"] = {}
        k_activations: dict[int, "Array"] = {}
        v_activations: dict[int, "Array"] = {}
        pool_strategy = self._resolve_pooling(model, pooling)

        try:
            arch = self._get_architecture(model)
            h = arch.embed_module(input_ids)

            for layer_idx in range(arch.num_layers):
                accessor = arch.layer_accessor(layer_idx)

                # Apply input layer norm
                input_norm = accessor.input_norm
                h_norm = input_norm(h) if input_norm is not None else h

                # Get attention module and extract Q, K, V
                attn = accessor.attention
                if attn is not None and hasattr(attn, "q_proj"):
                    q = attn.q_proj(h_norm)
                    k = attn.k_proj(h_norm)
                    v = attn.v_proj(h_norm)
                    mx.eval(q)
                    mx.eval(k)
                    mx.eval(v)

                    q_pooled = self._pool_activation(q, pool_strategy)
                    mx.eval(q_pooled)
                    q_activations[layer_idx] = q_pooled

                    k_pooled = self._pool_activation(k, pool_strategy)
                    mx.eval(k_pooled)
                    k_activations[layer_idx] = k_pooled

                    v_pooled = self._pool_activation(v, pool_strategy)
                    mx.eval(v_pooled)
                    v_activations[layer_idx] = v_pooled

                # Complete the layer forward for next iteration
                layer = arch.layers[layer_idx]
                result = layer(h)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result

        except ValueError:
            logger.debug("Model structure not compatible with attention activation collection")
        except Exception as e:
            logger.warning("Attention activation collection failed: %s", e)

        return q_activations, k_activations, v_activations


    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ):
        """
        Collect hidden + intermediate + gate + embedding activations in one batched pass.
        """
        from modelcypher.ports.activation_provider import ProbeActivationBatch

        import mlx.core as mx
        from mlx import nn

        if not texts:
            return ProbeActivationBatch(hidden=[], intermediate=[], gate=[], embedding=[])

        all_token_ids: list[list[int]] = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                all_token_ids.append(tokens)
            else:
                all_token_ids.append(list(tokens.ids))

        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = mx.array(padded)
        batch_size = len(texts)

        hidden_results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]
        intermediate_results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]
        gate_results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]
        embedding_results: list["Array"] = []
        all_tensors: list["Array"] = []

        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise RuntimeError("Model structure not compatible with probe batch collection.")

        inner = model.model
        if hasattr(inner, "embed_tokens"):
            h = inner.embed_tokens(input_ids)
        elif hasattr(inner, "wte"):
            h = inner.wte(input_ids)
        else:
            raise RuntimeError(
                "Cannot find embedding layer (embed_tokens or wte). "
                "Model architecture not supported for probe collection."
            )

        seq_lengths = [len(ids) for ids in all_token_ids]
        lengths = mx.array(seq_lengths, dtype=h.dtype)
        pos = mx.arange(max_len)
        pad_mask = pos[None, :] < lengths[:, None]
        mask = pad_mask.astype(h.dtype)
        denom = lengths[:, None]
        causal = pos[:, None] >= pos[None, :]
        attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
        attn_mask = attn_mask[:, None, :, :]
        pooled_embeddings = mx.sum(h * mask[:, :, None], axis=1) / denom
        for i in range(batch_size):
            embedding_results.append(pooled_embeddings[i])
        all_tensors.append(pooled_embeddings)

        for layer_idx, layer in enumerate(inner.layers):
            is_lfm2 = (
                hasattr(layer, "operator_norm")
                and hasattr(layer, "ffn_norm")
                and hasattr(layer, "feed_forward")
            )

            if is_lfm2:
                h_norm = layer.operator_norm(h)
                if hasattr(layer, "self_attn"):
                    attn_out = layer.self_attn(h_norm, mask=attn_mask, cache=None)
                elif hasattr(layer, "conv"):
                    attn_out = layer.conv(h_norm, mask=pad_mask, cache=None)
                else:
                    raise RuntimeError(
                        "LFM2 block missing attention/conv module for probe collection."
                    )
                h = h + attn_out
                h_post = layer.ffn_norm(h)
                ff_module = layer.feed_forward
            else:
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                else:
                    h_norm = h

                if hasattr(layer, "self_attn"):
                    attn_out = layer.self_attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                elif hasattr(layer, "attn"):
                    attn_out = layer.attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    raise RuntimeError(
                        "Model attention block not found; probe collection requires attention."
                    )

                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                else:
                    h_post = h

            if not is_lfm2:
                ff_module = None
                if hasattr(layer, "mlp"):
                    ff_module = layer.mlp
                elif hasattr(layer, "feed_forward"):
                    ff_module = layer.feed_forward

            intermediate = None
            gate = None
            mlp_out = None
            if ff_module is not None:
                if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                    up = ff_module.up_proj(h_post)
                    gate = ff_module.gate_proj(h_post)
                    intermediate = nn.silu(gate) * up
                    if hasattr(ff_module, "down_proj"):
                        mlp_out = ff_module.down_proj(intermediate)
                elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                    gate = ff_module.w1(h_post)
                    up = ff_module.w3(h_post)
                    intermediate = nn.silu(gate) * up
                    if hasattr(ff_module, "w2"):
                        mlp_out = ff_module.w2(intermediate)
                elif hasattr(ff_module, "fc1"):
                    intermediate = ff_module.fc1(h_post)

            if intermediate is None:
                raise RuntimeError(
                    "Model MLP block not found; probe collection requires intermediate activations."
                )

            pooled_intermediate = mx.sum(
                intermediate * mask[:, :, None], axis=1
            ) / denom
            for i in range(batch_size):
                intermediate_results[i][layer_idx] = pooled_intermediate[i]
            all_tensors.append(pooled_intermediate)

            if gate is not None:
                pooled_gate = mx.sum(gate * mask[:, :, None], axis=1) / denom
                for i in range(batch_size):
                    gate_results[i][layer_idx] = pooled_gate[i]
                all_tensors.append(pooled_gate)

            if mlp_out is None:
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                elif hasattr(layer, "feed_forward"):
                    mlp_out = layer.feed_forward(h_post)
                else:
                    raise RuntimeError(
                        "Model MLP output path not found; probe collection requires MLP output."
                    )

            h = h + mlp_out

            pooled_hidden = mx.sum(h * mask[:, :, None], axis=1) / denom
            for i in range(batch_size):
                hidden_results[i][layer_idx] = pooled_hidden[i]
            all_tensors.append(pooled_hidden)

        if all_tensors:
            mx.eval(*all_tensors)

        return ProbeActivationBatch(
            hidden=hidden_results,
            intermediate=intermediate_results,
            gate=gate_results,
            embedding=embedding_results,
        )


    # ==========================================================================
    # BATCHED METHODS - Process multiple texts in a single forward pass
    # ==========================================================================

    def collect_hidden_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """
        Collect per-layer hidden state activations for multiple texts in one pass.

        More efficient than calling collect_hidden_activations() multiple times
        because it batches forward passes, reducing kernel launch overhead.

        Returns list of dicts, one per input text.
        """
        import mlx.core as mx

        if not texts:
            return []

        # Tokenize all texts
        all_token_ids = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                all_token_ids.append(tokens)
            else:
                all_token_ids.append(list(tokens.ids))

        # Find max length for padding
        max_len = max(len(ids) for ids in all_token_ids)

        # Pad token ID (use 0 as default, or tokenizer's pad_token_id if available)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

        # Pad all sequences to max_len
        padded = []
        for ids in all_token_ids:
            padded.append(ids + [pad_id] * (max_len - len(ids)))

        input_ids = mx.array(padded)  # [batch_size, max_len]
        batch_size = len(texts)

        # Initialize result structure
        results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]

        # Collect all tensors lazily, eval once at the end for GPU efficiency
        all_tensors = []

        try:
            if hasattr(model, "forward_with_hidden_states"):
                _, hidden_states = model.forward_with_hidden_states(input_ids)
                for layer_idx, hidden in enumerate(hidden_states):
                    # hidden: [batch, seq, hidden_dim]
                    for i in range(batch_size):
                        # Use only non-padded tokens for mean pooling
                        seq_len = len(all_token_ids[i])
                        pooled = mx.mean(hidden[i, :seq_len, :], axis=0)
                        results[i][layer_idx] = pooled
                        all_tensors.append(pooled)

            elif hasattr(model, "model") and hasattr(model.model, "layers"):
                inner = model.model
                if hasattr(inner, "embed_tokens"):
                    h = inner.embed_tokens(input_ids)
                elif hasattr(inner, "wte"):
                    h = inner.wte(input_ids)
                else:
                    h = model.embed(input_ids) if hasattr(model, "embed") else None

                if h is not None:
                    for layer_idx, layer in enumerate(inner.layers):
                        result = layer(h)
                        if isinstance(result, tuple):
                            h = result[0]
                        else:
                            h = result
                        # h: [batch, seq, hidden_dim]
                        for i in range(batch_size):
                            seq_len = len(all_token_ids[i])
                            pooled = mx.mean(h[i, :seq_len, :], axis=0)
                            results[i][layer_idx] = pooled
                            all_tensors.append(pooled)
            else:
                output = model(input_ids)
                for i in range(batch_size):
                    seq_len = len(all_token_ids[i])
                    pooled = mx.mean(output[i, :seq_len, :], axis=0)
                    results[i][0] = pooled
                    all_tensors.append(pooled)

            # Single eval at the end - allows GPU to batch all operations
            if all_tensors:
                mx.eval(*all_tensors)

        except Exception as e:
            raise RuntimeError(f"Batch activation collection failed: {e}") from e

        return results

    def collect_intermediate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """
        Collect per-layer MLP intermediate activations for multiple texts in one pass.

        Returns list of dicts, one per input text.
        """
        import mlx.core as mx
        from mlx import nn

        if not texts:
            return []

        # Tokenize all texts
        all_token_ids = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                all_token_ids.append(tokens)
            else:
                all_token_ids.append(list(tokens.ids))

        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]
        all_tensors = []  # Collect lazily, eval once at end

        try:
            if not (hasattr(model, "model") and hasattr(model.model, "layers")):
                raise RuntimeError("Model not compatible with batch intermediate collection")

            inner = model.model
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                raise RuntimeError("Cannot find embedding layer for batch intermediate collection")

            for layer_idx, layer in enumerate(inner.layers):
                # Apply input layer norm
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                else:
                    h_norm = h

                # Apply self-attention
                if hasattr(layer, "self_attn"):
                    attn_out = layer.self_attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                elif hasattr(layer, "attn"):
                    attn_out = layer.attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = mx.zeros_like(h)

                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                else:
                    h_post = h

                # Extract MLP intermediate activation for batch
                # Try multiple architectures: standard transformer, LFM2/Mamba hybrid, GPT-style
                ff_module = None
                if hasattr(layer, "mlp"):
                    ff_module = layer.mlp
                elif hasattr(layer, "feed_forward"):
                    ff_module = layer.feed_forward

                if ff_module is not None:
                    if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                        up = ff_module.up_proj(h_post)
                        gate = ff_module.gate_proj(h_post)
                        intermediate = nn.silu(gate) * up
                        for i in range(batch_size):
                            seq_len = len(all_token_ids[i])
                            pooled = mx.mean(intermediate[i, :seq_len, :], axis=0)
                            results[i][layer_idx] = pooled
                            all_tensors.append(pooled)
                    elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                        # LFM2/Mamba-style SwiGLU (w1=gate, w3=up, w2=down)
                        gate = ff_module.w1(h_post)
                        up = ff_module.w3(h_post)
                        intermediate = nn.silu(gate) * up
                        for i in range(batch_size):
                            seq_len = len(all_token_ids[i])
                            pooled = mx.mean(intermediate[i, :seq_len, :], axis=0)
                            results[i][layer_idx] = pooled
                            all_tensors.append(pooled)
                    elif hasattr(ff_module, "fc1"):
                        intermediate = ff_module.fc1(h_post)
                        for i in range(batch_size):
                            seq_len = len(all_token_ids[i])
                            pooled = mx.mean(intermediate[i, :seq_len, :], axis=0)
                            results[i][layer_idx] = pooled
                            all_tensors.append(pooled)

                # Complete layer forward
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                    h = h + mlp_out
                elif hasattr(layer, "feed_forward"):
                    ff_out = layer.feed_forward(h_post)
                    h = h + ff_out
                else:
                    result = layer(h)
                    h = result[0] if isinstance(result, tuple) else result

            # Single eval at the end - allows GPU to batch all operations
            if all_tensors:
                mx.eval(*all_tensors)

        except Exception as e:
            raise RuntimeError(f"Batch intermediate collection failed: {e}") from e

        return results

    def collect_gate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """
        Collect per-layer PRE-SiLU gate_proj activations for multiple texts.

        Captures gate_proj output before SiLU for gate_proj/up_proj stitching.

        Returns list of dicts, one per input text.
        """
        import mlx.core as mx

        if not texts:
            return []

        all_token_ids = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                all_token_ids.append(tokens)
            else:
                all_token_ids.append(list(tokens.ids))

        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]
        all_tensors: list["Array"] = []

        try:
            if not (hasattr(model, "model") and hasattr(model.model, "layers")):
                raise RuntimeError("Model not compatible with batch gate collection")

            inner = model.model
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                raise RuntimeError("Cannot find embedding layer for batch gate collection")

            seq_lengths = [len(ids) for ids in all_token_ids]
            lengths = mx.array(seq_lengths, dtype=h.dtype)
            pos = mx.arange(max_len)
            pad_mask = pos[None, :] < lengths[:, None]
            mask = pad_mask.astype(h.dtype)
            denom = lengths[:, None]
            causal = pos[:, None] >= pos[None, :]
            attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
            attn_mask = attn_mask[:, None, :, :]

            for layer_idx, layer in enumerate(inner.layers):
                is_lfm2 = (
                    hasattr(layer, "operator_norm")
                    and hasattr(layer, "ffn_norm")
                    and hasattr(layer, "feed_forward")
                )

                if is_lfm2:
                    h_norm = layer.operator_norm(h)
                    if hasattr(layer, "self_attn"):
                        attn_out = layer.self_attn(h_norm, mask=attn_mask, cache=None)
                    elif hasattr(layer, "conv"):
                        attn_out = layer.conv(h_norm, mask=pad_mask, cache=None)
                    else:
                        attn_out = mx.zeros_like(h)
                    h = h + attn_out
                    h_post = layer.ffn_norm(h)
                    ff_module = layer.feed_forward
                else:
                    if hasattr(layer, "input_layernorm"):
                        h_norm = layer.input_layernorm(h)
                    elif hasattr(layer, "ln_1"):
                        h_norm = layer.ln_1(h)
                    else:
                        h_norm = h

                    if hasattr(layer, "self_attn"):
                        attn_out = layer.self_attn(h_norm)
                        if isinstance(attn_out, tuple):
                            attn_out = attn_out[0]
                    elif hasattr(layer, "attn"):
                        attn_out = layer.attn(h_norm)
                        if isinstance(attn_out, tuple):
                            attn_out = attn_out[0]
                    else:
                        attn_out = mx.zeros_like(h)

                    h = h + attn_out

                    if hasattr(layer, "post_attention_layernorm"):
                        h_post = layer.post_attention_layernorm(h)
                    elif hasattr(layer, "ln_2"):
                        h_post = layer.ln_2(h)
                    else:
                        h_post = h

                    ff_module = None
                    if hasattr(layer, "mlp"):
                        ff_module = layer.mlp
                    elif hasattr(layer, "feed_forward"):
                        ff_module = layer.feed_forward

                if ff_module is not None:
                    if hasattr(ff_module, "gate_proj"):
                        gate = ff_module.gate_proj(h_post)
                        pooled_gate = mx.sum(gate * mask[:, :, None], axis=1) / denom
                        for i in range(batch_size):
                            results[i][layer_idx] = pooled_gate[i]
                        all_tensors.append(pooled_gate)
                    elif hasattr(ff_module, "w1"):
                        gate = ff_module.w1(h_post)
                        pooled_gate = mx.sum(gate * mask[:, :, None], axis=1) / denom
                        for i in range(batch_size):
                            results[i][layer_idx] = pooled_gate[i]
                        all_tensors.append(pooled_gate)

                # Complete layer forward
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                    h = h + mlp_out
                elif hasattr(layer, "feed_forward"):
                    ff_out = layer.feed_forward(h_post)
                    h = h + ff_out
                else:
                    result = layer(h)
                    h = h + (result[0] if isinstance(result, tuple) else result)

            # Single eval at the end - allows GPU to batch all operations
            if all_tensors:
                mx.eval(*all_tensors)

        except Exception as e:
            raise RuntimeError(f"Batch gate collection failed: {e}") from e

        return results

    def collect_embedding_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list["Array"]:
        """
        Collect post-embedding activations for multiple texts in one pass.

        This is the batched version of collect_embedding_activations().
        More efficient because it shares tokenization overhead and allows
        GPU batching of embedding lookups.

        Returns list of MLX arrays, one per input text.
        """
        import mlx.core as mx

        if not texts:
            return []

        # Tokenize all texts
        all_token_ids = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                all_token_ids.append(tokens)
            else:
                all_token_ids.append(list(tokens.ids))

        # Find max length for padding
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

        # Pad all sequences
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = mx.array(padded)  # [batch_size, max_len]
        batch_size = len(texts)

        results: list["Array"] = []

        # Get embedding layer
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                raise RuntimeError(
                    "Cannot find embedding layer (embed_tokens or wte). "
                    "Model architecture not supported for embedding collection."
                )
        else:
            h = model.embed(input_ids) if hasattr(model, "embed") else None

        if h is None:
            raise RuntimeError(
                "No embedding output collected. Model.embed() returned None or "
                "model structure not compatible."
            )

        # h: [batch_size, max_len, hidden_dim]
        # Mean pool over non-padded tokens for each sample
        for i in range(batch_size):
            seq_len = len(all_token_ids[i])
            pooled = mx.mean(h[i, :seq_len, :], axis=0)
            results.append(pooled)

        # Single eval at the end
        if results:
            mx.eval(*results)

        return results

    # ==========================================================================
    # TRAJECTORY METHODS - For geometric manifold mapping
    # ==========================================================================

    def collect_trajectory_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> "TrajectoryActivations":
        """
        Collect COMPLETE trajectory activations for geometric manifold mapping.

        This is the foundation for PERFECT merging - collects ALL activation types
        in a SINGLE forward pass:
        - Hidden states (post-layer output)
        - Intermediate (MLP post-activation, before down_proj)
        - Embedding (post-embed_tokens)
        - Attention Q/K/V (separate for granular alignment)
        - Gate (pre-SiLU gate_proj output)

        Unlike mean-pooled methods, this preserves the FULL trajectory at every
        token position across all layers. A 100-token text yields 100 positions
        (plus 99 velocities for hidden states).

        Args:
            model: The loaded model (e.g., mlx_lm model).
            tokenizer: The tokenizer for encoding texts.
            texts: List of text inputs to process in a single forward pass.

        Returns:
            TrajectoryActivations containing ALL activation types.

        Raises:
            RuntimeError: If model structure is not compatible.
        """
        from modelcypher.ports.activation_provider import TrajectoryActivations

        import mlx.core as mx
        from mlx import nn

        if not texts:
            return TrajectoryActivations(
                positions={},
                velocities={},
                intermediate_positions={},
                embedding_positions=mx.zeros((0, 1)),
                q_positions={},
                k_positions={},
                v_positions={},
                gate_positions={},
                text_lengths=[],
                total_tokens=0,
                n_texts=0,
            )

        # Tokenize all texts
        all_token_ids: list[list[int]] = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                all_token_ids.append(tokens)
            else:
                all_token_ids.append(list(tokens.ids))

        # Track actual lengths before padding
        text_lengths = [len(ids) for ids in all_token_ids]
        total_tokens = sum(text_lengths)
        n_texts = len(texts)

        # Pad sequences for batched forward pass
        max_len = max(text_lengths)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = mx.array(padded)  # [batch_size, max_len]

        # Get model internals
        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise RuntimeError(
                "Model structure not compatible with trajectory collection. "
                "Requires model.model.layers attribute."
            )

        inner = model.model

        # Get embeddings
        if hasattr(inner, "embed_tokens"):
            h = inner.embed_tokens(input_ids)
        elif hasattr(inner, "wte"):
            h = inner.wte(input_ids)
        else:
            raise RuntimeError(
                "Cannot find embedding layer (embed_tokens or wte). "
                "Model architecture not supported for trajectory collection."
            )

        # h: [batch_size, max_len, hidden_dim]

        # Collect embedding positions (excluding padding)
        embedding_positions_list = []
        for i in range(n_texts):
            seq_len = text_lengths[i]
            text_embedding = h[i, :seq_len, :]  # [seq_len, hidden_dim]
            embedding_positions_list.append(text_embedding)
        embedding_positions = mx.concatenate(embedding_positions_list, axis=0)

        # Storage for all activation types
        positions: dict[int, "Array"] = {}
        velocities: dict[int, "Array"] = {}
        intermediate_positions: dict[int, "Array"] = {}
        q_positions: dict[int, "Array"] = {}
        k_positions: dict[int, "Array"] = {}
        v_positions: dict[int, "Array"] = {}
        gate_positions: dict[int, "Array"] = {}
        all_tensors: list["Array"] = [embedding_positions]

        # Build attention mask for proper attention computation
        seq_lengths_arr = mx.array(text_lengths, dtype=h.dtype)
        pos = mx.arange(max_len)
        pad_mask = pos[None, :] < seq_lengths_arr[:, None]
        causal = pos[:, None] >= pos[None, :]
        attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
        attn_mask = attn_mask[:, None, :, :]  # [batch, 1, seq, seq]

        for layer_idx, layer in enumerate(inner.layers):
            # Detect architecture type
            is_lfm2 = (
                hasattr(layer, "operator_norm")
                and hasattr(layer, "ffn_norm")
                and hasattr(layer, "feed_forward")
            )

            # === LAYER FORWARD (with activation extraction) ===
            if is_lfm2:
                # LFM2/Mamba hybrid architecture
                h_norm = layer.operator_norm(h)

                # Attention/Conv
                if hasattr(layer, "self_attn"):
                    attn_module = layer.self_attn
                    attn_out = attn_module(h_norm, mask=attn_mask, cache=None)
                elif hasattr(layer, "conv"):
                    attn_out = layer.conv(h_norm, mask=pad_mask, cache=None)
                    attn_module = None
                else:
                    raise RuntimeError("LFM2 block missing attention/conv module")

                h = h + attn_out
                h_post = layer.ffn_norm(h)
                ff_module = layer.feed_forward
            else:
                # Standard transformer (Llama/Qwen/Mistral)
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                else:
                    h_norm = h

                # Get attention module and extract Q/K/V
                attn_module = None
                if hasattr(layer, "self_attn"):
                    attn_module = layer.self_attn
                    attn_out = attn_module(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                elif hasattr(layer, "attn"):
                    attn_module = layer.attn
                    attn_out = attn_module(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = mx.zeros_like(h)

                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                else:
                    h_post = h

                ff_module = None
                if hasattr(layer, "mlp"):
                    ff_module = layer.mlp
                elif hasattr(layer, "feed_forward"):
                    ff_module = layer.feed_forward

            # === EXTRACT Q/K/V ACTIVATIONS ===
            if attn_module is not None and hasattr(attn_module, "q_proj"):
                q = attn_module.q_proj(h_norm)
                k = attn_module.k_proj(h_norm)
                v = attn_module.v_proj(h_norm)

                # Extract positions (excluding padding)
                q_pos_list = []
                k_pos_list = []
                v_pos_list = []
                for i in range(n_texts):
                    seq_len = text_lengths[i]
                    q_pos_list.append(q[i, :seq_len, :])
                    k_pos_list.append(k[i, :seq_len, :])
                    v_pos_list.append(v[i, :seq_len, :])

                layer_q_pos = mx.concatenate(q_pos_list, axis=0)
                layer_k_pos = mx.concatenate(k_pos_list, axis=0)
                layer_v_pos = mx.concatenate(v_pos_list, axis=0)

                q_positions[layer_idx] = layer_q_pos
                k_positions[layer_idx] = layer_k_pos
                v_positions[layer_idx] = layer_v_pos
                all_tensors.extend([layer_q_pos, layer_k_pos, layer_v_pos])

            # === EXTRACT INTERMEDIATE AND GATE ACTIVATIONS ===
            intermediate = None
            gate = None
            mlp_out = None

            if ff_module is not None:
                if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                    # Standard SwiGLU/SiLU (LLaMA, Qwen, Mistral)
                    up = ff_module.up_proj(h_post)
                    gate = ff_module.gate_proj(h_post)
                    intermediate = nn.silu(gate) * up
                    if hasattr(ff_module, "down_proj"):
                        mlp_out = ff_module.down_proj(intermediate)
                elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                    # LFM2/Mamba-style (w1=gate, w3=up, w2=down)
                    gate = ff_module.w1(h_post)
                    up = ff_module.w3(h_post)
                    intermediate = nn.silu(gate) * up
                    if hasattr(ff_module, "w2"):
                        mlp_out = ff_module.w2(intermediate)
                elif hasattr(ff_module, "fc1"):
                    # GPT-style MLP
                    intermediate = ff_module.fc1(h_post)

            # Extract intermediate positions (excluding padding)
            if intermediate is not None:
                int_pos_list = []
                for i in range(n_texts):
                    seq_len = text_lengths[i]
                    int_pos_list.append(intermediate[i, :seq_len, :])
                layer_int_pos = mx.concatenate(int_pos_list, axis=0)
                intermediate_positions[layer_idx] = layer_int_pos
                all_tensors.append(layer_int_pos)

            # Extract gate positions (excluding padding)
            if gate is not None:
                gate_pos_list = []
                for i in range(n_texts):
                    seq_len = text_lengths[i]
                    gate_pos_list.append(gate[i, :seq_len, :])
                layer_gate_pos = mx.concatenate(gate_pos_list, axis=0)
                gate_positions[layer_idx] = layer_gate_pos
                all_tensors.append(layer_gate_pos)

            # Complete MLP forward
            if mlp_out is None:
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                elif hasattr(layer, "feed_forward"):
                    mlp_out = layer.feed_forward(h_post)
                else:
                    raise RuntimeError("Cannot complete MLP forward pass")

            h = h + mlp_out

            # === EXTRACT HIDDEN STATE POSITIONS AND VELOCITIES ===
            layer_positions_list = []
            layer_velocities_list = []

            for i in range(n_texts):
                seq_len = text_lengths[i]
                text_positions = h[i, :seq_len, :]  # [seq_len, hidden_dim]
                layer_positions_list.append(text_positions)

                # Compute velocities: h[t+1] - h[t]
                if seq_len >= 2:
                    text_velocities = text_positions[1:, :] - text_positions[:-1, :]
                    layer_velocities_list.append(text_velocities)

            if layer_positions_list:
                layer_positions = mx.concatenate(layer_positions_list, axis=0)
                positions[layer_idx] = layer_positions
                all_tensors.append(layer_positions)

            if layer_velocities_list:
                layer_velocities = mx.concatenate(layer_velocities_list, axis=0)
                velocities[layer_idx] = layer_velocities
                all_tensors.append(layer_velocities)

        # Single eval for all tensors - maximizes GPU efficiency
        if all_tensors:
            mx.eval(*all_tensors)

        return TrajectoryActivations(
            positions=positions,
            velocities=velocities,
            intermediate_positions=intermediate_positions,
            embedding_positions=embedding_positions,
            q_positions=q_positions,
            k_positions=k_positions,
            v_positions=v_positions,
            gate_positions=gate_positions,
            text_lengths=text_lengths,
            total_tokens=total_tokens,
            n_texts=n_texts,
        )


def collect_trajectory_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        layer_idx: int,
        max_seq_len: int = 512,
    ) -> dict[str, "Array"] | None:
        """
        Collect full sequence trajectory activations (not mean-pooled) at a layer.

        This returns the raw activation sequence [seq_len, hidden_dim], enabling
        trajectory-based null-space discovery. Velocities and accelerations can
        be computed from the returned positions.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding text.
            text: The text input to process.
            layer_idx: Target layer index.
            max_seq_len: Maximum sequence length (truncate longer texts).

        Returns:
            Dict with:
            - "positions": [seq_len, hidden_dim] - raw activations per position
            - "seq_len": int - actual sequence length
            - "hidden_dim": int - hidden dimension
            Or None if collection fails.
        """
        import mlx.core as mx

        try:
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)

            # Truncate if needed
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]

            if len(token_ids) < 2:
                logger.debug("Text too short for trajectory collection (need >= 2 tokens)")
                return None

            input_ids = mx.array([token_ids])

            # Get model internals
            if not (hasattr(model, "model") and hasattr(model.model, "layers")):
                logger.debug("Model structure not compatible with trajectory collection")
                return None

            inner = model.model

            # Get embeddings
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                logger.debug("Cannot find embedding layer")
                return None

            # Forward through layers up to target
            for idx, layer in enumerate(inner.layers):
                if idx > layer_idx:
                    break
                result = layer(h)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result

            mx.eval(h)

            # h is [batch=1, seq_len, hidden_dim]
            # Squeeze batch dimension
            positions = mx.squeeze(h, axis=0)  # [seq_len, hidden_dim]
            mx.eval(positions)

            seq_len = positions.shape[0]
            hidden_dim = positions.shape[1]

            return {
                "positions": positions,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
            }

        except Exception as e:
            logger.warning("Trajectory collection failed for text '%s...': %s", text[:30], e)
            return None


def get_activation_provider() -> MLXActivationProvider:
    """Get the MLX activation provider instance."""
    return MLXActivationProvider()


def get_embedding_weights(model: Any) -> "Array":
    """Return embedding matrix [vocab_size, embed_dim] from model.

    Args:
        model: The MLX model.

    Returns:
        Embedding weight matrix.

    Raises:
        RuntimeError: If embedding layer cannot be found.
    """
    inner = model.model if hasattr(model, "model") else model

    if hasattr(inner, "embed_tokens"):
        return inner.embed_tokens.weight
    elif hasattr(inner, "wte"):
        return inner.wte.weight
    else:
        raise RuntimeError(
            "Cannot find embedding layer (embed_tokens or wte). "
            "Model architecture not supported for embedding access."
        )


def get_layer_activation(
    model: Any,
    input_ids: "Array",
    layer_idx: int,
) -> "Array | None":
    """Get mean-pooled activation at specific layer for given input.

    Args:
        model: The MLX model.
        input_ids: Token IDs [batch, seq_len] or [seq_len].
        layer_idx: Target layer index.

    Returns:
        Mean-pooled activation [hidden_dim], or None if failed.
    """
    import mlx.core as mx

    inner = model.model if hasattr(model, "model") else model

    if not hasattr(inner, "layers"):
        return None

    # Ensure 2D input
    if len(input_ids.shape) == 1:
        input_ids = mx.expand_dims(input_ids, axis=0)

    # Get embeddings
    if hasattr(inner, "embed_tokens"):
        h = inner.embed_tokens(input_ids)
    elif hasattr(inner, "wte"):
        h = inner.wte(input_ids)
    else:
        return None

    # Forward through layers
    for idx, layer in enumerate(inner.layers):
        if idx > layer_idx:
            break
        result = layer(h)
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result

        if idx == layer_idx:
            # Mean pool over batch and sequence
            pooled = mx.mean(h, axis=(0, 1))
            mx.eval(pooled)
            return pooled

    return None


def forward_embeddings_to_layer(
    model: Any,
    embeddings: "Array",
    layer_idx: int,
) -> "Array | None":
    """Forward continuous embeddings (not token IDs) to get activation at layer.

    This enables gradient-based probe generation by forwarding continuous
    embeddings through the model without tokenization.

    Args:
        model: The MLX model.
        embeddings: Continuous embeddings [batch, seq_len, embed_dim] or [seq_len, embed_dim].
        layer_idx: Target layer index.

    Returns:
        Mean-pooled activation [hidden_dim], or None if failed.
    """
    import mlx.core as mx

    inner = model.model if hasattr(model, "model") else model

    if not hasattr(inner, "layers"):
        return None

    # Ensure 3D input
    h = embeddings
    if len(h.shape) == 2:
        h = mx.expand_dims(h, axis=0)

    # Forward through layers
    for idx, layer in enumerate(inner.layers):
        if idx > layer_idx:
            break
        result = layer(h)
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result

        if idx == layer_idx:
            # Mean pool over batch and sequence
            pooled = mx.mean(h, axis=(0, 1))
            mx.eval(pooled)
            return pooled

    return None


__all__ = [
    "MLXActivationProvider",
    "get_activation_provider",
    "get_embedding_weights",
    "get_layer_activation",
    "forward_embeddings_to_layer",
]
