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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

logger = logging.getLogger(__name__)


class MLXActivationProvider:
    """
    MLX implementation of ActivationProvider protocol.

    Collects activations from mlx_lm models, keeping all tensors on Metal GPU.
    """

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, "Array"]:
        """
        Collect per-layer hidden state activations for a text input.

        Runs the text through the model and extracts the final hidden state
        (mean-pooled over sequence length) at each layer.

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

        try:
            if hasattr(model, "forward_with_hidden_states"):
                _, hidden_states = model.forward_with_hidden_states(input_ids)
                for layer_idx, hidden in enumerate(hidden_states):
                    pooled = mx.mean(hidden, axis=(0, 1))
                    mx.eval(pooled)
                    activations[layer_idx] = pooled
            elif hasattr(model, "model") and hasattr(model.model, "layers"):
                if hasattr(model.model, "embed_tokens"):
                    h = model.model.embed_tokens(input_ids)
                elif hasattr(model.model, "wte"):
                    h = model.model.wte(input_ids)
                else:
                    h = model.embed(input_ids) if hasattr(model, "embed") else None

                if h is not None:
                    for layer_idx, layer in enumerate(model.model.layers):
                        # Layer may return single tensor or (tensor, cache) tuple
                        result = layer(h)
                        if isinstance(result, tuple):
                            h = result[0]
                        else:
                            h = result
                        pooled = mx.mean(h, axis=(0, 1))
                        mx.eval(pooled)
                        activations[layer_idx] = pooled
            else:
                output = model(input_ids)
                mx.eval(output)
                pooled = mx.mean(output, axis=(0, 1))
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
    ) -> "Array":
        """
        Collect post-embedding activation for a text input.

        This captures the OUTPUT of embed_tokens (before layer 0 input_layernorm).
        Shape: [hidden_dim] (mean-pooled over sequence length).

        Used for GramAlign at the 1D→2D interface (token IDs → embedding space).
        Same CKA=1.0, same geodesic math - applied at the embedding dimension.

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

        try:
            if hasattr(model, "model"):
                inner = model.model
                if hasattr(inner, "embed_tokens"):
                    h = inner.embed_tokens(input_ids)
                elif hasattr(inner, "wte"):
                    h = inner.wte(input_ids)
                else:
                    logger.warning("Cannot find embedding layer")
                    return mx.zeros((960,))  # Fallback
            else:
                h = model.embed(input_ids) if hasattr(model, "embed") else None

            if h is not None:
                # h: [batch=1, seq_len, hidden_dim]
                # Mean pool over sequence to get [hidden_dim]
                pooled = mx.mean(h, axis=(0, 1))
                mx.eval(pooled)
                return pooled
            else:
                logger.warning("No embedding output collected")
                return mx.zeros((960,))

        except Exception as e:
            logger.warning("Embedding activation collection failed: %s", e)
            return mx.zeros((960,))

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, "Array"]:
        """
        Collect per-layer MLP intermediate activations for a text input.

        Captures the activation INSIDE the MLP (after gate_proj * up_proj, before down_proj).
        This is the intermediate representation space, distinct from the hidden space.

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

        try:
            if not (hasattr(model, "model") and hasattr(model.model, "layers")):
                logger.debug("Model structure not compatible with intermediate activation collection")
                return activations

            inner = model.model

            # Get embeddings
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                logger.debug("Cannot find embedding layer")
                return activations

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

                # Add residual
                h = h + attn_out

                # Post-attention norm
                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                else:
                    h_post = h

                # Extract MLP intermediate activation
                # Try multiple architectures: standard transformer, LFM2/Mamba hybrid, GPT-style
                ff_module = None
                if hasattr(layer, "mlp"):
                    ff_module = layer.mlp
                elif hasattr(layer, "feed_forward"):
                    ff_module = layer.feed_forward

                if ff_module is not None:
                    if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                        # Standard SwiGLU/SiLU architecture (LLaMA, Qwen, Mistral)
                        up = ff_module.up_proj(h_post)
                        gate = ff_module.gate_proj(h_post)
                        # Intermediate = silu(gate) * up (before down_proj)
                        intermediate = nn.silu(gate) * up
                        mx.eval(intermediate)
                        # Mean pool over sequence
                        pooled = mx.mean(intermediate, axis=(0, 1))
                        mx.eval(pooled)
                        activations[layer_idx] = pooled
                    elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                        # LFM2/Mamba-style SwiGLU (w1=gate, w3=up, w2=down)
                        gate = ff_module.w1(h_post)
                        up = ff_module.w3(h_post)
                        # Intermediate = silu(gate) * up (before w2/down_proj)
                        intermediate = nn.silu(gate) * up
                        mx.eval(intermediate)
                        pooled = mx.mean(intermediate, axis=(0, 1))
                        mx.eval(pooled)
                        activations[layer_idx] = pooled
                    elif hasattr(ff_module, "fc1") and hasattr(ff_module, "fc2"):
                        # GPT-style MLP (fc1 -> activation -> fc2)
                        intermediate = ff_module.fc1(h_post)
                        mx.eval(intermediate)
                        pooled = mx.mean(intermediate, axis=(0, 1))
                        mx.eval(pooled)
                        activations[layer_idx] = pooled
                    else:
                        logger.debug("Layer %d: Unknown MLP structure", layer_idx)

                # Complete the layer forward for next iteration
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                    h = h + mlp_out
                elif hasattr(layer, "feed_forward"):
                    ff_out = layer.feed_forward(h_post)
                    h = h + ff_out
                else:
                    result = layer(h)
                    if isinstance(result, tuple):
                        h = result[0]
                    else:
                        h = result

        except Exception as e:
            logger.warning("Intermediate activation collection failed: %s", e)

        return activations

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"], dict[int, "Array"]]:
        """
        Collect per-layer attention Q, K, and V activations for a text input.

        Returns THREE dicts for finest-level granular alignment:
        1. Q activations: [num_heads * head_dim] - for q_proj/o_proj stitching
        2. K activations: [num_kv_heads * head_dim] - for k_proj stitching
        3. V activations: [num_kv_heads * head_dim] - for v_proj stitching

        Each component gets its own CKA=1.0 alignment transform.

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

        try:
            if not (hasattr(model, "model") and hasattr(model.model, "layers")):
                logger.debug("Model structure not compatible with attention activation collection")
                return q_activations, k_activations, v_activations

            inner = model.model

            # Get embeddings
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                logger.debug("Cannot find embedding layer")
                return q_activations, k_activations, v_activations

            for layer_idx, layer in enumerate(inner.layers):
                # Apply input layer norm
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                else:
                    h_norm = h

                # Get attention module
                attn = layer.self_attn if hasattr(layer, "self_attn") else getattr(layer, "attn", None)

                if attn is not None:
                    # Compute Q, K, V projections separately
                    if hasattr(attn, "q_proj"):
                        q = attn.q_proj(h_norm)
                        k = attn.k_proj(h_norm)
                        v = attn.v_proj(h_norm)
                        mx.eval(q)
                        mx.eval(k)
                        mx.eval(v)

                        # Q activations
                        q_pooled = mx.mean(q, axis=(0, 1))
                        mx.eval(q_pooled)
                        q_activations[layer_idx] = q_pooled

                        # K activations (separate from V for granular alignment)
                        k_pooled = mx.mean(k, axis=(0, 1))
                        mx.eval(k_pooled)
                        k_activations[layer_idx] = k_pooled

                        # V activations (separate from K for granular alignment)
                        v_pooled = mx.mean(v, axis=(0, 1))
                        mx.eval(v_pooled)
                        v_activations[layer_idx] = v_pooled

                # Complete the layer forward for next iteration
                result = layer(h)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result

        except Exception as e:
            logger.warning("Attention activation collection failed: %s", e)

        return q_activations, k_activations, v_activations


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

        try:
            if hasattr(model, "forward_with_hidden_states"):
                _, hidden_states = model.forward_with_hidden_states(input_ids)
                for layer_idx, hidden in enumerate(hidden_states):
                    # hidden: [batch, seq, hidden_dim]
                    for i in range(batch_size):
                        # Use only non-padded tokens for mean pooling
                        seq_len = len(all_token_ids[i])
                        pooled = mx.mean(hidden[i, :seq_len, :], axis=0)
                        mx.eval(pooled)
                        results[i][layer_idx] = pooled

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
                            mx.eval(pooled)
                            results[i][layer_idx] = pooled
            else:
                output = model(input_ids)
                mx.eval(output)
                for i in range(batch_size):
                    seq_len = len(all_token_ids[i])
                    pooled = mx.mean(output[i, :seq_len, :], axis=0)
                    mx.eval(pooled)
                    results[i][0] = pooled

        except Exception as e:
            logger.warning("Batch activation collection failed: %s", e)
            # Fall back to sequential processing
            return [
                self.collect_hidden_activations(model, tokenizer, text)
                for text in texts
            ]

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

        try:
            if not (hasattr(model, "model") and hasattr(model.model, "layers")):
                logger.debug("Model not compatible with batch intermediate collection")
                return [
                    self.collect_intermediate_activations(model, tokenizer, text)
                    for text in texts
                ]

            inner = model.model
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                return [
                    self.collect_intermediate_activations(model, tokenizer, text)
                    for text in texts
                ]

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
                        mx.eval(intermediate)
                        for i in range(batch_size):
                            seq_len = len(all_token_ids[i])
                            pooled = mx.mean(intermediate[i, :seq_len, :], axis=0)
                            mx.eval(pooled)
                            results[i][layer_idx] = pooled
                    elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                        # LFM2/Mamba-style SwiGLU (w1=gate, w3=up, w2=down)
                        gate = ff_module.w1(h_post)
                        up = ff_module.w3(h_post)
                        intermediate = nn.silu(gate) * up
                        mx.eval(intermediate)
                        for i in range(batch_size):
                            seq_len = len(all_token_ids[i])
                            pooled = mx.mean(intermediate[i, :seq_len, :], axis=0)
                            mx.eval(pooled)
                            results[i][layer_idx] = pooled
                    elif hasattr(ff_module, "fc1"):
                        intermediate = ff_module.fc1(h_post)
                        mx.eval(intermediate)
                        for i in range(batch_size):
                            seq_len = len(all_token_ids[i])
                            pooled = mx.mean(intermediate[i, :seq_len, :], axis=0)
                            mx.eval(pooled)
                            results[i][layer_idx] = pooled

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

        except Exception as e:
            logger.warning("Batch intermediate collection failed: %s", e)
            return [
                self.collect_intermediate_activations(model, tokenizer, text)
                for text in texts
            ]

        return results

    def collect_attention_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> tuple[list[dict[int, "Array"]], list[dict[int, "Array"]], list[dict[int, "Array"]]]:
        """
        Collect per-layer attention Q, K, and V activations for multiple texts.

        Returns tuple of (q_activations_list, k_activations_list, v_activations_list).
        """
        import mlx.core as mx

        if not texts:
            return [], [], []

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

        q_results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]
        k_results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]
        v_results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]

        try:
            if not (hasattr(model, "model") and hasattr(model.model, "layers")):
                logger.debug("Model not compatible with batch attention collection")
                q_list, k_list, v_list = [], [], []
                for text in texts:
                    q, k, v = self.collect_attention_activations(model, tokenizer, text)
                    q_list.append(q)
                    k_list.append(k)
                    v_list.append(v)
                return q_list, k_list, v_list

            inner = model.model
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                q_list, k_list, v_list = [], [], []
                for text in texts:
                    q, k, v = self.collect_attention_activations(model, tokenizer, text)
                    q_list.append(q)
                    k_list.append(k)
                    v_list.append(v)
                return q_list, k_list, v_list

            for layer_idx, layer in enumerate(inner.layers):
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                else:
                    h_norm = h

                attn = layer.self_attn if hasattr(layer, "self_attn") else getattr(layer, "attn", None)

                if attn is not None and hasattr(attn, "q_proj"):
                    q = attn.q_proj(h_norm)
                    k = attn.k_proj(h_norm)
                    v = attn.v_proj(h_norm)
                    mx.eval(q)
                    mx.eval(k)
                    mx.eval(v)

                    for i in range(batch_size):
                        seq_len = len(all_token_ids[i])
                        q_pooled = mx.mean(q[i, :seq_len, :], axis=0)
                        k_pooled = mx.mean(k[i, :seq_len, :], axis=0)
                        v_pooled = mx.mean(v[i, :seq_len, :], axis=0)
                        mx.eval(q_pooled)
                        mx.eval(k_pooled)
                        mx.eval(v_pooled)
                        q_results[i][layer_idx] = q_pooled
                        k_results[i][layer_idx] = k_pooled
                        v_results[i][layer_idx] = v_pooled

                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result

        except Exception as e:
            logger.warning("Batch attention collection failed: %s", e)
            q_list, k_list, v_list = [], [], []
            for text in texts:
                q, k, v = self.collect_attention_activations(model, tokenizer, text)
                q_list.append(q)
                k_list.append(k)
                v_list.append(v)
            return q_list, k_list, v_list

        return q_results, k_results, v_results


def get_activation_provider() -> MLXActivationProvider:
    """Get the MLX activation provider instance."""
    return MLXActivationProvider()


__all__ = ["MLXActivationProvider", "get_activation_provider"]
