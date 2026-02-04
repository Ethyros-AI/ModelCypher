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

"""Unified Activation Provider - ONE provider that uses Backend.

This is THE activation provider. It detects the backend and collects activations.
No mlx_activation_provider.py, jax_activation_provider.py, cuda_activation_provider.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from modelcypher.ports.activation_provider import (
    ActivationProvider as ActivationProviderProtocol,
    ProbeActivationBatch,
    TrajectoryActivations,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_architecture import ModelArchitecturePort

logger = logging.getLogger(__name__)

PoolingStrategy = Literal["auto", "last", "mean", "max"]


class ActivationProvider(ActivationProviderProtocol):
    """Unified activation provider - uses Backend for tensor operations.

    Handles MLX, JAX, and CUDA models through a single interface.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        model_path: str | Path | None = None,
        pooling: PoolingStrategy = "auto",
    ) -> None:
        """Initialize with optional backend.

        Args:
            backend: If None, auto-detects from platform.
            model_path: Optional path to model directory (for config.json).
            pooling: Default pooling strategy for activation collection.
        """
        if backend is None:
            from modelcypher.core.domain._backend import get_default_backend
            backend = get_default_backend()
        self._backend = backend
        self._backend_type = type(backend).__name__.lower().replace("backend", "")
        self._model_path = Path(model_path) if model_path else None
        self._default_pooling = pooling
        self._architecture_cache: dict[int, "ModelArchitecturePort"] = {}

    def _get_architecture(self, model: Any) -> "ModelArchitecturePort":
        """Get or create architecture wrapper for model."""
        from modelcypher.adapters.model_architecture import get_model_architecture, load_config

        model_id = id(model)
        if model_id in self._architecture_cache:
            return self._architecture_cache[model_id]

        config = None
        if self._model_path is not None:
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
                logger.debug("Architecture detection failed, using mean pooling")
                return "mean"

        return strategy

    def _pool_activation(self, hidden: Any, pooling: str) -> Any:
        """Apply pooling strategy to hidden state."""
        if pooling == "last":
            return hidden[0, -1, :]
        elif pooling == "max":
            return self._backend.max(hidden[0], axis=0)
        else:
            return self._backend.mean(hidden, axis=(0, 1))

    def _tokenize(self, tokenizer: Any, text: str, token_ids: list[int] | None = None) -> list[int]:
        """Tokenize text or return provided token_ids."""
        if token_ids is not None:
            return token_ids
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            return tokens
        return list(tokens.ids)

    # =========================================================================
    # Core collection methods
    # =========================================================================

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, "Array"]:
        """Collect per-layer hidden state activations for a text input."""
        if self._backend_type == "mlx":
            return self._collect_hidden_mlx(model, tokenizer, text, token_ids)
        elif self._backend_type == "cuda":
            return self._collect_hidden_cuda(model, tokenizer, text, token_ids)
        elif self._backend_type == "jax":
            return self._collect_hidden_jax(model, tokenizer, text, token_ids)
        else:
            raise RuntimeError(f"Unknown backend type: {self._backend_type}")

    def collect_embedding_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> "Array":
        """Collect post-embedding activation for a text input."""
        if self._backend_type == "mlx":
            return self._collect_embedding_mlx(model, tokenizer, text, token_ids)
        elif self._backend_type == "cuda":
            return self._collect_embedding_cuda(model, tokenizer, text, token_ids)
        elif self._backend_type == "jax":
            return self._collect_embedding_jax(model, tokenizer, text, token_ids)
        else:
            raise RuntimeError(f"Unknown backend type: {self._backend_type}")

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, "Array"]:
        """Collect per-layer MLP intermediate activations."""
        if self._backend_type == "mlx":
            return self._collect_intermediate_mlx(model, tokenizer, text, token_ids)
        elif self._backend_type == "cuda":
            return self._collect_intermediate_cuda(model, tokenizer, text, token_ids)
        elif self._backend_type == "jax":
            return self._collect_intermediate_jax(model, tokenizer, text, token_ids)
        else:
            raise RuntimeError(f"Unknown backend type: {self._backend_type}")

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"]]:
        """Collect per-layer attention Q and KV activations."""
        if self._backend_type == "mlx":
            return self._collect_attention_mlx(model, tokenizer, text, token_ids)
        elif self._backend_type == "cuda":
            return self._collect_attention_cuda(model, tokenizer, text, token_ids)
        elif self._backend_type == "jax":
            return self._collect_attention_jax(model, tokenizer, text, token_ids)
        else:
            raise RuntimeError(f"Unknown backend type: {self._backend_type}")

    def collect_logits(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> "Array":
        """Collect logits for a text input."""
        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = self._backend.array([token_ids])

        logits = model(input_ids)
        self._backend.eval(logits)

        if logits.ndim == 3:
            last_logits = logits[0, -1, :]
        elif logits.ndim == 2:
            last_logits = logits[0, :]
        else:
            last_logits = logits

        self._backend.eval(last_logits)
        return last_logits

    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> ProbeActivationBatch:
        """Collect hidden + intermediate + gate + embedding activations in one batched pass."""
        if self._backend_type == "mlx":
            return self._collect_probe_batch_mlx(model, tokenizer, texts)
        else:
            # Fallback: sequential collection for non-MLX backends
            hidden: list[dict[int, "Array"]] = []
            intermediate: list[dict[int, "Array"]] = []
            gate: list[dict[int, "Array"]] = []
            embedding: list["Array"] = []

            for text in texts:
                hidden.append(self.collect_hidden_activations(model, tokenizer, text))
                intermediate.append(self.collect_intermediate_activations(model, tokenizer, text))
                gate.append({})
                embedding.append(self.collect_embedding_activations(model, tokenizer, text))

            return ProbeActivationBatch(
                hidden=hidden,
                intermediate=intermediate,
                gate=gate,
                embedding=embedding,
            )

    def collect_hidden_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """Collect per-layer hidden state activations for multiple texts."""
        if self._backend_type == "mlx":
            return self._collect_hidden_batch_mlx(model, tokenizer, texts)
        else:
            return [self.collect_hidden_activations(model, tokenizer, t) for t in texts]

    def collect_intermediate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """Collect per-layer MLP intermediate activations for multiple texts."""
        if self._backend_type == "mlx":
            return self._collect_intermediate_batch_mlx(model, tokenizer, texts)
        else:
            return [self.collect_intermediate_activations(model, tokenizer, t) for t in texts]

    def collect_gate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """Collect per-layer PRE-SiLU gate activations for multiple texts."""
        if self._backend_type == "mlx":
            return self._collect_gate_batch_mlx(model, tokenizer, texts)
        else:
            return [{} for _ in texts]

    def collect_trajectory_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> TrajectoryActivations:
        """Collect full trajectory activations for geometric manifold mapping."""
        if self._backend_type == "mlx":
            return self._collect_trajectory_mlx(model, tokenizer, texts)
        else:
            raise NotImplementedError(f"Trajectory collection not implemented for {self._backend_type}")

    # =========================================================================
    # MLX implementations
    # =========================================================================

    def _collect_hidden_mlx(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> dict[int, "Array"]:
        """MLX implementation of hidden activation collection."""
        import mlx.core as mx

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = mx.array([token_ids])
        activations: dict[int, "Array"] = {}
        pool_strategy = self._resolve_pooling(model)

        try:
            if hasattr(model, "forward_with_hidden_states"):
                _, hidden_states = model.forward_with_hidden_states(input_ids)
                for layer_idx, hidden in enumerate(hidden_states):
                    pooled = self._pool_activation(hidden, pool_strategy)
                    mx.eval(pooled)
                    activations[layer_idx] = pooled
                return activations

            try:
                arch = self._get_architecture(model)
                h = arch.embed_module(input_ids)

                for layer_idx, layer in enumerate(arch.layers):
                    result = layer(h)
                    h = result[0] if isinstance(result, tuple) else result
                    pooled = self._pool_activation(h, pool_strategy)
                    mx.eval(pooled)
                    activations[layer_idx] = pooled

            except ValueError:
                output = model(input_ids)
                mx.eval(output)
                pooled = self._pool_activation(output, pool_strategy)
                mx.eval(pooled)
                activations[0] = pooled

        except Exception as e:
            logger.warning("Activation collection failed for text '%s...': %s", text[:30], e)

        return activations

    def _collect_embedding_mlx(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> "Array":
        """MLX implementation of embedding activation collection."""
        import mlx.core as mx

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = mx.array([token_ids])
        pool_strategy = self._resolve_pooling(model)

        try:
            arch = self._get_architecture(model)
            h = arch.embed_module(input_ids)
        except ValueError as e:
            raise RuntimeError(f"Cannot find embedding layer: {e}") from e

        if h is None:
            raise RuntimeError("No embedding output collected.")

        pooled = self._pool_activation(h, pool_strategy)
        mx.eval(pooled)
        return pooled

    def _collect_intermediate_mlx(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> dict[int, "Array"]:
        """MLX implementation of intermediate activation collection."""
        import mlx.core as mx
        from mlx import nn

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = mx.array([token_ids])
        activations: dict[int, "Array"] = {}
        pool_strategy = self._resolve_pooling(model)

        try:
            arch = self._get_architecture(model)
            h = arch.embed_module(input_ids)

            for layer_idx in range(arch.num_layers):
                accessor = arch.layer_accessor(layer_idx)

                input_norm = accessor.input_norm
                h_norm = input_norm(h) if input_norm is not None else h

                attn = accessor.attention
                if attn is not None:
                    attn_out = attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = mx.zeros_like(h)

                h = h + attn_out

                post_norm = accessor.post_attn_norm
                h_post = post_norm(h) if post_norm is not None else h

                ff_module = accessor.mlp
                if ff_module is not None:
                    if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                        up = ff_module.up_proj(h_post)
                        gate = ff_module.gate_proj(h_post)
                        intermediate = nn.silu(gate) * up
                    elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                        gate = ff_module.w1(h_post)
                        up = ff_module.w3(h_post)
                        intermediate = nn.silu(gate) * up
                    elif hasattr(ff_module, "fc1"):
                        intermediate = ff_module.fc1(h_post)
                    else:
                        continue

                    mx.eval(intermediate)
                    pooled = self._pool_activation(intermediate, pool_strategy)
                    mx.eval(pooled)
                    activations[layer_idx] = pooled

                if ff_module is not None:
                    mlp_out = ff_module(h_post)
                    h = h + mlp_out
                else:
                    layer = arch.layers[layer_idx]
                    result = layer(h)
                    h = result[0] if isinstance(result, tuple) else result

        except ValueError:
            logger.debug("Model structure not compatible with intermediate activation collection")
        except Exception as e:
            logger.warning("Intermediate activation collection failed: %s", e)

        return activations

    def _collect_attention_mlx(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"], dict[int, "Array"]]:
        """MLX implementation of attention activation collection."""
        import mlx.core as mx

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = mx.array([token_ids])
        q_activations: dict[int, "Array"] = {}
        k_activations: dict[int, "Array"] = {}
        v_activations: dict[int, "Array"] = {}
        pool_strategy = self._resolve_pooling(model)

        try:
            arch = self._get_architecture(model)
            h = arch.embed_module(input_ids)

            for layer_idx in range(arch.num_layers):
                accessor = arch.layer_accessor(layer_idx)

                input_norm = accessor.input_norm
                h_norm = input_norm(h) if input_norm is not None else h

                attn = accessor.attention
                if attn is not None and hasattr(attn, "q_proj"):
                    q = attn.q_proj(h_norm)
                    k = attn.k_proj(h_norm)
                    v = attn.v_proj(h_norm)
                    mx.eval(q, k, v)

                    q_activations[layer_idx] = self._pool_activation(q, pool_strategy)
                    k_activations[layer_idx] = self._pool_activation(k, pool_strategy)
                    v_activations[layer_idx] = self._pool_activation(v, pool_strategy)

                layer = arch.layers[layer_idx]
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result

        except ValueError:
            logger.debug("Model structure not compatible with attention activation collection")
        except Exception as e:
            logger.warning("Attention activation collection failed: %s", e)

        return q_activations, k_activations, v_activations

    def _collect_probe_batch_mlx(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> ProbeActivationBatch:
        """MLX batched probe collection."""
        import mlx.core as mx
        from mlx import nn

        if not texts:
            return ProbeActivationBatch(hidden=[], intermediate=[], gate=[], embedding=[])

        all_token_ids: list[list[int]] = []
        for text in texts:
            all_token_ids.append(self._tokenize(tokenizer, text, None))

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
            raise RuntimeError("Cannot find embedding layer.")

        seq_lengths = [len(ids) for ids in all_token_ids]
        lengths = mx.array(seq_lengths, dtype=h.dtype)
        pos = mx.arange(max_len)
        pad_mask = pos[None, :] < lengths[:, None]
        mask = pad_mask.astype(h.dtype)
        denom = lengths[:, None]

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
                    causal = pos[:, None] >= pos[None, :]
                    attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
                    attn_mask = attn_mask[:, None, :, :]
                    attn_out = layer.self_attn(h_norm, mask=attn_mask, cache=None)
                elif hasattr(layer, "conv"):
                    attn_out = layer.conv(h_norm, mask=pad_mask, cache=None)
                else:
                    raise RuntimeError("LFM2 block missing attention/conv module.")
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
                    raise RuntimeError("Model attention block not found.")

                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                else:
                    h_post = h

                ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

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
                raise RuntimeError("Model MLP block not found.")

            pooled_intermediate = mx.sum(intermediate * mask[:, :, None], axis=1) / denom
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
                    raise RuntimeError("Model MLP output path not found.")

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

    def _collect_hidden_batch_mlx(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """MLX batched hidden activation collection."""
        import mlx.core as mx

        if not texts:
            return []

        all_token_ids = [self._tokenize(tokenizer, t, None) for t in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]
        all_tensors = []

        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
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
                        h = result[0] if isinstance(result, tuple) else result
                        for i in range(batch_size):
                            seq_len = len(all_token_ids[i])
                            pooled = mx.mean(h[i, :seq_len, :], axis=0)
                            results[i][layer_idx] = pooled
                            all_tensors.append(pooled)

            if all_tensors:
                mx.eval(*all_tensors)

        except Exception as e:
            raise RuntimeError(f"Batch activation collection failed: {e}") from e

        return results

    def _collect_intermediate_batch_mlx(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """MLX batched intermediate activation collection."""
        import mlx.core as mx
        from mlx import nn

        if not texts:
            return []

        all_token_ids = [self._tokenize(tokenizer, t, None) for t in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, "Array"]] = [{} for _ in range(batch_size)]
        all_tensors = []

        try:
            if not (hasattr(model, "model") and hasattr(model.model, "layers")):
                raise RuntimeError("Model not compatible with batch intermediate collection")

            inner = model.model
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                raise RuntimeError("Cannot find embedding layer")

            for layer_idx, layer in enumerate(inner.layers):
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

                ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

                if ff_module is not None:
                    if hasattr(ff_module, "up_proj") and hasattr(ff_module, "gate_proj"):
                        up = ff_module.up_proj(h_post)
                        gate = ff_module.gate_proj(h_post)
                        intermediate = nn.silu(gate) * up
                    elif hasattr(ff_module, "w1") and hasattr(ff_module, "w3"):
                        gate = ff_module.w1(h_post)
                        up = ff_module.w3(h_post)
                        intermediate = nn.silu(gate) * up
                    elif hasattr(ff_module, "fc1"):
                        intermediate = ff_module.fc1(h_post)
                    else:
                        intermediate = None

                    if intermediate is not None:
                        for i in range(batch_size):
                            seq_len = len(all_token_ids[i])
                            pooled = mx.mean(intermediate[i, :seq_len, :], axis=0)
                            results[i][layer_idx] = pooled
                            all_tensors.append(pooled)

                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                    h = h + mlp_out
                elif hasattr(layer, "feed_forward"):
                    ff_out = layer.feed_forward(h_post)
                    h = h + ff_out
                else:
                    result = layer(h)
                    h = result[0] if isinstance(result, tuple) else result

            if all_tensors:
                mx.eval(*all_tensors)

        except Exception as e:
            raise RuntimeError(f"Batch intermediate collection failed: {e}") from e

        return results

    def _collect_gate_batch_mlx(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, "Array"]]:
        """MLX batched gate activation collection."""
        import mlx.core as mx

        if not texts:
            return []

        all_token_ids = [self._tokenize(tokenizer, t, None) for t in texts]
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
                raise RuntimeError("Cannot find embedding layer")

            seq_lengths = [len(ids) for ids in all_token_ids]
            lengths = mx.array(seq_lengths, dtype=h.dtype)
            pos = mx.arange(max_len)
            pad_mask = pos[None, :] < lengths[:, None]
            mask = pad_mask.astype(h.dtype)
            denom = lengths[:, None]

            for layer_idx, layer in enumerate(inner.layers):
                is_lfm2 = (
                    hasattr(layer, "operator_norm")
                    and hasattr(layer, "ffn_norm")
                    and hasattr(layer, "feed_forward")
                )

                if is_lfm2:
                    h_norm = layer.operator_norm(h)
                    if hasattr(layer, "self_attn"):
                        causal = pos[:, None] >= pos[None, :]
                        attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
                        attn_mask = attn_mask[:, None, :, :]
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

                    ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

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

                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                    h = h + mlp_out
                elif hasattr(layer, "feed_forward"):
                    ff_out = layer.feed_forward(h_post)
                    h = h + ff_out
                else:
                    result = layer(h)
                    h = h + (result[0] if isinstance(result, tuple) else result)

            if all_tensors:
                mx.eval(*all_tensors)

        except Exception as e:
            raise RuntimeError(f"Batch gate collection failed: {e}") from e

        return results

    def _collect_trajectory_mlx(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> TrajectoryActivations:
        """MLX trajectory collection for geometric manifold mapping."""
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

        all_token_ids = [self._tokenize(tokenizer, t, None) for t in texts]
        text_lengths = [len(ids) for ids in all_token_ids]
        total_tokens = sum(text_lengths)
        n_texts = len(texts)

        max_len = max(text_lengths)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = mx.array(padded)

        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise RuntimeError("Model structure not compatible with trajectory collection.")

        inner = model.model

        if hasattr(inner, "embed_tokens"):
            h = inner.embed_tokens(input_ids)
        elif hasattr(inner, "wte"):
            h = inner.wte(input_ids)
        else:
            raise RuntimeError("Cannot find embedding layer.")

        embedding_positions_list = []
        for i in range(n_texts):
            seq_len = text_lengths[i]
            embedding_positions_list.append(h[i, :seq_len, :])
        embedding_positions = mx.concatenate(embedding_positions_list, axis=0)

        positions: dict[int, "Array"] = {}
        velocities: dict[int, "Array"] = {}
        intermediate_positions: dict[int, "Array"] = {}
        q_positions: dict[int, "Array"] = {}
        k_positions: dict[int, "Array"] = {}
        v_positions: dict[int, "Array"] = {}
        gate_positions: dict[int, "Array"] = {}
        all_tensors: list["Array"] = [embedding_positions]

        seq_lengths_arr = mx.array(text_lengths, dtype=h.dtype)
        pos = mx.arange(max_len)
        pad_mask = pos[None, :] < seq_lengths_arr[:, None]
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
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                else:
                    h_norm = h

                attn_module = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn_module is not None:
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

                ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

            # Q/K/V extraction
            if attn_module is not None and hasattr(attn_module, "q_proj"):
                q = attn_module.q_proj(h_norm)
                k = attn_module.k_proj(h_norm)
                v = attn_module.v_proj(h_norm)

                q_pos_list, k_pos_list, v_pos_list = [], [], []
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

            # Intermediate and gate extraction
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

            if intermediate is not None:
                int_pos_list = []
                for i in range(n_texts):
                    seq_len = text_lengths[i]
                    int_pos_list.append(intermediate[i, :seq_len, :])
                layer_int_pos = mx.concatenate(int_pos_list, axis=0)
                intermediate_positions[layer_idx] = layer_int_pos
                all_tensors.append(layer_int_pos)

            if gate is not None:
                gate_pos_list = []
                for i in range(n_texts):
                    seq_len = text_lengths[i]
                    gate_pos_list.append(gate[i, :seq_len, :])
                layer_gate_pos = mx.concatenate(gate_pos_list, axis=0)
                gate_positions[layer_idx] = layer_gate_pos
                all_tensors.append(layer_gate_pos)

            if mlp_out is None:
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                elif hasattr(layer, "feed_forward"):
                    mlp_out = layer.feed_forward(h_post)
                else:
                    raise RuntimeError("Cannot complete MLP forward pass")

            h = h + mlp_out

            # Hidden state positions and velocities
            layer_positions_list = []
            layer_velocities_list = []

            for i in range(n_texts):
                seq_len = text_lengths[i]
                text_positions = h[i, :seq_len, :]
                layer_positions_list.append(text_positions)

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

    # =========================================================================
    # CUDA implementations
    # =========================================================================

    def _collect_hidden_cuda(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> dict[int, "Array"]:
        """CUDA implementation of hidden activation collection."""
        import torch

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = torch.tensor([token_ids], device="cuda")
        activations: dict[int, "Array"] = {}

        try:
            if hasattr(model, "forward") and hasattr(model, "config"):
                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                        for layer_idx, hidden in enumerate(outputs.hidden_states):
                            activations[layer_idx] = hidden.mean(dim=(0, 1))
                        return activations

            if hasattr(model, "model") and hasattr(model.model, "layers"):
                hook_outputs: dict[int, Any] = {}

                def make_hook(layer_idx: int):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            output = output[0]
                        hook_outputs[layer_idx] = output.mean(dim=(0, 1)).detach()
                    return hook

                handles = []
                for layer_idx, layer in enumerate(model.model.layers):
                    handles.append(layer.register_forward_hook(make_hook(layer_idx)))

                try:
                    with torch.no_grad():
                        _ = model(input_ids)
                    activations = hook_outputs
                finally:
                    for handle in handles:
                        handle.remove()

        except Exception as e:
            logger.warning("Activation collection failed: %s", e)

        return activations

    def _collect_embedding_cuda(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> "Array":
        """CUDA implementation of embedding activation collection."""
        import torch

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = torch.tensor([token_ids], device="cuda")

        if hasattr(model, "forward") and hasattr(model, "config"):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    return outputs.hidden_states[0].mean(dim=(0, 1))

        raise RuntimeError("Embedding extraction not supported for this model type")

    def _collect_intermediate_cuda(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> dict[int, "Array"]:
        """CUDA implementation of intermediate activation collection."""
        import torch

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = torch.tensor([token_ids], device="cuda")
        activations: dict[int, "Array"] = {}

        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                hook_outputs: dict[int, Any] = {}

                def make_gate_hook(layer_idx: int):
                    def hook(module, input, output):
                        hook_outputs[layer_idx] = output.mean(dim=(0, 1)).detach()
                    return hook

                handles = []
                for layer_idx, layer in enumerate(model.model.layers):
                    mlp = getattr(layer, "mlp", None)
                    if mlp is not None:
                        if hasattr(mlp, "gate_proj"):
                            handles.append(mlp.gate_proj.register_forward_hook(make_gate_hook(layer_idx)))
                        elif hasattr(mlp, "fc1"):
                            handles.append(mlp.fc1.register_forward_hook(make_gate_hook(layer_idx)))

                try:
                    with torch.no_grad():
                        _ = model(input_ids)
                    activations = hook_outputs
                finally:
                    for handle in handles:
                        handle.remove()

        except Exception as e:
            logger.warning("Intermediate activation collection failed: %s", e)

        return activations

    def _collect_attention_cuda(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"]]:
        """CUDA implementation of attention activation collection."""
        import torch

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = torch.tensor([token_ids], device="cuda")
        q_activations: dict[int, "Array"] = {}
        kv_activations: dict[int, "Array"] = {}

        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                q_outputs: dict[int, Any] = {}
                k_outputs: dict[int, Any] = {}

                def make_q_hook(layer_idx: int):
                    def hook(module, input, output):
                        q_outputs[layer_idx] = output.mean(dim=(0, 1)).detach()
                    return hook

                def make_k_hook(layer_idx: int):
                    def hook(module, input, output):
                        k_outputs[layer_idx] = output.mean(dim=(0, 1)).detach()
                    return hook

                handles = []
                for layer_idx, layer in enumerate(model.model.layers):
                    attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                    if attn is not None:
                        if hasattr(attn, "q_proj"):
                            handles.append(attn.q_proj.register_forward_hook(make_q_hook(layer_idx)))
                        if hasattr(attn, "k_proj"):
                            handles.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))

                try:
                    with torch.no_grad():
                        _ = model(input_ids)
                    q_activations = q_outputs
                    kv_activations = k_outputs
                finally:
                    for handle in handles:
                        handle.remove()

        except Exception as e:
            logger.warning("Attention activation collection failed: %s", e)

        return q_activations, kv_activations

    # =========================================================================
    # JAX implementations
    # =========================================================================

    def _collect_hidden_jax(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> dict[int, "Array"]:
        """JAX implementation of hidden activation collection."""
        import jax.numpy as jnp

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = jnp.array([token_ids])
        activations: dict[int, "Array"] = {}

        try:
            if hasattr(model, "__call__") and hasattr(model, "config"):
                outputs = model(input_ids, output_hidden_states=True)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    for layer_idx, hidden in enumerate(outputs.hidden_states):
                        activations[layer_idx] = jnp.mean(hidden, axis=(0, 1))
                    return activations

            if hasattr(model, "forward_with_hidden_states"):
                _, hidden_states = model.forward_with_hidden_states(input_ids)
                for layer_idx, hidden in enumerate(hidden_states):
                    activations[layer_idx] = jnp.mean(hidden, axis=(0, 1))
                return activations

        except Exception as e:
            logger.warning("Activation collection failed: %s", e)

        return activations

    def _collect_embedding_jax(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> "Array":
        """JAX implementation of embedding activation collection."""
        import jax.numpy as jnp

        token_ids = self._tokenize(tokenizer, text, token_ids)
        input_ids = jnp.array([token_ids])

        if hasattr(model, "__call__") and hasattr(model, "config"):
            outputs = model(input_ids, output_hidden_states=True)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                return jnp.mean(outputs.hidden_states[0], axis=(0, 1))

        raise RuntimeError("Embedding extraction not supported for this JAX model")

    def _collect_intermediate_jax(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> dict[int, "Array"]:
        """JAX implementation of intermediate activation collection."""
        # JAX intermediate extraction requires model surgery - return empty
        return {}

    def _collect_attention_jax(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"]]:
        """JAX implementation of attention activation collection."""
        # JAX attention extraction requires model surgery - return empty
        return {}, {}


def get_activation_provider(backend: "Backend | None" = None) -> ActivationProvider:
    """Get an activation provider instance."""
    return ActivationProvider(backend)


__all__ = ["ActivationProvider", "get_activation_provider"]
