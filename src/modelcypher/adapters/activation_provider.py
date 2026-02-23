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

"""Activation provider adapter using Backend protocol.

This module implements the ActivationProvider port using the Backend abstraction,
making it framework-agnostic across supported runtimes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


class ActivationProviderAdapter:
    """Activation provider adapter using Backend protocol.

    This implementation uses the Backend abstraction to collect activations,
    making it framework-agnostic across supported runtimes.
    """

    def __init__(
        self,
        backend: "Backend",
        model_path: str | Path | None = None,
        pooling: str = "auto",
    ) -> None:
        self._backend = backend
        self._model_path = model_path
        self._pooling = pooling
        self._model = None
        self._tokenizer = None

    def _ensure_model_loaded(self) -> tuple[Any, Any]:
        """Ensure model and tokenizer are loaded."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        if self._model_path is None:
            raise RuntimeError("No model_path provided to ActivationProvider")

        from modelcypher.adapters.model_loader import load_model_for_training

        self._model, self._tokenizer = load_model_for_training(str(self._model_path))
        return self._model, self._tokenizer

    def _get_backbone(self, model: Any) -> tuple[Any, Any, Any]:
        """Get backbone components (embed_tokens, layers, norm) from model."""
        from modelcypher.adapters.model_backbone import resolve_model_backbone

        model_type = getattr(model, "model_type", "unknown")
        resolved = resolve_model_backbone(model, model_type)
        if not resolved:
            raise RuntimeError("Could not resolve model backbone")
        return resolved

    def _pool_activations(self, hidden: Any) -> Any:
        """Pool activations based on pooling strategy."""
        # Mean pooling over sequence dimension
        return self._backend.mean(hidden, axis=0)

    def _forward_through_backbone(
        self,
        input_ids: Any,
        embed_tokens: Any,
        layers: Any,
        norm: Any,
        target_layer: int,
    ) -> Any:
        """Forward pass through backbone to target layer."""
        from modelcypher.adapters.model_backbone import forward_through_backbone

        return forward_through_backbone(
            input_ids,
            embed_tokens,
            layers,
            norm,
            target_layer=target_layer,
            backend=self._backend,
        )

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Any]:
        """Collect per-layer hidden activations for text."""
        embed_tokens, layers, norm = self._get_backbone(model)
        num_layers = len(layers)

        if token_ids is None:
            tokens = tokenizer.encode(text)
        else:
            tokens = token_ids

        input_ids = self._backend.array([tokens])
        results = {}

        for layer_idx in range(num_layers):
            hidden = self._forward_through_backbone(
                input_ids, embed_tokens, layers, norm, target_layer=layer_idx
            )
            pooled = self._pool_activations(hidden[0])
            self._backend.eval(pooled)
            results[layer_idx] = pooled

        return results

    def collect_embedding_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> Any:
        """Collect post-embedding activations for text."""
        embed_tokens, _, _ = self._get_backbone(model)

        if token_ids is None:
            tokens = tokenizer.encode(text)
        else:
            tokens = token_ids

        input_ids = self._backend.array([tokens])
        embedded = embed_tokens(input_ids)
        pooled = self._pool_activations(embedded[0])
        self._backend.eval(pooled)
        return pooled

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Any]:
        """Collect per-layer MLP intermediate activations."""
        # This requires hooking into MLP intermediate outputs
        # For now, return hidden activations as a fallback
        return self.collect_hidden_activations(model, tokenizer, text, token_ids)

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, Any], dict[int, Any]]:
        """Collect per-layer attention Q and KV activations."""
        # This requires hooking into attention outputs
        # For now, return hidden activations as a fallback
        hidden = self.collect_hidden_activations(model, tokenizer, text, token_ids)
        return hidden, hidden

    def collect_attention_matrices(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, list[Any]]:
        """Collect per-layer, per-head attention weight matrices.

        Manually computes softmax(QK^T / sqrt(d_k)) since fused kernels
        don't expose weights. Handles GQA by expanding KV heads via
        expand_dims + tile + reshape (no framework imports).

        Returns:
            Dict mapping attention_layer_idx -> list of [seq_len, seq_len]
            arrays, one per attention head. Only attention layers are included
            (conv layers are skipped).
        """
        b = self._backend
        embed_tokens, layers, norm = self._get_backbone(model)

        if token_ids is None:
            tokens = tokenizer.encode(text)
        else:
            tokens = token_ids

        input_ids = b.array([tokens])
        captured_attn: dict[int, Any] = {}

        class _AttnCaptureWrapper:
            """Wraps a layer to capture attention weights during forward pass."""

            def __init__(self, layer, layer_idx):
                self._layer = layer
                self._layer_idx = layer_idx

            def __call__(self, x, mask=None, cache=None):
                if not getattr(self._layer, "is_attention_layer", False):
                    return self._layer(x, mask=mask, cache=cache)

                attn = self._layer.self_attn
                x_normed = self._layer.operator_norm(x)
                B_val, L, _D = x_normed.shape

                queries = attn.q_proj(x_normed)
                keys = attn.k_proj(x_normed)

                queries = attn.q_layernorm(
                    queries.reshape(B_val, L, attn.n_heads, -1)
                ).transpose(0, 2, 1, 3)
                keys = attn.k_layernorm(
                    keys.reshape(B_val, L, attn.n_kv_heads, -1)
                ).transpose(0, 2, 1, 3)

                if cache is not None:
                    queries = attn.rope(queries, offset=cache.offset)
                    keys = attn.rope(keys, offset=cache.offset)
                else:
                    queries = attn.rope(queries)
                    keys = attn.rope(keys)

                # GQA: expand KV heads to match Q heads
                # keys: [B, n_kv_heads, L, head_dim]
                # target: [B, n_heads, L, head_dim]
                n_rep = attn.n_heads // attn.n_kv_heads
                if n_rep > 1:
                    n_kv = keys.shape[1]
                    head_d = keys.shape[3]
                    keys_expanded = b.expand_dims(keys, axis=2)
                    keys_expanded = b.tile(
                        keys_expanded, (1, 1, n_rep, 1, 1)
                    )
                    keys_expanded = b.reshape(
                        keys_expanded, (B_val, n_kv * n_rep, L, head_d)
                    )
                else:
                    keys_expanded = keys

                # scores = QK^T / sqrt(d_k)
                # attn.scale = 1/sqrt(head_dim), derived from the model's
                # own geometry (head_dim set by architecture)
                scores = b.matmul(
                    queries,
                    b.transpose(keys_expanded, (0, 1, 3, 2)),
                ) * attn.scale

                # Additive causal mask: 0 for allowed, -large for masked
                # Structural constraint of autoregressive computation graph
                causal = b.create_causal_mask(L)
                scores = scores + causal

                weights = b.softmax(scores, axis=-1)
                b.eval(weights)
                captured_attn[self._layer_idx] = weights[0]

                return self._layer(x, mask=mask, cache=cache)

            def __getattr__(self, name):
                return getattr(self._layer, name)

        original_layers = list(layers)
        try:
            for i in range(len(layers)):
                layers[i] = _AttnCaptureWrapper(original_layers[i], i)
            _logits = model(input_ids)
            b.eval(_logits)
        finally:
            for i, layer in enumerate(original_layers):
                layers[i] = layer

        # Convert from [n_heads, L, L] arrays to list of [L, L] per head
        result: dict[int, list[Any]] = {}
        for layer_idx, weights in captured_attn.items():
            n_heads = weights.shape[0]
            result[layer_idx] = [weights[h] for h in range(n_heads)]

        return result

    def collect_logits(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> Any:
        """Collect logits for text."""
        if token_ids is None:
            tokens = tokenizer.encode(text)
        else:
            tokens = token_ids

        input_ids = self._backend.array([tokens])
        logits = model(input_ids)
        self._backend.eval(logits)
        # Return last token's logits
        return logits[0, -1, :]

    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ):
        """Collect activations for multiple texts."""
        from modelcypher.ports.activation_provider import ProbeActivationBatch

        hidden_list = []
        intermediate_list = []
        gate_list = []
        embedding_list = []

        for text in texts:
            hidden = self.collect_hidden_activations(model, tokenizer, text)
            embedding = self.collect_embedding_activations(model, tokenizer, text)
            hidden_list.append(hidden)
            intermediate_list.append(hidden)  # Fallback
            gate_list.append(hidden)  # Fallback
            embedding_list.append(embedding)

        return ProbeActivationBatch(
            hidden=hidden_list,
            intermediate=intermediate_list,
            gate=gate_list,
            embedding=embedding_list,
        )

    def collect_hidden_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect hidden activations for multiple texts."""
        return [self.collect_hidden_activations(model, tokenizer, text) for text in texts]

    def collect_intermediate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect intermediate activations for multiple texts."""
        return [self.collect_intermediate_activations(model, tokenizer, text) for text in texts]

    def collect_gate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect gate activations for multiple texts."""
        # Fallback to hidden activations
        return [self.collect_hidden_activations(model, tokenizer, text) for text in texts]

    def collect_attention_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> tuple[list[dict[int, Any]], list[dict[int, Any]]]:
        """Collect attention activations for multiple texts."""
        q_list = []
        kv_list = []
        for text in texts:
            q, kv = self.collect_attention_activations(model, tokenizer, text)
            q_list.append(q)
            kv_list.append(kv)
        return q_list, kv_list

    def collect_trajectory_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ):
        """Collect full trajectory activations for geometric manifold mapping.

        Runs a single forward pass through all layers, capturing hidden states
        at every token position at every layer. Also computes velocities
        (first differences between consecutive token positions).

        For the trajectory fields that require MLP/attention hooks (intermediate,
        gate, q/k/v), we populate them with empty dicts since the current
        backbone forward doesn't expose those internals.
        """
        from modelcypher.adapters.model_backbone import (
            _apply_layer_with_mask,
            _resolve_layer_mask,
            resolve_model_backbone,
        )
        from modelcypher.ports.activation_provider import TrajectoryActivations

        b = self._backend
        resolved = resolve_model_backbone(model)
        if not resolved:
            raise RuntimeError("Could not resolve model backbone for trajectory collection")
        embed_tokens, layers, norm = resolved
        num_layers = len(layers)

        # Tokenize all texts and concatenate
        all_token_ids: list[list[int]] = []
        text_lengths: list[int] = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if isinstance(tokens, list):
                all_token_ids.append(tokens)
            else:
                all_token_ids.append(b.tolist(tokens))
            text_lengths.append(len(all_token_ids[-1]))

        total_tokens = sum(text_lengths)
        n_texts = len(texts)

        # Process each text separately (variable lengths)
        # Collect per-layer positions: layer_idx -> list of per-text arrays
        layer_positions_parts: dict[int, list[Any]] = {i: [] for i in range(num_layers)}
        embedding_parts: list[Any] = []

        for text_idx, token_ids in enumerate(all_token_ids):
            input_ids = b.array([token_ids])
            hidden = embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
            numeric_mask = b.create_causal_mask(seq_len, hidden.dtype)

            # Save embedding positions (pre-layer-0)
            embedding_parts.append(hidden[0])  # [seq_len, hidden_dim]

            # Forward through all layers, capturing at each
            for layer_idx, layer in enumerate(layers):
                mask = _resolve_layer_mask(layer, numeric_mask)
                hidden = _apply_layer_with_mask(layer, hidden, mask)
                # hidden is [batch=1, seq_len, hidden_dim]
                layer_positions_parts[layer_idx].append(hidden[0])  # [seq_len, hidden_dim]

            # Apply final norm to last layer's output
            if norm is not None:
                normed = norm(hidden)
                layer_positions_parts[num_layers - 1][-1] = normed[0]

        # Concatenate per-text arrays into [total_tokens, hidden_dim] per layer
        positions: dict[int, Any] = {}
        for layer_idx in range(num_layers):
            parts = layer_positions_parts[layer_idx]
            if len(parts) == 1:
                positions[layer_idx] = parts[0]
            else:
                positions[layer_idx] = b.concatenate(parts, axis=0)

        embedding_positions = (
            embedding_parts[0]
            if len(embedding_parts) == 1
            else b.concatenate(embedding_parts, axis=0)
        )

        # Compute velocities: first differences between consecutive tokens
        # Skip boundaries between texts
        velocities: dict[int, Any] = {}
        for layer_idx in range(num_layers):
            vel_parts = []
            offset = 0
            for length in text_lengths:
                if length > 1:
                    text_pos = positions[layer_idx][offset : offset + length]
                    vel_parts.append(text_pos[1:] - text_pos[:-1])
                offset += length

            if vel_parts:
                if len(vel_parts) == 1:
                    velocities[layer_idx] = vel_parts[0]
                else:
                    velocities[layer_idx] = b.concatenate(vel_parts, axis=0)
            else:
                # No velocities possible (all single-token texts)
                hidden_dim = int(positions[layer_idx].shape[-1])
                velocities[layer_idx] = b.zeros((0, hidden_dim))

        # Eval all arrays
        eval_arrays = list(positions.values()) + list(velocities.values()) + [embedding_positions]
        b.eval(*eval_arrays)

        return TrajectoryActivations(
            positions=positions,
            velocities=velocities,
            intermediate_positions={},
            embedding_positions=embedding_positions,
            q_positions={},
            k_positions={},
            v_positions={},
            gate_positions={},
            text_lengths=text_lengths,
            total_tokens=total_tokens,
            n_texts=n_texts,
        )


def get_activation_provider(
    backend: "Backend | None" = None,
    model_path: str | Path | None = None,
    pooling: str = "auto",
) -> ActivationProviderAdapter:
    """Factory for ActivationProviderAdapter."""
    from modelcypher.backends import initialize_default_backend

    if backend is None:
        backend = initialize_default_backend()

    return ActivationProviderAdapter(backend=backend, model_path=model_path, pooling=pooling)


__all__ = ["ActivationProviderAdapter", "get_activation_provider"]
