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

"""Activation collection methods for :class:`MLXBackend`."""

from __future__ import annotations

from typing import Any

from modelcypher.core.domain.geometry.model_utils import resolve_model_base


class _MLXBackendActivationMixin:
    @staticmethod
    def _resolve_model_base(model: Any) -> Any:
        """Return the backbone object that has both .embed_tokens and .layers.

        Delegates to :func:`modelcypher.core.domain.geometry.model_utils.resolve_model_base`.
        """
        return resolve_model_base(model)

    def collect_hidden_activations(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        layer_indices: list[int] | None = None,
    ) -> dict[int, Any]:
        """Collect hidden state activations from model layers.

        Args:
            model: Model object from load_model.
            tokenizer: Tokenizer object from load_model.
            prompts: List of input prompts.
            layer_indices: Optional specific layers to collect (None = all).

        Returns:
            Dictionary mapping layer index to activations [batch, seq, hidden].
        """
        base = self._resolve_model_base(model)
        n_layers = len(base.layers)
        if layer_indices is None:
            layer_indices = list(range(n_layers))

        activations: dict[int, list[Any]] = {i: [] for i in layer_indices}

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = self.mx.array([tokens])

            # Embedding
            hidden = base.embed_tokens(input_ids)
            self.mx.eval(hidden)

            # Forward through layers
            for layer_idx, layer in enumerate(base.layers):
                hidden = layer(hidden, mask=None, cache=None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                self.mx.eval(hidden)

                if layer_idx in layer_indices:
                    activations[layer_idx].append(hidden)

        # Stack activations per layer
        result = {}
        for layer_idx, acts in activations.items():
            if acts:
                result[layer_idx] = self.mx.concatenate(acts, axis=0)
                self.mx.eval(result[layer_idx])

        return result

    def trace_norm_trajectory(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
    ) -> list[float]:
        """Trace the norm of hidden states through all layers.

        Args:
            model: Model object from load_model.
            tokenizer: Tokenizer object from load_model.
            prompt: Input prompt.

        Returns:
            List of norms, one per layer (including embedding).
        """
        tokens = tokenizer.encode(prompt)
        input_ids = self.mx.array([tokens])

        base = self._resolve_model_base(model)

        # Embedding
        hidden = base.embed_tokens(input_ids)
        self.mx.eval(hidden)

        norms = [float(self.mx.sqrt(self.mx.sum(hidden * hidden)).item())]

        # Each layer
        for layer in base.layers:
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            self.mx.eval(hidden)
            norms.append(float(self.mx.sqrt(self.mx.sum(hidden * hidden)).item()))

        return norms

    # --- Neural Network Operations ---

    def silu(self, array: Any) -> Any:
        """SiLU (Swish) activation function: x * sigmoid(x)."""
        from mlx import nn
        return nn.silu(array)

    # --- Memory Management ---

    def get_peak_memory_gb(self) -> float:
        """Get peak GPU memory usage in gigabytes."""
        if hasattr(self.mx, "get_peak_memory"):
            return self.mx.get_peak_memory() / (1024**3)
        return self.mx.metal.get_peak_memory() / (1024**3)

    def reset_peak_memory(self) -> None:
        """Reset peak GPU memory counter."""
        if hasattr(self.mx, "reset_peak_memory"):
            self.mx.reset_peak_memory()
            return
        if hasattr(self.mx, "metal") and hasattr(self.mx.metal, "reset_peak_memory"):
            self.mx.metal.reset_peak_memory()

    def get_active_memory_gb(self) -> float:
        """Get active GPU memory usage in gigabytes."""
        if hasattr(self.mx, "get_active_memory"):
            return self.mx.get_active_memory() / (1024**3)
        return self.mx.metal.get_active_memory() / (1024**3)

    # --- Extended Activation Collection ---

    def collect_embedding_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> Any:
        """Collect post-embedding activation for a text input."""
        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)
        self.mx.eval(h)

        # Mean pool over sequence
        pooled = self.mx.mean(h, axis=(0, 1))
        self.mx.eval(pooled)
        return pooled

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Any]:
        """Collect per-layer MLP intermediate activations."""
        from mlx import nn

        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])
        activations: dict[int, Any] = {}

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(base.layers):
            # Get layer components
            if hasattr(layer, "input_layernorm"):
                h_norm = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_norm = layer.ln_1(h)
            elif hasattr(layer, "operator_norm"):
                h_norm = layer.operator_norm(h)
            else:
                h_norm = h

            # Attention
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is not None:
                attn_out = attn(h_norm)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
            else:
                attn_out = self.mx.zeros_like(h)
            h = h + attn_out

            # Post-attention norm
            if hasattr(layer, "post_attention_layernorm"):
                h_post = layer.post_attention_layernorm(h)
            elif hasattr(layer, "ln_2"):
                h_post = layer.ln_2(h)
            elif hasattr(layer, "ffn_norm"):
                h_post = layer.ffn_norm(h)
            else:
                h_post = h

            # MLP intermediate
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
                    pooled = self.mx.mean(intermediate, axis=(0, 1))
                    self.mx.eval(pooled)
                    activations[layer_idx] = pooled

                # Complete MLP forward
                mlp_out = ff_module(h_post)
                h = h + mlp_out
            else:
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result

        return activations

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
        """Collect per-layer attention Q, K, V activations."""
        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])
        q_activations: dict[int, Any] = {}
        k_activations: dict[int, Any] = {}
        v_activations: dict[int, Any] = {}

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(base.layers):
            # Get layer norm
            if hasattr(layer, "input_layernorm"):
                h_norm = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_norm = layer.ln_1(h)
            elif hasattr(layer, "operator_norm"):
                h_norm = layer.operator_norm(h)
            else:
                h_norm = h

            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is not None and hasattr(attn, "q_proj"):
                q = attn.q_proj(h_norm)
                k = attn.k_proj(h_norm)
                v = attn.v_proj(h_norm)
                self.mx.eval(q, k, v)

                q_activations[layer_idx] = self.mx.mean(q, axis=(0, 1))
                k_activations[layer_idx] = self.mx.mean(k, axis=(0, 1))
                v_activations[layer_idx] = self.mx.mean(v, axis=(0, 1))

            # Forward through layer
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result

        return q_activations, k_activations, v_activations

    def _collect_attention_layer_results(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
        *,
        include_weights: bool = True,
        include_values: bool = False,
        include_scores: bool = False,
    ) -> list[tuple[int, list[Any] | None, list[Any] | None, list[Any] | None]]:
        """Shared core for attention matrix extraction.

        Runs a full forward pass, extracting per-head attention weights,
        pre-softmax score matrices, and/or value vectors from each attention
        layer. Conv/non-attention layers are skipped.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding text.
            text: The text input to process.
            token_ids: Optional pre-tokenized input.
            include_weights: If True, extract post-softmax attention weights.
            include_values: If True, also extract per-head V vectors.
            include_scores: If True, also extract pre-softmax attention scores.

        Returns:
            List of
            (layer_idx, head_weights_list_or_none, head_values_list_or_none,
            head_scores_list_or_none).
        """
        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])

        results: list[tuple[int, list[Any] | None, list[Any] | None, list[Any] | None]] = []
        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]

        for layer_idx, layer in enumerate(base.layers):
            # Detect attention module
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is not None and hasattr(attn, "q_proj"):
                # Get pre-attention norm
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                elif hasattr(layer, "operator_norm"):
                    h_norm = layer.operator_norm(h)
                else:
                    h_norm = h

                # Project Q and K (and V if requested)
                q = attn.q_proj(h_norm)
                k = attn.k_proj(h_norm)
                v = attn.v_proj(h_norm) if include_values else None

                # Determine head counts and dimensions.
                # Qwen2/Llama use "num_heads"; Qwen3.5 uses "num_attention_heads";
                # LFM2 uses "n_heads" / "n_kv_heads".
                num_heads = (
                    getattr(attn, "num_heads", None)
                    or getattr(attn, "num_attention_heads", None)
                    or getattr(attn, "n_heads", None)
                )
                num_kv_heads = (
                    getattr(attn, "num_key_value_heads", None)
                    or getattr(attn, "n_kv_heads", None)
                )
                if num_kv_heads is None:
                    num_kv_heads = num_heads
                if num_heads is None:
                    # Both num_heads and num_attention_heads are missing, and
                    # num_kv_heads was None too (set to num_heads above which is
                    # None).  Cannot determine head configuration.
                    q_dim = q.shape[-1]
                    k_dim = k.shape[-1]
                    raise ValueError(
                        f"Cannot determine head configuration for attention at "
                        f"layer {layer_idx}: no num_heads or num_attention_heads "
                        f"attribute found, and num_kv_heads is also unknown. "
                        f"q_proj output dim={q_dim}, k_proj output dim={k_dim}."
                    )

                # Derive head_dim from attn.head_dim when available (Qwen3.5),
                # else from k_proj output (safe for gated-Q architectures where
                # q_proj output = num_heads * head_dim * 2).
                head_dim = getattr(attn, "head_dim", None)
                if head_dim is None:
                    head_dim = k.shape[-1] // num_kv_heads

                # Qwen3.5 Attention uses a gated Q projection: q_proj outputs
                # num_heads * head_dim * 2, which is split into queries + gate.
                # Detect this by comparing q_proj output to expected Q size.
                batch = q.shape[0]
                expected_q_dim = num_heads * head_dim
                if q.shape[-1] == expected_q_dim * 2:
                    # Gated Q: reshape to [batch, seq, num_heads, head_dim*2],
                    # then split along last dim into queries and gate.
                    q = q.reshape(batch, seq_len, num_heads, head_dim * 2)
                    q, _gate = self.mx.split(q, 2, axis=-1)
                    # q is now [batch, seq, num_heads, head_dim]

                # Reshape: [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]

                # Apply Q/K normalization if present.
                # LFM2: q_layernorm / k_layernorm
                # Qwen3.5: q_norm / k_norm
                # Order: project -> reshape -> norm -> transpose -> RoPE.
                q_ln = (
                    getattr(attn, "q_layernorm", None)
                    or getattr(attn, "q_norm", None)
                )
                k_ln = (
                    getattr(attn, "k_layernorm", None)
                    or getattr(attn, "k_norm", None)
                )
                # Reshape Q/K/V to [batch, seq, num_heads, head_dim] if still 3D.
                # (Gated-Q path above already reshaped q to 4D.)
                if q.ndim == 3:
                    q = q.reshape(batch, seq_len, num_heads, head_dim)
                k = k.reshape(batch, seq_len, num_kv_heads, head_dim)

                if q_ln is not None or k_ln is not None:
                    if q_ln is not None:
                        q = q_ln(q)
                    if k_ln is not None:
                        k = k_ln(k)

                # Transpose to [batch, num_heads, seq, head_dim]
                q = q.transpose(0, 2, 1, 3)
                k = k.transpose(0, 2, 1, 3)
                if v is not None:
                    v = v.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(
                        0, 2, 1, 3
                    )

                # Apply RoPE (Rotary Position Embeddings) to Q and K.
                # Every standard attention module (Qwen2, Qwen3.5, Llama, LFM2)
                # stores RoPE as attn.rope and applies it after reshape, before
                # computing scores. Without RoPE the positional structure of
                # attention is wrong and downstream SVD results are meaningless.
                rope_fn = getattr(attn, "rope", None)
                if rope_fn is not None:
                    q = rope_fn(q)
                    k = rope_fn(k)

                # GQA expansion: repeat K (and V) heads to match Q head count
                if num_kv_heads < num_heads:
                    repeats = num_heads // num_kv_heads
                    k = self.mx.repeat(k, repeats, axis=1)
                    if v is not None:
                        v = self.mx.repeat(v, repeats, axis=1)

                # Compute attention scores: Q @ K^T / sqrt(head_dim)
                scale = head_dim ** -0.5
                scores = (q @ k.transpose(0, 1, 3, 2)) * scale

                # Causal mask: upper triangle = -inf
                causal_mask = self.mx.tril(self.mx.ones((seq_len, seq_len)))
                neg_inf = self.mx.array(float("-inf"))
                scores = self.mx.where(
                    causal_mask[None, None, :, :], scores, neg_inf
                )

                weights = self.mx.softmax(scores, axis=-1) if include_weights else None

                eval_args: list[Any] = []
                if weights is not None:
                    eval_args.append(weights)
                if v is not None:
                    eval_args.append(v)
                if include_scores:
                    eval_args.append(scores)
                if eval_args:
                    self.mx.eval(*eval_args)

                # Split into per-head [seq, seq] matrices (and value vectors)
                head_weights: list[Any] | None = [] if include_weights else None
                head_values: list[Any] | None = [] if include_values else None
                head_scores: list[Any] | None = [] if include_scores else None
                for head_i in range(num_heads):
                    if head_weights is not None and weights is not None:
                        head_weights.append(weights[0, head_i, :, :])
                    if head_values is not None:
                        head_values.append(v[0, head_i, :, :])
                    if head_scores is not None:
                        head_scores.append(scores[0, head_i, :, :])
                results.append((layer_idx, head_weights, head_values, head_scores))

            # Forward through layer so subsequent layers get correct hidden
            # states.  Attention layers need mask="causal" so internal SDPA
            # uses a proper causal mask (mask=None means unmasked, not
            # causal).  Conv layers (LFM2) cannot accept mask="causal" —
            # they expect None or an actual array mask.
            is_attn = attn is not None and hasattr(attn, "q_proj")
            layer_mask = "causal" if is_attn else None
            result = layer(h, mask=layer_mask, cache=None)
            h = result[0] if isinstance(result, tuple) else result

        return results

    def collect_attention_matrices(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, list[Any]]:
        """Collect per-layer, per-head attention weight matrices.

        Extracts the softmax(QK^T / sqrt(d_k)) attention weight matrices
        from each attention layer. Conv/non-attention layers are skipped.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding text.
            text: The text input to process.
            token_ids: Optional pre-tokenized input.

        Returns:
            Dict mapping layer_idx -> list of [seq_len, seq_len]
            arrays, one per attention head.
        """
        layer_results = self._collect_attention_layer_results(
            model,
            tokenizer,
            text,
            token_ids,
            include_weights=True,
            include_values=False,
            include_scores=False,
        )
        return {
            layer_idx: weights
            for layer_idx, weights, _, _ in layer_results
            if weights is not None
        }

    def collect_attention_score_matrices(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, list[Any]]:
        """Collect per-layer, per-head pre-softmax attention score matrices.

        Returns the causal-masked `QK^T / sqrt(d_k)` score matrices before
        row-wise softmax normalization.
        """
        layer_results = self._collect_attention_layer_results(
            model,
            tokenizer,
            text,
            token_ids,
            include_weights=False,
            include_values=False,
            include_scores=True,
        )
        return {
            layer_idx: scores
            for layer_idx, _, _, scores in layer_results
            if scores is not None
        }

    def collect_attention_matrices_with_values(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, list[Any]], dict[int, list[Any]]]:
        """Like collect_attention_matrices but also returns per-head V vectors.

        Returns:
            (attention_matrices, value_vectors) where:
            - attention_matrices: dict[layer_idx, list of [seq, seq]]
            - value_vectors: dict[layer_idx, list of [seq, head_dim]]
        """
        layer_results = self._collect_attention_layer_results(
            model,
            tokenizer,
            text,
            token_ids,
            include_weights=True,
            include_values=True,
            include_scores=False,
        )
        attn_result: dict[int, list[Any]] = {}
        val_result: dict[int, list[Any]] = {}
        for layer_idx, weights, values, _ in layer_results:
            if weights is not None:
                attn_result[layer_idx] = weights
            if values is not None:
                val_result[layer_idx] = values
        return attn_result, val_result

    def collect_hidden_with_attention_hook(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        attention_hook: Any | None = None,
        token_ids: list[int] | None = None,
    ) -> dict[int, Any]:
        """Forward pass with optional attention weight modification.

        Runs the model, manually decomposing attention layers to apply the hook
        to post-softmax weights before computing output. Non-attention layers
        pass through normally.

        For attention layers, the decomposition is:
            norm → Q,K,V → scores → softmax → **hook** → output=W@V → o_proj
            → residual → post_attn_norm → MLP → residual

        Args:
            model: The loaded model.
            tokenizer: The tokenizer.
            text: Input text.
            attention_hook: Optional callable(weights, layer_idx) -> weights.
                weights shape: [batch, num_heads, seq, seq].
                If None, runs a normal forward pass (baseline).
            token_ids: Optional pre-tokenized input.

        Returns:
            Dict mapping layer_idx -> mean-pooled hidden state [hidden_dim].
        """
        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        hidden_states: dict[int, Any] = {}

        for layer_idx, layer in enumerate(base.layers):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            is_attn = attn is not None and hasattr(attn, "q_proj")

            if is_attn and attention_hook is not None:
                # Manual attention decomposition with hook
                # Step 1: Pre-attention norm
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                elif hasattr(layer, "operator_norm"):
                    h_norm = layer.operator_norm(h)
                else:
                    h_norm = h

                # Step 2: Q, K, V projections
                q = attn.q_proj(h_norm)
                k = attn.k_proj(h_norm)
                v = attn.v_proj(h_norm)

                # Head counts
                num_heads = (
                    getattr(attn, "num_heads", None)
                    or getattr(attn, "num_attention_heads", None)
                    or getattr(attn, "n_heads", None)
                )
                num_kv_heads = (
                    getattr(attn, "num_key_value_heads", None)
                    or getattr(attn, "n_kv_heads", None)
                )
                if num_kv_heads is None:
                    num_kv_heads = num_heads

                head_dim = getattr(attn, "head_dim", None)
                if head_dim is None:
                    head_dim = k.shape[-1] // num_kv_heads

                batch = q.shape[0]
                expected_q_dim = num_heads * head_dim

                # Handle gated Q (Qwen3.5)
                gate = None
                if q.shape[-1] == expected_q_dim * 2:
                    q = q.reshape(batch, seq_len, num_heads, head_dim * 2)
                    q, gate = self.mx.split(q, 2, axis=-1)
                    gate = gate.reshape(batch, seq_len, -1)

                # Reshape Q/K/V
                if q.ndim == 3:
                    q = q.reshape(batch, seq_len, num_heads, head_dim)
                k = k.reshape(batch, seq_len, num_kv_heads, head_dim)
                v = v.reshape(batch, seq_len, num_kv_heads, head_dim)

                # Q/K normalization (LFM2: q_layernorm/k_layernorm, Qwen3.5: q_norm/k_norm)
                q_ln = getattr(attn, "q_layernorm", None) or getattr(attn, "q_norm", None)
                k_ln = getattr(attn, "k_layernorm", None) or getattr(attn, "k_norm", None)
                if q_ln is not None:
                    q = q_ln(q)
                if k_ln is not None:
                    k = k_ln(k)

                # Transpose to [batch, heads, seq, dim]
                q = q.transpose(0, 2, 1, 3)
                k = k.transpose(0, 2, 1, 3)
                v = v.transpose(0, 2, 1, 3)

                # RoPE
                rope_fn = getattr(attn, "rope", None)
                if rope_fn is not None:
                    q = rope_fn(q)
                    k = rope_fn(k)

                # GQA expansion
                if num_kv_heads < num_heads:
                    repeats = num_heads // num_kv_heads
                    k = self.mx.repeat(k, repeats, axis=1)
                    v = self.mx.repeat(v, repeats, axis=1)

                # Attention scores
                scale = head_dim ** -0.5
                scores = (q @ k.transpose(0, 1, 3, 2)) * scale

                # Causal mask
                causal_mask = self.mx.tril(self.mx.ones((seq_len, seq_len)))
                neg_inf = self.mx.array(float("-inf"))
                scores = self.mx.where(causal_mask[None, None, :, :], scores, neg_inf)

                # Softmax
                weights = self.mx.softmax(scores, axis=-1)

                # Apply hook
                weights = attention_hook(weights, layer_idx)

                # Output = weights @ V -> [batch, heads, seq, dim]
                attn_output = weights @ v
                attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
                    batch, seq_len, -1
                )

                # Gate (Qwen3.5)
                if gate is not None:
                    attn_output = attn_output * self.mx.sigmoid(gate)

                # O projection (o_proj for Qwen/Llama, out_proj for LFM2)
                o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
                attn_output = o_proj(attn_output)

                # Residual connection
                h = h + attn_output

                # Post-attention norm + MLP
                # Naming: post_attention_layernorm (Qwen/Llama), ffn_norm (LFM2), ln_2 (GPT)
                post_norm = (
                    getattr(layer, "post_attention_layernorm", None)
                    or getattr(layer, "ffn_norm", None)
                    or getattr(layer, "ln_2", None)
                )
                h_mlp = post_norm(h) if post_norm is not None else h

                # MLP: mlp (Qwen/Llama), feed_forward (LFM2)
                mlp_fn = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                h = h + mlp_fn(h_mlp)
                self.mx.eval(h)
            else:
                # Non-attention or no hook: standard forward
                layer_mask = "causal" if is_attn else None
                result = layer(h, mask=layer_mask, cache=None)
                h = result[0] if isinstance(result, tuple) else result
                self.mx.eval(h)

            # Collect mean-pooled hidden state
            hidden_states[layer_idx] = self.mx.mean(h, axis=1).squeeze(0)
            self.mx.eval(hidden_states[layer_idx])

        return hidden_states

    def collect_logits_with_attention_hook(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        attention_hook: Any | None = None,
        token_ids: list[int] | None = None,
    ) -> tuple[Any, dict[int, Any]]:
        """Forward pass with attention hook, returning logits and attention weights.

        Same hook mechanism as collect_hidden_with_attention_hook, but instead
        of collecting mean-pooled hidden states, applies final norm + unembedding
        to produce logits. Also collects post-softmax attention weights per layer.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer.
            text: Input text.
            attention_hook: Optional callable(weights, layer_idx) -> weights.
                weights shape: [batch, num_heads, seq, seq].
                If None, runs a normal forward pass (baseline).
            token_ids: Optional pre-tokenized input.

        Returns:
            Tuple of (logits [batch, seq, vocab], attn_weights_per_layer).
            attn_weights_per_layer maps layer_idx -> weights [batch, heads, seq, seq]
            for attention layers only.
        """
        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        attn_weights_per_layer: dict[int, Any] = {}

        for layer_idx, layer in enumerate(base.layers):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            is_attn = attn is not None and hasattr(attn, "q_proj")

            if is_attn:
                # Manual attention decomposition (same as collect_hidden_with_attention_hook)
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                elif hasattr(layer, "operator_norm"):
                    h_norm = layer.operator_norm(h)
                else:
                    h_norm = h

                q = attn.q_proj(h_norm)
                k = attn.k_proj(h_norm)
                v = attn.v_proj(h_norm)

                num_heads = (
                    getattr(attn, "num_heads", None)
                    or getattr(attn, "num_attention_heads", None)
                    or getattr(attn, "n_heads", None)
                )
                num_kv_heads = (
                    getattr(attn, "num_key_value_heads", None)
                    or getattr(attn, "n_kv_heads", None)
                )
                if num_kv_heads is None:
                    num_kv_heads = num_heads

                head_dim = getattr(attn, "head_dim", None)
                if head_dim is None:
                    head_dim = k.shape[-1] // num_kv_heads

                batch = q.shape[0]
                expected_q_dim = num_heads * head_dim

                gate = None
                if q.shape[-1] == expected_q_dim * 2:
                    q = q.reshape(batch, seq_len, num_heads, head_dim * 2)
                    q, gate = self.mx.split(q, 2, axis=-1)
                    gate = gate.reshape(batch, seq_len, -1)

                if q.ndim == 3:
                    q = q.reshape(batch, seq_len, num_heads, head_dim)
                k = k.reshape(batch, seq_len, num_kv_heads, head_dim)
                v = v.reshape(batch, seq_len, num_kv_heads, head_dim)

                q_ln = getattr(attn, "q_layernorm", None) or getattr(attn, "q_norm", None)
                k_ln = getattr(attn, "k_layernorm", None) or getattr(attn, "k_norm", None)
                if q_ln is not None:
                    q = q_ln(q)
                if k_ln is not None:
                    k = k_ln(k)

                q = q.transpose(0, 2, 1, 3)
                k = k.transpose(0, 2, 1, 3)
                v = v.transpose(0, 2, 1, 3)

                rope_fn = getattr(attn, "rope", None)
                if rope_fn is not None:
                    q = rope_fn(q)
                    k = rope_fn(k)

                if num_kv_heads < num_heads:
                    repeats = num_heads // num_kv_heads
                    k = self.mx.repeat(k, repeats, axis=1)
                    v = self.mx.repeat(v, repeats, axis=1)

                scale = head_dim ** -0.5
                scores = (q @ k.transpose(0, 1, 3, 2)) * scale

                causal_mask = self.mx.tril(self.mx.ones((seq_len, seq_len)))
                neg_inf = self.mx.array(float("-inf"))
                scores = self.mx.where(causal_mask[None, None, :, :], scores, neg_inf)

                weights = self.mx.softmax(scores, axis=-1)

                # Collect pre-hook weights for baseline measurement
                attn_weights_per_layer[layer_idx] = weights

                if attention_hook is not None:
                    weights = attention_hook(weights, layer_idx)

                attn_output = weights @ v
                attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
                    batch, seq_len, -1
                )

                if gate is not None:
                    attn_output = attn_output * self.mx.sigmoid(gate)

                o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
                attn_output = o_proj(attn_output)

                h = h + attn_output

                post_norm = (
                    getattr(layer, "post_attention_layernorm", None)
                    or getattr(layer, "ffn_norm", None)
                    or getattr(layer, "ln_2", None)
                )
                h_mlp = post_norm(h) if post_norm is not None else h

                mlp_fn = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                h = h + mlp_fn(h_mlp)
                self.mx.eval(h)
            else:
                layer_mask = "causal" if is_attn else None
                result = layer(h, mask=layer_mask, cache=None)
                h = result[0] if isinstance(result, tuple) else result
                self.mx.eval(h)

        # Final norm + unembedding → logits
        # Order matches mlx_training_adapter_core.py (lines 930-938):
        # norm (Qwen/Llama) takes precedence, then embedding_norm (LFM2).
        # In practice these are mutually exclusive, but the guard prevents
        # double-normalization if a model ever exposes both.
        if hasattr(base, "norm"):
            h = base.norm(h)
        elif hasattr(base, "embedding_norm"):
            h = base.embedding_norm(h)

        if hasattr(model, "lm_head"):
            logits = model.lm_head(h)
        else:
            logits = base.embed_tokens.as_linear(h)

        self.mx.eval(logits)
        return logits, attn_weights_per_layer

    def collect_logits(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> Any:
        """Collect logits for the last token position."""
        if token_ids is None:
            token_ids = tokenizer.encode(text)
        input_ids = self.mx.array([token_ids])

        logits = model(input_ids)
        self.mx.eval(logits)

        if logits.ndim == 3:
            last_logits = logits[0, -1, :]
        elif logits.ndim == 2:
            last_logits = logits[0, :]
        else:
            last_logits = logits

        self.mx.eval(last_logits)
        return last_logits

    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> Any:
        """Collect hidden + intermediate + gate + embedding activations in batch."""
        from mlx import nn

        from modelcypher.ports.activation_provider import ProbeActivationBatch

        if not texts:
            return ProbeActivationBatch(hidden=[], intermediate=[], gate=[], embedding=[])

        # Tokenize all texts
        all_token_ids = [tokenizer.encode(text) for text in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)
        batch_size = len(texts)

        hidden_results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        intermediate_results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        gate_results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        embedding_results: list[Any] = []
        all_tensors: list[Any] = []

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)

        # Compute mask for proper mean pooling
        seq_lengths = [len(ids) for ids in all_token_ids]
        lengths = self.mx.array(seq_lengths, dtype=h.dtype)
        pos = self.mx.arange(max_len)
        pad_mask = pos[None, :] < lengths[:, None]
        mask = pad_mask.astype(h.dtype)
        denom = lengths[:, None]

        # Embedding activations
        pooled_embeddings = self.mx.sum(h * mask[:, :, None], axis=1) / denom
        for i in range(batch_size):
            embedding_results.append(pooled_embeddings[i])
        all_tensors.append(pooled_embeddings)

        for layer_idx, layer in enumerate(base.layers):
            # Detect layer type
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
                    attn_out = self.mx.zeros_like(h)
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

                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn is not None:
                    attn_out = attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = self.mx.zeros_like(h)
                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                else:
                    h_post = h

                ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

            # MLP computation
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
                pooled_intermediate = self.mx.sum(intermediate * mask[:, :, None], axis=1) / denom
                for i in range(batch_size):
                    intermediate_results[i][layer_idx] = pooled_intermediate[i]
                all_tensors.append(pooled_intermediate)

            if gate is not None:
                pooled_gate = self.mx.sum(gate * mask[:, :, None], axis=1) / denom
                for i in range(batch_size):
                    gate_results[i][layer_idx] = pooled_gate[i]
                all_tensors.append(pooled_gate)

            if mlp_out is None:
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                elif hasattr(layer, "feed_forward"):
                    mlp_out = layer.feed_forward(h_post)
                else:
                    mlp_out = self.mx.zeros_like(h)
            h = h + mlp_out

            # Hidden state
            pooled_hidden = self.mx.sum(h * mask[:, :, None], axis=1) / denom
            for i in range(batch_size):
                hidden_results[i][layer_idx] = pooled_hidden[i]
            all_tensors.append(pooled_hidden)

        if all_tensors:
            self.mx.eval(*all_tensors)

        return ProbeActivationBatch(
            hidden=hidden_results,
            intermediate=intermediate_results,
            gate=gate_results,
            embedding=embedding_results,
        )

    def collect_routing_decisions(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> dict[int, Any]:
        """Collect selected expert IDs per token for MoE router layers.

        The collector computes top-k from gate logits before the layer's MLP
        call. The MLP then runs normally, which may recompute router logits
        internally. For deterministic routers this is identical; for noisy
        load-balancing routers the captured IDs can differ slightly.
        """
        if not texts:
            return {}

        all_token_ids = [tokenizer.encode(t) for t in texts]
        if not all_token_ids:
            return {}

        max_len = max(len(ids) for ids in all_token_ids)
        if max_len <= 0:
            return {}

        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)
        text_lengths = [len(ids) for ids in all_token_ids]

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)

        lengths = self.mx.array(text_lengths, dtype=h.dtype)
        pos = self.mx.arange(max_len)
        pad_mask = pos[None, :] < lengths[:, None]

        # LFM2-style attention mask when needed.
        causal = pos[:, None] >= pos[None, :]
        attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
        attn_mask = attn_mask[:, None, :, :]

        routing: dict[int, Any] = {}
        eval_tensors: list[Any] = []

        for layer_idx, layer in enumerate(base.layers):
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
                    attn_out = self.mx.zeros_like(h)
                h = h + attn_out
                h_post = layer.ffn_norm(h)
                ff_module = layer.feed_forward
            else:
                if hasattr(layer, "input_layernorm"):
                    h_norm = layer.input_layernorm(h)
                elif hasattr(layer, "ln_1"):
                    h_norm = layer.ln_1(h)
                elif hasattr(layer, "operator_norm"):
                    h_norm = layer.operator_norm(h)
                else:
                    h_norm = h

                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn is not None:
                    attn_out = attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = self.mx.zeros_like(h)
                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    h_post = layer.post_attention_layernorm(h)
                elif hasattr(layer, "ln_2"):
                    h_post = layer.ln_2(h)
                elif hasattr(layer, "ffn_norm"):
                    h_post = layer.ffn_norm(h)
                else:
                    h_post = h

                ff_module = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

            is_moe_block = (
                ff_module is not None
                and hasattr(ff_module, "gate")
                and (hasattr(ff_module, "experts") or hasattr(ff_module, "switch_mlp"))
            )
            if is_moe_block:
                gate_logits = ff_module.gate(h_post)
                num_experts = int(gate_logits.shape[-1])
                top_k_raw = (
                    getattr(ff_module, "num_experts_per_tok", None)
                    or getattr(ff_module, "top_k", None)
                    or 1
                )
                top_k = max(1, min(int(top_k_raw), num_experts))

                sorted_idx = self.mx.argsort(gate_logits, axis=-1)
                top_idx = sorted_idx[:, :, -top_k:]
                per_text = [
                    top_idx[i, : text_lengths[i], :]
                    for i in range(len(text_lengths))
                ]
                if per_text:
                    layer_selected = self.mx.concatenate(per_text, axis=0)
                else:
                    layer_selected = self.mx.zeros((0, top_k), dtype=self.mx.int32)
                routing[layer_idx] = layer_selected
                eval_tensors.append(layer_selected)

            if ff_module is not None:
                mlp_out = ff_module(h_post)
                h = h + mlp_out

        if eval_tensors:
            self.mx.eval(*eval_tensors)

        return routing

    def collect_hidden_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect per-layer hidden activations for multiple texts."""
        if not texts:
            return []

        all_token_ids = [tokenizer.encode(t) for t in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        all_tensors = []

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(base.layers):
            result = layer(h)
            h = result[0] if isinstance(result, tuple) else result

            for i in range(batch_size):
                seq_len = len(all_token_ids[i])
                pooled = self.mx.mean(h[i, :seq_len, :], axis=0)
                results[i][layer_idx] = pooled
                all_tensors.append(pooled)

        if all_tensors:
            self.mx.eval(*all_tensors)

        return results

    def collect_intermediate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect per-layer intermediate activations for multiple texts."""
        from mlx import nn

        if not texts:
            return []

        all_token_ids = [tokenizer.encode(t) for t in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        all_tensors = []

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(base.layers):
            if hasattr(layer, "input_layernorm"):
                h_norm = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_norm = layer.ln_1(h)
            elif hasattr(layer, "operator_norm"):
                h_norm = layer.operator_norm(h)
            else:
                h_norm = h

            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is not None:
                attn_out = attn(h_norm)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
            else:
                attn_out = self.mx.zeros_like(h)
            h = h + attn_out

            if hasattr(layer, "post_attention_layernorm"):
                h_post = layer.post_attention_layernorm(h)
            elif hasattr(layer, "ln_2"):
                h_post = layer.ln_2(h)
            elif hasattr(layer, "ffn_norm"):
                h_post = layer.ffn_norm(h)
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
                        pooled = self.mx.mean(intermediate[i, :seq_len, :], axis=0)
                        results[i][layer_idx] = pooled
                        all_tensors.append(pooled)

                mlp_out = ff_module(h_post)
                h = h + mlp_out
            else:
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result

        if all_tensors:
            self.mx.eval(*all_tensors)

        return results

    def collect_gate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect per-layer gate (pre-SiLU) activations for multiple texts."""
        if not texts:
            return []

        all_token_ids = [tokenizer.encode(t) for t in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)
        batch_size = len(texts)

        results: list[dict[int, Any]] = [{} for _ in range(batch_size)]
        all_tensors: list[Any] = []

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)

        seq_lengths = [len(ids) for ids in all_token_ids]
        lengths = self.mx.array(seq_lengths, dtype=h.dtype)
        pos = self.mx.arange(max_len)
        pad_mask = pos[None, :] < lengths[:, None]
        mask = pad_mask.astype(h.dtype)
        denom = lengths[:, None]

        for layer_idx, layer in enumerate(base.layers):
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
                    attn_out = self.mx.zeros_like(h)
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

                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn is not None:
                    attn_out = attn(h_norm)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                else:
                    attn_out = self.mx.zeros_like(h)
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
                    pooled_gate = self.mx.sum(gate * mask[:, :, None], axis=1) / denom
                    for i in range(batch_size):
                        results[i][layer_idx] = pooled_gate[i]
                    all_tensors.append(pooled_gate)
                elif hasattr(ff_module, "w1"):
                    gate = ff_module.w1(h_post)
                    pooled_gate = self.mx.sum(gate * mask[:, :, None], axis=1) / denom
                    for i in range(batch_size):
                        results[i][layer_idx] = pooled_gate[i]
                    all_tensors.append(pooled_gate)

                mlp_out = ff_module(h_post)
                h = h + mlp_out
            else:
                result = layer(h)
                h = h + (result[0] if isinstance(result, tuple) else result)

        if all_tensors:
            self.mx.eval(*all_tensors)

        return results

    def collect_trajectory_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> Any:
        """Collect full trajectory activations for manifold mapping."""
        from mlx import nn

        from modelcypher.ports.activation_provider import TrajectoryActivations

        if not texts:
            return TrajectoryActivations(
                positions={},
                velocities={},
                intermediate_positions={},
                embedding_positions=self.mx.zeros((0, 1)),
                q_positions={},
                k_positions={},
                v_positions={},
                gate_positions={},
                text_lengths=[],
                total_tokens=0,
                n_texts=0,
            )

        all_token_ids = [tokenizer.encode(t) for t in texts]
        text_lengths = [len(ids) for ids in all_token_ids]
        total_tokens = sum(text_lengths)
        n_texts = len(texts)

        max_len = max(text_lengths)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids]
        input_ids = self.mx.array(padded)

        base = self._resolve_model_base(model)
        h = base.embed_tokens(input_ids)

        # Embedding positions
        embedding_positions_list = []
        for i in range(n_texts):
            seq_len = text_lengths[i]
            embedding_positions_list.append(h[i, :seq_len, :])
        embedding_positions = self.mx.concatenate(embedding_positions_list, axis=0)

        positions: dict[int, Any] = {}
        velocities: dict[int, Any] = {}
        intermediate_positions: dict[int, Any] = {}
        q_positions: dict[int, Any] = {}
        k_positions: dict[int, Any] = {}
        v_positions: dict[int, Any] = {}
        gate_positions: dict[int, Any] = {}
        all_tensors: list[Any] = [embedding_positions]

        seq_lengths_arr = self.mx.array(text_lengths, dtype=h.dtype)
        pos = self.mx.arange(max_len)
        pad_mask = pos[None, :] < seq_lengths_arr[:, None]
        causal = pos[:, None] >= pos[None, :]
        attn_mask = pad_mask[:, :, None] & pad_mask[:, None, :] & causal[None, :, :]
        attn_mask = attn_mask[:, None, :, :]

        for layer_idx, layer in enumerate(base.layers):
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
                    attn_out = self.mx.zeros_like(h)
                    attn_module = None
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
                    attn_out = self.mx.zeros_like(h)
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

                layer_q_pos = self.mx.concatenate(q_pos_list, axis=0)
                layer_k_pos = self.mx.concatenate(k_pos_list, axis=0)
                layer_v_pos = self.mx.concatenate(v_pos_list, axis=0)

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
                layer_int_pos = self.mx.concatenate(int_pos_list, axis=0)
                intermediate_positions[layer_idx] = layer_int_pos
                all_tensors.append(layer_int_pos)

            if gate is not None:
                gate_pos_list = []
                for i in range(n_texts):
                    seq_len = text_lengths[i]
                    gate_pos_list.append(gate[i, :seq_len, :])
                layer_gate_pos = self.mx.concatenate(gate_pos_list, axis=0)
                gate_positions[layer_idx] = layer_gate_pos
                all_tensors.append(layer_gate_pos)

            if mlp_out is None:
                if hasattr(layer, "mlp"):
                    mlp_out = layer.mlp(h_post)
                elif hasattr(layer, "feed_forward"):
                    mlp_out = layer.feed_forward(h_post)
                else:
                    mlp_out = self.mx.zeros_like(h)
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
                layer_positions = self.mx.concatenate(layer_positions_list, axis=0)
                positions[layer_idx] = layer_positions
                all_tensors.append(layer_positions)

            if layer_velocities_list:
                layer_velocities = self.mx.concatenate(layer_velocities_list, axis=0)
                velocities[layer_idx] = layer_velocities
                all_tensors.append(layer_velocities)

        if all_tensors:
            self.mx.eval(*all_tensors)

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

    def compute_per_probe_gradients(
        self,
        model: Any,
        tokenizer: Any,
        probe_texts: list[str],
        weight_name: str,
    ) -> Any:
        """Compute per-probe CE gradients for a specific weight matrix.

        For each preservation probe, computes ∂CE(probe_i)/∂W where W is
        the weight identified by weight_name. Returns the gradient matrix
        G ∈ ℝ^{N × D} where D = product of weight dimensions.

        This is the behavior Jacobian: rows of G span the directions in
        weight space that affect model output on preservation probes. The
        null space of G contains directions where weight perturbation
        produces zero output change on the preservation set.

        Args:
            model: Loaded model (nn.Module).
            tokenizer: Tokenizer for encoding probe texts.
            probe_texts: Preservation probe texts.
            weight_name: Dot-separated path to weight parameter,
                e.g. "model.layers.5.mlp.down_proj.weight".

        Returns:
            mx.array of shape [n_probes, D] where D is the flattened
            weight dimension.
        """
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        gradients: list[Any] = []

        for text in probe_texts:
            tokens = mx.array([tokenizer.encode(text)])

            if tokens.shape[1] < 2:
                continue

            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            def _probe_loss(mdl):
                logits = mdl(inputs)
                loss = nn.losses.cross_entropy(
                    logits, targets, reduction="mean",
                )
                return loss

            loss_and_grad = nn.value_and_grad(model, _probe_loss)
            _loss_val, grads = loss_and_grad(model)

            # Extract gradient for the target weight from grad pytree
            layer_grad = None
            for name, tensor in tree_flatten(grads):
                if name == weight_name:
                    layer_grad = tensor
                    break

            if layer_grad is None:
                raise ValueError(
                    f"Weight '{weight_name}' not found in gradient pytree. "
                    "Check that the name matches a model parameter.",
                )

            gradients.append(layer_grad.reshape(-1))
            mx.eval(gradients[-1])

        if not gradients:
            raise ValueError(
                "No valid probes produced gradients "
                f"(received {len(probe_texts)} probe texts).",
            )

        G = mx.stack(gradients)  # [N, D]
        mx.eval(G)
        return G
