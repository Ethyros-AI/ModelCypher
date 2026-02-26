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

"""Sublayer activation collector for causal chain analysis.

Collects per-sublayer activations (h_in, h_post_attn, h_out) needed for
curvature decomposition into attention and MLP components.

For standard transformers (Qwen, Llama): full decomposition.
For LFM2 hybrid layers: total curvature only for non-attention layers.

Returns pure Python lists at the boundary (no numpy/mlx in return types)
to respect hexagonal import rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def collect_sublayer_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    num_layers: int,
    backend: Backend,
) -> list[dict]:
    """Collect per-sublayer activations for curvature decomposition.

    For each layer, collects:
        h_in:  input to the layer [N, d]
        h_post_attn: after attention + residual [N, d] (or None)
        h_out: after MLP + residual = layer output [N, d]

    Returns pure Python list[list[float]] arrays (not numpy) so the
    domain module can process them without importing numpy.

    Args:
        model: Loaded model (e.g., from ModelLoader).
        tokenizer: Tokenizer for the model.
        prompts: Probe texts to feed through the model.
        num_layers: Number of transformer layers to collect from.
        backend: Compute backend for mask creation.

    Returns:
        List of dicts per layer, each with:
            h_in: list[list[float]] [N_probes, d]
            h_out: list[list[float]] [N_probes, d]
            h_post_attn: list[list[float]] [N_probes, d] or None
            has_decomposition: bool
    """
    base = getattr(model, "model", model)
    embed = getattr(base, "embed_tokens", None)
    layers_list = getattr(base, "layers", None)
    if layers_list is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone layers")

    # Per-layer collectors (numpy for efficient stacking, converted at boundary)
    layer_h_in: list[list[np.ndarray]] = [[] for _ in range(num_layers)]
    layer_h_post_attn: list[list[np.ndarray | None]] = [[] for _ in range(num_layers)]
    layer_h_out: list[list[np.ndarray]] = [[] for _ in range(num_layers)]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)
        seq_len = input_ids.shape[1]

        try:
            numeric_mask = backend.create_causal_mask(seq_len, hidden.dtype)
        except Exception:
            numeric_mask = None

        for i, layer in enumerate(layers_list):
            if i >= num_layers:
                break

            h_in = hidden

            # Per-layer mask routing (LFM2 compatibility)
            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = numeric_mask

            # Try to decompose into attention + MLP sub-steps
            h_post_attn = None
            input_norm = getattr(layer, "input_layernorm", None)
            self_attn = getattr(layer, "self_attn", None)
            post_attn_norm = getattr(layer, "post_attention_layernorm", None)
            mlp = getattr(layer, "mlp", None)

            if input_norm is not None and self_attn is not None and mlp is not None:
                try:
                    normed = input_norm(h_in)
                    attn_out = self_attn(normed, mask=layer_mask)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_post_attn = h_in + attn_out

                    normed2 = (
                        post_attn_norm(h_post_attn)
                        if post_attn_norm is not None
                        else h_post_attn
                    )
                    mlp_out = mlp(normed2)
                    h_out = h_post_attn + mlp_out
                except Exception:
                    h_post_attn = None
                    h_out = _call_layer_fallback(layer, h_in, layer_mask)
            else:
                h_out = _call_layer_fallback(layer, h_in, layer_mask)

            # Last-token representation [1, d] → [d]
            h_in_last = h_in[:, -1, :].astype(mx.float32)
            h_out_last = h_out[:, -1, :].astype(mx.float32)
            mx.eval(h_in_last, h_out_last)

            layer_h_in[i].append(
                np.array(h_in_last[0].tolist(), dtype=np.float32)
            )
            layer_h_out[i].append(
                np.array(h_out_last[0].tolist(), dtype=np.float32)
            )

            if h_post_attn is not None:
                h_pa_last = h_post_attn[:, -1, :].astype(mx.float32)
                mx.eval(h_pa_last)
                layer_h_post_attn[i].append(
                    np.array(h_pa_last[0].tolist(), dtype=np.float32)
                )
            else:
                layer_h_post_attn[i].append(None)

            hidden = h_out

        mx.eval(hidden)

    # Build result — convert numpy arrays to pure Python lists at boundary
    result = []
    for i in range(num_layers):
        has_decomposition = all(x is not None for x in layer_h_post_attn[i])
        result.append(
            {
                "h_in": [arr.tolist() for arr in layer_h_in[i]],
                "h_out": [arr.tolist() for arr in layer_h_out[i]],
                "h_post_attn": (
                    [arr.tolist() for arr in layer_h_post_attn[i]]
                    if has_decomposition
                    else None
                ),
                "has_decomposition": has_decomposition,
            }
        )

    return result


def _call_layer_fallback(layer: Any, h_in: Any, mask: Any) -> Any:
    """Call a layer with fallback mask handling."""
    try:
        return layer(h_in, mask=mask)
    except (TypeError, ValueError):
        try:
            return layer(h_in, mask)
        except (TypeError, ValueError):
            return layer(h_in)
