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
CUDA/PyTorch Activation Provider - Collects activations from PyTorch models.

This is an ADAPTER in hexagonal architecture. It implements the ActivationProvider
protocol for the CUDA backend (NVIDIA GPU with PyTorch).

Usage:
    from modelcypher.adapters.cuda_activation_provider import CUDAActivationProvider

    provider = CUDAActivationProvider()
    hidden_acts = provider.collect_hidden_activations(model, tokenizer, text)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

logger = logging.getLogger(__name__)


class CUDAActivationProvider:
    """
    PyTorch/CUDA implementation of ActivationProvider protocol.

    Collects activations from PyTorch models, keeping all tensors on CUDA GPU.

    Supports:
    - HuggingFace Transformers with PreTrainedModel
    - Custom PyTorch models with register_forward_hook
    - Llama.cpp/ExLlamaV2 via hooks
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize CUDA activation provider.

        Args:
            device: PyTorch device ("cuda", "cuda:0", "cuda:1", etc.)
        """
        self.device = device
        try:
            import torch

            self.torch = torch
            self._available = torch.cuda.is_available()
            if not self._available:
                logger.warning("CUDA not available. PyTorch will fall back to CPU.")
        except ImportError:
            self._available = False
            self.torch = None
            logger.warning("PyTorch not available. Install with: pip install torch")

    @property
    def available(self) -> bool:
        """Check if CUDA backend is available."""
        return self._available

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

        Supports:
        - HuggingFace PreTrainedModel with output_hidden_states=True
        - Custom PyTorch models with forward_with_hidden_states method
        - Any model with layer hooks

        Returns PyTorch tensors on CUDA (stays on GPU).
        """
        if self.torch is None:
            raise RuntimeError("PyTorch not available. Install: pip install torch")

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            elif hasattr(tokens, "ids"):
                token_ids = list(tokens.ids)
            else:
                token_ids = tokens.tolist()
        input_ids = self.torch.tensor([token_ids], device=self.device)

        activations: dict[int, "Array"] = {}

        try:
            # Try HuggingFace PreTrainedModel pattern
            if hasattr(model, "forward") and hasattr(model, "config"):
                with self.torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                        for layer_idx, hidden in enumerate(outputs.hidden_states):
                            pooled = hidden.mean(dim=(0, 1))
                            activations[layer_idx] = pooled
                        return activations

            # Try custom forward_with_hidden_states pattern
            if hasattr(model, "forward_with_hidden_states"):
                with self.torch.no_grad():
                    _, hidden_states = model.forward_with_hidden_states(input_ids)
                    for layer_idx, hidden in enumerate(hidden_states):
                        pooled = hidden.mean(dim=(0, 1))
                        activations[layer_idx] = pooled
                    return activations

            # Try layer iteration with hooks
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
                    handle = layer.register_forward_hook(make_hook(layer_idx))
                    handles.append(handle)

                try:
                    with self.torch.no_grad():
                        _ = model(input_ids)
                    activations = hook_outputs
                finally:
                    for handle in handles:
                        handle.remove()

                return activations

            # Fallback: run model and capture single output
            if not activations:
                with self.torch.no_grad():
                    output = model(input_ids)
                    if isinstance(output, tuple):
                        output = output[0]
                    pooled = output.mean(dim=(0, 1))
                    activations[0] = pooled

        except Exception as e:
            logger.warning("Activation collection failed for text '%s...': %s", text[:30], e)

        if not activations:
            logger.debug("No activations collected for text: %s", text[:50])

        return activations

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, "Array"]:
        """
        Collect per-layer MLP intermediate activations for a text input.

        Uses forward hooks to capture the MLP intermediate activations
        (after gating, before output projection).

        Returns PyTorch tensors on CUDA (stays on GPU).
        """
        if self.torch is None:
            raise RuntimeError("PyTorch not available. Install: pip install torch")

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            elif hasattr(tokens, "ids"):
                token_ids = list(tokens.ids)
            else:
                token_ids = tokens.tolist()
        input_ids = self.torch.tensor([token_ids], device=self.device)

        activations: dict[int, "Array"] = {}

        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                hook_outputs: dict[int, Any] = {}

                def make_gate_hook(layer_idx: int):
                    def hook(module, input, output):
                        # This captures gate_proj output (or fc1 for GPT-style)
                        hook_outputs[layer_idx] = output.mean(dim=(0, 1)).detach()

                    return hook

                handles = []
                for layer_idx, layer in enumerate(model.model.layers):
                    mlp = getattr(layer, "mlp", None)
                    if mlp is not None:
                        if hasattr(mlp, "gate_proj"):
                            handle = mlp.gate_proj.register_forward_hook(make_gate_hook(layer_idx))
                            handles.append(handle)
                        elif hasattr(mlp, "fc1"):
                            handle = mlp.fc1.register_forward_hook(make_gate_hook(layer_idx))
                            handles.append(handle)

                try:
                    with self.torch.no_grad():
                        _ = model(input_ids)
                    activations = hook_outputs
                finally:
                    for handle in handles:
                        handle.remove()

        except Exception as e:
            logger.warning("Intermediate activation collection failed: %s", e)

        return activations

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"]]:
        """
        Collect per-layer attention Q and KV activations for a text input.

        Uses forward hooks on q_proj and k_proj to capture:
        1. Q activations: [num_heads * head_dim]
        2. KV activations: [num_kv_heads * head_dim] (for GQA models)

        Returns PyTorch tensors on CUDA (stays on GPU).
        """
        if self.torch is None:
            raise RuntimeError("PyTorch not available. Install: pip install torch")

        if token_ids is None:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            elif hasattr(tokens, "ids"):
                token_ids = list(tokens.ids)
            else:
                token_ids = tokens.tolist()
        input_ids = self.torch.tensor([token_ids], device=self.device)

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
                    with self.torch.no_grad():
                        _ = model(input_ids)
                    q_activations = q_outputs
                    kv_activations = k_outputs
                finally:
                    for handle in handles:
                        handle.remove()

        except Exception as e:
            logger.warning("Attention activation collection failed: %s", e)

        return q_activations, kv_activations


def get_activation_provider(device: str = "cuda") -> CUDAActivationProvider:
    """Get the CUDA activation provider instance."""
    return CUDAActivationProvider(device=device)


__all__ = ["CUDAActivationProvider", "get_activation_provider"]
