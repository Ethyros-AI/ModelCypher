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

"""Layer Entropy Projector (Entropy-Lens Implementation).

Projects hidden states at each transformer layer through the unembedding matrix
to compute per-layer entropy profiles. This implements the Entropy-Lens approach
from Ali et al. (2025) for real layer-wise entropy measurement.

Algorithm (per layer L):
    1. Capture hidden state h_L during forward pass
    2. Project through unembedding matrix: logits_L = h_L @ W_unembed.T
    3. Apply softmax: p = softmax(logits_L)
    4. Compute Shannon entropy: H_L = -sum(p * log(p))

This gives one real entropy value per layer showing how uncertainty evolves
through the model - NOT fabricated data like the previous implementation.

Theoretical Foundation
----------------------
The projection through the unembedding matrix measures entropy in the *output
embedding space*. Per Bertolotti & Cazzola (2024), output embeddings encode
*contextual similarity* (words appearing in similar contexts → similar vectors),
while input embeddings encode *semantic similarity* (words with similar meanings).

When using tied embeddings (same matrix for input/output), this assumes the
*distributional hypothesis* holds: that contextual and semantic similarity align.
This assumption is generally valid for language models trained on natural text.

References
----------
Ali et al. (2025) "Entropy-Lens: The Information Signature of Transformer Computations"
    arXiv:2502.16570 - Key finding: entropy profiles reveal family-specific computation patterns

Bertolotti & Cazzola (2024) "By Tying Embeddings You Are Assuming the Distributional Hypothesis"
    ICML 2024, PMLR 235:3584-3610 - Theoretical foundation for tied embedding interpretation
    https://proceedings.mlr.press/v235/bertolotti24a.html
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.entropy.layer_projector")


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class LayerEntropyResult:
    """Entropy measurement for a single layer.

    Attributes
    ----------
    layer_index : int
        Zero-based layer index.
    layer_name : str
        Human-readable layer name (e.g., "layers.0").
    mean_entropy : float
        Mean entropy across all probe prompts for this layer.
    entropy_variance : float
        Variance of entropy across probe prompts.
    sample_count : int
        Number of prompts used for measurement.
    min_entropy : float
        Minimum entropy observed.
    max_entropy : float
        Maximum entropy observed.
    """

    layer_index: int
    layer_name: str
    mean_entropy: float
    entropy_variance: float
    sample_count: int
    min_entropy: float = 0.0
    max_entropy: float = 0.0


@dataclass
class ModelLayerEntropyProfile:
    """Complete per-layer entropy profile for a model.

    Attributes
    ----------
    model_name : str
        Name/path of the profiled model.
    layer_results : dict[int, LayerEntropyResult]
        Entropy results keyed by layer index.
    probe_prompts : list[str]
        Prompts used for profiling.
    unembedding_source : str
        Source of unembedding matrix ("lm_head" or "embed_tokens_transposed").
    vocab_size : int
        Vocabulary size (for max entropy calculation).
    max_possible_entropy : float
        Maximum possible entropy = ln(vocab_size).
    created_at : datetime
        Timestamp of profile creation.
    """

    model_name: str
    layer_results: dict[int, LayerEntropyResult]
    probe_prompts: list[str]
    unembedding_source: str
    vocab_size: int
    max_possible_entropy: float
    created_at: datetime = field(default_factory=datetime.now)

    def entropy_trajectory(self) -> list[float]:
        """Return entropy values in layer order."""
        return [
            self.layer_results[i].mean_entropy
            for i in sorted(self.layer_results.keys())
        ]

    def normalized_trajectory(self) -> list[float]:
        """Return entropy values normalized to [0, 1] by max possible entropy."""
        if self.max_possible_entropy <= 0:
            return [0.0] * len(self.layer_results)
        return [
            self.layer_results[i].mean_entropy / self.max_possible_entropy
            for i in sorted(self.layer_results.keys())
        ]


# =============================================================================
# Layer Entropy Projector
# =============================================================================


class LayerEntropyProjector:
    """Projects hidden states to logits and computes per-layer entropy.

    Implements the Entropy-Lens approach: for each layer, project hidden states
    through the unembedding matrix to vocabulary space, then compute Shannon entropy.

    Parameters
    ----------
    backend : Backend, optional
        Compute backend. Defaults to MLXBackend.
    epsilon : float, optional
        Numerical stability constant for log operations.

    Examples
    --------
    Basic usage:

        projector = LayerEntropyProjector()
        projector.set_unembedding_matrix(model)

        # Profile all layers
        profile = projector.profile_model(model, tokenizer, prompts)

        # Get entropy trajectory
        trajectory = profile.entropy_trajectory()
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        epsilon: float = 1e-10,
    ) -> None:
        self._backend = backend or get_default_backend()
        self._epsilon = epsilon
        self._unembedding_matrix: "Array | None" = None
        self._vocab_size: int = 0
        self._hidden_dim: int = 0
        self._unembedding_source: str = ""

    def set_unembedding_matrix(self, model: Any) -> str:
        """Extract unembedding matrix from model.

        Tries multiple strategies to locate the unembedding matrix:
        1. model.lm_head.weight (explicit output projection)
        2. model.model.lm_head.weight (nested model structure)
        3. model.model.embed_tokens.weight (tied weights, may need transpose)
        4. model.embed_tokens.weight (direct access)

        Parameters
        ----------
        model : Any
            Loaded transformer model.

        Returns
        -------
        str
            Source identifier ("lm_head" or "embed_tokens_transposed").

        Raises
        ------
        ValueError
            If unembedding matrix cannot be found.
        """
        b = self._backend

        # Strategy 1: Explicit lm_head
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            weight = model.lm_head.weight
            self._unembedding_matrix = b.astype(weight, "float32")
            self._vocab_size = weight.shape[0]
            self._hidden_dim = weight.shape[1]
            self._unembedding_source = "lm_head"
            logger.info(
                f"Using lm_head: vocab={self._vocab_size}, hidden={self._hidden_dim}"
            )
            return self._unembedding_source

        # Strategy 2: Nested model.model.lm_head
        if (
            hasattr(model, "model")
            and hasattr(model.model, "lm_head")
            and hasattr(model.model.lm_head, "weight")
        ):
            weight = model.model.lm_head.weight
            self._unembedding_matrix = b.astype(weight, "float32")
            self._vocab_size = weight.shape[0]
            self._hidden_dim = weight.shape[1]
            self._unembedding_source = "lm_head"
            logger.info(
                f"Using model.model.lm_head: vocab={self._vocab_size}, hidden={self._hidden_dim}"
            )
            return self._unembedding_source

        # Strategy 3: Tied weights via model.model.embed_tokens
        # NOTE: Using input embeddings as output embeddings assumes the distributional
        # hypothesis holds (Bertolotti & Cazzola, ICML 2024). This is valid for most
        # language models trained on natural text, where semantic similarity (encoded
        # in input embeddings) aligns with contextual similarity (output embeddings).
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed = model.model.embed_tokens
            if hasattr(embed, "weight"):
                weight = embed.weight
                # embed_tokens.weight is [vocab_size, hidden_dim]
                # unembedding is also [vocab_size, hidden_dim] for tied weights
                self._unembedding_matrix = b.astype(weight, "float32")
                self._vocab_size = weight.shape[0]
                self._hidden_dim = weight.shape[1]
                self._unembedding_source = "embed_tokens_transposed"
                logger.info(
                    f"Using embed_tokens (tied): vocab={self._vocab_size}, hidden={self._hidden_dim}"
                )
                logger.debug(
                    "Tied embeddings assume distributional hypothesis (Bertolotti & Cazzola 2024)"
                )
                return self._unembedding_source

        # Strategy 4: Direct embed_tokens (same distributional hypothesis caveat)
        if hasattr(model, "embed_tokens") and hasattr(model.embed_tokens, "weight"):
            weight = model.embed_tokens.weight
            self._unembedding_matrix = b.astype(weight, "float32")
            self._vocab_size = weight.shape[0]
            self._hidden_dim = weight.shape[1]
            self._unembedding_source = "embed_tokens_transposed"
            logger.info(
                f"Using embed_tokens (tied): vocab={self._vocab_size}, hidden={self._hidden_dim}"
            )
            logger.debug(
                "Tied embeddings assume distributional hypothesis (Bertolotti & Cazzola 2024)"
            )
            return self._unembedding_source

        raise ValueError(
            "Could not locate unembedding matrix in model. "
            "Expected lm_head.weight or embed_tokens.weight"
        )

    def compute_layer_entropy(
        self,
        hidden_state: "Array",
    ) -> tuple[float, float]:
        """Project hidden state to logits and compute Shannon entropy.

        Parameters
        ----------
        hidden_state : Array
            Hidden state from a transformer layer.
            Shape: [batch, seq, hidden], [seq, hidden], or [hidden]

        Returns
        -------
        tuple[float, float]
            (entropy, variance) where:
            - entropy: Shannon entropy over vocabulary
            - variance: Always 0.0 (placeholder for API compatibility)

        Notes
        -----
        Algorithm:
        1. Extract last token if sequence
        2. Project: logits = hidden @ W_unembed.T
        3. Softmax with numerical stability
        4. Entropy: H = -sum(p * log(p))
        """
        if self._unembedding_matrix is None:
            raise ValueError("Unembedding matrix not set. Call set_unembedding_matrix first.")

        b = self._backend

        # Flatten to 1D hidden vector (last token)
        h = self._flatten_to_hidden(hidden_state)
        h = b.astype(h, "float32")

        # Project to vocabulary: logits = h @ W.T
        # W is [vocab, hidden], so W.T is [hidden, vocab]
        # h @ W.T gives [vocab]
        logits = b.matmul(h, b.transpose(self._unembedding_matrix))

        # Numerically stable softmax
        max_val = b.max(logits, keepdims=True)
        shifted = logits - max_val
        exp_shifted = b.exp(shifted)
        sum_exp = b.sum(exp_shifted, keepdims=True)
        probs = exp_shifted / sum_exp

        # Shannon entropy: -sum(p * log(p))
        log_probs = b.log(probs + self._epsilon)
        entropy = -b.sum(probs * log_probs)

        # Evaluate and convert
        b.eval(entropy)
        entropy_val = float(b.to_numpy(entropy).item())

        return entropy_val, 0.0

    def profile_model(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        target_layers: set[int] | None = None,
    ) -> ModelLayerEntropyProfile:
        """Profile all layers with real entropy measurement.

        Runs forward passes with hidden state capture at each layer,
        projects to vocabulary space, and computes entropy.

        Parameters
        ----------
        model : Any
            Loaded transformer model.
        tokenizer : Any
            Tokenizer for the model.
        prompts : list[str]
            Probe prompts to use for measurement.
        target_layers : set[int], optional
            Specific layers to profile. If None, profiles all layers.

        Returns
        -------
        ModelLayerEntropyProfile
            Complete per-layer entropy profile.
        """
        from pathlib import Path

        b = self._backend

        # Ensure unembedding matrix is set
        if self._unembedding_matrix is None:
            self.set_unembedding_matrix(model)

        # Get model structure
        base_model = getattr(model, "model", model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Could not find model.layers or model.model.layers")

        num_layers = len(layers)
        if target_layers is None:
            target_layers = set(range(num_layers))

        # Model name from path or class
        model_name = getattr(model, "name", None) or model.__class__.__name__

        logger.info(
            f"Profiling {model_name}: {num_layers} layers, {len(prompts)} prompts"
        )

        # Collect entropy for each layer across all prompts
        layer_entropies: dict[int, list[float]] = {i: [] for i in target_layers}

        for prompt_idx, prompt in enumerate(prompts):
            # Tokenize
            tokens = tokenizer.encode(prompt)
            if isinstance(tokens, list):
                input_ids = b.array([tokens])
            else:
                input_ids = tokens
                if input_ids.ndim == 1:
                    input_ids = b.reshape(input_ids, (1, -1))

            # Capture hidden states at each target layer
            captured_states = self._capture_layer_states(
                base_model, layers, input_ids, target_layers
            )

            # Compute entropy for each captured state
            for layer_idx, hidden_state in captured_states.items():
                entropy, _ = self.compute_layer_entropy(hidden_state)
                layer_entropies[layer_idx].append(entropy)

            if (prompt_idx + 1) % 10 == 0:
                logger.debug(f"Processed {prompt_idx + 1}/{len(prompts)} prompts")

        # Compute statistics for each layer
        layer_results: dict[int, LayerEntropyResult] = {}
        for layer_idx, entropies in layer_entropies.items():
            if not entropies:
                continue

            mean_entropy = sum(entropies) / len(entropies)
            variance = (
                sum((e - mean_entropy) ** 2 for e in entropies) / len(entropies)
                if len(entropies) > 1
                else 0.0
            )

            layer_results[layer_idx] = LayerEntropyResult(
                layer_index=layer_idx,
                layer_name=f"layers.{layer_idx}",
                mean_entropy=mean_entropy,
                entropy_variance=variance,
                sample_count=len(entropies),
                min_entropy=min(entropies),
                max_entropy=max(entropies),
            )

        max_entropy = math.log(self._vocab_size) if self._vocab_size > 0 else 0.0

        return ModelLayerEntropyProfile(
            model_name=model_name,
            layer_results=layer_results,
            probe_prompts=prompts,
            unembedding_source=self._unembedding_source,
            vocab_size=self._vocab_size,
            max_possible_entropy=max_entropy,
        )

    def _flatten_to_hidden(self, hidden_state: "Array") -> "Array":
        """Extract 1D hidden vector from various tensor shapes.

        Handles:
        - [batch, seq, hidden] -> last token of batch 0
        - [seq, hidden] -> last token
        - [hidden] -> as-is
        """
        if hidden_state.ndim == 3:
            # [batch, seq, hidden] -> take batch 0, last token
            return hidden_state[0, -1, :]
        elif hidden_state.ndim == 2:
            # [seq, hidden] -> take last token
            return hidden_state[-1, :]
        else:
            return hidden_state

    def _capture_layer_states(
        self,
        base_model: Any,
        layers: Any,
        input_ids: "Array",
        target_layers: set[int],
    ) -> dict[int, "Array"]:
        """Capture hidden states at target layers during forward pass.

        Uses direct layer replacement for MLX compatibility (no hooks).

        Parameters
        ----------
        base_model : Any
            The base model (model.model typically).
        layers : Any
            The layers list from the model.
        input_ids : Array
            Input token IDs.
        target_layers : set[int]
            Layers to capture.

        Returns
        -------
        dict[int, Array]
            Captured hidden states keyed by layer index.
        """
        b = self._backend
        captured: dict[int, "Array"] = {}

        # Create wrapper layers that capture output
        class _CaptureWrapper:
            def __init__(wrapper_self, layer: Any, layer_idx: int) -> None:
                wrapper_self._layer = layer
                wrapper_self._layer_idx = layer_idx

            def __call__(wrapper_self, *args: Any, **kwargs: Any) -> Any:
                output = wrapper_self._layer(*args, **kwargs)
                # Capture the hidden state (first element if tuple)
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured[wrapper_self._layer_idx] = hidden
                return output

            def __getattr__(wrapper_self, name: str) -> Any:
                return getattr(wrapper_self._layer, name)

        # Store original layers
        original_layers = list(layers)

        try:
            # Replace target layers with wrappers
            for i in target_layers:
                if 0 <= i < len(layers):
                    layers[i] = _CaptureWrapper(original_layers[i], i)

            # Forward pass
            _ = base_model(input_ids)
            b.eval(_)

        finally:
            # Restore original layers
            for i, layer in enumerate(original_layers):
                layers[i] = layer

        return captured
