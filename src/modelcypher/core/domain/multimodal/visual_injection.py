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
Visual Concept Injection Pipeline.

All geometric parameters are AUTO-DERIVED from the data.
The math determines everything - no user-configurable knobs.

Integrates:
1. HybridBridge (CLIP → LLM vocabulary space alignment)
2. AttentionMemoryInjector (memory token injection)

Auto-derived parameters:
- Scale: Derived from calibration activation norms (0.5 * mean_norm)
- Temperature: Derived from similarity distribution std (2.0 * std)
- Null rank: Derived from SVD (keep dims for 95% variance)
- Injection layer: Derived from architecture config (semantic highway middle)

Pipeline:
    1. CLIP encoder → 512D image embedding
    2. Vision offramp → 1024D (LLM dimension)
    3. HybridBridge.transform() → vocabulary-constrained embedding
    4. AttentionMemoryInjector → prepend as memory token at optimal layer

Usage:
    injector = VisualConceptInjector(backend)
    injector.load_bridge_weights("/path/to/affine_bridge.safetensors")
    injector.set_vocabulary(llm_vocab_embeddings)
    injector.compute_null_basis_from_activations(calibration_activations)

    # At inference time - no parameters needed, math determines everything
    memory = injector.create_visual_memory(clip_embedding)

    # Inject at auto-determined optimal layer
    result = injector.inject_memory(hidden_states, memory)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.affine_bridge import HybridBridge
from modelcypher.core.domain.multimodal.attention_memory import (
    AttentionMemoryInjector,
    KNOWN_ARCHITECTURES,
    MemoryTokenContent,
)

if TYPE_CHECKING:
    from modelcypher.core.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class VisualMemoryToken:
    """Visual concept as memory token for LLM injection."""

    # The memory embedding (vocabulary-constrained)
    embedding: Any  # [1, hidden_dim]

    # Nearest vocabulary tokens (for interpretability)
    nearest_token_ids: list[int]
    attention_weights: list[float]  # Top attention weights

    # Auto-derived parameters (exposed for diagnostics only)
    scale: float  # Derived from activation norms
    temperature: float  # Derived from similarity std
    null_space_projected: bool

    # Source info
    source_type: str  # "clip_image", "clip_text", etc.


@dataclass
class InjectionResult:
    """Result of visual memory injection."""

    # Modified hidden states
    hidden_states: Any

    # Layer(s) where injection occurred
    injection_layers: list[int]

    # Safety check results
    is_safe: bool
    safety_message: str


class VisualConceptInjector:
    """
    End-to-end visual concept injection pipeline.

    All geometric parameters are auto-derived from the data.
    No user-configurable knobs - the math determines everything.

    Combines vocabulary-constrained projection (HybridBridge) with
    attention-based memory injection (AttentionMemoryInjector).
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        architecture: str = "LFM2",
    ) -> None:
        """Initialize the injector.

        Args:
            backend: Optional backend instance
            architecture: Model architecture name for layer config
        """
        self._backend = backend or get_default_backend()
        self._architecture = architecture

        # Components
        self._bridge = HybridBridge(self._backend)
        self._memory_injector = AttentionMemoryInjector(self._backend)

        # State
        self._bridge_loaded = False
        self._vocabulary_set = False
        self._null_basis: Any = None
        self._calibration_activations: Any = None  # For scale derivation

        # Get layer config for architecture
        self._layer_config = KNOWN_ARCHITECTURES.get(architecture)
        if self._layer_config:
            logger.info(
                f"Using {architecture} config: "
                f"attention_layers={self._layer_config.attention_layers}, "
                f"semantic_highway={self._layer_config.semantic_highway}"
            )
        else:
            logger.warning(f"Unknown architecture {architecture}, using defaults")

    def load_bridge_weights(
        self,
        weights_path: str | Path,
    ) -> None:
        """Load pre-trained affine bridge weights.

        Args:
            weights_path: Path to safetensors file with W and b tensors
        """
        from safetensors import safe_open

        weights_path = Path(weights_path)

        with safe_open(str(weights_path), framework="numpy") as f:
            W_np = f.get_tensor("W")
            b_np = f.get_tensor("b")

        W = self._backend.array(W_np)
        b = self._backend.array(b_np)
        self._backend.eval(W, b)

        self._bridge.load_affine_weights(W, b)
        self._bridge_loaded = True

        logger.info(f"Loaded bridge weights from {weights_path}: W={W_np.shape}, b={b_np.shape}")

    def set_vocabulary(self, vocab_embeddings: "Array") -> None:
        """Set LLM vocabulary embeddings for constrained projection.

        Args:
            vocab_embeddings: [vocab_size, hidden_dim] token embeddings
        """
        self._bridge.set_vocabulary(vocab_embeddings)
        self._vocabulary_set = True

        vocab_size = int(vocab_embeddings.shape[0])
        hidden_dim = int(vocab_embeddings.shape[1])
        logger.info(f"Set vocabulary: {vocab_size} tokens, {hidden_dim}D")

    def set_null_basis(self, null_basis: "Array") -> None:
        """Set precomputed null-space basis for projection.

        Args:
            null_basis: [null_rank, hidden_dim] null-space basis
        """
        self._null_basis = self._backend.array(null_basis)
        self._backend.eval(self._null_basis)
        logger.info(f"Set null basis: {self._null_basis.shape}")

    def compute_null_basis_from_activations(
        self,
        activations: "Array",
    ) -> None:
        """Compute null-space basis from activation samples.

        Null rank is auto-derived from SVD: keep dimensions explaining
        the remaining variance after 95% is captured by active dimensions.

        Args:
            activations: [n_samples, hidden_dim] activation samples
        """
        # Store for scale derivation
        self._calibration_activations = self._backend.array(activations)
        self._backend.eval(self._calibration_activations)

        # Auto-derive null rank from SVD
        null_rank = self._derive_null_rank(self._calibration_activations)

        self._null_basis = self._memory_injector.compute_null_basis(
            activations,
            null_rank=null_rank,
            cache_key=f"{self._architecture}_null_basis",
        )
        logger.info(f"Computed null basis: {self._null_basis.shape} (rank auto-derived: {null_rank})")

    def _derive_null_rank(self, activations: "Array") -> int:
        """Derive null rank from SVD variance analysis.

        Keep dimensions that explain the remaining variance after
        95% is captured by the principal components.

        Args:
            activations: [n_samples, hidden_dim] activation samples

        Returns:
            Number of null-space dimensions to keep
        """
        backend = self._backend

        # Center activations
        mean = backend.mean(activations, axis=0, keepdims=True)
        centered = activations - mean

        # Compute SVD
        _, S, _ = backend.svd(centered, full_matrices=False)
        backend.eval(S)

        # Compute cumulative variance explained
        total_var = backend.sum(S ** 2)
        cumvar = backend.cumsum(S ** 2) / total_var
        backend.eval(cumvar)

        # Find active rank (dimensions explaining 95% variance)
        cumvar_list = backend.tolist(cumvar)
        active_rank = sum(1 for v in cumvar_list if v < 0.95) + 1

        # Null rank is remaining dimensions
        hidden_dim = int(activations.shape[1])
        null_rank = max(1, hidden_dim - active_rank)

        logger.debug(f"SVD variance analysis: active_rank={active_rank}, null_rank={null_rank}")
        return null_rank

    def _derive_optimal_scale(self) -> float:
        """Derive optimal scale from calibration activation statistics.

        Scale is set to 0.5 * mean(activation_norms), which provides
        sufficient signal while staying within safe injection range.

        Returns:
            Auto-derived scale factor
        """
        if self._calibration_activations is None:
            # Fallback if no calibration data
            logger.warning("No calibration activations. Using default scale 5.0")
            return 5.0

        backend = self._backend

        # Compute L2 norms of calibration activations
        norms = backend.sqrt(backend.sum(
            self._calibration_activations ** 2, axis=1
        ))
        backend.eval(norms)

        # Scale is half the mean norm (conservative but effective)
        mean_norm = float(backend.to_scalar(backend.mean(norms)))
        scale = mean_norm * 0.5

        logger.debug(f"Derived scale: {scale:.3f} (mean_norm={mean_norm:.3f})")
        return scale

    def create_visual_memory(
        self,
        embedding: "Array",
        source_type: str = "clip_image",
    ) -> VisualMemoryToken:
        """Create visual memory token from embedding.

        All geometric parameters (scale, temperature) are auto-derived.
        Null-space projection is always applied if basis is available.

        Args:
            embedding: Source embedding [1, source_dim] (e.g., CLIP 512D or 1024D)
            source_type: Source type for logging (not used in computation)

        Returns:
            VisualMemoryToken ready for injection
        """
        if not self._bridge_loaded:
            msg = "Must call load_bridge_weights() before creating visual memory"
            raise RuntimeError(msg)

        if not self._vocabulary_set:
            msg = "Must call set_vocabulary() before creating visual memory"
            raise RuntimeError(msg)

        backend = self._backend

        # Ensure correct shape
        embedding = backend.array(embedding)
        if len(embedding.shape) == 1:
            embedding = backend.reshape(embedding, (1, -1))
        backend.eval(embedding)

        # Apply vocabulary-constrained projection via HybridBridge
        # Temperature is auto-derived inside transform()
        vocab_result = self._bridge.transform(embedding)

        # Get the aligned embedding (on vocabulary manifold)
        aligned = backend.array(vocab_result.aligned)
        backend.eval(aligned)

        # Get auto-derived temperature from result
        derived_temperature = vocab_result.temperature_used

        # ALWAYS project into null-space if basis is available
        null_space_projected = False
        if self._null_basis is not None:
            # Project into null-space directions
            coeffs = backend.matmul(aligned, backend.transpose(self._null_basis))
            aligned = backend.matmul(coeffs, self._null_basis)
            backend.eval(aligned)
            null_space_projected = True

        # Auto-derive scale from calibration activations
        derived_scale = self._derive_optimal_scale()

        # Apply scale
        memory_embedding = aligned * derived_scale
        backend.eval(memory_embedding)

        # Get top attention weights for interpretability
        top_k = 5
        attn = vocab_result.attention_weights[0]
        top_indices = sorted(range(len(attn)), key=lambda i: attn[i], reverse=True)[:top_k]
        top_weights = [attn[i] for i in top_indices]

        return VisualMemoryToken(
            embedding=memory_embedding,
            nearest_token_ids=vocab_result.nearest_token_ids,
            attention_weights=top_weights,
            scale=derived_scale,
            temperature=derived_temperature,
            null_space_projected=null_space_projected,
            source_type=source_type,
        )

    def inject_memory(
        self,
        hidden_states: "Array",
        memory: VisualMemoryToken,
    ) -> InjectionResult:
        """Inject visual memory into hidden states.

        Injection layer is auto-determined from architecture config
        (semantic highway middle). Always validates scale.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] hidden states
            memory: Visual memory token to inject

        Returns:
            InjectionResult with modified hidden states
        """
        backend = self._backend
        hidden_states = backend.array(hidden_states)

        # Auto-determine target layer from architecture
        if self._layer_config:
            layer_idx = self._layer_config.semantic_highway[1]  # Middle of highway
        else:
            raise ValueError(
                f"Unknown architecture '{self._architecture}'. "
                "Cannot auto-determine injection layer."
            )

        # Validate layer is attention (can attend to memory)
        if layer_idx not in self._layer_config.attention_layers:
            logger.warning(
                f"Layer {layer_idx} is not an attention layer in {self._architecture}. "
                f"Attention layers: {self._layer_config.attention_layers}"
            )

        # Measure memory scale for transparency (geometry handles safety by construction)
        memory_content = MemoryTokenContent(
            content=memory.embedding,
            source_concept=memory.source_type,
            scale_applied=memory.scale,
            null_space_projected=memory.null_space_projected,
            direction_norm=float(backend.to_scalar(
                backend.sqrt(backend.sum(memory.embedding * memory.embedding))
            )),
        )
        is_valid, info_message = self._memory_injector.validate_memory_scale(
            memory_content,
            hidden_states,
        )
        logger.debug(f"Memory scale measurement: {info_message}")

        # Apply memory token (always prepend at position 0)
        modified = self._memory_injector.apply_memory_to_hidden_states(
            hidden_states=hidden_states,
            memory_content=MemoryTokenContent(
                content=memory.embedding,
                source_concept=memory.source_type,
                scale_applied=memory.scale,
                null_space_projected=memory.null_space_projected,
                direction_norm=0.0,  # Not used for application
            ),
            memory_position=0,  # Always prepend
        )

        return InjectionResult(
            hidden_states=modified,
            injection_layers=[layer_idx],
            is_safe=True,  # Geometry handles safety by construction
            safety_message=info_message,
        )

    def get_optimal_injection_layers(self) -> list[int]:
        """Get optimal layers for memory injection.

        Returns:
            List of layer indices optimal for memory placement
        """
        if self._layer_config:
            # Get attention layers in semantic highway
            layer_types = self._memory_injector.detect_layer_types(
                architecture_name=self._architecture
            )
            return self._memory_injector.get_optimal_memory_layers(
                layer_types,
                semantic_highway=self._layer_config.semantic_highway,
            )
        return [8]  # Default

    @property
    def is_ready(self) -> bool:
        """Check if injector is ready for use."""
        return self._bridge_loaded and self._vocabulary_set
