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

"""Training ports for framework-agnostic training operations.

Two protocols:
1. TrainingEngine - High-level job management (start, stop, status)
2. TrainingPort - Low-level framework operations (LoRA, gradients, optimizer)

Domain code uses the Backend protocol for numeric operations.
Adapters implement these ports for specific runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from modelcypher.core.domain.models import TrainingJob
    from modelcypher.core.domain.training import PreflightResult, TrainingSpec
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Data Classes for TrainingPort
# =============================================================================


@dataclass
class LoRALayerConfig:
    """Configuration for a single LoRA layer.

    All values are geometry-derived, not hyperparameters.
    """

    layer_key: str
    rank: int
    sigma_k: float  # Spectral scale bound
    in_features: int
    out_features: int


@dataclass
class ParameterInfo:
    """Information about a single parameter."""

    key: str
    shape: tuple[int, ...]
    dtype: str
    size: int


@dataclass
class GradientInfo:
    """Information about gradients from a training step."""

    param_key: str
    grad_norm: float
    param_norm: float


# =============================================================================
# TrainingPort Protocol (Low-Level Framework Operations)
# =============================================================================


@runtime_checkable
class TrainingPort(Protocol):
    """Abstract protocol for framework-specific training operations.

    This port abstracts ML framework-specific operations:
    - Model parameter access (tree_flatten, etc.)
    - LoRA layer creation and injection
    - Auto-differentiation (value_and_grad)
    - Optimizer updates
    - Model serialization

    Implementations: backend-specific training adapters
    """

    @property
    def backend(self) -> "Backend":
        """Get the compute backend used by this training adapter."""
        ...

    # =========================================================================
    # Model Inspection
    # =========================================================================

    def get_weight_matrices(
        self, model: Any, layer_pattern: str | None = None
    ) -> dict[str, "Array"]:
        """Extract weight matrices from a model.

        Args:
            model: Framework-specific model object.
            layer_pattern: Optional regex pattern to filter layer names.

        Returns:
            Dict mapping layer_key -> weight array (as Backend arrays).
        """
        ...

    def get_parameter_info(self, model: Any) -> list[ParameterInfo]:
        """Get information about all parameters in the model.

        Args:
            model: Framework-specific model object.

        Returns:
            List of ParameterInfo for all parameters.
        """
        ...

    def count_trainable_params(self, model: Any) -> int:
        """Count total trainable parameters.

        Args:
            model: Framework-specific model object.

        Returns:
            Total number of trainable parameters.
        """
        ...

    def get_num_layers(self, model: Any) -> int:
        """Get number of transformer layers in the model.

        Args:
            model: Framework-specific model object.

        Returns:
            Number of layers.
        """
        ...

    # =========================================================================
    # LoRA Operations
    # =========================================================================

    def apply_lora(
        self,
        model: Any,
        configs: list[LoRALayerConfig],
    ) -> dict[str, Any]:
        """Apply LoRA layers to specified model layers.

        Creates LoRA adapters with spectral-normalized initialization
        where ||B @ A||_spectral = σ_k at initialization.

        Args:
            model: Framework-specific model object.
            configs: List of LoRA configurations for each target layer.

        Returns:
            Dict mapping layer_key -> framework-specific LoRA layer handle.
        """
        ...

    def get_lora_weights(
        self, lora_layers: dict[str, Any]
    ) -> dict[str, tuple["Array", "Array"]]:
        """Extract LoRA weights (A, B matrices) from LoRA layers.

        Args:
            lora_layers: Dict from apply_lora().

        Returns:
            Dict mapping layer_key -> (A, B) weight arrays.
        """
        ...

    def compute_lora_spectral_norm(
        self, lora_layers: dict[str, Any], layer_key: str
    ) -> float:
        """Compute spectral norm of B @ A for a LoRA layer.

        Args:
            lora_layers: Dict from apply_lora().
            layer_key: Which layer to compute for.

        Returns:
            ||B @ A||_spectral
        """
        ...

    def enforce_spectral_bounds(
        self, lora_layers: dict[str, Any], configs: list[LoRALayerConfig]
    ) -> int:
        """Enforce spectral bound ||B @ A|| <= σ_k for all LoRA layers.

        Rescales B matrices where the bound is violated.

        Args:
            lora_layers: Dict from apply_lora().
            configs: LoRA configurations with sigma_k values.

        Returns:
            Number of layers that were rescaled (saturated).
        """
        ...

    # =========================================================================
    # Training Operations
    # =========================================================================

    def freeze_model(self, model: Any) -> None:
        """Freeze all model parameters (no gradients).

        Args:
            model: Framework-specific model object.
        """
        ...

    def unfreeze_lora(self, model: Any, lora_layers: dict[str, Any]) -> None:
        """Unfreeze only LoRA parameters for training.

        Args:
            model: Framework-specific model object.
            lora_layers: Dict from apply_lora().
        """
        ...

    def compute_loss_and_gradients(
        self,
        model: Any,
        input_ids: "Array",
        target_ids: "Array",
    ) -> tuple[float, dict[str, "Array"]]:
        """Compute loss and gradients for a batch.

        Args:
            model: Framework-specific model object.
            input_ids: Input token IDs [batch, seq].
            target_ids: Target token IDs [batch, seq].

        Returns:
            Tuple of (loss_value, gradient_dict).
            gradient_dict maps parameter keys to gradient arrays.
        """
        ...

    def apply_gradients(
        self,
        model: Any,
        gradients: dict[str, "Array"],
        learning_rates: dict[str, float],
        weight_decay: dict[str, float] | None = None,
    ) -> None:
        """Apply gradient updates to model parameters.

        Performs: param = param - lr * grad - decay * param

        Args:
            model: Framework-specific model object.
            gradients: Gradients from compute_loss_and_gradients().
            learning_rates: Per-parameter learning rates.
            weight_decay: Optional per-parameter weight decay.
        """
        ...

    def get_gradient_info(
        self, model: Any, gradients: dict[str, "Array"]
    ) -> list[GradientInfo]:
        """Get information about gradients for monitoring.

        Args:
            model: Framework-specific model object.
            gradients: Gradients from compute_loss_and_gradients().

        Returns:
            List of GradientInfo with norms for each parameter.
        """
        ...

    # =========================================================================
    # Forward Pass Operations
    # =========================================================================

    def tokenize(
        self, tokenizer: Any, text: str, max_length: int = 512
    ) -> "Array":
        """Tokenize text using the provided tokenizer.

        Args:
            tokenizer: Tokenizer object.
            text: Text to tokenize.
            max_length: Maximum sequence length.

        Returns:
            Token IDs as Backend array.
        """
        ...

    def forward(self, model: Any, input_ids: "Array") -> "Array":
        """Run forward pass through model.

        Args:
            model: Framework-specific model object.
            input_ids: Input token IDs [batch, seq].

        Returns:
            Logits [batch, seq, vocab].
        """
        ...

    def get_hidden_states(
        self, model: Any, input_ids: "Array", layer_idx: int
    ) -> "Array":
        """Get hidden states at a specific layer.

        Args:
            model: Framework-specific model object.
            input_ids: Input token IDs [batch, seq].
            layer_idx: Which layer to extract from.

        Returns:
            Hidden states [batch, seq, hidden_dim].
        """
        ...

    # =========================================================================
    # Serialization
    # =========================================================================

    def save_lora_adapter(
        self,
        lora_layers: dict[str, Any],
        configs: list[LoRALayerConfig],
        output_path: "Path",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save LoRA adapter weights and config.

        Args:
            lora_layers: Dict from apply_lora().
            configs: LoRA configurations.
            output_path: Directory to save to.
            metadata: Optional metadata to include.
        """
        ...

    def load_lora_adapter(
        self, model: Any, adapter_path: "Path"
    ) -> tuple[dict[str, Any], list[LoRALayerConfig]]:
        """Load LoRA adapter and apply to model.

        Args:
            model: Framework-specific model object.
            adapter_path: Directory containing adapter files.

        Returns:
            Tuple of (lora_layers, configs).
        """
        ...


# =============================================================================
# TrainingEngine Protocol (High-Level Job Management)
# =============================================================================


@runtime_checkable
class TrainingEngine(Protocol):
    """Port for running training jobs on a backend.

    High-level job management - start, stop, pause, resume, status.
    Uses TrainingPort internally for framework-specific operations.
    """

    def preflight(self, config: "TrainingSpec") -> "PreflightResult":
        """Validate training configuration and return a preflight report."""
        ...

    def start(
        self, config: "TrainingSpec", stream_events: bool = False
    ) -> tuple["TrainingJob", list[dict]]:
        """Start a training job and return the job plus any initial events."""
        ...

    def status(self, job_id: str) -> "TrainingJob":
        """Return the current status for a job."""
        ...

    def pause(self, job_id: str) -> "TrainingJob":
        """Pause a running training job."""
        ...

    def resume(self, job_id: str) -> "TrainingJob":
        """Resume a paused training job."""
        ...

    def cancel(self, job_id: str) -> "TrainingJob":
        """Cancel a training job."""
        ...

    def logs(self, job_id: str, tail: int = 100) -> list[str]:
        """Return the latest log lines for a job."""
        ...


__all__ = [
    # Data classes
    "LoRALayerConfig",
    "ParameterInfo",
    "GradientInfo",
    # Protocols
    "TrainingPort",
    "TrainingEngine",
]
