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

"""MLX Training Adapter implementing TrainingPort.

All MLX-specific training code lives here:
- nn.Module subclasses (GeometricLoRALinear)
- tree_flatten/tree_unflatten for parameter access
- value_and_grad for auto-differentiation
- MLX-specific model introspection

Domain code uses the Backend protocol for numeric operations.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.core.domain.training.geometric_optimizer import (
        OptimizerGeometryConfig,
    )

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from modelcypher.backends.mlx_backend import MLXBackend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    sqrt_scalar,
)
from modelcypher.ports.backend import Backend
from modelcypher.ports.training import (
    GradientInfo,
    LoRALayerConfig,
    ParameterInfo,
    TrainingPort,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MLX-Specific LoRA Layer
# =============================================================================


class GeometricLoRALinear(nn.Module):
    """MLX Linear layer with geometry-normalized LoRA.

    The LoRA delta is:
        delta = σ_k * (B @ A) / ||B @ A||_spectral

    Where σ_k is the smallest significant singular value of the base weight.
    This guarantees the perturbation respects the spectral structure.

    Initialization: ||B @ A||_spectral = σ_k at step 0
    Uses FULL geometric budget from step 0, derived from base weight spectral structure.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        sigma_k: float,
        rank: int,
        backend: Backend,
    ):
        super().__init__()

        in_features = base_layer.weight.shape[1]
        out_features = base_layer.weight.shape[0]

        self.base_weight = base_layer.weight
        self.base_bias = getattr(base_layer, "bias", None)
        self.sigma_k = sigma_k
        self.rank = rank
        self._backend = backend

        # Spectral-normalized initialization (geometry-derived)
        # Initialize so ||B @ A||_spectral = σ_k at step 0
        # Each matrix gets ||·||_spectral = sqrt(σ_k)
        sqrt_sigma_k = sqrt_scalar(sigma_k, backend)
        sqrt_eps = division_epsilon(backend, backend.array([sigma_k]))

        # Initialize A: [rank, in_features]
        A_init = mx.random.normal(shape=(rank, in_features))
        A_spectral = self._spectral_norm(A_init)
        self.lora_a = A_init * (sqrt_sigma_k / (float(A_spectral) + sqrt_eps))

        # Initialize B: [out_features, rank]
        B_init = mx.random.normal(shape=(out_features, rank))
        B_spectral = self._spectral_norm(B_init)
        self.lora_b = B_init * (sqrt_sigma_k / (float(B_spectral) + sqrt_eps))

        mx.eval(self.lora_a, self.lora_b)

        logger.debug(
            "Spectral init: σ_k=%.4f, ||A||=%.4f, ||B||=%.4f, target=%.4f",
            sigma_k,
            float(self._spectral_norm(self.lora_a)),
            float(self._spectral_norm(self.lora_b)),
            sqrt_sigma_k,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Base computation
        out = x @ self.base_weight.T
        if self.base_bias is not None:
            out = out + self.base_bias

        # LoRA delta: B @ A gives [out_features, in_features]
        delta = self.lora_b @ self.lora_a

        # Spectral normalization: scale by σ_k / ||B @ A||_spectral
        spectral_norm = self._spectral_norm(delta)

        # Normalize and scale by σ_k (add epsilon for numerical stability)
        delta_normalized = delta / (spectral_norm + 1e-8)
        lora_out = x @ (self.sigma_k * delta_normalized).T

        out = out + lora_out
        return out

    def _spectral_norm(self, M: mx.array, n_iters: int = 3) -> mx.array:
        """Power iteration for spectral norm.

        Uses deterministic initialization and avoids Python if-statements
        to ensure gradients flow properly through the computation.
        """
        # Initialize with deterministic vector (sum of columns)
        v = mx.ones((M.shape[1],)) / mx.sqrt(mx.array(M.shape[1], dtype=M.dtype))

        for _ in range(n_iters):
            u = M @ v
            u_norm = mx.maximum(mx.linalg.norm(u), mx.array(1e-8))
            u = u / u_norm

            v = M.T @ u
            v_norm = mx.maximum(mx.linalg.norm(v), mx.array(1e-8))
            v = v / v_norm

        return mx.linalg.norm(M @ v)


# =============================================================================
# MLX Training Adapter
# =============================================================================


class MLXTrainingAdapter(TrainingPort):
    """MLX implementation of TrainingPort.

    Handles all MLX-specific training operations including:
    - Model parameter access via tree_flatten
    - LoRA layer creation and injection
    - Auto-differentiation via value_and_grad
    - Gradient-based optimization
    """

    def __init__(self, backend: MLXBackend | None = None):
        self._backend = backend or MLXBackend()
        self._lora_configs: dict[str, LoRALayerConfig] = {}

    @property
    def backend(self) -> Backend:
        return self._backend

    # =========================================================================
    # Model Inspection
    # =========================================================================

    def get_weight_matrices(
        self, model: Any, layer_pattern: str | None = None
    ) -> dict[str, mx.array]:
        """Extract weight matrices from model."""
        flat_params = tree_flatten(model.parameters())
        result = {}

        pattern = re.compile(layer_pattern) if layer_pattern else None

        for key, param in flat_params:
            if not isinstance(param, mx.array):
                continue
            # Only 2D weight matrices
            if param.ndim != 2:
                continue
            # Skip very small matrices
            if min(param.shape) < 4:
                continue
            # Apply pattern filter
            if pattern and not pattern.search(key):
                continue

            result[key] = param

        return result

    def get_parameter_info(self, model: Any) -> list[ParameterInfo]:
        """Get information about all parameters."""
        flat_params = tree_flatten(model.parameters())
        result = []

        for key, param in flat_params:
            if not isinstance(param, mx.array):
                continue

            result.append(
                ParameterInfo(
                    key=key,
                    shape=tuple(param.shape),
                    dtype=str(param.dtype),
                    size=int(param.size),
                )
            )

        return result

    def count_trainable_params(self, model: Any) -> int:
        """Count total trainable parameters."""
        total = 0
        flat_params = tree_flatten(model.trainable_parameters())
        for _, param in flat_params:
            if isinstance(param, mx.array):
                total += int(param.size)
        return total

    def get_num_layers(self, model: Any) -> int:
        """Get number of transformer layers."""
        base_model = getattr(model, "model", model)
        layers = getattr(base_model, "layers", [])
        return len(layers)

    # =========================================================================
    # LoRA Operations
    # =========================================================================

    def apply_lora(
        self,
        model: Any,
        configs: list[LoRALayerConfig],
    ) -> dict[str, GeometricLoRALinear]:
        """Apply LoRA layers to specified model layers."""
        base_model = getattr(model, "model", model)
        layers = base_model.layers

        lora_layers = {}
        self._lora_configs = {}

        for config in configs:
            layer_key = config.layer_key

            # Parse layer key: model.layers.{idx}.self_attn.{proj}
            parts = layer_key.split(".")
            if len(parts) < 5:
                logger.warning("Invalid layer key format: %s, skipping", layer_key)
                continue

            try:
                layer_idx = int(parts[2])
                proj_name = parts[4]
            except (ValueError, IndexError):
                logger.warning("Could not parse layer key: %s, skipping", layer_key)
                continue

            if layer_idx >= len(layers):
                logger.warning(
                    "Layer index %d out of range for %s, skipping", layer_idx, layer_key
                )
                continue

            layer = layers[layer_idx]
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                logger.warning("No self_attn found in layer %d, skipping", layer_idx)
                continue

            base_linear = getattr(attn, proj_name, None)
            if base_linear is None or not hasattr(base_linear, "weight"):
                logger.warning(
                    "No valid linear layer at %s.%s, skipping", layer_idx, proj_name
                )
                continue

            # Create geometric LoRA layer
            lora_layer = GeometricLoRALinear(
                base_layer=base_linear,
                sigma_k=config.sigma_k,
                rank=config.rank,
                backend=self._backend,
            )

            # Replace in model
            setattr(attn, proj_name, lora_layer)
            lora_layers[layer_key] = lora_layer
            self._lora_configs[layer_key] = config

            logger.info(
                "Applied geometric LoRA to %s: rank=%d, σ_k=%.4f",
                layer_key,
                config.rank,
                config.sigma_k,
            )

        return lora_layers

    def get_lora_weights(
        self, lora_layers: dict[str, GeometricLoRALinear]
    ) -> dict[str, tuple[mx.array, mx.array]]:
        """Extract LoRA weights (A, B matrices)."""
        result = {}
        for layer_key, lora_layer in lora_layers.items():
            result[layer_key] = (lora_layer.lora_a, lora_layer.lora_b)
        return result

    def compute_lora_spectral_norm(
        self, lora_layers: dict[str, GeometricLoRALinear], layer_key: str
    ) -> float:
        """Compute spectral norm of B @ A."""
        if layer_key not in lora_layers:
            raise KeyError(f"Layer {layer_key} not in lora_layers")

        lora_layer = lora_layers[layer_key]
        delta = lora_layer.lora_b @ lora_layer.lora_a
        mx.eval(delta)

        spectral = lora_layer._spectral_norm(delta)
        mx.eval(spectral)

        return float(spectral)

    def enforce_spectral_bounds(
        self, lora_layers: dict[str, GeometricLoRALinear], configs: list[LoRALayerConfig]
    ) -> int:
        """Enforce spectral bound ||B @ A|| <= σ_k."""
        config_map = {c.layer_key: c for c in configs}
        n_saturated = 0

        for layer_key, lora_layer in lora_layers.items():
            config = config_map.get(layer_key)
            if config is None:
                continue

            sigma_k = config.sigma_k
            if sigma_k <= 0:
                continue

            # Compute current spectral norm
            spectral_norm = self.compute_lora_spectral_norm(lora_layers, layer_key)

            if spectral_norm > sigma_k:
                # Rescale B to enforce constraint
                scale = sigma_k / spectral_norm
                lora_layer.lora_b = lora_layer.lora_b * scale
                mx.eval(lora_layer.lora_b)
                n_saturated += 1

        return n_saturated

    # =========================================================================
    # Training Operations
    # =========================================================================

    def freeze_model(self, model: Any) -> None:
        """Freeze all model parameters."""
        model.freeze()

    def unfreeze_lora(
        self, model: Any, lora_layers: dict[str, GeometricLoRALinear]
    ) -> None:
        """Unfreeze only LoRA parameters."""
        for lora_layer in lora_layers.values():
            lora_layer.unfreeze()
            # Re-freeze base weights (they were unfrozen with the module)
            lora_layer.freeze(keys=["base_weight", "base_bias"], strict=False)

    def compute_loss_and_gradients(
        self,
        model: Any,
        input_ids: mx.array,
        target_ids: mx.array,
    ) -> tuple[float, dict[str, mx.array]]:
        """Compute loss and gradients for a batch."""

        def loss_fn(model):
            logits = model(input_ids)
            # Flatten for cross entropy
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = target_ids.reshape(-1)
            loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")
            return loss

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss)

        # Flatten gradients to dict
        flat_grads = tree_flatten(grads)
        grad_dict = {key: grad for key, grad in flat_grads if grad is not None}

        return float(loss), grad_dict

    def apply_gradients(
        self,
        model: Any,
        gradients: dict[str, mx.array],
        learning_rates: dict[str, float],
        weight_decay: dict[str, float] | None = None,
    ) -> None:
        """Apply gradient updates to model parameters."""
        weight_decay = weight_decay or {}

        flat_params = tree_flatten(model.parameters())
        param_dict = {key: param for key, param in flat_params}

        updates = []
        for key, param in flat_params:
            if key not in gradients:
                updates.append((key, param))
                continue

            grad = gradients[key]
            lr = learning_rates.get(key, 1e-4)
            decay = weight_decay.get(key, 0.0)

            # SGD update: w = w - lr*grad - decay*w
            new_param = param - lr * grad
            if decay > 0:
                new_param = new_param - decay * param

            updates.append((key, new_param))

        new_params = tree_unflatten(updates)
        model.update(new_params)

    def get_gradient_info(
        self, model: Any, gradients: dict[str, mx.array]
    ) -> list[GradientInfo]:
        """Get information about gradients."""
        flat_params = tree_flatten(model.parameters())
        param_dict = {key: param for key, param in flat_params}

        result = []
        for key, grad in gradients.items():
            if grad is None:
                continue

            grad_norm = float(mx.sqrt(mx.sum(grad * grad)))
            mx.eval(grad_norm)

            param = param_dict.get(key)
            if param is not None:
                param_norm = float(mx.sqrt(mx.sum(param * param)))
                mx.eval(param_norm)
            else:
                param_norm = 0.0

            result.append(
                GradientInfo(param_key=key, grad_norm=grad_norm, param_norm=param_norm)
            )

        return result

    # =========================================================================
    # Forward Pass Operations
    # =========================================================================

    def tokenize(self, tokenizer: Any, text: str, max_length: int = 512) -> mx.array:
        """Tokenize text."""
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return mx.array(tokens)

    def forward(self, model: Any, input_ids: mx.array) -> mx.array:
        """Run forward pass."""
        logits = model(input_ids)
        mx.eval(logits)
        return logits

    def get_hidden_states(
        self, model: Any, input_ids: mx.array, layer_idx: int
    ) -> mx.array:
        """Get hidden states at a specific layer."""
        base_model = getattr(model, "model", model)
        embed_module = getattr(base_model, "embed_tokens", None)
        layers = getattr(base_model, "layers", [])

        if embed_module is None:
            raise ValueError("Could not find embed_tokens module")

        if layer_idx >= len(layers):
            raise ValueError(f"Layer {layer_idx} out of range")

        # Get embeddings
        h = embed_module(input_ids)
        mx.eval(h)

        # Forward through layers up to layer_idx
        for i in range(layer_idx + 1):
            result = layers[i](h)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result
            mx.eval(h)

        return h

    # =========================================================================
    # Serialization
    # =========================================================================

    def save_lora_adapter(
        self,
        lora_layers: dict[str, GeometricLoRALinear],
        configs: list[LoRALayerConfig],
        output_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save LoRA adapter weights and config."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect weights
        weights = {}
        for layer_key, lora_layer in lora_layers.items():
            weights[f"{layer_key}.lora_a"] = lora_layer.lora_a
            weights[f"{layer_key}.lora_b"] = lora_layer.lora_b

        # Save weights
        weights_path = output_path / "lora_weights.safetensors"
        mx.save_safetensors(str(weights_path), weights)

        # Build config
        config_dict = {
            "type": "geometric_lora",
            "target_modules": [c.layer_key for c in configs],
            "layer_configs": {
                c.layer_key: {
                    "rank": c.rank,
                    "sigma_k": c.sigma_k,
                    "in_features": c.in_features,
                    "out_features": c.out_features,
                }
                for c in configs
            },
        }

        if metadata:
            config_dict["metadata"] = metadata

        # Save config
        config_path = output_path / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info("Saved geometric adapter to %s", output_path)

    def load_lora_adapter(
        self, model: Any, adapter_path: Path
    ) -> tuple[dict[str, GeometricLoRALinear], list[LoRALayerConfig]]:
        """Load LoRA adapter and apply to model."""
        adapter_path = Path(adapter_path)

        # Load config
        config_path = adapter_path / "adapter_config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        # Build configs
        configs = []
        for layer_key, layer_config in config_dict.get("layer_configs", {}).items():
            configs.append(
                LoRALayerConfig(
                    layer_key=layer_key,
                    rank=layer_config["rank"],
                    sigma_k=layer_config["sigma_k"],
                    in_features=layer_config["in_features"],
                    out_features=layer_config["out_features"],
                )
            )

        # Apply LoRA layers
        lora_layers = self.apply_lora(model, configs)

        # Load weights
        weights_path = adapter_path / "lora_weights.safetensors"
        weights = mx.load(str(weights_path))

        # Apply weights to LoRA layers
        for layer_key, lora_layer in lora_layers.items():
            a_key = f"{layer_key}.lora_a"
            b_key = f"{layer_key}.lora_b"

            if a_key in weights:
                lora_layer.lora_a = weights[a_key]
            if b_key in weights:
                lora_layer.lora_b = weights[b_key]

            mx.eval(lora_layer.lora_a, lora_layer.lora_b)

        logger.info("Loaded geometric adapter from %s", adapter_path)

        return lora_layers, configs


# =============================================================================
# MLX Geometric Optimizer
# =============================================================================


class MLXGeometricOptimizer:
    """MLX optimizer with geometry-derived per-layer scaling and BB adaptation.

    Uses the pure geometry config from geometric_optimizer module.
    No momentum. No magic hyperparameters. Gradient descent with learning rate
    derived from spectral structure and local curvature.

    First step:
        LR = 1 / σ_max_i (spectral bound, no gradient history yet)

    Subsequent steps:
        LR = (s·s) / (s·y) bounded to [σ_k/σ_max, 1/σ_max]
        where s = θ_k - θ_{k-1}, y = g_k - g_{k-1}
    """

    def __init__(
        self,
        base_decay: float = 0.0,
        gradient_clip_mode: str = "none",
        global_clip_value: float = 1.0,
    ):
        """Initialize geometric optimizer.

        Args:
            base_decay: Base weight decay (will be scaled per-layer by condition).
            gradient_clip_mode: One of "none", "global", "spectral" (kept for API compat).
            global_clip_value: Clip threshold for "global" mode.
        """
        self.base_decay = base_decay
        self.gradient_clip_mode = gradient_clip_mode
        self.global_clip_value = global_clip_value
        self._config: "OptimizerGeometryConfig | None" = None
        self._backend: Backend | None = None
        self._initialized = False
        self._learning_rate_override: float | None = None

        # Barzilai-Borwein state tracking
        self._prev_params: dict[str, mx.array] = {}
        self._prev_grads: dict[str, mx.array] = {}
        self._step_count: int = 0
        self._per_layer_lr: dict[str, float] = {}
        self._sdy_history: list[float] = []
        self._gradient_norms: dict[str, list[float]] = {}

    @property
    def base_lr(self) -> float | None:
        """Return base learning rate (1/max_sigma)."""
        return self._config.base_lr if self._config else None

    @property
    def learning_rate(self) -> float | None:
        """Return learning rate (for engine compatibility)."""
        if self._learning_rate_override is not None:
            return self._learning_rate_override
        return self.base_lr

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set learning rate override (for warmup)."""
        self._learning_rate_override = value

    @property
    def layer_configs(self) -> dict:
        """Return layer configs dict for backwards compatibility."""
        return self._config.layer_configs if self._config else {}

    @property
    def state(self) -> dict:
        """Return optimizer state for checkpointing."""
        return self.state_dict()

    @state.setter
    def state(self, value: dict) -> None:
        """Load optimizer state from checkpoint."""
        self.load_state_dict(value)

    def load_state(self, state: dict) -> None:
        """Load optimizer state (alias for load_state_dict)."""
        self.load_state_dict(state)

    def init_from_model(self, model: Any, backend: Backend | None = None) -> None:
        """Compute spectral structure and derive all parameters from geometry.

        Args:
            model: The MLX model to optimize
            backend: Backend for computations (defaults to MLXBackend)
        """
        from modelcypher.core.domain.training.geometric_optimizer import (
            derive_optimizer_geometry_config,
        )

        if backend is None:
            backend = MLXBackend()
        self._backend = backend

        logger.info("Analyzing model geometry for optimizer initialization...")

        # Get weight matrices from model
        flat_params = tree_flatten(model.parameters())
        weights = {}
        for key, param in flat_params:
            if isinstance(param, mx.array) and param.ndim == 2 and min(param.shape) >= 4:
                weights[key] = param

        # Derive config from geometry
        self._config = derive_optimizer_geometry_config(weights, backend)
        self._initialized = True

        logger.info(
            "Geometric optimizer initialized: base_lr=%.6f (from max σ=%.4f)",
            self._config.base_lr, self._config.max_sigma
        )

    def _compute_layer_lr(
        self, key: str, param: mx.array, grad: mx.array
    ) -> float:
        """Compute BB-adapted learning rate for a layer.

        Args:
            key: Parameter key
            param: Current parameter value
            grad: Current gradient

        Returns:
            Learning rate for this update
        """
        from modelcypher.core.domain.training.geometric_optimizer import (
            compute_barzilai_borwein_lr,
        )

        config = self._config.layer_configs.get(key)
        if config is None:
            return self._config.base_lr

        # First step: use spectral LR (no gradient history yet)
        if self._step_count == 0 or key not in self._prev_grads:
            return self._config.base_lr * config.lr_scale

        prev_param = self._prev_params.get(key)
        prev_grad = self._prev_grads.get(key)

        if prev_param is None or prev_grad is None:
            return self._config.base_lr * config.lr_scale

        # Compute s and y for BB
        s = param - prev_param
        y = grad - prev_grad

        s_flat = s.flatten()
        y_flat = y.flatten()

        s_dot_s = float(mx.sum(s_flat * s_flat))
        s_dot_y = float(mx.sum(s_flat * y_flat))

        # Track s·y for stability monitoring
        self._sdy_history.append(s_dot_y)
        if len(self._sdy_history) > 100:
            self._sdy_history = self._sdy_history[-100:]

        # Use pure function from config module
        return compute_barzilai_borwein_lr(
            s_dot_s, s_dot_y, config, self._config.base_lr
        )

    def update(self, model: Any, gradients: Any) -> None:
        """Apply geometry-derived gradient update with BB adaptation.

        Args:
            model: The MLX model
            gradients: Gradients from value_and_grad (tree structure)
        """
        if not self._initialized:
            raise RuntimeError("Optimizer not initialized. Call init_from_model first.")

        flat_params = tree_flatten(model.parameters())
        flat_grads = tree_flatten(gradients)

        param_dict = {key: param for key, param in flat_params}
        grad_dict = {key: grad for key, grad in flat_grads if grad is not None}

        updates = []
        for key, param in flat_params:
            grad = grad_dict.get(key)

            if grad is None:
                updates.append((key, param))
                continue

            # Compute BB-adapted learning rate
            lr = self._compute_layer_lr(key, param, grad)
            self._per_layer_lr[key] = lr

            # Weight decay (condition-scaled)
            config = self._config.layer_configs.get(key)
            if config and self.base_decay > 0:
                decay = self.base_decay * config.decay_scale
                new_param = param - lr * grad - decay * param
            else:
                new_param = param - lr * grad

            updates.append((key, new_param))

        # Store for next BB computation
        self._prev_params = {key: param_dict[key] for key in grad_dict}
        self._prev_grads = {key: grad_dict[key] for key in grad_dict}
        self._step_count += 1

        # Update model
        new_params = tree_unflatten(updates)
        model.update(new_params)

    def get_lr_stats(self) -> dict:
        """Return per-layer LR statistics for logging."""
        from modelcypher.core.domain.training.geometric_optimizer import (
            compute_lr_statistics,
        )
        return compute_lr_statistics(self._per_layer_lr, self._backend)

    def get_bb_stability(self) -> float:
        """Return variance of s·y values for stability monitoring."""
        if len(self._sdy_history) < 10:
            return float('inf')
        recent = self._sdy_history[-10:]
        mean_val = sum(recent) / len(recent)
        variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)
        return float(variance)

    def is_bb_stable(self, threshold: float = 1e-4) -> bool:
        """Check if BB curvature estimates have stabilized."""
        if len(self._sdy_history) < 10:
            return False
        recent = self._sdy_history[-10:]
        mean_sdy = sum(abs(x) for x in recent) / len(recent)
        if mean_sdy < 1e-10:
            return True
        mean_val = sum(recent) / len(recent)
        variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)
        return float(variance / (mean_sdy ** 2)) < threshold

    def get_gradient_stats(self) -> dict[str, float]:
        """Return gradient statistics for monitoring."""
        return self._gradient_norms.copy()

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        config_dict = self._config.to_dict() if self._config else {}
        return {
            "type": "geometric",
            "base_lr": self.base_lr,
            "base_decay": self.base_decay,
            "step_count": self._step_count,
            "sdy_history": self._sdy_history,
            "layer_configs": config_dict.get("layer_configs", {}),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load optimizer state from checkpoint."""
        from modelcypher.core.domain.training.geometric_optimizer import (
            OptimizerGeometryConfig,
            LayerOptimizerConfig,
        )
        self._step_count = state.get("step_count", 0)
        self._sdy_history = state.get("sdy_history", [])
        self.base_decay = state.get("base_decay", 0.0)

        # Reconstruct config if present
        if state.get("layer_configs") or state.get("config"):
            layer_configs_data = state.get("layer_configs") or state.get("config", {}).get("layer_configs", {})
            layer_configs = {}
            for key, cfg_dict in layer_configs_data.items():
                layer_configs[key] = LayerOptimizerConfig(
                    layer_key=key,
                    sigma_max=cfg_dict.get("sigma_max", 1.0),
                    sigma_k=cfg_dict.get("sigma_k", 0.01),
                    lr_scale=cfg_dict.get("lr_scale", 1.0),
                    epsilon=cfg_dict.get("epsilon", 1e-8),
                    decay_scale=cfg_dict.get("decay_scale", 1.0),
                )

            base_lr = state.get("base_lr", 1e-4)
            max_sigma = 1.0 / base_lr if base_lr > 0 else 1.0

            self._config = OptimizerGeometryConfig(
                base_lr=base_lr,
                max_sigma=max_sigma,
                layer_configs=layer_configs,
            )
            self._initialized = True


# Alias for backwards compatibility
GeometricOptimizer = MLXGeometricOptimizer


__all__ = [
    "GeometricLoRALinear",
    "MLXTrainingAdapter",
    "MLXGeometricOptimizer",
    "GeometricOptimizer",
]
