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
SAE Training Loop.

Trains Sparse Autoencoders on model activations using gradient descent.
Supports:
- Adam optimizer with configurable learning rate
- Learning rate warmup and decay
- Dead feature resampling
- Decoder normalization
- Checkpointing

References:
    - "Scaling Monosemanticity" (Anthropic, 2024) - Training methodology
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.interpretability.sae import (
    SAEConfig,
    SAEEncodingResult,
    SAEWeights,
    SparseAutoencoder,
    derive_sparsity_coefficient,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SAETrainingConfig:
    """Configuration for SAE training.

    Attributes
    ----------
    learning_rate : float
        Base learning rate for Adam optimizer.
    num_steps : int
        Total number of training steps.
    batch_size : int
        Batch size for training.
    warmup_steps : int
        Number of warmup steps for learning rate.
    sparsity_coefficient : float | None
        L1 sparsity coefficient. If None, derived from data.
    dead_feature_threshold : int
        Steps without activation before feature is considered dead.
    resample_dead_features : bool
        Whether to resample dead features during training.
    resample_frequency : int
        How often to check and resample dead features (in steps).
    normalize_decoder_frequency : int
        How often to normalize decoder columns (in steps).
    log_frequency : int
        How often to log training metrics (in steps).
    checkpoint_frequency : int | None
        How often to save checkpoints (in steps). None = no checkpoints.
    """

    learning_rate: float = 1e-4
    num_steps: int = 10000
    batch_size: int = 4096
    warmup_steps: int = 1000
    sparsity_coefficient: float | None = None
    dead_feature_threshold: int = 10000
    resample_dead_features: bool = True
    resample_frequency: int = 1000
    normalize_decoder_frequency: int = 100
    log_frequency: int = 100
    checkpoint_frequency: int | None = None


@dataclass
class TrainingState:
    """Mutable training state.

    Attributes
    ----------
    weights : SAEWeights
        Current SAE weights.
    step : int
        Current training step.
    m_enc : Any
        Adam first moment for encoder weights.
    v_enc : Any
        Adam second moment for encoder weights.
    m_dec : Any
        Adam first moment for decoder weights.
    v_dec : Any
        Adam second moment for decoder weights.
    m_b_enc : Any
        Adam first moment for encoder bias.
    v_b_enc : Any
        Adam second moment for encoder bias.
    m_b_dec : Any
        Adam first moment for decoder bias.
    v_b_dec : Any
        Adam second moment for decoder bias.
    feature_activation_counts : Any
        Number of times each feature has activated.
    total_loss : float
        Accumulated loss for logging.
    total_recon_loss : float
        Accumulated reconstruction loss.
    total_l1_loss : float
        Accumulated L1 loss.
    total_sparsity : float
        Accumulated sparsity.
    log_count : int
        Number of steps since last log.
    """

    weights: SAEWeights
    step: int = 0
    m_enc: Any = None
    v_enc: Any = None
    m_dec: Any = None
    v_dec: Any = None
    m_b_enc: Any = None
    v_b_enc: Any = None
    m_b_dec: Any = None
    v_b_dec: Any = None
    feature_activation_counts: Any = None
    total_loss: float = 0.0
    total_recon_loss: float = 0.0
    total_l1_loss: float = 0.0
    total_sparsity: float = 0.0
    log_count: int = 0


@dataclass(frozen=True)
class SAETrainingResult:
    """Result of SAE training.

    Attributes
    ----------
    weights : SAEWeights
        Trained SAE weights.
    final_loss : float
        Final training loss.
    final_reconstruction_loss : float
        Final reconstruction loss.
    final_sparsity : float
        Final L0 sparsity.
    dead_features : int
        Number of features that never activated.
    training_steps : int
        Total steps trained.
    training_time_seconds : float
        Total training time.
    loss_history : list[float]
        Loss at each logging step.
    """

    weights: SAEWeights
    final_loss: float
    final_reconstruction_loss: float
    final_sparsity: float
    dead_features: int
    training_steps: int
    training_time_seconds: float
    loss_history: list[float] = field(default_factory=list)


class SAETrainer:
    """Trains Sparse Autoencoders.

    Example
    -------
    >>> config = SAETrainingConfig(num_steps=5000)
    >>> trainer = SAETrainer(sae_config, config)
    >>> result = trainer.train(activation_iterator)
    >>> trained_weights = result.weights
    """

    def __init__(
        self,
        sae_config: SAEConfig,
        training_config: SAETrainingConfig,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize trainer.

        Parameters
        ----------
        sae_config : SAEConfig
            SAE architecture configuration.
        training_config : SAETrainingConfig
            Training hyperparameters.
        backend : Backend, optional
            Computation backend.
        """
        self._sae_config = sae_config
        self._training_config = training_config
        self._backend = backend or get_default_backend()
        self._sae = SparseAutoencoder(sae_config, self._backend)

    def train(
        self,
        activation_iterator: Iterator[Any] | Callable[[], Any],
        initial_weights: SAEWeights | None = None,
        on_checkpoint: Callable[[SAEWeights, int], None] | None = None,
    ) -> SAETrainingResult:
        """Train SAE on activations.

        Parameters
        ----------
        activation_iterator : Iterator or Callable
            Iterator yielding activation batches, or callable that returns a batch.
            Each batch should have shape [batch_size, hidden_dim].
        initial_weights : SAEWeights, optional
            Initial weights. If None, randomly initialized.
        on_checkpoint : Callable, optional
            Called with (weights, step) at each checkpoint.

        Returns
        -------
        SAETrainingResult
            Training result with trained weights.
        """
        b = self._backend
        config = self._training_config

        # Initialize state
        weights = initial_weights or self._sae.initialize_weights()
        state = self._initialize_state(weights)

        # Derive sparsity coefficient if needed
        sparsity_coeff = config.sparsity_coefficient
        if sparsity_coeff is None:
            # Get a sample batch to derive coefficient
            batch = self._get_batch(activation_iterator)
            sparsity_coeff = derive_sparsity_coefficient(batch, b)
            logger.info(f"Derived sparsity coefficient: {sparsity_coeff:.6f}")

        loss_history: list[float] = []
        start_time = time.perf_counter()

        for step in range(config.num_steps):
            state.step = step

            # Get batch
            batch = self._get_batch(activation_iterator)
            batch = b.array(batch) if not hasattr(batch, "shape") else batch
            b.eval(batch)

            # Compute gradients and update
            state = self._training_step(state, batch, sparsity_coeff)

            # Normalize decoder periodically
            if (step + 1) % config.normalize_decoder_frequency == 0:
                state = self._normalize_decoder(state)

            # Resample dead features
            if config.resample_dead_features and (step + 1) % config.resample_frequency == 0:
                state = self._resample_dead_features(state, batch)

            # Log metrics
            if (step + 1) % config.log_frequency == 0:
                avg_loss = state.total_loss / max(state.log_count, 1)
                avg_recon = state.total_recon_loss / max(state.log_count, 1)
                avg_l1 = state.total_l1_loss / max(state.log_count, 1)
                avg_sparsity = state.total_sparsity / max(state.log_count, 1)

                loss_history.append(avg_loss)
                logger.info(
                    f"Step {step + 1}/{config.num_steps}: "
                    f"loss={avg_loss:.4f}, recon={avg_recon:.4f}, "
                    f"l1={avg_l1:.4f}, L0={avg_sparsity:.1f}"
                )

                # Reset accumulators
                state.total_loss = 0.0
                state.total_recon_loss = 0.0
                state.total_l1_loss = 0.0
                state.total_sparsity = 0.0
                state.log_count = 0

            # Checkpoint
            if config.checkpoint_frequency and (step + 1) % config.checkpoint_frequency == 0:
                if on_checkpoint:
                    on_checkpoint(state.weights, step + 1)

        training_time = time.perf_counter() - start_time

        # Count dead features
        eps = regularization_epsilon(b, state.feature_activation_counts)
        dead_mask = state.feature_activation_counts < eps
        b.eval(dead_mask)
        dead_features = int(b.to_scalar(b.sum(b.astype(dead_mask, "float32"))))

        # Final metrics
        final_batch = self._get_batch(activation_iterator)
        final_result = self._sae.encode(final_batch, state.weights)

        return SAETrainingResult(
            weights=state.weights,
            final_loss=final_result.reconstruction_loss + sparsity_coeff * final_result.l1_loss,
            final_reconstruction_loss=final_result.reconstruction_loss,
            final_sparsity=final_result.sparsity,
            dead_features=dead_features,
            training_steps=config.num_steps,
            training_time_seconds=training_time,
            loss_history=loss_history,
        )

    def _initialize_state(self, weights: SAEWeights) -> TrainingState:
        """Initialize training state with Adam moments."""
        b = self._backend
        hidden_dim = self._sae_config.hidden_dim
        latent_dim = self._sae_config.latent_dim

        return TrainingState(
            weights=weights,
            m_enc=b.zeros((hidden_dim, latent_dim)),
            v_enc=b.zeros((hidden_dim, latent_dim)),
            m_dec=b.zeros((latent_dim, hidden_dim)),
            v_dec=b.zeros((latent_dim, hidden_dim)),
            m_b_enc=b.zeros((latent_dim,)),
            v_b_enc=b.zeros((latent_dim,)),
            m_b_dec=b.zeros((hidden_dim,)),
            v_b_dec=b.zeros((hidden_dim,)),
            feature_activation_counts=b.zeros((latent_dim,)),
        )

    def _training_step(
        self,
        state: TrainingState,
        batch: Any,
        sparsity_coeff: float,
    ) -> TrainingState:
        """Perform one training step."""
        b = self._backend
        config = self._training_config
        weights = state.weights

        # Compute forward pass and loss
        result = self._sae.encode(batch, weights)
        loss = result.reconstruction_loss + sparsity_coeff * result.l1_loss

        # Accumulate metrics
        state.total_loss += loss
        state.total_recon_loss += result.reconstruction_loss
        state.total_l1_loss += result.l1_loss
        state.total_sparsity += result.sparsity
        state.log_count += 1

        # Update feature activation counts
        active_count = b.sum(b.astype(result.active_features, "float32"), axis=0)
        b.eval(active_count)
        state.feature_activation_counts = state.feature_activation_counts + active_count
        b.eval(state.feature_activation_counts)

        # Compute gradients manually (autodiff through our encode function)
        # This is a simplified gradient computation - in production you'd use
        # backend.value_and_grad() if available
        grads = self._compute_gradients(batch, weights, result, sparsity_coeff)

        # Learning rate with warmup
        lr = self._get_learning_rate(state.step)

        # Adam update
        state = self._adam_update(state, grads, lr)

        return state

    def _compute_gradients(
        self,
        batch: Any,
        weights: SAEWeights,
        result: SAEEncodingResult,
        sparsity_coeff: float,
    ) -> dict[str, Any]:
        """Compute gradients for SAE weights.

        Uses analytical gradients for the SAE architecture.
        """
        b = self._backend
        batch_size = int(batch.shape[0])
        if batch_size == 0:
            return {
                "W_enc": b.zeros_like(weights.W_enc),
                "b_enc": b.zeros_like(weights.b_enc),
                "W_dec": b.zeros_like(weights.W_dec),
                "b_dec": b.zeros_like(weights.b_dec),
            }

        x = b.astype(batch, "float32")
        W_enc = b.astype(weights.W_enc, "float32")
        b_enc = b.astype(weights.b_enc, "float32")
        W_dec = b.astype(weights.W_dec, "float32")
        b_dec = b.astype(weights.b_dec, "float32")
        z = b.astype(result.sparse_codes, "float32")
        x_hat = b.astype(result.reconstruction, "float32")
        b.eval(x, W_enc, b_enc, W_dec, b_dec, z, x_hat)

        # Reconstruction loss gradient: d/dx_hat (||x - x_hat||^2) = -2(x - x_hat)
        recon_grad = -2.0 * (x - x_hat) / batch_size

        # Gradient through decoder: W_dec, b_dec
        # x_hat = z @ W_dec + b_dec
        # dL/dW_dec = z^T @ dL/dx_hat
        # dL/db_dec = sum(dL/dx_hat)
        grad_W_dec = b.matmul(b.transpose(z), recon_grad)
        grad_b_dec = b.sum(recon_grad, axis=0)
        b.eval(grad_W_dec, grad_b_dec)

        # Gradient through encoder
        # z = ReLU(x_centered @ W_enc + b_enc)
        # dL/dz = dL/dx_hat @ W_dec^T + sparsity_coeff * sign(z)
        grad_z = b.matmul(recon_grad, b.transpose(W_dec))

        # L1 gradient: sign(z) where z > 0
        eps = regularization_epsilon(b, z)
        relu_mask = z > eps
        l1_grad = sparsity_coeff * b.where(relu_mask, b.sign(z), b.zeros_like(z)) / batch_size
        grad_z = grad_z + l1_grad
        b.eval(grad_z)

        # ReLU gradient: pass through where z > 0
        grad_pre_relu = b.where(relu_mask, grad_z, b.zeros_like(grad_z))
        b.eval(grad_pre_relu)

        # dL/dW_enc = x_centered^T @ grad_pre_relu
        x_centered = x - b.reshape(b_dec, (1, -1))
        grad_W_enc = b.matmul(b.transpose(x_centered), grad_pre_relu)
        grad_b_enc = b.sum(grad_pre_relu, axis=0)
        b.eval(grad_W_enc, grad_b_enc)

        # b_dec also affects encoder input
        # dL/db_dec += -sum(grad_pre_relu @ W_enc^T, axis=0)
        grad_b_dec_encoder = -b.sum(b.matmul(grad_pre_relu, b.transpose(W_enc)), axis=0)
        grad_b_dec = grad_b_dec + grad_b_dec_encoder
        b.eval(grad_b_dec)

        return {
            "W_enc": grad_W_enc,
            "b_enc": grad_b_enc,
            "W_dec": grad_W_dec,
            "b_dec": grad_b_dec,
        }

    def _adam_update(
        self,
        state: TrainingState,
        grads: dict[str, Any],
        lr: float,
    ) -> TrainingState:
        """Apply Adam optimizer update."""
        b = self._backend
        beta1, beta2 = 0.9, 0.999
        eps = machine_epsilon(b, state.weights.W_enc)
        t = state.step + 1

        # Bias correction
        bc1 = 1 - beta1**t
        bc2 = 1 - beta2**t

        def adam_step(
            param: Any,
            grad: Any,
            m: Any,
            v: Any,
        ) -> tuple[Any, Any, Any]:
            m_new = beta1 * m + (1 - beta1) * grad
            v_new = beta2 * v + (1 - beta2) * grad * grad
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            param_new = param - lr * m_hat / (b.sqrt(v_hat) + eps)
            b.eval(param_new, m_new, v_new)
            return param_new, m_new, v_new

        # Update encoder
        W_enc, m_enc, v_enc = adam_step(
            b.astype(state.weights.W_enc, "float32"),
            grads["W_enc"],
            state.m_enc,
            state.v_enc,
        )
        b_enc, m_b_enc, v_b_enc = adam_step(
            b.astype(state.weights.b_enc, "float32"),
            grads["b_enc"],
            state.m_b_enc,
            state.v_b_enc,
        )

        # Update decoder
        W_dec, m_dec, v_dec = adam_step(
            b.astype(state.weights.W_dec, "float32"),
            grads["W_dec"],
            state.m_dec,
            state.v_dec,
        )
        b_dec, m_b_dec, v_b_dec = adam_step(
            b.astype(state.weights.b_dec, "float32"),
            grads["b_dec"],
            state.m_b_dec,
            state.v_b_dec,
        )

        # Create new weights
        new_weights = SAEWeights(
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=b_dec,
            config=state.weights.config,
        )

        state.weights = new_weights
        state.m_enc = m_enc
        state.v_enc = v_enc
        state.m_dec = m_dec
        state.v_dec = v_dec
        state.m_b_enc = m_b_enc
        state.v_b_enc = v_b_enc
        state.m_b_dec = m_b_dec
        state.v_b_dec = v_b_dec

        return state

    def _get_learning_rate(self, step: int) -> float:
        """Get learning rate with warmup."""
        config = self._training_config
        if step < config.warmup_steps:
            # Linear warmup
            return config.learning_rate * (step + 1) / config.warmup_steps
        return config.learning_rate

    def _normalize_decoder(self, state: TrainingState) -> TrainingState:
        """Normalize decoder columns to unit geodesic norm."""
        b = self._backend
        W_dec = b.astype(state.weights.W_dec, "float32")

        norms = geodesic_norms(W_dec, b)
        b.eval(norms)

        eps = regularization_epsilon(b, W_dec)
        latent_dim = int(W_dec.shape[0])
        norms_safe = b.maximum(norms, b.full((latent_dim,), eps))
        norms_broadcast = b.reshape(norms_safe, (latent_dim, 1))
        W_dec_normalized = W_dec / norms_broadcast
        b.eval(W_dec_normalized)

        new_weights = SAEWeights(
            W_enc=state.weights.W_enc,
            b_enc=state.weights.b_enc,
            W_dec=W_dec_normalized,
            b_dec=state.weights.b_dec,
            config=state.weights.config,
        )
        state.weights = new_weights
        return state

    def _resample_dead_features(self, state: TrainingState, batch: Any) -> TrainingState:
        """Resample features that haven't activated."""
        b = self._backend
        config = self._training_config

        eps = regularization_epsilon(b, state.feature_activation_counts)
        dead_mask = state.feature_activation_counts < eps
        b.eval(dead_mask)

        dead_count = int(b.to_scalar(b.sum(b.astype(dead_mask, "float32"))))
        if dead_count == 0:
            return state

        logger.info(f"Resampling {dead_count} dead features")

        # Find dead feature indices
        latent_dim = self._sae_config.latent_dim
        dead_indices = []
        for i in range(latent_dim):
            is_dead = bool(b.to_scalar(dead_mask[i]))
            if is_dead:
                dead_indices.append(i)

        # Resample from high-loss samples
        # First, find samples with highest reconstruction loss
        result = self._sae.encode(batch, state.weights)
        recon_diff = batch - result.reconstruction
        per_sample_loss = geodesic_norms(recon_diff, b)
        b.eval(per_sample_loss)

        # Get indices of high-loss samples
        batch_size = int(batch.shape[0])
        sorted_indices = b.argsort(per_sample_loss)
        # Take top dead_count samples (highest loss)
        reversed_idx = b.arange(batch_size - 1, -1, -1)
        high_loss_indices = b.take(sorted_indices, reversed_idx, axis=0)[:dead_count]
        b.eval(high_loss_indices)

        # Update encoder and decoder for dead features
        W_enc = b.astype(state.weights.W_enc, "float32")
        W_dec = b.astype(state.weights.W_dec, "float32")
        b_enc = b.astype(state.weights.b_enc, "float32")

        for i, dead_idx in enumerate(dead_indices):
            if i >= dead_count:
                break
            # Get high-loss sample
            sample_idx = int(b.to_scalar(high_loss_indices[i]))
            sample = batch[sample_idx, :]

            # Normalize sample for decoder direction
            sample_norm = geodesic_norms(b.reshape(sample, (1, -1)), b)
            b.eval(sample_norm)
            norm_val = float(b.to_scalar(sample_norm[0]))
            if norm_val > eps:
                direction = sample / norm_val
            else:
                direction = sample

            # Set decoder row to this direction
            W_dec = self._set_row(W_dec, dead_idx, direction, b)

            # Set encoder column to decoder row transpose * small scale
            scale = 0.1  # Small scale to avoid immediate dominance
            W_enc = self._set_col(W_enc, dead_idx, direction * scale, b)

            # Reset encoder bias to zero
            b_enc = self._set_element(b_enc, dead_idx, 0.0, b)

        b.eval(W_enc, W_dec, b_enc)

        # Reset activation counts for resampled features
        for dead_idx in dead_indices[:dead_count]:
            state.feature_activation_counts = self._set_element(
                state.feature_activation_counts, dead_idx, 0.0, b
            )
        b.eval(state.feature_activation_counts)

        new_weights = SAEWeights(
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=state.weights.b_dec,
            config=state.weights.config,
        )
        state.weights = new_weights
        return state

    def _set_row(self, matrix: Any, row_idx: int, values: Any, backend: Any) -> Any:
        """Set a row in a matrix (backend-agnostic)."""
        # Create mask for the row
        n_rows = int(matrix.shape[0])
        mask = backend.arange(n_rows) == row_idx
        mask = backend.reshape(mask, (n_rows, 1))

        # Broadcast values to matrix shape
        values_row = backend.reshape(values, (1, -1))

        # Use where to set row
        result = backend.where(mask, values_row, matrix)
        backend.eval(result)
        return result

    def _set_col(self, matrix: Any, col_idx: int, values: Any, backend: Any) -> Any:
        """Set a column in a matrix (backend-agnostic)."""
        n_cols = int(matrix.shape[1])
        mask = backend.arange(n_cols) == col_idx
        mask = backend.reshape(mask, (1, n_cols))

        values_col = backend.reshape(values, (-1, 1))
        result = backend.where(mask, values_col, matrix)
        backend.eval(result)
        return result

    def _set_element(self, vector: Any, idx: int, value: float, backend: Any) -> Any:
        """Set an element in a vector (backend-agnostic)."""
        n = int(vector.shape[0])
        mask = backend.arange(n) == idx
        result = backend.where(mask, backend.full((n,), value), vector)
        backend.eval(result)
        return result

    def _get_batch(self, activation_source: Iterator[Any] | Callable[[], Any]) -> Any:
        """Get a batch from the activation source."""
        if callable(activation_source) and not hasattr(activation_source, "__next__"):
            return activation_source()
        try:
            return next(activation_source)  # type: ignore
        except StopIteration:
            raise ValueError("Activation iterator exhausted")


__all__ = [
    "SAETrainingConfig",
    "TrainingState",
    "SAETrainingResult",
    "SAETrainer",
]
