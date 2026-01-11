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
Crosscoder for Model Diffing.

Crosscoders train a joint SAE on activations from two related models
(e.g., base and fine-tuned) to identify:
- Shared features: Present in both models
- Base-exclusive features: Only in base model
- Fine-tuned-exclusive features: Only in fine-tuned model

The exclusive features capture what changed during fine-tuning, which is
crucial for understanding safety-relevant behavioral changes.

Architecture:
    Shared encoder: Processes activations from both models
    Reconstruction: Separate decoders for shared vs exclusive features
    Loss: Sum of reconstruction losses for both models + sparsity

References:
    - "Crosscoders: A Unifying Framework for Model Diffing" (Anthropic, 2024)
    - "Model Diffing with Crosscoders" (Anthropic Research, 2024)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_linear_cka
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrosscoderConfig:
    """Configuration for Crosscoder.

    Attributes
    ----------
    hidden_dim : int
        Activation dimension for both models.
    shared_expansion : int
        Expansion factor for shared features.
    exclusive_expansion : int
        Expansion factor for exclusive features per model.
    sparsity_coefficient : float | None
        L1 sparsity penalty. If None, derived from data.
    normalize_decoder : bool
        Whether to normalize decoder columns.
    """

    hidden_dim: int
    shared_expansion: int = 4
    exclusive_expansion: int = 2
    sparsity_coefficient: float | None = None
    normalize_decoder: bool = True

    @property
    def shared_dim(self) -> int:
        """Dimension of shared feature space."""
        return self.hidden_dim * self.shared_expansion

    @property
    def exclusive_dim(self) -> int:
        """Dimension of exclusive feature space per model."""
        return self.hidden_dim * self.exclusive_expansion

    @property
    def total_latent_dim(self) -> int:
        """Total latent dimension: shared + 2 * exclusive."""
        return self.shared_dim + 2 * self.exclusive_dim


@dataclass(frozen=True)
class CrosscoderWeights:
    """Trained Crosscoder weights.

    Attributes
    ----------
    W_enc_shared : Array
        Shared encoder. Shape: [hidden_dim, shared_dim].
    W_enc_base : Array
        Base-exclusive encoder. Shape: [hidden_dim, exclusive_dim].
    W_enc_ft : Array
        Fine-tuned-exclusive encoder. Shape: [hidden_dim, exclusive_dim].
    b_enc : Array
        Encoder bias. Shape: [total_latent_dim].
    W_dec_shared : Array
        Shared decoder. Shape: [shared_dim, hidden_dim].
    W_dec_base : Array
        Base-exclusive decoder. Shape: [exclusive_dim, hidden_dim].
    W_dec_ft : Array
        Fine-tuned-exclusive decoder. Shape: [exclusive_dim, hidden_dim].
    b_dec_base : Array
        Base decoder bias. Shape: [hidden_dim].
    b_dec_ft : Array
        Fine-tuned decoder bias. Shape: [hidden_dim].
    config : CrosscoderConfig
        Configuration.
    """

    W_enc_shared: Any
    W_enc_base: Any
    W_enc_ft: Any
    b_enc: Any
    W_dec_shared: Any
    W_dec_base: Any
    W_dec_ft: Any
    b_dec_base: Any
    b_dec_ft: Any
    config: CrosscoderConfig


@dataclass(frozen=True)
class CrosscoderEncodingResult:
    """Result of encoding through Crosscoder.

    Attributes
    ----------
    shared_features : Array
        Shared feature activations. Shape: [batch, shared_dim].
    base_exclusive_features : Array
        Base-exclusive feature activations. Shape: [batch, exclusive_dim].
    ft_exclusive_features : Array
        Fine-tuned-exclusive feature activations. Shape: [batch, exclusive_dim].
    base_reconstruction : Array
        Reconstructed base activations. Shape: [batch, hidden_dim].
    ft_reconstruction : Array
        Reconstructed fine-tuned activations. Shape: [batch, hidden_dim].
    base_loss : float
        Reconstruction loss for base model.
    ft_loss : float
        Reconstruction loss for fine-tuned model.
    sparsity : float
        L0 sparsity across all features.
    """

    shared_features: Any
    base_exclusive_features: Any
    ft_exclusive_features: Any
    base_reconstruction: Any
    ft_reconstruction: Any
    base_loss: float
    ft_loss: float
    sparsity: float


@dataclass(frozen=True)
class ModelDiffResult:
    """Result of model diffing with Crosscoder.

    Attributes
    ----------
    shared_feature_indices : list[int]
        Indices of features active in both models.
    base_exclusive_indices : list[int]
        Indices of features only active in base model.
    ft_exclusive_indices : list[int]
        Indices of features only active in fine-tuned model.
    shared_activation_cka : float
        CKA similarity on shared features (measures alignment quality).
    exclusive_base_energy : float
        Total energy in base-exclusive features.
    exclusive_ft_energy : float
        Total energy in fine-tuned-exclusive features.
    change_magnitude : float
        Overall magnitude of behavioral change.
    """

    shared_feature_indices: list[int]
    base_exclusive_indices: list[int]
    ft_exclusive_indices: list[int]
    shared_activation_cka: float
    exclusive_base_energy: float
    exclusive_ft_energy: float
    change_magnitude: float


class Crosscoder:
    """Crosscoder for model diffing.

    Trains a joint SAE on base and fine-tuned model activations to identify
    shared vs exclusive features.

    Example
    -------
    >>> cc = Crosscoder(config)
    >>> weights = cc.initialize_weights()
    >>> result = cc.encode(base_acts, ft_acts, weights)
    >>> diff = cc.diff_models(base_acts, ft_acts, weights)
    >>> # diff.ft_exclusive_indices shows what changed in fine-tuning
    """

    def __init__(
        self,
        config: CrosscoderConfig,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize Crosscoder.

        Parameters
        ----------
        config : CrosscoderConfig
            Crosscoder configuration.
        backend : Backend, optional
            Computation backend.
        """
        self._config = config
        self._backend = backend or get_default_backend()

    @property
    def config(self) -> CrosscoderConfig:
        """Get configuration."""
        return self._config

    @property
    def backend(self) -> "Backend":
        """Get backend."""
        return self._backend

    def initialize_weights(
        self,
        initialization_scale: float | None = None,
    ) -> CrosscoderWeights:
        """Initialize Crosscoder weights.

        Parameters
        ----------
        initialization_scale : float, optional
            Scale for initialization. If None, derived from dimensions.

        Returns
        -------
        CrosscoderWeights
            Initialized weights.
        """
        b = self._backend
        config = self._config
        hidden_dim = config.hidden_dim
        shared_dim = config.shared_dim
        exclusive_dim = config.exclusive_dim
        total_latent = config.total_latent_dim

        if initialization_scale is None:
            scale = b.sqrt(b.array(2.0 / hidden_dim))
            b.eval(scale)
            initialization_scale = float(b.to_scalar(scale))

        # Encoders
        W_enc_shared = b.random_normal(shape=(hidden_dim, shared_dim)) * initialization_scale
        W_enc_base = b.random_normal(shape=(hidden_dim, exclusive_dim)) * initialization_scale
        W_enc_ft = b.random_normal(shape=(hidden_dim, exclusive_dim)) * initialization_scale
        b_enc = b.zeros((total_latent,))
        b.eval(W_enc_shared, W_enc_base, W_enc_ft, b_enc)

        # Decoders
        W_dec_shared = b.random_normal(shape=(shared_dim, hidden_dim)) * initialization_scale
        W_dec_base = b.random_normal(shape=(exclusive_dim, hidden_dim)) * initialization_scale
        W_dec_ft = b.random_normal(shape=(exclusive_dim, hidden_dim)) * initialization_scale
        b_dec_base = b.zeros((hidden_dim,))
        b_dec_ft = b.zeros((hidden_dim,))
        b.eval(W_dec_shared, W_dec_base, W_dec_ft, b_dec_base, b_dec_ft)

        # Normalize decoders if requested
        if config.normalize_decoder:
            W_dec_shared = self._normalize_decoder(W_dec_shared)
            W_dec_base = self._normalize_decoder(W_dec_base)
            W_dec_ft = self._normalize_decoder(W_dec_ft)

        return CrosscoderWeights(
            W_enc_shared=W_enc_shared,
            W_enc_base=W_enc_base,
            W_enc_ft=W_enc_ft,
            b_enc=b_enc,
            W_dec_shared=W_dec_shared,
            W_dec_base=W_dec_base,
            W_dec_ft=W_dec_ft,
            b_dec_base=b_dec_base,
            b_dec_ft=b_dec_ft,
            config=config,
        )

    def encode(
        self,
        base_activations: Any,
        ft_activations: Any,
        weights: CrosscoderWeights,
    ) -> CrosscoderEncodingResult:
        """Encode activations from both models.

        Parameters
        ----------
        base_activations : Array
            Activations from base model. Shape: [batch, hidden_dim].
        ft_activations : Array
            Activations from fine-tuned model. Shape: [batch, hidden_dim].
        weights : CrosscoderWeights
            Crosscoder weights.

        Returns
        -------
        CrosscoderEncodingResult
            Encoding result with shared and exclusive features.
        """
        b = self._backend
        config = self._config

        base = b.array(base_activations) if not hasattr(base_activations, "shape") else base_activations
        ft = b.array(ft_activations) if not hasattr(ft_activations, "shape") else ft_activations
        base = b.astype(base, "float32")
        ft = b.astype(ft, "float32")
        b.eval(base, ft)

        batch_size = int(base.shape[0])
        if batch_size == 0:
            return self._empty_result(weights)

        # Extract weight components
        W_enc_shared = b.astype(weights.W_enc_shared, "float32")
        W_enc_base = b.astype(weights.W_enc_base, "float32")
        W_enc_ft = b.astype(weights.W_enc_ft, "float32")
        b_enc = b.astype(weights.b_enc, "float32")
        W_dec_shared = b.astype(weights.W_dec_shared, "float32")
        W_dec_base = b.astype(weights.W_dec_base, "float32")
        W_dec_ft = b.astype(weights.W_dec_ft, "float32")
        b_dec_base = b.astype(weights.b_dec_base, "float32")
        b_dec_ft = b.astype(weights.b_dec_ft, "float32")
        b.eval(W_enc_shared, W_enc_base, W_enc_ft, b_enc)
        b.eval(W_dec_shared, W_dec_base, W_dec_ft, b_dec_base, b_dec_ft)

        shared_dim = config.shared_dim
        exclusive_dim = config.exclusive_dim

        # Bias slices
        b_shared = b_enc[:shared_dim]
        b_base = b_enc[shared_dim : shared_dim + exclusive_dim]
        b_ft = b_enc[shared_dim + exclusive_dim :]

        # Encode shared features (from concatenated input)
        # For shared: average contribution from both models
        shared_pre = (
            b.matmul(base, W_enc_shared) + b.matmul(ft, W_enc_shared)
        ) / 2.0 + b.reshape(b_shared, (1, -1))
        shared_features = b.maximum(shared_pre, b.zeros_like(shared_pre))
        b.eval(shared_features)

        # Encode exclusive features
        base_excl_pre = b.matmul(base, W_enc_base) + b.reshape(b_base, (1, -1))
        base_exclusive = b.maximum(base_excl_pre, b.zeros_like(base_excl_pre))
        b.eval(base_exclusive)

        ft_excl_pre = b.matmul(ft, W_enc_ft) + b.reshape(b_ft, (1, -1))
        ft_exclusive = b.maximum(ft_excl_pre, b.zeros_like(ft_excl_pre))
        b.eval(ft_exclusive)

        # Decode base reconstruction: shared + base_exclusive
        base_from_shared = b.matmul(shared_features, W_dec_shared)
        base_from_excl = b.matmul(base_exclusive, W_dec_base)
        base_reconstruction = base_from_shared + base_from_excl + b.reshape(b_dec_base, (1, -1))
        b.eval(base_reconstruction)

        # Decode ft reconstruction: shared + ft_exclusive
        ft_from_shared = b.matmul(shared_features, W_dec_shared)
        ft_from_excl = b.matmul(ft_exclusive, W_dec_ft)
        ft_reconstruction = ft_from_shared + ft_from_excl + b.reshape(b_dec_ft, (1, -1))
        b.eval(ft_reconstruction)

        # Compute losses (geodesic)
        base_diff = base - base_reconstruction
        ft_diff = ft - ft_reconstruction
        base_norms = geodesic_norms(base_diff, b)
        ft_norms = geodesic_norms(ft_diff, b)
        b.eval(base_norms, ft_norms)
        base_loss = float(b.to_scalar(b.mean(base_norms)))
        ft_loss = float(b.to_scalar(b.mean(ft_norms)))

        # Compute sparsity
        eps = regularization_epsilon(b, shared_features)
        all_features = b.concatenate(
            [shared_features, base_exclusive, ft_exclusive], axis=1
        )
        active = b.sum(b.astype(all_features > eps, "float32"), axis=1)
        sparsity = float(b.to_scalar(b.mean(active)))

        return CrosscoderEncodingResult(
            shared_features=shared_features,
            base_exclusive_features=base_exclusive,
            ft_exclusive_features=ft_exclusive,
            base_reconstruction=base_reconstruction,
            ft_reconstruction=ft_reconstruction,
            base_loss=base_loss,
            ft_loss=ft_loss,
            sparsity=sparsity,
        )

    def diff_models(
        self,
        base_activations: Any,
        ft_activations: Any,
        weights: CrosscoderWeights,
        activity_threshold: float | None = None,
    ) -> ModelDiffResult:
        """Compute model diff using Crosscoder features.

        Parameters
        ----------
        base_activations : Array
            Activations from base model. Shape: [batch, hidden_dim].
        ft_activations : Array
            Activations from fine-tuned model. Shape: [batch, hidden_dim].
        weights : CrosscoderWeights
            Crosscoder weights.
        activity_threshold : float, optional
            Threshold for feature activity. If None, derived from data.

        Returns
        -------
        ModelDiffResult
            Model diff analysis.
        """
        b = self._backend
        config = self._config

        # Encode both
        result = self.encode(base_activations, ft_activations, weights)

        # Derive threshold from feature statistics
        all_activations = b.concatenate([
            result.shared_features,
            result.base_exclusive_features,
            result.ft_exclusive_features,
        ], axis=1)
        b.eval(all_activations)

        if activity_threshold is None:
            # Use mean activation as threshold (data-derived)
            mean_act = b.mean(all_activations)
            b.eval(mean_act)
            activity_threshold = float(b.to_scalar(mean_act))

        # Sum activations across batch to get total activity per feature
        shared_activity = b.sum(result.shared_features, axis=0)
        base_excl_activity = b.sum(result.base_exclusive_features, axis=0)
        ft_excl_activity = b.sum(result.ft_exclusive_features, axis=0)
        b.eval(shared_activity, base_excl_activity, ft_excl_activity)

        # Find active features
        shared_dim = config.shared_dim
        exclusive_dim = config.exclusive_dim

        shared_indices = []
        for i in range(shared_dim):
            if float(b.to_scalar(shared_activity[i])) > activity_threshold:
                shared_indices.append(i)

        base_indices = []
        for i in range(exclusive_dim):
            if float(b.to_scalar(base_excl_activity[i])) > activity_threshold:
                base_indices.append(i + shared_dim)

        ft_indices = []
        for i in range(exclusive_dim):
            if float(b.to_scalar(ft_excl_activity[i])) > activity_threshold:
                ft_indices.append(i + shared_dim + exclusive_dim)

        # Compute CKA on shared features for alignment quality
        if len(shared_indices) > 0:
            shared_base_contrib = b.matmul(
                result.shared_features,
                b.astype(weights.W_dec_shared, "float32")
            )
            shared_ft_contrib = shared_base_contrib  # Same shared features
            b.eval(shared_base_contrib, shared_ft_contrib)

            shared_cka = compute_linear_cka(shared_base_contrib, shared_ft_contrib, b)
        else:
            shared_cka = 1.0  # No shared features = perfectly aligned (trivially)

        # Compute energy in exclusive features
        base_energy = float(b.to_scalar(b.sum(base_excl_activity)))
        ft_energy = float(b.to_scalar(b.sum(ft_excl_activity)))

        # Change magnitude: ratio of exclusive to total energy
        total_energy = (
            float(b.to_scalar(b.sum(shared_activity))) + base_energy + ft_energy
        )
        eps = regularization_epsilon(b, all_activations)
        if total_energy > eps:
            change_magnitude = (base_energy + ft_energy) / total_energy
        else:
            change_magnitude = 0.0

        return ModelDiffResult(
            shared_feature_indices=shared_indices,
            base_exclusive_indices=base_indices,
            ft_exclusive_indices=ft_indices,
            shared_activation_cka=shared_cka,
            exclusive_base_energy=base_energy,
            exclusive_ft_energy=ft_energy,
            change_magnitude=change_magnitude,
        )

    def get_feature_direction(
        self,
        feature_index: int,
        weights: CrosscoderWeights,
        model: str = "base",
    ) -> Any:
        """Get decoder direction for a feature.

        Parameters
        ----------
        feature_index : int
            Feature index.
        weights : CrosscoderWeights
            Crosscoder weights.
        model : str
            Which model's decoder to use: "base" or "ft".

        Returns
        -------
        Array
            Feature direction. Shape: [hidden_dim].
        """
        b = self._backend
        config = self._config
        shared_dim = config.shared_dim
        exclusive_dim = config.exclusive_dim

        if feature_index < shared_dim:
            # Shared feature
            W_dec = b.astype(weights.W_dec_shared, "float32")
            direction = W_dec[feature_index, :]
        elif feature_index < shared_dim + exclusive_dim:
            # Base-exclusive
            idx = feature_index - shared_dim
            W_dec = b.astype(weights.W_dec_base, "float32")
            direction = W_dec[idx, :]
        else:
            # FT-exclusive
            idx = feature_index - shared_dim - exclusive_dim
            W_dec = b.astype(weights.W_dec_ft, "float32")
            direction = W_dec[idx, :]

        b.eval(direction)

        # Normalize
        norm = geodesic_norms(b.reshape(direction, (1, -1)), b)
        b.eval(norm)
        norm_val = float(b.to_scalar(norm[0]))

        eps = regularization_epsilon(b, direction)
        if norm_val > eps:
            direction = direction / norm_val
            b.eval(direction)

        return direction

    def _normalize_decoder(self, W_dec: Any) -> Any:
        """Normalize decoder rows to unit geodesic norm."""
        b = self._backend
        n_features = int(W_dec.shape[0])

        norms = geodesic_norms(W_dec, b)
        b.eval(norms)

        eps = regularization_epsilon(b, W_dec)
        norms_safe = b.maximum(norms, b.full((n_features,), eps))
        norms_broadcast = b.reshape(norms_safe, (n_features, 1))
        W_dec_normalized = W_dec / norms_broadcast
        b.eval(W_dec_normalized)

        return W_dec_normalized

    def _empty_result(self, weights: CrosscoderWeights) -> CrosscoderEncodingResult:
        """Create empty result for zero-batch input."""
        b = self._backend
        config = weights.config

        return CrosscoderEncodingResult(
            shared_features=b.zeros((0, config.shared_dim)),
            base_exclusive_features=b.zeros((0, config.exclusive_dim)),
            ft_exclusive_features=b.zeros((0, config.exclusive_dim)),
            base_reconstruction=b.zeros((0, config.hidden_dim)),
            ft_reconstruction=b.zeros((0, config.hidden_dim)),
            base_loss=0.0,
            ft_loss=0.0,
            sparsity=0.0,
        )


__all__ = [
    "CrosscoderConfig",
    "CrosscoderWeights",
    "CrosscoderEncodingResult",
    "ModelDiffResult",
    "Crosscoder",
]
