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
Sparse Autoencoder (SAE) for Mechanistic Interpretability.

Decomposes polysemantic neural network activations into sparse, monosemantic
features. This is the foundation for understanding what individual neurons
and circuits compute.

Architecture:
    encode: x -> ReLU(W_enc @ (x - b_dec) + b_enc) -> sparse z
    decode: z -> W_dec @ z + b_dec -> reconstruction
    loss:   geodesic_mse(x, x_hat) + lambda * L1(z)

Key principles:
    - Geodesic reconstruction loss (not Euclidean MSE)
    - No hardcoded sparsity thresholds (derived from activation variance)
    - Backend-agnostic (MLX/JAX/CUDA via Backend protocol)

References:
    - "Towards Monosemanticity" (Anthropic, 2023)
    - "Scaling Monosemanticity" (Anthropic, 2024)
    - "Sparse Autoencoders Find Highly Interpretable Features" (Cunningham et al., 2023)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SAEConfig:
    """Configuration for Sparse Autoencoder.

    Attributes
    ----------
    hidden_dim : int
        Model activation dimension (d_model).
    expansion_factor : int
        Expansion ratio for latent dimension. latent_dim = hidden_dim * expansion_factor.
        Typical values: 4x for small models, 8-32x for larger models.
    sparsity_coefficient : float | None
        L1 sparsity penalty coefficient. If None, derived from activation variance.
        Higher values = sparser features.
    normalize_decoder : bool
        Whether to normalize decoder columns to unit norm after each update.
        Standard practice in SAE training.
    tied_weights : bool
        Whether encoder weights are tied to decoder transpose.
        Reduces parameters but may limit expressiveness.
    """

    hidden_dim: int
    expansion_factor: int = 8
    sparsity_coefficient: float | None = None
    normalize_decoder: bool = True
    tied_weights: bool = False

    @property
    def latent_dim(self) -> int:
        """Dimension of sparse latent space."""
        return self.hidden_dim * self.expansion_factor


@dataclass(frozen=True)
class SAEWeights:
    """Trained SAE weights.

    Attributes
    ----------
    W_enc : Array
        Encoder weights. Shape: [hidden_dim, latent_dim].
    b_enc : Array
        Encoder bias. Shape: [latent_dim].
    W_dec : Array
        Decoder weights. Shape: [latent_dim, hidden_dim].
    b_dec : Array
        Decoder bias (pre-encoder centering). Shape: [hidden_dim].
    config : SAEConfig
        Configuration used to create these weights.
    """

    W_enc: Any
    b_enc: Any
    W_dec: Any
    b_dec: Any
    config: SAEConfig


@dataclass(frozen=True)
class SAEEncodingResult:
    """Result of encoding activations through SAE.

    Attributes
    ----------
    sparse_codes : Array
        Sparse latent codes. Shape: [batch, latent_dim].
    reconstruction : Array
        Reconstructed activations. Shape: [batch, hidden_dim].
    reconstruction_loss : float
        Geodesic reconstruction loss (mean over batch).
    sparsity : float
        L0 sparsity - average number of active features per sample.
    l1_loss : float
        L1 sparsity loss (mean over batch).
    active_features : Array
        Boolean mask of active features. Shape: [batch, latent_dim].
    feature_activations : Array
        Per-feature activation magnitudes (summed over batch). Shape: [latent_dim].
    """

    sparse_codes: Any
    reconstruction: Any
    reconstruction_loss: float
    sparsity: float
    l1_loss: float
    active_features: Any
    feature_activations: Any


@dataclass(frozen=True)
class TopKFeature:
    """A top-k activated feature.

    Attributes
    ----------
    index : int
        Feature index in latent space.
    activation : float
        Activation magnitude.
    """

    index: int
    activation: float


@dataclass(frozen=True)
class FeatureAnalysis:
    """Analysis of feature activations for a batch.

    Attributes
    ----------
    top_k_features : list[TopKFeature]
        Top-k most active features across batch.
    active_feature_count : int
        Number of features with non-zero activation.
    total_features : int
        Total number of features in SAE.
    activation_histogram : Any
        Histogram of feature activations. Shape: [num_bins].
    """

    top_k_features: list[TopKFeature]
    active_feature_count: int
    total_features: int
    activation_histogram: Any


class SparseAutoencoder:
    """Sparse Autoencoder for mechanistic interpretability.

    Learns sparse, monosemantic features from polysemantic neural activations.
    Uses geodesic reconstruction loss to respect manifold geometry.

    Example
    -------
    >>> sae = SparseAutoencoder(config)
    >>> weights = sae.initialize_weights()
    >>> result = sae.encode(activations, weights)
    >>> # result.sparse_codes contains monosemantic features
    >>> # result.reconstruction_loss measures fidelity
    """

    def __init__(self, config: SAEConfig, backend: "Backend | None" = None) -> None:
        """Initialize SAE.

        Parameters
        ----------
        config : SAEConfig
            SAE configuration.
        backend : Backend, optional
            Computation backend. If None, uses default.
        """
        self._config = config
        self._backend = backend or get_default_backend()

    @property
    def config(self) -> SAEConfig:
        """Get SAE configuration."""
        return self._config

    @property
    def backend(self) -> "Backend":
        """Get computation backend."""
        return self._backend

    def initialize_weights(
        self,
        initialization_scale: float | None = None,
    ) -> SAEWeights:
        """Initialize SAE weights.

        Uses Kaiming initialization scaled by derived factor.

        Parameters
        ----------
        initialization_scale : float, optional
            Scale for weight initialization. If None, derived from dimensions.

        Returns
        -------
        SAEWeights
            Initialized weights.
        """
        b = self._backend
        config = self._config
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim

        # Derive initialization scale from dimensions (no arbitrary constants)
        if initialization_scale is None:
            # Kaiming-style: sqrt(2 / fan_in) for ReLU
            initialization_scale = b.sqrt(b.array(2.0 / hidden_dim))
            b.eval(initialization_scale)
            initialization_scale = float(b.to_scalar(initialization_scale))

        # Initialize encoder weights (standard normal scaled by init scale)
        W_enc = b.random_normal(shape=(hidden_dim, latent_dim)) * initialization_scale
        b.eval(W_enc)

        # Initialize decoder weights
        if config.tied_weights:
            W_dec = b.transpose(W_enc)
        else:
            W_dec = b.random_normal(shape=(latent_dim, hidden_dim)) * initialization_scale
        b.eval(W_dec)

        # Normalize decoder columns if requested
        if config.normalize_decoder:
            W_dec = self._normalize_decoder(W_dec)

        # Initialize biases to zero
        b_enc = b.zeros((latent_dim,))
        b_dec = b.zeros((hidden_dim,))
        b.eval(b_enc, b_dec)

        return SAEWeights(
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=b_dec,
            config=config,
        )

    def encode(
        self,
        activations: Any,
        weights: SAEWeights,
        sparsity_coefficient: float | None = None,
    ) -> SAEEncodingResult:
        """Encode activations through SAE.

        Parameters
        ----------
        activations : Array
            Input activations. Shape: [batch, hidden_dim].
        weights : SAEWeights
            Trained SAE weights.
        sparsity_coefficient : float, optional
            Override sparsity coefficient for L1 loss computation.

        Returns
        -------
        SAEEncodingResult
            Encoding result with sparse codes and metrics.
        """
        b = self._backend
        activations = b.array(activations) if not hasattr(activations, "shape") else activations
        b.eval(activations)

        batch_size = int(activations.shape[0])
        if batch_size == 0:
            return self._empty_result(weights)

        # Cast to float32 for numerical stability
        x = b.astype(activations, "float32")
        W_enc = b.astype(weights.W_enc, "float32")
        b_enc = b.astype(weights.b_enc, "float32")
        W_dec = b.astype(weights.W_dec, "float32")
        b_dec = b.astype(weights.b_dec, "float32")
        b.eval(x, W_enc, b_enc, W_dec, b_dec)

        # Encode: z = ReLU(W_enc @ (x - b_dec) + b_enc)
        # Centering by b_dec improves reconstruction
        x_centered = x - b.reshape(b_dec, (1, -1))
        pre_activation = b.matmul(x_centered, W_enc) + b.reshape(b_enc, (1, -1))
        sparse_codes = b.maximum(pre_activation, b.zeros_like(pre_activation))
        b.eval(sparse_codes)

        # Decode: x_hat = W_dec @ z + b_dec
        reconstruction = b.matmul(sparse_codes, W_dec) + b.reshape(b_dec, (1, -1))
        b.eval(reconstruction)

        # Geodesic reconstruction loss (NOT Euclidean MSE)
        diff = x - reconstruction
        norms = geodesic_norms(diff, b)
        b.eval(norms)
        reconstruction_loss = float(b.to_scalar(b.mean(norms)))

        # L1 sparsity loss
        l1_per_sample = b.sum(b.abs(sparse_codes), axis=1)
        l1_loss = float(b.to_scalar(b.mean(l1_per_sample)))

        # Sparsity statistics
        eps = regularization_epsilon(b, sparse_codes)
        active_mask = sparse_codes > eps
        b.eval(active_mask)

        # L0 sparsity: average number of active features
        active_count_per_sample = b.sum(b.astype(active_mask, "float32"), axis=1)
        sparsity = float(b.to_scalar(b.mean(active_count_per_sample)))

        # Per-feature activation magnitudes
        feature_activations = b.sum(sparse_codes, axis=0)
        b.eval(feature_activations)

        return SAEEncodingResult(
            sparse_codes=sparse_codes,
            reconstruction=reconstruction,
            reconstruction_loss=reconstruction_loss,
            sparsity=sparsity,
            l1_loss=l1_loss,
            active_features=active_mask,
            feature_activations=feature_activations,
        )

    def decode(self, sparse_codes: Any, weights: SAEWeights) -> Any:
        """Decode sparse codes back to activation space.

        Parameters
        ----------
        sparse_codes : Array
            Sparse latent codes. Shape: [batch, latent_dim].
        weights : SAEWeights
            Trained SAE weights.

        Returns
        -------
        Array
            Reconstructed activations. Shape: [batch, hidden_dim].
        """
        b = self._backend
        codes = b.array(sparse_codes) if not hasattr(sparse_codes, "shape") else sparse_codes
        codes = b.astype(codes, "float32")
        W_dec = b.astype(weights.W_dec, "float32")
        b_dec = b.astype(weights.b_dec, "float32")
        b.eval(codes, W_dec, b_dec)

        reconstruction = b.matmul(codes, W_dec) + b.reshape(b_dec, (1, -1))
        b.eval(reconstruction)
        return reconstruction

    def compute_loss(
        self,
        activations: Any,
        weights: SAEWeights,
        sparsity_coefficient: float | None = None,
    ) -> tuple[float, SAEEncodingResult]:
        """Compute total SAE loss.

        Parameters
        ----------
        activations : Array
            Input activations. Shape: [batch, hidden_dim].
        weights : SAEWeights
            SAE weights.
        sparsity_coefficient : float, optional
            L1 sparsity coefficient. If None, uses config or derives from data.

        Returns
        -------
        tuple[float, SAEEncodingResult]
            Total loss and encoding result.
        """
        result = self.encode(activations, weights, sparsity_coefficient)

        # Determine sparsity coefficient
        coeff = sparsity_coefficient or self._config.sparsity_coefficient
        if coeff is None:
            # Derive from activation variance (no arbitrary constants)
            b = self._backend
            act = b.array(activations) if not hasattr(activations, "shape") else activations
            var = b.var(act)
            b.eval(var)
            var_val = float(b.to_scalar(var))
            # Scale inversely with variance - higher variance needs less sparsity pressure
            eps = regularization_epsilon(b, act)
            coeff = 1.0 / max(var_val, eps)

        total_loss = result.reconstruction_loss + coeff * result.l1_loss
        return total_loss, result

    def analyze_features(
        self,
        activations: Any,
        weights: SAEWeights,
        top_k: int = 10,
        num_bins: int = 50,
    ) -> FeatureAnalysis:
        """Analyze which features activate for given inputs.

        Parameters
        ----------
        activations : Array
            Input activations. Shape: [batch, hidden_dim].
        weights : SAEWeights
            Trained SAE weights.
        top_k : int
            Number of top features to return.
        num_bins : int
            Number of histogram bins for activation distribution.

        Returns
        -------
        FeatureAnalysis
            Analysis of feature activations.
        """
        b = self._backend
        result = self.encode(activations, weights)

        # Get per-feature activation sums
        feature_acts = result.feature_activations
        latent_dim = int(feature_acts.shape[0])

        # Find top-k features
        sorted_indices = b.argsort(feature_acts)
        b.eval(sorted_indices)
        # Reverse to get descending order
        reversed_indices = b.arange(latent_dim - 1, -1, -1)
        top_indices = b.take(sorted_indices, reversed_indices, axis=0)[:top_k]
        b.eval(top_indices)

        top_k_features = []
        for i in range(min(top_k, latent_dim)):
            idx = int(b.to_scalar(top_indices[i]))
            act = float(b.to_scalar(feature_acts[idx]))
            top_k_features.append(TopKFeature(index=idx, activation=act))

        # Count active features
        eps = regularization_epsilon(b, feature_acts)
        active_mask = feature_acts > eps
        active_count = int(b.to_scalar(b.sum(b.astype(active_mask, "float32"))))

        # Build activation histogram
        max_act = b.max(feature_acts)
        b.eval(max_act)
        max_act_val = float(b.to_scalar(max_act))

        if max_act_val > eps:
            bin_edges = b.linspace(0.0, max_act_val, num_bins + 1)
            histogram = b.zeros((num_bins,))
            # Simple histogram computation
            for bin_idx in range(num_bins):
                low = b.take(bin_edges, b.array(bin_idx), axis=0)
                high = b.take(bin_edges, b.array(bin_idx + 1), axis=0)
                in_bin = (feature_acts >= low) & (feature_acts < high)
                histogram = b.where(
                    b.arange(num_bins) == bin_idx,
                    b.sum(b.astype(in_bin, "float32")),
                    histogram,
                )
            b.eval(histogram)
        else:
            histogram = b.zeros((num_bins,))
            b.eval(histogram)

        return FeatureAnalysis(
            top_k_features=top_k_features,
            active_feature_count=active_count,
            total_features=latent_dim,
            activation_histogram=histogram,
        )

    def get_feature_direction(self, feature_index: int, weights: SAEWeights) -> Any:
        """Get the decoder direction for a specific feature.

        This represents the direction in activation space that the feature
        encodes. Can be used for feature steering.

        Parameters
        ----------
        feature_index : int
            Index of feature in latent space.
        weights : SAEWeights
            Trained SAE weights.

        Returns
        -------
        Array
            Feature direction. Shape: [hidden_dim].
        """
        b = self._backend
        W_dec = b.astype(weights.W_dec, "float32")
        b.eval(W_dec)

        # Extract row from decoder (latent_dim, hidden_dim)
        direction = W_dec[feature_index, :]
        b.eval(direction)

        # Normalize to unit length
        norm = geodesic_norms(b.reshape(direction, (1, -1)), b)
        b.eval(norm)
        norm_val = float(b.to_scalar(norm[0]))

        eps = regularization_epsilon(b, direction)
        if norm_val > eps:
            direction = direction / norm_val
            b.eval(direction)

        return direction

    def _normalize_decoder(self, W_dec: Any) -> Any:
        """Normalize decoder columns to unit geodesic norm."""
        b = self._backend
        latent_dim = int(W_dec.shape[0])

        # Compute geodesic norms for each row (feature direction)
        norms = geodesic_norms(W_dec, b)
        b.eval(norms)

        # Normalize each row
        eps = regularization_epsilon(b, W_dec)
        norms_safe = b.maximum(norms, b.full((latent_dim,), eps))
        norms_broadcast = b.reshape(norms_safe, (latent_dim, 1))
        W_dec_normalized = W_dec / norms_broadcast
        b.eval(W_dec_normalized)

        return W_dec_normalized

    def _empty_result(self, weights: SAEWeights) -> SAEEncodingResult:
        """Create empty result for zero-batch input."""
        b = self._backend
        latent_dim = weights.config.latent_dim

        return SAEEncodingResult(
            sparse_codes=b.zeros((0, latent_dim)),
            reconstruction=b.zeros((0, weights.config.hidden_dim)),
            reconstruction_loss=0.0,
            sparsity=0.0,
            l1_loss=0.0,
            active_features=b.zeros((0, latent_dim), dtype="bool"),
            feature_activations=b.zeros((latent_dim,)),
        )


def derive_sparsity_coefficient(
    activations: Any,
    backend: "Backend | None" = None,
) -> float:
    """Derive appropriate sparsity coefficient from activation statistics.

    Uses activation variance to determine sparsity pressure. Higher variance
    activations need less sparsity pressure to achieve same L0.

    Parameters
    ----------
    activations : Array
        Sample activations. Shape: [n_samples, hidden_dim].
    backend : Backend, optional
        Computation backend.

    Returns
    -------
    float
        Derived sparsity coefficient.
    """
    b = backend or get_default_backend()
    acts = b.array(activations) if not hasattr(activations, "shape") else activations
    b.eval(acts)

    if int(acts.shape[0]) == 0:
        return 1.0

    # Compute activation variance
    var = b.var(acts)
    b.eval(var)
    var_val = float(b.to_scalar(var))

    eps = regularization_epsilon(b, acts)
    if var_val < eps:
        return 1.0

    # Inverse relationship: high variance = low coefficient
    # This ensures consistent sparsity across different activation scales
    return 1.0 / var_val


__all__ = [
    "SAEConfig",
    "SAEWeights",
    "SAEEncodingResult",
    "TopKFeature",
    "FeatureAnalysis",
    "SparseAutoencoder",
    "derive_sparsity_coefficient",
]
