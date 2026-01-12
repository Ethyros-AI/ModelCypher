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
Transcoder for Cross-Layer MLP Replacement.

Transcoders are like SAEs but trained to predict MLP output from MLP input.
This enables replacing the MLP computation with an interpretable sparse
representation, allowing circuit tracing through the network.

Architecture:
    encode: mlp_input -> ReLU(W_enc @ mlp_input + b_enc) -> sparse features
    decode: features -> W_dec @ features + b_dec -> mlp_output_approximation
    loss:   geodesic_mse(mlp_output, decoded) + lambda * L1(features)

Key insight: Unlike SAEs which reconstruct the same activation, transcoders
transform from MLP input space to MLP output space. This captures what
computation the MLP performs in interpretable terms.

References:
    - "Transcoders Find Interpretable LLM Feature Circuits" (Anthropic, 2024)
    - "Scaling Monosemanticity" (Anthropic, 2024)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    regularization_epsilon,
    ulp_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscoderConfig:
    """Configuration for Transcoder.

    Attributes
    ----------
    input_dim : int
        MLP input dimension (usually hidden_dim).
    output_dim : int
        MLP output dimension (usually hidden_dim).
    expansion_factor : int
        Expansion ratio for latent dimension.
    sparsity_coefficient : float | None
        L1 sparsity penalty. If None, derived from data.
    normalize_decoder : bool
        Whether to normalize decoder columns.
    """

    input_dim: int
    output_dim: int
    expansion_factor: int = 8
    sparsity_coefficient: float | None = None
    normalize_decoder: bool = True

    @property
    def latent_dim(self) -> int:
        """Dimension of sparse latent space."""
        return self.input_dim * self.expansion_factor


@dataclass(frozen=True)
class TranscoderWeights:
    """Trained Transcoder weights.

    Attributes
    ----------
    W_enc : Array
        Encoder weights. Shape: [input_dim, latent_dim].
    b_enc : Array
        Encoder bias. Shape: [latent_dim].
    W_dec : Array
        Decoder weights. Shape: [latent_dim, output_dim].
    b_dec : Array
        Decoder bias. Shape: [output_dim].
    config : TranscoderConfig
        Configuration used to create these weights.
    """

    W_enc: Any
    b_enc: Any
    W_dec: Any
    b_dec: Any
    config: TranscoderConfig


@dataclass(frozen=True)
class TranscoderResult:
    """Result of transcoding MLP input to output.

    Attributes
    ----------
    sparse_features : Array
        Sparse feature activations. Shape: [batch, latent_dim].
    predicted_output : Array
        Predicted MLP output. Shape: [batch, output_dim].
    reconstruction_loss : float
        Geodesic loss between predicted and actual MLP output.
    sparsity : float
        L0 sparsity - average active features per sample.
    l1_loss : float
        L1 sparsity loss.
    active_features : Array
        Boolean mask of active features. Shape: [batch, latent_dim].
    """

    sparse_features: Any
    predicted_output: Any
    reconstruction_loss: float
    sparsity: float
    l1_loss: float
    active_features: Any


@dataclass(frozen=True)
class FeatureContribution:
    """Contribution of a feature to the transcoding.

    Attributes
    ----------
    feature_index : int
        Index of the feature.
    activation : float
        Feature activation magnitude.
    contribution_to_output : Array
        Direction added to output. Shape: [output_dim].
    contribution_magnitude : float
        Geodesic magnitude of contribution.
    """

    feature_index: int
    activation: float
    contribution_to_output: Any
    contribution_magnitude: float


class Transcoder:
    """Transcoder for interpretable MLP replacement.

    Learns sparse features that explain what computation the MLP performs.
    Can be used to trace circuits through the network by examining which
    features activate for different inputs.

    Example
    -------
    >>> tc = Transcoder(config)
    >>> weights = tc.initialize_weights()
    >>> result = tc.transcode(mlp_input, mlp_output, weights)
    >>> # result.sparse_features shows what MLP computes
    """

    def __init__(
        self,
        config: TranscoderConfig,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize Transcoder.

        Parameters
        ----------
        config : TranscoderConfig
            Transcoder configuration.
        backend : Backend, optional
            Computation backend.
        """
        self._config = config
        self._backend = backend or get_default_backend()

    @property
    def config(self) -> TranscoderConfig:
        """Get configuration."""
        return self._config

    @property
    def backend(self) -> "Backend":
        """Get backend."""
        return self._backend

    def initialize_weights(
        self,
        initialization_scale: float | None = None,
    ) -> TranscoderWeights:
        """Initialize Transcoder weights.

        Parameters
        ----------
        initialization_scale : float, optional
            Scale for weight initialization. If None, derived from dimensions.

        Returns
        -------
        TranscoderWeights
            Initialized weights.
        """
        b = self._backend
        config = self._config
        input_dim = config.input_dim
        output_dim = config.output_dim
        latent_dim = config.latent_dim

        if initialization_scale is None:
            # Kaiming-style initialization
            scale = b.sqrt(b.array(2.0 / input_dim))
            b.eval(scale)
            initialization_scale = float(b.to_scalar(scale))

        # Encoder: [input_dim, latent_dim]
        W_enc = b.random_normal(shape=(input_dim, latent_dim)) * initialization_scale
        b.eval(W_enc)

        # Decoder: [latent_dim, output_dim]
        W_dec = b.random_normal(shape=(latent_dim, output_dim)) * initialization_scale
        b.eval(W_dec)

        if config.normalize_decoder:
            W_dec = self._normalize_decoder(W_dec)

        b_enc = b.zeros((latent_dim,))
        b_dec = b.zeros((output_dim,))
        b.eval(b_enc, b_dec)

        return TranscoderWeights(
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=b_dec,
            config=config,
        )

    def transcode(
        self,
        mlp_input: Any,
        mlp_output: Any,
        weights: TranscoderWeights,
        sparsity_coefficient: float | None = None,
    ) -> TranscoderResult:
        """Transcode MLP input to output via sparse features.

        Parameters
        ----------
        mlp_input : Array
            MLP layer input. Shape: [batch, input_dim].
        mlp_output : Array
            Actual MLP layer output (for loss computation).
            Shape: [batch, output_dim].
        weights : TranscoderWeights
            Transcoder weights.
        sparsity_coefficient : float, optional
            Override sparsity coefficient.

        Returns
        -------
        TranscoderResult
            Transcoding result with features and metrics.
        """
        b = self._backend
        mlp_input = b.array(mlp_input) if not hasattr(mlp_input, "shape") else mlp_input
        mlp_output = b.array(mlp_output) if not hasattr(mlp_output, "shape") else mlp_output
        b.eval(mlp_input, mlp_output)

        batch_size = int(mlp_input.shape[0])
        if batch_size == 0:
            return self._empty_result(weights)

        # Cast for numerical stability
        x = b.astype(mlp_input, "float32")
        y = b.astype(mlp_output, "float32")
        W_enc = b.astype(weights.W_enc, "float32")
        b_enc = b.astype(weights.b_enc, "float32")
        W_dec = b.astype(weights.W_dec, "float32")
        b_dec = b.astype(weights.b_dec, "float32")
        b.eval(x, y, W_enc, b_enc, W_dec, b_dec)

        # Encode: features = ReLU(x @ W_enc + b_enc)
        pre_activation = b.matmul(x, W_enc) + b.reshape(b_enc, (1, -1))
        sparse_features = b.maximum(pre_activation, b.zeros_like(pre_activation))
        b.eval(sparse_features)

        # Decode: y_hat = features @ W_dec + b_dec
        predicted_output = b.matmul(sparse_features, W_dec) + b.reshape(b_dec, (1, -1))
        b.eval(predicted_output)

        # Geodesic reconstruction loss
        diff = y - predicted_output
        norms = geodesic_norms(diff, b)
        b.eval(norms)
        reconstruction_loss = float(b.to_scalar(b.mean(norms)))

        # L1 sparsity loss
        l1_per_sample = b.sum(b.abs(sparse_features), axis=1)
        l1_loss = float(b.to_scalar(b.mean(l1_per_sample)))

        # Sparsity statistics
        eps = regularization_epsilon(b, sparse_features)
        active_mask = sparse_features > eps
        b.eval(active_mask)

        active_count = b.sum(b.astype(active_mask, "float32"), axis=1)
        sparsity = float(b.to_scalar(b.mean(active_count)))

        return TranscoderResult(
            sparse_features=sparse_features,
            predicted_output=predicted_output,
            reconstruction_loss=reconstruction_loss,
            sparsity=sparsity,
            l1_loss=l1_loss,
            active_features=active_mask,
        )

    def encode_only(self, mlp_input: Any, weights: TranscoderWeights) -> Any:
        """Encode MLP input to sparse features.

        Parameters
        ----------
        mlp_input : Array
            MLP layer input. Shape: [batch, input_dim].
        weights : TranscoderWeights
            Transcoder weights.

        Returns
        -------
        Array
            Sparse features. Shape: [batch, latent_dim].
        """
        b = self._backend
        x = b.array(mlp_input) if not hasattr(mlp_input, "shape") else mlp_input
        x = b.astype(x, "float32")
        W_enc = b.astype(weights.W_enc, "float32")
        b_enc = b.astype(weights.b_enc, "float32")
        b.eval(x, W_enc, b_enc)

        pre_activation = b.matmul(x, W_enc) + b.reshape(b_enc, (1, -1))
        features = b.maximum(pre_activation, b.zeros_like(pre_activation))
        b.eval(features)
        return features

    def decode_only(self, sparse_features: Any, weights: TranscoderWeights) -> Any:
        """Decode sparse features to MLP output.

        Parameters
        ----------
        sparse_features : Array
            Sparse features. Shape: [batch, latent_dim].
        weights : TranscoderWeights
            Transcoder weights.

        Returns
        -------
        Array
            Predicted MLP output. Shape: [batch, output_dim].
        """
        b = self._backend
        f = b.array(sparse_features) if not hasattr(sparse_features, "shape") else sparse_features
        f = b.astype(f, "float32")
        W_dec = b.astype(weights.W_dec, "float32")
        b_dec = b.astype(weights.b_dec, "float32")
        b.eval(f, W_dec, b_dec)

        output = b.matmul(f, W_dec) + b.reshape(b_dec, (1, -1))
        b.eval(output)
        return output

    def analyze_feature_contributions(
        self,
        sparse_features: Any,
        weights: TranscoderWeights,
        top_k: int = 10,
    ) -> list[FeatureContribution]:
        """Analyze which features contribute most to the output.

        Parameters
        ----------
        sparse_features : Array
            Sparse features for a single sample. Shape: [latent_dim].
        weights : TranscoderWeights
            Transcoder weights.
        top_k : int
            Number of top features to return.

        Returns
        -------
        list[FeatureContribution]
            Top-k contributing features.
        """
        b = self._backend
        features = b.array(sparse_features) if not hasattr(sparse_features, "shape") else sparse_features
        if len(b.shape(features)) == 2:
            features = features[0]  # Take first sample
        features = b.astype(features, "float32")
        W_dec = b.astype(weights.W_dec, "float32")
        b.eval(features, W_dec)

        latent_dim = int(features.shape[0])

        # Find top-k active features
        sorted_indices = b.argsort(b.abs(features))
        b.eval(sorted_indices)
        reversed_idx = b.arange(latent_dim - 1, -1, -1)
        top_indices = b.take(sorted_indices, reversed_idx, axis=0)[:top_k]
        b.eval(top_indices)

        max_abs = b.max(b.abs(features))
        b.eval(max_abs)
        min_activation = ulp_scalar(float(b.to_scalar(max_abs)), b)

        contributions = []
        for i in range(min(top_k, latent_dim)):
            idx = int(b.to_scalar(top_indices[i]))
            activation = float(b.to_scalar(features[idx]))

            if abs(activation) <= min_activation:
                continue

            # Contribution = activation * decoder_row
            decoder_row = W_dec[idx, :]
            contribution = activation * decoder_row
            b.eval(contribution)

            # Geodesic magnitude
            contrib_norm = geodesic_norms(b.reshape(contribution, (1, -1)), b)
            b.eval(contrib_norm)
            magnitude = float(b.to_scalar(contrib_norm[0]))

            contributions.append(FeatureContribution(
                feature_index=idx,
                activation=activation,
                contribution_to_output=contribution,
                contribution_magnitude=magnitude,
            ))

        return contributions

    def replace_mlp_forward(
        self,
        model: Any,
        layer_index: int,
        weights: TranscoderWeights,
    ) -> "_TranscoderContext":
        """Context manager to replace MLP with transcoder during inference.

        This enables circuit tracing by seeing which features activate
        when the model processes different inputs.

        Parameters
        ----------
        model : Any
            Model to modify.
        layer_index : int
            Layer whose MLP to replace.
        weights : TranscoderWeights
            Transcoder weights.

        Returns
        -------
        _TranscoderContext
            Context manager for temporary MLP replacement.

        Example
        -------
        >>> with tc.replace_mlp_forward(model, layer=16, weights=tc_weights):
        ...     output = model(input_ids)
        ...     # MLP at layer 16 is replaced with transcoder
        """
        base_model = getattr(model, "model", model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise RuntimeError("Model does not expose transformer layers.")

        return _TranscoderContext(
            layers=layers,
            layer_index=layer_index,
            transcoder=self,
            weights=weights,
        )

    def _normalize_decoder(self, W_dec: Any) -> Any:
        """Normalize decoder rows to unit geodesic norm."""
        b = self._backend
        latent_dim = int(W_dec.shape[0])

        norms = geodesic_norms(W_dec, b)
        b.eval(norms)

        eps = regularization_epsilon(b, W_dec)
        norms_safe = b.maximum(norms, b.full((latent_dim,), eps))
        norms_broadcast = b.reshape(norms_safe, (latent_dim, 1))
        W_dec_normalized = W_dec / norms_broadcast
        b.eval(W_dec_normalized)

        return W_dec_normalized

    def _empty_result(self, weights: TranscoderWeights) -> TranscoderResult:
        """Create empty result for zero-batch input."""
        b = self._backend
        config = weights.config

        return TranscoderResult(
            sparse_features=b.zeros((0, config.latent_dim)),
            predicted_output=b.zeros((0, config.output_dim)),
            reconstruction_loss=0.0,
            sparsity=0.0,
            l1_loss=0.0,
            active_features=b.zeros((0, config.latent_dim), dtype="bool"),
        )


class _TranscoderContext:
    """Context manager for MLP replacement."""

    def __init__(
        self,
        layers: list[Any],
        layer_index: int,
        transcoder: Transcoder,
        weights: TranscoderWeights,
    ) -> None:
        self._layers = layers
        self._layer_index = layer_index
        self._transcoder = transcoder
        self._weights = weights
        self._original_mlp: Any = None

    def __enter__(self) -> "_TranscoderContext":
        layer = self._layers[self._layer_index]
        # Save original MLP
        self._original_mlp = getattr(layer, "mlp", None)
        if self._original_mlp is None:
            raise RuntimeError(f"Layer {self._layer_index} has no mlp attribute")

        # Replace with transcoder wrapper
        wrapper = _TranscoderMLPWrapper(
            original_mlp=self._original_mlp,
            transcoder=self._transcoder,
            weights=self._weights,
        )
        setattr(layer, "mlp", wrapper)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Restore original MLP
        if self._original_mlp is not None:
            layer = self._layers[self._layer_index]
            setattr(layer, "mlp", self._original_mlp)


class _TranscoderMLPWrapper:
    """Wrapper that replaces MLP computation with transcoder."""

    def __init__(
        self,
        original_mlp: Any,
        transcoder: Transcoder,
        weights: TranscoderWeights,
    ) -> None:
        self._original_mlp = original_mlp
        self._transcoder = transcoder
        self._weights = weights
        self.captured_features: Any = None

    def __call__(self, x: Any) -> Any:
        # Encode to sparse features
        features = self._transcoder.encode_only(x, self._weights)
        self.captured_features = features

        # Decode to output
        output = self._transcoder.decode_only(features, self._weights)
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_mlp, name)


__all__ = [
    "TranscoderConfig",
    "TranscoderWeights",
    "TranscoderResult",
    "FeatureContribution",
    "Transcoder",
]
