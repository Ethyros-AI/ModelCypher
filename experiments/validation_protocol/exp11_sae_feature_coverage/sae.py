# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sparse Autoencoder for activation space decomposition.

SAEs decompose activations into sparse, interpretable features.
Used to identify which dimensions our probes don't cover.

Based on: arXiv:2506.23845 "Use Sparse Autoencoders to Discover Unknown Concepts"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class SAEConfig:
    """SAE configuration.

    Defaults follow common practice from the literature.
    """
    input_dim: int
    expansion_factor: int = 8  # hidden_dim = input_dim * expansion_factor
    sparsity_coefficient: float = 0.04  # L1 penalty weight
    learning_rate: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 10

    @property
    def hidden_dim(self) -> int:
        return self.input_dim * self.expansion_factor


class SparseAutoencoder:
    """Sparse Autoencoder with L1 sparsity penalty.

    Architecture:
        encode: x -> ReLU(W_enc @ (x - b_pre) + b_enc)
        decode: z -> W_dec @ z + b_dec

    Sparsity is encouraged via L1 penalty on activations in the loss function.
    """

    def __init__(self, config: SAEConfig):
        import mlx.core as mx

        self.config = config
        d = config.input_dim
        h = config.hidden_dim

        # Initialize weights with Xavier/Glorot initialization
        scale = (2.0 / (d + h)) ** 0.5

        self.W_enc = mx.random.normal((h, d)) * scale
        self.b_enc = mx.zeros((h,))
        self.b_pre = mx.zeros((d,))

        self.W_dec = mx.random.normal((d, h)) * scale
        self.b_dec = mx.zeros((d,))

        # Normalize decoder columns to unit norm
        norms = mx.sqrt(mx.sum(self.W_dec ** 2, axis=0, keepdims=True))
        self.W_dec = self.W_dec / (norms + 1e-8)

    def encode(self, x):
        """Encode input to sparse representation.

        Uses ReLU activation with L1 penalty for sparsity (standard SAE approach).
        The L1 loss in the loss function encourages sparsity.
        """
        import mlx.core as mx

        # Pre-bias subtraction
        x_centered = x - self.b_pre

        # Linear transform + ReLU
        z = x_centered @ self.W_enc.T + self.b_enc
        z_sparse = mx.maximum(z, 0)  # ReLU activation

        return z_sparse

    def decode(self, z):
        """Decode sparse representation back to input space."""
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x):
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x):
        """Compute reconstruction loss + L1 sparsity penalty."""
        import mlx.core as mx

        x_hat, z = self.forward(x)

        # MSE reconstruction loss
        recon_loss = mx.mean((x - x_hat) ** 2)

        # L1 sparsity loss on activations
        l1_loss = mx.mean(mx.abs(z))

        total_loss = recon_loss + self.config.sparsity_coefficient * l1_loss

        return total_loss, recon_loss, l1_loss

    def parameters(self):
        """Return all trainable parameters as dict for MLX optimizer."""
        return {
            "W_enc": self.W_enc,
            "b_enc": self.b_enc,
            "b_pre": self.b_pre,
            "W_dec": self.W_dec,
            "b_dec": self.b_dec,
        }

    def set_parameters(self, params):
        """Set parameters from dict."""
        self.W_enc = params["W_enc"]
        self.b_enc = params["b_enc"]
        self.b_pre = params["b_pre"]
        self.W_dec = params["W_dec"]
        self.b_dec = params["b_dec"]


def train_sae(
    activations,
    config: SAEConfig,
    verbose: bool = True,
) -> SparseAutoencoder:
    """Train SAE on activation data.

    Args:
        activations: [n_samples, input_dim] array of activations
        config: SAE configuration
        verbose: Print training progress

    Returns:
        Trained SparseAutoencoder
    """
    import mlx.core as mx
    import mlx.optimizers as optim

    sae = SparseAutoencoder(config)
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    n_samples = activations.shape[0]
    n_batches = max(1, n_samples // config.batch_size)

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_l1 = 0.0

        for batch_idx in range(n_batches):
            start = batch_idx * config.batch_size
            end = min(start + config.batch_size, n_samples)
            batch = activations[start:end]

            # Compute loss and gradients
            params = sae.parameters()

            def loss_fn(p):
                sae.set_parameters(p)
                total, _, _ = sae.loss(batch)
                return total

            loss_val, grads = mx.value_and_grad(loss_fn)(params)

            # Get individual loss components for logging
            sae.set_parameters(params)
            _, recon_loss, l1_loss = sae.loss(batch)

            # Update parameters
            new_params = optimizer.apply_gradients(grads, params)
            sae.set_parameters(new_params)

            # Normalize decoder columns to unit norm
            norms = mx.sqrt(mx.sum(sae.W_dec ** 2, axis=0, keepdims=True))
            sae.W_dec = sae.W_dec / (norms + 1e-8)

            mx.eval(sae.W_enc, sae.W_dec, sae.b_enc, sae.b_dec, sae.b_pre)

            total_loss += float(loss_val)
            total_recon += float(recon_loss)
            total_l1 += float(l1_loss)

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_l1 = total_l1 / n_batches

        if verbose:
            print(f"Epoch {epoch+1}/{config.num_epochs}: "
                  f"loss={avg_loss:.4f}, recon={avg_recon:.4f}, l1={avg_l1:.4f}")

    return sae


def find_dormant_features(
    sae: SparseAutoencoder,
    probe_activations,
    threshold: float = 0.01,
) -> tuple:
    """Find SAE features that never activate on probe set.

    Args:
        sae: Trained sparse autoencoder
        probe_activations: [n_probes, input_dim] probe activation matrix
        threshold: Activation level below which feature is considered dormant

    Returns:
        (dormant_mask, max_activations, n_dormant, n_total)
    """
    import mlx.core as mx

    # Encode all probes
    encoded = sae.encode(probe_activations)
    mx.eval(encoded)

    # Find max activation per feature across all probes
    max_activation = mx.max(encoded, axis=0)
    mx.eval(max_activation)

    # Identify dormant features
    dormant_mask = max_activation < threshold
    mx.eval(dormant_mask)

    n_dormant = int(mx.sum(dormant_mask.astype(mx.int32)))
    n_total = encoded.shape[1]

    return dormant_mask, max_activation, n_dormant, n_total
