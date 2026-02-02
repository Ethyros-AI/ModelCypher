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
Token-level data filtering via SAE latent labeling.

Implements the token labeling pipeline from "Shaping capabilities with
token-level data filtering" (arXiv:2601.21571v1).

Key ideas:
- Use SAE latents to identify tokens belonging to specific domains
- Label tokens where 2+ domain-specific latents activate above 4σ threshold
- Expand labels to adjacent tokens for context

References:
    - "Shaping capabilities with token-level data filtering" (Anthropic, 2025)
    - "Towards Monosemanticity" (Anthropic, 2023)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class TokenLabelingConfig:
    """Configuration for SAE-based token labeling.

    All thresholds are derived from the paper or from data statistics.

    Attributes
    ----------
    min_active_latents : int
        Minimum number of domain latents that must fire to label a token.
        Paper uses 2 latents as threshold (Section 3.2).
    activation_threshold_sigma : float
        Number of standard deviations above mean for latent activation.
        Paper uses 4σ threshold (Section 3.2).
    expand_adjacent : bool
        Whether to expand labels to adjacent tokens.
        Paper expands by 1 token on each side.
    expansion_radius : int
        Number of tokens to expand on each side when expand_adjacent=True.
    """

    min_active_latents: int = 2
    activation_threshold_sigma: float = 4.0
    expand_adjacent: bool = True
    expansion_radius: int = 1


@dataclass(frozen=True)
class LatentActivationStats:
    """Statistics for latent activations across a dataset.

    Used to compute the threshold for "active" latents.

    Attributes
    ----------
    mean : Array
        Per-latent mean activation. Shape: [latent_dim].
    std : Array
        Per-latent standard deviation. Shape: [latent_dim].
    sample_count : int
        Number of tokens used to compute statistics.
    """

    mean: "Array"
    std: "Array"
    sample_count: int


@dataclass(frozen=True)
class TokenLabelResult:
    """Result of token labeling operation.

    Attributes
    ----------
    labels : Array
        Binary labels for each token. Shape: [total_tokens].
        1 = token belongs to target domain, 0 = does not.
    confidence_scores : Array
        Confidence score for each label. Shape: [total_tokens].
        Based on number of active latents and their activation strength.
    active_latent_counts : Array
        Number of domain latents active at each position. Shape: [total_tokens].
    text_lengths : list[int]
        Length of each text in tokens, for reconstructing text boundaries.
    """

    labels: "Array"
    confidence_scores: "Array"
    active_latent_counts: "Array"
    text_lengths: list[int]


class SAETokenLabeler:
    """Labels tokens using SAE latent activations.

    Implements the labeling pipeline from arXiv:2601.21571v1:
    1. Identify domain-specific latents (provided by user)
    2. Compute activation statistics for thresholding
    3. Label tokens where enough domain latents fire above threshold
    4. Optionally expand labels to adjacent tokens

    Example
    -------
    >>> labeler = SAETokenLabeler(config)
    >>> stats = labeler.compute_activation_stats(sae_activations)
    >>> result = labeler.label_tokens(
    ...     activations=sae_activations,
    ...     domain_latent_indices=[42, 137, 256],
    ...     stats=stats,
    ...     text_lengths=[100, 150, 75],
    ... )
    >>> # result.labels contains binary token labels
    """

    def __init__(
        self,
        config: TokenLabelingConfig | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize labeler.

        Parameters
        ----------
        config : TokenLabelingConfig, optional
            Labeling configuration. Uses defaults if None.
        backend : Backend, optional
            Computation backend. If None, uses default.
        """
        self._config = config or TokenLabelingConfig()
        self._backend = backend or get_default_backend()

    @property
    def config(self) -> TokenLabelingConfig:
        """Get labeling configuration."""
        return self._config

    @property
    def backend(self) -> "Backend":
        """Get computation backend."""
        return self._backend

    def compute_activation_stats(
        self,
        activations: "Array",
    ) -> LatentActivationStats:
        """Compute activation statistics for thresholding.

        Parameters
        ----------
        activations : Array
            SAE latent activations. Shape: [total_tokens, latent_dim].

        Returns
        -------
        LatentActivationStats
            Per-latent mean and standard deviation.
        """
        b = self._backend
        acts = b.array(activations) if not hasattr(activations, "shape") else activations
        acts = b.astype(acts, "float32")
        b.eval(acts)

        n_tokens = int(acts.shape[0])
        if n_tokens == 0:
            latent_dim = int(acts.shape[1]) if acts.ndim > 1 else 0
            return LatentActivationStats(
                mean=b.zeros((latent_dim,)),
                std=b.ones((latent_dim,)),
                sample_count=0,
            )

        # Compute per-latent mean and std
        mean = b.mean(acts, axis=0)
        b.eval(mean)

        # Compute variance manually for numerical stability
        diff = acts - b.reshape(mean, (1, -1))
        var = b.mean(diff * diff, axis=0)
        std = b.sqrt(var)
        b.eval(std)

        return LatentActivationStats(
            mean=mean,
            std=std,
            sample_count=n_tokens,
        )

    def label_tokens(
        self,
        activations: "Array",
        domain_latent_indices: list[int],
        stats: LatentActivationStats,
        text_lengths: list[int],
    ) -> TokenLabelResult:
        """Label tokens based on SAE latent activations.

        A token is labeled positive if at least `min_active_latents` domain
        latents activate above the `activation_threshold_sigma` threshold.

        Parameters
        ----------
        activations : Array
            SAE latent activations. Shape: [total_tokens, latent_dim].
        domain_latent_indices : list[int]
            Indices of latents associated with the target domain.
        stats : LatentActivationStats
            Activation statistics for computing thresholds.
        text_lengths : list[int]
            Length of each text in tokens.

        Returns
        -------
        TokenLabelResult
            Token labels and metadata.
        """
        b = self._backend
        config = self._config

        acts = b.array(activations) if not hasattr(activations, "shape") else activations
        acts = b.astype(acts, "float32")
        b.eval(acts)

        n_tokens = int(acts.shape[0])
        latent_dim = int(acts.shape[1])

        if n_tokens == 0 or not domain_latent_indices:
            return TokenLabelResult(
                labels=b.zeros((n_tokens,), dtype="int32"),
                confidence_scores=b.zeros((n_tokens,)),
                active_latent_counts=b.zeros((n_tokens,), dtype="int32"),
                text_lengths=text_lengths,
            )

        # Extract domain latent activations
        domain_indices = b.array(domain_latent_indices, dtype="int32")
        b.eval(domain_indices)
        domain_acts = b.take(acts, domain_indices, axis=1)  # [n_tokens, n_domain]
        b.eval(domain_acts)

        # Get stats for domain latents
        domain_mean = b.take(stats.mean, domain_indices, axis=0)
        domain_std = b.take(stats.std, domain_indices, axis=0)
        b.eval(domain_mean, domain_std)

        # Compute threshold: mean + sigma * std
        eps = division_epsilon(b, domain_std)
        domain_std_safe = b.maximum(domain_std, b.full(domain_std.shape, eps))
        threshold = domain_mean + config.activation_threshold_sigma * domain_std_safe
        threshold = b.reshape(threshold, (1, -1))
        b.eval(threshold)

        # Count active latents per token
        above_threshold = domain_acts > threshold
        active_counts = b.sum(b.astype(above_threshold, "int32"), axis=1)
        b.eval(active_counts)

        # Label tokens with enough active latents
        labels = b.astype(active_counts >= config.min_active_latents, "int32")
        b.eval(labels)

        # Compute confidence scores
        # Confidence = (active_count / n_domain_latents) * (mean_activation / threshold)
        n_domain = len(domain_latent_indices)
        count_ratio = b.astype(active_counts, "float32") / float(n_domain)

        # Mean activation of domain latents above threshold
        domain_acts_masked = b.where(
            above_threshold,
            domain_acts,
            b.zeros_like(domain_acts),
        )
        threshold_scalar = b.mean(threshold)
        b.eval(threshold_scalar)
        threshold_val = float(b.to_scalar(threshold_scalar))

        active_sum = b.sum(domain_acts_masked, axis=1)
        active_count_float = b.astype(active_counts, "float32")
        eps_arr = b.full(active_count_float.shape, eps)
        active_count_safe = b.maximum(active_count_float, eps_arr)
        mean_active = active_sum / active_count_safe
        activation_ratio = mean_active / max(threshold_val, eps)

        confidence = count_ratio * b.minimum(activation_ratio, b.ones_like(activation_ratio))
        b.eval(confidence)

        # Expand labels if configured
        if config.expand_adjacent:
            labels = self._expand_labels(labels, text_lengths)
            b.eval(labels)

        return TokenLabelResult(
            labels=labels,
            confidence_scores=confidence,
            active_latent_counts=active_counts,
            text_lengths=text_lengths,
        )

    def _expand_labels(
        self,
        labels: "Array",
        text_lengths: list[int],
    ) -> "Array":
        """Expand positive labels to adjacent tokens within text boundaries.

        Parameters
        ----------
        labels : Array
            Binary labels. Shape: [total_tokens].
        text_lengths : list[int]
            Length of each text in tokens.

        Returns
        -------
        Array
            Expanded labels. Shape: [total_tokens].
        """
        b = self._backend
        config = self._config
        radius = config.expansion_radius

        labels_int = b.astype(labels, "int32")
        b.eval(labels_int)
        n_tokens = int(labels.shape[0])

        if n_tokens == 0 or radius <= 0:
            return labels_int

        # Convert to list for manipulation (expansion is text-boundary aware)
        labels_list = [int(x) for x in b.tolist(labels_int)]
        expanded_list = labels_list.copy()

        # Process each text separately to respect boundaries
        offset = 0
        for length in text_lengths:
            if length <= 0:
                continue

            text_end = min(offset + length, n_tokens)

            # Find original positive labels in this text (use original, not expanded)
            for i in range(offset, text_end):
                if labels_list[i] == 1:
                    # Expand to neighbors within text boundary
                    for delta in range(-radius, radius + 1):
                        neighbor = i + delta
                        if offset <= neighbor < text_end:
                            expanded_list[neighbor] = 1

            offset = text_end

        expanded = b.array(expanded_list, dtype="int32")
        b.eval(expanded)
        return expanded

    def calibrate_threshold(
        self,
        activations: "Array",
        domain_latent_indices: list[int],
        target_positive_rate: float = 0.1,
    ) -> float:
        """Calibrate activation threshold to achieve target positive rate.

        Instead of using a fixed sigma threshold, find the threshold that
        labels approximately `target_positive_rate` of tokens as positive.

        Parameters
        ----------
        activations : Array
            SAE latent activations. Shape: [total_tokens, latent_dim].
        domain_latent_indices : list[int]
            Indices of latents associated with the target domain.
        target_positive_rate : float
            Target fraction of tokens to label positive.

        Returns
        -------
        float
            Calibrated sigma threshold.
        """
        b = self._backend
        config = self._config

        acts = b.array(activations) if not hasattr(activations, "shape") else activations
        acts = b.astype(acts, "float32")
        b.eval(acts)

        n_tokens = int(acts.shape[0])
        if n_tokens == 0 or not domain_latent_indices:
            return config.activation_threshold_sigma

        # Compute stats
        stats = self.compute_activation_stats(acts)

        # Extract domain latent activations
        domain_indices = b.array(domain_latent_indices, dtype="int32")
        b.eval(domain_indices)
        domain_acts = b.take(acts, domain_indices, axis=1)
        domain_mean = b.take(stats.mean, domain_indices, axis=0)
        domain_std = b.take(stats.std, domain_indices, axis=0)
        b.eval(domain_acts, domain_mean, domain_std)

        eps = division_epsilon(b, domain_std)
        domain_std_safe = b.maximum(domain_std, b.full(domain_std.shape, eps))

        # Binary search for threshold (in units of standard deviations).
        # Upper bound 10σ chosen to be well beyond typical activation ranges;
        # if threshold needs >10σ, the target rate is likely too low.
        low_sigma = 0.0
        high_sigma = 10.0
        target_count = int(target_positive_rate * n_tokens)

        for _ in range(20):  # Binary search iterations (log2(10/precision) ≈ 20)
            mid_sigma = (low_sigma + high_sigma) / 2.0
            threshold = domain_mean + mid_sigma * domain_std_safe
            threshold = b.reshape(threshold, (1, -1))
            b.eval(threshold)

            above = domain_acts > threshold
            active_counts = b.sum(b.astype(above, "int32"), axis=1)
            labels = b.astype(active_counts >= config.min_active_latents, "int32")
            positive_count = b.sum(labels)
            b.eval(positive_count)
            count = int(b.to_scalar(positive_count))

            if count > target_count:
                low_sigma = mid_sigma
            else:
                high_sigma = mid_sigma

            # Early exit if within 1% of target (or at least 1 token).
            # 1% tolerance balances precision vs. computation.
            if abs(count - target_count) <= max(1, int(0.01 * n_tokens)):
                break

        return mid_sigma


__all__ = [
    "TokenLabelingConfig",
    "LatentActivationStats",
    "TokenLabelResult",
    "SAETokenLabeler",
]
