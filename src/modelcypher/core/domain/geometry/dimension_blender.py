"""
Dimension Blender.

Computes per-dimension alpha vectors for model merging based on
probe activation patterns and activation correlations.

Two complementary approaches:

1. Domain-based: Classify dimensions by their domain affinity (COMPUTATIONAL,
   LINGUISTIC, etc.) and assign domain-specific alphas.

2. Correlation-based: For each dimension, compute correlation between source
   and target activations across probes. High correlation = trust either model;
   low correlation = trust target for stability.

Reference:
- Each dimension's weight is computed from its OWN correlation strength

Mathematical Foundation
-----------------------
For dimension d with activations source_acts[:, d] and target_acts[:, d]:

1. correlation_d = cosine_similarity(source[:, d], target[:, d])

2. weight_d = sigmoid((1 - correlation_d) * scale)
   - High correlation → low weight → trust either → use default alpha
   - Low correlation → high weight → use stability alpha (trust target)

3. final_alpha = weight * stability_alpha + (1 - weight) * base_alpha
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )

from modelcypher.core.domain.agents.unified_atlas import (
    AtlasDomain,
    AtlasProbe,
    UnifiedAtlasInventory,
)

logger = logging.getLogger(__name__)


@dataclass
class DimensionDomainScores:
    """Domain affinity scores for a single dimension."""
    dimension_index: int
    scores: dict[AtlasDomain, float] = field(default_factory=dict)
    total_activation: float = 0.0
    dominant_domain: Optional[AtlasDomain] = None
    confidence: float = 0.0  # How confident we are in the classification

    def normalize(self) -> None:
        """Normalize scores and compute dominant domain."""
        if not self.scores:
            return

        total = sum(self.scores.values())
        if total > 0:
            for domain in self.scores:
                self.scores[domain] /= total

        # Find dominant domain
        if self.scores:
            self.dominant_domain = max(self.scores.keys(), key=lambda d: self.scores[d])
            self.confidence = self.scores[self.dominant_domain]


@dataclass
class LayerDimensionProfile:
    """Per-dimension domain profiles for a single layer."""
    layer_index: int
    dimension_count: int
    dimension_scores: dict[int, DimensionDomainScores] = field(default_factory=dict)

    def get_domain_distribution(self) -> dict[AtlasDomain, int]:
        """Get count of dimensions dominated by each domain."""
        distribution: dict[AtlasDomain, int] = {}
        for scores in self.dimension_scores.values():
            if scores.dominant_domain:
                distribution[scores.dominant_domain] = distribution.get(
                    scores.dominant_domain, 0
                ) + 1
        return distribution


@dataclass(frozen=True)
class DimensionBlendConfig:
    """
    Configuration for dimension-level blending.

    The domain_alpha_map controls which model to favor for each domain:
    - 0.0 = fully favor target model
    - 0.5 = equal blend
    - 1.0 = fully favor source model
    """
    # Domain -> alpha preference
    domain_alpha_map: dict[AtlasDomain, float] = field(default_factory=dict)

    # Minimum total activation to consider a dimension classified
    activation_threshold: float = 0.05

    # Default alpha for unclassified or low-confidence dimensions
    default_alpha: float = 0.5

    # Minimum confidence to apply domain-specific alpha
    confidence_threshold: float = 0.3

    # Smoothing factor: blend domain alpha toward default (0 = pure domain, 1 = pure default)
    smoothing: float = 0.2

    def __hash__(self) -> int:
        return hash((
            tuple(sorted(self.domain_alpha_map.items())),
            self.activation_threshold,
            self.default_alpha,
            self.confidence_threshold,
            self.smoothing,
        ))


# Default domain affinity map for Instruct → Coder merges
# Higher alpha = more source contribution
# For source=Instruct, target=Coder: high alpha keeps reasoning, low alpha keeps coding
INSTRUCT_TO_CODER_AFFINITY: dict[AtlasDomain, float] = {
    # Favor Coder (target) - low alpha
    # Note: computational/structural dominate (60%+), so we keep some Instruct
    AtlasDomain.COMPUTATIONAL: 0.35,  # Was 0.2 - too aggressive
    AtlasDomain.STRUCTURAL: 0.35,     # Was 0.25 - too aggressive

    # Favor Instruct (source) - high alpha
    AtlasDomain.LINGUISTIC: 0.75,
    AtlasDomain.MENTAL: 0.8,
    AtlasDomain.AFFECTIVE: 0.85,
    AtlasDomain.RELATIONAL: 0.75,

    # Lean toward Instruct for general reasoning (preserve language model)
    AtlasDomain.LOGICAL: 0.55,   # Was 0.5 - slight Instruct preference
    AtlasDomain.MATHEMATICAL: 0.5,
    AtlasDomain.TEMPORAL: 0.6,   # Was 0.5 - Instruct better at temporal reasoning
    AtlasDomain.SPATIAL: 0.55,   # Was 0.5 - slight Instruct preference
}

# Balanced affinity - preserves more of both capabilities
BALANCED_AFFINITY: dict[AtlasDomain, float] = {
    AtlasDomain.COMPUTATIONAL: 0.4,
    AtlasDomain.STRUCTURAL: 0.4,
    AtlasDomain.LINGUISTIC: 0.7,
    AtlasDomain.MENTAL: 0.7,
    AtlasDomain.AFFECTIVE: 0.75,
    AtlasDomain.RELATIONAL: 0.65,
    AtlasDomain.LOGICAL: 0.5,
    AtlasDomain.MATHEMATICAL: 0.5,
    AtlasDomain.TEMPORAL: 0.55,
    AtlasDomain.SPATIAL: 0.5,
}

# Inverse: for Coder → Instruct merges
CODER_TO_INSTRUCT_AFFINITY: dict[AtlasDomain, float] = {
    domain: 1.0 - alpha for domain, alpha in INSTRUCT_TO_CODER_AFFINITY.items()
}


class DimensionBlender:
    """
    Computes per-dimension alpha vectors from probe activations.

    The core insight: different dimensions in the hidden state
    specialize for different tasks. By analyzing which probes
    activate which dimensions, we can blend each dimension
    according to its specialization.
    """

    @staticmethod
    def build_probe_domain_map(probes: list[AtlasProbe]) -> dict[str, AtlasDomain]:
        """Build mapping from probe ID to domain."""
        return {probe.probe_id: probe.domain for probe in probes}

    @staticmethod
    def compute_dimension_profiles(
        fingerprints: list[dict],
        probe_domain_map: dict[str, AtlasDomain],
        layer_indices: list[int],
        hidden_dim: int,
    ) -> dict[int, LayerDimensionProfile]:
        """
        Compute per-dimension domain scores from fingerprints.

        Args:
            fingerprints: List of fingerprint dicts with probe_id and per-layer activations
            probe_domain_map: Mapping from probe_id to AtlasDomain
            layer_indices: Which layers to analyze
            hidden_dim: Size of hidden dimension

        Returns:
            Dict mapping layer_index to LayerDimensionProfile
        """
        profiles: dict[int, LayerDimensionProfile] = {}

        for layer_idx in layer_indices:
            profile = LayerDimensionProfile(
                layer_index=layer_idx,
                dimension_count=hidden_dim,
            )

            # Initialize dimension scores
            for dim_idx in range(hidden_dim):
                profile.dimension_scores[dim_idx] = DimensionDomainScores(
                    dimension_index=dim_idx
                )

            profiles[layer_idx] = profile

        # Aggregate activations by domain for each dimension
        for fp in fingerprints:
            probe_id = fp.get("probe_id", "")
            domain = probe_domain_map.get(probe_id)
            if domain is None:
                continue

            activated_dims = fp.get("activated_dimensions", {})
            for layer_idx_str, dims in activated_dims.items():
                try:
                    layer_idx = int(layer_idx_str)
                except ValueError:
                    continue

                if layer_idx not in profiles:
                    continue

                profile = profiles[layer_idx]
                for dim_data in dims:
                    dim_idx = dim_data.get("dimension", dim_data.get("index", -1))
                    activation = dim_data.get("activation", 0.0)

                    if dim_idx < 0 or dim_idx >= hidden_dim:
                        continue

                    scores = profile.dimension_scores[dim_idx]
                    scores.scores[domain] = scores.scores.get(domain, 0.0) + activation
                    scores.total_activation += activation

        # Normalize scores and compute dominant domains
        for profile in profiles.values():
            for scores in profile.dimension_scores.values():
                scores.normalize()

        return profiles

    @staticmethod
    def compute_alpha_vector(
        profile: LayerDimensionProfile,
        config: DimensionBlendConfig,
    ) -> np.ndarray:
        """
        Compute alpha vector for a single layer.

        Args:
            profile: Dimension domain profiles for the layer
            config: Blend configuration with domain affinity map

        Returns:
            Alpha vector of shape (hidden_dim,) with per-dimension alpha values
        """
        alpha = np.full(profile.dimension_count, config.default_alpha, dtype=np.float32)

        classified_count = 0
        for dim_idx, scores in profile.dimension_scores.items():
            if scores.total_activation < config.activation_threshold:
                continue

            if scores.confidence < config.confidence_threshold:
                continue

            if scores.dominant_domain is None:
                continue

            # Get domain-specific alpha
            domain_alpha = config.domain_alpha_map.get(
                scores.dominant_domain,
                config.default_alpha
            )

            # Apply confidence-weighted smoothing toward default
            # High confidence = more domain-specific
            # Low confidence = more default
            effective_confidence = min(1.0, scores.confidence / 0.5)  # Normalize to 0-1 range
            smoothed_alpha = (
                domain_alpha * (1.0 - config.smoothing) * effective_confidence +
                config.default_alpha * (1.0 - effective_confidence * (1.0 - config.smoothing))
            )

            alpha[dim_idx] = np.clip(smoothed_alpha, 0.0, 1.0)
            classified_count += 1

        logger.debug(
            "Layer %d: %d/%d dimensions classified (%.1f%%)",
            profile.layer_index,
            classified_count,
            profile.dimension_count,
            100.0 * classified_count / max(1, profile.dimension_count),
        )

        return alpha

    @classmethod
    def compute_alpha_vectors(
        cls,
        profiles: dict[int, LayerDimensionProfile],
        config: DimensionBlendConfig,
    ) -> dict[int, np.ndarray]:
        """
        Compute per-layer alpha vectors from dimension profiles.

        Args:
            profiles: Dict of layer_index -> LayerDimensionProfile
            config: Blend configuration

        Returns:
            Dict mapping layer_index to alpha vector (shape: hidden_dim,)
        """
        return {
            layer_idx: cls.compute_alpha_vector(profile, config)
            for layer_idx, profile in profiles.items()
        }

    @classmethod
    def summarize_profiles(
        cls,
        profiles: dict[int, LayerDimensionProfile],
    ) -> dict:
        """Generate summary statistics for dimension profiles."""
        summary = {
            "layer_count": len(profiles),
            "layers": {},
        }

        for layer_idx, profile in sorted(profiles.items()):
            dist = profile.get_domain_distribution()
            total_classified = sum(dist.values())

            layer_summary = {
                "dimension_count": profile.dimension_count,
                "classified_count": total_classified,
                "classification_rate": total_classified / max(1, profile.dimension_count),
                "domain_distribution": {d.value: c for d, c in dist.items()},
            }
            summary["layers"][layer_idx] = layer_summary

        return summary


# =============================================================================
# Correlation-Based Dimension Weighting
# =============================================================================


@dataclass(frozen=True)
class CorrelationWeightConfig:
    """
    Configuration for correlation-based dimension weighting.

    The correlation between source and target activations determines
    how much to trust the default alpha vs stability alpha.
    """

    # Scale factor for sigmoid transformation
    # Higher scale = sharper transition based on correlation
    correlation_scale: float = 5.0

    # Base alpha used when correlation is high (dimensions agree)
    base_alpha: float = 0.5

    # Stability alpha used when correlation is low (dimensions disagree)
    # Higher = trust target more for stability
    stability_alpha: float = 0.7

    # Epsilon for numerical stability
    epsilon: float = 1e-8

    # Minimum correlation before applying stability bias
    min_correlation_for_default: float = 0.8

    @classmethod
    def default(cls) -> CorrelationWeightConfig:
        """Default configuration."""
        return cls()

    @classmethod
    def conservative(cls) -> CorrelationWeightConfig:
        """Conservative: more stability bias."""
        return cls(stability_alpha=0.8, correlation_scale=3.0)

    @classmethod
    def aggressive(cls) -> CorrelationWeightConfig:
        """Aggressive: less stability bias."""
        return cls(stability_alpha=0.6, correlation_scale=7.0)


@dataclass
class DimensionCorrelations:
    """Per-dimension correlation metrics between source and target."""

    # Correlations per dimension [hidden_dim]
    correlations: np.ndarray

    # Mean correlation across all dimensions
    mean_correlation: float

    # Standard deviation of correlations
    std_correlation: float

    # Number of high-correlation dimensions (agree well)
    high_correlation_count: int

    # Number of low-correlation dimensions (disagree)
    low_correlation_count: int

    @property
    def agreement_ratio(self) -> float:
        """Fraction of dimensions with high correlation."""
        total = len(self.correlations)
        return self.high_correlation_count / max(total, 1)


def compute_dimension_correlations(
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    config: Optional[CorrelationWeightConfig] = None,
) -> DimensionCorrelations:
    """
    Compute per-dimension correlations between source and target activations.

    Args:
        source_activations: Source model activations [num_probes, hidden_dim]
        target_activations: Target model activations [num_probes, hidden_dim]
        config: Correlation weight configuration

    Returns:
        DimensionCorrelations with per-dimension correlation values
    """
    if config is None:
        config = CorrelationWeightConfig.default()

    # Ensure same shape
    if source_activations.shape != target_activations.shape:
        raise ValueError(
            f"Shape mismatch: {source_activations.shape} vs {target_activations.shape}"
        )

    num_probes, hidden_dim = source_activations.shape

    # Compute cosine similarity per dimension
    # For each dimension d: cos_sim(source[:, d], target[:, d])
    correlations = np.zeros(hidden_dim, dtype=np.float32)

    for d in range(hidden_dim):
        s_col = source_activations[:, d]
        t_col = target_activations[:, d]

        s_norm = np.linalg.norm(s_col)
        t_norm = np.linalg.norm(t_col)

        if s_norm < config.epsilon or t_norm < config.epsilon:
            # If either is near-zero, correlation undefined → assume disagreement
            correlations[d] = 0.0
        else:
            correlations[d] = np.dot(s_col, t_col) / (s_norm * t_norm)

    # Clamp to [-1, 1]
    correlations = np.clip(correlations, -1.0, 1.0)

    mean_corr = float(np.mean(correlations))
    std_corr = float(np.std(correlations))

    high_threshold = config.min_correlation_for_default
    low_threshold = 0.5

    high_count = int(np.sum(correlations >= high_threshold))
    low_count = int(np.sum(correlations < low_threshold))

    return DimensionCorrelations(
        correlations=correlations,
        mean_correlation=mean_corr,
        std_correlation=std_corr,
        high_correlation_count=high_count,
        low_correlation_count=low_count,
    )


def compute_correlation_weights(
    correlations: DimensionCorrelations,
    config: Optional[CorrelationWeightConfig] = None,
) -> np.ndarray:
    """
    Compute per-dimension weights from correlations.

    High correlation → low weight (use base alpha)
    Low correlation → high weight (use stability alpha)

    Args:
        correlations: Per-dimension correlation metrics
        config: Correlation weight configuration

    Returns:
        Weight vector [hidden_dim] in range [0, 1]
    """
    if config is None:
        config = CorrelationWeightConfig.default()

    # Transform: weight = sigmoid((1 - correlation) * scale)
    # correlation=1.0 → weight≈0 → base_alpha
    # correlation=0.0 → weight≈0.99 → stability_alpha
    # correlation=-1.0 → weight≈1 → stability_alpha (maximum disagreement)

    disagreement = 1.0 - correlations.correlations  # [0, 2] range
    weights = _sigmoid((disagreement - 1.0) * config.correlation_scale)

    return weights.astype(np.float32)


def apply_correlation_weights_to_alpha(
    base_alpha_vector: np.ndarray,
    correlation_weights: np.ndarray,
    config: Optional[CorrelationWeightConfig] = None,
) -> np.ndarray:
    """
    Apply correlation weights to modulate alpha values.

    Args:
        base_alpha_vector: Per-dimension base alpha [hidden_dim]
        correlation_weights: Per-dimension weights from correlations [hidden_dim]
        config: Correlation weight configuration

    Returns:
        Modulated alpha vector [hidden_dim]
    """
    if config is None:
        config = CorrelationWeightConfig.default()

    # final_alpha = (1 - weight) * base_alpha + weight * stability_alpha
    # weight=0 → base_alpha (correlation high, trust either)
    # weight=1 → stability_alpha (correlation low, trust target)

    modulated = (
        (1.0 - correlation_weights) * base_alpha_vector
        + correlation_weights * config.stability_alpha
    )

    return np.clip(modulated, 0.0, 1.0).astype(np.float32)


def compute_correlation_based_alpha(
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    base_alpha: float = 0.5,
    config: Optional[CorrelationWeightConfig] = None,
) -> tuple[np.ndarray, DimensionCorrelations]:
    """
    Compute per-dimension alpha based on activation correlations.

    This is the main entry point for correlation-based blending.
    Dimensions with high source-target correlation use the base alpha.
    Dimensions with low correlation use the stability alpha (trust target).

    Args:
        source_activations: Source model activations [num_probes, hidden_dim]
        target_activations: Target model activations [num_probes, hidden_dim]
        base_alpha: Default alpha value
        config: Correlation weight configuration

    Returns:
        Tuple of (alpha_vector, correlations)
    """
    if config is None:
        config = CorrelationWeightConfig.default()

    hidden_dim = source_activations.shape[1]
    base_alpha_vec = np.full(hidden_dim, base_alpha, dtype=np.float32)

    correlations = compute_dimension_correlations(
        source_activations, target_activations, config
    )

    weights = compute_correlation_weights(correlations, config)
    alpha_vector = apply_correlation_weights_to_alpha(base_alpha_vec, weights, config)

    logger.debug(
        "Correlation-based alpha: mean_corr=%.3f, agree_ratio=%.2f%%",
        correlations.mean_correlation,
        correlations.agreement_ratio * 100,
    )

    return alpha_vector, correlations


def blend_domain_and_correlation_alpha(
    domain_alpha: np.ndarray,
    correlation_alpha: np.ndarray,
    blend_ratio: float = 0.5,
) -> np.ndarray:
    """
    Blend domain-based and correlation-based alpha vectors.

    This combines the domain classification signal with the
    correlation stability signal.

    Args:
        domain_alpha: Alpha from domain classification [hidden_dim]
        correlation_alpha: Alpha from correlation weighting [hidden_dim]
        blend_ratio: Weight for correlation alpha (0 = pure domain, 1 = pure correlation)

    Returns:
        Blended alpha vector [hidden_dim]
    """
    blend_ratio = max(0.0, min(1.0, blend_ratio))

    blended = (1.0 - blend_ratio) * domain_alpha + blend_ratio * correlation_alpha

    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def correlation_summary(correlations: DimensionCorrelations) -> dict:
    """
    Generate summary statistics for dimension correlations.

    Args:
        correlations: Per-dimension correlation metrics

    Returns:
        Summary dictionary
    """
    return {
        "hidden_dim": len(correlations.correlations),
        "mean_correlation": correlations.mean_correlation,
        "std_correlation": correlations.std_correlation,
        "min_correlation": float(np.min(correlations.correlations)),
        "max_correlation": float(np.max(correlations.correlations)),
        "high_correlation_count": correlations.high_correlation_count,
        "low_correlation_count": correlations.low_correlation_count,
        "agreement_ratio": correlations.agreement_ratio,
    }
