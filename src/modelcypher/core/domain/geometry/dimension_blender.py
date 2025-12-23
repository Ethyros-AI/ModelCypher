"""
Dimension Blender.

Computes per-dimension alpha vectors for model merging based on
probe activation patterns. Dimensions that strongly activate for
coding probes get different alpha values than dimensions that
activate for reasoning probes.

This enables surgical merging at the finest granularity - individual
hidden state dimensions - rather than entire layers or modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

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
