"""
Verb-Noun Dimension Classifier.

Classifies embedding dimensions as Verb (skill/trajectory) or Noun (knowledge/position).
This enables smarter model merging by treating skill and knowledge dimensions differently.

Mathematical Foundation
-----------------------
For each dimension `d` at layer `l`, we compute:

    NounStability(l,d) = 1 - CoeffVar[primeActivation(p,l,d) for p in primes]
    VerbVariance(l,d)  = Var[gateActivation(g,l,d) for g in gates]
    Ratio(l,d) = VerbVariance / (NounStability + epsilon)

    High ratio → Verb dimension (trust Source for skills)
    Low ratio  → Noun dimension (trust Target for knowledge)

Usage
-----
For an Instruct → Coder merge:
    - Verb dimensions get low alpha (trust Coder - has coding skills)
    - Noun dimensions get high alpha (trust Instruct - has language knowledge)

This is the opposite of domain-based classification which looks at
*what* activates a dimension rather than *how* it behaves.

Reference
---------
- Wierzbicka (1996) "Semantics: Primes and Universals" - semantic primes as knowledge anchors
- Schönfinkel (1924) "Über die Bausteine der mathematischen Logik" - combinators as minimal verbs
- Ported from TrainingCypher/VerbNounDimensionClassifier.swift
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DimensionClass(str, Enum):
    """Classification of a single dimension."""

    VERB = "verb"  # Skill dimension - high variance, trust Source (skill donor)
    NOUN = "noun"  # Knowledge dimension - high stability, trust Target (knowledge base)
    MIXED = "mixed"  # Mixed dimension - use standard blending


@dataclass(frozen=True)
class VerbNounConfig:
    """Configuration for verb/noun classification."""

    # Threshold above which a dimension is classified as Verb
    # Ratio = VerbVariance / NounStability. Higher ratio → more verb-like.
    verb_threshold: float = 2.0

    # Threshold below which a dimension is classified as Noun
    noun_threshold: float = 0.5

    # Epsilon to prevent division by zero
    epsilon: float = 1e-6

    # Blend weight for Verb dimensions (0 = full Target, 1 = full Source)
    # Verb = skills → trust Source (skill donor) more → HIGH alpha
    verb_alpha: float = 0.8  # 80% Source, 20% Target

    # Blend weight for Noun dimensions
    # Noun = knowledge → trust Target (knowledge base) more → LOW alpha
    noun_alpha: float = 0.2  # 20% Source, 80% Target

    # Blend weight for Mixed dimensions
    mixed_alpha: float = 0.5

    # Strength of verb/noun modulation (0 = disabled, 1 = full effect)
    # This interpolates between correlation-based and verb/noun-based weights
    # Swift default is 0.7
    modulation_strength: float = 0.7

    # Minimum activation variance to consider a dimension active
    min_activation_variance: float = 1e-8

    @classmethod
    def default(cls) -> VerbNounConfig:
        """Default configuration."""
        return cls()

    @classmethod
    def conservative(cls) -> VerbNounConfig:
        """Conservative: less aggressive verb/noun separation."""
        return cls(
            verb_threshold=3.0,
            noun_threshold=0.3,
            verb_alpha=0.7,  # Closer to 0.5, less extreme
            noun_alpha=0.3,  # Closer to 0.5, less extreme
            modulation_strength=0.5,  # Swift conservative uses 0.5
        )

    @classmethod
    def aggressive(cls) -> VerbNounConfig:
        """Aggressive: strong verb/noun separation."""
        return cls(
            verb_threshold=1.5,
            noun_threshold=0.7,
            verb_alpha=0.9,  # Strongly trust Source for skills
            noun_alpha=0.1,  # Strongly trust Target for knowledge
            modulation_strength=0.9,  # Swift aggressive uses 0.9
        )


@dataclass
class DimensionResult:
    """Per-dimension classification result."""

    dimension: int  # Index of the dimension
    classification: DimensionClass  # verb, noun, or mixed
    noun_stability: float  # 0-1, higher = more stable
    verb_variance: float  # Higher = more variable across gates
    ratio: float  # VerbVariance / NounStability
    alpha: float  # Recommended blend weight for this dimension


@dataclass
class VerbNounClassification:
    """Full classification result for all dimensions."""

    dimensions: list[DimensionResult]
    alpha_vector: np.ndarray  # Per-dimension blend weights
    verb_count: int
    noun_count: int
    mixed_count: int
    mean_noun_stability: float
    mean_verb_variance: float
    overall_ratio: float

    @property
    def total_dimensions(self) -> int:
        """Total dimension count."""
        return len(self.dimensions)

    @property
    def verb_fraction(self) -> float:
        """Fraction of dimensions classified as verb."""
        if self.total_dimensions == 0:
            return 0.0
        return self.verb_count / self.total_dimensions

    @property
    def noun_fraction(self) -> float:
        """Fraction of dimensions classified as noun."""
        if self.total_dimensions == 0:
            return 0.0
        return self.noun_count / self.total_dimensions


@dataclass
class LayerVerbNounClassification:
    """Multi-layer classification result."""

    layer_classifications: dict[int, VerbNounClassification]
    alpha_vectors_by_layer: dict[int, np.ndarray]

    @property
    def mean_verb_fraction(self) -> float:
        """Mean verb fraction across layers."""
        if not self.layer_classifications:
            return 0.0
        return np.mean([
            c.verb_fraction for c in self.layer_classifications.values()
        ])

    @property
    def mean_noun_fraction(self) -> float:
        """Mean noun fraction across layers."""
        if not self.layer_classifications:
            return 0.0
        return np.mean([
            c.noun_fraction for c in self.layer_classifications.values()
        ])


class VerbNounDimensionClassifier:
    """
    Classifies embedding dimensions as Verb (skill/trajectory) or Noun (knowledge/position).

    The key insight: different dimensions in the hidden state have different behaviors.
    - Verb dimensions: high variance across computational gates (skills/operations)
    - Noun dimensions: low variance, stable across semantic primes (knowledge anchors)

    For model merging:
    - Verb dimensions should trust Source (skill donor) more
    - Noun dimensions should trust Target (knowledge base) more
    """

    @staticmethod
    def compute_noun_stability(
        prime_activations: np.ndarray,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute noun stability for each dimension.

        Noun stability = 1 - coefficient of variation of prime activations.
        High stability means primes activate this dimension consistently (it's a concept anchor).

        Args:
            prime_activations: [num_primes, hidden_dim] matrix
            epsilon: Small value to prevent division by zero

        Returns:
            [hidden_dim] array of stability scores (0-1)
        """
        # Compute mean and variance along the prime axis
        mean = np.mean(prime_activations, axis=0)
        variance = np.var(prime_activations, axis=0)

        # Coefficient of variation = std / |mean|
        std = np.sqrt(variance + epsilon)
        abs_mean = np.abs(mean) + epsilon
        coeff_var = std / abs_mean

        # Stability = 1 - normalized_coeffVar (clamped to [0, 1])
        # Normalize by a reasonable max coeff_var (e.g., 2.0)
        normalized_coeff_var = np.minimum(coeff_var / 2.0, 1.0)
        stability = 1.0 - normalized_coeff_var

        return stability.astype(np.float32)

    @staticmethod
    def compute_verb_variance(gate_activations: np.ndarray) -> np.ndarray:
        """
        Compute verb variance for each dimension.

        Verb variance = variance of gate activations across different gates.
        High variance means different operations use this dimension differently (it's a trajectory).

        Args:
            gate_activations: [num_gates, hidden_dim] matrix

        Returns:
            [hidden_dim] array of variance scores
        """
        variance = np.var(gate_activations, axis=0)
        return variance.astype(np.float32)

    @classmethod
    def classify(
        cls,
        prime_activations: np.ndarray,
        gate_activations: np.ndarray,
        config: Optional[VerbNounConfig] = None,
    ) -> VerbNounClassification:
        """
        Classify dimensions based on semantic prime and computational gate activations.

        Args:
            prime_activations: [num_primes, hidden_dim] matrix
            gate_activations: [num_gates, hidden_dim] matrix
            config: Classification configuration

        Returns:
            Classification result with per-dimension blend weights
        """
        if config is None:
            config = VerbNounConfig.default()

        hidden_dim = prime_activations.shape[1]

        logger.debug(
            "Classifying %d dimensions: primes=%d, gates=%d",
            hidden_dim,
            prime_activations.shape[0],
            gate_activations.shape[0],
        )

        # Compute per-dimension statistics
        noun_stabilities = cls.compute_noun_stability(
            prime_activations, config.epsilon
        )
        verb_variances = cls.compute_verb_variance(gate_activations)

        # Classify each dimension
        dimension_results: list[DimensionResult] = []
        alpha_vector = np.full(hidden_dim, config.mixed_alpha, dtype=np.float32)
        verb_count = 0
        noun_count = 0
        mixed_count = 0

        for dim in range(hidden_dim):
            noun_stab = float(noun_stabilities[dim])
            verb_var = float(verb_variances[dim])
            ratio = verb_var / (noun_stab + config.epsilon)

            if ratio > config.verb_threshold:
                classification = DimensionClass.VERB
                alpha = config.verb_alpha
                verb_count += 1
            elif ratio < config.noun_threshold:
                classification = DimensionClass.NOUN
                alpha = config.noun_alpha
                noun_count += 1
            else:
                classification = DimensionClass.MIXED
                alpha = config.mixed_alpha
                mixed_count += 1

            result = DimensionResult(
                dimension=dim,
                classification=classification,
                noun_stability=noun_stab,
                verb_variance=verb_var,
                ratio=ratio,
                alpha=alpha,
            )
            dimension_results.append(result)
            alpha_vector[dim] = alpha

        # Compute aggregate statistics
        mean_noun_stability = float(np.mean(noun_stabilities))
        mean_verb_variance = float(np.mean(verb_variances))
        overall_ratio = mean_verb_variance / (mean_noun_stability + config.epsilon)

        logger.info(
            "Classification complete: %d verb (%.1f%%), %d noun (%.1f%%), %d mixed. "
            "Ratio=%.2f",
            verb_count,
            100.0 * verb_count / max(hidden_dim, 1),
            noun_count,
            100.0 * noun_count / max(hidden_dim, 1),
            mixed_count,
            overall_ratio,
        )

        return VerbNounClassification(
            dimensions=dimension_results,
            alpha_vector=alpha_vector,
            verb_count=verb_count,
            noun_count=noun_count,
            mixed_count=mixed_count,
            mean_noun_stability=mean_noun_stability,
            mean_verb_variance=mean_verb_variance,
            overall_ratio=overall_ratio,
        )

    @classmethod
    def classify_from_fingerprints(
        cls,
        fingerprints: list[dict],
        prime_probe_ids: set[str],
        gate_probe_ids: set[str],
        layer_indices: list[int],
        hidden_dim: int,
        config: Optional[VerbNounConfig] = None,
    ) -> LayerVerbNounClassification:
        """
        Classify dimensions from probe fingerprints.

        Separates fingerprints into prime probes (for noun stability)
        and gate probes (for verb variance), then classifies each layer.

        Args:
            fingerprints: List of fingerprint dicts with probe_id and activated_dimensions
            prime_probe_ids: Set of probe IDs that are semantic primes
            gate_probe_ids: Set of probe IDs that are computational gates
            layer_indices: Which layers to classify
            hidden_dim: Size of hidden dimension
            config: Classification configuration

        Returns:
            Multi-layer classification result
        """
        if config is None:
            config = VerbNounConfig.default()

        # Separate fingerprints by type
        prime_fps = [
            fp for fp in fingerprints
            if fp.get("probe_id", "") in prime_probe_ids
        ]
        gate_fps = [
            fp for fp in fingerprints
            if fp.get("probe_id", "") in gate_probe_ids
        ]

        logger.debug(
            "Found %d prime fingerprints and %d gate fingerprints",
            len(prime_fps),
            len(gate_fps),
        )

        layer_classifications: dict[int, VerbNounClassification] = {}
        alpha_vectors_by_layer: dict[int, np.ndarray] = {}

        for layer_idx in layer_indices:
            # Build activation matrices for this layer
            prime_activations = cls._build_activation_matrix(
                prime_fps, layer_idx, hidden_dim
            )
            gate_activations = cls._build_activation_matrix(
                gate_fps, layer_idx, hidden_dim
            )

            if prime_activations.shape[0] < 5 or gate_activations.shape[0] < 5:
                logger.warning(
                    "Layer %d: insufficient probes (primes=%d, gates=%d), using default",
                    layer_idx,
                    prime_activations.shape[0],
                    gate_activations.shape[0],
                )
                # Use default mixed alpha for all dimensions
                alpha_vectors_by_layer[layer_idx] = np.full(
                    hidden_dim, config.mixed_alpha, dtype=np.float32
                )
                continue

            classification = cls.classify(
                prime_activations, gate_activations, config
            )
            layer_classifications[layer_idx] = classification
            alpha_vectors_by_layer[layer_idx] = classification.alpha_vector

        return LayerVerbNounClassification(
            layer_classifications=layer_classifications,
            alpha_vectors_by_layer=alpha_vectors_by_layer,
        )

    @staticmethod
    def _build_activation_matrix(
        fingerprints: list[dict],
        layer_idx: int,
        hidden_dim: int,
    ) -> np.ndarray:
        """
        Build activation matrix for a single layer from fingerprints.

        Args:
            fingerprints: Fingerprints to process
            layer_idx: Layer to extract
            hidden_dim: Size of hidden dimension

        Returns:
            [num_probes, hidden_dim] activation matrix
        """
        rows = []
        layer_key = str(layer_idx)

        for fp in fingerprints:
            activated_dims = fp.get("activated_dimensions", {})
            if layer_key not in activated_dims:
                continue

            # Initialize row with zeros
            row = np.zeros(hidden_dim, dtype=np.float32)

            for dim_data in activated_dims[layer_key]:
                dim_idx = dim_data.get("dimension", dim_data.get("index", -1))
                activation = dim_data.get("activation", 0.0)

                if 0 <= dim_idx < hidden_dim:
                    row[dim_idx] = activation

            rows.append(row)

        if not rows:
            return np.zeros((1, hidden_dim), dtype=np.float32)

        return np.array(rows, dtype=np.float32)

    @staticmethod
    def modulate_weights(
        correlation_weights: np.ndarray,
        vn_classification: VerbNounClassification,
        strength: float = 0.3,
    ) -> np.ndarray:
        """
        Modulate existing blend weights with verb/noun classification.

        Combines correlation-based weights with verb/noun signals:
            final_weight = (1 - strength) * correlation_weight + strength * vn_weight

        IMPORTANT: This BLENDS the signals rather than replacing. The default
        strength (0.3) means correlation weights dominate while verb/noun
        provides subtle adjustment.

        Args:
            correlation_weights: Existing per-dimension weights (from correlation analysis)
            vn_classification: Verb/noun classification result
            strength: Modulation strength (0 = ignore v/n, 1 = full v/n effect)
                      Default 0.3 keeps correlation as primary signal

        Returns:
            Modulated blend weights
        """
        if len(correlation_weights) != len(vn_classification.alpha_vector):
            logger.warning(
                "Weight count mismatch: %d vs %d",
                len(correlation_weights),
                len(vn_classification.alpha_vector),
            )
            return correlation_weights

        strength = np.clip(strength, 0.0, 1.0)

        return (
            (1.0 - strength) * correlation_weights +
            strength * vn_classification.alpha_vector
        ).astype(np.float32)


# Probe type detection helpers
def get_prime_probe_ids() -> set[str]:
    """
    Get IDs of semantic prime probes.

    These are probes that test stable knowledge representations.
    """
    from modelcypher.core.domain.agents.unified_atlas import (
        AtlasDomain,
        UnifiedAtlasInventory,
    )

    # Linguistic and mental probes are primarily "noun" (knowledge) probes
    prime_domains = {
        AtlasDomain.LINGUISTIC,
        AtlasDomain.MENTAL,
        AtlasDomain.AFFECTIVE,
        AtlasDomain.RELATIONAL,
        AtlasDomain.TEMPORAL,
        AtlasDomain.SPATIAL,
    }

    probes = UnifiedAtlasInventory.all_probes()
    return {
        probe.probe_id
        for probe in probes
        if probe.domain in prime_domains
    }


def get_gate_probe_ids() -> set[str]:
    """
    Get IDs of computational gate probes.

    These are probes that test skill/operation representations.
    """
    from modelcypher.core.domain.agents.unified_atlas import (
        AtlasDomain,
        UnifiedAtlasInventory,
    )

    # Computational and structural probes are primarily "verb" (skill) probes
    gate_domains = {
        AtlasDomain.COMPUTATIONAL,
        AtlasDomain.STRUCTURAL,
        AtlasDomain.LOGICAL,
        AtlasDomain.MATHEMATICAL,
    }

    probes = UnifiedAtlasInventory.all_probes()
    return {
        probe.probe_id
        for probe in probes
        if probe.domain in gate_domains
    }


def modulate_with_confidence(
    base_alpha: np.ndarray,
    vn_classification: VerbNounClassification,
    modulation_strength: float = 0.3,
    min_confidence: float = 0.3,
) -> np.ndarray:
    """
    Modulate alpha with verb/noun signal weighted by classification confidence.

    Dimensions with high classification confidence are modulated more strongly.
    Dimensions with low confidence retain more of the base alpha.

    Args:
        base_alpha: Base per-dimension alpha from correlation or other source
        vn_classification: Verb/noun classification with per-dimension results
        modulation_strength: Maximum modulation strength (scaled by confidence)
        min_confidence: Minimum confidence to apply any modulation

    Returns:
        Modulated alpha vector
    """
    if len(base_alpha) != len(vn_classification.alpha_vector):
        logger.warning(
            "Dimension mismatch: base=%d, vn=%d",
            len(base_alpha),
            len(vn_classification.alpha_vector),
        )
        return base_alpha

    result = base_alpha.copy()

    for dim_result in vn_classification.dimensions:
        dim = dim_result.dimension
        if dim >= len(result):
            continue

        # Skip low-confidence classifications
        if dim_result.classification == DimensionClass.MIXED:
            continue

        # Compute effective strength based on how extreme the ratio is
        # More extreme ratio = higher confidence in classification
        if dim_result.classification == DimensionClass.VERB:
            # For verb: higher ratio = higher confidence
            confidence = min(1.0, dim_result.ratio / 5.0)  # Saturate at ratio=5
        else:
            # For noun: lower ratio = higher confidence
            confidence = min(1.0, 1.0 / (dim_result.ratio + 0.1))

        if confidence < min_confidence:
            continue

        # Scale modulation by confidence
        effective_strength = modulation_strength * confidence

        # Blend toward VN alpha
        result[dim] = (
            (1.0 - effective_strength) * base_alpha[dim]
            + effective_strength * dim_result.alpha
        )

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def summarize_verb_noun_classification(
    classification: VerbNounClassification,
) -> dict:
    """
    Generate summary statistics for verb/noun classification.

    Args:
        classification: Classification result

    Returns:
        Summary dictionary
    """
    return {
        "total_dimensions": classification.total_dimensions,
        "verb_count": classification.verb_count,
        "noun_count": classification.noun_count,
        "mixed_count": classification.mixed_count,
        "verb_fraction": classification.verb_fraction,
        "noun_fraction": classification.noun_fraction,
        "mean_noun_stability": classification.mean_noun_stability,
        "mean_verb_variance": classification.mean_verb_variance,
        "overall_ratio": classification.overall_ratio,
        "mean_alpha": float(np.mean(classification.alpha_vector)),
        "alpha_std": float(np.std(classification.alpha_vector)),
    }
