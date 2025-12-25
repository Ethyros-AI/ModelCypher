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
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain._backend import get_default_backend

logger = logging.getLogger(__name__)


class DimensionClass(str, Enum):
    """Classification of a single dimension."""

    VERB = "verb"  # Skill dimension - high variance, trust Source (skill donor)
    NOUN = "noun"  # Knowledge dimension - high stability, trust Target (knowledge base)
    MIXED = "mixed"  # Mixed dimension - use standard blending


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    return math.exp(x) / (1 + math.exp(x))


def ratio_to_alpha(ratio: float, scale: float = 1.5, epsilon: float = 1e-6) -> float:
    """Derive alpha from variance ratio using geometric transformation.

    The ratio (VerbVariance / NounStability) encodes how verb-like vs noun-like
    a dimension is. This function transforms it to an alpha value:

    - High ratio → verb-like → high alpha (trust source for skills)
    - Low ratio → noun-like → low alpha (trust target for knowledge)
    - ratio=1 → mixed → alpha=0.5 (balanced)

    Mathematical foundation:
    - log(ratio) maps: ratio=1 → 0, ratio>1 → positive, ratio<1 → negative
    - sigmoid bounds the output to [0, 1]
    - scale controls the steepness of the transition

    Args:
        ratio: VerbVariance / NounStability
        scale: Controls transition sharpness (default 1.5)
        epsilon: Prevents log(0)

    Returns:
        Alpha in [0, 1]
    """
    # Protect against zero/negative ratios
    safe_ratio = max(ratio, epsilon)

    # Log transform: ratio=1 → 0, ratio>1 → positive, ratio<1 → negative
    log_ratio = math.log(safe_ratio)

    # Sigmoid with scale: maps to [0, 1] with controlled steepness
    # scale=1.5: ratio=2.0 → alpha≈0.73, ratio=0.5 → alpha≈0.27
    # scale=2.5: ratio=2.0 → alpha≈0.85, ratio=0.5 → alpha≈0.15
    alpha = _sigmoid(log_ratio * scale)

    return alpha


@dataclass(frozen=True)
class VerbNounConfig:
    """Configuration for verb/noun classification.

    IMPORTANT: Alpha values are derived from the variance ratio geometry, not hardcoded.
    The ratio (VerbVariance / NounStability) directly determines alpha:
    - High ratio → verb-like → high alpha (trust source for skills)
    - Low ratio → noun-like → low alpha (trust target for knowledge)
    - ratio=1 → mixed → alpha=0.5

    The sigmoid(log(ratio) * scale) transformation provides a smooth, bounded mapping.
    """

    # Threshold above which a dimension is classified as Verb
    # Ratio = VerbVariance / NounStability. Higher ratio → more verb-like.
    verb_threshold: float = 2.0

    # Threshold below which a dimension is classified as Noun
    noun_threshold: float = 0.5

    # Epsilon to prevent division by zero
    epsilon: float = 1e-6

    # Scale factor for ratio→alpha transformation
    # Higher scale = sharper transition between verb and noun alphas
    # sigmoid(log(ratio) * scale): scale=1.5 gives moderate transition
    alpha_scale: float = 1.5

    # Strength of verb/noun modulation (0 = disabled, 1 = full effect)
    # This interpolates between correlation-based and verb/noun-based weights
    modulation_strength: float = 0.7

    # Minimum activation variance to consider a dimension active
    min_activation_variance: float = 1e-8

    @classmethod
    def default(cls) -> VerbNounConfig:
        """Default configuration."""
        return cls()

    @classmethod
    def conservative(cls) -> VerbNounConfig:
        """Conservative: less aggressive verb/noun separation.

        Uses higher thresholds and lower scale for gentler alpha transitions.
        """
        return cls(
            verb_threshold=3.0,
            noun_threshold=0.3,
            alpha_scale=1.0,  # Gentler transition
            modulation_strength=0.5,
        )

    @classmethod
    def aggressive(cls) -> VerbNounConfig:
        """Aggressive: strong verb/noun separation.

        Uses lower thresholds and higher scale for sharper alpha transitions.
        """
        return cls(
            verb_threshold=1.5,
            noun_threshold=0.7,
            alpha_scale=2.5,  # Sharper transition
            modulation_strength=0.9,
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
    alpha_vector: "Array"  # Per-dimension blend weights
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
    alpha_vectors_by_layer: dict[int, "Array"]

    @property
    def mean_verb_fraction(self) -> float:
        """Mean verb fraction across layers."""
        if not self.layer_classifications:
            return 0.0
        fractions = [c.verb_fraction for c in self.layer_classifications.values()]
        return sum(fractions) / len(fractions)

    @property
    def mean_noun_fraction(self) -> float:
        """Mean noun fraction across layers."""
        if not self.layer_classifications:
            return 0.0
        fractions = [c.noun_fraction for c in self.layer_classifications.values()]
        return sum(fractions) / len(fractions)


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
        prime_activations: "Array",
        epsilon: float = 1e-6,
    ) -> "Array":
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
        backend = get_default_backend()

        # Compute mean and variance along the prime axis
        mean = backend.mean(prime_activations, axis=0)
        variance = backend.var(prime_activations, axis=0)

        # Coefficient of variation = std / |mean|
        std = backend.sqrt(variance + epsilon)
        abs_mean = backend.abs(mean) + epsilon
        coeff_var = std / abs_mean

        # Stability = 1 - normalized_coeffVar (clamped to [0, 1])
        # Normalize by a reasonable max coeff_var (e.g., 2.0)
        normalized_coeff_var = backend.clip(coeff_var / 2.0, 0.0, 1.0)
        stability = 1.0 - normalized_coeff_var

        return stability

    @staticmethod
    def compute_verb_variance(gate_activations: "Array") -> "Array":
        """
        Compute verb variance for each dimension.

        Verb variance = variance of gate activations across different gates.
        High variance means different operations use this dimension differently (it's a trajectory).

        Args:
            gate_activations: [num_gates, hidden_dim] matrix

        Returns:
            [hidden_dim] array of variance scores
        """
        backend = get_default_backend()
        variance = backend.var(gate_activations, axis=0)
        return variance

    @classmethod
    def classify(
        cls,
        prime_activations: "Array",
        gate_activations: "Array",
        config: VerbNounConfig | None = None,
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

        backend = get_default_backend()
        hidden_dim = prime_activations.shape[1]

        logger.debug(
            "Classifying %d dimensions: primes=%d, gates=%d",
            hidden_dim,
            prime_activations.shape[0],
            gate_activations.shape[0],
        )

        # Compute per-dimension statistics
        noun_stabilities = cls.compute_noun_stability(prime_activations, config.epsilon)
        verb_variances = cls.compute_verb_variance(gate_activations)

        # Classify each dimension
        # Alpha is derived from variance ratio geometry, not fixed values
        dimension_results: list[DimensionResult] = []
        alpha_vector = backend.ones((hidden_dim,)) * 0.5  # Default to 0.5 (will be overwritten)
        verb_count = 0
        noun_count = 0
        mixed_count = 0

        for dim in range(hidden_dim):
            noun_stab = float(backend.to_numpy(noun_stabilities[dim]))
            verb_var = float(backend.to_numpy(verb_variances[dim]))
            ratio = verb_var / (noun_stab + config.epsilon)

            # Derive alpha from the ratio itself using geometric transformation
            # sigmoid(log(ratio) * scale) maps ratio to alpha in [0, 1]
            alpha = ratio_to_alpha(ratio, config.alpha_scale, config.epsilon)

            # Classification based on ratio thresholds (for counting/logging)
            if ratio > config.verb_threshold:
                classification = DimensionClass.VERB
                verb_count += 1
            elif ratio < config.noun_threshold:
                classification = DimensionClass.NOUN
                noun_count += 1
            else:
                classification = DimensionClass.MIXED
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
        mean_noun_stability = float(backend.to_numpy(backend.mean(noun_stabilities)))
        mean_verb_variance = float(backend.to_numpy(backend.mean(verb_variances)))
        overall_ratio = mean_verb_variance / (mean_noun_stability + config.epsilon)

        logger.info(
            "Classification complete: %d verb (%.1f%%), %d noun (%.1f%%), %d mixed. Ratio=%.2f",
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
        config: VerbNounConfig | None = None,
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

        backend = get_default_backend()

        # Separate fingerprints by type
        prime_fps = [fp for fp in fingerprints if fp.get("probe_id", "") in prime_probe_ids]
        gate_fps = [fp for fp in fingerprints if fp.get("probe_id", "") in gate_probe_ids]

        logger.debug(
            "Found %d prime fingerprints and %d gate fingerprints",
            len(prime_fps),
            len(gate_fps),
        )

        layer_classifications: dict[int, VerbNounClassification] = {}
        alpha_vectors_by_layer: dict[int, "Array"] = {}

        for layer_idx in layer_indices:
            # Build activation matrices for this layer
            prime_activations = cls._build_activation_matrix(prime_fps, layer_idx, hidden_dim)
            gate_activations = cls._build_activation_matrix(gate_fps, layer_idx, hidden_dim)

            if prime_activations.shape[0] < 5 or gate_activations.shape[0] < 5:
                logger.warning(
                    "Layer %d: insufficient probes (primes=%d, gates=%d), using default",
                    layer_idx,
                    prime_activations.shape[0],
                    gate_activations.shape[0],
                )
                # Use balanced alpha (0.5) when we can't classify - equivalent to ratio=1
                alpha_vectors_by_layer[layer_idx] = backend.ones((hidden_dim,)) * 0.5
                continue

            classification = cls.classify(prime_activations, gate_activations, config)
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
    ) -> "Array":
        """
        Build activation matrix for a single layer from fingerprints.

        Args:
            fingerprints: Fingerprints to process
            layer_idx: Layer to extract
            hidden_dim: Size of hidden dimension

        Returns:
            [num_probes, hidden_dim] activation matrix
        """
        backend = get_default_backend()
        rows = []
        layer_key = str(layer_idx)

        for fp in fingerprints:
            activated_dims = fp.get("activated_dimensions", {})
            if layer_key not in activated_dims:
                continue

            # Initialize row with zeros
            row = [0.0] * hidden_dim

            for dim_data in activated_dims[layer_key]:
                dim_idx = dim_data.get("dimension", dim_data.get("index", -1))
                activation = dim_data.get("activation", 0.0)

                if 0 <= dim_idx < hidden_dim:
                    row[dim_idx] = activation

            rows.append(row)

        if not rows:
            return backend.zeros((1, hidden_dim))

        return backend.array(rows)

    @staticmethod
    def modulate_weights(
        correlation_weights: "Array",
        vn_classification: VerbNounClassification,
        strength: float = 0.3,
    ) -> "Array":
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
        backend = get_default_backend()

        if len(correlation_weights) != len(vn_classification.alpha_vector):
            logger.warning(
                "Weight count mismatch: %d vs %d",
                len(correlation_weights),
                len(vn_classification.alpha_vector),
            )
            return correlation_weights

        strength = max(0.0, min(1.0, strength))

        return (
            (1.0 - strength) * correlation_weights + strength * vn_classification.alpha_vector
        )


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
    return {probe.probe_id for probe in probes if probe.domain in prime_domains}


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
    return {probe.probe_id for probe in probes if probe.domain in gate_domains}


def modulate_with_confidence(
    base_alpha: "Array",
    vn_classification: VerbNounClassification,
    modulation_strength: float = 0.3,
    min_confidence: float = 0.3,
) -> "Array":
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
    backend = get_default_backend()

    if len(base_alpha) != len(vn_classification.alpha_vector):
        logger.warning(
            "Dimension mismatch: base=%d, vn=%d",
            len(base_alpha),
            len(vn_classification.alpha_vector),
        )
        return base_alpha

    result = backend.array(backend.to_numpy(base_alpha).copy())

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
        result[dim] = (1.0 - effective_strength) * base_alpha[
            dim
        ] + effective_strength * dim_result.alpha

    return backend.clip(result, 0.0, 1.0)


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
    backend = get_default_backend()

    mean_alpha = float(backend.to_numpy(backend.mean(classification.alpha_vector)))

    # Compute std manually
    alpha_mean = backend.mean(classification.alpha_vector)
    variance = backend.mean((classification.alpha_vector - alpha_mean) ** 2)
    alpha_std = float(backend.to_numpy(backend.sqrt(variance)))

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
        "mean_alpha": mean_alpha,
        "alpha_std": alpha_std,
    }
