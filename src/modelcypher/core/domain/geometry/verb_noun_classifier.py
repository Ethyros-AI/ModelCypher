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

"""Verb-Noun Dimension Classifier.

Classifies embedding dimensions as Verb (skill/trajectory) or Noun (knowledge/position)
for per-dimension alpha blending during model merges.

Notes
-----
For each dimension d at layer l:

    NounStability(l,d) = 1 - CoeffVar[primeActivation(p,l,d) for p in primes]
    VerbVariance(l,d)  = Var[gateActivation(g,l,d) for g in gates]
    Ratio(l,d) = VerbVariance / (NounStability + epsilon)
    Alpha(l,d) = sigmoid(log(Ratio) * scale)

References
----------
.. [1] Wierzbicka, "Semantics: Primes and Universals", 1996.
.. [2] Schönfinkel, "Über die Bausteine der mathematischen Logik", 1924.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from modelcypher.core.domain._backend import get_default_backend

logger = logging.getLogger(__name__)


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
    """Configuration for verb/noun analysis.

    Attributes
    ----------
    epsilon : float
        Prevent division by zero in ratio computation.
    alpha_scale : float
        Scale for sigmoid(log(ratio) * scale) transformation.
    min_activation_variance : float
        Minimum variance to consider a dimension active.
    """

    epsilon: float = 1e-6
    alpha_scale: float = 1.5
    min_activation_variance: float = 1e-8

    @classmethod
    def default(cls) -> VerbNounConfig:
        """Default configuration."""
        return cls()

    @classmethod
    def conservative(cls) -> VerbNounConfig:
        """Conservative: gentler alpha transitions."""
        return cls(alpha_scale=1.0)

    @classmethod
    def aggressive(cls) -> VerbNounConfig:
        """Aggressive: sharper alpha transitions."""
        return cls(alpha_scale=2.5)


@dataclass
class DimensionResult:
    """Per-dimension verb/noun analysis result.

    Attributes
    ----------
    dimension : int
        Index of the dimension.
    noun_stability : float
        Stability score in [0, 1]. Higher = stable across semantic primes.
    verb_variance : float
        Variance across computational gates.
    ratio : float
        VerbVariance / NounStability.
    alpha : float
        Geometry-derived blend weight in [0, 1].
    """

    dimension: int
    noun_stability: float
    verb_variance: float
    ratio: float
    alpha: float


@dataclass
class VerbNounClassification:
    """Full verb/noun analysis result for all dimensions.

    Attributes
    ----------
    dimensions : list[DimensionResult]
        Per-dimension analysis results.
    alpha_vector : Array
        Per-dimension blend weights derived from geometry, shape (hidden_dim,).
    mean_noun_stability : float
        Mean stability across dimensions.
    mean_verb_variance : float
        Mean variance across dimensions.
    overall_ratio : float
        mean_verb_variance / mean_noun_stability.
    """

    dimensions: list[DimensionResult]
    alpha_vector: "Array"
    mean_noun_stability: float
    mean_verb_variance: float
    overall_ratio: float

    @property
    def total_dimensions(self) -> int:
        """Total dimension count."""
        return len(self.dimensions)


@dataclass
class LayerVerbNounClassification:
    """Multi-layer analysis result."""

    layer_classifications: dict[int, VerbNounClassification]
    """Per-layer analysis results."""

    alpha_vectors_by_layer: dict[int, "Array"]
    """Per-layer alpha vectors derived from geometry."""

    @property
    def mean_overall_ratio(self) -> float:
        """Mean ratio across layers."""
        if not self.layer_classifications:
            return 1.0
        ratios = [c.overall_ratio for c in self.layer_classifications.values()]
        return sum(ratios) / len(ratios)


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
        """Classify dimensions as verb-like or noun-like.

        Parameters
        ----------
        prime_activations : Array
            Activations from semantic primes, shape (num_primes, hidden_dim).
        gate_activations : Array
            Activations from computational gates, shape (num_gates, hidden_dim).
        config : VerbNounConfig, optional
            Analysis configuration.

        Returns
        -------
        VerbNounClassification
            Per-dimension ratios and geometry-derived alpha vector.
        """
        if config is None:
            config = VerbNounConfig.default()

        backend = get_default_backend()
        hidden_dim = prime_activations.shape[1]

        logger.debug(
            "Analyzing %d dimensions: primes=%d, gates=%d",
            hidden_dim,
            prime_activations.shape[0],
            gate_activations.shape[0],
        )

        # Compute per-dimension statistics
        noun_stabilities = cls.compute_noun_stability(prime_activations, config.epsilon)
        verb_variances = cls.compute_verb_variance(gate_activations)

        # Compute per-dimension results
        # Alpha is derived from variance ratio geometry
        dimension_results: list[DimensionResult] = []
        alpha_vector = backend.ones((hidden_dim,)) * 0.5

        for dim in range(hidden_dim):
            noun_stab = float(backend.to_numpy(noun_stabilities[dim]))
            verb_var = float(backend.to_numpy(verb_variances[dim]))
            ratio = verb_var / (noun_stab + config.epsilon)

            # Derive alpha from the ratio using geometric transformation
            # sigmoid(log(ratio) * scale) maps ratio to alpha in [0, 1]
            alpha = ratio_to_alpha(ratio, config.alpha_scale, config.epsilon)

            result = DimensionResult(
                dimension=dim,
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
            "Analysis complete: %d dimensions, overall ratio=%.2f",
            hidden_dim,
            overall_ratio,
        )

        return VerbNounClassification(
            dimensions=dimension_results,
            alpha_vector=alpha_vector,
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
    min_ratio_extremity: float = 0.3,
) -> "Array":
    """
    Modulate alpha with verb/noun signal weighted by ratio extremity.

    Dimensions with extreme ratios (far from 1.0) are modulated more strongly.
    Dimensions with ratio ≈ 1.0 retain more of the base alpha.

    The ratio IS the confidence: more extreme = higher confidence.

    Args:
        base_alpha: Base per-dimension alpha from correlation or other source
        vn_classification: Verb/noun analysis with per-dimension results
        modulation_strength: Maximum modulation strength (scaled by extremity)
        min_ratio_extremity: Minimum ratio extremity to apply modulation

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

        # Compute extremity: how far is the ratio from 1.0?
        # ratio=1 → extremity=0 (mixed, low confidence)
        # ratio=5 → extremity≈0.8 (verb-like, high confidence)
        # ratio=0.2 → extremity≈0.8 (noun-like, high confidence)
        log_ratio = abs(math.log(max(dim_result.ratio, 1e-6)))
        extremity = min(1.0, log_ratio / 2.0)  # Saturate at log_ratio=2 (ratio ≈ 7.4)

        if extremity < min_ratio_extremity:
            continue

        # Scale modulation by extremity
        effective_strength = modulation_strength * extremity

        # Blend toward geometry-derived alpha
        result[dim] = (1.0 - effective_strength) * base_alpha[
            dim
        ] + effective_strength * dim_result.alpha

    return backend.clip(result, 0.0, 1.0)


def summarize_verb_noun_classification(
    classification: VerbNounClassification,
) -> dict:
    """Generate summary statistics for verb/noun analysis.

    Parameters
    ----------
    classification : VerbNounClassification
        Analysis result.

    Returns
    -------
    dict
        Summary with total_dimensions, mean_noun_stability, mean_verb_variance,
        overall_ratio, mean_alpha, alpha_std.
    """
    backend = get_default_backend()

    mean_alpha = float(backend.to_numpy(backend.mean(classification.alpha_vector)))

    # Compute std manually
    alpha_mean = backend.mean(classification.alpha_vector)
    variance = backend.mean((classification.alpha_vector - alpha_mean) ** 2)
    alpha_std = float(backend.to_numpy(backend.sqrt(variance)))

    return {
        "total_dimensions": classification.total_dimensions,
        "mean_noun_stability": classification.mean_noun_stability,
        "mean_verb_variance": classification.mean_verb_variance,
        "overall_ratio": classification.overall_ratio,
        "mean_alpha": mean_alpha,
        "alpha_std": alpha_std,
    }
