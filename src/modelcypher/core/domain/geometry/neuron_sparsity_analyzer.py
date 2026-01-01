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
Per-neuron sparsity analysis for fine-grained knowledge grafting.

This module extends layer-level sparsity analysis to individual neurons,
enabling identification of sparse neurons suitable for knowledge transfer.

Integrates with:
- HiddenStateExtractor: Captures per-token, per-layer activations
- DomainSignalProfile: Layer-level sparsity scoring
- SparseRegionLocator: Domain comparison logic
"""

import logging
import math
from dataclasses import dataclass, field

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NeuronSparsityConfig:
    """Configuration for per-neuron sparsity analysis.

    Thresholds should be derived from data, not guessed. Use the class method
    `from_activation_distribution()` to calibrate thresholds from actual activations.

    Default thresholds are conservative placeholders that signal the need for
    calibration - they will work but are not optimal for any specific model.
    """

    activation_threshold: float | None = None
    """Activation magnitude below this is considered inactive.

    If None, derived from machine epsilon scaled to activation magnitude.
    A neuron with |activation| < threshold is considered inactive for that sample.
    """

    sparsity_sigma: float = 2.0
    """Standard deviations above mean sparsity to qualify as 'sparse'.

    Used when sparsity_threshold is None to derive threshold from distribution.
    """

    sparsity_threshold: float | None = None
    """Neurons with sparsity above this are candidates for grafting.

    If None, derived as mean + sparsity_sigma * std of the sparsity distribution.
    """

    dead_neuron_sigma: float = 3.0
    """Standard deviations above mean sparsity to qualify as 'dead'.

    Used when dead_neuron_threshold is None.
    """

    dead_neuron_threshold: float | None = None
    """Neurons with sparsity above this are considered dead.

    If None, derived as mean + dead_neuron_sigma * std, clamped to 0.99 max.
    """

    min_prompts: int = 20
    """Minimum number of prompts for statistical significance."""

    normalize_activations: bool = True
    """Whether to normalize activations per sample before analysis."""

    use_absolute_values: bool = True
    """Whether to use |activation| instead of raw values."""

    @classmethod
    def from_activation_distribution(
        cls,
        activations: list[list[float]],
        sparsity_sigma: float = 2.0,
        dead_neuron_sigma: float = 3.0,
    ) -> "NeuronSparsityConfig":
        """Derive thresholds from actual activation distribution.

        This is the preferred way to create a config - let the data tell us
        what 'sparse' means rather than guessing with magic numbers.

        Args:
            activations: Flattened list of activation values from model.
            sparsity_sigma: Std devs above mean for sparse classification.
            dead_neuron_sigma: Std devs above mean for dead classification.

        Returns:
            NeuronSparsityConfig with data-derived thresholds.

        Raises:
            ValueError: If activations list is empty or contains no values.
        """
        if not activations or not any(activations):
            raise ValueError(
                "Cannot derive thresholds from empty activations. "
                "Provide activation data from model inference."
            )

        # Flatten all activations
        all_values = [abs(v) for row in activations for v in row if v is not None]
        if not all_values:
            raise ValueError(
                "Cannot derive thresholds: all activation values are None. "
                "Check activation extraction."
            )

        # Activation threshold: noise floor based on distribution
        # Use median of small values as noise floor, or machine epsilon scaled
        sorted_vals = sorted(all_values)
        n = len(sorted_vals)

        # Get 10th percentile as noise floor estimate
        p10_idx = max(0, int(n * 0.10) - 1)
        noise_floor = sorted_vals[p10_idx]

        # Ensure threshold is at least machine epsilon * max_activation
        max_val = sorted_vals[-1] if sorted_vals else 1.0
        min_threshold = max_val * 1e-6  # 6 orders of magnitude below max
        activation_threshold = max(noise_floor, min_threshold)

        return cls(
            activation_threshold=activation_threshold,
            sparsity_sigma=sparsity_sigma,
            dead_neuron_sigma=dead_neuron_sigma,
            # Leave sparsity/dead thresholds as None - derived during analysis
        )


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class NeuronStats:
    """Statistics for a single neuron across prompts.

    Captures activation patterns to determine if a neuron is sparse
    enough for knowledge grafting.
    """

    layer: int
    """Layer index containing this neuron."""

    neuron_idx: int
    """Index within the layer's hidden dimension."""

    mean_activation: float
    """Mean activation magnitude across all prompts."""

    max_activation: float
    """Maximum activation magnitude observed."""

    min_activation: float
    """Minimum activation magnitude observed."""

    activation_variance: float
    """Variance of activation magnitude across prompts."""

    active_fraction: float
    """Fraction of prompts where |activation| > threshold."""

    prompt_count: int
    """Number of prompts used to compute statistics."""

    @property
    def sparsity_score(self) -> float:
        """Sparsity score: 1 - active_fraction (higher = more sparse)."""
        return 1.0 - self.active_fraction

    @property
    def is_dead(self) -> bool:
        """Whether this neuron never activates."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([self.max_activation]))
        return self.max_activation < eps

    @property
    def coefficient_of_variation(self) -> float:
        """CV = std / mean, measures relative variability."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([self.mean_activation]))
        if self.mean_activation < eps:
            return 0.0
        return math.sqrt(self.activation_variance) / self.mean_activation


@dataclass
class NeuronSparsityMap:
    """Per-neuron sparsity analysis across all layers.

    Provides methods to identify sparse neurons suitable for knowledge grafting.
    Thresholds are derived from the sparsity distribution when not explicitly set.
    """

    stats: dict[int, list[NeuronStats]]
    """layer_index -> list of NeuronStats for each neuron."""

    config: NeuronSparsityConfig
    """Configuration used for this analysis."""

    total_prompts: int
    """Total number of prompts used in analysis."""

    # Cached derived thresholds
    _derived_sparsity_threshold: float | None = None
    _derived_dead_threshold: float | None = None

    def _derive_thresholds(self) -> tuple[float, float]:
        """Derive sparsity and dead thresholds from the actual distribution.

        Uses mean + N*sigma where N is configurable. This lets the data
        tell us what 'sparse' means for this specific model.

        Returns:
            (sparsity_threshold, dead_neuron_threshold)
        """
        if self._derived_sparsity_threshold is not None:
            return self._derived_sparsity_threshold, self._derived_dead_threshold

        # Collect all sparsity scores
        all_scores = [n.sparsity_score for neurons in self.stats.values() for n in neurons]
        if not all_scores:
            raise ValueError(
                "Cannot derive thresholds: no neuron statistics available. "
                "Collect activation data first."
            )

        n = len(all_scores)
        mean = sum(all_scores) / n
        variance = sum((s - mean) ** 2 for s in all_scores) / n
        std = math.sqrt(variance)

        # Derive thresholds: mean + sigma * std_devs
        sparsity_thresh = mean + self.config.sparsity_sigma * std
        dead_thresh = mean + self.config.dead_neuron_sigma * std

        # Clamp to valid range [0, 1]
        self._derived_sparsity_threshold = max(0.0, min(1.0, sparsity_thresh))
        self._derived_dead_threshold = max(0.0, min(1.0, dead_thresh))

        return self._derived_sparsity_threshold, self._derived_dead_threshold

    @property
    def effective_sparsity_threshold(self) -> float:
        """The sparsity threshold in use (explicit or derived)."""
        if self.config.sparsity_threshold is not None:
            return self.config.sparsity_threshold
        return self._derive_thresholds()[0]

    @property
    def effective_dead_threshold(self) -> float:
        """The dead neuron threshold in use (explicit or derived)."""
        if self.config.dead_neuron_threshold is not None:
            return self.config.dead_neuron_threshold
        return self._derive_thresholds()[1]

    @property
    def sparse_neurons(self) -> dict[int, list[int]]:
        """Layer -> list of sparse neuron indices."""
        thresh = self.effective_sparsity_threshold
        result: dict[int, list[int]] = {}
        for layer, neurons in self.stats.items():
            sparse = [n.neuron_idx for n in neurons if n.sparsity_score >= thresh]
            if sparse:
                result[layer] = sparse
        return result

    @property
    def dead_neurons(self) -> dict[int, list[int]]:
        """Layer -> list of never-activating neuron indices."""
        thresh = self.effective_dead_threshold
        result: dict[int, list[int]] = {}
        for layer, neurons in self.stats.items():
            dead = [n.neuron_idx for n in neurons if n.sparsity_score >= thresh]
            if dead:
                result[layer] = dead
        return result

    def get_graft_candidates(self, threshold: float | None = None) -> dict[int, list[int]]:
        """Return neurons sparse enough for knowledge grafting.

        Args:
            threshold: Override sparsity threshold (default: derived from distribution)

        Returns:
            Dict mapping layer index to list of graftable neuron indices.
        """
        thresh = threshold if threshold is not None else self.effective_sparsity_threshold
        result: dict[int, list[int]] = {}
        for layer, neurons in self.stats.items():
            candidates = [n.neuron_idx for n in neurons if n.sparsity_score >= thresh]
            if candidates:
                result[layer] = candidates
        return result

    def get_layer_summary(self, layer: int) -> dict[str, float]:
        """Get summary statistics for a layer.

        Returns:
            Dict with mean_sparsity, sparse_fraction, dead_fraction, etc.
        """
        if layer not in self.stats:
            return {}

        neurons = self.stats[layer]
        total = len(neurons)
        if total == 0:
            return {}

        sparsity_scores = [n.sparsity_score for n in neurons]
        sparse_count = sum(1 for s in sparsity_scores if s >= self.effective_sparsity_threshold)
        dead_count = sum(1 for s in sparsity_scores if s >= self.effective_dead_threshold)

        return {
            "total_neurons": total,
            "mean_sparsity": sum(sparsity_scores) / total,
            "max_sparsity": max(sparsity_scores),
            "min_sparsity": min(sparsity_scores),
            "sparse_count": sparse_count,
            "sparse_fraction": sparse_count / total,
            "dead_count": dead_count,
            "dead_fraction": dead_count / total,
            "mean_activation": sum(n.mean_activation for n in neurons) / total,
        }

    def summary(self) -> dict[str, any]:
        """Get overall summary of neuron sparsity analysis."""
        total_neurons = sum(len(neurons) for neurons in self.stats.values())
        total_sparse = sum(len(v) for v in self.sparse_neurons.values())
        total_dead = sum(len(v) for v in self.dead_neurons.values())

        all_sparsity = [n.sparsity_score for neurons in self.stats.values() for n in neurons]

        # Compute distribution stats for context
        mean_sparsity = sum(all_sparsity) / len(all_sparsity) if all_sparsity else 0
        variance = (
            sum((s - mean_sparsity) ** 2 for s in all_sparsity) / len(all_sparsity)
            if all_sparsity
            else 0
        )
        std_sparsity = math.sqrt(variance)

        return {
            "num_layers": len(self.stats),
            "total_neurons": total_neurons,
            "total_sparse": total_sparse,
            "sparse_fraction": total_sparse / total_neurons if total_neurons > 0 else 0,
            "total_dead": total_dead,
            "dead_fraction": total_dead / total_neurons if total_neurons > 0 else 0,
            "mean_sparsity": mean_sparsity,
            "std_sparsity": std_sparsity,
            "total_prompts": self.total_prompts,
            "graft_candidates": sum(len(v) for v in self.get_graft_candidates().values()),
            "thresholds": {
                "sparsity": self.effective_sparsity_threshold,
                "dead_neuron": self.effective_dead_threshold,
                "sparsity_derived": self.config.sparsity_threshold is None,
                "dead_derived": self.config.dead_neuron_threshold is None,
            },
        }


# =============================================================================
# Activation Collection
# =============================================================================


@dataclass
class NeuronActivationCollector:
    """Collects per-neuron activation statistics across prompts.

    Usage:
        collector = NeuronActivationCollector(config)
        for prompt_activations in all_activations:
            collector.add_sample(prompt_activations)
        sparsity_map = collector.compute_sparsity_map()
    """

    config: NeuronSparsityConfig = field(default_factory=NeuronSparsityConfig)

    # Internal storage: layer -> neuron_idx -> list of activation values
    _activations: dict[int, dict[int, list[float]]] = field(default_factory=dict, repr=False)
    _sample_count: int = field(default=0, repr=False)

    def add_sample(self, layer_activations: dict[int, list[float]]) -> None:
        """Add a single prompt's activations across all layers.

        Args:
            layer_activations: Dict mapping layer_index to activation vector.
                Each activation vector has shape [hidden_dim].
        """
        self._sample_count += 1

        for layer, activations in layer_activations.items():
            if layer not in self._activations:
                self._activations[layer] = {}

            # Process each neuron's activation
            for neuron_idx, activation in enumerate(activations):
                if neuron_idx not in self._activations[layer]:
                    self._activations[layer][neuron_idx] = []

                # Normalize and take absolute value if configured
                value = abs(activation) if self.config.use_absolute_values else activation
                self._activations[layer][neuron_idx].append(value)

    def add_batch(self, batch_activations: list[dict[int, list[float]]]) -> None:
        """Add multiple samples at once.

        Args:
            batch_activations: List of per-prompt activation dicts.
        """
        for sample in batch_activations:
            self.add_sample(sample)

    def _derive_activation_threshold(self) -> float:
        """Derive activation threshold from collected data.

        Uses the 10th percentile of all activation magnitudes as the noise floor,
        with a minimum of machine epsilon scaled to the activation range.

        Returns:
            Derived activation threshold.
        """
        # Collect all activation magnitudes
        all_values = []
        for layer_data in self._activations.values():
            for neuron_values in layer_data.values():
                all_values.extend(neuron_values)

        if not all_values:
            raise ValueError(
                "Cannot derive activation threshold: no activations collected. "
                "Add samples first."
            )

        sorted_vals = sorted(all_values)
        n = len(sorted_vals)

        # 10th percentile as noise floor
        p10_idx = max(0, int(n * 0.10) - 1)
        noise_floor = sorted_vals[p10_idx]

        # Ensure at least 6 orders of magnitude below max
        max_val = sorted_vals[-1] if sorted_vals else 1.0
        min_threshold = max_val * 1e-6

        return max(noise_floor, min_threshold)

    def compute_sparsity_map(self) -> NeuronSparsityMap:
        """Compute neuron sparsity statistics from collected activations.

        Returns:
            NeuronSparsityMap with per-neuron statistics.

        Raises:
            ValueError: If insufficient samples collected.
        """
        if self._sample_count < self.config.min_prompts:
            logger.warning(
                f"Only {self._sample_count} samples collected, "
                f"minimum recommended is {self.config.min_prompts}"
            )

        # Derive activation threshold if not explicitly set
        activation_threshold = self.config.activation_threshold
        if activation_threshold is None:
            activation_threshold = self._derive_activation_threshold()
            logger.debug(f"Derived activation threshold: {activation_threshold:.2e}")

        stats: dict[int, list[NeuronStats]] = {}

        for layer, neuron_data in self._activations.items():
            layer_stats = []

            for neuron_idx, values in neuron_data.items():
                if not values:
                    continue

                # Compute statistics
                n = len(values)
                mean_val = sum(values) / n
                max_val = max(values)
                min_val = min(values)

                # Variance: E[(x - mean)^2]
                variance = sum((v - mean_val) ** 2 for v in values) / n

                # Active fraction: proportion above threshold
                active_count = sum(1 for v in values if v > activation_threshold)
                active_fraction = active_count / n

                neuron_stat = NeuronStats(
                    layer=layer,
                    neuron_idx=neuron_idx,
                    mean_activation=mean_val,
                    max_activation=max_val,
                    min_activation=min_val,
                    activation_variance=variance,
                    active_fraction=active_fraction,
                    prompt_count=n,
                )
                layer_stats.append(neuron_stat)

            # Sort by neuron index for consistent ordering
            layer_stats.sort(key=lambda x: x.neuron_idx)
            stats[layer] = layer_stats

        return NeuronSparsityMap(
            stats=stats,
            config=self.config,
            total_prompts=self._sample_count,
        )

    def clear(self) -> None:
        """Clear collected activations."""
        self._activations.clear()
        self._sample_count = 0


# =============================================================================
# Analysis Functions
# =============================================================================


def compute_neuron_sparsity_map(
    activations: dict[int, list[list[float]]],
    config: NeuronSparsityConfig | None = None,
) -> NeuronSparsityMap:
    """Compute per-neuron sparsity from activation data.

    Args:
        activations: Dict mapping layer_index to list of activation vectors.
            Each inner list is [prompt_idx][neuron_idx].
        config: Analysis configuration.

    Returns:
        NeuronSparsityMap with per-neuron statistics.
    """
    cfg = config or NeuronSparsityConfig()
    collector = NeuronActivationCollector(cfg)

    # Transpose: activations[layer][prompt][neuron] -> per-prompt dicts
    if not activations:
        return NeuronSparsityMap(stats={}, config=cfg, total_prompts=0)

    # Get number of prompts from first layer
    first_layer = next(iter(activations.values()))
    num_prompts = len(first_layer)

    for prompt_idx in range(num_prompts):
        prompt_data: dict[int, list[float]] = {}
        for layer, layer_acts in activations.items():
            if prompt_idx < len(layer_acts):
                prompt_data[layer] = layer_acts[prompt_idx]
        collector.add_sample(prompt_data)

    return collector.compute_sparsity_map()


def compare_neuron_sparsity(
    source_map: NeuronSparsityMap,
    target_map: NeuronSparsityMap,
) -> dict[str, any]:
    """Compare neuron sparsity between source and target models.

    Identifies neurons that are:
    - Sparse in source, active in target (good graft targets)
    - Active in both (collision risk)
    - Sparse in both (unused in both)

    Args:
        source_map: Sparsity analysis of source model.
        target_map: Sparsity analysis of target model.

    Returns:
        Dict with comparison statistics and graft recommendations.
    """
    source_sparse = source_map.sparse_neurons
    target_sparse = target_map.sparse_neurons

    # Find common layers
    common_layers = set(source_sparse.keys()) & set(target_map.stats.keys())

    graft_candidates: dict[int, list[int]] = {}
    collision_neurons: dict[int, list[int]] = {}
    both_sparse: dict[int, list[int]] = {}

    for layer in common_layers:
        source_set = set(source_sparse.get(layer, []))
        target_set = set(target_sparse.get(layer, []))

        # Sparse in target but not source = good for grafting from source
        graft = list(target_set - source_set)
        if graft:
            graft_candidates[layer] = graft

        # Active in both = collision risk
        source_active = set(
            n.neuron_idx
            for n in source_map.stats.get(layer, [])
            if n.sparsity_score < source_map.effective_sparsity_threshold
        )
        target_active = set(
            n.neuron_idx
            for n in target_map.stats.get(layer, [])
            if n.sparsity_score < target_map.effective_sparsity_threshold
        )
        collision = list(source_active & target_active)
        if collision:
            collision_neurons[layer] = collision

        # Sparse in both = unused
        both = list(source_set & target_set)
        if both:
            both_sparse[layer] = both

    total_graft = sum(len(v) for v in graft_candidates.values())
    total_collision = sum(len(v) for v in collision_neurons.values())
    total_both_sparse = sum(len(v) for v in both_sparse.values())

    return {
        "graft_candidates": graft_candidates,
        "collision_neurons": collision_neurons,
        "both_sparse": both_sparse,
        "total_graft_candidates": total_graft,
        "total_collision_neurons": total_collision,
        "total_both_sparse": total_both_sparse,
        "graft_potential": total_graft / (total_graft + total_collision + 1),
    }


def identify_domain_specific_neurons(
    baseline_map: NeuronSparsityMap,
    domain_map: NeuronSparsityMap,
    specificity_threshold: float | None = None,
    specificity_sigma: float = 2.0,
) -> dict[int, list[tuple[int, float]]]:
    """Identify neurons that activate specifically for a domain.

    Compares domain activations against baseline to find neurons
    that are unusually active for the domain.

    Args:
        baseline_map: Sparsity from general prompts.
        domain_map: Sparsity from domain-specific prompts.
        specificity_threshold: Minimum sparsity difference for specificity.
            If None, derived as mean + specificity_sigma * std of all
            specificity scores.
        specificity_sigma: Standard deviations above mean for threshold
            when specificity_threshold is None.

    Returns:
        Dict mapping layer to (neuron_idx, specificity_score) tuples.
    """
    # First pass: compute all specificity scores to derive threshold if needed
    all_specificities: list[float] = []
    layer_specificities: dict[int, list[tuple[int, float]]] = {}

    for layer in baseline_map.stats:
        if layer not in domain_map.stats:
            continue

        baseline_neurons = {n.neuron_idx: n for n in baseline_map.stats[layer]}
        domain_neurons = {n.neuron_idx: n for n in domain_map.stats[layer]}

        layer_scores = []
        for neuron_idx, domain_stat in domain_neurons.items():
            baseline_stat = baseline_neurons.get(neuron_idx)
            if baseline_stat is None:
                continue

            # Specificity = baseline_sparsity - domain_sparsity
            # Higher = neuron activates more for domain than baseline
            specificity = baseline_stat.sparsity_score - domain_stat.sparsity_score
            all_specificities.append(specificity)
            layer_scores.append((neuron_idx, specificity))

        if layer_scores:
            layer_specificities[layer] = layer_scores

    # Derive threshold if not provided
    if specificity_threshold is None:
        if not all_specificities:
            raise ValueError(
                "Cannot derive specificity threshold: no matching neurons between "
                "baseline and domain maps. Check that maps have overlapping layers."
            )
        n = len(all_specificities)
        mean = sum(all_specificities) / n
        variance = sum((s - mean) ** 2 for s in all_specificities) / n
        std = math.sqrt(variance)
        specificity_threshold = mean + specificity_sigma * std
        logger.debug(
            f"Derived specificity threshold: {specificity_threshold:.4f} "
            f"(mean={mean:.4f}, std={std:.4f})"
        )

    # Second pass: filter by threshold
    domain_specific: dict[int, list[tuple[int, float]]] = {}
    for layer, scores in layer_specificities.items():
        specific_neurons = [
            (neuron_idx, specificity)
            for neuron_idx, specificity in scores
            if specificity >= specificity_threshold
        ]
        if specific_neurons:
            # Sort by specificity descending
            specific_neurons.sort(key=lambda x: -x[1])
            domain_specific[layer] = specific_neurons

    return domain_specific
