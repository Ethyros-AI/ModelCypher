"""
Per-neuron sparsity analysis for fine-grained knowledge grafting.

This module extends layer-level sparsity analysis to individual neurons,
enabling identification of sparse neurons suitable for knowledge transfer.

Integrates with:
- HiddenStateExtractor: Captures per-token, per-layer activations
- DomainSignalProfile: Layer-level sparsity scoring
- SparseRegionLocator: Domain comparison logic
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import math

try:
    import mlx.core as mx
except ImportError:
    mx = None  # Allow import for type checking

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NeuronSparsityConfig:
    """Configuration for per-neuron sparsity analysis."""

    activation_threshold: float = 0.01
    """Activation magnitude below this is considered inactive."""

    sparsity_threshold: float = 0.8
    """Neurons with sparsity above this are candidates for grafting."""

    dead_neuron_threshold: float = 0.99
    """Neurons with sparsity above this are considered dead."""

    min_prompts: int = 20
    """Minimum number of prompts for statistical significance."""

    normalize_activations: bool = True
    """Whether to normalize activations per sample before analysis."""

    use_absolute_values: bool = True
    """Whether to use |activation| instead of raw values."""


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
        return self.max_activation < 1e-10

    @property
    def coefficient_of_variation(self) -> float:
        """CV = std / mean, measures relative variability."""
        if self.mean_activation < 1e-10:
            return 0.0
        return math.sqrt(self.activation_variance) / self.mean_activation


@dataclass
class NeuronSparsityMap:
    """Per-neuron sparsity analysis across all layers.

    Provides methods to identify sparse neurons suitable for knowledge grafting.
    """

    stats: Dict[int, List[NeuronStats]]
    """layer_index -> list of NeuronStats for each neuron."""

    config: NeuronSparsityConfig
    """Configuration used for this analysis."""

    total_prompts: int
    """Total number of prompts used in analysis."""

    @property
    def sparse_neurons(self) -> Dict[int, List[int]]:
        """Layer -> list of sparse neuron indices."""
        result: Dict[int, List[int]] = {}
        for layer, neurons in self.stats.items():
            sparse = [
                n.neuron_idx
                for n in neurons
                if n.sparsity_score >= self.config.sparsity_threshold
            ]
            if sparse:
                result[layer] = sparse
        return result

    @property
    def dead_neurons(self) -> Dict[int, List[int]]:
        """Layer -> list of never-activating neuron indices."""
        result: Dict[int, List[int]] = {}
        for layer, neurons in self.stats.items():
            dead = [
                n.neuron_idx
                for n in neurons
                if n.sparsity_score >= self.config.dead_neuron_threshold
            ]
            if dead:
                result[layer] = dead
        return result

    def get_graft_candidates(
        self, threshold: Optional[float] = None
    ) -> Dict[int, List[int]]:
        """Return neurons sparse enough for knowledge grafting.

        Args:
            threshold: Override sparsity threshold (default: config value)

        Returns:
            Dict mapping layer index to list of graftable neuron indices.
        """
        thresh = threshold if threshold is not None else self.config.sparsity_threshold
        result: Dict[int, List[int]] = {}
        for layer, neurons in self.stats.items():
            candidates = [n.neuron_idx for n in neurons if n.sparsity_score >= thresh]
            if candidates:
                result[layer] = candidates
        return result

    def get_layer_summary(self, layer: int) -> Dict[str, float]:
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
        sparse_count = sum(
            1 for s in sparsity_scores if s >= self.config.sparsity_threshold
        )
        dead_count = sum(
            1 for s in sparsity_scores if s >= self.config.dead_neuron_threshold
        )

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

    def summary(self) -> Dict[str, any]:
        """Get overall summary of neuron sparsity analysis."""
        total_neurons = sum(len(neurons) for neurons in self.stats.values())
        total_sparse = sum(len(v) for v in self.sparse_neurons.values())
        total_dead = sum(len(v) for v in self.dead_neurons.values())

        all_sparsity = [
            n.sparsity_score for neurons in self.stats.values() for n in neurons
        ]

        return {
            "num_layers": len(self.stats),
            "total_neurons": total_neurons,
            "total_sparse": total_sparse,
            "sparse_fraction": total_sparse / total_neurons if total_neurons > 0 else 0,
            "total_dead": total_dead,
            "dead_fraction": total_dead / total_neurons if total_neurons > 0 else 0,
            "mean_sparsity": sum(all_sparsity) / len(all_sparsity)
            if all_sparsity
            else 0,
            "total_prompts": self.total_prompts,
            "graft_candidates": sum(
                len(v) for v in self.get_graft_candidates().values()
            ),
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
    _activations: Dict[int, Dict[int, List[float]]] = field(
        default_factory=dict, repr=False
    )
    _sample_count: int = field(default=0, repr=False)

    def add_sample(self, layer_activations: Dict[int, List[float]]) -> None:
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

    def add_batch(
        self, batch_activations: List[Dict[int, List[float]]]
    ) -> None:
        """Add multiple samples at once.

        Args:
            batch_activations: List of per-prompt activation dicts.
        """
        for sample in batch_activations:
            self.add_sample(sample)

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

        stats: Dict[int, List[NeuronStats]] = {}

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
                active_count = sum(
                    1 for v in values if v > self.config.activation_threshold
                )
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
    activations: Dict[int, List[List[float]]],
    config: Optional[NeuronSparsityConfig] = None,
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
        prompt_data: Dict[int, List[float]] = {}
        for layer, layer_acts in activations.items():
            if prompt_idx < len(layer_acts):
                prompt_data[layer] = layer_acts[prompt_idx]
        collector.add_sample(prompt_data)

    return collector.compute_sparsity_map()


def compare_neuron_sparsity(
    source_map: NeuronSparsityMap,
    target_map: NeuronSparsityMap,
) -> Dict[str, any]:
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

    graft_candidates: Dict[int, List[int]] = {}
    collision_neurons: Dict[int, List[int]] = {}
    both_sparse: Dict[int, List[int]] = {}

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
            if n.sparsity_score < source_map.config.sparsity_threshold
        )
        target_active = set(
            n.neuron_idx
            for n in target_map.stats.get(layer, [])
            if n.sparsity_score < target_map.config.sparsity_threshold
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
    specificity_threshold: float = 0.3,
) -> Dict[int, List[Tuple[int, float]]]:
    """Identify neurons that activate specifically for a domain.

    Compares domain activations against baseline to find neurons
    that are unusually active for the domain.

    Args:
        baseline_map: Sparsity from general prompts.
        domain_map: Sparsity from domain-specific prompts.
        specificity_threshold: Minimum sparsity difference for specificity.

    Returns:
        Dict mapping layer to (neuron_idx, specificity_score) tuples.
    """
    domain_specific: Dict[int, List[Tuple[int, float]]] = {}

    for layer in baseline_map.stats:
        if layer not in domain_map.stats:
            continue

        baseline_neurons = {n.neuron_idx: n for n in baseline_map.stats[layer]}
        domain_neurons = {n.neuron_idx: n for n in domain_map.stats[layer]}

        specific_neurons = []
        for neuron_idx, domain_stat in domain_neurons.items():
            baseline_stat = baseline_neurons.get(neuron_idx)
            if baseline_stat is None:
                continue

            # Specificity = baseline_sparsity - domain_sparsity
            # Higher = neuron activates more for domain than baseline
            specificity = baseline_stat.sparsity_score - domain_stat.sparsity_score

            if specificity >= specificity_threshold:
                specific_neurons.append((neuron_idx, specificity))

        if specific_neurons:
            # Sort by specificity descending
            specific_neurons.sort(key=lambda x: -x[1])
            domain_specific[layer] = specific_neurons

    return domain_specific
