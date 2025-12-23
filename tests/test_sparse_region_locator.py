from __future__ import annotations

from modelcypher.core.domain.geometry.sparse_region_locator import (
    Configuration,
    LayerActivationStats,
    SparseRegionLocator,
)


def test_sparse_region_locator_analysis() -> None:
    locator = SparseRegionLocator(
        Configuration(base_rank=10, sparsity_threshold=0.3, max_skip_layers=2, use_dare_alignment=False)
    )
    domain_stats = [
        LayerActivationStats(layer_index=0, mean_activation=0.5, max_activation=0.5, activation_variance=0.0, prompt_count=2),
        LayerActivationStats(layer_index=1, mean_activation=0.2, max_activation=0.2, activation_variance=0.0, prompt_count=2),
    ]
    baseline_stats = [
        LayerActivationStats(layer_index=0, mean_activation=1.0, max_activation=1.0, activation_variance=0.0, prompt_count=2),
        LayerActivationStats(layer_index=1, mean_activation=0.4, max_activation=0.4, activation_variance=0.0, prompt_count=2),
    ]

    result = locator.analyze(domain_stats=domain_stats, baseline_stats=baseline_stats, domain="test")
    assert result.sparse_layers == [0, 1]
    assert result.skip_layers == []
    assert result.recommendation.overall_rank == 7
    assert result.recommendation.alpha == 14

    from_activations = locator.analyze_from_activations(
        domain_activations=[{0: 0.5, 1: 0.2}],
        baseline_activations=[{0: 1.0, 1: 0.4}],
        domain="test",
    )
    assert from_activations.sparse_layers == [0, 1]


def test_sparse_region_locator_configuration_defaults() -> None:
    """Configuration has sensible defaults."""
    config = Configuration()
    assert config.base_rank > 0
    assert 0.0 < config.sparsity_threshold < 1.0
    assert config.max_skip_layers >= 0


def test_layer_activation_stats_creation() -> None:
    """LayerActivationStats can be created with all fields."""
    stats = LayerActivationStats(
        layer_index=5,
        mean_activation=0.75,
        max_activation=1.5,
        activation_variance=0.1,
        prompt_count=10,
    )
    assert stats.layer_index == 5
    assert stats.mean_activation == 0.75
    assert stats.max_activation == 1.5
    assert stats.activation_variance == 0.1
    assert stats.prompt_count == 10


def test_sparse_region_locator_high_sparsity() -> None:
    """High sparsity layers are correctly identified."""
    locator = SparseRegionLocator(
        Configuration(base_rank=10, sparsity_threshold=0.5, max_skip_layers=2)
    )
    # Domain has much lower activation than baseline = high sparsity
    domain_stats = [
        LayerActivationStats(layer_index=0, mean_activation=0.1, max_activation=0.1, activation_variance=0.0, prompt_count=2),
    ]
    baseline_stats = [
        LayerActivationStats(layer_index=0, mean_activation=1.0, max_activation=1.0, activation_variance=0.0, prompt_count=2),
    ]
    result = locator.analyze(domain_stats=domain_stats, baseline_stats=baseline_stats, domain="test")
    assert 0 in result.sparse_layers


def test_sparse_region_locator_recommendation_properties() -> None:
    """Recommendation has valid properties."""
    locator = SparseRegionLocator(Configuration(base_rank=10))
    domain_stats = [
        LayerActivationStats(layer_index=0, mean_activation=0.5, max_activation=0.5, activation_variance=0.0, prompt_count=2),
    ]
    baseline_stats = [
        LayerActivationStats(layer_index=0, mean_activation=1.0, max_activation=1.0, activation_variance=0.0, prompt_count=2),
    ]
    result = locator.analyze(domain_stats=domain_stats, baseline_stats=baseline_stats, domain="test")
    assert result.recommendation.overall_rank > 0
    assert result.recommendation.alpha >= result.recommendation.overall_rank
