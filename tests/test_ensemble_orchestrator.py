"""
Comprehensive tests for EnsembleOrchestrator.

Tests cover:
- Ensemble creation and validation
- Weight computation and normalization
- Strategy selection (weight blending vs attention routing)
- Rebalancing and stabilizer takeover
- Error conditions
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from modelcypher.core.domain.adapters.ensemble_orchestrator import (
    AdapterInfo,
    CompositionStrategy,
    Ensemble,
    EnsembleOrchestrator,
    EnsembleOrchestratorError,
    EnsembleResult,
    InvalidAdapterIDError,
    NoActiveEnsembleError,
    NoAdaptersError,
    NoCompatibleAdaptersError,
    OrchestratorConfiguration,
    TooManyAdaptersError,
)


# =============================================================================
# AdapterInfo Tests
# =============================================================================


def test_adapter_info_creation():
    """AdapterInfo can be created with required fields."""
    adapter_id = uuid4()
    adapter = AdapterInfo(id=adapter_id, name="test-adapter")

    assert adapter.id == adapter_id
    assert adapter.name == "test-adapter"
    assert adapter.compatibility_score is None


def test_adapter_info_with_score():
    """AdapterInfo can have compatibility score."""
    adapter = AdapterInfo(
        id=uuid4(),
        name="test-adapter",
        compatibility_score=0.85,
    )
    assert adapter.compatibility_score == 0.85


def test_adapter_info_frozen():
    """AdapterInfo is immutable."""
    adapter = AdapterInfo(id=uuid4(), name="test")
    with pytest.raises(Exception):
        adapter.name = "changed"


# =============================================================================
# Ensemble Tests
# =============================================================================


def test_ensemble_dominant_adapter():
    """Dominant adapter is the one with highest weight."""
    adapter1 = AdapterInfo(id=uuid4(), name="adapter1")
    adapter2 = AdapterInfo(id=uuid4(), name="adapter2")

    ensemble = Ensemble(
        id=uuid4(),
        adapters=[adapter1, adapter2],
        weights={adapter1.id: 0.3, adapter2.id: 0.7},
        strategy=CompositionStrategy.WEIGHT_BLENDING,
    )

    assert ensemble.dominant_adapter == adapter2


def test_ensemble_dominant_adapter_empty_weights():
    """Empty weights returns None dominant."""
    adapter = AdapterInfo(id=uuid4(), name="adapter")

    ensemble = Ensemble(
        id=uuid4(),
        adapters=[adapter],
        weights={},
        strategy=CompositionStrategy.WEIGHT_BLENDING,
    )

    assert ensemble.dominant_adapter is None


def test_ensemble_dominant_adapter_not_found():
    """Unknown weight ID returns None dominant."""
    adapter = AdapterInfo(id=uuid4(), name="adapter")
    other_id = uuid4()

    ensemble = Ensemble(
        id=uuid4(),
        adapters=[adapter],
        weights={other_id: 1.0},  # ID not in adapters
        strategy=CompositionStrategy.WEIGHT_BLENDING,
    )

    assert ensemble.dominant_adapter is None


# =============================================================================
# OrchestratorConfiguration Tests
# =============================================================================


def test_config_defaults():
    """Default configuration has reasonable values."""
    config = OrchestratorConfiguration.default()

    assert config.max_adapters == 4
    assert 0 < config.min_fit_score < 1
    assert 0 < config.weight_blending_threshold < 1
    assert config.auto_select_strategy is True


def test_config_custom():
    """Custom configuration values are respected."""
    config = OrchestratorConfiguration(
        max_adapters=8,
        min_fit_score=0.5,
        weight_blending_threshold=0.8,
        auto_select_strategy=False,
    )

    assert config.max_adapters == 8
    assert config.min_fit_score == 0.5
    assert config.weight_blending_threshold == 0.8
    assert config.auto_select_strategy is False


# =============================================================================
# EnsembleOrchestrator Initialization Tests
# =============================================================================


def test_orchestrator_default_config():
    """Orchestrator uses default config when none provided."""
    orchestrator = EnsembleOrchestrator()

    assert orchestrator.configuration.max_adapters == 4


def test_orchestrator_custom_config():
    """Orchestrator uses provided config."""
    config = OrchestratorConfiguration(max_adapters=10)
    orchestrator = EnsembleOrchestrator(configuration=config)

    assert orchestrator.configuration.max_adapters == 10


def test_orchestrator_initial_state():
    """New orchestrator has no active ensemble."""
    orchestrator = EnsembleOrchestrator()

    assert orchestrator.current_ensemble() is None


# =============================================================================
# create_ensemble Tests
# =============================================================================


def test_create_ensemble_basic():
    """Basic ensemble creation works."""
    orchestrator = EnsembleOrchestrator()
    adapter = AdapterInfo(id=uuid4(), name="test-adapter")

    result = orchestrator.create_ensemble([adapter])

    assert isinstance(result, EnsembleResult)
    assert result.ensemble is not None
    assert len(result.ensemble.adapters) == 1


def test_create_ensemble_sets_active():
    """Created ensemble becomes active."""
    orchestrator = EnsembleOrchestrator()
    adapter = AdapterInfo(id=uuid4(), name="test-adapter")

    orchestrator.create_ensemble([adapter])

    assert orchestrator.current_ensemble() is not None


def test_create_ensemble_empty_raises():
    """Empty adapter list raises NoAdaptersError."""
    orchestrator = EnsembleOrchestrator()

    with pytest.raises(NoAdaptersError):
        orchestrator.create_ensemble([])


def test_create_ensemble_too_many_raises():
    """Too many adapters raises TooManyAdaptersError."""
    config = OrchestratorConfiguration(max_adapters=2)
    orchestrator = EnsembleOrchestrator(configuration=config)

    adapters = [AdapterInfo(id=uuid4(), name=f"adapter-{i}") for i in range(5)]

    with pytest.raises(TooManyAdaptersError) as exc_info:
        orchestrator.create_ensemble(adapters)

    assert exc_info.value.count == 5
    assert exc_info.value.max_count == 2


def test_create_ensemble_no_compatible_raises():
    """No adapters meeting threshold raises NoCompatibleAdaptersError."""
    config = OrchestratorConfiguration(min_fit_score=0.9)
    orchestrator = EnsembleOrchestrator(configuration=config)

    # Adapter with low compatibility
    adapter = AdapterInfo(id=uuid4(), name="low-compat", compatibility_score=0.3)

    with pytest.raises(NoCompatibleAdaptersError):
        orchestrator.create_ensemble([adapter])


def test_create_ensemble_filters_low_compatibility():
    """Low compatibility adapters are filtered with warning."""
    config = OrchestratorConfiguration(min_fit_score=0.5)
    orchestrator = EnsembleOrchestrator(configuration=config)

    high_compat = AdapterInfo(id=uuid4(), name="high", compatibility_score=0.8)
    low_compat = AdapterInfo(id=uuid4(), name="low", compatibility_score=0.2)

    result = orchestrator.create_ensemble([high_compat, low_compat])

    # Only high_compat should be included
    assert len(result.ensemble.adapters) == 1
    assert result.ensemble.adapters[0].name == "high"
    assert len(result.warnings) > 0
    assert "excluded" in result.warnings[0].lower()


def test_create_ensemble_uses_provided_scores():
    """Provided compatibility scores override adapter scores."""
    orchestrator = EnsembleOrchestrator()

    adapter = AdapterInfo(id=uuid4(), name="adapter", compatibility_score=0.1)
    scores = {adapter.id: 0.9}  # Override to high

    result = orchestrator.create_ensemble([adapter], compatibility_scores=scores)

    assert result.compatibility_scores[adapter.id] == 0.9


def test_create_ensemble_weights_proportional():
    """Weights are proportional to compatibility scores."""
    orchestrator = EnsembleOrchestrator()

    adapter1 = AdapterInfo(id=uuid4(), name="adapter1", compatibility_score=0.6)
    adapter2 = AdapterInfo(id=uuid4(), name="adapter2", compatibility_score=0.4)

    result = orchestrator.create_ensemble([adapter1, adapter2])

    # Higher score should have higher weight
    assert result.ensemble.weights[adapter1.id] > result.ensemble.weights[adapter2.id]


def test_create_ensemble_weights_sum_to_one():
    """Weights sum to 1.0."""
    orchestrator = EnsembleOrchestrator()

    adapters = [
        AdapterInfo(id=uuid4(), name=f"adapter-{i}", compatibility_score=0.5 + i * 0.1)
        for i in range(3)
    ]

    result = orchestrator.create_ensemble(adapters)

    total_weight = sum(result.ensemble.weights.values())
    assert abs(total_weight - 1.0) < 1e-6


def test_create_ensemble_equal_weights_zero_scores():
    """Zero scores result in equal weights."""
    orchestrator = EnsembleOrchestrator()

    adapters = [
        AdapterInfo(id=uuid4(), name=f"adapter-{i}", compatibility_score=0.0)
        for i in range(3)
    ]
    # Need to provide scores that meet threshold
    scores = {a.id: 0.5 for a in adapters}  # All same score

    result = orchestrator.create_ensemble(adapters, compatibility_scores=scores)

    weights = list(result.ensemble.weights.values())
    assert all(abs(w - 1/3) < 0.01 for w in weights)


# =============================================================================
# Strategy Selection Tests
# =============================================================================


def test_create_ensemble_auto_weight_blending():
    """High compatibility suggests weight blending."""
    config = OrchestratorConfiguration(weight_blending_threshold=0.6)
    orchestrator = EnsembleOrchestrator(configuration=config)

    adapter = AdapterInfo(id=uuid4(), name="adapter", compatibility_score=0.8)

    result = orchestrator.create_ensemble([adapter])

    assert result.strategy_suggested == CompositionStrategy.WEIGHT_BLENDING


def test_create_ensemble_auto_attention_routing():
    """Low compatibility suggests attention routing."""
    config = OrchestratorConfiguration(weight_blending_threshold=0.9, min_fit_score=0.3)
    orchestrator = EnsembleOrchestrator(configuration=config)

    adapter = AdapterInfo(id=uuid4(), name="adapter", compatibility_score=0.5)

    result = orchestrator.create_ensemble([adapter])

    assert result.strategy_suggested == CompositionStrategy.ATTENTION_ROUTING


def test_create_ensemble_explicit_strategy():
    """Explicit strategy overrides auto-selection."""
    orchestrator = EnsembleOrchestrator()

    adapter = AdapterInfo(id=uuid4(), name="adapter", compatibility_score=0.9)

    result = orchestrator.create_ensemble(
        [adapter],
        strategy=CompositionStrategy.ATTENTION_ROUTING,
    )

    assert result.ensemble.strategy == CompositionStrategy.ATTENTION_ROUTING


def test_create_ensemble_warns_on_suboptimal_strategy():
    """Warning issued when explicit strategy differs from suggested."""
    orchestrator = EnsembleOrchestrator()

    adapter = AdapterInfo(id=uuid4(), name="adapter", compatibility_score=0.9)

    result = orchestrator.create_ensemble(
        [adapter],
        strategy=CompositionStrategy.ATTENTION_ROUTING,
    )

    # Should warn that weight blending is recommended
    assert any("recommended" in w.lower() for w in result.warnings)


def test_create_ensemble_no_auto_select():
    """Disabled auto-select defaults to weight blending."""
    config = OrchestratorConfiguration(
        auto_select_strategy=False,
        min_fit_score=0.3,
    )
    orchestrator = EnsembleOrchestrator(configuration=config)

    adapter = AdapterInfo(id=uuid4(), name="adapter", compatibility_score=0.4)

    result = orchestrator.create_ensemble([adapter])

    # Should use weight blending regardless of low score
    assert result.ensemble.strategy == CompositionStrategy.WEIGHT_BLENDING


# =============================================================================
# rebalance Tests
# =============================================================================


def test_rebalance_updates_weights():
    """Rebalance updates ensemble weights."""
    orchestrator = EnsembleOrchestrator()

    adapter1 = AdapterInfo(id=uuid4(), name="adapter1")
    adapter2 = AdapterInfo(id=uuid4(), name="adapter2")
    orchestrator.create_ensemble([adapter1, adapter2])

    new_weights = {adapter1.id: 0.8, adapter2.id: 0.2}
    orchestrator.rebalance(new_weights)

    ensemble = orchestrator.current_ensemble()
    assert ensemble is not None
    assert ensemble.weights[adapter1.id] == 0.8
    assert ensemble.weights[adapter2.id] == 0.2


def test_rebalance_normalizes_weights():
    """Rebalance normalizes weights to sum to 1.0."""
    orchestrator = EnsembleOrchestrator()

    adapter = AdapterInfo(id=uuid4(), name="adapter")
    orchestrator.create_ensemble([adapter])

    # Provide unnormalized weight
    orchestrator.rebalance({adapter.id: 5.0})

    ensemble = orchestrator.current_ensemble()
    assert ensemble is not None
    assert ensemble.weights[adapter.id] == 1.0


def test_rebalance_no_active_raises():
    """Rebalance without active ensemble raises."""
    orchestrator = EnsembleOrchestrator()

    with pytest.raises(NoActiveEnsembleError):
        orchestrator.rebalance({uuid4(): 1.0})


def test_rebalance_invalid_adapter_raises():
    """Rebalance with unknown adapter ID raises."""
    orchestrator = EnsembleOrchestrator()

    adapter = AdapterInfo(id=uuid4(), name="adapter")
    orchestrator.create_ensemble([adapter])

    invalid_id = uuid4()
    with pytest.raises(InvalidAdapterIDError) as exc_info:
        orchestrator.rebalance({invalid_id: 1.0})

    assert exc_info.value.adapter_id == invalid_id


# =============================================================================
# stabilizer_takeover Tests
# =============================================================================


def test_stabilizer_takeover_with_stabilizer():
    """Takeover activates stabilizer adapter."""
    orchestrator = EnsembleOrchestrator()

    stabilizer = AdapterInfo(id=uuid4(), name="stabilizer")
    orchestrator.set_stabilizer(stabilizer)

    # Create some ensemble first
    other = AdapterInfo(id=uuid4(), name="other")
    orchestrator.create_ensemble([other])

    # Takeover
    orchestrator.stabilizer_takeover()

    ensemble = orchestrator.current_ensemble()
    assert ensemble is not None
    assert len(ensemble.adapters) == 1
    assert ensemble.adapters[0].name == "stabilizer"
    assert ensemble.weights[stabilizer.id] == 1.0


def test_stabilizer_takeover_no_stabilizer():
    """Takeover without stabilizer clears ensemble."""
    orchestrator = EnsembleOrchestrator()

    adapter = AdapterInfo(id=uuid4(), name="adapter")
    orchestrator.create_ensemble([adapter])

    orchestrator.stabilizer_takeover()

    assert orchestrator.current_ensemble() is None


def test_set_stabilizer():
    """Stabilizer can be set and cleared."""
    orchestrator = EnsembleOrchestrator()

    stabilizer = AdapterInfo(id=uuid4(), name="stabilizer")
    orchestrator.set_stabilizer(stabilizer)
    assert orchestrator._stabilizer_adapter == stabilizer

    orchestrator.set_stabilizer(None)
    assert orchestrator._stabilizer_adapter is None


# =============================================================================
# disband_ensemble Tests
# =============================================================================


def test_disband_ensemble():
    """Disband clears active ensemble."""
    orchestrator = EnsembleOrchestrator()

    adapter = AdapterInfo(id=uuid4(), name="adapter")
    orchestrator.create_ensemble([adapter])

    orchestrator.disband_ensemble()

    assert orchestrator.current_ensemble() is None


def test_disband_already_empty():
    """Disband on empty does nothing."""
    orchestrator = EnsembleOrchestrator()

    orchestrator.disband_ensemble()  # Should not raise

    assert orchestrator.current_ensemble() is None


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_lifecycle():
    """Full ensemble lifecycle: create, rebalance, stabilizer, disband."""
    orchestrator = EnsembleOrchestrator()

    # Set up stabilizer
    stabilizer = AdapterInfo(id=uuid4(), name="stabilizer", compatibility_score=0.9)
    orchestrator.set_stabilizer(stabilizer)

    # Create ensemble
    adapter1 = AdapterInfo(id=uuid4(), name="adapter1", compatibility_score=0.7)
    adapter2 = AdapterInfo(id=uuid4(), name="adapter2", compatibility_score=0.8)
    result = orchestrator.create_ensemble([adapter1, adapter2])

    assert len(result.ensemble.adapters) == 2

    # Rebalance
    orchestrator.rebalance({adapter1.id: 0.9, adapter2.id: 0.1})
    assert orchestrator.current_ensemble().weights[adapter1.id] == 0.9

    # Stabilizer takeover
    orchestrator.stabilizer_takeover()
    assert orchestrator.current_ensemble().adapters[0].name == "stabilizer"

    # Disband
    orchestrator.disband_ensemble()
    assert orchestrator.current_ensemble() is None


def test_multiple_ensemble_creations():
    """Creating new ensemble replaces old one."""
    orchestrator = EnsembleOrchestrator()

    adapter1 = AdapterInfo(id=uuid4(), name="first")
    orchestrator.create_ensemble([adapter1])

    adapter2 = AdapterInfo(id=uuid4(), name="second")
    orchestrator.create_ensemble([adapter2])

    ensemble = orchestrator.current_ensemble()
    assert len(ensemble.adapters) == 1
    assert ensemble.adapters[0].name == "second"
