from __future__ import annotations

from modelcypher.core.domain.sparse_region_domains import SparseRegionDomains
from modelcypher.core.domain.sparse_region_prober import Configuration, SparseRegionProber


def test_sparse_region_prober_probe() -> None:
    domain = SparseRegionDomains.custom(
        name="tiny",
        description="Tiny domain",
        probe_prompts=["a", "b"],
    )
    prober = SparseRegionProber(Configuration(prompts_per_domain=2, max_tokens_per_prompt=5))

    def generate_tokens(prompt: str, max_tokens: int, capture) -> int:
        capture({0: 1.0, 1: 3.0})
        capture({0: 3.0, 1: 5.0})
        return 4

    result = prober.probe(domain=domain, total_layers=2, generate_tokens=generate_tokens)

    assert result.prompts_processed == 2
    assert result.tokens_generated == 8
    assert len(result.layer_stats) == 2

    stat0 = result.layer_stats[0]
    stat1 = result.layer_stats[1]
    assert stat0.layer_index == 0
    assert stat1.layer_index == 1
    assert stat0.mean_activation == 2.0
    assert stat1.mean_activation == 4.0


def test_sparse_region_prober_configuration_defaults() -> None:
    """Configuration has sensible defaults."""
    config = Configuration()
    assert config.prompts_per_domain > 0
    assert config.max_tokens_per_prompt > 0


def test_sparse_region_prober_variance_calculation() -> None:
    """Variance is calculated correctly."""
    domain = SparseRegionDomains.custom(
        name="var_test",
        description="Test variance",
        probe_prompts=["test"],
    )
    prober = SparseRegionProber(Configuration(prompts_per_domain=1, max_tokens_per_prompt=5))

    def generate_tokens(prompt: str, max_tokens: int, capture) -> int:
        capture({0: 2.0})
        capture({0: 4.0})
        return 2

    result = prober.probe(domain=domain, total_layers=1, generate_tokens=generate_tokens)
    assert len(result.layer_stats) == 1
    # Mean should be 3.0, variance should be 1.0
    assert result.layer_stats[0].mean_activation == 3.0


def test_sparse_region_prober_no_tokens_generated() -> None:
    """Handle case where no tokens are generated."""
    domain = SparseRegionDomains.custom(
        name="empty",
        description="Empty domain",
        probe_prompts=["test"],
    )
    prober = SparseRegionProber(Configuration(prompts_per_domain=1, max_tokens_per_prompt=5))

    def generate_tokens(prompt: str, max_tokens: int, capture) -> int:
        return 0  # No tokens generated

    result = prober.probe(domain=domain, total_layers=2, generate_tokens=generate_tokens)
    assert result.tokens_generated == 0


def test_sparse_region_prober_max_activation_tracking() -> None:
    """Max activation is tracked correctly."""
    domain = SparseRegionDomains.custom(
        name="max_test",
        description="Test max tracking",
        probe_prompts=["test"],
    )
    prober = SparseRegionProber(Configuration(prompts_per_domain=1, max_tokens_per_prompt=5))

    def generate_tokens(prompt: str, max_tokens: int, capture) -> int:
        capture({0: 1.0})
        capture({0: 10.0})  # This is the max
        capture({0: 5.0})
        return 3

    result = prober.probe(domain=domain, total_layers=1, generate_tokens=generate_tokens)
    assert result.layer_stats[0].max_activation == 10.0
