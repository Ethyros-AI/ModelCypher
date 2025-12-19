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
