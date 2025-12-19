from __future__ import annotations

from modelcypher.core.domain.sparse_region_domains import (
    DomainCategory,
    ProbeCorpus,
    SparseRegionDomains,
    create_probe_corpora,
)


def test_domain_lookup_and_category() -> None:
    domain = SparseRegionDomains.domain_named("code")
    assert domain is not None
    assert domain.name == "code"

    scientific = SparseRegionDomains.domains_in_category(DomainCategory.scientific)
    assert any(item.name == "math" for item in scientific)


def test_probe_corpus_and_create_corpora() -> None:
    domain = SparseRegionDomains.domain_named("creative")
    assert domain is not None
    corpus = ProbeCorpus(domain=domain, max_prompts=3, shuffle=False)
    assert corpus.count == 3
    assert len(corpus.prompts) == 3

    target, baseline = create_probe_corpora(domain, prompts_per_domain=2)
    assert target.count == 2
    assert baseline.count == 2
