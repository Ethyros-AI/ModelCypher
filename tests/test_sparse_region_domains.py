from __future__ import annotations

from modelcypher.core.domain.geometry.sparse_region_domains import (
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


def test_domain_not_found_returns_none() -> None:
    """Unknown domain name returns None."""
    domain = SparseRegionDomains.domain_named("nonexistent_domain_xyz")
    assert domain is None


def test_all_domains_have_prompts() -> None:
    """All domains have at least one probe prompt."""
    for category in DomainCategory:
        domains = SparseRegionDomains.domains_in_category(category)
        for domain in domains:
            assert len(domain.probe_prompts) > 0


def test_custom_domain_creation() -> None:
    """Custom domains can be created."""
    domain = SparseRegionDomains.custom(
        name="test_domain",
        description="Test description",
        probe_prompts=["prompt1", "prompt2"],
    )
    assert domain.name == "test_domain"
    assert domain.description == "Test description"
    assert len(domain.probe_prompts) == 2


def test_probe_corpus_shuffle() -> None:
    """Shuffled corpus may have different order."""
    domain = SparseRegionDomains.domain_named("code")
    assert domain is not None
    corpus1 = ProbeCorpus(domain=domain, max_prompts=10, shuffle=False)
    corpus2 = ProbeCorpus(domain=domain, max_prompts=10, shuffle=False)
    # Without shuffle, order should be deterministic
    assert corpus1.prompts == corpus2.prompts


def test_domain_category_enum_values() -> None:
    """All expected category values exist."""
    category_names = [c.value for c in DomainCategory]
    assert "scientific" in category_names
    assert "creative" in category_names
