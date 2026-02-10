from __future__ import annotations

import logging

import yaml

import modelcypher.core.domain.domains as domains_mod
from modelcypher.core.domain.domains import AtlasDomain


def test_resolve_domain_and_resolve_domains_with_warnings(caplog) -> None:
    domains_mod._load_taxonomy_aliases.cache_clear()

    assert domains_mod.resolve_domain("mathematical") == AtlasDomain.MATHEMATICAL
    assert domains_mod.resolve_domain("MATH") == AtlasDomain.MATHEMATICAL
    assert domains_mod.resolve_domain("unknown-domain") is None

    with caplog.at_level(logging.WARNING):
        resolved = domains_mod.resolve_domains(
            ["math", "relational", "math", "nope"]
        )

    assert resolved == [AtlasDomain.MATHEMATICAL, AtlasDomain.RELATIONAL]
    assert any("Unknown domain" in rec.message for rec in caplog.records)


def test_load_taxonomy_aliases_from_yaml(monkeypatch, tmp_path) -> None:
    pkg_root = tmp_path / "pkg"
    domain_dir = pkg_root / "core" / "domain"
    domain_dir.mkdir(parents=True)
    fake_file = domain_dir / "domains.py"
    fake_file.write_text("# fake", encoding="utf-8")

    taxonomy_dir = pkg_root / "data"
    taxonomy_dir.mkdir(parents=True)
    taxonomy_path = taxonomy_dir / "domain_taxonomy.yaml"
    taxonomy_path.write_text(
        yaml.safe_dump(
            {
                "core_domains": {
                    "mathematical": {
                        "atlas_domain": "MATHEMATICAL",
                        "aliases": ["calcx", "numbersx"],
                    },
                    "custom_logic": {
                        "atlas_domain": "LOGICAL",
                        "aliases": ["logicx"],
                    },
                },
                "composite_domains": {
                    "reasoning": {
                        "components": ["logical", "mathematical"],
                        "aliases": ["reasonx"],
                    },
                    "empty_component": {
                        "components": [],
                        "aliases": ["should_skip"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(domains_mod, "__file__", str(fake_file))
    domains_mod._load_taxonomy_aliases.cache_clear()

    aliases = domains_mod._load_taxonomy_aliases()

    assert aliases["calcx"] == AtlasDomain.MATHEMATICAL
    assert aliases["numbersx"] == AtlasDomain.MATHEMATICAL
    assert aliases["logicx"] == AtlasDomain.LOGICAL
    assert aliases["reasoning"] == AtlasDomain.LOGICAL
    assert aliases["reasonx"] == AtlasDomain.LOGICAL


def test_load_taxonomy_aliases_yaml_failure_falls_back(monkeypatch, tmp_path, caplog) -> None:
    pkg_root = tmp_path / "pkg"
    domain_dir = pkg_root / "core" / "domain"
    domain_dir.mkdir(parents=True)
    fake_file = domain_dir / "domains.py"
    fake_file.write_text("# fake", encoding="utf-8")

    taxonomy_dir = pkg_root / "data"
    taxonomy_dir.mkdir(parents=True)
    taxonomy_path = taxonomy_dir / "domain_taxonomy.yaml"
    taxonomy_path.write_text("core_domains: [", encoding="utf-8")

    monkeypatch.setattr(domains_mod, "__file__", str(fake_file))
    domains_mod._load_taxonomy_aliases.cache_clear()

    with caplog.at_level(logging.WARNING):
        aliases = domains_mod._load_taxonomy_aliases()

    assert aliases["math"] == AtlasDomain.MATHEMATICAL
    assert any("Failed to load domain taxonomy" in rec.message for rec in caplog.records)


def test_list_helpers_return_domains_and_aliases() -> None:
    domains = domains_mod.list_all_domains()
    aliases = domains_mod.list_domain_aliases()

    assert AtlasDomain.MATHEMATICAL in domains
    assert aliases["math"] == AtlasDomain.MATHEMATICAL
