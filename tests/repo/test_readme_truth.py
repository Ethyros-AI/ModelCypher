"""Contracts between README claims and retained repository evidence."""

from __future__ import annotations

import re
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_review_is_not_older_than_latest_sota_audit() -> None:
    readme_match = re.search(
        r"Current evidence state \(reviewed (?P<date>\d{4}-\d{2}-\d{2})\)",
        _read(README),
    )
    audit_match = re.search(
        r"^\*\*Date:\*\* (?P<date>\d{4}-\d{2}-\d{2})",
        _read(ROOT / "docs/research/SOTA-AUDIT-2026-07.md"),
        re.MULTILINE,
    )
    assert readme_match is not None
    assert audit_match is not None
    assert date.fromisoformat(readme_match["date"]) >= date.fromisoformat(
        audit_match["date"]
    )


def test_readme_pipeline_verdict_matches_retained_artifact() -> None:
    report = _read(ROOT / "docs/research/reports/pipeline_validation/REPORT.md")
    readme = _read(README)
    assert "structural pass count: `5 / 5`" in report
    assert "inference pass count: `3 / 5`" in report
    assert "`all_pass = false`" in report
    assert "structural pass `5/5`, inference pass `3/5`, `all_pass = false`" in readme


def test_readme_g5_verdict_discloses_seed_count_and_failed_gates() -> None:
    report = _read(ROOT / "docs/research/reports/g5_8b_validation_multiseed/REPORT.md")
    readme = _read(README)
    seed_match = re.search(r"tracked seeds in aggregate verdict: `(\d+)`", report)
    assert seed_match is not None
    seed_count = int(seed_match.group(1))
    assert f"`n_seeds={seed_count}`" in readme
    failed = re.findall(r"- `(\w+) = 0`", report)
    assert failed
    assert all(f"`{name}`" in readme for name in failed)
    assert f"[EMPIRICAL: {seed_count} seed]" in readme


def test_readme_quantization_summary_matches_retained_artifact() -> None:
    report = _read(ROOT / "docs/research/reports/quantization_frontier/REPORT.md")
    assert "retained models: `3`" in report
    assert "`all_ppl_improved = true`" in report
    assert "`all_degen_improved = true`" in report
    assert "all 3 retained models" in _read(README)


def test_readme_platform_claims_match_declared_extras_and_parity_policy() -> None:
    pyproject = tomllib.loads(_read(ROOT / "pyproject.toml"))
    extras = pyproject["tool"]["poetry"]["extras"]
    assert extras["cuda"] == ["torch"]
    assert extras["jax"] == ["jax", "jaxlib"]

    readme = _read(README)
    guide = _read(ROOT / "docs/BACKEND-COMPARISON.md")
    for command in (
        "poetry install --extras cuda",
        "poetry install --extras jax",
    ):
        assert command in readme
        assert command in guide
    assert "The MLX path is the only end-to-end product surface today" in readme
    assert "loader/CLI parity partial" in readme
    assert "CI/domain-test fallback only" in guide


def test_readme_research_rows_preserve_evidence_boundaries() -> None:
    readme = _read(README)
    paper_one = _read(ROOT / "papers/paper-1-invariant-semantic-structure.md")
    paper_five = _read(ROOT / "papers/paper-5-semantic-highway.md")

    assert "[PROVEN: fitted probes only]" in readme
    assert "[PROVEN: fitted training probes only]" in paper_one
    assert "held-out and cross-model" in readme
    assert "[EXPLORATORY]" in paper_five
    assert "published-profile replication is pending under WS4.2" in readme


def test_readme_relative_links_resolve() -> None:
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", _read(README))
    relative_links = [
        link.split("#", 1)[0]
        for link in links
        if link and not link.startswith(("http://", "https://", "#", "mailto:"))
    ]
    missing = [link for link in relative_links if not (ROOT / link).exists()]
    assert relative_links
    assert missing == []
