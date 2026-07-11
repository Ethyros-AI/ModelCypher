#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Scan the repo for product-maintenance doctrine drift.

This is a candidate generator, not an auto-adjudicator. It groups suspicious
lines by doctrine category so the audit report can classify them as:
keep (exact alternate), delete, or refactor.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("src", "tests", "docs", "scripts")
SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "data",
    "results",
}
SKIP_FILES = {
    "docs/research/PRODUCT-MAINTENANCE-AUDIT-2026-03.md",
    "scripts/report_doctrine_audit.py",
    "tests/repo/test_doctrine_audit.py",
    "tests/domain/training/test_mission_alignment_training.py",
}


@dataclass(frozen=True)
class Candidate:
    category: str
    suggested_severity: str
    path: str
    line: int
    pattern: str
    snippet: str


CATEGORY_RULES: dict[str, dict[str, object]] = {
    "permissive_unknown_default": {
        "severity": "P0",
        "patterns": (
            r'get\("capability_transfer",\s*"true"\)',
            r'get\("training_objective",\s*"unknown"\)',
            r"default true for",
            r"permissive fallback",
            r"return 1\.0\s*#\s*fallback",
            r"returns 1\.0\s*\(permissive fallback\)",
        ),
    },
    "backward_compat_shim": {
        "severity": "P1",
        "patterns": (
            r"backward compatibility",
            r"backwards compatibility",
            r"compatibility shim",
            r"retained for compatibility",
            r"convenience functions for",
            r"re-exports? .*compatibility",
        ),
    },
    "legacy_alias_or_deprecated": {
        "severity": "P1",
        "patterns": (
            r"\blegacy\b",
            r"\bdeprecated\b",
            r"alias retained",
            r"old interface",
        ),
    },
    "override_or_bypass": {
        "severity": "P2",
        "patterns": (
            r"\boverride\b",
            r"\bbypass\w*\b",
            r"allow_.*invalid",
            r"research_allow_",
            r"lr_override",
            r"scale_bound_override",
            r"rank_override",
        ),
    },
    "heuristic_or_product_language": {
        "severity": "P2",
        "patterns": (
            r"\bheuristic\w*\b",
            r"\breasonable\b",
            r"\bgood enough\b",
            r"\bfine\b",
            r"non-breaking",
            r"user[- ]friendly",
            r"for users",
        ),
    },
}


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SCAN_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in SKIP_FILES:
                continue
            if path.suffix.lower() not in {".py", ".md"}:
                continue
            files.append(path)
    return sorted(files)


def collect_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in _iter_files():
        rel = path.relative_to(ROOT).as_posix()
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line_no, line in enumerate(lines, start=1):
            lowered = line.lower()
            for category, rule in CATEGORY_RULES.items():
                severity = str(rule["severity"])
                for pattern in rule["patterns"]:
                    if re.search(pattern, lowered, re.IGNORECASE):
                        candidates.append(
                            Candidate(
                                category=category,
                                suggested_severity=severity,
                                path=rel,
                                line=line_no,
                                pattern=str(pattern),
                                snippet=line.strip(),
                            ),
                        )
    return sorted(
        candidates,
        key=lambda c: (c.suggested_severity, c.category, c.path, c.line, c.pattern),
    )


def _render_markdown(candidates: list[Candidate]) -> str:
    lines = [
        "# Doctrine Audit Candidate Inventory",
        "",
        "Candidate scan only. Manual adjudication is required before any keep/delete decision.",
        "",
    ]
    if not candidates:
        lines.append("No candidates found.")
        return "\n".join(lines)

    categories = sorted({candidate.category for candidate in candidates})
    for category in categories:
        group = [candidate for candidate in candidates if candidate.category == category]
        severity = group[0].suggested_severity
        lines.append(f"## {severity} — {category}")
        lines.append("")
        for candidate in group:
            lines.append(
                f"- `{candidate.path}:{candidate.line}` — `{candidate.snippet}` "
                f"(pattern: `{candidate.pattern}`)"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of markdown.",
    )
    parser.add_argument(
        "--fail-on-p0p1",
        action="store_true",
        help="Exit non-zero if any P0/P1 candidates are found.",
    )
    args = parser.parse_args()

    candidates = collect_candidates()
    if args.json:
        print(json.dumps([asdict(candidate) for candidate in candidates], indent=2))
    else:
        print(_render_markdown(candidates))

    if args.fail_on_p0p1 and any(
        candidate.suggested_severity in {"P0", "P1"} for candidate in candidates
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
