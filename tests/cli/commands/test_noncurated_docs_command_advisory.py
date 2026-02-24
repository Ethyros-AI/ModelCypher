# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Advisory command-contract audit for non-curated docs.

This test intentionally does not fail CI on command drift in non-curated docs.
It reports counts to make drift visible while curated docs remain strict.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from typer.testing import CliRunner

from tests.cli.contracts.command_contract import CLIContractValidator, validate_markdown_file


runner = CliRunner()
validator = CLIContractValidator(runner)

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs"
CURATED_RELATIVE = {
    Path("README.md"),
    Path("docs/START-HERE.md"),
    Path("docs/INFERENCE.md"),
    Path("docs/CLI-REFERENCE.md"),
    Path("docs/TRAINING-GUIDE.md"),
}


def _noncurated_markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in DOCS_ROOT.rglob("*.md"):
        rel = path.relative_to(REPO_ROOT)
        if rel in CURATED_RELATIVE:
            continue
        files.append(path)
    return sorted(files)


def test_noncurated_docs_command_contract_advisory() -> None:
    files = _noncurated_markdown_files()
    assert files, "Expected non-curated docs to scan."

    issue_counter: Counter[str] = Counter()
    file_issue_counts: list[tuple[Path, int]] = []
    total_examples = 0
    total_issues = 0

    for path in files:
        examples, issues = validate_markdown_file(path, validator)
        total_examples += len(examples)
        total_issues += len(issues)
        if issues:
            file_issue_counts.append((path, len(issues)))
            for _, issue in issues:
                issue_counter[issue.code] += 1

    print(
        (
            "[advisory] non-curated docs command audit: "
            f"files={len(files)}, examples={total_examples}, issues={total_issues}"
        )
    )
    if file_issue_counts:
        top_files = sorted(file_issue_counts, key=lambda x: x[1], reverse=True)[:10]
        print("[advisory] top files by issue count:")
        for path, count in top_files:
            print(f"  - {path}: {count}")
    if issue_counter:
        print("[advisory] issue code distribution:")
        for code, count in issue_counter.most_common():
            print(f"  - {code}: {count}")
