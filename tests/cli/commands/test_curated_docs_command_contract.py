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

"""Strict command-contract checks for curated user docs."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tests.cli.contracts.command_contract import CLIContractValidator, validate_markdown_file

runner = CliRunner()
validator = CLIContractValidator(runner)

REPO_ROOT = Path(__file__).resolve().parents[3]
CURATED_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "START-HERE.md",
    REPO_ROOT / "docs" / "INFERENCE.md",
    REPO_ROOT / "docs" / "CLI-REFERENCE.md",
    REPO_ROOT / "docs" / "TRAINING-GUIDE.md",
]


def test_curated_docs_have_executable_mc_examples() -> None:
    for path in CURATED_DOCS:
        examples, _ = validate_markdown_file(path, validator)
        assert examples, f"Expected at least one CLI command example in {path}"


def test_curated_docs_command_contract() -> None:
    all_issues: list[str] = []

    for path in CURATED_DOCS:
        _, issues = validate_markdown_file(path, validator)
        for example, issue in issues:
            all_issues.append(
                (
                    f"{path}:{example.line_no}: {example.command}\n"
                    f"  - {issue.code}: {issue.detail}"
                )
            )

    assert not all_issues, "Curated docs command-contract violations:\n" + "\n".join(all_issues)
