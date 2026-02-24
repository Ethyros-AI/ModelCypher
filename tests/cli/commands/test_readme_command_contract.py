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

"""README ↔ CLI contract tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tests.cli.contracts.command_contract import (
    CLIContractValidator,
    extract_bash_command_examples,
    tokenize,
    validate_markdown_file,
)

runner = CliRunner()
validator = CLIContractValidator(runner)
README_PATH = Path(__file__).resolve().parents[3] / "README.md"


def test_readme_contains_mc_commands_for_contract() -> None:
    examples, _ = validate_markdown_file(README_PATH, validator)
    assert examples, "Expected at least one `poetry run mc ...` command in README bash blocks."

    expected_anchors = [
        "train run",
        "model info",
        "analyze dimension-profile",
        "analyze lora-svd",
    ]
    for anchor in expected_anchors:
        assert any(anchor in ex.command for ex in examples), f"Missing README CLI anchor: {anchor}"


def test_readme_mc_commands_match_cli_contract() -> None:
    examples, issues = validate_markdown_file(README_PATH, validator)
    assert examples, "No README commands found to validate."

    assert not issues, "\n".join(
        [
            (
                f"{README_PATH}:{example.line_no}: {example.command}\n"
                f"  - {issue.code}: {issue.detail}"
            )
            for example, issue in issues
        ]
    )


def test_contract_rejects_invalid_option_example() -> None:
    bad_line = 'poetry run mc analyze dimension-profile --model /path/to/model --prompt "2+2"'
    issues = validator.validate_command_line(bad_line)
    assert any(issue.code == "OPTION_NOT_IN_HELP" for issue in issues), issues


def test_contract_rejects_missing_required_positional_shape() -> None:
    bad_line = "poetry run mc analyze lora-svd --base /path/to/model"
    issues = validator.validate_command_line(bad_line)
    assert any(issue.code == "MISSING_REQUIRED_POSITIONAL" for issue in issues), issues


def test_tokenize_ignores_inline_comments() -> None:
    tokens = tokenize("poetry run mc --help  # verify install")
    assert tokens == ["poetry", "run", "mc", "--help"]


def test_option_equals_syntax_is_supported() -> None:
    line = (
        "poetry run mc analyze lora-svd /path/to/adapter "
        "--base=/path/to/model --top-k=10"
    )
    issues = validator.validate_command_line(line)
    assert not issues, issues


def test_bash_extractor_supports_line_continuations() -> None:
    text = (
        "```bash\n"
        "poetry run mc train run \\\n"
        "  --model /path/to/model \\\n"
        "  --data /path/to/data.jsonl\n"
        "```\n"
    )
    examples = extract_bash_command_examples(text)
    assert examples
    assert examples[0].command.startswith("poetry run mc train run")
    assert "--model /path/to/model" in examples[0].command
