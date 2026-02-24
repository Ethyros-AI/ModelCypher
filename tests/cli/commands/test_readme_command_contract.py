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

"""README ↔ CLI contract tests.

Treats README `poetry run mc ...` examples as an executable interface contract:
- Command path must resolve via `--help`
- Explicit options must exist in command help
- Positional shape must match usage signature
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex

from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()

README_PATH = Path(__file__).resolve().parents[3] / "README.md"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    detail: str


def _read_readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _extract_bash_mc_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_bash_block = False

    for raw in text.splitlines():
        stripped = raw.strip()

        if not in_bash_block:
            if stripped.lower() == "```bash":
                in_bash_block = True
            continue

        if stripped.startswith("```"):
            in_bash_block = False
            continue

        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("poetry run mc"):
            lines.append(stripped)

    return lines


def _tokenize(line: str) -> list[str]:
    return shlex.split(line, comments=True, posix=True)


def _help_result(command_path: list[str], cli_runner: CliRunner):
    return cli_runner.invoke(app, [*command_path, "--help"])


def _help_text(command_path: list[str], cli_runner: CliRunner) -> str:
    result = _help_result(command_path, cli_runner)
    if result.exit_code != 0:
        return ""
    return result.stdout


def _extract_usage_line(help_text: str) -> str:
    match = re.search(r"Usage:\s+([^\n]+)", help_text)
    if not match:
        return ""
    return match.group(1).strip()


def _resolve_command_path(argv: list[str], cli_runner: CliRunner) -> list[str] | None:
    """Resolve command path depth as [], [group], or [group, subcommand]."""
    if not argv or argv[0] != "mc":
        return None

    candidates: list[list[str]] = [[]]
    tail = argv[1:]

    if tail and not tail[0].startswith("-"):
        candidates.append([tail[0]])
        if len(tail) > 1 and not tail[1].startswith("-"):
            candidates.append([tail[0], tail[1]])

    valid: list[list[str]] = []
    for candidate in candidates:
        if _help_result(candidate, cli_runner).exit_code == 0:
            valid.append(candidate)

    if not valid:
        return None

    return max(valid, key=len)


def _split_options_and_positionals(argv_tail: list[str]) -> tuple[list[str], list[str]]:
    options: list[str] = []
    positionals: list[str] = []

    i = 0
    while i < len(argv_tail):
        token = argv_tail[i]

        if token == "--":
            positionals.extend(argv_tail[i + 1 :])
            break

        if token.startswith("--") and token != "--":
            options.append(token.split("=", 1)[0])
            if "=" in token:
                i += 1
                continue
            if i + 1 < len(argv_tail) and not argv_tail[i + 1].startswith("-"):
                i += 2
                continue
            i += 1
            continue

        if token.startswith("-") and token != "-":
            options.append(token)
            if i + 1 < len(argv_tail) and not argv_tail[i + 1].startswith("-"):
                i += 2
                continue
            i += 1
            continue

        positionals.append(token)
        i += 1

    return options, positionals


def _usage_declares_positionals(help_text: str) -> bool:
    usage = _extract_usage_line(help_text)
    if not usage:
        return False

    placeholders = re.findall(r"\b[A-Z][A-Z0-9_-]*\b", usage)
    filtered = [p for p in placeholders if p not in {"OPTIONS", "COMMAND", "ARGS"}]
    return bool(filtered)


def _usage_required_positional_count(help_text: str) -> int:
    usage = _extract_usage_line(help_text)
    if not usage:
        return 0

    placeholders = re.findall(r"\b[A-Z][A-Z0-9_-]*\b", usage)
    optional_placeholders = set(re.findall(r"\[([A-Z][A-Z0-9_-]*)\]", usage))

    required = [
        p
        for p in placeholders
        if p not in {"OPTIONS", "COMMAND", "ARGS"} and p not in optional_placeholders
    ]
    return len(required)


def _known_options(help_text: str) -> set[str]:
    long_options = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", help_text))
    short_options = set(re.findall(r"(?<!\S)-[A-Za-z0-9]{1,3}(?!\S)", help_text))
    return long_options | short_options


def _validate_command_line(line: str, cli_runner: CliRunner) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    tokens = _tokenize(line)

    if len(tokens) < 3 or tokens[:3] != ["poetry", "run", "mc"]:
        return [ValidationIssue("INVALID_PREFIX", f"Line does not start with `poetry run mc`: {line}")]

    argv = tokens[2:]  # ["mc", ...]
    command_path = _resolve_command_path(argv, cli_runner)
    if command_path is None:
        return [ValidationIssue("UNKNOWN_COMMAND_PATH", f"Could not resolve command path: {line}")]

    help_text = _help_text(command_path, cli_runner)
    usage_line = _extract_usage_line(help_text)

    leading_non_option: list[str] = []
    for token in argv[1:]:
        if token.startswith("-"):
            break
        leading_non_option.append(token)

    if len(leading_non_option) > len(command_path) and "COMMAND" in usage_line:
        unknown = leading_non_option[len(command_path)]
        parent = " ".join(command_path) if command_path else "mc"
        return [
            ValidationIssue(
                "UNKNOWN_COMMAND_PATH",
                f"Unknown subcommand `{unknown}` under `{parent}` in line: {line}",
            )
        ]

    tail = argv[1 + len(command_path) :]
    explicit_options, positionals = _split_options_and_positionals(tail)

    known_options = _known_options(help_text)
    for option in explicit_options:
        if option in {"--help", "-h"}:
            continue
        if option not in known_options:
            issues.append(
                ValidationIssue(
                    "OPTION_NOT_IN_HELP",
                    f"Option `{option}` is not present in help for `{ ' '.join(command_path) or 'mc' }`",
                )
            )

    declares_positionals = _usage_declares_positionals(help_text)
    if positionals and not declares_positionals:
        issues.append(
            ValidationIssue(
                "POSITIONAL_SHAPE",
                f"Positional tokens {positionals} provided, but usage declares no positional placeholders",
            )
        )

    required_positional_count = _usage_required_positional_count(help_text)
    if required_positional_count > 0 and len(positionals) < required_positional_count:
        issues.append(
            ValidationIssue(
                "MISSING_REQUIRED_POSITIONAL",
                (
                    f"Usage requires at least {required_positional_count} positional token(s), "
                    f"but got {len(positionals)}"
                ),
            )
        )

    return issues


def test_readme_contains_mc_commands_for_contract() -> None:
    text = _read_readme_text()
    lines = _extract_bash_mc_lines(text)

    assert lines, "Expected at least one `poetry run mc ...` line in README bash blocks."

    expected_anchors = [
        "train run",
        "model info",
        "analyze dimension-profile",
        "analyze lora-svd",
    ]
    for anchor in expected_anchors:
        assert any(anchor in line for line in lines), f"Missing README CLI anchor: {anchor}"


def test_readme_mc_commands_match_cli_contract() -> None:
    text = _read_readme_text()
    lines = _extract_bash_mc_lines(text)

    assert lines, "No `poetry run mc ...` lines found to validate."

    all_issues: list[tuple[str, ValidationIssue]] = []
    for line in lines:
        issues = _validate_command_line(line, runner)
        all_issues.extend((line, issue) for issue in issues)

    assert not all_issues, "\n".join(
        [f"{line}\n  - {issue.code}: {issue.detail}" for line, issue in all_issues]
    )


def test_contract_rejects_invalid_option_example() -> None:
    bad_line = 'poetry run mc analyze dimension-profile --model /path/to/model --prompt "2+2"'
    issues = _validate_command_line(bad_line, runner)

    assert any(issue.code == "OPTION_NOT_IN_HELP" for issue in issues), issues


def test_contract_rejects_missing_required_positional_shape() -> None:
    bad_line = "poetry run mc analyze lora-svd --base /path/to/model"
    issues = _validate_command_line(bad_line, runner)

    assert any(issue.code == "MISSING_REQUIRED_POSITIONAL" for issue in issues), issues


def test_tokenize_ignores_inline_comments() -> None:
    tokens = _tokenize("poetry run mc --help  # verify install")
    assert tokens == ["poetry", "run", "mc", "--help"]


def test_option_equals_syntax_is_supported() -> None:
    line = (
        "poetry run mc analyze lora-svd /path/to/adapter "
        "--base=/path/to/model --top-k=10"
    )
    issues = _validate_command_line(line, runner)
    assert not issues, issues
