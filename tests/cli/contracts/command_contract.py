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

"""Reusable command-contract validation helpers for docs/tests."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
from pathlib import Path

from typer.testing import CliRunner

from modelcypher.cli.app import app


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    detail: str


@dataclass(frozen=True)
class CommandExample:
    command: str
    line_no: int


def tokenize(line: str) -> list[str]:
    return shlex.split(line, comments=True, posix=True)


def normalize_mc_command(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped == "mc" or stripped.startswith("mc "):
        return f"poetry run {stripped}"
    if stripped.startswith("poetry run mc"):
        return stripped
    return None


def extract_bash_command_examples(text: str) -> list[CommandExample]:
    """Extract normalized mc commands from fenced bash blocks.

    Supports line continuations with trailing backslash.
    Only accepts command lines starting with `mc` or `poetry run mc`.
    """
    examples: list[CommandExample] = []
    in_bash_block = False
    buffer: str | None = None
    buffer_line_no = 0

    def _emit(raw_line: str, line_no: int) -> None:
        normalized = normalize_mc_command(raw_line)
        if normalized is not None:
            examples.append(CommandExample(command=normalized, line_no=line_no))

    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()

        if not in_bash_block:
            if stripped.lower() == "```bash":
                in_bash_block = True
            continue

        if stripped.startswith("```"):
            if buffer is not None:
                _emit(buffer, buffer_line_no)
                buffer = None
                buffer_line_no = 0
            in_bash_block = False
            continue

        if not stripped or stripped.startswith("#"):
            if buffer is not None:
                _emit(buffer, buffer_line_no)
                buffer = None
                buffer_line_no = 0
            continue

        if buffer is None:
            buffer = stripped
            buffer_line_no = lineno
        else:
            buffer = f"{buffer} {stripped}"

        if buffer.endswith("\\"):
            buffer = buffer[:-1].rstrip()
            continue

        _emit(buffer, buffer_line_no)
        buffer = None
        buffer_line_no = 0

    if in_bash_block and buffer is not None:
        _emit(buffer, buffer_line_no)

    return examples


def read_markdown_command_examples(path: Path) -> list[CommandExample]:
    return extract_bash_command_examples(path.read_text(encoding="utf-8"))


class CLIContractValidator:
    """Validates docs command examples against CLI help signatures."""

    def __init__(self, runner: CliRunner) -> None:
        self.runner = runner
        self._help_cache: dict[tuple[str, ...], tuple[int, str]] = {}
        self._path_cache: dict[tuple[str, ...], tuple[str, ...] | None] = {}
        self._known_option_cache: dict[tuple[str, ...], set[str]] = {}
        self._usage_cache: dict[tuple[str, ...], str] = {}
        self._required_positional_cache: dict[tuple[str, ...], int] = {}

    def _help_result(self, command_path: list[str]) -> tuple[int, str]:
        key = tuple(command_path)
        if key not in self._help_cache:
            result = self.runner.invoke(app, [*command_path, "--help"])
            self._help_cache[key] = (result.exit_code, result.stdout)
        return self._help_cache[key]

    def _help_text(self, command_path: list[str]) -> str:
        _, stdout = self._help_result(command_path)
        return stdout

    @staticmethod
    def _extract_usage_line(help_text: str) -> str:
        match = re.search(r"Usage:\s+([^\n]+)", help_text)
        if not match:
            return ""
        return match.group(1).strip()

    def _resolve_command_path(self, argv: list[str]) -> list[str] | None:
        key = tuple(argv)
        if key in self._path_cache:
            cached = self._path_cache[key]
            return list(cached) if cached is not None else None

        if not argv or argv[0] != "mc":
            self._path_cache[key] = None
            return None

        # Root command must be resolvable.
        root_exit, _ = self._help_result([])
        if root_exit != 0:
            self._path_cache[key] = None
            return None

        tail = argv[1:]
        leading_non_options: list[str] = []
        for token in tail:
            if token.startswith("-"):
                break
            leading_non_options.append(token)

        resolved: list[str] = []
        for token in leading_non_options:
            candidate = [*resolved, token]
            exit_code, stdout = self._help_result(candidate)
            if exit_code != 0:
                break

            resolved = candidate
            usage = self._extract_usage_line(stdout)
            # Leaf command reached; remaining non-option tokens are positionals.
            if "COMMAND" not in usage:
                break

        self._path_cache[key] = tuple(resolved)
        return resolved

    @staticmethod
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

    def _known_options(self, command_path: list[str]) -> set[str]:
        key = tuple(command_path)
        if key in self._known_option_cache:
            return self._known_option_cache[key]

        help_text = self._help_text(command_path)
        long_options = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", help_text))
        short_options = set(re.findall(r"(?<!\S)-[A-Za-z0-9]{1,3}(?!\S)", help_text))
        known = long_options | short_options
        self._known_option_cache[key] = known
        return known

    def _usage_line(self, command_path: list[str]) -> str:
        key = tuple(command_path)
        if key in self._usage_cache:
            return self._usage_cache[key]
        usage = self._extract_usage_line(self._help_text(command_path))
        self._usage_cache[key] = usage
        return usage

    def _usage_declares_positionals(self, command_path: list[str]) -> bool:
        usage = self._usage_line(command_path)
        if not usage:
            return False
        placeholders = re.findall(r"\b[A-Z][A-Z0-9_-]*\b", usage)
        filtered = [p for p in placeholders if p not in {"OPTIONS", "COMMAND", "ARGS"}]
        return bool(filtered)

    def _required_positional_count(self, command_path: list[str]) -> int:
        key = tuple(command_path)
        if key in self._required_positional_cache:
            return self._required_positional_cache[key]

        usage = self._usage_line(command_path)
        if not usage:
            self._required_positional_cache[key] = 0
            return 0

        placeholders = re.findall(r"\b[A-Z][A-Z0-9_-]*\b", usage)
        optional_placeholders = set(re.findall(r"\[([A-Z][A-Z0-9_-]*)\]", usage))
        required = [
            p
            for p in placeholders
            if p not in {"OPTIONS", "COMMAND", "ARGS"} and p not in optional_placeholders
        ]
        count = len(required)
        self._required_positional_cache[key] = count
        return count

    def validate_command_line(self, line: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        try:
            tokens = tokenize(line)
        except ValueError as exc:
            return [ValidationIssue("INVALID_PREFIX", f"Unable to tokenize command line: {exc}")]

        if len(tokens) < 3 or tokens[:3] != ["poetry", "run", "mc"]:
            return [ValidationIssue("INVALID_PREFIX", f"Line does not start with `poetry run mc`: {line}")]

        argv = tokens[2:]  # ["mc", ...]
        command_path = self._resolve_command_path(argv)
        if command_path is None:
            return [ValidationIssue("UNKNOWN_COMMAND_PATH", f"Could not resolve command path: {line}")]

        usage_line = self._usage_line(command_path)

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
        explicit_options, positionals = self._split_options_and_positionals(tail)

        known_options = self._known_options(command_path)
        for option in explicit_options:
            if option in {"--help", "-h"}:
                continue
            if option not in known_options:
                issues.append(
                    ValidationIssue(
                        "OPTION_NOT_IN_HELP",
                        f"Option `{option}` is not present in help for `{' '.join(command_path) or 'mc'}`",
                    )
                )

        declares_positionals = self._usage_declares_positionals(command_path)
        if positionals and not declares_positionals:
            issues.append(
                ValidationIssue(
                    "POSITIONAL_SHAPE",
                    f"Positional tokens {positionals} provided, but usage declares no positional placeholders",
                )
            )

        required_positionals = self._required_positional_count(command_path)
        if required_positionals > 0 and len(positionals) < required_positionals:
            issues.append(
                ValidationIssue(
                    "MISSING_REQUIRED_POSITIONAL",
                    (
                        f"Usage requires at least {required_positionals} positional token(s), "
                        f"but got {len(positionals)}"
                    ),
                )
            )

        return issues


def validate_markdown_file(
    path: Path,
    validator: CLIContractValidator,
) -> tuple[list[CommandExample], list[tuple[CommandExample, ValidationIssue]]]:
    examples = read_markdown_command_examples(path)
    issues: list[tuple[CommandExample, ValidationIssue]] = []
    for example in examples:
        for issue in validator.validate_command_line(example.command):
            issues.append((example, issue))
    return examples, issues
