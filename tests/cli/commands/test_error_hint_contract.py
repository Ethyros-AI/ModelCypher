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

"""Contract test: every ErrorDetail instantiation in CLI must include a hint.

Agents use the hint field to decide how to self-correct. Missing hints
leave agents with no recovery suggestion, degrading the CLI's agent-friendliness.

This test AST-parses all CLI command files and asserts that every
ErrorDetail(...) call includes a `hint=` keyword argument.
"""

from __future__ import annotations

import ast
from pathlib import Path

CLI_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "src" / "modelcypher" / "cli"


def _find_errordetail_calls_missing_hint(filepath: Path) -> list[tuple[str, int]]:
    """Return (filename, line_number) pairs for ErrorDetail calls missing hint=."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    violations: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Match ErrorDetail(...) or errors.ErrorDetail(...)
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr

        if name != "ErrorDetail":
            continue

        # Check if hint= is present as a keyword argument
        kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
        if "hint" not in kw_names:
            violations.append((str(filepath.relative_to(CLI_ROOT.parent.parent.parent)), node.lineno))

    return violations


def test_all_errordetail_calls_have_hint() -> None:
    """Every ErrorDetail instantiation in CLI code must include hint=."""
    all_violations: list[tuple[str, int]] = []

    for py_file in sorted(CLI_ROOT.rglob("*.py")):
        violations = _find_errordetail_calls_missing_hint(py_file)
        all_violations.extend(violations)

    if all_violations:
        report = "\n".join(f"  {path}:{line}" for path, line in all_violations)
        raise AssertionError(
            f"ErrorDetail calls missing hint= ({len(all_violations)} found):\n{report}\n\n"
            "Every ErrorDetail must include a hint= kwarg so agents can self-correct."
        )
