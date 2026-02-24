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

from __future__ import annotations

import ast
from pathlib import Path


_FORBIDDEN_FRAMEWORK_IMPORTS = {"mlx", "jax", "torch"}
_FORBIDDEN_CORE_IMPORTS = {"numpy"}


def _import_stmt_text(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        return "import " + ", ".join(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = "." * node.level + (node.module or "")
        imported = ", ".join(alias.name for alias in node.names)
        return f"from {module} import {imported}"
    return "<unknown import>"


def _scan_import_roots(tree: ast.AST) -> list[tuple[str, int, str]]:
    imports: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                imports.append((root, node.lineno, _import_stmt_text(node)))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".", 1)[0]
                imports.append((root, node.lineno, _import_stmt_text(node)))
    return imports


def _framework_import_violations(tree: ast.AST, file_rel: str) -> list[str]:
    violations: list[str] = []
    for root, lineno, stmt in _scan_import_roots(tree):
        if root in _FORBIDDEN_FRAMEWORK_IMPORTS:
            violations.append(f"{file_rel}:{lineno} {stmt}")
    return violations


def _numpy_import_violations(tree: ast.AST, file_rel: str) -> list[str]:
    violations: list[str] = []
    for root, lineno, stmt in _scan_import_roots(tree):
        if root in _FORBIDDEN_CORE_IMPORTS:
            violations.append(f"{file_rel}:{lineno} {stmt}")
    return violations


def _parse_file(path: Path, file_rel: str) -> ast.AST:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=file_rel)


def test_framework_imports_confined_to_backends() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "modelcypher"
    violations: list[str] = []

    for path in src_root.rglob("*.py"):
        if "backends" in path.parts:
            continue
        file_rel = str(path.relative_to(repo_root))
        tree = _parse_file(path, file_rel)
        violations.extend(_framework_import_violations(tree, file_rel))

    assert not violations, (
        "Framework imports must remain inside src/modelcypher/backends/:\n"
        + "\n".join(sorted(violations))
    )


def test_numpy_imports_forbidden_in_core() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "modelcypher" / "core"
    violations: list[str] = []

    for segment in ("domain", "use_cases"):
        for path in (src_root / segment).rglob("*.py"):
            file_rel = str(path.relative_to(repo_root))
            tree = _parse_file(path, file_rel)
            violations.extend(_numpy_import_violations(tree, file_rel))

    assert not violations, (
        "numpy imports are forbidden in core/domain and core/use_cases:\n"
        + "\n".join(sorted(violations))
    )


def test_negative_control_detects_forbidden_framework_imports() -> None:
    tree = ast.parse(
        "import mlx.core as mx\n"
        "from torch import nn\n"
        "from jax import numpy as jnp\n"
    )
    violations = _framework_import_violations(tree, "<snippet>")
    assert violations
    assert any("mlx" in v for v in violations)
    assert any("torch" in v for v in violations)
    assert any("jax" in v for v in violations)


def test_negative_control_allows_modelcypher_backend_import_path() -> None:
    tree = ast.parse(
        "from modelcypher.backends import get_backend\n"
        "import modelcypher.backends.mlx_backend\n"
    )
    violations = _framework_import_violations(tree, "<snippet>")
    assert not violations
