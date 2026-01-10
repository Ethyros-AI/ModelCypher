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


_FORBIDDEN_PREFIXES = (
    "modelcypher.adapters",
    "modelcypher.infrastructure",
    "modelcypher.backends",
    "modelcypher.cli",
    "modelcypher.mcp",
)
_FORBIDDEN_ROOTS = {"adapters", "infrastructure", "backends", "cli", "mcp"}
_BACKENDS_ALLOWLIST = {
    "src/modelcypher/core/domain/_backend.py",
}


def _module_parts(path: Path, src_root: Path) -> list[str]:
    rel = path.relative_to(src_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        return parts[:-1]
    parts[-1] = parts[-1].removesuffix(".py")
    return parts


def _resolve_import_from(node: ast.ImportFrom, module_parts: list[str]) -> str | None:
    if node.level == 0:
        return node.module
    if node.level > len(module_parts):
        return node.module
    base = module_parts[:-node.level]
    if node.module:
        return ".".join(base + node.module.split("."))
    return ".".join(base)


def _is_forbidden(module: str | None, file_rel: str) -> bool:
    if module is None:
        return False
    for prefix in _FORBIDDEN_PREFIXES:
        if module == prefix or module.startswith(prefix + "."):
            if prefix == "modelcypher.backends" and file_rel in _BACKENDS_ALLOWLIST:
                return False
            return True
    return False


def _scan_forbidden_imports(root: Path, src_root: Path) -> list[str]:
    violations: list[str] = []
    for path in root.rglob("*.py"):
        file_rel = str(path.relative_to(src_root))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=file_rel)
        module_parts = _module_parts(path, src_root)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name, file_rel):
                        violations.append(
                            f"{file_rel}:{node.lineno} import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                full_module = _resolve_import_from(node, module_parts)
                if _is_forbidden(full_module, file_rel):
                    violations.append(
                        f"{file_rel}:{node.lineno} from {full_module} import ..."
                    )
                if node.module == "modelcypher":
                    for alias in node.names:
                        if alias.name in _FORBIDDEN_ROOTS:
                            module = f"modelcypher.{alias.name}"
                            if _is_forbidden(module, file_rel):
                                violations.append(
                                    f"{file_rel}:{node.lineno} from {module} import ..."
                                )
    return violations


def test_core_hexagonal_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    core_root = src_root / "modelcypher" / "core"
    domain_dir = core_root / "domain"
    use_cases_dir = core_root / "use_cases"

    violations = _scan_forbidden_imports(domain_dir, src_root)
    violations += _scan_forbidden_imports(use_cases_dir, src_root)

    assert not violations, "Hexagonal boundary violations:\n" + "\n".join(
        sorted(violations)
    )
