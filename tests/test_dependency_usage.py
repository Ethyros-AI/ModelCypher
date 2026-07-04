# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

_IMPORT_MODULE_BY_DEP = {
    "PyYAML": "yaml",
    "huggingface-hub": "huggingface_hub",
    "lm-eval": "lm_eval",
    "mlx-lm": "mlx_lm",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
}


def _mandatory_dependencies(pyproject_path: Path) -> dict[str, str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = data["tool"]["poetry"]["dependencies"]
    mandatory: dict[str, str] = {}
    for dep_name, dep_spec in dependencies.items():
        if dep_name == "python":
            continue
        if isinstance(dep_spec, dict) and dep_spec.get("optional"):
            continue
        module_name = _IMPORT_MODULE_BY_DEP.get(dep_name, dep_name.replace("-", "_"))
        mandatory[dep_name] = module_name
    return mandatory


def _source_import_roots(src_root: Path) -> set[str]:
    imports: set[str] = set()
    for path in src_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", maxsplit=1)[0])
    return imports


def test_mandatory_dependencies_are_imported_by_source() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mandatory = _mandatory_dependencies(repo_root / "pyproject.toml")
    source_imports = _source_import_roots(repo_root / "src" / "modelcypher")

    missing = {
        dep_name: module_name
        for dep_name, module_name in mandatory.items()
        if module_name not in source_imports
    }

    assert not missing, (
        "Mandatory dependencies must be imported by src/modelcypher or demoted "
        f"to an extra: {missing}"
    )
