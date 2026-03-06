# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"


def test_deleted_probe_shim_stays_deleted() -> None:
    shim_path = _ROOT / "src/modelcypher/core/domain/geometry/orthogonal_probe_generator.py"
    assert not shim_path.exists()


def test_runtime_source_omits_deleted_loader_aliases_and_probe_shim_imports() -> None:
    banned_fragments = [
        ".load_model_for_training(",
        "def load_model_for_training(",
        "def get_model_loader(",
        "def load_model_weights_only(",
        "orthogonal_probe_generator",
        "layer_agreement_rate",
        "legacy_profile_path",
    ]
    violations: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        content = path.read_text(encoding="utf-8")
        for fragment in banned_fragments:
            if fragment in content:
                violations.append(f"{path.relative_to(_ROOT)}: contains '{fragment}'")

    assert not violations, "\n".join(violations)


def test_runtime_source_omits_backward_compatibility_language() -> None:
    banned_fragments = [
        "backward compatibility",
        "backwards compatibility",
        "backward compat",
        "backwards compat",
    ]
    violations: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        content = path.read_text(encoding="utf-8").lower()
        for fragment in banned_fragments:
            if fragment in content:
                violations.append(f"{path.relative_to(_ROOT)}: contains '{fragment}'")

    assert not violations, "\n".join(violations)
