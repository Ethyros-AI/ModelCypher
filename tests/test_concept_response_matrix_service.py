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

import json
from pathlib import Path

from modelcypher.core.use_cases.concept_response_matrix_service import (
    CRMBuildConfig,
    ConceptResponseMatrixService,
)
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix


class _FakeHiddenStateEngine:
    def __init__(self, layer_count: int, hidden_dim: int) -> None:
        self.layer_count = layer_count
        self.hidden_dim = hidden_dim

    def capture_hidden_states(
        self,
        model: str,
        prompt: str,
        adapter: str | None = None,
        target_layers: set[int] | None = None,
    ) -> dict[int, list[float]]:
        layers = target_layers or set(range(self.layer_count))
        states = {}
        for layer in layers:
            value = float(len(prompt) + layer + 1)
            states[layer] = [value] * self.hidden_dim
        return states


def _write_model_config(path: Path, layers: int, hidden_dim: int) -> None:
    config = {
        "num_hidden_layers": layers,
        "hidden_size": hidden_dim,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_crm_build_and_compare(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_model_config(model_dir, layers=2, hidden_dim=2)

    engine = _FakeHiddenStateEngine(layer_count=2, hidden_dim=2)
    service = ConceptResponseMatrixService(engine=engine)
    output_path = tmp_path / "crm.json"
    config = CRMBuildConfig(
        include_primes=True,
        include_gates=False,
        include_polyglot=False,
        max_prompts_per_anchor=1,
        max_anchors=2,
    )

    summary = service.build(
        model_path=str(model_dir),
        output_path=str(output_path),
        config=config,
    )

    assert output_path.exists()
    assert summary.anchor_count == 2
    assert summary.layer_count == 2
    assert summary.hidden_dim == 2

    crm = ConceptResponseMatrix.load(str(output_path))
    assert crm.anchor_metadata.total_count == 2
    assert crm.layer_count == 2
    assert crm.hidden_dim == 2

    output_path_2 = tmp_path / "crm_2.json"
    service.build(
        model_path=str(model_dir),
        output_path=str(output_path_2),
        config=config,
    )

    compare = service.compare(str(output_path), str(output_path_2))
    assert compare.common_anchor_count == summary.anchor_count
    assert compare.cka_matrix is None
    assert compare.overall_alignment > 0.99
