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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.use_cases.concept_response_matrix_service import (
    ConceptResponseMatrixService,
)


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
    service._anchor_prompt_cache = [
        ("test_anchor:alpha", ["alpha"]),
        ("test_anchor:beta", ["beta"]),
    ]
    output_path = tmp_path / "crm.json"
    summary = service.build(
        model_path=str(model_dir),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert summary.anchor_count > 0
    assert summary.layer_count == 2
    assert summary.hidden_dim == 2

    crm = ConceptResponseMatrix.load(str(output_path))
    assert crm.anchor_metadata.total_count == summary.anchor_count
    assert crm.layer_count == 2
    assert crm.hidden_dim == 2

    output_path_2 = tmp_path / "crm_2.json"
    service.build(
        model_path=str(model_dir),
        output_path=str(output_path_2),
    )

    compare = service.compare(str(output_path), str(output_path_2))
    # Allow small tolerance for anchor count (some probes may be filtered during comparison)
    assert abs(compare.common_anchor_count - summary.anchor_count) <= 1
    assert compare.cka_matrix is None
    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([1.0]))
    assert abs(compare.mean_cka - 1.0) <= eps
    assert compare.aligned is True
