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

"""Tests for experimental composite adapter builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modelcypher.core.domain.inference.composite_adapter_builder import CompositeAdapterBuilder


class _MockCompositeBackend:
    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.loaded_weights: dict[str, dict[str, Any]] = {}
        self.saved_weights: dict[str, dict[str, Any]] = {}

    def load_safetensors(self, path: str) -> dict[str, Any]:
        if path not in self.loaded_weights:
            raise FileNotFoundError(path)
        return dict(self.loaded_weights[path])

    def save_safetensors(
        self,
        path: str,
        weights: dict[str, Any],
        metadata: dict[str, str] | None = None,
    ) -> None:
        del metadata
        self.saved_weights[path] = dict(weights)
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"mock")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _arr(backend: Any, rows: int, cols: int, value: float) -> Any:
    return backend.full((rows, cols), fill_value=value, dtype="float32")


def _layer_keys(layer_idx: int) -> tuple[str, str]:
    base = f"model.layers.{layer_idx}.self_attn.q_proj"
    return f"{base}.lora_a", f"{base}.lora_b"


def test_build_cherry_picks_selected_layer_weights(tmp_path: Path, any_backend: Any) -> None:
    backend = _MockCompositeBackend(any_backend)
    builder = CompositeAdapterBuilder(backend)

    path_a = str((tmp_path / "adapter_a.safetensors").resolve())
    path_b = str((tmp_path / "adapter_b.safetensors").resolve())
    Path(path_a).write_bytes(b"a")
    Path(path_b).write_bytes(b"b")

    a0_a, a0_b = _layer_keys(0)
    a1_a, a1_b = _layer_keys(1)
    backend.loaded_weights[path_a] = {
        a0_a: _arr(any_backend, 4, 2, 1.0),
        a0_b: _arr(any_backend, 2, 3, 1.0),
        a1_a: _arr(any_backend, 4, 2, 9.0),
        a1_b: _arr(any_backend, 2, 3, 9.0),
    }
    backend.loaded_weights[path_b] = {
        a0_a: _arr(any_backend, 4, 2, 7.0),
        a0_b: _arr(any_backend, 2, 3, 7.0),
        a1_a: _arr(any_backend, 4, 2, 2.0),
        a1_b: _arr(any_backend, 2, 3, 2.0),
    }

    out_dir = str((tmp_path / "composite").resolve())
    output = builder.build_composite_adapter(
        adapter_weight_paths={"adapter_a": path_a, "adapter_b": path_b},
        adapter_configs={
            "adapter_a": {"num_layers": 2, "target_modules": ["self_attn.q_proj"]},
            "adapter_b": {"num_layers": 2, "target_modules": ["self_attn.q_proj"]},
        },
        routing_decisions={0: "adapter_a", 1: "adapter_b"},
        output_directory=out_dir,
    )

    assert output == out_dir
    saved_path = str(Path(out_dir) / "adapters.safetensors")
    assert saved_path in backend.saved_weights
    saved = backend.saved_weights[saved_path]

    vals_0 = any_backend.tolist(saved[a0_a])
    vals_1 = any_backend.tolist(saved[a1_a])
    assert float(vals_0[0][0]) == pytest.approx(1.0)
    assert float(vals_1[0][0]) == pytest.approx(2.0)


def test_build_handles_rank_mismatch_with_zero_padding(tmp_path: Path, any_backend: Any) -> None:
    backend = _MockCompositeBackend(any_backend)
    builder = CompositeAdapterBuilder(backend)

    path_a = str((tmp_path / "adapter_a.safetensors").resolve())
    path_b = str((tmp_path / "adapter_b.safetensors").resolve())
    Path(path_a).write_bytes(b"a")
    Path(path_b).write_bytes(b"b")

    a0_a, a0_b = _layer_keys(0)
    a1_a, a1_b = _layer_keys(1)
    backend.loaded_weights[path_a] = {
        a0_a: _arr(any_backend, 4, 2, 1.0),
        a0_b: _arr(any_backend, 2, 3, 1.0),
    }
    backend.loaded_weights[path_b] = {
        a1_a: _arr(any_backend, 4, 4, 2.0),
        a1_b: _arr(any_backend, 4, 3, 2.0),
    }

    out_dir = str((tmp_path / "composite_rank").resolve())
    builder.build_composite_adapter(
        adapter_weight_paths={"adapter_a": path_a, "adapter_b": path_b},
        adapter_configs={
            "adapter_a": {"num_layers": 2, "target_modules": ["self_attn.q_proj"]},
            "adapter_b": {"num_layers": 2, "target_modules": ["self_attn.q_proj"]},
        },
        routing_decisions={0: "adapter_a", 1: "adapter_b"},
        output_directory=out_dir,
    )

    saved = backend.saved_weights[str(Path(out_dir) / "adapters.safetensors")]
    padded_a = saved[a0_a]
    padded_b = saved[a0_b]

    assert padded_a.shape == (4, 4)
    assert padded_b.shape == (4, 3)
    padded_a_rows = any_backend.tolist(padded_a)
    padded_b_rows = any_backend.tolist(padded_b)
    assert float(padded_a_rows[0][2]) == pytest.approx(0.0)
    assert float(padded_a_rows[0][3]) == pytest.approx(0.0)
    assert float(padded_b_rows[2][0]) == pytest.approx(0.0)
    assert float(padded_b_rows[3][0]) == pytest.approx(0.0)


def test_build_writes_valid_adapter_config(tmp_path: Path, any_backend: Any) -> None:
    backend = _MockCompositeBackend(any_backend)
    builder = CompositeAdapterBuilder(backend)

    path_a = str((tmp_path / "adapter_a.safetensors").resolve())
    path_b = str((tmp_path / "adapter_b.safetensors").resolve())
    Path(path_a).write_bytes(b"a")
    Path(path_b).write_bytes(b"b")

    a0_a, a0_b = _layer_keys(0)
    a1_a, a1_b = _layer_keys(1)
    backend.loaded_weights[path_a] = {
        a0_a: _arr(any_backend, 4, 2, 1.0),
        a0_b: _arr(any_backend, 2, 3, 1.0),
    }
    backend.loaded_weights[path_b] = {
        a1_a: _arr(any_backend, 4, 4, 2.0),
        a1_b: _arr(any_backend, 4, 3, 2.0),
    }

    out_dir = str((tmp_path / "composite_cfg").resolve())
    builder.build_composite_adapter(
        adapter_weight_paths={"adapter_a": path_a, "adapter_b": path_b},
        adapter_configs={
            "adapter_a": {"num_layers": 16, "target_modules": ["self_attn.q_proj"]},
            "adapter_b": {"num_layers": 16, "target_modules": ["self_attn.v_proj"]},
        },
        routing_decisions={0: "adapter_a", 1: "adapter_b"},
        output_directory=out_dir,
    )

    config = json.loads((Path(out_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["fine_tune_type"] == "lora"
    assert config["num_layers"] == 16
    assert config["lora_parameters"]["rank"] == 4
    assert config["lora_parameters"]["scale"] == pytest.approx(1.0)
    assert config["lora_parameters"]["dropout"] == pytest.approx(0.0)
    assert config["lora_parameters"]["keys"] == ["self_attn.q_proj", "self_attn.v_proj"]


def test_build_raises_on_empty_routing(any_backend: Any, tmp_path: Path) -> None:
    backend = _MockCompositeBackend(any_backend)
    builder = CompositeAdapterBuilder(backend)

    with pytest.raises(ValueError, match="routing_decisions"):
        builder.build_composite_adapter(
            adapter_weight_paths={"adapter_a": str(tmp_path / "a.safetensors")},
            adapter_configs={"adapter_a": {}},
            routing_decisions={},
            output_directory=str(tmp_path / "out"),
        )


def test_build_raises_on_unknown_adapter_id(any_backend: Any, tmp_path: Path) -> None:
    backend = _MockCompositeBackend(any_backend)
    builder = CompositeAdapterBuilder(backend)

    with pytest.raises(ValueError, match="unknown adapter ids"):
        builder.build_composite_adapter(
            adapter_weight_paths={"adapter_a": str(tmp_path / "a.safetensors")},
            adapter_configs={"adapter_a": {}},
            routing_decisions={0: "adapter_missing"},
            output_directory=str(tmp_path / "out"),
        )
