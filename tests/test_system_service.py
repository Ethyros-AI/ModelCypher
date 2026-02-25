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

from pathlib import Path
from types import SimpleNamespace

from modelcypher.core.use_cases.system_service import SystemService


class _DummyStore:
    def __init__(self) -> None:
        self.paths = SimpleNamespace(base=Path("."))


class _DummyTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _DummyModel:
    def __init__(self) -> None:
        self._params = {"w": _DummyTensor((2, 3))}
        self._trainable = {"tw": _DummyTensor((2, 2))}

    def parameters(self):
        return self._params

    def trainable_parameters(self):
        return self._trainable


class _DummyBackend:
    def clear_cache(self) -> None:
        return None

    def reset_peak_memory(self) -> None:
        return None

    def get_active_memory_gb(self) -> float:
        return 1.0

    def get_peak_memory_gb(self) -> float:
        return 2.0

    def load_model(self, _path: str):
        return _DummyModel(), object()

    def encode_tokens(self, _tokenizer, _text: str) -> list[int]:
        return [1, 2, 3]

    def collect_logits(self, _model, _tokenizer, _text: str, token_ids=None):
        del token_ids
        return _DummyTensor((3,))

    def eval(self, *arrays) -> None:
        del arrays
        return None

    def generate(self, _model, _tokenizer, _text: str, max_tokens: int = 512, **kwargs) -> str:
        del max_tokens, kwargs
        return "ok"

    def tree_flatten(self, params):
        if isinstance(params, dict):
            return list(params.items())
        raise TypeError("expected dict")

    def shape(self, tensor):
        return tensor.shape


def test_system_probe_cuda_payload() -> None:
    service = SystemService(_DummyStore())
    payload = service.probe("cuda")
    assert payload["target"] == "cuda"
    assert "backend" in payload
    assert "available" in payload["backend"]
    assert "systemInfo" in payload["backend"]


def test_system_probe_jax_payload() -> None:
    service = SystemService(_DummyStore())
    payload = service.probe("jax")
    assert payload["target"] == "jax"
    assert "backend" in payload
    assert "available" in payload["backend"]
    assert "systemInfo" in payload["backend"]


def test_system_readiness_includes_backends() -> None:
    service = SystemService(_DummyStore())
    payload = service.readiness()
    assert "backendVersions" in payload
    assert "backends" in payload
    assert "preferredBackend" in payload


def test_memory_profile_emits_required_fields(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"quantization": "4bit"}', encoding="utf-8")

    service = SystemService(_DummyStore(), backend=_DummyBackend())
    payload = service.memory_profile(
        model=str(model_dir),
        prompt="hello",
        train_probe=True,
        decode_tokens=8,
    )

    assert payload["model_id"] == "model"
    assert payload["precision_bits"] == 4
    assert payload["param_count"] == 6
    assert payload["train_probe"] is not None
    assert payload["train_probe"]["n_trainable_params"] == 4
    assert payload["decode_slope"]["windows"]
    for stage in payload["memory_stages"]:
        assert stage["peak_gb"] >= stage["active_gb"]
