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

"""Integration tests for experimental adapter routing orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modelcypher.core.use_cases.adapter_routing_service import AdapterRoutingService


class _MockRoutingBackend:
    """Delegating backend with deterministic model-loading and activation collection."""

    def __init__(
        self,
        backend: Any,
        base_activations: dict[int, Any],
        adapter_activations: dict[str, dict[int, Any]],
    ) -> None:
        self._backend = backend
        self._base_activations = base_activations
        self._adapter_activations = adapter_activations

    def load_model(
        self,
        path: str,
        adapter_path: str | None = None,
    ) -> tuple[dict[str, str | None], dict[str, str | None]]:
        model = {"path": path, "adapter_path": adapter_path}
        tokenizer = {"path": path, "adapter_path": adapter_path}
        return model, tokenizer

    def collect_hidden_activations(
        self,
        model: dict[str, str | None],
        tokenizer: dict[str, str | None],
        prompts: list[str],
        layer_indices: list[int] | None = None,
    ) -> dict[int, Any]:
        del tokenizer, prompts
        adapter_path = model.get("adapter_path")
        activations = (
            self._base_activations
            if adapter_path is None
            else self._adapter_activations[str(adapter_path)]
        )
        if layer_indices is None:
            return dict(activations)
        return {index: activations[index] for index in layer_indices if index in activations}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _write_adapter_config(
    path: Path,
    *,
    rank: int,
    alpha: float,
    target_modules: list[str],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def _build_backend_and_paths(tmp_path: Path, any_backend: Any) -> tuple[_MockRoutingBackend, str, list[str]]:
    base_layer_acts = {
        0: any_backend.array([[1.0, 2.0, 3.0]], dtype="float32"),
        1: any_backend.array([[2.0, 1.0, 0.5]], dtype="float32"),
    }

    adapter_one_path = tmp_path / "adapter_math"
    adapter_two_path = tmp_path / "adapter_code"
    _write_adapter_config(
        adapter_one_path,
        rank=8,
        alpha=16.0,
        target_modules=["layers.0.q_proj", "layers.1.q_proj"],
    )
    _write_adapter_config(
        adapter_two_path,
        rank=4,
        alpha=4.0,
        target_modules=["layers.0.v_proj"],
    )

    adapter_layer_acts = {
        str(adapter_one_path.resolve()): {
            0: any_backend.array([[1.1, 2.0, 3.1]], dtype="float32"),
            1: any_backend.array([[2.1, 1.1, 0.4]], dtype="float32"),
        },
        str(adapter_two_path.resolve()): {
            0: any_backend.array([[4.0, 0.2, 0.2]], dtype="float32"),
            1: any_backend.array([[0.1, 3.5, 4.0]], dtype="float32"),
        },
    }

    backend = _MockRoutingBackend(any_backend, base_layer_acts, adapter_layer_acts)
    base_model_path = str((tmp_path / "base_model").resolve())
    adapter_paths = [str(adapter_one_path.resolve()), str(adapter_two_path.resolve())]
    return backend, base_model_path, adapter_paths


def test_load_adapter_pool_reads_identity_metadata(tmp_path: Path, any_backend: Any) -> None:
    backend, base_model_path, adapter_paths = _build_backend_and_paths(tmp_path, any_backend)
    service = AdapterRoutingService(backend)

    pool = service.load_adapter_pool(base_model_path, adapter_paths)

    assert pool.base_model_path == str(Path(base_model_path).resolve())
    assert set(pool.adapter_models.keys()) == {"adapter_math", "adapter_code"}
    identities = {identity.id: identity for identity in pool.adapter_identities}
    assert identities["adapter_math"].rank == 8
    assert identities["adapter_math"].scale == pytest.approx(2.0)
    assert identities["adapter_math"].module_keys == ("layers.0.q_proj", "layers.1.q_proj")
    assert identities["adapter_code"].rank == 4
    assert identities["adapter_code"].scale == pytest.approx(1.0)
    assert identities["adapter_code"].module_keys == ("layers.0.v_proj",)


def test_collect_routing_measurements_kl_and_cosine_selectors(tmp_path: Path, any_backend: Any) -> None:
    backend, base_model_path, adapter_paths = _build_backend_and_paths(tmp_path, any_backend)
    service = AdapterRoutingService(backend)
    pool = service.load_adapter_pool(base_model_path, adapter_paths)

    min_trace = service.collect_routing_measurements(
        pool=pool,
        prompt="What is 2+2?",
        selection_method="min_kl",
    )
    max_trace = service.collect_routing_measurements(
        pool=pool,
        prompt="What is 2+2?",
        selection_method="max_kl",
    )
    max_cosine_trace = service.collect_routing_measurements(
        pool=pool,
        prompt="What is 2+2?",
        selection_method="max_cosine",
    )
    min_cosine_trace = service.collect_routing_measurements(
        pool=pool,
        prompt="What is 2+2?",
        selection_method="min_cosine",
    )

    assert min_trace.n_adapters == 2
    assert min_trace.n_layers == 2
    assert len(min_trace.layer_snapshots) == 2
    assert min_trace.selected_adapter_per_layer == {0: "adapter_math", 1: "adapter_math"}
    assert min_trace.output_kl_vs_single is None

    assert max_trace.selected_adapter_per_layer == {0: "adapter_code", 1: "adapter_code"}
    assert max_trace.output_kl_vs_single is None
    assert max_cosine_trace.selected_adapter_per_layer == {0: "adapter_math", 1: "adapter_math"}
    assert min_cosine_trace.selected_adapter_per_layer == {0: "adapter_code", 1: "adapter_code"}


def test_collect_routing_measurements_none_selection(tmp_path: Path, any_backend: Any) -> None:
    backend, base_model_path, adapter_paths = _build_backend_and_paths(tmp_path, any_backend)
    service = AdapterRoutingService(backend)
    pool = service.load_adapter_pool(base_model_path, adapter_paths)

    trace = service.collect_routing_measurements(
        pool=pool,
        prompt="Evaluate geometry.",
        selection_method="none",
    )

    assert trace.selection_method == "none"
    assert trace.selected_adapter_per_layer == {}
    assert trace.n_layers == 2


def test_run_evaluation_returns_trace_per_prompt(tmp_path: Path, any_backend: Any) -> None:
    backend, base_model_path, adapter_paths = _build_backend_and_paths(tmp_path, any_backend)
    service = AdapterRoutingService(backend)
    pool = service.load_adapter_pool(base_model_path, adapter_paths)

    traces = service.run_evaluation(
        pool=pool,
        prompts=["Prompt A", "Prompt B"],
        selection_method="none",
    )

    assert len(traces) == 2
    assert traces[0].prompt == "Prompt A"
    assert traces[1].prompt == "Prompt B"


def test_invalid_selection_method_raises(tmp_path: Path, any_backend: Any) -> None:
    backend, base_model_path, adapter_paths = _build_backend_and_paths(tmp_path, any_backend)
    service = AdapterRoutingService(backend)
    pool = service.load_adapter_pool(base_model_path, adapter_paths)

    with pytest.raises(ValueError, match="selection_method"):
        service.collect_routing_measurements(
            pool=pool,
            prompt="Invalid selection",
            selection_method="median_kl",
        )
