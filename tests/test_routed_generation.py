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

"""Integration tests for experimental routed generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modelcypher.core.use_cases.adapter_routing_service import AdapterRoutingService


class _MockRoutingBackend:
    def __init__(
        self,
        backend: Any,
        base_activations: dict[int, Any],
        adapter_activations: dict[str, dict[int, Any]],
    ) -> None:
        self._backend = backend
        self._base_activations = base_activations
        self._adapter_activations = adapter_activations
        self._weights_by_path: dict[str, dict[str, Any]] = {}
        self._save_calls = 0

    @property
    def save_calls(self) -> int:
        return self._save_calls

    def register_weights(self, path: str, weights: dict[str, Any]) -> None:
        self._weights_by_path[path] = dict(weights)

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

    def load_safetensors(self, path: str) -> dict[str, Any]:
        return dict(self._weights_by_path[path])

    def save_safetensors(
        self,
        path: str,
        weights: dict[str, Any],
        metadata: dict[str, str] | None = None,
    ) -> None:
        del metadata
        self._save_calls += 1
        self._weights_by_path[path] = dict(weights)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mock")

    def generate(
        self,
        model: dict[str, str | None],
        tokenizer: dict[str, str | None],
        prompt: str,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        del tokenizer, kwargs
        adapter_path = model.get("adapter_path")
        tag = "base" if adapter_path is None else str(Path(str(adapter_path)).name)
        return f"response[{tag}]::{prompt}::max_tokens={max_tokens}"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _layer_keys(layer_idx: int) -> tuple[str, str]:
    base = f"model.layers.{layer_idx}.self_attn.q_proj"
    return f"{base}.lora_a", f"{base}.lora_b"


def _write_adapter_dir(
    backend: _MockRoutingBackend,
    base_backend: Any,
    path: Path,
    *,
    rank: int,
    alpha: float,
    target_modules: list[str],
    weights: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "rank": rank,
        "lora_alpha": alpha,
        "num_layers": 2,
        "target_modules": target_modules,
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    weights_path = path / "adapters.safetensors"
    weights_path.write_bytes(b"mock")
    backend.register_weights(str(weights_path.resolve()), weights)
    _ = base_backend  # Keep signature aligned with call sites


def _setup_backend(tmp_path: Path, any_backend: Any) -> tuple[_MockRoutingBackend, str, list[str]]:
    base_activations = {
        0: any_backend.array([[1.0, 2.0, 3.0]], dtype="float32"),
        1: any_backend.array([[2.0, 1.0, 0.5]], dtype="float32"),
    }

    adapter_math_path = (tmp_path / "adapter_math").resolve()
    adapter_code_path = (tmp_path / "adapter_code").resolve()

    a0_a, a0_b = _layer_keys(0)
    a1_a, a1_b = _layer_keys(1)
    math_weights = {
        a0_a: any_backend.full((4, 2), fill_value=1.0, dtype="float32"),
        a0_b: any_backend.full((2, 3), fill_value=1.0, dtype="float32"),
        a1_a: any_backend.full((4, 2), fill_value=1.0, dtype="float32"),
        a1_b: any_backend.full((2, 3), fill_value=1.0, dtype="float32"),
    }
    code_weights = {
        a0_a: any_backend.full((4, 2), fill_value=2.0, dtype="float32"),
        a0_b: any_backend.full((2, 3), fill_value=2.0, dtype="float32"),
        a1_a: any_backend.full((4, 2), fill_value=2.0, dtype="float32"),
        a1_b: any_backend.full((2, 3), fill_value=2.0, dtype="float32"),
    }

    adapter_activations = {
        str(adapter_math_path): {
            0: any_backend.array([[1.1, 2.0, 3.1]], dtype="float32"),
            1: any_backend.array([[2.1, 1.1, 0.4]], dtype="float32"),
        },
        str(adapter_code_path): {
            0: any_backend.array([[4.0, 0.2, 0.2]], dtype="float32"),
            1: any_backend.array([[0.1, 3.5, 4.0]], dtype="float32"),
        },
    }

    backend = _MockRoutingBackend(any_backend, base_activations, adapter_activations)
    _write_adapter_dir(
        backend,
        any_backend,
        adapter_math_path,
        rank=2,
        alpha=2.0,
        target_modules=["self_attn.q_proj"],
        weights=math_weights,
    )
    _write_adapter_dir(
        backend,
        any_backend,
        adapter_code_path,
        rank=2,
        alpha=2.0,
        target_modules=["self_attn.q_proj"],
        weights=code_weights,
    )

    base_model_path = str((tmp_path / "base_model").resolve())
    adapter_paths = [str(adapter_math_path), str(adapter_code_path)]
    return backend, base_model_path, adapter_paths


def test_route_and_generate_returns_comparison_payload(tmp_path: Path, any_backend: Any) -> None:
    backend, base_model_path, adapter_paths = _setup_backend(tmp_path, any_backend)
    service = AdapterRoutingService(backend)
    pool = service.load_adapter_pool(base_model_path, adapter_paths)

    result = service.route_and_generate(
        pool=pool,
        prompt="What is 2+2?",
        max_tokens=32,
        selection_method="min_kl",
    )

    composite_path = Path(result.composite_adapter_path)
    assert result.prompt == "What is 2+2?"
    assert result.routing_trace.selected_adapter_per_layer == {0: "adapter_math", 1: "adapter_math"}
    assert result.base_response.startswith("response[base]")
    assert "response[" in result.routed_response
    assert set(result.single_adapter_responses) == {"adapter_code", "adapter_math"}
    assert composite_path.exists()
    assert (composite_path / "adapters.safetensors").exists()
    assert (composite_path / "adapter_config.json").exists()


def test_run_generation_evaluation_returns_one_result_per_prompt(
    tmp_path: Path,
    any_backend: Any,
) -> None:
    backend, base_model_path, adapter_paths = _setup_backend(tmp_path, any_backend)
    service = AdapterRoutingService(backend)
    pool = service.load_adapter_pool(base_model_path, adapter_paths)

    results = service.run_generation_evaluation(
        pool=pool,
        prompts=["Prompt A", "Prompt B"],
        max_tokens=16,
        selection_method="min_kl",
    )

    assert len(results) == 2
    assert results[0].prompt == "Prompt A"
    assert results[1].prompt == "Prompt B"


def test_route_and_generate_with_prebuilt_composite_skips_rebuild(
    tmp_path: Path,
    any_backend: Any,
) -> None:
    backend, base_model_path, adapter_paths = _setup_backend(tmp_path, any_backend)
    service = AdapterRoutingService(backend)
    pool = service.load_adapter_pool(base_model_path, adapter_paths)

    prebuilt_dir = str((tmp_path / "prebuilt_composite").resolve())
    service.build_composite_adapter(
        pool=pool,
        routing_decisions={0: "adapter_math", 1: "adapter_math"},
        output_directory=prebuilt_dir,
    )
    save_calls_before = backend.save_calls

    result = service.route_and_generate(
        pool=pool,
        prompt="Reuse composite",
        max_tokens=24,
        selection_method="min_kl",
        composite_dir=prebuilt_dir,
    )

    assert backend.save_calls == save_calls_before
    assert result.composite_adapter_path == prebuilt_dir

