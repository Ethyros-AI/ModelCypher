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

"""Tests for experimental adapter divergence profile service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modelcypher.core.use_cases.adapter_divergence_profile_service import (
    AdapterDivergenceProfileService,
)
from modelcypher.core.use_cases.adapter_routing_service import AdapterRoutingService


class _MockRoutingBackend:
    """Delegating backend with prompt-conditioned activation streams."""

    def __init__(
        self,
        backend: Any,
        base_activations_by_prompt: dict[str, dict[int, Any]],
        adapter_activations_by_prompt: dict[str, dict[str, dict[int, Any]]],
    ) -> None:
        self._backend = backend
        self._base_activations_by_prompt = base_activations_by_prompt
        self._adapter_activations_by_prompt = adapter_activations_by_prompt

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
        del tokenizer
        prompt = prompts[0]
        adapter_path = model.get("adapter_path")
        activations = (
            self._base_activations_by_prompt[prompt]
            if adapter_path is None
            else self._adapter_activations_by_prompt[prompt][str(adapter_path)]
        )
        if layer_indices is None:
            return dict(activations)
        return {index: activations[index] for index in layer_indices if index in activations}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _write_adapter_config(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "r": 4,
        "lora_alpha": 8.0,
        "target_modules": ["layers.0.q_proj", "layers.1.q_proj"],
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def _build_profile_service(
    tmp_path: Path,
    any_backend: Any,
) -> tuple[AdapterDivergenceProfileService, AdapterRoutingService, str, list[str], list[str]]:
    prompts = ["prompt_a", "prompt_b"]

    adapter_alpha_path = (tmp_path / "adapter_alpha").resolve()
    adapter_beta_path = (tmp_path / "adapter_beta").resolve()
    _write_adapter_config(adapter_alpha_path)
    _write_adapter_config(adapter_beta_path)

    base_by_prompt = {
        "prompt_a": {
            0: any_backend.array([[1.0, 0.0]], dtype="float32"),
            1: any_backend.array([[0.0, 1.0]], dtype="float32"),
        },
        "prompt_b": {
            0: any_backend.array([[1.0, 0.0]], dtype="float32"),
            1: any_backend.array([[0.0, 1.0]], dtype="float32"),
        },
    }

    adapter_by_prompt = {
        "prompt_a": {
            str(adapter_alpha_path): {
                0: any_backend.array([[1.0, 0.0]], dtype="float32"),
                1: any_backend.array([[0.0, 1.0]], dtype="float32"),
            },
            str(adapter_beta_path): {
                0: any_backend.array([[0.0, 1.0]], dtype="float32"),
                1: any_backend.array([[1.0, 0.0]], dtype="float32"),
            },
        },
        "prompt_b": {
            str(adapter_alpha_path): {
                0: any_backend.array([[0.7, 0.3]], dtype="float32"),
                1: any_backend.array([[0.0, 1.0]], dtype="float32"),
            },
            str(adapter_beta_path): {
                0: any_backend.array([[1.0, 0.0]], dtype="float32"),
                1: any_backend.array([[0.6, 0.4]], dtype="float32"),
            },
        },
    }

    backend = _MockRoutingBackend(any_backend, base_by_prompt, adapter_by_prompt)
    routing_service = AdapterRoutingService(backend)
    profile_service = AdapterDivergenceProfileService(routing_service=routing_service)
    base_model_path = str((tmp_path / "base_model").resolve())
    adapter_paths = [str(adapter_alpha_path), str(adapter_beta_path)]
    return profile_service, routing_service, base_model_path, adapter_paths, prompts


def _layer_entry(per_layer: list[dict[str, Any]], layer_index: int) -> dict[str, Any]:
    for entry in per_layer:
        if int(entry["layer_index"]) == layer_index:
            return entry
    pytest.fail(f"Missing layer entry for {layer_index}")


def test_compute_profile_contains_expected_structure(tmp_path: Path, any_backend: Any) -> None:
    service, _, base_model_path, adapter_paths, prompts = _build_profile_service(tmp_path, any_backend)

    profile = service.compute_profile(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        prompts=prompts,
    )

    assert profile["n_prompts"] == 2
    assert profile["n_layers"] == 2
    assert profile["adapter_ids"] == ["adapter_alpha", "adapter_beta"]
    assert set(profile["per_adapter"]) == {"adapter_alpha", "adapter_beta"}
    assert set(profile["pairwise"]) == {"adapter_alpha_vs_adapter_beta"}
    assert set(profile["routing_potential"]) == {"adapter_alpha_vs_adapter_beta"}


def test_compute_profile_per_adapter_means_match_trace_measurements(
    tmp_path: Path,
    any_backend: Any,
) -> None:
    service, routing_service, base_model_path, adapter_paths, prompts = _build_profile_service(
        tmp_path,
        any_backend,
    )
    pool = routing_service.load_adapter_pool(base_model_path, adapter_paths)

    traces = [
        routing_service.collect_routing_measurements(
            pool=pool,
            prompt=prompt,
            selection_method="none",
        )
        for prompt in prompts
    ]
    profile = service.compute_profile(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        prompts=prompts,
    )

    alpha_l0_kls: list[float] = []
    alpha_cosines: list[float] = []
    for trace in traces:
        for snapshot in trace.layer_snapshots:
            for measurement in snapshot.measurements:
                if measurement.adapter_id == "adapter_alpha" and measurement.layer_index == 0:
                    alpha_l0_kls.append(float(measurement.kl_divergence))
                if measurement.adapter_id == "adapter_alpha":
                    alpha_cosines.append(float(measurement.cosine_similarity))

    alpha_profile = profile["per_adapter"]["adapter_alpha"]
    alpha_l0_entry = _layer_entry(alpha_profile["per_layer"], layer_index=0)
    expected_mean_l0_kl = sum(alpha_l0_kls) / len(alpha_l0_kls)
    expected_mean_cosine = sum(alpha_cosines) / len(alpha_cosines)

    assert alpha_l0_entry["mean_kl"] == pytest.approx(expected_mean_l0_kl)
    assert alpha_profile["aggregate"]["mean_cosine"] == pytest.approx(expected_mean_cosine)


def test_compute_profile_pairwise_comparison_for_two_adapters(
    tmp_path: Path,
    any_backend: Any,
) -> None:
    service, _, base_model_path, adapter_paths, prompts = _build_profile_service(tmp_path, any_backend)
    profile = service.compute_profile(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        prompts=prompts,
    )

    pair = profile["pairwise"]["adapter_alpha_vs_adapter_beta"]

    assert pair["dominant_adapter_rate"] == pytest.approx(0.75)
    assert pair["mean_kl_gap"] > 0.0
    assert pair["mean_cosine_gap"] > 0.0


def test_compute_profile_routing_potential_inverts_agreement_rate(
    tmp_path: Path,
    any_backend: Any,
) -> None:
    service, _, base_model_path, adapter_paths, prompts = _build_profile_service(tmp_path, any_backend)
    profile = service.compute_profile(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        prompts=prompts,
    )

    key = "adapter_alpha_vs_adapter_beta"
    agreement = profile["pairwise"][key]["dominant_adapter_rate"]
    assert profile["routing_potential"][key] == pytest.approx(1.0 - agreement)
