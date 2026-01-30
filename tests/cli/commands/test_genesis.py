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

"""Tests for genesis CLI commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.ports.activation_provider import ProbeActivationBatch

runner = CliRunner()


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        _ = text
        return [1, 2, 3]

    def decode(self, token_ids: list[int]) -> str:
        _ = token_ids
        # Satisfies all canaries: ["4"/"four"], ["paris"], ["cannot"/"won't"/"refuse"]
        return "4 paris cannot"


@dataclass
class _FakeState:
    token_id: int | None


class _FakeGeometricInference:
    def __init__(self, model, backend) -> None:
        _ = (model, backend)

    def generate(self, input_ids: list[int]):
        _ = input_ids
        for i in range(3):
            yield _FakeState(token_id=i)


class _FakeActivationProvider:
    def collect_probe_activations_batch(self, model, tokenizer, texts: list[str]) -> ProbeActivationBatch:
        _ = (model, tokenizer)
        b = get_default_backend()
        hidden: list[dict[int, object]] = []
        intermediate: list[dict[int, object]] = []
        gate: list[dict[int, object]] = []
        embedding: list[object] = []

        for i, _text in enumerate(texts):
            row: dict[int, object] = {}
            for layer_idx in (0, 1, 2):
                row[layer_idx] = b.array(
                    [float(i), float(layer_idx), 1.0, 0.0],
                    dtype="float32",
                )
            hidden.append(row)
            intermediate.append({})
            gate.append({})
            embedding.append(b.zeros((4,), dtype="float32"))

        return ProbeActivationBatch(
            hidden=hidden,
            intermediate=intermediate,
            gate=gate,
            embedding=embedding,
        )


@dataclass(frozen=True)
class _FakeDecisionAction:
    value: str = "proceed"


@dataclass(frozen=True)
class _FakeDecision:
    action: _FakeDecisionAction = field(default_factory=_FakeDecisionAction)


@dataclass
class _FakeRunState:
    token_id: int | None
    thinking_iterations: int = 0
    encoding_results: list[object] = field(default_factory=list)
    surprise_event: object | None = None
    probe_embedding: object | None = None
    decision: _FakeDecision = field(default_factory=_FakeDecision)
    attractor_state: object | None = None


class _FakeModel:
    def parameters(self):
        # Empty weights are sufficient for the save/load control test (save is patched).
        return []


class _FakeGeometricInferenceRun:
    def __init__(self, model, backend) -> None:
        _ = (model, backend)
        self._stats = {
            "null_space_state": {
                "capacity_fraction": 0.123,
                "total_variance": 1.0,
                "null_variance": 1.0,
            },
            "attractor": {"escape_count": 0},
        }

    def generate(self, input_ids, seed_embedding=None, append_tokens=True):
        _ = (input_ids, seed_embedding, append_tokens)
        yield _FakeRunState(token_id=1)

    def get_stats(self) -> dict:
        return self._stats


class TestGenesisValidateCommand:
    def test_genesis_validate_reference_cka_linear(self, tmp_path):
        model_dir = tmp_path / "model"
        ref_dir = tmp_path / "ref"
        model_dir.mkdir()
        ref_dir.mkdir()

        fake_model_a = object()
        fake_model_b = object()
        fake_tokenizer = _FakeTokenizer()

        with patch("mlx_lm.load", side_effect=[(fake_model_a, fake_tokenizer), (fake_model_b, fake_tokenizer)]):
            with patch(
                "modelcypher.cli.composition.get_activation_provider",
                return_value=_FakeActivationProvider(),
            ):
                with patch(
                    "modelcypher.core.domain.continual.geometric_inference.GeometricInference",
                    _FakeGeometricInference,
                ):
                    result = runner.invoke(
                        app,
                        [
                            "genesis",
                            "validate",
                            "--model",
                            str(model_dir),
                            "--reference",
                            str(ref_dir),
                            "--output",
                            "json",
                        ],
                    )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["model"] == str(model_dir)
        assert payload["cka_comparison"]["status"] == "computed"
        assert payload["cka_comparison"]["kernel"] == "linear"
        assert payload["cka_comparison"]["probe_count"] >= 2
        assert payload["cka_comparison"]["layers_compared"] == [0, 1, 2]
        assert payload["cka_comparison"]["cka_min"] == pytest.approx(1.0, abs=1e-6)
        assert payload["cka_comparison"]["cka_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_genesis_validate_reference_cka_probes_file(self, tmp_path):
        model_dir = tmp_path / "model"
        ref_dir = tmp_path / "ref"
        model_dir.mkdir()
        ref_dir.mkdir()
        probes_path = tmp_path / "probes.txt"
        probes_path.write_text("probe a\nprobe b\nprobe c\n")

        fake_model_a = object()
        fake_model_b = object()
        fake_tokenizer = _FakeTokenizer()

        with patch("mlx_lm.load", side_effect=[(fake_model_a, fake_tokenizer), (fake_model_b, fake_tokenizer)]):
            with patch(
                "modelcypher.cli.composition.get_activation_provider",
                return_value=_FakeActivationProvider(),
            ):
                with patch(
                    "modelcypher.core.domain.continual.geometric_inference.GeometricInference",
                    _FakeGeometricInference,
                ):
                    result = runner.invoke(
                        app,
                        [
                            "genesis",
                            "validate",
                            "--model",
                            str(model_dir),
                            "--reference",
                            str(ref_dir),
                            "--cka-probes",
                            str(probes_path),
                            "--output",
                            "json",
                        ],
                    )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["cka_comparison"]["probe_count"] == 3
        assert payload["cka_comparison"]["probes"] == ["probe a", "probe b", "probe c"]


class TestGenesisRunCommand:
    def test_genesis_run_reports_cka_preservation(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        fake_model = _FakeModel()
        fake_tokenizer = _FakeTokenizer()

        with patch("mlx_lm.load", return_value=(fake_model, fake_tokenizer)):
            with patch(
                "modelcypher.cli.composition.get_activation_provider",
                return_value=_FakeActivationProvider(),
            ):
                with patch(
                    "modelcypher.core.domain.continual.geometric_inference.GeometricInference",
                    _FakeGeometricInferenceRun,
                ):
                    result = runner.invoke(
                        app,
                        [
                            "--ai",
                            "genesis",
                            "run",
                            "--model",
                            str(model_dir),
                            "--prompt",
                            "hello",
                        ],
                    )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["genesis"]["capacity_remaining"] == pytest.approx(0.123, abs=1e-9)
        assert payload["genesis"]["cka_preserved"] == pytest.approx(1.0, abs=1e-6)
        assert payload["cka"]["status"] == "computed"
        assert payload["cka"]["kernel"] == "linear"
        assert payload["cka"]["layers_compared"] == [0, 1, 2]
        assert payload["cka"]["cka_min"] == pytest.approx(1.0, abs=1e-6)
        assert payload["cka"]["cka_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_genesis_run_cka_save_load_control(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        fake_model = _FakeModel()
        fake_tokenizer = _FakeTokenizer()

        with patch("mlx_lm.load", side_effect=[(fake_model, fake_tokenizer), (fake_model, fake_tokenizer)]):
            with patch(
                "modelcypher.cli.composition.get_activation_provider",
                return_value=_FakeActivationProvider(),
            ):
                with patch(
                    "modelcypher.core.domain.continual.geometric_inference.GeometricInference",
                    _FakeGeometricInferenceRun,
                ):
                    with patch("mlx.core.save_safetensors", return_value=None):
                        result = runner.invoke(
                            app,
                            [
                                "--ai",
                                "genesis",
                                "run",
                                "--model",
                                str(model_dir),
                                "--prompt",
                                "hello",
                                "--cka-control",
                                "save-load",
                            ],
                        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["cka"]["control"]["status"] in {"computed", "failed"}
        if payload["cka"]["control"]["status"] == "computed":
            assert payload["cka"]["control"]["cka_mean"] == pytest.approx(1.0, abs=1e-6)


class TestGenesisStatusCommand:
    def test_genesis_status_includes_cka_summary(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        metadata = {
            "genesis_timestamp": "2026-01-30T00:00:00",
            "source_model": str(model_dir),
            "cka": {
                "kernel": "linear",
                "probe_count": 3,
                "cka_min": 0.91,
                "cka_mean": 0.95,
                "layers_compared": [0, 1],
                "control": {
                    "status": "computed",
                    "cka_min": 0.99,
                    "cka_mean": 0.995,
                    "probe_count": 3,
                },
            },
        }
        (model_dir / "genesis_metadata.json").write_text(json.dumps(metadata))

        result = runner.invoke(
            app,
            ["genesis", "status", "--model", str(model_dir), "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["has_genesis"] is True
        assert payload["cka_summary"]["kernel"] == "linear"
        assert payload["cka_summary"]["probe_count"] == 3
        assert payload["cka_summary"]["cka_min"] == pytest.approx(0.91, abs=1e-9)
        assert payload["cka_summary"]["cka_mean"] == pytest.approx(0.95, abs=1e-9)
        assert payload["cka_summary"]["layers_compared"] == [0, 1]
        assert payload["cka_summary"]["control"]["status"] == "computed"
