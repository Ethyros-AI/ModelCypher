# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from modelcypher.core.domain.training.exceptions import TrainingDerivationError
from modelcypher.core.use_cases.dataset_training_service import DatasetTrainingService


@dataclass
class _FloatInfo:
    eps: float


class _DummyBackend:
    def random_seed(self, _seed: int) -> None:
        return None

    def load_model(self, _model_path: str):
        return object(), object()

    def finfo(self, _dtype=None):
        return _FloatInfo(eps=2.0 ** -23)


class _DummyAdapter:
    pass


@dataclass
class _Geom:
    sigma_k: float
    sigma_max: float
    tail_dims: int
    spectral_gap: float


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_pilot_variance_split_meets_target_standard_error():
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())

    losses = [
        1.0,
        1.0001,
        0.9999,
        1.00005,
        0.99995,
        1.00002,
        0.99998,
        1.0,
        1.00003,
        0.99997,
    ]
    n_eval, info = service._derive_validation_split_from_losses(
        sample_losses=losses,
        n_total=len(losses),
    )

    assert 0 < n_eval < len(losses)

    val_losses = losses[:n_eval]
    mean_loss = sum(val_losses) / len(val_losses)
    target_se = info["sqrt_eps"] * max(1.0, abs(mean_loss))
    variance = (
        sum((value - mean_loss) ** 2 for value in val_losses) / (len(val_losses) - 1)
        if len(val_losses) > 1
        else 0.0
    )
    standard_error = math.sqrt(variance / len(val_losses))
    assert standard_error <= target_se


def test_answer_mask_without_answer_start_fails_fast(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    rows = [
        {"text": "Q: 1+1? A: 2"},
        {"text": "Q: 2+2? A: 4"},
    ]
    _write_jsonl(train_path, rows)
    _write_jsonl(eval_path, rows)

    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        service.train_from_dataset(
            model_path=model_dir,
            dataset_path=train_path,
            eval_dataset_path=eval_path,
            answer_mask=True,
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_answer_mask_metadata"
    assert err.diagnostics["missing_answer_start_count"] == len(rows) * 2


def test_geometry_manifest_written_with_sigma_k(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())
    service._write_geometry_manifest(
        adapter_dir=adapter_dir,
        model_path=model_dir,
        target_modules=["model.layers.0.self_attn.q_proj.weight"],
        geometries={
            "model.layers.0.self_attn.q_proj.weight": _Geom(
                sigma_k=1.5,
                sigma_max=2.0,
                tail_dims=8,
                spectral_gap=0.2,
            ),
        },
    )

    manifest_path = adapter_dir / "geometry_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["sigma_k_by_module"]["model.layers.0.self_attn.q_proj.weight"] == pytest.approx(1.5)
