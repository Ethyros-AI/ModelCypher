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

"""Tests for derived_training_validation_service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.derived_training_validation_service import (
    DerivedTrainingValidationService,
)


@dataclass
class _FakeTrainResult:
    baseline_loss: float
    post_loss: float
    baseline_perplexity: float
    post_perplexity: float
    spectral_bounds_ok: bool = True
    max_spectral_ratio: float = 0.0
    stop_reason: str = "stop"
    train_iters: int = 10
    training_time_seconds: float = 1.0
    min_cka: float | None = None
    mean_cka: float | None = None
    adapter_saturation_median_ratio: float | None = None


class _FakeDatasetTrainingService:
    def __init__(self, results: list[_FakeTrainResult]):
        self._results = list(results)
        self.calls: list[dict] = []

    def train_from_dataset_research(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self._results:
            raise RuntimeError("No fake result available")
        return self._results.pop(0)


def test_validate_all_trials_pass_when_metrics_improve(tmp_path):
    fake = _FakeDatasetTrainingService(
        results=[
            _FakeTrainResult(
                baseline_loss=2.0,
                post_loss=1.4,
                baseline_perplexity=7.0,
                post_perplexity=4.0,
            ),
            _FakeTrainResult(
                baseline_loss=2.2,
                post_loss=1.6,
                baseline_perplexity=8.1,
                post_perplexity=5.0,
            ),
        ]
    )
    service = DerivedTrainingValidationService(
        dataset_training_service=fake,
        backend=get_default_backend(),
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"text":"hello"}\n', encoding="utf-8")

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=2,
        base_seed=100,
    )

    assert result.all_passed is True
    assert result.pass_count == 2
    assert result.fail_count == 0
    assert result.seeds == (100, 101)
    assert len(result.counterexamples) == 0
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["auto_regime"] is True
        assert call["no_save"] is True


def test_validate_records_counterexamples(tmp_path):
    fake = _FakeDatasetTrainingService(
        results=[
            _FakeTrainResult(
                baseline_loss=1.0,
                post_loss=1.2,
                baseline_perplexity=3.0,
                post_perplexity=3.4,
            ),
        ]
    )
    service = DerivedTrainingValidationService(
        dataset_training_service=fake,
        backend=get_default_backend(),
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"text":"hello"}\n', encoding="utf-8")

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=7,
    )

    assert result.all_passed is False
    assert result.pass_count == 0
    assert result.fail_count == 1
    assert len(result.counterexamples) == 1
    failure = result.counterexamples[0]
    assert "loss_not_improved" in failure.failure_modes
    assert "perplexity_not_improved" in failure.failure_modes


def test_validate_requires_positive_trial_count(tmp_path):
    fake = _FakeDatasetTrainingService(results=[])
    service = DerivedTrainingValidationService(
        dataset_training_service=fake,
        backend=get_default_backend(),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"text":"hello"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="trials must be positive"):
        service.validate(
            model_path=Path(model_dir),
            dataset_path=Path(data_file),
            eval_dataset_path=None,
            trials=0,
        )


def _make_service(tmp_path, results):
    fake = _FakeDatasetTrainingService(results=results)
    service = DerivedTrainingValidationService(
        dataset_training_service=fake,
        backend=get_default_backend(),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"text":"hello"}\n', encoding="utf-8")
    return service, model_dir, data_file


def test_safety_cap_hit_is_failure(tmp_path):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            stop_reason="safety_cap (100000 iters)",
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert result.all_passed is False
    assert "safety_cap_hit" in result.counterexamples[0].failure_modes


def test_cka_degraded_is_failure(tmp_path):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            min_cka=0.5,
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert result.all_passed is False
    assert "cka_degraded" in result.counterexamples[0].failure_modes


def test_adapter_saturation_exceeded_is_failure(tmp_path):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            adapter_saturation_median_ratio=1.01,
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert result.all_passed is False
    assert "adapter_saturation_exceeded" in result.counterexamples[0].failure_modes


def test_new_gates_pass_when_healthy(tmp_path):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            stop_reason="loss_stable",
            min_cka=0.9999,
            adapter_saturation_median_ratio=0.85,
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert result.all_passed is True
    trial = result.trial_results[0]
    assert trial.adapter_saturation_median_ratio == pytest.approx(0.85)
    assert trial.min_cka == pytest.approx(0.9999)


def test_trial_to_dict_includes_adapter_saturation(tmp_path):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            adapter_saturation_median_ratio=0.75,
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    d = result.trial_results[0].to_dict()
    assert "adapter_saturation_median_ratio" in d
    assert d["adapter_saturation_median_ratio"] == pytest.approx(0.75)
