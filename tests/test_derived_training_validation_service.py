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
    _Phase5Metrics,
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
    per_layer_cka: dict[int, float] | None = None
    per_layer_gram_epsilon: dict[int, float] | None = None
    per_layer_cka_bound: dict[int, float] | None = None
    min_cka_layer: int | None = None
    adapter_saturation_median_ratio: float | None = None
    max_effective_gain_ratio: float | None = None
    seq_length_used: int = 64
    adapter_path: str | None = "adapter-path"


class _FakeDatasetTrainingService:
    def __init__(self, results: list[_FakeTrainResult]):
        self._results = list(results)
        self.calls: list[dict] = []
        self.derive_regime_calls = 0

    def train_from_dataset_research(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self._results:
            raise RuntimeError("No fake result available")
        return self._results.pop(0)

    def _derive_regime_n_from_ci(self) -> int:
        self.derive_regime_calls += 1
        return 2


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


def test_cka_shift_without_bound_data_is_not_failure(tmp_path):
    """CKA shift alone (no bound data) is diagnostic, not a structural fail."""
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
    # CKA shift without bound data: no structural failure.
    assert result.all_passed is True
    trial = result.trial_results[0]
    assert "cka_bound_violation" not in trial.failure_modes
    assert trial.cka_margin_to_bound is None


def test_cka_bound_violation_is_failure(tmp_path):
    """Actual CKA below theoretical bound triggers structural failure."""
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            min_cka=0.5,
            per_layer_cka={0: 0.99, 1: 0.5},
            per_layer_cka_bound={0: 0.95, 1: 0.9},
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert result.all_passed is False
    trial = result.counterexamples[0]
    assert "cka_bound_violation" in trial.failure_modes
    # margin = min(0.99-0.95, 0.5-0.9) = min(0.04, -0.4) = -0.4
    assert trial.cka_margin_to_bound == pytest.approx(-0.4, abs=0.01)


def test_cka_margin_positive_no_violation(tmp_path):
    """When actual CKA >= theoretical bound everywhere, no violation."""
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            min_cka=0.93,
            per_layer_cka={0: 0.99, 1: 0.93},
            per_layer_cka_bound={0: 0.85, 1: 0.47},
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert result.all_passed is True
    trial = result.trial_results[0]
    assert "cka_bound_violation" not in trial.failure_modes
    # margin = min(0.99-0.85, 0.93-0.47) = min(0.14, 0.46) = 0.14
    assert trial.cka_margin_to_bound == pytest.approx(0.14, abs=0.01)


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


def test_gain_divergence_is_failure(tmp_path):
    """Bounded-gain violation triggers gain_divergence failure mode."""
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            max_effective_gain_ratio=1.5,  # > 1.0 + sqrt(eps)
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert result.all_passed is False
    assert "gain_divergence" in result.counterexamples[0].failure_modes


def test_gain_bounded_is_not_failure(tmp_path):
    """Gain ratio at or below ceiling does not trigger failure."""
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            max_effective_gain_ratio=0.95,  # <= 1.0
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert "gain_divergence" not in result.trial_results[0].failure_modes


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


def test_phase5_cka_shift_inference_healthy(tmp_path, monkeypatch):
    """CKA shift with healthy inference: structural passes (shift is diagnostic),
    inference passes, co-occurrence shows cka_shift_without_inference_degradation."""
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.0,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            min_cka=0.5,
            per_layer_cka={0: 0.99, 1: 0.5},
            per_layer_cka_bound={0: 0.85, 1: 0.40},
            min_cka_layer=1,
        ),
    ])

    monkeypatch.setattr(
        service,
        "_run_phase5_for_trial",
        lambda **_kwargs: _Phase5Metrics(
            baseline_n_correct=8,
            baseline_n_total=10,
            adapted_n_correct=8,
            adapted_n_total=10,
            baseline_max_4gram_repeat=0.10,
            adapted_max_4gram_repeat=0.10,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts",
    )

    trial = result.trial_results[0]
    # CKA shift is diagnostic only; actual >= bound, so no structural failure.
    assert trial.structural_passed is True
    assert trial.inference_passed is True
    assert "cka_bound_violation" not in trial.failure_modes
    assert "online_eval_degraded" not in trial.failure_modes
    assert "fourgram_degenerated" not in trial.failure_modes
    assert trial.cooccurrence_class == "cka_shift_without_inference_degradation"


def test_phase5_online_eval_drop_triggers_failure(tmp_path, monkeypatch):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.0,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            min_cka=0.99999,
        ),
    ])

    monkeypatch.setattr(
        service,
        "_run_phase5_for_trial",
        lambda **_kwargs: _Phase5Metrics(
            baseline_n_correct=9,
            baseline_n_total=10,
            adapted_n_correct=8,
            adapted_n_total=10,
            baseline_max_4gram_repeat=0.10,
            adapted_max_4gram_repeat=0.10,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts",
    )

    trial = result.trial_results[0]
    assert "online_eval_degraded" in trial.failure_modes
    assert trial.inference_passed is False
    assert trial.structural_passed is True
    assert result.inference_fail_count == 1
    assert result.structural_fail_count == 0


def test_phase5_fourgram_crossing_triggers_failure(tmp_path, monkeypatch):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.0,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            min_cka=0.99999,
        ),
    ])

    monkeypatch.setattr(
        service,
        "_run_phase5_for_trial",
        lambda **_kwargs: _Phase5Metrics(
            baseline_n_correct=8,
            baseline_n_total=10,
            adapted_n_correct=8,
            adapted_n_total=10,
            baseline_max_4gram_repeat=0.10,
            adapted_max_4gram_repeat=0.25,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts",
    )

    trial = result.trial_results[0]
    assert "fourgram_degenerated" in trial.failure_modes
    assert trial.inference_passed is False
    assert trial.structural_passed is True


def test_phase5_probe_derivation_is_deterministic(tmp_path, monkeypatch):
    service_a, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.0,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
        ),
    ])
    service_b, _, _ = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.0,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
        ),
    ])

    monkeypatch.setattr(
        service_a,
        "_run_phase5_for_trial",
        lambda **_kwargs: _Phase5Metrics(
            baseline_n_correct=8,
            baseline_n_total=10,
            adapted_n_correct=8,
            adapted_n_total=10,
            baseline_max_4gram_repeat=0.10,
            adapted_max_4gram_repeat=0.10,
        ),
    )
    monkeypatch.setattr(
        service_b,
        "_run_phase5_for_trial",
        lambda **_kwargs: _Phase5Metrics(
            baseline_n_correct=8,
            baseline_n_total=10,
            adapted_n_correct=8,
            adapted_n_total=10,
            baseline_max_4gram_repeat=0.10,
            adapted_max_4gram_repeat=0.10,
        ),
    )

    result_a = service_a.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts_a",
    )
    result_b = service_b.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts_b",
    )

    assert result_a.phase5_probe_count == 10
    assert result_b.phase5_probe_count == 10
    assert result_a.phase5_probe_seed == result_b.phase5_probe_seed
    assert service_a._dataset_training_service.derive_regime_calls == 1
    assert service_b._dataset_training_service.derive_regime_calls == 1
