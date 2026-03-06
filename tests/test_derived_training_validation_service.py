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
from modelcypher.core.domain.training.online_eval import OnlineEvalResult
from modelcypher.core.use_cases.derived_training_validation_service import (
    DerivedTrainingValidationService,
    _Phase5Context,
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
    per_layer_null_observability: dict[int, dict[str, float | int]] | None = None
    per_layer_null_accessibility: dict[int, dict[str, float | int]] | None = None
    per_module_null_accessibility: dict[str, dict[str, float | int]] | None = None
    min_cka_layer: int | None = None
    adapter_saturation_median_ratio: float | None = None
    max_effective_gain_ratio: float | None = None
    epoch_metrics: list[dict[str, object]] | None = None
    seq_length_used: int = 64
    moe_router_stability: float | None = None
    mode_connectivity_barrier: float | None = None
    mode_connectivity_normalized_barrier: float | None = None
    mode_connectivity_method: str | None = None
    degeneration_max_ngram_repeat: float | None = None
    degeneration_mean_ngram_repeat: float | None = None
    degeneration_ngram_order: int | None = None
    rss_final_cosine: float | None = None
    rss_final_spearman: float | None = None
    rss_final_top1: float | None = None
    per_layer_signal_ranks: dict[int, dict[str, float | int]] | None = None
    inference_per_layer_cka: dict[int, float] | None = None
    moe_saturated_during_training: list[str] | None = None
    moe_targets: object | None = None
    dim_final_used_fraction: float | None = None
    dim_final_null_fraction: float | None = None
    dim_null_recruitment_from_baseline: float | None = None
    benchmark_baseline: dict[str, float] | None = None
    benchmark_post: dict[str, float] | None = None
    adapter_path: str | None = "adapter-path"


class _FakeDatasetTrainingService:
    def __init__(self, results: list[_FakeTrainResult]):
        self._results = list(results)
        self.calls: list[dict] = []

    def train_from_dataset_research(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self._results:
            raise RuntimeError("No fake result available")
        return self._results.pop(0)



def _make_online_eval_result(
    *,
    n_correct: int,
    n_total: int,
    correct_ids: set[str],
    epoch: int = 0,
) -> OnlineEvalResult:
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return OnlineEvalResult(
        epoch=epoch,
        accuracy=accuracy,
        n_correct=n_correct,
        n_total=n_total,
        correct_ids=frozenset(correct_ids),
        baseline_n_correct=n_correct,
        baseline_accuracy=accuracy,
        n_lost=0,
        n_gained=0,
        degraded=False,
    )


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


def test_shared_pipeline_gate_reports_spectral_failure(tmp_path):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            spectral_bounds_ok=False,
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert result.all_passed is False
    assert "spectral_bounds_violation" in result.counterexamples[0].failure_modes


def test_shared_pipeline_gate_keeps_unresolved_core_non_blocking_in_validation(tmp_path):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            spectral_bounds_ok=None,
            per_layer_cka=None,
            per_layer_cka_bound=None,
            stop_reason="certificate",
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    assert result.all_passed is True
    trial = result.trial_results[0]
    assert "spectral_bounds_unavailable" not in trial.failure_modes
    assert "cka_bound_unavailable" not in trial.failure_modes


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


def test_moe_router_stability_round_trips_through_trial(tmp_path):
    """moe_router_stability extracted from train result and serialized in to_dict()."""
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            moe_router_stability=0.00123,
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    trial = result.trial_results[0]
    assert trial.moe_router_stability == pytest.approx(0.00123)
    d = trial.to_dict()
    assert "moe_router_stability" in d
    assert d["moe_router_stability"] == pytest.approx(0.00123)


def test_moe_router_stability_none_when_absent(tmp_path):
    """moe_router_stability is None for non-MoE models."""
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    trial = result.trial_results[0]
    assert trial.moe_router_stability is None
    d = trial.to_dict()
    assert d["moe_router_stability"] is None


@dataclass
class _FakeMoETargets:
    n_trainable_experts: int = 4
    saturated: list = None
    skipped: list = None

    def __post_init__(self):
        if self.saturated is None:
            self.saturated = [(0, 1)]
        if self.skipped is None:
            self.skipped = [(0, 2), (0, 3)]


def test_all_diagnostic_fields_round_trip(tmp_path):
    """All 21 new diagnostic fields round-trip through _build_trial → to_dict()."""
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            # Group A: Mode connectivity
            mode_connectivity_barrier=0.015,
            mode_connectivity_normalized_barrier=0.003,
            mode_connectivity_method="linear_interpolation",
            # Group B: Train-time degeneration
            degeneration_max_ngram_repeat=0.12,
            degeneration_mean_ngram_repeat=0.05,
            degeneration_ngram_order=4,
            # Group C: RSS similarity
            rss_final_cosine=0.987,
            rss_final_spearman=0.945,
            rss_final_top1=0.92,
            # Group D: Signal ranks and inference CKA
            per_layer_signal_ranks={
                0: {"signal_rank": 12, "noise_rank": 52},
                1: {"signal_rank": 8, "noise_rank": 56},
            },
            inference_per_layer_cka={0: 0.998, 1: 0.995},
            # Group E: MoE diagnostics
            moe_saturated_during_training=["layers.0.experts.1"],
            moe_targets=_FakeMoETargets(),
            # Group F: Dimensional recruitment and benchmarks
            dim_final_used_fraction=0.41,
            dim_final_null_fraction=0.59,
            dim_null_recruitment_from_baseline=0.03,
            benchmark_baseline={"overall": 0.72, "math": 0.65},
            benchmark_post={"overall": 0.78, "math": 0.71},
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    trial = result.trial_results[0]
    d = trial.to_dict()

    # Group A
    assert trial.mode_connectivity_barrier == pytest.approx(0.015)
    assert d["mode_connectivity_barrier"] == pytest.approx(0.015)
    assert trial.mode_connectivity_normalized_barrier == pytest.approx(0.003)
    assert d["mode_connectivity_normalized_barrier"] == pytest.approx(0.003)
    assert trial.mode_connectivity_method == "linear_interpolation"
    assert d["mode_connectivity_method"] == "linear_interpolation"

    # Group B
    assert trial.degeneration_max_ngram_repeat == pytest.approx(0.12)
    assert d["degeneration_max_ngram_repeat"] == pytest.approx(0.12)
    assert trial.degeneration_mean_ngram_repeat == pytest.approx(0.05)
    assert d["degeneration_mean_ngram_repeat"] == pytest.approx(0.05)
    assert trial.degeneration_ngram_order == 4
    assert d["degeneration_ngram_order"] == 4

    # Group C
    assert trial.rss_final_cosine == pytest.approx(0.987)
    assert d["rss_final_cosine"] == pytest.approx(0.987)
    assert trial.rss_final_spearman == pytest.approx(0.945)
    assert d["rss_final_spearman"] == pytest.approx(0.945)
    assert trial.rss_final_top1 == pytest.approx(0.92)
    assert d["rss_final_top1"] == pytest.approx(0.92)

    # Group D
    assert trial.per_layer_signal_ranks is not None
    assert trial.per_layer_signal_ranks[0]["signal_rank"] == 12
    assert d["per_layer_signal_ranks"][1]["noise_rank"] == 56
    assert trial.inference_per_layer_cka is not None
    assert trial.inference_per_layer_cka[0] == pytest.approx(0.998)
    assert d["inference_per_layer_cka"][1] == pytest.approx(0.995)

    # Group E
    assert trial.moe_saturated_during_training == ["layers.0.experts.1"]
    assert d["moe_saturated_during_training"] == ["layers.0.experts.1"]
    assert trial.moe_n_targets == 4
    assert d["moe_n_targets"] == 4
    assert trial.moe_n_saturated == 1
    assert d["moe_n_saturated"] == 1
    assert trial.moe_n_skipped == 2
    assert d["moe_n_skipped"] == 2

    # Group F
    assert trial.dim_final_used_fraction == pytest.approx(0.41)
    assert d["dim_final_used_fraction"] == pytest.approx(0.41)
    assert trial.dim_final_null_fraction == pytest.approx(0.59)
    assert d["dim_final_null_fraction"] == pytest.approx(0.59)
    assert trial.dim_null_recruitment_from_baseline == pytest.approx(0.03)
    assert d["dim_null_recruitment_from_baseline"] == pytest.approx(0.03)
    assert trial.benchmark_baseline == {"overall": 0.72, "math": 0.65}
    assert d["benchmark_baseline"] == {"overall": 0.72, "math": 0.65}
    assert trial.benchmark_post == {"overall": 0.78, "math": 0.71}
    assert d["benchmark_post"] == {"overall": 0.78, "math": 0.71}
    # Computed benchmark_delta
    assert d["benchmark_delta"]["overall"] == pytest.approx(0.06)
    assert d["benchmark_delta"]["math"] == pytest.approx(0.06)


def test_diagnostic_fields_none_when_absent(tmp_path):
    """All diagnostic fields are None for models that don't produce them."""
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
        ),
    ])
    result = service.validate(
        model_path=model_dir, dataset_path=data_file,
        eval_dataset_path=None, trials=1, base_seed=1,
    )
    trial = result.trial_results[0]
    d = trial.to_dict()

    assert trial.mode_connectivity_barrier is None
    assert trial.degeneration_max_ngram_repeat is None
    assert trial.rss_final_cosine is None
    assert trial.per_layer_signal_ranks is None
    assert trial.inference_per_layer_cka is None
    assert trial.moe_saturated_during_training is None
    assert trial.moe_n_targets is None
    assert trial.dim_final_used_fraction is None
    assert trial.benchmark_baseline is None
    assert trial.benchmark_post is None
    assert "benchmark_delta" not in d


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
            baseline_max_ngram_repeat=0.10,
            adapted_max_ngram_repeat=0.10,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        phase5_probe_count=10,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts",
    )

    trial = result.trial_results[0]
    # CKA shift is diagnostic only; actual >= bound, so no structural failure.
    assert trial.structural_passed is True
    assert trial.inference_passed is True
    assert "cka_bound_violation" not in trial.failure_modes
    assert "online_eval_degraded" not in trial.failure_modes
    assert "ngram_degenerated" not in trial.failure_modes
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
            baseline_max_ngram_repeat=0.10,
            adapted_max_ngram_repeat=0.10,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        phase5_probe_count=10,
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
            baseline_max_ngram_repeat=0.10,
            adapted_max_ngram_repeat=0.25,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        phase5_probe_count=10,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts",
    )

    trial = result.trial_results[0]
    assert "ngram_degenerated" in trial.failure_modes
    assert trial.inference_passed is False
    assert trial.structural_passed is True


def test_argmax_not_certified_triggers_inference_failure(tmp_path, monkeypatch):
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
            baseline_max_ngram_repeat=0.10,
            adapted_max_ngram_repeat=0.10,
            max_logit_delta_inf=0.40,
            argmax_cert_gap=-0.05,
            argmax_preservation_certified=False,
            argmax_n_correct_flipped=1,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        phase5_probe_count=10,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts",
    )

    trial = result.trial_results[0]
    assert "argmax_not_certified" in trial.failure_modes
    assert trial.inference_passed is False
    assert result.all_passed is False
    assert trial.argmax_n_correct_flipped == 1


def test_argmax_certified_does_not_trigger_failure(tmp_path, monkeypatch):
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
            baseline_max_ngram_repeat=0.10,
            adapted_max_ngram_repeat=0.10,
            max_logit_delta_inf=0.05,
            argmax_cert_gap=0.25,
            argmax_preservation_certified=True,
            argmax_n_correct_flipped=0,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        phase5_probe_count=10,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts",
    )

    trial = result.trial_results[0]
    assert "argmax_not_certified" not in trial.failure_modes
    assert trial.inference_passed is True
    assert trial.argmax_n_correct_flipped == 0


def test_argmax_cert_none_does_not_trigger_failure(tmp_path, monkeypatch):
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
            baseline_max_ngram_repeat=0.10,
            adapted_max_ngram_repeat=0.10,
            max_logit_delta_inf=None,
            argmax_cert_gap=None,
            argmax_preservation_certified=None,
            argmax_n_correct_flipped=None,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        phase5_probe_count=10,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts",
    )

    trial = result.trial_results[0]
    assert "argmax_not_certified" not in trial.failure_modes
    assert trial.inference_passed is True


@pytest.mark.parametrize(
    ("adapted_margins", "expected_gap", "expected_certified", "expected_flipped"),
    [
        # Both correct probes keep positive margin → certified
        ({"p1": 0.75, "p2": 0.55}, 0.55, True, 0),
        # One correct probe margin goes negative → not certified
        ({"p1": 0.75, "p2": -0.10}, -0.10, False, 1),
        # Both correct probes flip → not certified
        ({"p1": -0.3, "p2": -0.1}, -0.3, False, 2),
    ],
)
def test_phase5_argmax_certificate_from_adapted_margins(
    tmp_path,
    monkeypatch,
    adapted_margins,
    expected_gap,
    expected_certified,
    expected_flipped,
):
    service, model_dir, _ = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.0,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
        ),
    ])

    context = _Phase5Context(
        enabled=True,
        probe_count=3,
        probe_seed=1,
        artifact_root=tmp_path / "artifacts",
        keep_all_artifacts=True,
        probe_problems=[],
    )

    # p1, p2 correct at baseline; p3 incorrect at baseline
    baseline_eval = _make_online_eval_result(
        n_correct=2,
        n_total=3,
        correct_ids={"p1", "p2"},
    )
    adapted_eval = _make_online_eval_result(
        n_correct=2,
        n_total=3,
        correct_ids={"p1", "p2"},
    )

    # p3 has margin -0.5 (baseline wrong) — must not affect cert
    full_adapted_margins = dict(adapted_margins)
    full_adapted_margins["p3"] = -0.5

    def _fake_run_probe_eval(**kwargs):
        if kwargs["adapter_path"] is None:
            return (
                baseline_eval,
                0.10,
                {"p1": 0.8, "p2": 0.6, "p3": -0.5},
                None,
                {"p1": object(), "p2": object(), "p3": object()},
                4,  # ngram_order
            )
        return (
            adapted_eval,
            0.10,
            full_adapted_margins,
            0.5,
            None,
            4,  # ngram_order
        )

    monkeypatch.setattr(service, "_run_probe_eval", _fake_run_probe_eval)

    metrics = service._run_phase5_for_trial(
        context=context,
        model_path=model_dir,
        train_result=_FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.0,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
        ),
        epoch_index=0,
    )

    assert metrics.argmax_cert_gap == pytest.approx(expected_gap, abs=1e-12)
    assert metrics.argmax_preservation_certified is expected_certified
    assert metrics.argmax_n_correct_flipped == expected_flipped


def test_phase5_argmax_certificate_fields_round_trip_in_trial_dict(
    tmp_path,
    monkeypatch,
):
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
            baseline_max_ngram_repeat=0.10,
            adapted_max_ngram_repeat=0.10,
            max_logit_delta_inf=0.12,
            argmax_cert_gap=0.36,
            argmax_preservation_certified=True,
            argmax_n_correct_flipped=0,
        ),
    )

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        phase5_probe_count=10,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts",
    )

    trial = result.trial_results[0]
    payload = trial.to_dict()
    assert trial.max_logit_delta_inf == pytest.approx(0.12)
    assert trial.argmax_cert_gap == pytest.approx(0.36)
    assert trial.argmax_preservation_certified is True
    assert trial.argmax_n_correct_flipped == 0
    assert payload["max_logit_delta_inf"] == pytest.approx(0.12)
    assert payload["argmax_cert_gap"] == pytest.approx(0.36)
    assert payload["argmax_preservation_certified"] is True
    assert payload["argmax_n_correct_flipped"] == 0


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
            baseline_max_ngram_repeat=0.10,
            adapted_max_ngram_repeat=0.10,
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
            baseline_max_ngram_repeat=0.10,
            adapted_max_ngram_repeat=0.10,
        ),
    )

    result_a = service_a.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        phase5_probe_count=10,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts_a",
    )
    result_b = service_b.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=1,
        phase5_probe_count=10,
        enable_phase5_inference=True,
        artifact_root=tmp_path / "artifacts_b",
    )

    assert result_a.phase5_probe_count == 10
    assert result_b.phase5_probe_count == 10
    assert result_a.phase5_probe_seed == result_b.phase5_probe_seed


def test_null_space_diagnostics_round_trip_and_summaries(tmp_path):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            per_layer_null_observability={
                0: {"condition_number": 12.0, "coverage_ratio": 1.2},
                1: {"condition_number": 20.0, "coverage_ratio": 1.3},
            },
            per_layer_null_accessibility={
                0: {"behavioral_preserved_fraction": 0.41, "module_count": 3},
                1: {"behavioral_preserved_fraction": 0.22, "module_count": 3},
            },
            per_module_null_accessibility={
                "model.layers.0.self_attn.q_proj.weight": {
                    "behavioral_preserved_fraction": 0.50,
                    "condition_number": 12.0,
                },
                "model.layers.1.self_attn.q_proj.weight": {
                    "behavioral_preserved_fraction": 0.20,
                    "condition_number": 20.0,
                },
            },
        ),
    ])

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=11,
    )
    trial = result.trial_results[0]

    assert trial.per_layer_null_observability is not None
    assert trial.per_layer_null_accessibility is not None
    assert trial.per_module_null_accessibility is not None
    assert trial.null_access_min_behavioral_preserved_fraction == pytest.approx(0.22)
    assert trial.null_access_min_behavioral_preserved_layer == 1
    assert trial.null_observability_max_condition_number == pytest.approx(20.0)
    assert trial.null_observability_max_condition_layer == 1

    payload = trial.to_dict()
    assert payload["per_layer_null_observability"][1]["condition_number"] == pytest.approx(20.0)
    assert payload["per_layer_null_accessibility"][1]["behavioral_preserved_fraction"] == pytest.approx(0.22)
    assert (
        payload["per_module_null_accessibility"][
            "model.layers.1.self_attn.q_proj.weight"
        ]["behavioral_preserved_fraction"]
        == pytest.approx(0.20)
    )


def test_epoch_geometry_trace_and_first_degraded_epochs(tmp_path):
    service, model_dir, data_file = _make_service(tmp_path, [
        _FakeTrainResult(
            baseline_loss=2.0,
            post_loss=1.4,
            baseline_perplexity=7.0,
            post_perplexity=4.0,
            epoch_metrics=[
                {
                    "epoch": 1,
                    "adapter_saturation_median_ratio": 0.12,
                    "dim_final_used_fraction": 0.41,
                    "dim_final_null_fraction": 0.59,
                    "dim_null_recruitment_from_baseline": 0.01,
                    "online_eval_pre_n_correct": 1,
                    "online_eval_pre_n_total": 2,
                    "online_eval_pre_degraded": False,
                    "online_eval_post_n_correct": 1,
                    "online_eval_post_n_total": 2,
                    "online_eval_post_degraded": False,
                },
                {
                    "epoch": 2,
                    "adapter_saturation_median_ratio": 0.18,
                    "dim_final_used_fraction": 0.44,
                    "dim_final_null_fraction": 0.56,
                    "dim_null_recruitment_from_baseline": 0.03,
                    "online_eval_pre_n_correct": 0,
                    "online_eval_pre_n_total": 2,
                    "online_eval_pre_degraded": True,
                    "online_eval_post_n_correct": 1,
                    "online_eval_post_n_total": 2,
                    "online_eval_post_degraded": False,
                },
                {
                    "epoch": 3,
                    "adapter_saturation_median_ratio": 0.22,
                    "dim_final_used_fraction": 0.45,
                    "dim_final_null_fraction": 0.55,
                    "dim_null_recruitment_from_baseline": 0.04,
                    "online_eval_pre_n_correct": 0,
                    "online_eval_pre_n_total": 2,
                    "online_eval_pre_degraded": True,
                    "online_eval_post_n_correct": 0,
                    "online_eval_post_n_total": 2,
                    "online_eval_post_degraded": True,
                },
            ],
        ),
    ])

    result = service.validate(
        model_path=model_dir,
        dataset_path=data_file,
        eval_dataset_path=None,
        trials=1,
        base_seed=21,
    )
    trial = result.trial_results[0]
    assert trial.online_eval_first_pre_degraded_epoch == 2
    assert trial.online_eval_first_post_degraded_epoch == 3
    assert trial.epoch_geometry_trace is not None
    assert len(trial.epoch_geometry_trace) == 3
    assert trial.epoch_geometry_trace[0]["epoch"] == 1
    assert trial.epoch_geometry_trace[1]["online_eval_pre_degraded"] is True
