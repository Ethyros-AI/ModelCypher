# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import modelcypher.core.use_cases.dataset_training_service as dataset_training_service_module
from modelcypher.cli.app import app
from modelcypher.core.domain.training.exceptions import TrainingDerivationError
from modelcypher.core.domain.training.regime_selection import PerTypeRegime
from modelcypher.core.use_cases.dataset_training_service import DatasetTrainingService

runner = CliRunner()


@dataclass
class _FloatInfo:
    eps: float


class _DummyBackend:
    def random_seed(self, _seed: int) -> None:
        return None

    def load_model(self, _model_path: str):
        return object(), object()

    def encode_tokens(self, _tokenizer, text: str) -> list[str]:
        return text.split()

    def decode_tokens(self, _tokenizer, token_ids: list[str]) -> str:
        return " ".join(token_ids)

    def generate(self, _model, _tokenizer, prompt: str, max_tokens: int = 512, **_kwargs) -> str:
        del max_tokens
        return f"{prompt} completion"

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
    full_rank: int = 2
    shannon_effective_rank: float = 1.0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class _AutoRetentionBackend(_DummyBackend):
    def __init__(self, fail_on_greedy: bool = False) -> None:
        self.fail_on_greedy = fail_on_greedy
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, _model, _tokenizer, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "kwargs": dict(kwargs),
            },
        )
        if self.fail_on_greedy and kwargs.get("temp") == 0.0:
            raise TypeError("temp/top_p not supported")
        return f"{prompt} ::completion"


class _FlowModel:
    def trainable_parameters(self) -> dict:
        return {}


class _FlowBackend(_DummyBackend):
    def load_model(self, _model_path: str):
        return _FlowModel(), object()

    def tree_flatten(self, _value) -> list[tuple[str, object]]:
        return []

    def get_num_layers(self, _model) -> int:
        return 0


class _FlowAdapter:
    def prepare_dataset(self, samples: list[dict], _tokenizer) -> list[dict]:
        return list(samples)

    def evaluate_loss(self, **_kwargs):
        return 1.0, 2.0

    def extract_weight_matrices(self, _model) -> dict[str, object]:
        return {"model.layers.0.self_attn.q_proj.weight": object()}

    def inject_nb_lora(self, *_args, **_kwargs) -> int:
        return 1

    def freeze_and_apply_lora(self, _model) -> None:
        return None

    def derive_critical_batch_size(self, _model, _train_dataset, _seq_length) -> int:
        return 1

    def train_loop(self, **_kwargs):
        return [(1, 1.0, 1.0)], "max_iters", []

    def verify_bounds(self, _model):
        return True, 0.5, {}


class _CliResult:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self._payload = payload or {"ok": True}

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


class _CliCaptureDatasetService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def train_from_dataset(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _CliResult()


def _patch_lightweight_training(monkeypatch: pytest.MonkeyPatch, service: DatasetTrainingService) -> None:
    monkeypatch.setattr(
        service,
        "_collect_probe_activations",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        dataset_training_service_module,
        "analyze_weight_geometries",
        lambda *_args, **_kwargs: {
            "model.layers.0.self_attn.q_proj.weight": _Geom(
                sigma_k=1.0,
                sigma_max=1.0,
                tail_dims=1,
                spectral_gap=0.1,
            ),
        },
    )
    monkeypatch.setattr(
        dataset_training_service_module,
        "derive_optimizer_geometry_config",
        lambda *_args, **_kwargs: SimpleNamespace(n_layers=1, base_lr=1e-3),
    )
    monkeypatch.setattr(
        dataset_training_service_module,
        "select_target_modules",
        lambda geometries: list(geometries.keys()),
    )
    monkeypatch.setattr(
        dataset_training_service_module,
        "compute_coupled_ranks",
        lambda _geometries, target_modules: {module: 1 for module in target_modules},
    )
    monkeypatch.setattr(
        dataset_training_service_module,
        "apply_data_rank_ceiling",
        lambda ranks, **_kwargs: dict(ranks),
    )
    monkeypatch.setattr(
        dataset_training_service_module,
        "estimate_nb_lora_parameter_count",
        lambda *_args, **_kwargs: 1,
    )


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
    n_eval, _split_info = service._derive_validation_split_from_losses(
        sample_losses=losses,
        n_total=len(losses),
    )

    assert 0 < n_eval < len(losses)

    val_losses = losses[:n_eval]
    mean_loss = sum(val_losses) / len(val_losses)
    target_se = math.sqrt(math.ldexp(1.0, -23)) * max(1.0, abs(mean_loss))
    variance = (
        sum((value - mean_loss) ** 2 for value in val_losses) / (len(val_losses) - 1)
        if len(val_losses) > 1
        else 0.0
    )
    standard_error = math.sqrt(variance / len(val_losses))
    assert standard_error <= target_se


def test_pilot_variance_split_uses_ieee754_sqrt_eps():
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())
    losses = [1.0, 1.0, 1.0, 1.0]

    _n_eval, split_info = service._derive_validation_split_from_losses(
        sample_losses=losses,
        n_total=len(losses),
    )

    expected_sqrt_eps = math.sqrt(math.ldexp(1.0, -23))
    assert split_info["sqrt_eps"] == pytest.approx(expected_sqrt_eps, rel=1e-12)


def test_train_from_dataset_auto_regime_default_is_enabled():
    signature = inspect.signature(DatasetTrainingService.train_from_dataset)
    assert signature.parameters["auto_regime"].default is True


def test_auto_regime_outcome_filter_drops_ce_types():
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())

    problems = [
        SimpleNamespace(problem_type="syllogistic_chain"),
        SimpleNamespace(problem_type="contrapositive"),
        SimpleNamespace(problem_type="multi_step_arithmetic"),
        SimpleNamespace(problem_type="unknown_type"),
    ]
    per_type_regime = {
        "syllogistic_chain": PerTypeRegime(
            problem_type="syllogistic_chain",
            n_correct=0,
            n_total=5,
            observed_accuracy=0.0,
            ci_lower=0.0,
            ci_upper=0.1,
            chance_rate=0.5,
            regime="ce",
            rationale="k=0",
        ),
        "contrapositive": PerTypeRegime(
            problem_type="contrapositive",
            n_correct=2,
            n_total=5,
            observed_accuracy=0.4,
            ci_lower=0.2,
            ci_upper=0.8,
            chance_rate=0.5,
            regime="reinforce_entropy",
            rationale="ci lower below chance",
        ),
        "multi_step_arithmetic": PerTypeRegime(
            problem_type="multi_step_arithmetic",
            n_correct=3,
            n_total=5,
            observed_accuracy=0.6,
            ci_lower=0.3,
            ci_upper=0.9,
            chance_rate=0.0,
            regime="reinforce",
            rationale="ci lower above chance",
        ),
    }

    filtered, dropped = service._filter_outcome_problems_by_regime(
        problems,
        per_type_regime,
    )

    assert [p.problem_type for p in filtered] == [
        "contrapositive",
        "multi_step_arithmetic",
    ]
    assert dropped["syllogistic_chain"] == 1
    # Unknown problem type defaults to CE for conservative filtering
    assert dropped["unknown_type"] == 1


def test_train_run_no_auto_regime_flag_suppresses_regime_selection(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text":"hello"}\n', encoding="utf-8")

    capture = _CliCaptureDatasetService()
    monkeypatch.setattr(
        "modelcypher.cli.composition.get_dataset_training_service",
        lambda: capture,
    )

    help_result = runner.invoke(app, ["train", "run", "--help"])
    assert help_result.exit_code == 0
    assert "--auto-regime" in help_result.stdout
    assert "--no-auto-regime" in help_result.stdout

    result = runner.invoke(
        app,
        ["train", "run", "--model", str(model_dir), "--data", str(data_path), "--no-auto-regime"],
    )
    assert result.exit_code == 0
    assert capture.calls
    assert capture.calls[0]["auto_regime"] is False


def test_pilot_variance_split_requires_two_samples():
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        service._derive_validation_split_from_losses(
            sample_losses=[1.0],
            n_total=1,
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_validation_resolution"
    assert err.diagnostics["n_total"] == 1


def test_pilot_variance_split_fails_when_required_eval_consumes_dataset():
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())
    losses = [0.01, 100.0]

    with pytest.raises(TrainingDerivationError) as excinfo:
        service._derive_validation_split_from_losses(
            sample_losses=losses,
            n_total=2,
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_validation_resolution"
    diagnostics = err.diagnostics or {}
    assert diagnostics["n_total"] == 2
    assert diagnostics["n_val_required"] >= 2


def test_pilot_variance_split_fails_on_loss_count_mismatch():
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())

    with pytest.raises(TrainingDerivationError) as excinfo:
        service._derive_validation_split_from_losses(
            sample_losses=[1.0, 2.0],
            n_total=3,
        )

    err = excinfo.value
    assert err.failure_class == "unavailable_measurement"
    diagnostics = err.diagnostics or {}
    assert diagnostics["n_total"] == 3
    assert diagnostics["n_losses"] == 2


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


def test_collect_auto_retention_applies_prompt_rules_and_format():
    backend = _AutoRetentionBackend()
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=backend)

    long_prompt = " ".join(f"tok{i}" for i in range(150))
    samples = [
        {"text": f"{long_prompt}\nsecond line should be excluded"},
        {"text": ""},
        {"text": 42},
    ]

    retention = service._collect_auto_retention(
        model=object(),
        tokenizer=object(),
        train_samples=samples,
        seq_length=64,
        seed=7,
        n_retention=3,
    )

    assert len(retention) == 1
    prompt_used = backend.generate_calls[0]["prompt"]
    assert isinstance(prompt_used, str)
    assert "\n" not in prompt_used
    assert len(prompt_used.split()) == 128
    assert retention[0] == {"text": f"{prompt_used} ::completion"}
    assert backend.generate_calls[0]["kwargs"]["temp"] == 0.0
    assert backend.generate_calls[0]["kwargs"]["top_p"] == 1.0


def test_collect_auto_retention_uses_greedy_fallback_on_typeerror():
    backend = _AutoRetentionBackend(fail_on_greedy=True)
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=backend)

    retention = service._collect_auto_retention(
        model=object(),
        tokenizer=object(),
        train_samples=[{"text": "prompt text"}],
        seq_length=32,
        seed=1,
        n_retention=1,
    )

    assert len(retention) == 1
    assert len(backend.generate_calls) == 2
    first_call_kwargs = backend.generate_calls[0]["kwargs"]
    second_call_kwargs = backend.generate_calls[1]["kwargs"]
    assert "temp" in first_call_kwargs
    assert "top_p" in first_call_kwargs
    assert "temp" not in second_call_kwargs
    assert "top_p" not in second_call_kwargs


def test_collect_auto_retention_default_bound_is_200():
    backend = _AutoRetentionBackend()
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=backend)
    train_samples = [{"text": f"sample {i}"} for i in range(250)]

    retention = service._collect_auto_retention(
        model=object(),
        tokenizer=object(),
        train_samples=train_samples,
        seq_length=16,
        seed=123,
        n_retention=None,
    )

    assert len(retention) == 200
    assert len(backend.generate_calls) == 200


def test_train_from_dataset_auto_regime_adds_regime_metadata(monkeypatch, tmp_path: Path):
    from modelcypher.core.domain.training.online_eval import OnlineEvalResult
    from modelcypher.core.domain.training.regime_selection import (
        PerTypeRegime as RegimePerType,
        TrainingRegime,
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}, {"text": "train 2"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    baseline_result = OnlineEvalResult(
        epoch=0,
        accuracy=0.0,
        n_correct=0,
        n_total=5,
        correct_ids=frozenset(),
        baseline_n_correct=0,
        baseline_accuracy=0.0,
        n_lost=0,
        n_gained=0,
        degraded=False,
        per_type_accuracy={"syllogistic_chain": 0.0},
        per_type_correct={"syllogistic_chain": 0},
        per_type_total={"syllogistic_chain": 5},
    )
    regime_result = TrainingRegime(
        global_regime="ce",
        per_type={
            "syllogistic_chain": RegimePerType(
                problem_type="syllogistic_chain",
                n_correct=0,
                n_total=5,
                observed_accuracy=0.0,
                ci_lower=0.0,
                ci_upper=0.2,
                chance_rate=0.5,
                regime="ce",
                rationale="k=0",
            ),
        },
        use_ce=True,
        use_outcome_training=False,
        use_entropy_regularization=False,
        confidence_level=0.8,
        n_total=5,
        derivation_log=("test",),
    )

    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.create_eval_problem_set",
        lambda **_kwargs: [object()],
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.evaluate_correctness",
        lambda **_kwargs: baseline_result,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.regime_selection.select_training_regime",
        lambda _baseline_result: regime_result,
    )

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        max_iters=1,
    )

    assert result.regime_global == "ce"
    assert result.regime_per_type is not None
    assert result.regime_per_type["syllogistic_chain"]["regime"] == "ce"
    payload = result.to_dict()
    assert payload["regime_global"] == "ce"
    assert payload["regime_per_type"]["syllogistic_chain"]["regime"] == "ce"


def test_auto_regime_preserves_explicit_entropy_regularization(monkeypatch, tmp_path: Path):
    from modelcypher.core.domain.training.online_eval import OnlineEvalResult
    from modelcypher.core.domain.training.regime_selection import TrainingRegime

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    baseline_result = OnlineEvalResult(
        epoch=0,
        accuracy=0.0,
        n_correct=0,
        n_total=5,
        correct_ids=frozenset(),
        baseline_n_correct=0,
        baseline_accuracy=0.0,
        n_lost=0,
        n_gained=0,
        degraded=False,
        per_type_accuracy={"syllogistic_chain": 0.0},
        per_type_correct={"syllogistic_chain": 0},
        per_type_total={"syllogistic_chain": 5},
    )
    regime_result = TrainingRegime(
        global_regime="ce",
        per_type={},
        use_ce=True,
        use_outcome_training=False,
        use_entropy_regularization=False,
        confidence_level=0.8,
        n_total=5,
        derivation_log=("test",),
    )

    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.create_eval_problem_set",
        lambda **_kwargs: [object()],
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.evaluate_correctness",
        lambda **_kwargs: baseline_result,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.regime_selection.select_training_regime",
        lambda _baseline_result: regime_result,
    )

    captured_train_loop_kwargs: dict[str, object] = {}

    def _capture_train_loop(**kwargs):
        captured_train_loop_kwargs.update(kwargs)
        return [(1, 1.0, 1.0)], "max_iters", []

    monkeypatch.setattr(service._adapter, "train_loop", _capture_train_loop)

    service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        max_iters=1,
        entropy_regularization=True,
    )

    assert captured_train_loop_kwargs["entropy_regularization"] is True


def test_auto_regime_selection_failure_raises_training_derivation_error(
    monkeypatch,
    tmp_path: Path,
):
    from modelcypher.core.domain.training.online_eval import OnlineEvalResult

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    baseline_result = OnlineEvalResult(
        epoch=0,
        accuracy=0.0,
        n_correct=0,
        n_total=5,
        correct_ids=frozenset(),
        baseline_n_correct=0,
        baseline_accuracy=0.0,
        n_lost=0,
        n_gained=0,
        degraded=False,
        per_type_accuracy={},
        per_type_correct={},
        per_type_total={},
    )

    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.create_eval_problem_set",
        lambda **_kwargs: [object()],
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.evaluate_correctness",
        lambda **_kwargs: baseline_result,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.regime_selection.select_training_regime",
        lambda _baseline_result: (_ for _ in ()).throw(ValueError("no per-type totals")),
    )

    with pytest.raises(TrainingDerivationError) as excinfo:
        service.train_from_dataset(
            model_path=model_dir,
            dataset_path=train_path,
            eval_dataset_path=eval_path,
            max_iters=1,
        )

    err = excinfo.value
    assert err.failure_class == "insufficient_baseline_measurement"
    assert "reason" in (err.diagnostics or {})


def test_train_from_dataset_auto_retention_mix_fraction(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(
        train_path,
        [{"text": "train 1"}, {"text": "train 2"}, {"text": "train 3"}],
    )
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)

    auto_retention = [{"text": "r0"}, {"text": "r1"}]
    monkeypatch.setattr(
        service,
        "_collect_auto_retention",
        lambda *_args, **_kwargs: list(auto_retention),
    )

    merge_call: dict[str, object] = {}

    def _fake_merge(primary, retention, fraction):
        merge_call["primary_len"] = len(primary)
        merge_call["retention_len"] = len(retention)
        merge_call["fraction"] = fraction
        return list(primary) + list(retention)

    monkeypatch.setattr(
        dataset_training_service_module,
        "merge_datasets_with_fraction",
        _fake_merge,
    )

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        max_iters=1,
        auto_regime=False,
    )

    assert merge_call["primary_len"] == 3
    assert merge_call["retention_len"] == 2
    assert merge_call["fraction"] == pytest.approx(2 / 5)
    assert result.auto_retention_samples_collected == 2
    assert result.to_dict()["auto_retention_samples_collected"] == 2


def test_train_from_dataset_manual_retention_skips_auto(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    retention_path = tmp_path / "retention.jsonl"
    _write_jsonl(
        train_path,
        [{"text": "train 1"}, {"text": "train 2"}, {"text": "train 3"}],
    )
    _write_jsonl(eval_path, [{"text": "eval 1"}])
    _write_jsonl(retention_path, [{"text": "manual retention"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)

    monkeypatch.setattr(
        service,
        "_collect_auto_retention",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("auto retention should not run when manual retention is provided"),
        ),
    )

    merge_call: dict[str, object] = {}

    def _fake_merge(primary, retention, fraction):
        merge_call["primary_len"] = len(primary)
        merge_call["retention_len"] = len(retention)
        merge_call["fraction"] = fraction
        return list(primary) + list(retention)

    monkeypatch.setattr(
        dataset_training_service_module,
        "merge_datasets_with_fraction",
        _fake_merge,
    )

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        retention_dataset_path=retention_path,
        retention_fraction=0.2,
        max_iters=1,
        auto_regime=False,
    )

    assert merge_call["primary_len"] == 3
    assert merge_call["retention_len"] == 1
    assert merge_call["fraction"] == pytest.approx(0.2)
    assert result.auto_retention_samples_collected == 0
    assert result.to_dict()["auto_retention_samples_collected"] == 0
