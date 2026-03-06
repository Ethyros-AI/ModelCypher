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
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.exceptions import TrainingDerivationError
from modelcypher.core.domain.training.geometric_lora import LayerGeometry
from modelcypher.core.domain.star.problem_generator import StarProblemGenerator
from modelcypher.core.use_cases.dataset_training_service import (
    DatasetTrainResult,
    DatasetTrainingService,
)

runner = CliRunner()


@dataclass
class _FloatInfo:
    eps: float


class _DummyBackend:
    def random_seed(self, _seed: int) -> None:
        return None

    def load_model(self, _model_path: str):
        return object(), object()

    def prepare_model_for_training(self, _model, _model_path: str) -> int:
        return 0

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

    def save_adapter(self, model, output_path, metadata=None):
        from pathlib import Path

        p = Path(output_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def verify_bounds(self, _model):
        return True, 0.5, {}


class _FlowAdapterFailingSpectral(_FlowAdapter):
    def verify_bounds(self, _model):
        return False, 1.5, {}


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
        lambda *_args, **_kwargs: {0: [object()]},
    )
    monkeypatch.setattr(
        service,
        "_collect_inference_probe_activations",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        service,
        "_verify_capability_preservation",
        lambda *_args, **_kwargs: {
            "min_cka": 0.99,
            "mean_cka": 0.99,
            "per_layer_cka": {"0": 0.99},
            "per_layer_cka_bound": {"0": 0.99},
        },
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
        "compute_per_layer_signal_ranks",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        dataset_training_service_module,
        "apply_signal_rank_ceiling",
        lambda ranks, *_args, **_kwargs: dict(ranks),
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



def test_train_run_unifies_instrumentation_flags(monkeypatch, tmp_path: Path):
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
    assert "--seq-length" in help_result.stdout
    assert "--lr" in help_result.stdout
    assert "--seed" in help_result.stdout
    assert "--topo-monitor" in help_result.stdout
    assert "--dim-monitor" in help_result.stdout
    assert "--no-save" in help_result.stdout
    assert "--auto-regime" not in help_result.stdout
    assert "--no-auto-regime" not in help_result.stdout

    result = runner.invoke(
        app,
        ["train", "run", "--model", str(model_dir), "--data", str(data_path)],
    )
    assert result.exit_code == 0
    assert capture.calls
    call = capture.calls[0]
    assert set(call) == {
        "model_path",
        "dataset_path",
        "output_path",
        "eval_dataset_path",
        "seq_length",
        "lr_override",
        "seed",
        "topo_monitor",
        "dim_monitor",
        "no_save",
        "benchmark_suite",
        "target_experts",
        "entropy_regularization",
    }



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


def test_pilot_variance_split_two_samples_keeps_one_training_sample():
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())
    losses = [0.01, 100.0]

    n_eval, diagnostics = service._derive_validation_split_from_losses(
        sample_losses=losses,
        n_total=2,
    )
    assert n_eval == 1
    assert diagnostics["n_eval"] == 1
    assert diagnostics["n_train"] == 1


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


def test_train_from_dataset_rejects_image_samples_for_non_vl_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    rows = [{"text": "<|image_pad|> describe image", "image_path": "img.jpg"}]
    _write_jsonl(train_path, rows)
    _write_jsonl(eval_path, rows)

    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=_DummyBackend())
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError, match="model has no vision_config"):
        service.train_from_dataset(
            model_path=model_dir,
            dataset_path=train_path,
            eval_dataset_path=eval_path,
            no_save=True,
        )


def test_train_from_dataset_uses_vl_loader_and_preprocessor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"vision_config": {"spatial_merge_size": 2}}),
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "<|image_pad|> train sample", "image_path": "img1.jpg"}])
    _write_jsonl(eval_path, [{"text": "<|image_pad|> eval sample", "image_path": "img2.jpg"}])

    class _NoFallbackLoadBackend(_FlowBackend):
        def __init__(self) -> None:
            super().__init__()
            self.load_model_called = False

        def load_model(self, _model_path: str):
            self.load_model_called = True
            raise AssertionError("backend.load_model should not be used for VL models")

    class _FakeTokens:
        def __init__(self, length: int) -> None:
            self.shape = (length,)

    class _FakeVLPreprocessor:
        def __init__(self, calls: dict[str, int]) -> None:
            self._calls = calls

        def prepare_vl_dataset(self, samples: list[dict], _tokenizer) -> list[dict]:
            self._calls["preprocess"] += 1
            return [
                {
                    "tokens": _FakeTokens(65),
                    "pixel_values": object(),
                    "position_ids": object(),
                    "n_text_tokens": 65,
                    "image_token_id": 151655,
                    "video_token_id": 151656,
                }
                for _ in samples
            ]

    backend = _NoFallbackLoadBackend()
    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=backend)
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    calls = {"vl_load": 0, "preprocess": 0}

    def _fake_vl_loader(_model_path: str):
        calls["vl_load"] += 1
        return _FlowModel(), object()

    monkeypatch.setattr(
        "modelcypher.backends._mlx_qwen35_vl_encoder.is_qwen35_vl",
        lambda _model_path: True,
    )
    monkeypatch.setattr(
        "modelcypher.backends._mlx_qwen35_vl_encoder.load_qwen35_vl_model",
        _fake_vl_loader,
    )
    monkeypatch.setattr(
        "modelcypher.backends._mlx_vl_preprocessor.VLPreprocessor.from_model_path",
        classmethod(lambda _cls, _model_path: _FakeVLPreprocessor(calls)),
    )

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        no_save=True,
        max_iters_cap=1,
    )

    assert isinstance(result, DatasetTrainResult)
    assert calls["vl_load"] == 1
    assert calls["preprocess"] == 2
    assert backend.load_model_called is False


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
    assert len(prompt_used.split()) == 64
    # Retention samples are bounded to seq_length by token count.
    # For full-window prompts, completion overage is truncated away.
    assert retention[0] == {"text": prompt_used}
    assert backend.generate_calls[0]["max_tokens"] == 1
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


def test_collect_auto_retention_default_uses_all_training_prompts():
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

    assert len(retention) == 250
    assert len(backend.generate_calls) == 250



def test_train_from_dataset_fails_pipeline_gate(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(
        adapter=_FlowAdapterFailingSpectral(),
        backend=_FlowBackend(),
    )
    _patch_lightweight_training(monkeypatch, service)

    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    with pytest.raises(TrainingDerivationError) as excinfo:
        service.train_from_dataset(
            model_path=model_dir,
            dataset_path=train_path,
            output_path=tmp_path / "adapter",
            eval_dataset_path=eval_path,
        )

    err = excinfo.value
    assert err.failure_class == "pipeline_gate_failed"
    diagnostics = err.diagnostics or {}
    assert diagnostics["operator"] == "pipeline_gate_v1"
    assert "spectral_bounds_violation" in diagnostics["failure_modes"]


def test_train_from_dataset_exposes_pipeline_gate_metadata(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)

    monkeypatch.setattr(
        service,
        "_collect_probe_activations",
        lambda *_args, **_kwargs: {0: [object()]},
    )
    monkeypatch.setattr(
        service,
        "_verify_capability_preservation",
        lambda *_args, **_kwargs: {
            "min_cka": 0.99,
            "mean_cka": 0.99,
            "per_layer_cka": {"0": 0.99},
            "per_layer_cka_bound": {"0": 0.99},
        },
    )
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        output_path=tmp_path / "adapter",
        eval_dataset_path=eval_path,
    )

    assert result.pipeline_gate_operator == "pipeline_gate_v1"
    assert result.pipeline_gate_passed is True
    assert result.pipeline_gate_failure_modes == []
    payload = result.to_dict()
    assert payload["pipeline_gate_operator"] == "pipeline_gate_v1"
    assert payload["pipeline_gate_passed"] is True
    assert payload["pipeline_gate_failure_modes"] == []
    assert "spectral_bounds" in payload["pipeline_gate_checks"]


def test_train_from_dataset_seed_override_is_respected(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    backend = _FlowBackend()
    service = DatasetTrainingService(
        adapter=_FlowAdapter(),
        backend=backend,
    )
    _patch_lightweight_training(monkeypatch, service)

    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])
    seen_seeds: list[int] = []
    monkeypatch.setattr(
        backend,
        "random_seed",
        lambda seed: seen_seeds.append(seed),
    )

    service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        no_save=True,
        seed=1234,
    )

    assert seen_seeds == [1234]


def test_train_from_dataset_auto_output_path_uses_derived_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(
        adapter=_FlowAdapter(),
        backend=_FlowBackend(),
    )
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_derive_training_seed", lambda **_kwargs: 4242)

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
    )

    assert result.adapter_path is not None
    assert Path(result.adapter_path).name == "model-nblora-4242"



def test_quantization_frontier_precheck_runs_before_init_adapter_merge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    ref_model_dir = tmp_path / "model_fp"
    ref_model_dir.mkdir()
    init_adapter_dir = tmp_path / "adapter"
    init_adapter_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    candidate_model = _FlowModel()
    candidate_model.merged = False
    reference_model = _FlowModel()
    monkeypatch.setattr(
        service._backend,
        "load_model",
        lambda model_path: (
            candidate_model if model_path == str(model_dir) else reference_model,
            object(),
        ),
    )
    order: list[str] = []

    def _frontier(**kwargs):
        assert kwargs["model"] is candidate_model
        assert candidate_model.merged is False
        order.append("frontier")
        return {
            "valid": True,
            "failure_modes": [],
            "n_probes": 1,
            "n_layers": 1,
            "raw_weyl": {"n_crossing": 1, "all_non_crossing": False},
        }

    def _merge(model, _adapter_path):
        assert order == ["frontier"]
        model.merged = True
        order.append("merge")
        return 1

    monkeypatch.setattr(service, "_run_quantization_frontier_precheck", _frontier)
    monkeypatch.setattr(
        service._adapter,
        "apply_standard_lora_adapter",
        _merge,
        raising=False,
    )

    service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        quantization_reference_model_path=ref_model_dir,
        init_adapter_path=init_adapter_dir,
        no_save=True,
    )
    assert order == ["frontier", "merge"]


def test_quantization_frontier_validity_allows_raw_weyl_crossing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    ref_model_dir = tmp_path / "model_fp"
    ref_model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "_run_quantization_frontier_precheck",
        lambda **_kwargs: {
            "operator": "quantization_frontier_precheck_v1",
            "valid": True,
            "failure_modes": [],
            "n_probes": 3,
            "n_layers": 1,
            "min_cka": 0.92,
            "mean_cka": 0.92,
            "per_layer_cka": {0: 0.92},
            "per_layer_gram_epsilon": {0: 0.04},
            "per_layer_cka_bound": {0: 0.90},
            "per_layer_hidden_probe_d_eff": {0: 2.0},
            "per_layer_hidden_probe_k_eff": {0: 2},
            "per_layer_hidden_probe_gap_eff": {0: 0.5},
            "per_layer_hidden_probe_rho_out": {0: 0.2},
            "subspace_source": "hidden_probe_output",
            "raw_weyl": {
                "n_layers": 1,
                "n_crossing": 1,
                "all_non_crossing": False,
                "max_error_over_gap_half": 1700.5,
            },
        },
    )

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        quantization_reference_model_path=ref_model_dir,
        no_save=True,
    )

    payload = result.to_dict()
    assert payload["quantization_frontier_precheck"]["valid"] is True
    assert payload["quantization_frontier_precheck"]["raw_weyl"]["all_non_crossing"] is False
    assert payload["quantization_frontier_precheck"]["raw_weyl"]["n_crossing"] == 1


def test_quantization_frontier_invalid_fails_closed_without_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    ref_model_dir = tmp_path / "model_fp"
    ref_model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "_run_quantization_frontier_precheck",
        lambda **_kwargs: {
            "valid": False,
            "failure_modes": ["degenerate_centered_gram"],
            "n_probes": 3,
            "raw_weyl": {"n_crossing": 1},
        },
    )

    with pytest.raises(TrainingDerivationError) as excinfo:
        service.train_from_dataset(
            model_path=model_dir,
            dataset_path=train_path,
            eval_dataset_path=eval_path,
            quantization_reference_model_path=ref_model_dir,
            no_save=True,
        )
    assert excinfo.value.failure_class == "quantization_frontier_unavailable"


def test_quantization_frontier_invalid_allows_explicit_research_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    ref_model_dir = tmp_path / "model_fp"
    ref_model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "_run_quantization_frontier_precheck",
        lambda **_kwargs: {
            "valid": False,
            "failure_modes": ["degenerate_centered_gram"],
            "n_probes": 3,
            "raw_weyl": {"n_crossing": 1},
        },
    )

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        quantization_reference_model_path=ref_model_dir,
        research_allow_quantization_frontier_invalid=True,
        no_save=True,
    )

    payload = result.to_dict()
    assert payload["quantization_frontier_precheck"]["valid"] is False
    assert payload["quantization_frontier_precheck"]["failure_modes"] == [
        "degenerate_centered_gram"
    ]


def test_quantization_frontier_precheck_not_run_without_reference_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "_run_quantization_frontier_precheck",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("frontier should not run")),
    )

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        no_save=True,
    )

    payload = result.to_dict()
    assert "quantization_frontier_precheck" not in payload


def test_research_online_eval_problem_set_path_overrides_generated_eval_problems(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from modelcypher.core.domain.training.online_eval import OnlineEvalResult

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    problem = StarProblemGenerator(seed=11).generate(1)[0]
    problem_set_path = tmp_path / "online_eval_problems.json"
    problem_set_path.write_text(
        json.dumps([problem.to_problem_record()], indent=2),
        encoding="utf-8",
    )

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    baseline_result = OnlineEvalResult(
        epoch=0,
        accuracy=1.0,
        n_correct=1,
        n_total=1,
        correct_ids=frozenset({problem.problem_id}),
        baseline_n_correct=1,
        baseline_accuracy=1.0,
        n_lost=0,
        n_gained=0,
        degraded=False,
        per_type_accuracy={problem.problem_type: 1.0},
        per_type_correct={problem.problem_type: 1},
        per_type_total={problem.problem_type: 1},
    )

    def _unexpected_create_eval_problem_set(**_kwargs):
        raise AssertionError("create_eval_problem_set should not be called")

    captured_eval: dict[str, object] = {}

    def _capture_eval_correctness(**kwargs):
        captured_eval["problems"] = kwargs["problems"]
        return baseline_result

    captured_train_loop_kwargs: dict[str, object] = {}

    def _capture_train_loop(**kwargs):
        captured_train_loop_kwargs.update(kwargs)
        return [(1, 1.0, 1.0)], "max_iters", []

    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.create_eval_problem_set",
        _unexpected_create_eval_problem_set,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.training.online_eval.evaluate_correctness",
        _capture_eval_correctness,
    )
    monkeypatch.setattr(service._adapter, "train_loop", _capture_train_loop)

    service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        online_eval=True,
        research_online_eval_problem_set_path=problem_set_path,
        no_save=True,
    )

    problems = captured_eval["problems"]
    assert isinstance(problems, list)
    assert len(problems) == 1
    assert problems[0].problem_id == problem.problem_id
    assert len(captured_train_loop_kwargs["online_eval_problems"]) == 1



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
        no_save=True,
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
        no_save=True,
    )

    assert merge_call["primary_len"] == 3
    assert merge_call["retention_len"] == 1
    # retention_fraction now always derived from data ratio: 1 / (3 + 1) = 0.25
    assert merge_call["fraction"] == pytest.approx(0.25)
    assert result.auto_retention_samples_collected == 0
    assert result.to_dict()["auto_retention_samples_collected"] == 0


def test_train_from_dataset_seq_length_includes_manual_retention(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    retention_path = tmp_path / "retention.jsonl"
    _write_jsonl(train_path, [{"text": "short sample"}])
    _write_jsonl(eval_path, [{"text": "eval"}])
    long_retention = " ".join(f"tok{i}" for i in range(70))
    _write_jsonl(retention_path, [{"text": long_retention}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(
        service,
        "_collect_auto_retention",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("auto retention should not run when manual retention is provided"),
        ),
    )

    monkeypatch.setattr(
        dataset_training_service_module,
        "merge_datasets_with_fraction",
        lambda primary, retention, _fraction: list(primary) + list(retention),
    )

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        retention_dataset_path=retention_path,
        no_save=True,
    )

    simd_width = dataset_training_service_module._MLX_SIMD_WIDTH
    max_tokens = len(long_retention.split())
    expected_seq_length = ((max_tokens + simd_width - 1) // simd_width) * simd_width
    assert result.seq_length_used == expected_seq_length


def test_train_from_dataset_seq_length_includes_eval_data(monkeypatch, tmp_path: Path):
    """seq_length must cover eval samples to prevent truncation at evaluation time."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "short"}])
    # Eval sample much longer than any train sample.
    long_eval = " ".join(f"tok{i}" for i in range(100))
    _write_jsonl(eval_path, [{"text": long_eval}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        no_save=True,
    )

    simd_width = dataset_training_service_module._MLX_SIMD_WIDTH
    max_tokens = len(long_eval.split())
    expected_seq_length = ((max_tokens + simd_width - 1) // simd_width) * simd_width
    assert result.seq_length_used == expected_seq_length



def test_train_from_dataset_populates_moe_targets(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({
            "model_type": "qwen3",
            "num_hidden_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
        }),
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, [{"text": "train 1"}])
    _write_jsonl(eval_path, [{"text": "eval 1"}])

    service = DatasetTrainingService(adapter=_FlowAdapter(), backend=_FlowBackend())
    _patch_lightweight_training(monkeypatch, service)
    monkeypatch.setattr(service, "_collect_auto_retention", lambda *_args, **_kwargs: [])

    def _geom(key: str) -> LayerGeometry:
        return LayerGeometry(
            layer_key=key,
            shape=(64, 16),
            sigma_max=1.0,
            sigma_k=0.5,
            effective_rank=16,
            full_rank=16,
            decay_ratio=2.0,
            tail_dims=4,
            shannon_effective_rank=12.0,
            spectral_gap=0.1,
        )

    moe_keys = [
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.2.up_proj.weight",
        "model.layers.0.mlp.experts.2.down_proj.weight",
    ]
    monkeypatch.setattr(
        dataset_training_service_module,
        "analyze_weight_geometries",
        lambda *_args, **_kwargs: {key: _geom(key) for key in moe_keys},
    )
    monkeypatch.setattr(
        dataset_training_service_module,
        "select_target_modules",
        lambda _geometries: list(moe_keys),
    )

    result = service.train_from_dataset(
        model_path=model_dir,
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        no_save=True,
        target_experts="L0.E2",
    )

    assert result.moe_targets is not None
    assert result.moe_targets.n_trainable_experts == 1
    assert result.moe_targets.topology.num_experts == 4
    assert result.moe_targets.target_module_keys == sorted(moe_keys)
    payload = result.to_dict()
    assert "moe_targets" in payload
    assert payload["moe_targets"]["n_targets"] == 1


def test_dataset_train_result_to_dict_includes_null_space_diagnostics():
    result = DatasetTrainResult(
        train_iters=10,
        initial_loss=2.0,
        final_loss=1.0,
        stop_reason="certificate",
        baseline_loss=2.0,
        baseline_perplexity=8.0,
        post_loss=1.0,
        post_perplexity=4.0,
        n_lora_layers=1,
        n_trainable_params=100,
        adapter_path=None,
        spectral_bounds_ok=True,
        max_spectral_ratio=0.5,
        training_time_seconds=1.0,
        per_layer_null_observability={0: {"condition_number": 10.0, "coverage_ratio": 1.5}},
        per_layer_null_accessibility={0: {"behavioral_preserved_fraction": 0.25}},
        per_module_null_accessibility={
            "model.layers.0.self_attn.q_proj.weight": {
                "behavioral_preserved_fraction": 0.25,
            }
        },
    )
    payload = result.to_dict()
    assert payload["per_layer_null_observability"][0]["condition_number"] == pytest.approx(10.0)
    assert payload["per_layer_null_accessibility"][0]["behavioral_preserved_fraction"] == pytest.approx(0.25)
    assert (
        payload["per_module_null_accessibility"][
            "model.layers.0.self_attn.q_proj.weight"
        ]["behavioral_preserved_fraction"]
        == pytest.approx(0.25)
    )


def test_verify_capability_preservation_without_delta_extractor_keeps_null_access_none(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = get_default_backend()
    service = DatasetTrainingService(adapter=_DummyAdapter(), backend=backend)

    def _fake_collect_hidden_activations(_self, _model, _tokenizer, texts):
        text = str(texts[0])
        val = float(len(text))
        layer = backend.array([[[val, 0.0, 0.0]]], dtype="float32")
        return {0: layer}

    monkeypatch.setattr(
        type(backend),
        "collect_hidden_activations",
        _fake_collect_hidden_activations,
    )

    base_activations = {
        0: [
            backend.array([1.0, 0.0, 0.0], dtype="float32"),
            backend.array([2.0, 0.0, 0.0], dtype="float32"),
        ],
    }
    eval_samples = [{"text": "a"}, {"text": "ab"}]
    result = service._verify_capability_preservation(
        model=object(),
        tokenizer=object(),
        base_activations=base_activations,
        eval_samples=eval_samples,
    )

    assert result["per_layer_null_observability"] is not None
    assert result["per_layer_null_accessibility"] is None
    assert result["per_module_null_accessibility"] is None


# ---------------------------------------------------------------------------
# Signal-rank ceiling unit tests
# ---------------------------------------------------------------------------

class _FakeSignalRankResult:
    """Minimal stand-in for SignalRankResult (avoids importing geometry module)."""
    def __init__(self, signal_rank: int):
        self.signal_rank = signal_rank
        self.noise_rank = 0
        self.mp_upper_edge = 0.0
        self.signal_variance_fraction = 0.0


def test_signal_rank_ceiling_reduces_ranks():
    """Modules in a measured layer get capped to signal_rank."""
    from modelcypher.core.domain.training.geometric_lora import apply_signal_rank_ceiling

    ranks = {
        "model.layers.0.self_attn.q_proj.weight": 500,
        "model.layers.0.self_attn.k_proj.weight": 800,
    }
    signal = {0: _FakeSignalRankResult(signal_rank=25)}
    result = apply_signal_rank_ceiling(ranks, signal)
    assert result["model.layers.0.self_attn.q_proj.weight"] == 25
    assert result["model.layers.0.self_attn.k_proj.weight"] == 25


def test_signal_rank_ceiling_floors_at_one():
    """signal_rank=0 should produce floor of 1, not 0."""
    from modelcypher.core.domain.training.geometric_lora import apply_signal_rank_ceiling

    ranks = {"model.layers.0.self_attn.q_proj.weight": 500}
    signal = {0: _FakeSignalRankResult(signal_rank=0)}
    result = apply_signal_rank_ceiling(ranks, signal)
    assert result["model.layers.0.self_attn.q_proj.weight"] == 1


def test_signal_rank_ceiling_preserves_zero_rank():
    """Modules with rank=0 (not targetable) stay at 0."""
    from modelcypher.core.domain.training.geometric_lora import apply_signal_rank_ceiling

    ranks = {"model.layers.0.self_attn.q_proj.weight": 0}
    signal = {0: _FakeSignalRankResult(signal_rank=25)}
    result = apply_signal_rank_ceiling(ranks, signal)
    assert result["model.layers.0.self_attn.q_proj.weight"] == 0


def test_signal_rank_ceiling_preserves_unmeasured_layers():
    """Modules in layers without signal-rank measurement keep original rank."""
    from modelcypher.core.domain.training.geometric_lora import apply_signal_rank_ceiling

    ranks = {
        "model.layers.0.self_attn.q_proj.weight": 500,
        "model.layers.5.self_attn.q_proj.weight": 300,
    }
    # Only layer 0 measured
    signal = {0: _FakeSignalRankResult(signal_rank=25)}
    result = apply_signal_rank_ceiling(ranks, signal)
    assert result["model.layers.0.self_attn.q_proj.weight"] == 25
    assert result["model.layers.5.self_attn.q_proj.weight"] == 300
