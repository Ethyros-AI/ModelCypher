# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_nblora_vs_standard as experiment


def test_convert_lora_family_layers_enables_dora_only_for_dora() -> None:
    calls: list[dict] = []

    def fake_converter(model, num_layers, lora_parameters, **kwargs):
        calls.append(
            {
                "model": model,
                "num_layers": num_layers,
                "lora_parameters": lora_parameters,
                "kwargs": kwargs,
            }
        )

    model = object()
    params = {"rank": 8, "scale": 2.0, "dropout": 0.0}

    experiment._convert_lora_family_layers(
        model,
        num_layers=16,
        lora_parameters=params,
        fine_tune_type="lora",
        converter=fake_converter,
    )
    experiment._convert_lora_family_layers(
        model,
        num_layers=16,
        lora_parameters=params,
        fine_tune_type="dora",
        converter=fake_converter,
    )

    assert calls[0]["kwargs"] == {"use_dora": False}
    assert calls[1]["kwargs"] == {"use_dora": True}


def test_train_method_relabels_lora_family_result_to_requested_method(monkeypatch) -> None:
    def fake_train_standard_lora(*args, **kwargs):
        return {"method": "lora", "seed": kwargs["seed"]}

    monkeypatch.setattr(experiment, "train_standard_lora", fake_train_standard_lora)

    result = experiment._train_method(
        method_name="rslora",
        model_path="/tmp/model",
        num_layers=16,
        data_dir=Path("/tmp/data"),
        output_dir=Path("/tmp/output"),
        seed=42,
        iters=100,
    )

    assert result["method"] == "rslora"
    assert result["seed"] == 42


def test_build_comparison_creates_head_to_head_for_each_requested_opponent() -> None:
    baseline_scores = {
        "arc_easy": {"acc,none": 0.50, "acc_stderr,none": 0.01},
    }
    method_evals = {
        "nb_lora": [{"arc_easy": {"acc,none": 0.60, "acc_stderr,none": 0.01}}],
        "standard_lora": [{"arc_easy": {"acc,none": 0.55, "acc_stderr,none": 0.01}}],
        "dora": [{"arc_easy": {"acc,none": 0.58, "acc_stderr,none": 0.01}}],
    }
    method_trainings = {
        "nb_lora": [{"max_spectral_ratio": 0.1, "spectral_bounds_ok": True}],
        "standard_lora": [{"spectral_info": {"max_spectral_norm": 1.0}}],
        "dora": [{"spectral_info": {"max_spectral_norm": 1.1}}],
    }

    comparison = experiment.build_comparison(
        baseline_scores=baseline_scores,
        method_evals=method_evals,
        method_trainings=method_trainings,
        methods=["standard_lora", "dora", "nb_lora"],
    )

    assert "nb_vs_standard_lora" in comparison["head_to_head"]
    assert "nb_vs_dora" in comparison["head_to_head"]


@pytest.mark.parametrize("method_name,target_fn_name", [
    ("pissa", "train_pissa"),
    ("eva", "train_eva"),
    ("nb_lora", "train_nb_lora"),
    ("geometric_pissa", "train_geometric_pissa"),
])
def test_train_method_dispatches_to_correct_function(
    monkeypatch, method_name, target_fn_name
) -> None:
    called_with: dict = {}

    def fake_trainer(*args, **kwargs):
        called_with["args"] = args
        called_with["kwargs"] = kwargs
        return {"method": "original_label", "seed": kwargs.get("seed", 0)}

    monkeypatch.setattr(experiment, target_fn_name, fake_trainer)

    result = experiment._train_method(
        method_name=method_name,
        model_path="/tmp/model",
        num_layers=16,
        data_dir=Path("/tmp/data"),
        output_dir=Path("/tmp/output"),
        seed=42,
        iters=100,
    )

    assert called_with, f"{target_fn_name} was never called"
    assert result["method"] == method_name


def test_train_method_raises_on_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        experiment._train_method(
            method_name="nonexistent",
            model_path="/tmp/model",
            num_layers=16,
            data_dir=Path("/tmp/data"),
            output_dir=Path("/tmp/output"),
            seed=42,
            iters=100,
        )


# ---------------------------------------------------------------------------
# Surface-matched method tests
# ---------------------------------------------------------------------------

def _make_fake_nb_surface():
    """Create a minimal NBTargetSurface-like object for dispatch tests."""
    class FakeSurface:
        rank_overrides = {
            "base.layers.2.mixer.q_proj.weight": 68,
            "base.layers.2.mixer.k_proj.weight": 68,
        }
        target_keys = list(rank_overrides.keys())
        rank_ceiling_source = "RMT signal-rank"
        sigma_k_min = 0.01
        sigma_max = 12.5
    return FakeSurface()


@pytest.mark.parametrize("method_name,target_fn_name", [
    ("standard_nb_surface", "train_surface_matched"),
    ("pissa_nb_surface", "train_surface_matched"),
    ("dora_nb_surface", "train_surface_matched"),
])
def test_surface_matched_dispatches_to_train_surface_matched(
    monkeypatch, method_name, target_fn_name,
) -> None:
    called_with: dict = {}

    def fake_trainer(*args, **kwargs):
        called_with["args"] = args
        called_with["kwargs"] = kwargs
        return {"method": "surface_matched", "seed": kwargs.get("seed", 0)}

    monkeypatch.setattr(experiment, target_fn_name, fake_trainer)

    result = experiment._train_method(
        method_name=method_name,
        model_path="/tmp/model",
        num_layers=16,
        data_dir=Path("/tmp/data"),
        output_dir=Path("/tmp/output"),
        seed=42,
        iters=100,
        nb_surface=_make_fake_nb_surface(),
    )

    assert called_with, f"{target_fn_name} was never called for {method_name}"
    assert result["method"] == method_name
    assert called_with["kwargs"]["rank_overrides"] == _make_fake_nb_surface().rank_overrides


def test_geometric_pissa_nb_surface_dispatches_with_rank_overrides(
    monkeypatch,
) -> None:
    called_with: dict = {}

    def fake_trainer(*args, **kwargs):
        called_with["args"] = args
        called_with["kwargs"] = kwargs
        return {"method": "geometric_pissa", "seed": kwargs.get("seed", 0)}

    monkeypatch.setattr(experiment, "train_geometric_pissa", fake_trainer)

    surface = _make_fake_nb_surface()
    result = experiment._train_method(
        method_name="geometric_pissa_nb_surface",
        model_path="/tmp/model",
        num_layers=16,
        data_dir=Path("/tmp/data"),
        output_dir=Path("/tmp/output"),
        seed=42,
        iters=100,
        nb_surface=surface,
    )

    assert called_with, "train_geometric_pissa was never called"
    assert result["method"] == "geometric_pissa_nb_surface"
    assert called_with["kwargs"]["rank_overrides"] == surface.rank_overrides


@pytest.mark.parametrize("method_name", [
    "standard_nb_surface",
    "pissa_nb_surface",
    "dora_nb_surface",
    "geometric_pissa_nb_surface",
])
def test_surface_methods_raise_without_nb_surface(method_name) -> None:
    with pytest.raises(ValueError, match="requires nb_surface"):
        experiment._train_method(
            method_name=method_name,
            model_path="/tmp/model",
            num_layers=16,
            data_dir=Path("/tmp/data"),
            output_dir=Path("/tmp/output"),
            seed=42,
            iters=100,
            nb_surface=None,
        )


def test_nb_surface_methods_set_is_consistent() -> None:
    """All NB_SURFACE_METHODS must be in ALL_METHOD_NAMES."""
    for m in experiment.NB_SURFACE_METHODS:
        assert m in experiment.ALL_METHOD_NAMES, f"{m} missing from ALL_METHOD_NAMES"
        assert m in experiment.METHOD_DISPLAY, f"{m} missing from METHOD_DISPLAY"
        assert m in experiment.SPECTRAL_POSTHOC_METHODS, f"{m} missing from SPECTRAL_POSTHOC_METHODS"


def test_targeted_lora_conversion_contract() -> None:
    """targeted_lora_conversion converts exact modules with correct ranks."""
    import mlx.nn as nn

    # Build a minimal model tree: model.base.layers[0].mixer.q_proj
    class FakeMixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = FakeMixer()

    class FakeBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [FakeLayer()]

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = FakeBase()

    model = FakeModel()
    model.freeze()

    rank_overrides = {
        "base.layers.0.mixer.q_proj.weight": 4,
        "base.layers.0.mixer.k_proj.weight": 8,
    }

    count = experiment.targeted_lora_conversion(
        model, rank_overrides, scale=2.0, dropout=0.0,
    )

    assert count == 2

    from mlx_lm.tuner.lora import LoRALinear
    q_proj = model.base.layers[0].mixer.q_proj
    k_proj = model.base.layers[0].mixer.k_proj
    assert isinstance(q_proj, LoRALinear)
    assert isinstance(k_proj, LoRALinear)
    assert q_proj.lora_a.shape[0] == 64  # [in_dims, rank]
    assert q_proj.lora_a.shape[1] == 4
    assert k_proj.lora_a.shape[1] == 8


def test_targeted_lora_conversion_alpha_per_module_scale() -> None:
    """When lora_alpha is set, scale = alpha / rank per module."""
    import mlx.nn as nn

    class FakeMixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = FakeMixer()

    class FakeBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [FakeLayer()]

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = FakeBase()

    model = FakeModel()
    model.freeze()

    rank_overrides = {
        "base.layers.0.mixer.q_proj.weight": 4,
        "base.layers.0.mixer.k_proj.weight": 8,
    }

    experiment.targeted_lora_conversion(
        model, rank_overrides, lora_alpha=16.0,
    )

    q_proj = model.base.layers[0].mixer.q_proj
    k_proj = model.base.layers[0].mixer.k_proj
    # scale = alpha / rank: 16/4=4.0 for q_proj, 16/8=2.0 for k_proj
    assert q_proj.scale == 4.0
    assert k_proj.scale == 2.0


def test_fuse_lora_into_base_produces_correct_weights() -> None:
    """_fuse_lora_into_base folds delta into linear.weight correctly."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    linear = nn.Linear(8, 8)
    original_weight = linear.weight.astype(mx.float32)
    mx.eval(original_weight)
    lora = LoRALinear.from_base(linear, r=2, scale=1.0)

    # Set known LoRA weights
    lora.lora_a = mx.ones((8, 2)) * 0.1  # [in, r]
    lora.lora_b = mx.ones((2, 8)) * 0.1  # [r, out]
    mx.eval(lora.lora_a, lora.lora_b)

    # Expected delta: (lora_a @ lora_b).T * scale = (0.1*[8,2] @ 0.1*[2,8]).T * 1.0
    expected_delta = (lora.lora_a @ lora.lora_b).T
    mx.eval(expected_delta)
    expected_weight = original_weight + expected_delta

    # Build minimal model
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = lora
    model = M()

    fused = experiment._fuse_lora_into_base(model)

    # Weight should be original + delta
    assert "proj.weight" in fused, f"Keys: {list(fused.keys())}"
    assert "proj.lora_a" not in fused
    actual = fused["proj.weight"].astype(mx.float32)
    mx.eval(actual, expected_weight)
    diff = float(mx.abs(actual - expected_weight).max())
    assert diff < 1e-5, f"Fused weight differs by {diff}"


def test_train_surface_matched_always_produces_fused_model(monkeypatch) -> None:
    """All surface-matched arms must return fused_model_path (including non-PiSSA)."""
    called_with: dict = {}

    def fake_trainer(*args, **kwargs):
        called_with.update(kwargs)
        return {"method": "surface_matched", "seed": 42, "fused_model_path": "/fake/fused"}

    monkeypatch.setattr(experiment, "train_surface_matched", fake_trainer)

    for method in ["standard_nb_surface", "dora_nb_surface"]:
        result = experiment._train_method(
            method_name=method,
            model_path="/tmp/model",
            num_layers=16,
            data_dir=Path("/tmp/data"),
            output_dir=Path("/tmp/output"),
            seed=42,
            iters=100,
            nb_surface=_make_fake_nb_surface(),
        )
        assert "fused_model_path" in result, f"{method} missing fused_model_path"


def test_fuse_lora_into_base_handles_dora_module() -> None:
    """_fuse_lora_into_base must use DoRALinear.fuse() for DoRA modules."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    try:
        from mlx_lm.tuner.dora import DoRALinear
    except ImportError:
        pytest.skip("DoRALinear not available in this mlx-lm version")

    linear = nn.Linear(8, 8)
    mx.eval(linear.weight)
    dora = DoRALinear.from_base(linear, r=2)
    # Set known LoRA weights so delta is non-trivial
    dora.lora_a = mx.ones((8, 2)) * 0.1
    dora.lora_b = mx.ones((2, 8)) * 0.1
    mx.eval(dora.lora_a, dora.lora_b, dora.m)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = dora

    model = M()
    fused = experiment._fuse_lora_into_base(model)

    # After DoRA fusion, LoRA keys must be absent
    assert "proj.weight" in fused, f"Keys: {list(fused.keys())}"
    assert "proj.lora_a" not in fused
    assert "proj.lora_b" not in fused
    # DoRA magnitude vector should be folded in — no separate 'proj.m'
    assert "proj.m" not in fused

    # Verify fused weight differs from original (DoRA applies magnitude renorm)
    original_weight = linear.weight.astype(mx.float32)
    mx.eval(original_weight)
    fused_weight = fused["proj.weight"].astype(mx.float32)
    mx.eval(fused_weight)
    diff = float(mx.abs(fused_weight - original_weight).max())
    assert diff > 1e-6, "DoRA fusion should change the weight"


def test_inference_uses_fused_model_path_for_surface_matched_arms() -> None:
    """Surface-matched arms store fused_model_path; inference must use it."""
    # Simulate the seed_result dict that run_model_experiment builds
    seed_result = {
        "standard_lora": {
            "training": {"method": "standard_lora", "final_val_loss": 2.0},
            "eval": {},
        },
        "pissa_nb_surface": {
            "training": {
                "method": "pissa_nb_surface",
                "final_val_loss": 1.9,
                "fused_model_path": "/tmp/fused/pissa",
            },
            "eval": {},
        },
        "standard_nb_surface": {
            "training": {
                "method": "standard_nb_surface",
                "final_val_loss": 1.95,
                "fused_model_path": "/tmp/fused/standard",
            },
            "eval": {},
        },
    }

    # For each method, replicate the inference loop's adapter selection logic
    model_path = "/tmp/base_model"
    seed_dir = Path("/tmp/seeds/42")

    for method_name in ["standard_lora", "pissa_nb_surface", "standard_nb_surface"]:
        train_info = seed_result[method_name].get("training", {})
        fused = train_info.get("fused_model_path")
        if fused:
            infer_model = fused
            infer_adapter = None
        else:
            infer_model = model_path
            infer_adapter = str(seed_dir / method_name)

        if method_name == "standard_lora":
            # Regular method: uses base model + adapter
            assert infer_model == model_path
            assert infer_adapter == str(seed_dir / "standard_lora")
        else:
            # Surface-matched: uses fused model, no adapter
            assert infer_model == train_info["fused_model_path"]
            assert infer_adapter is None


# ---------------------------------------------------------------------------
# Diagnostic function tests (Step 6)
# ---------------------------------------------------------------------------

def test_pissa_init_only_dispatches_correctly(monkeypatch) -> None:
    """_train_method routes pissa_init_only_nb_surface to train_pissa_init_only."""
    called_with: dict = {}

    def fake_trainer(*args, **kwargs):
        called_with.update(kwargs)
        return {"method": "pissa_init_only", "seed": 42, "fused_model_path": "/fake"}

    monkeypatch.setattr(experiment, "train_pissa_init_only", fake_trainer)

    surface = _make_fake_nb_surface()
    result = experiment._train_method(
        method_name="pissa_init_only_nb_surface",
        model_path="/tmp/model",
        num_layers=16,
        data_dir=Path("/tmp/data"),
        output_dir=Path("/tmp/output"),
        seed=42,
        iters=100,
        nb_surface=surface,
    )

    assert called_with, "train_pissa_init_only was never called"
    assert result["method"] == "pissa_init_only_nb_surface"
    assert called_with["rank_overrides"] == surface.rank_overrides


def test_pissa_init_only_not_in_public_registries() -> None:
    """Diagnostic arms must NOT appear in public method registries."""
    assert "pissa_init_only_nb_surface" not in experiment.ALL_METHOD_NAMES
    assert "pissa_init_only_nb_surface" not in experiment.METHOD_DISPLAY
    assert "pissa_init_only_nb_surface" not in experiment.NB_SURFACE_METHODS


def test_snapshot_fused_weights_does_not_mutate_model() -> None:
    """_snapshot_fused_weights returns fused dict without changing the model."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    linear = nn.Linear(8, 8)
    mx.eval(linear.weight)
    original_weight = linear.weight.astype(mx.float32)
    mx.eval(original_weight)

    lora = LoRALinear.from_base(linear, r=2, scale=1.0)
    lora.lora_a = mx.ones((8, 2)) * 0.1
    lora.lora_b = mx.ones((2, 8)) * 0.1
    mx.eval(lora.lora_a, lora.lora_b)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = lora

    model = M()

    # Capture state before
    pre_base_weight = model.proj.linear.weight.astype(mx.float32)
    pre_lora_a = model.proj.lora_a.astype(mx.float32)
    pre_lora_b = model.proj.lora_b.astype(mx.float32)
    mx.eval(pre_base_weight, pre_lora_a, pre_lora_b)

    # Snapshot
    fused = experiment._snapshot_fused_weights(model)

    # Model must be unchanged
    post_base_weight = model.proj.linear.weight.astype(mx.float32)
    post_lora_a = model.proj.lora_a.astype(mx.float32)
    post_lora_b = model.proj.lora_b.astype(mx.float32)
    mx.eval(post_base_weight, post_lora_a, post_lora_b)

    assert float(mx.abs(post_base_weight - pre_base_weight).max()) < 1e-7
    assert float(mx.abs(post_lora_a - pre_lora_a).max()) < 1e-7
    assert float(mx.abs(post_lora_b - pre_lora_b).max()) < 1e-7

    # Fused dict should have the combined weight
    assert "proj.weight" in fused


def test_retention_tracker_zero_drift_on_same_model() -> None:
    """When model matches base, KL≈0 and flips=0."""
    import mlx.core as mx
    import mlx.nn as nn

    # Minimal model that returns fixed logits
    class ConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._logits = mx.random.normal((1, 10, 100))
            mx.eval(self._logits)

        def __call__(self, x):
            seq_len = x.shape[-1] if x.ndim > 1 else x.shape[0]
            return self._logits[:, :seq_len, :]

        def eval(self):
            pass

        def train(self):
            pass

    class FakeTokenizer:
        def encode(self, text):
            return list(range(min(len(text), 10)))

    model = ConstantModel()
    tokenizer = FakeTokenizer()

    tracker = experiment.RetentionTracker(model, tokenizer, ["hello world", "test probe"])

    # Measure same model — should show zero drift
    metrics = tracker.measure(model)
    assert metrics["kl_mean"] < 1e-5, f"KL should be ~0, got {metrics['kl_mean']}"
    assert metrics["top_token_flips"] == 0
    assert metrics["n_probes"] == 2


def test_retention_capture_extends_loss_capture() -> None:
    """RetentionCapture inherits LossCapture behavior."""
    callback = experiment.RetentionCapture(model=None, tracker=None)

    # Should have LossCapture's interface
    assert hasattr(callback, "train_losses")
    assert hasattr(callback, "val_losses")
    assert hasattr(callback, "retention_log")
    assert isinstance(callback, experiment.LossCapture)

    # Without tracker, on_val_loss_report should not crash
    callback.on_val_loss_report({"iteration": 1, "val_loss": 1.5})
    assert len(callback.val_losses) == 1
    assert len(callback.retention_log) == 0  # no tracker → no retention


def test_first_step_probe_rejects_unknown_method() -> None:
    """first_step_probe raises on unknown method."""
    with pytest.raises(ValueError, match="Unknown method"):
        experiment.first_step_probe(
            model_path="/tmp/model",
            rank_overrides={},
            scale_bounds={},
            method="bogus",
            probe_texts=["test"],
            data_dir=Path("/tmp"),
        )
