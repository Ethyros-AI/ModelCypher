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
