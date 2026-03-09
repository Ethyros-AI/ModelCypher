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
