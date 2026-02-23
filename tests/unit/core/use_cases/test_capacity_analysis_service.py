# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.use_cases.capacity_analysis_service import CapacityAnalysisService


class _StubModelLoader:
    def __init__(self, weights):
        self._weights = weights

    def load_weights(self, model_path: str):
        return self._weights

    def iter_weights(self, model_path: str):
        for name in sorted(self._weights.keys()):
            yield name, self._weights[name]


class _InterruptingModelLoader(_StubModelLoader):
    def __init__(self, weights, fail_after: int):
        super().__init__(weights)
        self._fail_after = fail_after

    def iter_weights(self, model_path: str):
        for index, name in enumerate(sorted(self._weights.keys())):
            if index >= self._fail_after:
                raise RuntimeError("simulated interruption")
            yield name, self._weights[name]


def test_capacity_service_target_module_and_dim_filters(any_backend) -> None:
    b = any_backend
    weights = {
        "model.layers.0.self_attn.q_proj.weight": b.eye(4),
        "model.layers.0.self_attn.v_proj.weight": b.eye(8),
        "model.layers.0.self_attn.k_proj.weight": b.eye(6),
        "model.layers.0.self_attn.q_proj.bias": b.zeros((4,)),
    }
    loader = _StubModelLoader(weights)
    service = CapacityAnalysisService(backend=b, model_loader=loader)

    report = service.analyze(
        model_path="/tmp/fake-model",
        target_modules=["q_proj,v_proj"],
        min_dim=5,
        max_dim=8,
    )

    assert report.analyzed_layers == 1
    assert report.layer_reports[0].layer_name == "model.layers.0.self_attn.v_proj.weight"
    assert report.target_modules == ["q_proj", "v_proj"]
    assert report.min_dim == 5
    assert report.max_dim == 8


def test_capacity_service_rejects_invalid_dim_range(any_backend) -> None:
    b = any_backend
    weights = {"layer.weight": b.eye(4)}
    loader = _StubModelLoader(weights)
    service = CapacityAnalysisService(backend=b, model_loader=loader)

    with pytest.raises(ValueError, match="min_dim must be <= max_dim"):
        service.analyze("/tmp/fake-model", min_dim=8, max_dim=4)


def test_capacity_service_resume_matches_uninterrupted(any_backend, tmp_path) -> None:
    b = any_backend
    weights = {
        "model.layers.0.self_attn.q_proj.weight": b.eye(4),
        "model.layers.0.self_attn.v_proj.weight": b.eye(8),
        "model.layers.0.self_attn.k_proj.weight": b.eye(6),
        "model.layers.0.self_attn.q_proj.bias": b.zeros((4,)),
    }
    checkpoint_path = tmp_path / "capacity_checkpoint.json"

    interrupted = CapacityAnalysisService(
        backend=b,
        model_loader=_InterruptingModelLoader(weights, fail_after=2),
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.analyze(
            model_path="/tmp/fake-model",
            checkpoint_path=checkpoint_path,
            resume=False,
        )

    assert checkpoint_path.exists()

    resumed_service = CapacityAnalysisService(
        backend=b,
        model_loader=_StubModelLoader(weights),
    )
    resumed = resumed_service.analyze(
        model_path="/tmp/fake-model",
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    fresh_service = CapacityAnalysisService(
        backend=b,
        model_loader=_StubModelLoader(weights),
    )
    fresh = fresh_service.analyze(model_path="/tmp/fake-model")

    assert resumed.to_dict() == fresh.to_dict()


def test_capacity_service_resume_rejects_model_mismatch(any_backend, tmp_path) -> None:
    b = any_backend
    weights = {"layer.weight": b.eye(4)}
    checkpoint_path = tmp_path / "capacity_checkpoint.json"
    loader = _StubModelLoader(weights)
    service = CapacityAnalysisService(backend=b, model_loader=loader)

    service.analyze(
        model_path="/tmp/fake-model",
        checkpoint_path=checkpoint_path,
        resume=False,
    )

    with pytest.raises(ValueError, match="Checkpoint model path mismatch"):
        service.analyze(
            model_path="/tmp/other-model",
            checkpoint_path=checkpoint_path,
            resume=True,
        )
