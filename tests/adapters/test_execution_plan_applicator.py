from __future__ import annotations

from dataclasses import dataclass

import pytest

from modelcypher.adapters.execution_plan_applicator import apply_execution_plan
from modelcypher.core.domain.inference import LayerExecutionPlan


@dataclass
class _FakeBase:
    embed_tokens: object
    layers: list[object]


@dataclass
class _FakeModel:
    model: _FakeBase


def _make_model(layer_count: int) -> _FakeModel:
    layers = [object() for _ in range(layer_count)]
    return _FakeModel(model=_FakeBase(embed_tokens=object(), layers=layers))


def test_apply_execution_plan_reorders_layers_by_reference_and_restores() -> None:
    model = _make_model(4)
    original_layers = model.model.layers
    plan = LayerExecutionPlan.from_rys(4, 1, 3)

    with apply_execution_plan(model, plan):
        planned_layers = model.model.layers
        assert planned_layers is not original_layers
        assert len(planned_layers) == plan.execution_layer_count
        for execution_step, source_idx in enumerate(plan.layer_indices):
            assert planned_layers[execution_step] is original_layers[source_idx]

    assert model.model.layers is original_layers
    assert model.model.layers == original_layers


def test_apply_execution_plan_restores_layers_after_exception() -> None:
    model = _make_model(5)
    original_layers = model.model.layers
    plan = LayerExecutionPlan.from_rys(5, 2, 4)

    with pytest.raises(RuntimeError, match="boom"):
        with apply_execution_plan(model, plan):
            raise RuntimeError("boom")

    assert model.model.layers is original_layers


def test_apply_execution_plan_rejects_non_list_layers() -> None:
    model = _FakeModel(model=_FakeBase(embed_tokens=object(), layers=list(object() for _ in range(3))))
    model.model.layers = tuple(model.model.layers)  # type: ignore[assignment]

    with pytest.raises(NotImplementedError, match="mutable Python list"):
        with apply_execution_plan(model, LayerExecutionPlan.identity(3)):
            pass
