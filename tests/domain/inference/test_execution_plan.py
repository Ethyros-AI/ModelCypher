from __future__ import annotations

import pytest

from modelcypher.core.domain.inference import LayerExecutionPlan


def test_identity_plan_uses_original_layer_order() -> None:
    plan = LayerExecutionPlan.identity(4)

    assert plan.base_layer_count == 4
    assert plan.layer_indices == (0, 1, 2, 3)
    assert plan.plan_kind == "identity"
    assert plan.label == "identity"
    assert plan.execution_layer_count == 4


def test_rys_plan_duplicates_expected_block() -> None:
    plan = LayerExecutionPlan.from_rys(8, 2, 5)

    assert plan.layer_indices == (0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7)
    assert plan.plan_kind == "rys"
    assert plan.label == "rys_2_5"
    assert plan.execution_layer_count == 11


@pytest.mark.parametrize(
    ("base_layer_count", "start", "end"),
    [
        (0, 0, 0),
        (4, -1, 2),
        (4, 2, 2),
        (4, 3, 2),
        (4, 1, 5),
    ],
)
def test_rys_plan_rejects_invalid_bounds(
    base_layer_count: int,
    start: int,
    end: int,
) -> None:
    with pytest.raises(ValueError):
        LayerExecutionPlan.from_rys(base_layer_count, start, end)


def test_explicit_plan_rejects_out_of_bounds_indices() -> None:
    with pytest.raises(ValueError):
        LayerExecutionPlan.from_indices(3, [0, 3])


def test_to_dict_is_json_friendly_and_equality_is_stable() -> None:
    left = LayerExecutionPlan.from_indices(5, [0, 2, 4], label="skip")
    right = LayerExecutionPlan.from_indices(5, (0, 2, 4), label="skip")

    assert left == right
    assert left.to_dict() == {
        "baseLayerCount": 5,
        "layerIndices": [0, 2, 4],
        "planKind": "explicit",
        "label": "skip",
    }
