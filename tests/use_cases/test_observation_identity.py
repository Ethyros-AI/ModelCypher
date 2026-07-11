"""Observation-bundle reproducibility identity tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from modelcypher.core.use_cases.observation_identity import (
    build_context_state,
    build_measurement_operator,
    build_precision_state,
    validate_observation_identity,
)


class _Backend:
    def get_system_info(self) -> dict[str, object]:
        return {
            "available": True,
            "version": "test",
            "device_platforms": ["cpu"],
        }


def _prompt_manifest() -> dict[str, object]:
    return {
        "schema": "mc.analyze.prompt_family.v2",
        "name": "icl_order",
        "metadata": {
            "demonstration_order": ["demo_a", "demo_b"],
            "label_mapping": {"positive": "A", "negative": "B"},
        },
        "variants": [
            {
                "case_id": "case_1",
                "variant_id": "control",
                "text": "demo a\ndemo b\nquery",
            },
            {
                "case_id": "case_1",
                "variant_id": "reversed",
                "text": "demo b\ndemo a\nquery",
                "comparison_to": "control",
            },
        ],
    }


def test_context_digest_is_stable_and_order_sensitive() -> None:
    manifest = _prompt_manifest()
    first = build_context_state(manifest)
    second = build_context_state(copy.deepcopy(manifest))
    assert first == second
    assert first["metadata"] == manifest["metadata"]
    assert first["variantOrder"] == [
        {"caseId": "case_1", "variantId": "control"},
        {"caseId": "case_1", "variantId": "reversed"},
    ]

    reversed_manifest = copy.deepcopy(manifest)
    reversed_manifest["variants"].reverse()  # type: ignore[union-attr]
    reversed_state = build_context_state(reversed_manifest)
    assert (
        reversed_state["promptFamilyDigest"]["value"]
        != first["promptFamilyDigest"]["value"]
    )


def test_precision_state_reads_declared_dtype_and_quantization(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    config = {
        "torch_dtype": "bfloat16",
        "quantization_config": {"bits": 4, "group_size": 64},
    }
    for model_dir in (left, right):
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    state = build_precision_state(
        backend=_Backend(),  # type: ignore[arg-type]
        targets=(
            {"label": "left", "model": str(left), "adapter": None},
            {"label": "right", "model": str(right), "adapter": None},
        ),
    )

    assert state["allTargetsDeclared"] is True
    assert state["declarationsMatch"] is True
    assert state["targets"][0]["declared"] == {
        "dtype": "bfloat16",
        "quantization": {"bits": 4, "group_size": 64},
    }

    right_config = dict(config)
    right_config["torch_dtype"] = "float16"
    (right / "config.json").write_text(json.dumps(right_config), encoding="utf-8")
    mismatch = build_precision_state(
        backend=_Backend(),  # type: ignore[arg-type]
        targets=(
            {"label": "left", "model": str(left), "adapter": None},
            {"label": "right", "model": str(right), "adapter": None},
        ),
    )
    assert mismatch["declarationsMatch"] is False


def test_measurement_operator_records_exact_surface_and_comparison_policy() -> None:
    operator = build_measurement_operator(
        workflow="compare",
        spaces=("hidden", "q"),
        max_tokens=13,
        include_within_target_comparisons=True,
        include_between_target_comparisons=True,
    )
    assert operator["parameters"] == {
        "spaces": ["hidden", "q"],
        "maxTokens": 13,
        "withinTargetComparisons": True,
        "betweenTargetComparisons": True,
    }
    assert "ActivationProvider.collect_hidden_activations" in operator["operators"]
    assert "ActivationProvider.collect_attention_activations:q" in operator["operators"]
    assert operator["inputPolicy"]["betweenTargetAlignment"] == (
        "same_case_variant_prompt_text"
    )


def test_v2_identity_validation_rejects_tampered_context() -> None:
    prompt_manifest = _prompt_manifest()
    manifest = {
        "bundleVersion": "mc.analyze.bundle.v2",
        "promptFamilyManifest": prompt_manifest,
        "contextState": build_context_state(prompt_manifest),
        "precisionState": build_precision_state(
            backend=_Backend(),  # type: ignore[arg-type]
            targets=({"label": "base", "model": "remote/model", "adapter": None},),
        ),
        "measurementOperator": build_measurement_operator(
            workflow="capture",
            spaces=("hidden",),
            max_tokens=7,
            include_within_target_comparisons=False,
            include_between_target_comparisons=False,
        ),
    }
    validate_observation_identity(manifest)

    manifest["promptFamilyManifest"]["variants"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="context digest does not match"):
        validate_observation_identity(manifest)
