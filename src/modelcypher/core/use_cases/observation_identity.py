"""Reproducibility identities for workflow-first observation bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from modelcypher.ports.backend import Backend


CONTEXT_STATE_SCHEMA = "mc.analyze.context_state.v1"
PRECISION_STATE_SCHEMA = "mc.analyze.precision_state.v1"
MEASUREMENT_OPERATOR_SCHEMA = "mc.analyze.measurement_operator.v1"
MEASUREMENT_OPERATOR_ID = "modelcypher.workflow_observation.v1"


def canonical_json_digest(payload: dict[str, Any]) -> str:
    """Return the SHA-256 identity of a canonical JSON object."""
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_context_state(prompt_family_manifest: dict[str, Any]) -> dict[str, Any]:
    """Identify the exact ordered prompt context consumed by a bundle."""
    variants = prompt_family_manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Context identity requires a non-empty prompt-family manifest")

    variant_order = []
    for row in variants:
        if not isinstance(row, dict):
            raise ValueError("Context identity requires prompt variants to be objects")
        variant_order.append(
            {
                "caseId": str(row.get("case_id", "")),
                "variantId": str(row.get("variant_id", "")),
            }
        )

    return {
        "schema": CONTEXT_STATE_SCHEMA,
        "promptFamilyName": prompt_family_manifest.get("name"),
        "promptFamilySchema": prompt_family_manifest.get("schema"),
        "promptFamilyDigest": {
            "algorithm": "sha256",
            "value": canonical_json_digest(prompt_family_manifest),
        },
        "variantOrder": variant_order,
        "metadata": _json_safe(prompt_family_manifest.get("metadata", {})),
        "promptTextPolicy": "exact_manifest_text",
    }


def build_precision_state(
    *,
    backend: "Backend",
    targets: "Sequence[dict[str, Any]]",
) -> dict[str, Any]:
    """Record backend runtime and model-declared precision for each target."""
    target_states = [_target_precision_state(target) for target in targets]
    declared = [state["declared"] for state in target_states]
    comparable = [
        {
            "dtype": declaration.get("dtype"),
            "quantization": declaration.get("quantization"),
        }
        for declaration in declared
    ]
    all_declared = all(
        declaration.get("dtype") is not None
        or declaration.get("quantization") is not None
        for declaration in declared
    )
    declarations_match = (
        len({json.dumps(item, sort_keys=True) for item in comparable}) == 1
        if all_declared
        else None
    )

    return {
        "schema": PRECISION_STATE_SCHEMA,
        "backend": {
            "type": type(backend).__name__,
            "system": _json_safe(backend.get_system_info()),
        },
        "targets": target_states,
        "allTargetsDeclared": all_declared,
        "declarationsMatch": declarations_match,
    }


def build_measurement_operator(
    *,
    workflow: str,
    spaces: Sequence[str],
    max_tokens: int,
    include_within_target_comparisons: bool,
    include_between_target_comparisons: bool,
) -> dict[str, Any]:
    """Describe the deterministic operator path that produced the bundle."""
    operators = [
        "ModelLoaderPort.generate",
        "ActivationProvider.collect_trajectory_batch",
        "BehavioralAnalyzer.analyze_entropy_trajectory",
        "GeometryAnalysisService.analyze_reasoning_flow",
        "GeodesicTrajectoryService.measure_layer_profile",
        "ChainAnalysisService.analyze_chain",
    ]
    space_operators = {
        "hidden": "ActivationProvider.collect_hidden_activations",
        "embedding": "ActivationProvider.collect_embedding_activations",
        "intermediate": "ActivationProvider.collect_intermediate_activations",
        "gate": "ActivationProvider.collect_gate_activations_batch",
        "q": "ActivationProvider.collect_attention_activations:q",
        "k": "ActivationProvider.collect_attention_activations:k",
        "v": "ActivationProvider.collect_attention_activations:v",
    }
    operators.extend(space_operators[space] for space in spaces)
    return {
        "schema": MEASUREMENT_OPERATOR_SCHEMA,
        "id": MEASUREMENT_OPERATOR_ID,
        "workflow": workflow,
        "parameters": {
            "spaces": list(spaces),
            "maxTokens": max_tokens,
            "withinTargetComparisons": include_within_target_comparisons,
            "betweenTargetComparisons": include_between_target_comparisons,
        },
        "inputPolicy": {
            "generation": "exact_prompt_text",
            "activationCollection": "exact_prompt_text",
            "betweenTargetAlignment": "same_case_variant_prompt_text",
        },
        "operators": operators,
    }


def validate_observation_identity(manifest: dict[str, Any]) -> None:
    """Reject a v2 manifest whose required identities are absent or inconsistent."""
    required = ("contextState", "precisionState", "measurementOperator")
    missing = [field for field in required if not isinstance(manifest.get(field), dict)]
    if missing:
        raise ValueError(
            "Observation bundle identity is missing required objects: "
            + ", ".join(missing)
        )

    context = manifest["contextState"]
    precision = manifest["precisionState"]
    operator = manifest["measurementOperator"]
    if context.get("schema") != CONTEXT_STATE_SCHEMA:
        raise ValueError("Observation bundle contextState schema is unsupported")
    if precision.get("schema") != PRECISION_STATE_SCHEMA:
        raise ValueError("Observation bundle precisionState schema is unsupported")
    if operator.get("schema") != MEASUREMENT_OPERATOR_SCHEMA:
        raise ValueError("Observation bundle measurementOperator schema is unsupported")
    if operator.get("id") != MEASUREMENT_OPERATOR_ID:
        raise ValueError("Observation bundle measurementOperator id is unsupported")

    prompt_manifest = manifest.get("promptFamilyManifest")
    if not isinstance(prompt_manifest, dict):
        raise ValueError("Observation bundle is missing promptFamilyManifest")
    digest = context.get("promptFamilyDigest")
    if not isinstance(digest, dict) or digest.get("algorithm") != "sha256":
        raise ValueError("Observation bundle context digest must use sha256")
    if digest.get("value") != canonical_json_digest(prompt_manifest):
        raise ValueError("Observation bundle context digest does not match promptFamilyManifest")


def _target_precision_state(target: dict[str, Any]) -> dict[str, Any]:
    model_ref = str(target.get("model", ""))
    config_path = _resolve_config_path(model_ref)
    config: dict[str, Any] = {}
    if config_path is not None:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            config = payload

    quantization = config.get("quantization_config", config.get("quantization"))
    declaration = {
        "dtype": _json_safe(config.get("torch_dtype", config.get("dtype"))),
        "quantization": _json_safe(quantization),
    }
    return {
        "targetLabel": target.get("label"),
        "model": model_ref,
        "adapter": target.get("adapter"),
        "declarationSource": str(config_path) if config_path is not None else None,
        "configDigest": canonical_json_digest(config) if config else None,
        "declared": declaration,
    }


def _resolve_config_path(model_ref: str) -> Path | None:
    if not model_ref:
        return None
    path = Path(model_ref).expanduser()
    candidate = path / "config.json" if path.is_dir() else path
    if candidate.name != "config.json" or not candidate.is_file():
        return None
    return candidate.resolve()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
