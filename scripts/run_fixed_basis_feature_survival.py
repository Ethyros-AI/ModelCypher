#!/usr/bin/env python3
"""Owner-run fixed-basis feature-survival measurement across model states."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelcypher.adapters.model_loader import ModelLoader
from modelcypher.backends import get_activation_provider, get_backend
from modelcypher.core.domain.geometry.fixed_basis_survival import (
    measure_fixed_basis_survival,
)
from modelcypher.core.use_cases.observation_identity import (
    build_context_state,
    build_precision_state,
)

DEFAULT_MANIFEST = Path(
    "docs/research/replication/ws4_2/fixed_basis_feature_survival.manifest.json"
)
OUTPUT_SCHEMA = "mc.research.fixed_basis_feature_survival.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure feature survival in a frozen reference-fitted basis.",
    )
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    return parser


def run_replication(
    *,
    reference_model_path: str,
    candidate_model_path: str,
    basis_path: str,
    probes_path: str,
    output_dir: str,
    manifest_path: str = str(DEFAULT_MANIFEST),
    backend: Any | None = None,
    model_loader: Any | None = None,
    activation_provider: Any | None = None,
) -> Path:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    prompts = _load_probes(Path(probes_path))

    resolved_backend = backend or get_backend("mlx")
    resolved_loader = model_loader or ModelLoader(resolved_backend)
    resolved_provider = activation_provider or get_activation_provider(resolved_backend)
    reference_tokens, reference = _collect_model_activations(
        model_path=reference_model_path,
        prompts=prompts,
        layers=None,
        backend=resolved_backend,
        loader=resolved_loader,
        provider=resolved_provider,
        expected_tokens=None,
    )
    candidate_tokens, candidate = _collect_model_activations(
        model_path=candidate_model_path,
        prompts=prompts,
        layers=sorted(reference),
        backend=resolved_backend,
        loader=resolved_loader,
        provider=resolved_provider,
        expected_tokens=reference_tokens,
    )
    if candidate_tokens != reference_tokens:
        raise ValueError("Candidate tokenization differs from the reference tokenization")

    basis_file = Path(basis_path).expanduser().resolve()
    basis_weights = resolved_backend.load_safetensors(str(basis_file))
    basis_by_layer = _resolve_basis_layers(basis_weights)
    missing_reference = sorted(set(basis_by_layer) - set(reference))
    missing_candidate = sorted(set(basis_by_layer) - set(candidate))
    if missing_reference or missing_candidate:
        raise ValueError(
            "Frozen basis references unavailable layers: "
            f"reference={missing_reference}, candidate={missing_candidate}"
        )

    rows: list[dict[str, Any]] = []
    for layer in sorted(basis_by_layer):
        reference_array = resolved_backend.array(reference[layer])
        candidate_array = resolved_backend.array(candidate[layer])
        result = measure_fixed_basis_survival(
            reference_array,
            candidate_array,
            basis_by_layer[layer],
            backend=resolved_backend,
        )
        rows.append(
            {
                "layer": layer,
                "observationCount": len(prompts),
                "basisFeatureCount": int(basis_by_layer[layer].shape[0]),
                "referenceResidualRatio": result.reference_residual_ratio,
                "candidateResidualRatio": result.candidate_residual_ratio,
                "coefficientRelativeChange": result.coefficient_relative_change,
                "coefficientCosine": result.coefficient_cosine,
                "referenceFeatureEnergy": resolved_backend.tolist(
                    result.reference_feature_energy
                ),
                "candidateFeatureEnergy": resolved_backend.tolist(
                    result.candidate_feature_energy
                ),
                "featureEnergyRatio": resolved_backend.tolist(result.feature_energy_ratio),
            }
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prompt_manifest = _prompt_manifest(prompts, Path(probes_path))
    runtime_manifest = {
        **manifest,
        "schema": OUTPUT_SCHEMA,
        "sourceManifest": str(manifest_file),
        "runId": destination.name,
        "timestampUtc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_commit(),
        "referenceModel": str(Path(reference_model_path).expanduser().resolve()),
        "candidateModel": str(Path(candidate_model_path).expanduser().resolve()),
        "basis": str(basis_file),
        "basisDigestSha256": _file_digest(basis_file),
        "probePath": str(Path(probes_path).expanduser().resolve()),
        "probeDigestSha256": _file_digest(Path(probes_path)),
        "contextState": build_context_state(prompt_manifest),
        "precisionState": build_precision_state(
            backend=resolved_backend,
            targets=[
                {
                    "label": "reference",
                    "model": reference_model_path,
                    "adapter": None,
                },
                {
                    "label": "candidate",
                    "model": candidate_model_path,
                    "adapter": None,
                },
            ],
        ),
        "measurementOperator": {
            "id": "modelcypher.fixed_basis_feature_survival.v1",
            "activation": "mean_pooled_post_block_residual",
            "basisPolicy": "reference_fitted_and_frozen",
            "observationAlignment": "identical_prompt_order_and_token_ids",
            "projection": "Moore-Penrose least-squares coefficients",
            "thresholdPolicy": "none_raw_measurements_only",
        },
        "basisKeys": {
            str(layer): key for layer, (key, _value) in _basis_entries(basis_weights).items()
        },
    }
    summary = {
        "schema": OUTPUT_SCHEMA,
        "runId": destination.name,
        "status": "owner_review_required",
        "validationClaim": None,
        "layers": rows,
    }
    _write_json(destination / "run_manifest.json", runtime_manifest)
    _write_json(destination / "summary.json", summary)
    _write_jsonl(destination / "layer_measurements.jsonl", rows)
    (destination / "ledger.tsv").write_text(_ledger_header() + "\n", encoding="utf-8")
    (destination / "REPORT.md").write_text(
        _report(summary, manifest_file),
        encoding="utf-8",
    )

    # TODO(owner): run the fixed-basis comparison on real precision states per R4.
    return destination


def _collect_model_activations(
    *,
    model_path: str,
    prompts: list[str],
    layers: list[int] | None,
    backend,
    loader,
    provider,
    expected_tokens: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[tuple[int, ...], ...], dict[int, list[list[float]]]]:
    model, tokenizer = loader.load_model(model_path)
    token_rows: list[tuple[int, ...]] = []
    selected_layers = list(layers) if layers is not None else None
    activations: dict[int, list[list[float]]] = (
        {layer: [] for layer in selected_layers} if selected_layers is not None else {}
    )
    for prompt_index, prompt in enumerate(prompts):
        token_ids = tuple(int(value) for value in tokenizer.encode(prompt))
        if expected_tokens is not None and token_ids != expected_tokens[prompt_index]:
            raise ValueError(
                f"Token identity mismatch at prompt index {prompt_index}: {model_path}"
            )
        token_rows.append(token_ids)
        hidden = provider.collect_hidden_activations(
            model,
            tokenizer,
            prompt,
            token_ids=list(token_ids),
        )
        if selected_layers is None:
            selected_layers = sorted(int(layer) for layer in hidden)
            activations = {layer: [] for layer in selected_layers}
        missing = [layer for layer in selected_layers if layer not in hidden]
        if missing:
            raise ValueError(f"Model did not expose basis layers: {missing}")
        for layer in selected_layers:
            activations[layer].append(
                [float(value) for value in backend.tolist(hidden[layer])]
            )
    del model, tokenizer
    if hasattr(backend, "clear_cache"):
        backend.clear_cache()
    gc.collect()
    return tuple(token_rows), activations


def _basis_layer_index(key: str) -> int:
    for prefix in ("layer_", "layer."):
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
    raise ValueError(
        f"Unsupported basis key {key!r}; expected layer_<index> or layer.<index>"
    )


def _basis_entries(weights: dict[str, Any]) -> dict[int, tuple[str, Any]]:
    entries: dict[int, tuple[str, Any]] = {}
    for key, value in weights.items():
        layer = _basis_layer_index(key)
        if layer in entries:
            raise ValueError(f"Multiple frozen bases declared for layer {layer}")
        entries[layer] = (key, value)
    if not entries:
        raise ValueError("Frozen basis file contains no layer matrices")
    return entries


def _resolve_basis_layers(weights: dict[str, Any]) -> dict[int, Any]:
    layers = {layer: value for layer, (_key, value) in _basis_entries(weights).items()}
    for layer, value in layers.items():
        if len(value.shape) != 2:
            raise ValueError(f"Frozen basis for layer {layer} is not a matrix")
    return layers


def _load_probes(path: Path) -> list[str]:
    resolved = path.expanduser().resolve()
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix == ".json":
        payload = json.loads(text)
        values = payload if isinstance(payload, list) else payload.get("variants", [])
        prompts = [value if isinstance(value, str) else value["text"] for value in values]
    elif resolved.suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
        prompts = [
            value
            if isinstance(value, str)
            else value["text"]
            if "text" in value
            else value["prompt"]
            for value in values
        ]
    else:
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        raise ValueError("Probe input is empty")
    return prompts


def _prompt_manifest(prompts: list[str], source: Path) -> dict[str, Any]:
    return {
        "schema": "mc.prompt_family.v1",
        "name": source.stem,
        "metadata": {"source": str(source.expanduser().resolve())},
        "variants": [
            {
                "case_id": f"prompt_{index}",
                "variant_id": "precision_pair",
                "text": text,
            }
            for index, text in enumerate(prompts)
        ],
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = (
        "linkedBlocker",
        "workOrder",
        "claimContract",
        "primaryObservable",
        "explicitFalsifier",
        "mutableSurface",
        "frozenSurfaces",
        "baselineCommand",
        "comparisonBudget",
        "artifactDirectory",
        "ledgerPath",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError("Replication manifest is missing: " + ", ".join(missing))


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.expanduser().resolve().read_bytes()).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ledger_header() -> str:
    return "\t".join(
        (
            "run_id",
            "timestamp_utc",
            "commit",
            "status",
            "claim",
            "mutable_surface",
            "frozen_surfaces",
            "command",
            "primary_observable",
            "artifact_dir",
            "next_falsifier",
        )
    )


def _report(summary: dict[str, Any], manifest_path: Path) -> str:
    return "\n".join(
        (
            "# Fixed-Basis Feature Survival",
            "",
            "**Status:** owner review required; no validation claim has been made.",
            f"**Manifest:** `{manifest_path}`",
            f"**Measured layers:** `{len(summary['layers'])}`",
            "",
            "Review the raw per-feature energy ratios and reconstruction terms,",
            "compare them with perplexity and whole-state geometry in the same R4",
            "sweep, then append the protocol decision to `ledger.tsv`.",
            "",
        )
    )


def main() -> int:
    args = build_parser().parse_args()
    output = run_replication(
        reference_model_path=args.reference_model,
        candidate_model_path=args.candidate_model,
        basis_path=args.basis,
        probes_path=args.probes,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
