#!/usr/bin/env python3
"""Owner-run replication of contextual curvature and next-token entropy.

The paper-specific settings are loaded from a tracked manifest. This script
does not contain fallback experiment constants and never emits a validation
verdict; it writes raw observations and an owner-review report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from modelcypher.adapters.model_loader import ModelLoader
from modelcypher.backends import get_backend
from modelcypher.core.domain.geometry.contextual_curvature import (
    compute_contextual_curvature,
)
from modelcypher.core.use_cases.observation_identity import (
    build_context_state,
    build_precision_state,
    canonical_json_digest,
)

DEFAULT_MANIFEST = Path(
    "docs/research/replication/ws4_2/contextual_curvature.manifest.json"
)
OUTPUT_SCHEMA = "mc.research.contextual_curvature_replication.v1"
ARMS = (
    "full_space",
    "random_subspace",
    "activation_subspace",
    "trajectory_subspace",
    "planar_subspace",
)


@dataclass(frozen=True)
class PromptTrace:
    prompt_index: int
    text: str
    token_ids: tuple[int, ...]
    layer_positions: dict[int, np.ndarray]
    logits: np.ndarray
    entropy_bits: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the owner-only WS4.2 contextual-curvature replication.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--target-layer", type=int)
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="Write the layer scan without residual interventions.",
    )
    return parser


def next_token_entropy_bits(logits: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy in bits for each row of logits."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Logits must have shape [tokens, vocabulary]")
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    log_partition = np.log(np.sum(exp_values, axis=1)) + np.max(values, axis=1)
    expected_logit = np.sum(probabilities * values, axis=1)
    return (log_partition - expected_logit) / math.log(2.0)


def pearson_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Return raw or importance-weighted Pearson correlation."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Pearson inputs must be equal-length vectors")
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != x.shape:
            raise ValueError("Pearson weights must match the observations")
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0:
        return float("nan")
    x_mean = float(np.sum(w * x) / weight_sum)
    y_mean = float(np.sum(w * y) / weight_sum)
    x_centered = x - x_mean
    y_centered = y - y_mean
    covariance = float(np.sum(w * x_centered * y_centered))
    x_energy = float(np.sum(w * x_centered * x_centered))
    y_energy = float(np.sum(w * y_centered * y_centered))
    denominator = math.sqrt(x_energy * y_energy)
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return covariance / denominator


def cross_validated_ols_correlation(
    predictor: np.ndarray,
    target: np.ndarray,
    *,
    folds: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Paper-matched OLS, fold-wise Pearson, Fisher pooling, and t interval."""
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if len(x) < folds:
        raise ValueError("Cross-validation requires at least one observation per fold")
    fold_indices = np.array_split(rng.permutation(len(x)), folds)
    correlations: list[float] = []
    all_indices = np.arange(len(x))
    for test_indices in fold_indices:
        train_mask = np.ones(len(x), dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]
        design = np.column_stack((np.ones(len(train_indices)), x[train_indices]))
        coefficients = np.linalg.lstsq(design, y[train_indices], rcond=None)[0]
        predictions = np.column_stack(
            (np.ones(len(test_indices)), x[test_indices])
        ) @ coefficients
        correlations.append(pearson_correlation(predictions, y[test_indices]))

    finite = np.asarray([value for value in correlations if np.isfinite(value)])
    if len(finite) != folds:
        raise ValueError("At least one OLS fold produced a degenerate correlation")
    unit_bound = np.nextafter(1.0, 0.0)
    z_values = np.arctanh(np.clip(finite, -unit_bound, unit_bound))
    z_mean = float(np.mean(z_values))
    standard_error = float(np.std(z_values, ddof=1) / math.sqrt(folds))
    alpha = 1.0 - confidence
    critical = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df=folds - 1))
    return {
        "pooledPearson": float(np.tanh(z_mean)),
        "confidenceInterval": [
            float(np.tanh(z_mean - critical * standard_error)),
            float(np.tanh(z_mean + critical * standard_error)),
        ],
        "foldPearson": correlations,
        "folds": folds,
    }


def importance_weights(
    reference: np.ndarray,
    family: np.ndarray,
    *,
    bins: int,
    epsilon: float,
    cap: float,
) -> np.ndarray:
    """Implement the paper's equal-width |delta-C| importance reweighting."""
    reference_values = np.abs(np.asarray(reference, dtype=np.float64))
    family_values = np.abs(np.asarray(family, dtype=np.float64))
    lower = float(np.min(reference_values))
    upper = float(np.max(reference_values))
    if upper <= lower:
        return np.ones_like(family_values)
    edges = np.linspace(lower, upper, bins + 1)
    reference_indices = np.clip(
        np.digitize(reference_values, edges[1:-1]),
        0,
        bins - 1,
    )
    family_indices = np.clip(
        np.digitize(family_values, edges[1:-1]),
        0,
        bins - 1,
    )
    reference_histogram = np.bincount(reference_indices, minlength=bins)
    family_histogram = np.bincount(family_indices, minlength=bins)
    q = reference_histogram / max(int(np.sum(reference_histogram)), 1)
    p = family_histogram / max(int(np.sum(family_histogram)), 1)
    return np.clip(
        q[family_indices] / (p[family_indices] + epsilon),
        1.0 / cap,
        cap,
    )


def run_replication(
    *,
    model_path: str,
    probes_path: str,
    output_dir: str,
    manifest_path: str = str(DEFAULT_MANIFEST),
    target_layer: int | None = None,
    observe_only: bool = False,
) -> Path:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    parameters = manifest["paperParameters"]
    _validate_manifest(manifest)
    prompts = _load_probes(Path(probes_path))
    manifest_digest = canonical_json_digest(manifest)
    seed_hex_chars = 64 // 4
    rng = np.random.default_rng(int(manifest_digest[:seed_hex_chars], 16))

    backend = get_backend("mlx")
    model, tokenizer = ModelLoader(backend).load_model(model_path)
    if not hasattr(backend, "collect_logits_with_residual_intervention"):
        raise RuntimeError("MLX backend lacks the residual intervention hook")

    traces = [
        _collect_trace(
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            text=text,
            prompt_index=index,
        )
        for index, text in enumerate(prompts)
    ]
    layer_scan = _build_layer_scan(
        traces,
        backend=backend,
        window_size=int(parameters["windowSize"]["value"]),
        minimum_token_position=int(parameters["minimumTokenPosition"]["value"]),
        folds=int(parameters["crossValidationFolds"]["value"]),
        confidence=float(parameters["confidenceLevel"]["value"]),
        rng=rng,
    )
    selected_layer = target_layer
    if selected_layer is None:
        selected_layer = min(
            layer_scan,
            key=lambda row: (row["meanContextualCurvatureRadians"], row["layer"]),
        )["layer"]

    intervention_rows: list[dict[str, Any]] = []
    arm_summary: list[dict[str, Any]] = []
    if not observe_only:
        intervention_rows = _run_interventions(
            traces,
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            target_layer=selected_layer,
            parameters=parameters,
            rng=rng,
        )
        arm_summary = _summarize_interventions(
            intervention_rows,
            parameters=parameters,
            rng=rng,
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit = _git_commit()
    prompt_manifest = _prompt_manifest(prompts, Path(probes_path))
    runtime_manifest = {
        **manifest,
        "schema": OUTPUT_SCHEMA,
        "sourceManifest": str(manifest_file),
        "sourceManifestDigest": manifest_digest,
        "runId": destination.name,
        "timestampUtc": timestamp,
        "commit": commit,
        "model": str(Path(model_path).expanduser().resolve()),
        "probePath": str(Path(probes_path).expanduser().resolve()),
        "probeDigestSha256": _file_digest(Path(probes_path)),
        "contextState": build_context_state(prompt_manifest),
        "precisionState": build_precision_state(
            backend=backend,
            targets=[{"label": "model", "model": model_path, "adapter": None}],
        ),
        "measurementOperator": {
            "id": "modelcypher.contextual_curvature_replication.v1",
            "trajectory": "post_transformer_block_residual_stream",
            "curvature": "compute_contextual_curvature",
            "entropy": "softmax_shannon_bits",
            "intervention": "post_block_single_token_additive_delta",
            "importanceReference": "full_space_at_selected_intervention_layer",
            "targetLayerPolicy": (
                "explicit" if target_layer is not None else "minimum_mean_contextual_curvature"
            ),
        },
        "selectedInterventionLayer": selected_layer,
        "observeOnly": observe_only,
    }
    summary = {
        "schema": OUTPUT_SCHEMA,
        "runId": destination.name,
        "status": "owner_review_required",
        "validationClaim": None,
        "layerScan": layer_scan,
        "selectedInterventionLayer": selected_layer,
        "interventionArms": arm_summary,
        "rawInterventionCount": len(intervention_rows),
    }
    _write_json(destination / "run_manifest.json", runtime_manifest)
    _write_json(destination / "summary.json", summary)
    _write_jsonl(destination / "interventions.jsonl", intervention_rows)
    (destination / "ledger.tsv").write_text(_ledger_header() + "\n", encoding="utf-8")
    (destination / "REPORT.md").write_text(
        _report(summary, manifest_file),
        encoding="utf-8",
    )

    # TODO(owner): run the MLX real-model replication and review discrepancies per WS4.2.
    return destination


def _collect_trace(*, backend, model, tokenizer, text: str, prompt_index: int) -> PromptTrace:
    token_ids = tuple(int(value) for value in tokenizer.encode(text))
    hidden = backend.collect_hidden_activations(
        model,
        tokenizer,
        [text],
        mask_mode="causal",
    )
    logits = backend.collect_logits_with_residual_intervention(
        model,
        tokenizer,
        text,
        token_ids=list(token_ids),
    )
    layer_positions = {
        int(layer): np.asarray(backend.tolist(values[0]), dtype=np.float64)
        for layer, values in hidden.items()
    }
    logits_array = np.asarray(backend.tolist(logits[0]), dtype=np.float64)
    trace = PromptTrace(
        prompt_index=prompt_index,
        text=text,
        token_ids=token_ids,
        layer_positions=layer_positions,
        logits=logits_array,
        entropy_bits=next_token_entropy_bits(logits_array),
    )
    del hidden, logits
    if hasattr(backend, "clear_cache"):
        backend.clear_cache()
    return trace


def _build_layer_scan(
    traces: list[PromptTrace],
    *,
    backend,
    window_size: int,
    minimum_token_position: int,
    folds: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_ids = sorted(set.intersection(*(set(trace.layer_positions) for trace in traces)))
    for layer in layer_ids:
        curvature_values: list[float] = []
        entropy_values: list[float] = []
        for trace in traces:
            profile = compute_contextual_curvature(
                backend.array(trace.layer_positions[layer]),
                backend=backend,
                window_size=window_size,
            )
            values = backend.tolist(profile.contextual_curvature_radians)
            for position, curvature in zip(profile.token_positions, values, strict=True):
                if position >= minimum_token_position:
                    curvature_values.append(float(curvature))
                    entropy_values.append(float(trace.entropy_bits[position]))
        cv = cross_validated_ols_correlation(
            np.asarray(curvature_values),
            np.asarray(entropy_values),
            folds=folds,
            confidence=confidence,
            rng=rng,
        )
        rows.append(
            {
                "layer": layer,
                "observationCount": len(curvature_values),
                "meanContextualCurvatureRadians": float(np.mean(curvature_values)),
                "entropyPrediction": cv,
            }
        )
    return rows


def _run_interventions(
    traces: list[PromptTrace],
    *,
    backend,
    model,
    tokenizer,
    target_layer: int,
    parameters: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    window_size = int(parameters["windowSize"]["value"])
    minimum_position = int(parameters["minimumTokenPosition"]["value"])
    subspace_dimension = int(parameters["subspaceDimension"]["value"])
    perturbation_count = int(parameters["perturbationsPerToken"]["value"])
    scale = float(parameters["perturbationScale"]["value"])
    activation_rows = np.concatenate(
        [trace.layer_positions[target_layer] for trace in traces],
        axis=0,
    )
    activation_basis = _principal_basis(activation_rows, subspace_dimension)
    hidden_width = activation_rows.shape[1]
    rows: list[dict[str, Any]] = []

    for trace in traces:
        positions = trace.layer_positions[target_layer]
        velocities = np.diff(positions, axis=0)
        baseline_profile = compute_contextual_curvature(
            backend.array(positions),
            backend=backend,
            window_size=window_size,
        )
        baseline_curvature = {
            position: float(value)
            for position, value in zip(
                baseline_profile.token_positions,
                backend.tolist(baseline_profile.contextual_curvature_radians),
                strict=True,
            )
        }
        last_position = len(trace.token_ids) - 2
        for token_position in range(minimum_position, last_position + 1):
            step_norm = float(np.linalg.norm(velocities[token_position]))
            random_basis = _random_basis(hidden_width, subspace_dimension, rng)
            trajectory_basis = _principal_basis(
                velocities[: token_position + 1],
                subspace_dimension,
            )
            planar_basis = _row_space_basis(
                velocities[token_position - subspace_dimension : token_position],
                subspace_dimension,
            )
            arm_bases = {
                "full_space": None,
                "random_subspace": random_basis,
                "activation_subspace": activation_basis,
                "trajectory_subspace": trajectory_basis,
                "planar_subspace": planar_basis,
            }
            for arm in ARMS:
                basis = arm_bases[arm]
                for perturbation_index in range(perturbation_count):
                    direction = _sample_direction(hidden_width, basis, rng)
                    delta = direction * (scale * step_norm)
                    perturbed_positions = positions.copy()
                    perturbed_positions[token_position] += delta
                    profile = compute_contextual_curvature(
                        backend.array(perturbed_positions),
                        backend=backend,
                        window_size=window_size,
                    )
                    profile_index = token_position - profile.token_positions[0]
                    changed_curvature = float(
                        backend.tolist(profile.contextual_curvature_radians)[profile_index]
                    )
                    perturbed_logits = backend.collect_logits_with_residual_intervention(
                        model,
                        tokenizer,
                        trace.text,
                        target_layer=target_layer,
                        token_position=token_position,
                        delta=backend.array(delta),
                        token_ids=list(trace.token_ids),
                    )
                    logits_row = np.asarray(
                        backend.tolist(perturbed_logits[0, token_position, :]),
                        dtype=np.float64,
                    )[None, :]
                    changed_entropy = float(next_token_entropy_bits(logits_row)[0])
                    baseline_logits = trace.logits[token_position]
                    rank_correlation = float(
                        scipy_stats.spearmanr(baseline_logits, logits_row[0]).statistic
                    )
                    rows.append(
                        {
                            "promptIndex": trace.prompt_index,
                            "tokenPosition": token_position,
                            "layer": target_layer,
                            "arm": arm,
                            "perturbationIndex": perturbation_index,
                            "deltaCurvatureRadians": (
                                changed_curvature - baseline_curvature[token_position]
                            ),
                            "deltaEntropyBits": (
                                changed_entropy - float(trace.entropy_bits[token_position])
                            ),
                            "outputRankSpearman": rank_correlation,
                        }
                    )
    return rows


def _summarize_interventions(
    rows: list[dict[str, Any]],
    *,
    parameters: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    full_reference = np.asarray(
        [row["deltaCurvatureRadians"] for row in rows if row["arm"] == "full_space"]
    )
    bins = int(parameters["importanceBins"]["value"])
    epsilon = float(parameters["importanceEpsilon"]["value"])
    cap = float(parameters["importanceWeightCap"]["value"])
    bootstrap_count = int(parameters["bootstrapReplicates"]["value"])
    confidence = float(parameters["confidenceLevel"]["value"])
    summary: list[dict[str, Any]] = []
    unit_bound = np.nextafter(1.0, 0.0)

    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        family_curvature = np.asarray([row["deltaCurvatureRadians"] for row in arm_rows])
        family_weights = importance_weights(
            full_reference,
            family_curvature,
            bins=bins,
            epsilon=epsilon,
            cap=cap,
        )
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, row in enumerate(arm_rows):
            grouped.setdefault((row["promptIndex"], row["tokenPosition"]), []).append(index)
        token_correlations: list[float] = []
        for indices in grouped.values():
            selected = np.asarray(indices)
            correlation = pearson_correlation(
                family_curvature[selected],
                np.asarray([arm_rows[index]["deltaEntropyBits"] for index in indices]),
                weights=family_weights[selected],
            )
            if np.isfinite(correlation):
                token_correlations.append(correlation)
        z_values = np.arctanh(
            np.clip(np.asarray(token_correlations), -unit_bound, unit_bound)
        )
        bootstrap = np.asarray(
            [
                np.tanh(np.mean(rng.choice(z_values, size=len(z_values), replace=True)))
                for _ in range(bootstrap_count)
            ]
        )
        tail = (1.0 - confidence) / 2.0
        summary.append(
            {
                "arm": arm,
                "tokenCount": len(token_correlations),
                "weightedPearson": float(np.tanh(np.mean(z_values))),
                "bootstrapConfidenceInterval": [
                    float(np.quantile(bootstrap, tail)),
                    float(np.quantile(bootstrap, 1.0 - tail)),
                ],
            }
        )
    return summary


def _principal_basis(values: np.ndarray, dimensions: int) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    return _row_space_basis(centered, dimensions)


def _row_space_basis(values: np.ndarray, dimensions: int) -> np.ndarray:
    _left, singular_values, right = np.linalg.svd(values, full_matrices=False)
    rank = int(np.sum(singular_values > np.finfo(singular_values.dtype).eps * singular_values[0]))
    if rank < dimensions:
        raise ValueError("Declared subspace dimension exceeds the measured rank")
    return right[:dimensions]


def _random_basis(
    width: int,
    dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = rng.normal(size=(width, dimensions))
    basis, _ = np.linalg.qr(matrix)
    return basis[:, :dimensions].T


def _sample_direction(
    width: int,
    basis: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if basis is None:
        direction = rng.normal(size=width)
    else:
        direction = rng.normal(size=basis.shape[0]) @ basis
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Random perturbation direction is numerically degenerate")
    return direction / norm


def _load_probes(path: Path) -> list[str]:
    text = path.expanduser().resolve().read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            values = payload
        else:
            values = payload.get("variants", payload.get("prompts", []))
        prompts = [value if isinstance(value, str) else value["text"] for value in values]
    elif path.suffix == ".jsonl":
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
                "variant_id": "observed",
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
        "paperParameters",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError("Replication manifest is missing: " + ", ".join(missing))
    for name, setting in manifest["paperParameters"].items():
        if not isinstance(setting, dict) or "value" not in setting or "source" not in setting:
            raise ValueError(f"Paper parameter {name} lacks value/source identity")


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
            "# WS4.2 Contextual Curvature Replication",
            "",
            "**Status:** owner review required; no validation claim has been made.",
            f"**Manifest:** `{manifest_path}`",
            f"**Selected intervention layer:** `{summary['selectedInterventionLayer']}`",
            f"**Layer observations:** `{len(summary['layerScan'])}`",
            f"**Raw interventions:** `{summary['rawInterventionCount']}`",
            "",
            "Review `summary.json` and `interventions.jsonl`, compare the qualitative",
            "profile with the cited paper, document discrepancies, and only then append",
            "a protocol status row to `ledger.tsv`.",
            "",
        )
    )


def main() -> int:
    args = build_parser().parse_args()
    output = run_replication(
        model_path=args.model,
        probes_path=args.probes,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        target_layer=args.target_layer,
        observe_only=args.observe_only,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
