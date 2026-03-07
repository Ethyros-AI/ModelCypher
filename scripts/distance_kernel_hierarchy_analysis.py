#!/usr/bin/env python3
"""Distance-kernel hierarchy falsifier protocol runner.

This script measures one narrow claim:

    For attention heads where distance explains substantial variance in
    the attention profile (as determined by AICc model selection on
    calibration data), monotone exponential decay (M1) is the sufficient
    kernel, and the simpler constant model (M0) is inadequate.
    This holds cross-family.

The output is a run directory with machine-readable artifacts only.
Interpretation and promotion live in the protocol document.

Usage:
    poetry run python scripts/distance_kernel_hierarchy_analysis.py --smoke
    poetry run python scripts/distance_kernel_hierarchy_analysis.py --models /path/to/model
    poetry run python scripts/distance_kernel_hierarchy_analysis.py \
        --models /path/to/model_a \
        --models /path/to/model_b \
        --probe-file docs/research/wave_kernel_probe_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

try:
    from scripts import validate_distance_kernel_hierarchy_artifacts as artifact_validator
except ImportError:  # pragma: no cover - direct script execution path
    import validate_distance_kernel_hierarchy_artifacts as artifact_validator

from modelcypher.adapters.activation_provider import ActivationProviderAdapter
from modelcypher.adapters.model_loader import ModelLoader
from modelcypher.backends import initialize_default_backend

logger = logging.getLogger(__name__)

PROTOCOL_ID = "F-DKH-01"
ARTIFACT_SCHEMA_VERSION = "v1"
CLAIM_FORM = (
    "observable = f(geometry_state, architecture_state, scale_state, "
    "precision_state, measurement_operator)"
)
CLAIM_DESCRIPTION = (
    "For attention heads where distance explains substantial variance, "
    "monotone exponential decay (M1) is the sufficient kernel and the "
    "simpler constant model (M0) is inadequate. This holds cross-family."
)
RESULTS_ROOT = Path("results/distance_kernel_hierarchy")
DEFAULT_PROBE_FILE = Path("docs/research/wave_kernel_probe_manifest.json")
DEFAULT_MODEL_PATHS = (
    Path("/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16"),
    Path("/Volumes/CodeCypher/models/mlx-community/Llama-3.2-3B-Instruct-bf16"),
    Path("/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Base-bf16"),
)
MODEL_RECORD_REQUIRED_KEYS = {
    "record_type",
    "protocol",
    "model_path",
    "model_name",
    "family",
    "status",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeSpec:
    """One probe entry from the committed manifest."""

    id: str
    family: str
    text: str
    smoke_only: bool = False


@dataclass(frozen=True)
class DistanceProfile:
    """Distance-conditioned mean profile with per-distance sample counts."""

    distances: list[int]
    means: list[float]
    counts: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "distances": self.distances,
            "means": self.means,
            "counts": self.counts,
        }

    @property
    def n_points(self) -> int:
        return len(self.distances)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _infer_model_family(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered.startswith("qwen"):
        return "Qwen"
    if lowered.startswith("llama"):
        return "Llama"
    if lowered.startswith("lfm2"):
        return "LFM2"
    return "unknown"


def _infer_precision_state(model_name: str) -> str:
    lowered = model_name.lower()
    if "4bit" in lowered:
        return "4bit"
    if "8bit" in lowered:
        return "8bit"
    if "fp16" in lowered:
        return "fp16"
    if "bf16" in lowered:
        return "bf16"
    return "unknown"


def _resolve_model_paths(raw_paths: list[str] | None) -> tuple[list[Path], bool]:
    if raw_paths:
        return [Path(raw).expanduser().resolve() for raw in raw_paths], False
    return [path.resolve() for path in DEFAULT_MODEL_PATHS], True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _load_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _processed_model_paths(existing_rows: list[dict[str, Any]]) -> set[str]:
    processed: set[str] = set()
    for row in existing_rows:
        if row.get("record_type") == "head_classification" and row.get("status") == "ok":
            processed.add(str(Path(row["model_path"]).resolve()))
    return processed


def _head_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("record_type") == "head_classification" and row.get("status") == "ok"
    ]


def _model_skip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("record_type") == "model_skip"
    ]


# ---------------------------------------------------------------------------
# Probe management
# ---------------------------------------------------------------------------


def load_probe_manifest(path: Path) -> list[ProbeSpec]:
    """Load the committed probe manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    probes_raw = payload["probes"] if isinstance(payload, dict) else payload
    probes: list[ProbeSpec] = []
    for entry in probes_raw:
        probes.append(
            ProbeSpec(
                id=str(entry["id"]),
                family=str(entry["family"]),
                text=str(entry["text"]),
                smoke_only=bool(entry.get("smoke_only", False)),
            )
        )
    return probes


def select_probes(probes: list[ProbeSpec], smoke: bool) -> list[ProbeSpec]:
    """Select either smoke-only or promotable probes."""
    if smoke:
        selected = [probe for probe in probes if probe.smoke_only]
    else:
        selected = [probe for probe in probes if not probe.smoke_only]
    if not selected:
        mode = "smoke" if smoke else "promotable"
        raise ValueError(f"No {mode} probes found in manifest.")
    return selected


def assign_probe_splits(probes: list[ProbeSpec]) -> dict[str, str]:
    """Assign calibration/holdout within each probe family using a stable hash."""
    grouped: dict[str, list[ProbeSpec]] = {}
    for probe in probes:
        grouped.setdefault(probe.family, []).append(probe)

    split_map: dict[str, str] = {}
    for family, family_probes in grouped.items():
        sorted_family = sorted(
            family_probes,
            key=lambda probe: (
                _stable_hash(f"{family}:{probe.id}:{probe.text}"),
                probe.id,
            ),
        )
        if len(sorted_family) == 1:
            split_map[sorted_family[0].id] = "calibration"
            continue
        for idx, probe in enumerate(sorted_family):
            split_map[probe.id] = "calibration" if idx % 2 == 0 else "holdout"
    return split_map


# ---------------------------------------------------------------------------
# Model functions (M0 constant, M1 monotone decay)
# ---------------------------------------------------------------------------


def monotonic_decay(d: np.ndarray, a: float, gamma: float) -> np.ndarray:
    """M1: Pure exponential decay."""
    return a * np.exp(-gamma * d)


# ---------------------------------------------------------------------------
# Profile measurement
# ---------------------------------------------------------------------------


def compute_prompt_measurements(attn_matrix: np.ndarray) -> tuple[DistanceProfile, float]:
    """Return the per-distance profile and nonparametric distance R2."""
    seq_len = int(attn_matrix.shape[0])
    by_distance: dict[int, list[float]] = {}
    all_values: list[float] = []

    for i in range(seq_len):
        for j in range(i + 1):
            distance = i - j
            value = float(attn_matrix[i, j])
            by_distance.setdefault(distance, []).append(value)
            all_values.append(value)

    distances = sorted(by_distance)
    means = [float(np.mean(by_distance[d])) for d in distances]
    counts = [len(by_distance[d]) for d in distances]
    profile = DistanceProfile(distances=distances, means=means, counts=counts)

    all_arr = np.asarray(all_values, dtype=np.float64)
    grand_mean = float(np.mean(all_arr))
    ss_tot = float(np.sum((all_arr - grand_mean) ** 2))
    if ss_tot < 1e-15:
        return profile, 1.0

    predicted: list[float] = []
    mean_lookup = {distance: mean for distance, mean in zip(distances, means, strict=True)}
    for i in range(seq_len):
        for j in range(i + 1):
            predicted.append(mean_lookup[i - j])

    pred_arr = np.asarray(predicted, dtype=np.float64)
    ss_res = float(np.sum((all_arr - pred_arr) ** 2))
    distance_r2 = 1.0 - (ss_res / ss_tot)
    return profile, float(distance_r2)


def aggregate_profiles(profiles: list[DistanceProfile]) -> DistanceProfile | None:
    """Aggregate prompt-level profiles into one count-weighted profile."""
    if not profiles:
        return None

    weighted_sum: dict[int, float] = {}
    count_sum: dict[int, int] = {}
    for profile in profiles:
        for distance, mean, count in zip(
            profile.distances,
            profile.means,
            profile.counts,
            strict=True,
        ):
            weighted_sum[distance] = weighted_sum.get(distance, 0.0) + (mean * count)
            count_sum[distance] = count_sum.get(distance, 0) + count

    distances = sorted(weighted_sum)
    means = [weighted_sum[d] / count_sum[d] for d in distances]
    counts = [count_sum[d] for d in distances]
    return DistanceProfile(distances=distances, means=means, counts=counts)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def _profile_arrays(profile: DistanceProfile) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(profile.distances, dtype=np.float64)
    means = np.asarray(profile.means, dtype=np.float64)
    return distances, means


def _profile_error(
    profile: DistanceProfile | None,
    predictions: np.ndarray | None,
) -> tuple[float | None, float | None, int]:
    if profile is None or predictions is None:
        return None, None, 0
    _, observed = _profile_arrays(profile)
    residual = observed - predictions
    sse = float(np.sum(residual * residual))
    rmse = float(np.sqrt(sse / max(1, observed.shape[0])))
    return sse, rmse, int(observed.shape[0])


def _information_criteria(
    sse: float | None,
    n_points: int,
    param_count: int,
) -> tuple[float | None, float | None]:
    if sse is None or n_points <= 0:
        return None, None
    mse = max(sse / max(1, n_points), np.finfo(np.float64).tiny)
    aic = n_points * math.log(mse) + (2 * param_count)
    bic = n_points * math.log(mse) + (param_count * math.log(n_points))
    if n_points > param_count + 1:
        correction = (2 * param_count * (param_count + 1)) / (n_points - param_count - 1)
        aicc = aic + correction
    else:
        aicc = None
    return _finite_or_none(aicc), _finite_or_none(bic)


def _m0_predictions(
    profile: DistanceProfile | None,
    mean_value: float | None,
) -> np.ndarray | None:
    if profile is None or mean_value is None:
        return None
    return np.full(len(profile.distances), mean_value, dtype=np.float64)


def _m1_predictions(
    profile: DistanceProfile | None,
    params: dict[str, float] | None,
) -> np.ndarray | None:
    if profile is None or params is None:
        return None
    distances, _ = _profile_arrays(profile)
    return monotonic_decay(distances, params["a"], params["gamma"])


# ---------------------------------------------------------------------------
# AICc classification
# ---------------------------------------------------------------------------


def delta_penalty(n: int) -> float:
    """Analytic AICc penalty difference between M1 (k=2) and M0 (k=1).

    delta_penalty(n) = [2*2 + 2*2*3/(n-3)] - [2*1 + 2*1*2/(n-2)]
                     = 2 + 12/(n-3) - 4/(n-2)

    Derivation: Burnham & Anderson (2002), AICc correction formula.
    """
    if n <= 3:
        return float("inf")
    return 2.0 + 12.0 / (n - 3) - 4.0 / (n - 2)


def classify_head_aicc(
    aicc_m0: float | None,
    aicc_m1: float | None,
    n_points: int,
) -> dict[str, Any]:
    """Classify a head as M0-class or M1-class using AICc model selection.

    delta_aicc = AICc(M0) - AICc(M1)
    - delta_aicc > 0 -> M1-class (decay parameter justified)
    - delta_aicc <= 0 -> M0-class (constant sufficient)
    """
    if aicc_m0 is None or aicc_m1 is None:
        return {
            "head_classification": None,
            "delta_aicc_m0_minus_m1": None,
            "classification_clear": None,
            "analytic_penalty": None,
        }

    delta = aicc_m0 - aicc_m1
    penalty = delta_penalty(n_points)
    classification = "m1_class" if delta > 0.0 else "m0_class"
    clear = abs(delta) > penalty

    return {
        "head_classification": classification,
        "delta_aicc_m0_minus_m1": _finite_or_none(delta),
        "classification_clear": clear,
        "analytic_penalty": _finite_or_none(penalty),
    }


# ---------------------------------------------------------------------------
# Model fitting (M0 + M1 only)
# ---------------------------------------------------------------------------


def fit_profile_models(
    calibration_profile: DistanceProfile | None,
    holdout_profile: DistanceProfile | None,
) -> dict[str, dict[str, Any]]:
    """Fit M0/M1 on calibration and evaluate on holdout."""
    results: dict[str, dict[str, Any]] = {
        "m0": {
            "param_count": 1,
            "fit_ok": False,
            "params": {},
            "calibration": {"sse": None, "rmse": None, "n_points": 0, "aicc": None, "bic": None},
            "holdout": {"sse": None, "rmse": None, "n_points": 0},
        },
        "m1": {
            "param_count": 2,
            "fit_ok": False,
            "params": {},
            "calibration": {"sse": None, "rmse": None, "n_points": 0, "aicc": None, "bic": None},
            "holdout": {"sse": None, "rmse": None, "n_points": 0},
            "error": None,
        },
    }

    if calibration_profile is None or calibration_profile.n_points == 0:
        return results

    distances, values = _profile_arrays(calibration_profile)

    # M0: constant baseline
    mean_value = float(np.mean(values))
    cal_pred = _m0_predictions(calibration_profile, mean_value)
    cal_sse, cal_rmse, cal_n = _profile_error(calibration_profile, cal_pred)
    hold_pred = _m0_predictions(holdout_profile, mean_value)
    hold_sse, hold_rmse, hold_n = _profile_error(holdout_profile, hold_pred)
    aicc, bic = _information_criteria(cal_sse, cal_n, results["m0"]["param_count"])
    results["m0"]["fit_ok"] = True
    results["m0"]["params"] = {"mean": mean_value}
    results["m0"]["calibration"] = {
        "sse": _finite_or_none(cal_sse),
        "rmse": _finite_or_none(cal_rmse),
        "n_points": cal_n,
        "aicc": aicc,
        "bic": bic,
    }
    results["m0"]["holdout"] = {
        "sse": _finite_or_none(hold_sse),
        "rmse": _finite_or_none(hold_rmse),
        "n_points": hold_n,
    }

    # M1: monotone exponential decay
    try:
        initial_a = max(float(values[0]), np.finfo(np.float64).tiny)
        popt, _ = curve_fit(
            monotonic_decay,
            distances,
            values,
            p0=[initial_a, 0.1],
            bounds=([0.0, 0.0], [np.inf, 50.0]),
            maxfev=5000,
        )
        params = {"a": float(popt[0]), "gamma": float(popt[1])}
        cal_pred = _m1_predictions(calibration_profile, params)
        cal_sse, cal_rmse, cal_n = _profile_error(calibration_profile, cal_pred)
        hold_pred = _m1_predictions(holdout_profile, params)
        hold_sse, hold_rmse, hold_n = _profile_error(holdout_profile, hold_pred)
        aicc, bic = _information_criteria(cal_sse, cal_n, results["m1"]["param_count"])
        results["m1"]["fit_ok"] = True
        results["m1"]["params"] = params
        results["m1"]["calibration"] = {
            "sse": _finite_or_none(cal_sse),
            "rmse": _finite_or_none(cal_rmse),
            "n_points": cal_n,
            "aicc": aicc,
            "bic": bic,
        }
        results["m1"]["holdout"] = {
            "sse": _finite_or_none(hold_sse),
            "rmse": _finite_or_none(hold_rmse),
            "n_points": hold_n,
        }
    except (RuntimeError, ValueError) as exc:
        results["m1"]["error"] = str(exc)

    return results


def _best_model_by_holdout(fits: dict[str, dict[str, Any]]) -> str | None:
    candidates: list[tuple[str, float]] = []
    for model_name in ("m0", "m1"):
        rmse = fits[model_name]["holdout"]["rmse"]
        if rmse is None:
            continue
        candidates.append((model_name, float(rmse)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Per-head row construction
# ---------------------------------------------------------------------------


def _head_row_from_measurements(
    *,
    model_path: Path,
    prompt_measurements: list[dict[str, Any]],
    layer_idx: int,
    head_idx: int,
) -> dict[str, Any]:
    model_name = model_path.name
    family = _infer_model_family(model_name)
    precision_state = _infer_precision_state(model_name)

    calibration_profiles = [
        DistanceProfile(**measurement["profile"])
        for measurement in prompt_measurements
        if measurement["split"] == "calibration"
    ]
    holdout_profiles = [
        DistanceProfile(**measurement["profile"])
        for measurement in prompt_measurements
        if measurement["split"] == "holdout"
    ]
    calibration_profile = aggregate_profiles(calibration_profiles)
    holdout_profile = aggregate_profiles(holdout_profiles)

    if calibration_profile is None or holdout_profile is None:
        status = "insufficient_data"
        reason = "missing_calibration_profile" if calibration_profile is None else "missing_holdout_profile"
        return {
            "record_type": "head_classification",
            "protocol": PROTOCOL_ID,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_path": str(model_path),
            "model_name": model_name,
            "family": family,
            "precision_state": precision_state,
            "layer": layer_idx,
            "head": head_idx,
            "status": status,
            "reason": reason,
            "prompt_measurements": prompt_measurements,
            "calibration_profile": calibration_profile.to_dict() if calibration_profile else None,
            "holdout_profile": holdout_profile.to_dict() if holdout_profile else None,
        }

    fits = fit_profile_models(calibration_profile, holdout_profile)
    mean_prompt_distance_r2 = float(np.mean([m["distance_r2"] for m in prompt_measurements]))
    median_prompt_distance_r2 = float(np.median([m["distance_r2"] for m in prompt_measurements]))

    # AICc classification
    cal_n = fits["m0"]["calibration"]["n_points"]
    classification = classify_head_aicc(
        aicc_m0=fits["m0"]["calibration"]["aicc"],
        aicc_m1=fits["m1"]["calibration"]["aicc"],
        n_points=cal_n,
    )

    holdout_best = _best_model_by_holdout(fits)
    holdout_agrees = None
    if classification["head_classification"] is not None and holdout_best is not None:
        expected_best = "m1" if classification["head_classification"] == "m1_class" else "m0"
        holdout_agrees = holdout_best == expected_best

    return {
        "record_type": "head_classification",
        "protocol": PROTOCOL_ID,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_path": str(model_path),
        "model_name": model_name,
        "family": family,
        "precision_state": precision_state,
        "layer": layer_idx,
        "head": head_idx,
        "status": "ok",
        "prompt_measurements": prompt_measurements,
        "calibration_profile": calibration_profile.to_dict(),
        "holdout_profile": holdout_profile.to_dict(),
        "fits": fits,
        "head_summary": {
            "mean_prompt_distance_r2": mean_prompt_distance_r2,
            "median_prompt_distance_r2": median_prompt_distance_r2,
            "head_classification": classification["head_classification"],
            "delta_aicc_m0_minus_m1": classification["delta_aicc_m0_minus_m1"],
            "classification_clear": classification["classification_clear"],
            "analytic_penalty": classification["analytic_penalty"],
            "holdout_best_model": holdout_best,
            "holdout_agrees": holdout_agrees,
        },
    }


# ---------------------------------------------------------------------------
# Model analysis
# ---------------------------------------------------------------------------


def analyze_model(
    model_path: Path,
    probes: list[ProbeSpec],
    split_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Analyze one model path and return JSON-serializable record rows."""
    model_name = model_path.name
    family = _infer_model_family(model_name)
    precision_state = _infer_precision_state(model_name)

    if not model_path.exists():
        return [{
            "record_type": "model_skip",
            "protocol": PROTOCOL_ID,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_path": str(model_path),
            "model_name": model_name,
            "family": family,
            "precision_state": precision_state,
            "status": "skipped",
            "reason": "model_path_missing",
        }]

    try:
        backend = initialize_default_backend()
        loader = ModelLoader(backend)
        provider = ActivationProviderAdapter(backend=backend, model_path=str(model_path))
        model, tokenizer = loader.load_model(str(model_path))
    except Exception as exc:  # pragma: no cover - hardware-dependent
        return [{
            "record_type": "model_skip",
            "protocol": PROTOCOL_ID,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_path": str(model_path),
            "model_name": model_name,
            "family": family,
            "precision_state": precision_state,
            "status": "skipped",
            "reason": "model_load_failed",
            "error": str(exc),
        }]

    prompt_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}
    prompt_errors: list[dict[str, Any]] = []

    for probe in probes:
        try:
            token_ids = tokenizer.encode(probe.text)
            attn_matrices = provider.collect_attention_matrices(
                model,
                tokenizer,
                probe.text,
                token_ids=token_ids,
            )
        except Exception as exc:  # pragma: no cover - hardware-dependent
            prompt_errors.append({
                "prompt_id": probe.id,
                "prompt_family": probe.family,
                "split": split_map[probe.id],
                "status": "prompt_error",
                "error": str(exc),
            })
            continue

        if not attn_matrices:
            prompt_errors.append({
                "prompt_id": probe.id,
                "prompt_family": probe.family,
                "split": split_map[probe.id],
                "status": "prompt_error",
                "error": "no_attention_layers_returned",
            })
            continue

        for layer_idx, head_matrices in attn_matrices.items():
            for head_idx, attn_mat in enumerate(head_matrices):
                attn_np = np.asarray(attn_mat.tolist(), dtype=np.float64)
                profile, distance_r2 = compute_prompt_measurements(attn_np)
                prompt_rows.setdefault((layer_idx, head_idx), []).append({
                    "prompt_id": probe.id,
                    "prompt_family": probe.family,
                    "split": split_map[probe.id],
                    "token_count": len(token_ids),
                    "distance_r2": float(distance_r2),
                    "profile": profile.to_dict(),
                })

    if not prompt_rows:
        return [{
            "record_type": "model_skip",
            "protocol": PROTOCOL_ID,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_path": str(model_path),
            "model_name": model_name,
            "family": family,
            "precision_state": precision_state,
            "status": "skipped",
            "reason": "no_attention_measurements",
            "prompt_errors": prompt_errors,
        }]

    rows: list[dict[str, Any]] = []
    for (layer_idx, head_idx), measurements in sorted(prompt_rows.items()):
        row = _head_row_from_measurements(
            model_path=model_path,
            prompt_measurements=measurements,
            layer_idx=layer_idx,
            head_idx=head_idx,
        )
        if prompt_errors:
            row["prompt_errors"] = prompt_errors
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def build_model_family_summary(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    requested_models: list[Path],
) -> dict[str, Any]:
    """Summarize rows by model and family."""
    model_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        model_rows.setdefault(row["model_path"], []).append(row)

    model_summaries: list[dict[str, Any]] = []
    family_accumulator: dict[str, dict[str, list[float]]] = {}

    for requested_model in requested_models:
        model_key = str(requested_model.resolve())
        per_model_rows = model_rows.get(model_key, [])
        head_rows = _head_rows(per_model_rows)
        skip_rows = _model_skip_rows(per_model_rows)
        model_name = requested_model.name
        family = _infer_model_family(model_name)

        if not head_rows:
            reason = skip_rows[0].get("reason") if skip_rows else "no_rows"
            model_summaries.append({
                "model_path": model_key,
                "model_name": model_name,
                "family": family,
                "status": "skipped",
                "reason": reason,
                "n_head_rows": 0,
            })
            continue

        distance_r2_values = [
            row["head_summary"]["mean_prompt_distance_r2"]
            for row in head_rows
        ]
        classifications = [
            row["head_summary"]["head_classification"]
            for row in head_rows
            if row["head_summary"]["head_classification"] is not None
        ]
        m1_count = sum(1 for c in classifications if c == "m1_class")
        m0_count = sum(1 for c in classifications if c == "m0_class")
        total_classified = m1_count + m0_count
        m1_fraction = m1_count / total_classified if total_classified > 0 else None
        m0_fraction = m0_count / total_classified if total_classified > 0 else None

        # P-DKH-2: holdout superiority for M1-classified heads
        m1_heads = [
            row for row in head_rows
            if row["head_summary"]["head_classification"] == "m1_class"
        ]
        m1_holdout_m1_wins = sum(
            1 for row in m1_heads
            if row["head_summary"]["holdout_best_model"] == "m1"
        )
        m1_holdout_superiority = (
            m1_holdout_m1_wins / len(m1_heads) if m1_heads else None
        )

        # P-DKH-5: AICc-holdout concordance for clear classifications
        clear_heads = [
            row for row in head_rows
            if row["head_summary"]["classification_clear"] is True
        ]
        concordant = sum(
            1 for row in clear_heads
            if row["head_summary"]["holdout_agrees"] is True
        )
        aicc_holdout_concordance = (
            concordant / len(clear_heads) if clear_heads else None
        )

        holdout_best_counts: dict[str, int] = {}
        for row in head_rows:
            best = row["head_summary"]["holdout_best_model"]
            if best is not None:
                holdout_best_counts[best] = holdout_best_counts.get(best, 0) + 1

        # Accumulate family-level data
        fam_data = family_accumulator.setdefault(family, {
            "m1_fractions": [],
            "distance_r2_values": [],
        })
        if m1_fraction is not None:
            fam_data["m1_fractions"].append(m1_fraction)
        fam_data["distance_r2_values"].extend(distance_r2_values)

        model_summaries.append({
            "model_path": model_key,
            "model_name": model_name,
            "family": family,
            "status": "ok",
            "n_head_rows": len(head_rows),
            "mean_prompt_distance_r2": float(np.mean(distance_r2_values)),
            "median_prompt_distance_r2": float(np.median(distance_r2_values)),
            "m1_fraction": _finite_or_none(m1_fraction),
            "m0_fraction": _finite_or_none(m0_fraction),
            "m1_count": m1_count,
            "m0_count": m0_count,
            "total_classified": total_classified,
            "m1_holdout_superiority": _finite_or_none(m1_holdout_superiority),
            "aicc_holdout_concordance": _finite_or_none(aicc_holdout_concordance),
            "n_clear_heads": len(clear_heads),
            "holdout_best_model_counts": holdout_best_counts,
        })

    family_summaries: list[dict[str, Any]] = []
    for family in sorted({s["family"] for s in model_summaries}):
        family_models = [
            s for s in model_summaries
            if s["family"] == family and s["status"] == "ok"
        ]
        fam_data = family_accumulator.get(family, {
            "m1_fractions": [],
            "distance_r2_values": [],
        })
        mean_m1_fraction = (
            float(np.mean(fam_data["m1_fractions"]))
            if fam_data["m1_fractions"] else None
        )
        mean_distance_r2 = (
            float(np.mean(fam_data["distance_r2_values"]))
            if fam_data["distance_r2_values"] else None
        )

        family_summaries.append({
            "family": family,
            "n_models": len(family_models),
            "model_names": [s["model_name"] for s in family_models],
            "mean_m1_fraction": _finite_or_none(mean_m1_fraction),
            "mean_distance_r2": _finite_or_none(mean_distance_r2),
        })

    return {
        "run_id": run_id,
        "protocol": PROTOCOL_ID,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "models": model_summaries,
        "families": family_summaries,
    }


# ---------------------------------------------------------------------------
# Falsifier evaluation
# ---------------------------------------------------------------------------


def _evaluate_p_dkh_1(model_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """P-DKH-1: Hierarchy existence -- M1 fraction strictly between 0 and 1."""
    family_results: dict[str, dict[str, Any]] = {}
    for s in model_summaries:
        if s["status"] != "ok":
            continue
        fam = s["family"]
        frac = s.get("m1_fraction")
        if frac is None:
            family_results.setdefault(fam, {"verdict": "insufficient_data", "m1_fractions": []})
            continue
        family_results.setdefault(fam, {"verdict": None, "m1_fractions": []})
        family_results[fam]["m1_fractions"].append(frac)

    for fam, result in family_results.items():
        fracs = result["m1_fractions"]
        if not fracs:
            result["verdict"] = "insufficient_data"
            continue
        mean_frac = float(np.mean(fracs))
        if mean_frac == 0.0 or mean_frac == 1.0:
            result["verdict"] = "FAIL"
        else:
            result["verdict"] = "PASS"
        result["mean_m1_fraction"] = mean_frac

    all_verdicts = [r["verdict"] for r in family_results.values()]
    if any(v == "FAIL" for v in all_verdicts):
        overall = "FAIL"
    elif all(v == "PASS" for v in all_verdicts):
        overall = "PASS"
    else:
        overall = "insufficient_data"

    return {
        "prediction": "P-DKH-1",
        "description": "Hierarchy existence: M1 fraction strictly between 0 and 1 per family",
        "verdict": overall,
        "families": family_results,
    }


def _evaluate_p_dkh_2(model_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """P-DKH-2: M1 holdout superiority >50% for M1-classified heads."""
    family_results: dict[str, dict[str, Any]] = {}
    for s in model_summaries:
        if s["status"] != "ok":
            continue
        fam = s["family"]
        sup = s.get("m1_holdout_superiority")
        family_results.setdefault(fam, {"verdict": None, "superiority_values": []})
        if sup is not None:
            family_results[fam]["superiority_values"].append(sup)

    for fam, result in family_results.items():
        vals = result["superiority_values"]
        if not vals:
            result["verdict"] = "insufficient_data"
            continue
        mean_sup = float(np.mean(vals))
        result["mean_m1_holdout_superiority"] = mean_sup
        result["verdict"] = "PASS" if mean_sup > 0.5 else "FAIL"

    all_verdicts = [r["verdict"] for r in family_results.values()]
    if any(v == "FAIL" for v in all_verdicts):
        overall = "FAIL"
    elif all(v == "PASS" for v in all_verdicts):
        overall = "PASS"
    else:
        overall = "insufficient_data"

    return {
        "prediction": "P-DKH-2",
        "description": "M1 holdout superiority: M1 wins >50% of holdout for M1-classified heads",
        "verdict": overall,
        "families": family_results,
    }


def _evaluate_p_dkh_3(
    family_summaries: list[dict[str, Any]],
    model_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """P-DKH-3: Cross-family consistency -- within < between variance."""
    family_fractions: dict[str, list[float]] = {}
    for s in model_summaries:
        if s["status"] != "ok" or s.get("m1_fraction") is None:
            continue
        family_fractions.setdefault(s["family"], []).append(s["m1_fraction"])

    families_with_multiple = {f: v for f, v in family_fractions.items() if len(v) > 1}
    if not families_with_multiple:
        return {
            "prediction": "P-DKH-3",
            "description": "Cross-family consistency: within-family var < between-family var",
            "verdict": "non_adjudicating",
            "reason": "Single-model families -- requires multiple models per family.",
        }

    within_variances = []
    for fracs in families_with_multiple.values():
        within_variances.append(float(np.var(fracs)))
    mean_within = float(np.mean(within_variances))

    all_means = []
    for fracs in family_fractions.values():
        all_means.append(float(np.mean(fracs)))
    between_var = float(np.var(all_means)) if len(all_means) > 1 else 0.0

    verdict = "PASS" if mean_within < between_var else "FAIL"

    return {
        "prediction": "P-DKH-3",
        "description": "Cross-family consistency: within-family var < between-family var",
        "verdict": verdict,
        "mean_within_family_variance": mean_within,
        "between_family_variance": between_var,
    }


def _evaluate_p_dkh_4(model_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """P-DKH-4: Content residual dominance -- mean distance_r2 < 0.5 per family."""
    family_results: dict[str, dict[str, Any]] = {}
    for s in model_summaries:
        if s["status"] != "ok":
            continue
        fam = s["family"]
        r2 = s.get("mean_prompt_distance_r2")
        family_results.setdefault(fam, {"verdict": None, "r2_values": []})
        if r2 is not None:
            family_results[fam]["r2_values"].append(r2)

    for fam, result in family_results.items():
        vals = result["r2_values"]
        if not vals:
            result["verdict"] = "insufficient_data"
            continue
        mean_r2 = float(np.mean(vals))
        result["mean_distance_r2"] = mean_r2
        result["verdict"] = "PASS" if mean_r2 < 0.5 else "FAIL"

    all_verdicts = [r["verdict"] for r in family_results.values()]
    if any(v == "FAIL" for v in all_verdicts):
        overall = "FAIL"
    elif all(v == "PASS" for v in all_verdicts):
        overall = "PASS"
    else:
        overall = "insufficient_data"

    return {
        "prediction": "P-DKH-4",
        "description": "Content residual dominance: mean distance_r2 < 0.5 per family",
        "verdict": overall,
        "families": family_results,
    }


def _evaluate_p_dkh_5(model_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """P-DKH-5: AICc-holdout concordance >90% for clear classifications."""
    family_results: dict[str, dict[str, Any]] = {}
    for s in model_summaries:
        if s["status"] != "ok":
            continue
        fam = s["family"]
        conc = s.get("aicc_holdout_concordance")
        n_clear = s.get("n_clear_heads", 0)
        family_results.setdefault(fam, {"verdict": None, "concordance_values": [], "n_clear_total": 0})
        if conc is not None:
            family_results[fam]["concordance_values"].append(conc)
        family_results[fam]["n_clear_total"] += n_clear

    for fam, result in family_results.items():
        vals = result["concordance_values"]
        if not vals:
            result["verdict"] = "insufficient_data"
            continue
        mean_conc = float(np.mean(vals))
        result["mean_concordance"] = mean_conc
        result["verdict"] = "PASS" if mean_conc > 0.9 else "FAIL"

    all_verdicts = [r["verdict"] for r in family_results.values()]
    if any(v == "FAIL" for v in all_verdicts):
        overall = "FAIL"
    elif all(v == "PASS" for v in all_verdicts):
        overall = "PASS"
    else:
        overall = "insufficient_data"

    return {
        "prediction": "P-DKH-5",
        "description": "AICc-holdout concordance >90% for clear classifications",
        "verdict": overall,
        "families": family_results,
    }


def build_falsifier_outcome(
    *,
    run_id: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Derive protocol outcome from prediction evaluations."""
    p1 = _evaluate_p_dkh_1(summary["models"])
    p2 = _evaluate_p_dkh_2(summary["models"])
    p3 = _evaluate_p_dkh_3(summary["families"], summary["models"])
    p4 = _evaluate_p_dkh_4(summary["models"])
    p5 = _evaluate_p_dkh_5(summary["models"])

    predictions = [p1, p2, p3, p4, p5]
    verdicts = [p["verdict"] for p in predictions]

    promotable_verdicts = [v for v in verdicts if v not in ("non_adjudicating", "insufficient_data")]
    if not promotable_verdicts:
        overall = "insufficient_data"
        reason = "No predictions could be adjudicated."
    elif all(v == "PASS" for v in promotable_verdicts):
        overall = "all_predictions_pass"
        reason = "All adjudicating predictions passed."
    elif any(v == "FAIL" for v in promotable_verdicts):
        failed = [p["prediction"] for p in predictions if p["verdict"] == "FAIL"]
        overall = "partial_falsification"
        reason = f"Failed predictions: {', '.join(failed)}"
    else:
        overall = "mixed"
        reason = "Mixed verdicts across predictions."

    promotion_blocked = overall != "all_predictions_pass"

    return {
        "run_id": run_id,
        "protocol": PROTOCOL_ID,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "claim": CLAIM_DESCRIPTION,
        "claim_form": CLAIM_FORM,
        "observable": "AICc model selection (M0 vs M1) with holdout validation",
        "overall": overall,
        "promotion_blocked": promotion_blocked,
        "reason": reason,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def build_run_manifest(
    *,
    run_id: str,
    output_dir: Path,
    probe_file: Path,
    smoke: bool,
    collect_missing: bool,
    probes: list[ProbeSpec],
    split_map: dict[str, str],
    requested_models: list[Path],
    default_model_matrix_used: bool,
) -> dict[str, Any]:
    family_split_counts: dict[str, dict[str, int]] = {}
    for probe in probes:
        family_counts = family_split_counts.setdefault(
            probe.family,
            {"calibration": 0, "holdout": 0},
        )
        family_counts[split_map[probe.id]] += 1

    return {
        "run_id": run_id,
        "protocol": PROTOCOL_ID,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "output_dir": str(output_dir),
        "probe_file": str(probe_file),
        "smoke": smoke,
        "collect_missing": collect_missing,
        "default_model_matrix_used": default_model_matrix_used,
        "claim": CLAIM_DESCRIPTION,
        "claim_form": CLAIM_FORM,
        "aicc_reference": "Burnham & Anderson (2002), Model Selection and Multimodel Inference",
        "split_rule": {
            "hash": "sha256(first_8_bytes)",
            "assignment": (
                "Within each family, probes are sorted by stable hash of "
                "family:id:text. Even indices -> calibration, odd indices -> holdout. "
                "Single-probe families remain calibration-only."
            ),
        },
        "models_requested": [
            {
                "model_path": str(model.resolve()),
                "model_name": model.name,
                "family": _infer_model_family(model.name),
                "precision_state": _infer_precision_state(model.name),
                "exists": model.exists(),
            }
            for model in requested_models
        ],
        "probe_counts": {
            "total": len(probes),
            "by_family": family_split_counts,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_run_dir(output: Path | None, collect_missing: bool) -> Path:
    if output is not None:
        return output.expanduser().resolve()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / run_id
    if collect_missing:
        raise ValueError("--collect-missing requires --output to point at an existing or intended run directory.")
    return run_dir.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distance-kernel hierarchy falsifier protocol runner.",
    )
    parser.add_argument(
        "--models",
        action="append",
        default=None,
        help="Repeated path to a model directory. Defaults to the 3-family local matrix.",
    )
    parser.add_argument(
        "--probe-file",
        type=Path,
        default=DEFAULT_PROBE_FILE,
        help="Committed probe manifest JSON.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke-only probes from the manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Run directory to write artifacts. Defaults to results/distance_kernel_hierarchy/<run_id>/",
    )
    parser.add_argument(
        "--collect-missing",
        action="store_true",
        help="Reuse an existing run directory and analyze only models without successful head rows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    probe_file = args.probe_file.expanduser().resolve()
    probes_all = load_probe_manifest(probe_file)
    probes = select_probes(probes_all, smoke=args.smoke)
    split_map = assign_probe_splits(probes)
    requested_models, default_model_matrix_used = _resolve_model_paths(args.models)
    run_dir = _resolve_run_dir(args.output, args.collect_missing)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name

    existing_rows: list[dict[str, Any]] = []
    if args.collect_missing:
        existing_rows = _load_existing_jsonl(run_dir / "per_head_classification.jsonl")
        already_processed = _processed_model_paths(existing_rows)
        requested_models = [
            model for model in requested_models
            if str(model.resolve()) not in already_processed
        ]
        logger.info("Collect-missing mode: %d model(s) remain.", len(requested_models))

    logger.info("Distance-kernel hierarchy falsifier run_id=%s", run_id)
    logger.info("Probe file: %s", probe_file)
    logger.info("Selected probes: %d", len(probes))
    logger.info("Models to analyze this pass: %d", len(requested_models))

    new_rows: list[dict[str, Any]] = []
    for model_path in requested_models:
        logger.info("Analyzing model: %s", model_path)
        new_rows.extend(analyze_model(model_path, probes, split_map))

    all_rows = existing_rows + new_rows
    manifest = build_run_manifest(
        run_id=run_id,
        output_dir=run_dir,
        probe_file=probe_file,
        smoke=args.smoke,
        collect_missing=args.collect_missing,
        probes=probes,
        split_map=split_map,
        requested_models=_resolve_model_paths(args.models)[0] if args.models else list(DEFAULT_MODEL_PATHS),
        default_model_matrix_used=default_model_matrix_used,
    )
    summary = build_model_family_summary(
        run_id=run_id,
        rows=all_rows,
        requested_models=_resolve_model_paths(args.models)[0] if args.models else list(DEFAULT_MODEL_PATHS),
    )
    outcome = build_falsifier_outcome(run_id=run_id, summary=summary)

    _write_json(run_dir / "run_manifest.json", manifest)
    _write_jsonl(run_dir / "per_head_classification.jsonl", all_rows)
    _write_json(run_dir / "model_family_summary.json", summary)
    _write_json(run_dir / "falsifier_outcome.json", outcome)

    validation = artifact_validator.validate_run_dir(run_dir, include_self=False)
    validation["validated_at"] = _utc_now()
    _write_json(run_dir / "artifact_validation.json", validation)

    status = "PASS" if validation["ok"] else "FAIL"
    logger.info(
        "Artifact validation %s for %s (%d files checked)",
        status,
        run_dir,
        len(validation["files_checked"]),
    )
    for warning in validation["warnings"]:
        logger.warning("Validator warning: %s", warning)
    for error in validation["errors"]:
        logger.error("Validator error: %s", error)

    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
