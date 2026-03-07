#!/usr/bin/env python3
"""Wave-kernel falsifier protocol runner.

This script measures one narrow claim:

    Does a damped oscillation kernel explain attention better than a
    monotone decay kernel once we control for prompt split, fit
    degeneracy, and architecture family?

The output is a run directory with machine-readable artifacts only.
Interpretation and promotion live in the protocol document.

Usage:
    poetry run python scripts/wave_field_analysis.py --smoke
    poetry run python scripts/wave_field_analysis.py --models /path/to/model
    poetry run python scripts/wave_field_analysis.py \
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
    from scripts import validate_wave_kernel_falsifier_artifacts as artifact_validator
except ImportError:  # pragma: no cover - direct script execution path
    import validate_wave_kernel_falsifier_artifacts as artifact_validator

from modelcypher.adapters.activation_provider import ActivationProviderAdapter
from modelcypher.adapters.model_loader import ModelLoader
from modelcypher.backends import initialize_default_backend

logger = logging.getLogger(__name__)

PROTOCOL_ID = "F-WAVE-01"
ARTIFACT_SCHEMA_VERSION = "v1"
CLAIM_FORM = (
    "observable = f(geometry_state, architecture_state, scale_state, "
    "precision_state, measurement_operator)"
)
CLAIM_DESCRIPTION = (
    "A damped oscillation kernel improves holdout fit over monotone decay on "
    "attention distance profiles once boundary-equivalent M2 fits are excluded."
)
RESULTS_ROOT = Path("results/wave_kernel_falsifier")
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


def monotonic_decay(d: np.ndarray, a: float, gamma: float) -> np.ndarray:
    """M1: Pure exponential decay."""
    return a * np.exp(-gamma * d)


def damped_oscillation(
    d: np.ndarray,
    a: float,
    gamma: float,
    omega: float,
    phi: float,
) -> np.ndarray:
    """M2: Damped oscillation."""
    return a * np.exp(-gamma * d) * np.cos(omega * d + phi)


def compute_prompt_measurements(attn_matrix: np.ndarray) -> tuple[DistanceProfile, float]:
    """Return the per-distance profile and nonparametric distance R²."""
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


def _m2_boundary_equivalent(
    distances: np.ndarray,
    params: dict[str, float],
) -> bool:
    if distances.shape[0] < 3:
        return True

    predicted = damped_oscillation(
        distances,
        params["a"],
        params["gamma"],
        params["omega"],
        params["phi"],
    )

    zero_crossing = any(
        predicted[idx] == 0.0 or predicted[idx] * predicted[idx + 1] < 0.0
        for idx in range(predicted.shape[0] - 1)
    )
    diffs = np.diff(predicted)
    interior_extremum = any(
        (diffs[idx - 1] > 0.0 and diffs[idx] < 0.0)
        or (diffs[idx - 1] < 0.0 and diffs[idx] > 0.0)
        for idx in range(1, diffs.shape[0])
    )
    return not (zero_crossing or interior_extremum)


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


def _m2_predictions(
    profile: DistanceProfile | None,
    params: dict[str, float] | None,
) -> np.ndarray | None:
    if profile is None or params is None:
        return None
    distances, _ = _profile_arrays(profile)
    return damped_oscillation(
        distances,
        params["a"],
        params["gamma"],
        params["omega"],
        params["phi"],
    )


def fit_profile_models(
    calibration_profile: DistanceProfile | None,
    holdout_profile: DistanceProfile | None,
) -> dict[str, dict[str, Any]]:
    """Fit M0/M1/M2 on calibration and evaluate on holdout."""
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
        "m2": {
            "param_count": 4,
            "fit_ok": False,
            "params": {},
            "boundary_equivalent": None,
            "calibration": {"sse": None, "rmse": None, "n_points": 0, "aicc": None, "bic": None},
            "holdout": {"sse": None, "rmse": None, "n_points": 0},
            "error": None,
        },
    }

    if calibration_profile is None or calibration_profile.n_points == 0:
        return results

    distances, values = _profile_arrays(calibration_profile)

    # M0
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

    # M1
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

    # M2
    try:
        if results["m1"]["params"]:
            m1_params = results["m1"]["params"]
            initial_a = max(float(m1_params["a"]), np.finfo(np.float64).tiny)
            initial_gamma = float(m1_params["gamma"])
        else:
            initial_a = max(float(values[0]), np.finfo(np.float64).tiny)
            initial_gamma = 0.1

        popt, _ = curve_fit(
            damped_oscillation,
            distances,
            values,
            p0=[initial_a, initial_gamma, 0.5, 0.0],
            bounds=([0.0, 0.0, 0.0, -np.pi], [np.inf, 50.0, 50.0, np.pi]),
            maxfev=10000,
        )
        params = {
            "a": float(popt[0]),
            "gamma": float(popt[1]),
            "omega": float(popt[2]),
            "phi": float(popt[3]),
        }
        boundary_equivalent = _m2_boundary_equivalent(distances, params)
        cal_pred = _m2_predictions(calibration_profile, params)
        cal_sse, cal_rmse, cal_n = _profile_error(calibration_profile, cal_pred)
        hold_pred = _m2_predictions(holdout_profile, params)
        hold_sse, hold_rmse, hold_n = _profile_error(holdout_profile, hold_pred)
        aicc, bic = _information_criteria(cal_sse, cal_n, results["m2"]["param_count"])
        results["m2"]["fit_ok"] = True
        results["m2"]["params"] = params
        results["m2"]["boundary_equivalent"] = boundary_equivalent
        results["m2"]["calibration"] = {
            "sse": _finite_or_none(cal_sse),
            "rmse": _finite_or_none(cal_rmse),
            "n_points": cal_n,
            "aicc": aicc,
            "bic": bic,
        }
        results["m2"]["holdout"] = {
            "sse": _finite_or_none(hold_sse),
            "rmse": _finite_or_none(hold_rmse),
            "n_points": hold_n,
        }
    except (RuntimeError, ValueError) as exc:
        results["m2"]["error"] = str(exc)

    return results


def _best_model_by_holdout(fits: dict[str, dict[str, Any]]) -> str | None:
    candidates: list[tuple[str, float]] = []
    for model_name in ("m0", "m1", "m2"):
        rmse = fits[model_name]["holdout"]["rmse"]
        if rmse is None:
            continue
        candidates.append((model_name, float(rmse)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[0][0]


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
            "record_type": "head_fit",
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
    holdout_delta = None
    if fits["m1"]["holdout"]["rmse"] is not None and fits["m2"]["holdout"]["rmse"] is not None:
        holdout_delta = float(fits["m2"]["holdout"]["rmse"] - fits["m1"]["holdout"]["rmse"])

    m2_boundary_equivalent = fits["m2"].get("boundary_equivalent")
    wave_support = (
        holdout_delta is not None
        and holdout_delta < 0.0
        and m2_boundary_equivalent is False
    )

    return {
        "record_type": "head_fit",
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
            "holdout_rmse_delta_m2_minus_m1": _finite_or_none(holdout_delta),
            "holdout_best_model": _best_model_by_holdout(fits),
            "wave_support_on_holdout": wave_support,
            "boundary_equivalent": m2_boundary_equivalent,
        },
    }


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
        if row.get("record_type") == "head_fit" and row.get("status") == "ok":
            processed.add(str(Path(row["model_path"]).resolve()))
    return processed


def _head_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("record_type") == "head_fit" and row.get("status") == "ok"
    ]


def _model_skip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("record_type") == "model_skip"
    ]


def _direction_from_deltas(deltas: list[float]) -> str:
    if not deltas:
        return "insufficient_data"
    mean_delta = float(np.mean(deltas))
    median_delta = float(np.median(deltas))
    if mean_delta < 0.0 and median_delta < 0.0:
        return "wave_favored"
    if mean_delta > 0.0 and median_delta > 0.0:
        return "decay_favored"
    return "mixed"


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
    family_accumulator: dict[str, list[float]] = {}

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

        distance_r2 = [
            row["head_summary"]["mean_prompt_distance_r2"]
            for row in head_rows
        ]
        holdout_deltas = [
            row["head_summary"]["holdout_rmse_delta_m2_minus_m1"]
            for row in head_rows
            if row["head_summary"]["holdout_rmse_delta_m2_minus_m1"] is not None
        ]
        nonboundary_deltas = [
            row["head_summary"]["holdout_rmse_delta_m2_minus_m1"]
            for row in head_rows
            if row["head_summary"]["holdout_rmse_delta_m2_minus_m1"] is not None
            and row["head_summary"]["boundary_equivalent"] is False
        ]
        boundary_flags = [
            row["head_summary"]["boundary_equivalent"]
            for row in head_rows
            if row["head_summary"]["boundary_equivalent"] is not None
        ]
        wave_support_count = sum(
            1 for row in head_rows if row["head_summary"]["wave_support_on_holdout"]
        )
        best_model_counts: dict[str, int] = {}
        for row in head_rows:
            best_model = row["head_summary"]["holdout_best_model"]
            if best_model is None:
                continue
            best_model_counts[best_model] = best_model_counts.get(best_model, 0) + 1

        direction = _direction_from_deltas(nonboundary_deltas)
        family_accumulator.setdefault(family, []).extend(nonboundary_deltas)

        model_summaries.append({
            "model_path": model_key,
            "model_name": model_name,
            "family": family,
            "status": "ok",
            "n_head_rows": len(head_rows),
            "mean_prompt_distance_r2": float(np.mean(distance_r2)),
            "median_prompt_distance_r2": float(np.median(distance_r2)),
            "mean_holdout_delta_m2_minus_m1": _finite_or_none(float(np.mean(holdout_deltas))) if holdout_deltas else None,
            "median_holdout_delta_m2_minus_m1": _finite_or_none(float(np.median(holdout_deltas))) if holdout_deltas else None,
            "nonboundary_head_count": len(nonboundary_deltas),
            "boundary_equivalent_head_fraction": (
                float(sum(bool(flag) for flag in boundary_flags) / len(boundary_flags))
                if boundary_flags else None
            ),
            "wave_support_head_fraction": wave_support_count / len(head_rows),
            "holdout_best_model_counts": best_model_counts,
            "direction": direction,
        })

    family_summaries: list[dict[str, Any]] = []
    for family in sorted({summary["family"] for summary in model_summaries}):
        family_models = [
            summary for summary in model_summaries
            if summary["family"] == family and summary["status"] == "ok"
        ]
        family_deltas = family_accumulator.get(family, [])
        family_summaries.append({
            "family": family,
            "n_models": len(family_models),
            "model_names": [summary["model_name"] for summary in family_models],
            "nonboundary_head_count": len(family_deltas),
            "mean_holdout_delta_m2_minus_m1": _finite_or_none(float(np.mean(family_deltas))) if family_deltas else None,
            "median_holdout_delta_m2_minus_m1": _finite_or_none(float(np.median(family_deltas))) if family_deltas else None,
            "direction": _direction_from_deltas(family_deltas),
        })

    return {
        "run_id": run_id,
        "protocol": PROTOCOL_ID,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "models": model_summaries,
        "families": family_summaries,
    }


def build_falsifier_outcome(
    *,
    run_id: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Derive protocol outcome from family-level directions."""
    family_directions = [
        family["direction"]
        for family in summary["families"]
        if family["direction"] != "insufficient_data"
    ]

    if not family_directions:
        overall = "insufficient_data"
        reason = "No family produced non-boundary holdout deltas."
    elif all(direction == "wave_favored" for direction in family_directions):
        overall = "consistent_with_wave_claim"
        reason = "All families favored M2 on non-boundary holdout heads."
    elif all(direction == "decay_favored" for direction in family_directions):
        overall = "falsified_by_decay"
        reason = "All families favored M1 on non-boundary holdout heads."
    else:
        overall = "architecture_conditioned_mixed"
        reason = "Families disagreed in direction; promotion is blocked."

    return {
        "run_id": run_id,
        "protocol": PROTOCOL_ID,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "claim": CLAIM_DESCRIPTION,
        "claim_form": CLAIM_FORM,
        "observable": "holdout_rmse_delta_m2_minus_m1 on non-boundary heads",
        "overall": overall,
        "promotion_blocked": overall != "consistent_with_wave_claim",
        "reason": reason,
        "family_outcomes": summary["families"],
        "model_outcomes": summary["models"],
    }


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
        description="Wave-kernel falsifier protocol runner.",
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
        help="Run directory to write artifacts. Defaults to results/wave_kernel_falsifier/<run_id>/",
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
        existing_rows = _load_existing_jsonl(run_dir / "per_head_fit_table.jsonl")
        already_processed = _processed_model_paths(existing_rows)
        requested_models = [
            model for model in requested_models
            if str(model.resolve()) not in already_processed
        ]
        logger.info("Collect-missing mode: %d model(s) remain.", len(requested_models))

    logger.info("Wave-kernel falsifier run_id=%s", run_id)
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
    _write_jsonl(run_dir / "per_head_fit_table.jsonl", all_rows)
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
