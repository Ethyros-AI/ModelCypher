# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""LoRA geometry baseline artifact loading and calibration utilities."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BASELINE_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[4] / "results" / "real_adapter_analysis" / "summary.json"
)
DEFAULT_FOUR_CONDITION_RESULTS_PATH = (
    Path(__file__).resolve().parents[4] / "results" / "four_condition" / "raw_measurements.json"
)


def resolve_baseline_artifact_path(baseline_artifact: str | None) -> Path | None:
    """Resolve explicit or default baseline artifact path."""
    if baseline_artifact is not None:
        artifact_path = Path(baseline_artifact).expanduser().resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Baseline artifact not found: {artifact_path}")
        return artifact_path

    if DEFAULT_BASELINE_ARTIFACT_PATH.exists():
        return DEFAULT_BASELINE_ARTIFACT_PATH
    return None


def load_reference_baseline(
    baseline_artifact: str | None,
) -> dict[str, Any] | None:
    """Load measured random baseline from an artifact, if available."""
    artifact_path = resolve_baseline_artifact_path(baseline_artifact)
    if artifact_path is None:
        return None

    artifact = _read_json_object(artifact_path)

    findings = artifact.get("findings")
    if not isinstance(findings, dict):
        raise ValueError(f"Invalid baseline artifact (missing findings): {artifact_path}")

    random_baseline = findings.get("synthetic_random_baseline")
    if not isinstance(random_baseline, dict):
        raise ValueError(
            f"Invalid baseline artifact (missing synthetic_random_baseline): {artifact_path}"
        )

    amplification_cv = _validated_positive_scalar(
        random_baseline.get("amplification_cv"),
        field_name="findings.synthetic_random_baseline.amplification_cv",
        source_path=artifact_path,
    )
    weyl_utilization = _validated_positive_scalar(
        random_baseline.get("weyl_utilization"),
        field_name="findings.synthetic_random_baseline.weyl_utilization",
        source_path=artifact_path,
    )

    result: dict[str, Any] = {
        "type": "synthetic_random_baseline",
        "amplification_cv": amplification_cv,
        "weyl_utilization": weyl_utilization,
        "source": random_baseline.get("source", "unknown"),
        "artifact_path": str(artifact_path),
    }
    if "sample_count" in random_baseline:
        result["sample_count"] = int(random_baseline["sample_count"])
    if "amplification_cv_range" in random_baseline:
        result["amplification_cv_range"] = random_baseline["amplification_cv_range"]
    if "weyl_utilization_range" in random_baseline:
        result["weyl_utilization_range"] = random_baseline["weyl_utilization_range"]
    if "experiment_date" in artifact:
        result["experiment_date"] = artifact["experiment_date"]
    return result


def calibrate_reference_baseline(
    four_condition_results: str | None,
    output_artifact: str | None,
    source_label: str | None = None,
) -> dict[str, Any]:
    """Calibrate synthetic-random baseline from four-condition measurements.

    Args:
        four_condition_results: Path to four-condition raw measurements JSON.
            Defaults to ``results/four_condition/raw_measurements.json``.
        output_artifact: Path to output artifact summary JSON.
            Defaults to ``results/real_adapter_analysis/summary.json``.
        source_label: Optional explicit source label for artifact provenance.

    Returns:
        Calibration payload including output path and calibrated baseline.
    """
    four_condition_path = (
        Path(four_condition_results).expanduser().resolve()
        if four_condition_results is not None
        else DEFAULT_FOUR_CONDITION_RESULTS_PATH
    )
    output_path = (
        Path(output_artifact).expanduser().resolve()
        if output_artifact is not None
        else DEFAULT_BASELINE_ARTIFACT_PATH
    )

    if not four_condition_path.exists():
        raise FileNotFoundError(f"Four-condition results not found: {four_condition_path}")

    four_condition_data = _read_json_object(four_condition_path)
    synthetic_baseline = _derive_synthetic_random_baseline(
        four_condition_data,
        source_path=four_condition_path,
        source_label=source_label,
    )

    artifact: dict[str, Any] = {}
    if output_path.exists():
        artifact = _read_json_object(output_path)

    findings = artifact.get("findings")
    if findings is None:
        findings = {}
    if not isinstance(findings, dict):
        raise ValueError(f"Invalid artifact (findings must be an object): {output_path}")

    findings["synthetic_random_baseline"] = synthetic_baseline
    artifact["findings"] = findings
    artifact["experiment_date"] = datetime.now(timezone.utc).date().isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    baseline = load_reference_baseline(str(output_path))
    if baseline is None:
        raise ValueError(f"Failed to load calibrated baseline from artifact: {output_path}")

    return {
        "output_artifact": str(output_path),
        "source_four_condition_results": str(four_condition_path),
        "reference_baseline": baseline,
    }


def _derive_synthetic_random_baseline(
    four_condition_data: dict[str, Any],
    source_path: Path,
    source_label: str | None,
) -> dict[str, Any]:
    """Compute synthetic random baseline from four-condition raw measurements."""
    conditions = four_condition_data.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError(
            f"Invalid four-condition results (missing conditions object): {source_path}"
        )

    pure_random = conditions.get("pure_random")
    if not isinstance(pure_random, list) or len(pure_random) == 0:
        raise ValueError(
            f"Invalid four-condition results (missing non-empty conditions.pure_random): {source_path}"
        )

    amplification_values: list[float] = []
    weyl_values: list[float] = []
    for idx, measurement in enumerate(pure_random):
        if not isinstance(measurement, dict):
            raise ValueError(
                f"Invalid four-condition results (conditions.pure_random[{idx}] is not an object): {source_path}"
            )
        amplification_values.append(
            _validated_positive_scalar(
                measurement.get("mean_amplification_cv"),
                field_name=f"conditions.pure_random[{idx}].mean_amplification_cv",
                source_path=source_path,
            )
        )
        weyl_values.append(
            _validated_positive_scalar(
                measurement.get("mean_weyl_utilization"),
                field_name=f"conditions.pure_random[{idx}].mean_weyl_utilization",
                source_path=source_path,
            )
        )

    sample_count = len(amplification_values)
    return {
        "amplification_cv": float(sum(amplification_values) / sample_count),
        "weyl_utilization": float(sum(weyl_values) / sample_count),
        "amplification_cv_range": [
            float(min(amplification_values)),
            float(max(amplification_values)),
        ],
        "weyl_utilization_range": [
            float(min(weyl_values)),
            float(max(weyl_values)),
        ],
        "sample_count": sample_count,
        "source": source_label or f"Calibrated from {source_path}",
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _validated_positive_scalar(value: Any, field_name: str, source_path: Path) -> float:
    """Parse and validate a positive finite scalar."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid scalar for {field_name}: {source_path}") from e
    if not math.isfinite(scalar):
        raise ValueError(f"Invalid scalar ({field_name} is not finite): {source_path}")
    if scalar <= 0.0:
        raise ValueError(f"Invalid scalar ({field_name} must be > 0): {source_path}")
    return scalar
