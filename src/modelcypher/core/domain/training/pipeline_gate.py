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

"""Pipeline gate for strict promotability checks.

`pipeline_gate_v1` evaluates mission-level geometric invariants from measured
training outputs. Strict mode may fail closed for unresolved core checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

_PIPELINE_GATE_OPERATOR = "pipeline_gate_v1"


@dataclass(frozen=True)
class PipelineGateInput:
    """Inputs required to evaluate pipeline promotability."""

    spectral_bounds_ok: bool | None
    stop_reason: str | None
    per_layer_cka: Mapping[int | str, float] | None = None
    per_layer_cka_bound: Mapping[int | str, float] | None = None
    adapter_saturation_median_ratio: float | None = None
    max_effective_gain_ratio: float | None = None
    online_eval_stop_basis_degraded_significant: bool | None = None
    epoch_metrics: list[Mapping[str, Any]] | None = None
    strict_fail_closed_core: bool = False


@dataclass(frozen=True)
class PipelineGateCheck:
    """Single gate check verdict."""

    name: str
    status: str  # "pass" | "fail" | "unresolved"
    required: bool
    failure_mode: str | None = None
    message: str | None = None
    value: float | bool | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "required": self.required,
        }
        if self.failure_mode is not None:
            payload["failure_mode"] = self.failure_mode
        if self.message is not None:
            payload["message"] = self.message
        if self.value is not None:
            payload["value"] = self.value
        if self.details is not None:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True)
class PipelineGateVerdict:
    """Aggregate gate verdict."""

    operator: str
    passed: bool
    failure_modes: tuple[str, ...]
    unresolved_required: tuple[str, ...]
    checks: dict[str, PipelineGateCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "passed": self.passed,
            "failure_modes": list(self.failure_modes),
            "unresolved_required": list(self.unresolved_required),
            "checks": {
                name: check.to_dict()
                for name, check in self.checks.items()
            },
        }


def _as_layer_map(
    values: Mapping[int | str, float] | None,
) -> dict[int, float] | None:
    if values is None:
        return None
    layer_map: dict[int, float] = {}
    for raw_key, raw_val in values.items():
        try:
            layer_key = int(raw_key)
            layer_map[layer_key] = float(raw_val)
        except (TypeError, ValueError):
            continue
    return layer_map


def _check_spectral_bounds(input_data: PipelineGateInput) -> PipelineGateCheck:
    if input_data.spectral_bounds_ok is None:
        return PipelineGateCheck(
            name="spectral_bounds",
            status="unresolved",
            required=True,
            failure_mode="spectral_bounds_unavailable",
            message="spectral bound measurement unavailable",
        )
    if bool(input_data.spectral_bounds_ok):
        return PipelineGateCheck(
            name="spectral_bounds",
            status="pass",
            required=True,
            value=True,
        )
    return PipelineGateCheck(
        name="spectral_bounds",
        status="fail",
        required=True,
        failure_mode="spectral_bounds_violation",
        value=False,
        message="spectral bounds failed",
    )


def _check_safety_cap(input_data: PipelineGateInput) -> PipelineGateCheck:
    if input_data.stop_reason is None:
        return PipelineGateCheck(
            name="safety_cap_stop_reason",
            status="unresolved",
            required=False,
            message="stop_reason unavailable",
        )
    stop_reason = str(input_data.stop_reason)
    if stop_reason.startswith("safety_cap"):
        return PipelineGateCheck(
            name="safety_cap_stop_reason",
            status="fail",
            required=False,
            failure_mode="safety_cap_hit",
            value=True,
            message=f"stop_reason={stop_reason}",
        )
    return PipelineGateCheck(
        name="safety_cap_stop_reason",
        status="pass",
        required=False,
        value=False,
    )


def _check_cka_bound_bundle(
    input_data: PipelineGateInput,
    sqrt_eps: float,
) -> PipelineGateCheck:
    per_layer_cka = _as_layer_map(input_data.per_layer_cka)
    per_layer_bound = _as_layer_map(input_data.per_layer_cka_bound)
    if not per_layer_cka or not per_layer_bound:
        return PipelineGateCheck(
            name="cka_bound_bundle",
            status="unresolved",
            required=True,
            failure_mode="cka_bound_unavailable",
            message="cka-bound bundle unavailable",
            details={
                "has_per_layer_cka": bool(per_layer_cka),
                "has_per_layer_cka_bound": bool(per_layer_bound),
            },
        )
    margins = [
        float(actual) - float(per_layer_bound[layer_idx])
        for layer_idx, actual in per_layer_cka.items()
        if layer_idx in per_layer_bound
    ]
    if not margins:
        return PipelineGateCheck(
            name="cka_bound_bundle",
            status="unresolved",
            required=True,
            failure_mode="cka_bound_unavailable",
            message="cka-bound bundle has no overlapping layers",
            details={
                "n_cka_layers": len(per_layer_cka),
                "n_bound_layers": len(per_layer_bound),
            },
        )
    min_margin = min(margins)
    if min_margin >= -sqrt_eps:
        return PipelineGateCheck(
            name="cka_bound_bundle",
            status="pass",
            required=True,
            value=min_margin,
            details={
                "min_margin": min_margin,
                "sqrt_eps": sqrt_eps,
                "n_overlap_layers": len(margins),
            },
        )
    return PipelineGateCheck(
        name="cka_bound_bundle",
        status="fail",
        required=True,
        failure_mode="cka_bound_violation",
        value=min_margin,
        message="cka fell below perturbation-theory bound",
        details={
            "min_margin": min_margin,
            "sqrt_eps": sqrt_eps,
            "n_overlap_layers": len(margins),
        },
    )


def _check_adapter_saturation(input_data: PipelineGateInput) -> PipelineGateCheck:
    sat = input_data.adapter_saturation_median_ratio
    if sat is None:
        return PipelineGateCheck(
            name="adapter_saturation",
            status="unresolved",
            required=False,
            message="adapter saturation unavailable",
        )
    sat_val = float(sat)
    if sat_val < 1.0:
        return PipelineGateCheck(
            name="adapter_saturation",
            status="pass",
            required=False,
            value=sat_val,
        )
    return PipelineGateCheck(
        name="adapter_saturation",
        status="fail",
        required=False,
        failure_mode="adapter_saturation_exceeded",
        value=sat_val,
        message="adapter saturation ratio exceeded 1.0",
    )


def _check_gain_ratio(
    input_data: PipelineGateInput,
    sqrt_eps: float,
) -> PipelineGateCheck:
    gain_ratio = input_data.max_effective_gain_ratio
    if gain_ratio is None:
        return PipelineGateCheck(
            name="max_effective_gain_ratio",
            status="unresolved",
            required=False,
            message="gain ratio unavailable",
        )
    gain_val = float(gain_ratio)
    ceiling = 1.0 + sqrt_eps
    if gain_val <= ceiling:
        return PipelineGateCheck(
            name="max_effective_gain_ratio",
            status="pass",
            required=False,
            value=gain_val,
            details={"ceiling": ceiling},
        )
    return PipelineGateCheck(
        name="max_effective_gain_ratio",
        status="fail",
        required=False,
        failure_mode="gain_divergence",
        value=gain_val,
        message="max effective gain ratio exceeded numerical ceiling",
        details={"ceiling": ceiling},
    )


def _collect_online_eval_signals(input_data: PipelineGateInput) -> list[bool]:
    if input_data.online_eval_stop_basis_degraded_significant is not None:
        return [bool(input_data.online_eval_stop_basis_degraded_significant)]
    signals: list[bool] = []
    if input_data.epoch_metrics is None:
        return signals
    for metric in input_data.epoch_metrics:
        signal = metric.get("online_eval_stop_basis_degraded_significant")
        if signal is not None:
            signals.append(bool(signal))
    return signals


def _check_online_eval_stop_basis(input_data: PipelineGateInput) -> PipelineGateCheck:
    signals = _collect_online_eval_signals(input_data)
    if not signals:
        return PipelineGateCheck(
            name="online_eval_stop_basis",
            status="unresolved",
            required=False,
            message="online-eval stop-basis signal unavailable",
        )
    degraded = any(signals)
    if degraded:
        return PipelineGateCheck(
            name="online_eval_stop_basis",
            status="fail",
            required=False,
            failure_mode="online_eval_degraded_significant",
            value=True,
            message="online eval degradation marked significant",
            details={"n_signals": len(signals)},
        )
    return PipelineGateCheck(
        name="online_eval_stop_basis",
        status="pass",
        required=False,
        value=False,
        details={"n_signals": len(signals)},
    )


def evaluate_pipeline_gate(
    input_data: PipelineGateInput,
    eps: float,
) -> PipelineGateVerdict:
    """Evaluate mission-level promotability checks.

    Parameters
    ----------
    input_data:
        Raw measured outputs from the training pipeline.
    eps:
        IEEE-754 machine epsilon for the active precision.
    """
    sqrt_eps = math.sqrt(float(eps))

    checks: dict[str, PipelineGateCheck] = {
        "spectral_bounds": _check_spectral_bounds(input_data),
        "safety_cap_stop_reason": _check_safety_cap(input_data),
        "cka_bound_bundle": _check_cka_bound_bundle(input_data, sqrt_eps),
        "adapter_saturation": _check_adapter_saturation(input_data),
        "max_effective_gain_ratio": _check_gain_ratio(input_data, sqrt_eps),
        "online_eval_stop_basis": _check_online_eval_stop_basis(input_data),
    }

    failure_modes: list[str] = []
    unresolved_required: list[str] = []
    for check in checks.values():
        if check.status == "fail" and check.failure_mode is not None:
            failure_modes.append(check.failure_mode)
        if (
            check.status == "unresolved"
            and check.required
            and input_data.strict_fail_closed_core
        ):
            unresolved_required.append(check.name)
            if check.failure_mode is not None:
                failure_modes.append(check.failure_mode)

    # De-duplicate while preserving order.
    dedup_failure_modes = tuple(dict.fromkeys(failure_modes))
    return PipelineGateVerdict(
        operator=_PIPELINE_GATE_OPERATOR,
        passed=len(dedup_failure_modes) == 0,
        failure_modes=dedup_failure_modes,
        unresolved_required=tuple(unresolved_required),
        checks=checks,
    )


__all__ = [
    "PipelineGateCheck",
    "PipelineGateInput",
    "PipelineGateVerdict",
    "evaluate_pipeline_gate",
]
