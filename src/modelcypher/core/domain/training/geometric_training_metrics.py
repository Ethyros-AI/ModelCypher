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

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from enum import Enum


class GeometricInstrumentationLevel(str, Enum):
    minimal = "minimal"
    moderate = "moderate"
    full = "full"
    research = "research"

    @property
    def description(self) -> str:
        if self is GeometricInstrumentationLevel.minimal:
            return "Minimal - gradient norms only"
        if self is GeometricInstrumentationLevel.moderate:
            return "Moderate - adds curvature estimation"
        if self is GeometricInstrumentationLevel.full:
            return "Full - all metrics every step"
        return "Research - includes loss landscape sampling"

    @property
    def hessian_computation_interval(self) -> int:
        if self is GeometricInstrumentationLevel.minimal:
            return sys.maxsize
        if self is GeometricInstrumentationLevel.moderate:
            return 10
        return 1

    @property
    def compute_per_layer_metrics(self) -> bool:
        return self in (GeometricInstrumentationLevel.full, GeometricInstrumentationLevel.research)

    @property
    def compute_loss_landscape(self) -> bool:
        return self is GeometricInstrumentationLevel.research

    @property
    def compute_top_eigenvalue(self) -> bool:
        return self is not GeometricInstrumentationLevel.minimal

    @property
    def metrics_collected(self) -> list[str]:
        if self is GeometricInstrumentationLevel.minimal:
            return ["Gradient norms", "Parameter divergence"]
        if self is GeometricInstrumentationLevel.moderate:
            return ["Gradient norms", "Parameter divergence", "Curvature estimation (Hessian trace)"]
        if self is GeometricInstrumentationLevel.full:
            return [
                "Gradient norms",
                "Parameter divergence",
                "Curvature estimation",
                "Per-layer statistics",
                "Effective step ratio",
            ]
        return ["All metrics", "Loss landscape sampling", "Trajectory projections"]


class GeometryMetricKey:
    hessian_trace = "geometry/hessian_trace"
    top_eigenvalue = "geometry/top_eigenvalue"
    condition_proxy = "geometry/condition_proxy"
    flatness_score = "geometry/flatness_score"
    gradient_variance = "geometry/gradient_variance"
    gradient_snr = "geometry/gradient_snr"
    effective_step_ratio = "geometry/effective_step_ratio"
    param_divergence = "geometry/param_divergence"
    param_cosine_similarity = "geometry/param_cosine_similarity"
    refusal_distance = "geometry/refusal_distance"
    refusal_projection = "geometry/refusal_projection"
    refusal_approaching = "geometry/refusal_approaching"
    refusal_strength = "geometry/refusal_strength"
    dare_effective_sparsity = "geometry/dare_effective_sparsity"
    dare_essential_fraction = "geometry/dare_essential_fraction"
    dare_recommended_drop_rate = "geometry/dare_recommended_drop_rate"
    dora_magnitude_change = "geometry/dora_magnitude_change"
    dora_directional_drift = "geometry/dora_directional_drift"
    dora_magnitude_to_direction_ratio = "geometry/dora_mag_dir_ratio"
    circuit_breaker_tripped = "geometry/circuit_breaker_tripped"
    circuit_breaker_confidence = "geometry/circuit_breaker_confidence"
    circuit_breaker_severity = "geometry/circuit_breaker_severity"
    persona_overall_drift = "geometry/persona/overall_drift"

    @staticmethod
    def layer_grad_norm(layer_name: str) -> str:
        return f"geometry/layer/{layer_name}/grad_norm"

    @staticmethod
    def layer_grad_fraction(layer_name: str) -> str:
        return f"geometry/layer/{layer_name}/grad_frac"

    @staticmethod
    def persona_position(vector_id: str) -> str:
        return f"geometry/persona/{vector_id}/position"

    @staticmethod
    def persona_delta(vector_id: str) -> str:
        return f"geometry/persona/{vector_id}/delta"


@dataclass(frozen=True)
class GeometricTrainingMetrics:
    hessian_trace_estimate: float | None = None
    top_hessian_eigenvalue: float | None = None
    hessian_condition_proxy: float | None = None
    gradient_variance: float | None = None
    gradient_snr: float | None = None
    effective_step_ratio: float | None = None
    per_layer_gradient_norms: dict[str, float] = field(default_factory=dict)
    per_layer_gradient_fractions: dict[str, float] = field(default_factory=dict)
    active_layers: list[str] = field(default_factory=list)
    parameter_divergence: float | None = None
    parameter_cosine_similarity: float | None = None
    refusal_distance: float | None = None
    is_approaching_refusal: bool | None = None
    dare_effective_sparsity: float | None = None
    dora_magnitude_change: float | None = None
    dora_directional_drift: float | None = None
    persona_drift_magnitude: float | None = None
    drifting_traits: list[str] = field(default_factory=list)
    circuit_breaker_severity: float | None = None
    circuit_breaker_tripped: bool | None = None

    @property
    def flatness_score(self) -> float | None:
        if self.top_hessian_eigenvalue is None or self.top_hessian_eigenvalue <= 0:
            return None
        log_eigen = math.log10(self.top_hessian_eigenvalue + 0.001)
        normalized = 1 - (log_eigen + 1) / 3
        return max(0.0, min(1.0, normalized))

    @property
    def flatness_assessment(self) -> str:
        score = self.flatness_score
        if score is None:
            return "Unknown"
        if score > 0.7:
            return "Flat (good)"
        if score > 0.4:
            return "Moderate"
        return "Sharp (risk)"

    @property
    def snr_assessment(self) -> str:
        if self.gradient_snr is None:
            return "Unknown"
        if self.gradient_snr > 10:
            return "Strong signal"
        if self.gradient_snr > 1:
            return "Adequate"
        return "Noisy"

    def to_metrics_dict(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.hessian_trace_estimate is not None:
            metrics[GeometryMetricKey.hessian_trace] = float(self.hessian_trace_estimate)
        if self.top_hessian_eigenvalue is not None:
            metrics[GeometryMetricKey.top_eigenvalue] = float(self.top_hessian_eigenvalue)
        if self.hessian_condition_proxy is not None:
            metrics[GeometryMetricKey.condition_proxy] = float(self.hessian_condition_proxy)
        if self.gradient_variance is not None:
            metrics[GeometryMetricKey.gradient_variance] = float(self.gradient_variance)
        if self.gradient_snr is not None:
            metrics[GeometryMetricKey.gradient_snr] = float(self.gradient_snr)
        if self.effective_step_ratio is not None:
            metrics[GeometryMetricKey.effective_step_ratio] = float(self.effective_step_ratio)
        if self.parameter_divergence is not None:
            metrics[GeometryMetricKey.param_divergence] = float(self.parameter_divergence)
        if self.parameter_cosine_similarity is not None:
            metrics[GeometryMetricKey.param_cosine_similarity] = float(self.parameter_cosine_similarity)
        if self.flatness_score is not None:
            metrics[GeometryMetricKey.flatness_score] = float(self.flatness_score)

        for layer, norm in self.per_layer_gradient_norms.items():
            metrics[GeometryMetricKey.layer_grad_norm(layer)] = float(norm)
        for layer, fraction in self.per_layer_gradient_fractions.items():
            metrics[GeometryMetricKey.layer_grad_fraction(layer)] = float(fraction)

        if self.refusal_distance is not None:
            metrics[GeometryMetricKey.refusal_distance] = float(self.refusal_distance)
        if self.is_approaching_refusal is not None:
            metrics[GeometryMetricKey.refusal_approaching] = 1.0 if self.is_approaching_refusal else 0.0
        if self.dare_effective_sparsity is not None:
            metrics[GeometryMetricKey.dare_effective_sparsity] = float(self.dare_effective_sparsity)
        if self.dora_magnitude_change is not None:
            metrics[GeometryMetricKey.dora_magnitude_change] = float(self.dora_magnitude_change)
        if self.dora_directional_drift is not None:
            metrics[GeometryMetricKey.dora_directional_drift] = float(self.dora_directional_drift)
        if self.persona_drift_magnitude is not None:
            metrics[GeometryMetricKey.persona_overall_drift] = float(self.persona_drift_magnitude)
        if self.circuit_breaker_severity is not None:
            metrics[GeometryMetricKey.circuit_breaker_severity] = float(self.circuit_breaker_severity)
        if self.circuit_breaker_tripped is not None:
            metrics[GeometryMetricKey.circuit_breaker_tripped] = 1.0 if self.circuit_breaker_tripped else 0.0

        return metrics

    @classmethod
    def from_progress_metrics(cls, metrics: dict[str, float]) -> "GeometricTrainingMetrics" | None:
        if not metrics:
            return None
        has_geometry = any(key.startswith("geometry/") for key in metrics)
        if not has_geometry:
            return None

        layer_norms: dict[str, float] = {}
        layer_fractions: dict[str, float] = {}
        active_layers: list[str] = []

        layer_prefix = "geometry/layer/"
        grad_norm_suffix = "/grad_norm"
        grad_frac_suffix = "/grad_frac"

        for key, value in metrics.items():
            if not key.startswith(layer_prefix):
                continue
            if key.endswith(grad_norm_suffix):
                layer_name = key[len(layer_prefix) : -len(grad_norm_suffix)]
                layer_norms[layer_name] = float(value)
                continue
            if key.endswith(grad_frac_suffix):
                layer_name = key[len(layer_prefix) : -len(grad_frac_suffix)]
                layer_fractions[layer_name] = float(value)
                if value > 0.05:
                    active_layers.append(layer_name)

        drifting_traits: list[str] = []
        persona_delta_prefix = "geometry/persona/"
        persona_delta_suffix = "/delta"
        for key, value in metrics.items():
            if not key.startswith(persona_delta_prefix) or not key.endswith(persona_delta_suffix):
                continue
            if abs(value) > 0.2:
                trait = key[len(persona_delta_prefix) : -len(persona_delta_suffix)]
                drifting_traits.append(trait)

        return cls(
            hessian_trace_estimate=_float_or_none(metrics.get(GeometryMetricKey.hessian_trace)),
            top_hessian_eigenvalue=_float_or_none(metrics.get(GeometryMetricKey.top_eigenvalue)),
            hessian_condition_proxy=_float_or_none(metrics.get(GeometryMetricKey.condition_proxy)),
            gradient_variance=_float_or_none(metrics.get(GeometryMetricKey.gradient_variance)),
            gradient_snr=_float_or_none(metrics.get(GeometryMetricKey.gradient_snr)),
            effective_step_ratio=_float_or_none(metrics.get(GeometryMetricKey.effective_step_ratio)),
            per_layer_gradient_norms=layer_norms,
            per_layer_gradient_fractions=layer_fractions,
            active_layers=sorted(active_layers),
            parameter_divergence=_float_or_none(metrics.get(GeometryMetricKey.param_divergence)),
            parameter_cosine_similarity=_float_or_none(metrics.get(GeometryMetricKey.param_cosine_similarity)),
            refusal_distance=_float_or_none(metrics.get(GeometryMetricKey.refusal_distance)),
            is_approaching_refusal=(
                metrics.get(GeometryMetricKey.refusal_approaching, 0) > 0.5
                if GeometryMetricKey.refusal_approaching in metrics
                else None
            ),
            dare_effective_sparsity=_float_or_none(metrics.get(GeometryMetricKey.dare_effective_sparsity)),
            dora_magnitude_change=_float_or_none(metrics.get(GeometryMetricKey.dora_magnitude_change)),
            dora_directional_drift=_float_or_none(metrics.get(GeometryMetricKey.dora_directional_drift)),
            persona_drift_magnitude=_float_or_none(metrics.get(GeometryMetricKey.persona_overall_drift)),
            drifting_traits=sorted(drifting_traits),
            circuit_breaker_severity=_float_or_none(metrics.get(GeometryMetricKey.circuit_breaker_severity)),
            circuit_breaker_tripped=(
                metrics.get(GeometryMetricKey.circuit_breaker_tripped, 0) > 0.5
                if GeometryMetricKey.circuit_breaker_tripped in metrics
                else None
            ),
        )


@dataclass(frozen=True)
class MetricEntry:
    step: int
    metrics: GeometricTrainingMetrics


@dataclass
class GeometricMetricsHistory:
    entries: list[MetricEntry] = field(default_factory=list)

    def append(self, step: int, metrics: GeometricTrainingMetrics) -> None:
        self.entries.append(MetricEntry(step=step, metrics=metrics))

    @property
    def flatness_history(self) -> list[tuple[int, float]]:
        history: list[tuple[int, float]] = []
        for entry in self.entries:
            score = entry.metrics.flatness_score
            if score is None:
                continue
            history.append((entry.step, score))
        return history

    @property
    def snr_history(self) -> list[tuple[int, float]]:
        history: list[tuple[int, float]] = []
        for entry in self.entries:
            snr = entry.metrics.gradient_snr
            if snr is None:
                continue
            history.append((entry.step, snr))
        return history

    @property
    def divergence_history(self) -> list[tuple[int, float]]:
        history: list[tuple[int, float]] = []
        for entry in self.entries:
            divergence = entry.metrics.parameter_divergence
            if divergence is None:
                continue
            history.append((entry.step, divergence))
        return history

    def to_payload(self) -> list[dict]:
        payload: list[dict] = []
        for entry in self.entries:
            payload.append({"step": entry.step, "metrics": entry.metrics.to_metrics_dict()})
        return payload

    @classmethod
    def from_payload(cls, payload: list[dict]) -> "GeometricMetricsHistory":
        history = cls()
        for entry in payload or []:
            step = entry.get("step")
            metrics_payload = entry.get("metrics")
            if step is None or not isinstance(metrics_payload, dict):
                continue
            metrics = GeometricTrainingMetrics.from_progress_metrics(metrics_payload)
            if metrics is None:
                continue
            history.append(int(step), metrics)
        return history


def _float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)
