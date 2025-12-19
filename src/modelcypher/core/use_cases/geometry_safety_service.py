from __future__ import annotations

from dataclasses import dataclass

from modelcypher.core.domain.circuit_breaker import (
    CircuitBreakerIntegration,
    CircuitBreakerState,
    Configuration,
    InputSignals,
)
from modelcypher.core.domain.geometric_training_metrics import GeometricTrainingMetrics
from modelcypher.core.use_cases.geometry_training_service import GeometryTrainingService


@dataclass(frozen=True)
class PersonaDriftInfo:
    overall_drift_magnitude: float
    assessment: str
    drifting_traits: list[str]
    refusal_distance: float | None
    is_approaching_refusal: bool | None


class GeometrySafetyService:
    def __init__(self, training_service: GeometryTrainingService | None = None) -> None:
        self.training_service = training_service or GeometryTrainingService()

    def evaluate_circuit_breaker(
        self,
        job_id: str | None = None,
        entropy_signal: float | None = None,
        refusal_distance: float | None = None,
        persona_drift_magnitude: float | None = None,
        has_oscillation: bool = False,
        configuration: Configuration | None = None,
    ) -> tuple[CircuitBreakerState, InputSignals]:
        signals = InputSignals(
            entropy_signal=entropy_signal,
            refusal_distance=refusal_distance,
            persona_drift_magnitude=persona_drift_magnitude,
            has_oscillation=has_oscillation,
        )

        if job_id:
            metrics = self.training_service.get_metrics(job_id)
            if metrics:
                resolved_refusal = (
                    metrics.refusal_distance if metrics.refusal_distance is not None else refusal_distance
                )
                resolved_persona = (
                    metrics.persona_drift_magnitude
                    if metrics.persona_drift_magnitude is not None
                    else persona_drift_magnitude
                )
                signals = InputSignals(
                    entropy_signal=entropy_signal,
                    refusal_distance=resolved_refusal,
                    is_approaching_refusal=metrics.is_approaching_refusal,
                    persona_drift_magnitude=resolved_persona,
                    drifting_traits=metrics.drifting_traits,
                    has_oscillation=has_oscillation,
                )

        state = CircuitBreakerIntegration.evaluate(signals, configuration=configuration)
        return state, signals

    def persona_drift(self, job_id: str) -> PersonaDriftInfo | None:
        metrics = self.training_service.get_metrics(job_id)
        if metrics is None:
            return None

        drift_magnitude = metrics.persona_drift_magnitude or 0.0
        if drift_magnitude < 0.1:
            assessment = "minimal"
        elif drift_magnitude < 0.3:
            assessment = "moderate"
        elif drift_magnitude < 0.5:
            assessment = "significant"
        else:
            assessment = "critical"

        return PersonaDriftInfo(
            overall_drift_magnitude=drift_magnitude,
            assessment=assessment,
            drifting_traits=metrics.drifting_traits,
            refusal_distance=metrics.refusal_distance,
            is_approaching_refusal=metrics.is_approaching_refusal,
        )

    @staticmethod
    def persona_interpretation(info: PersonaDriftInfo) -> str:
        if info.assessment == "minimal":
            return "Persona alignment stable. Training is not significantly affecting character traits."
        if info.assessment == "moderate":
            return "Moderate persona drift detected. Monitor closely for alignment degradation."
        if info.assessment == "significant":
            return "Significant persona drift. Consider pausing training to evaluate alignment."
        if info.assessment == "critical":
            return "Critical persona drift. Recommend immediate training intervention."
        return f"Persona drift magnitude: {info.overall_drift_magnitude:.3f}"
