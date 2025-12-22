from __future__ import annotations

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.geometric_training_metrics import (
    GeometricMetricsHistory,
    GeometricTrainingMetrics,
)


class GeometryTrainingService:
    def __init__(self, store: FileSystemStore | None = None) -> None:
        self.store = store or FileSystemStore()

    def get_metrics(self, job_id: str) -> GeometricTrainingMetrics | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        metrics = job.metrics or {}
        return GeometricTrainingMetrics.from_progress_metrics(metrics)

    def get_history(self, job_id: str) -> GeometricMetricsHistory:
        job = self.store.get_job(job_id)
        if job is None or not job.metrics_history:
            return GeometricMetricsHistory()
        return GeometricMetricsHistory.from_payload(job.metrics_history)

    def training_status_payload(
        self,
        job_id: str,
        output_format: str = "full",
        require_metrics: bool = True,
    ) -> dict:
        job = self.store.get_job(job_id)
        if job is None:
            raise ValueError(f"Job '{job_id}' not found")

        metrics = GeometricTrainingMetrics.from_progress_metrics(job.metrics or {})
        if metrics is None and require_metrics:
            raise ValueError(f"Job '{job_id}' not found or has no geometric metrics")
        has_metrics = metrics is not None
        if metrics is None:
            metrics = GeometricTrainingMetrics()
        checkpoints = self.store.list_checkpoints(job_id)
        step = max((checkpoint.step for checkpoint in checkpoints), default=0)
        if step == 0 and job is not None:
            if job.current_step:
                step = job.current_step
            elif job.total_steps:
                step = int((job.current_step / job.total_steps) * 100)

        thresholds = None
        if output_format == "full":
            thresholds = {
                "snrLow": 1.0,
                "snrAdequate": 10.0,
                "flatnessWarning": 0.4,
                "circuitBreakerThreshold": 0.75,
            }

        interpretation_parts: list[str] = []
        if metrics.flatness_score is not None:
            interpretation_parts.append(
                f"Flatness: {metrics.flatness_assessment} ({metrics.flatness_score:.2f})"
            )
        if metrics.gradient_snr is not None:
            interpretation_parts.append(
                f"SNR: {metrics.snr_assessment} ({metrics.gradient_snr:.1f})"
            )
        if metrics.circuit_breaker_tripped:
            interpretation_parts.append("WARNING: Circuit breaker tripped")
        interpretation = (
            "No geometric metrics available" if not interpretation_parts else ". ".join(interpretation_parts)
        )

        return {
            "_schema": "mc.geometry.training_status.v1",
            "jobId": job_id,
            "step": step,
            "flatnessScore": metrics.flatness_score,
            "flatnessAssessment": (
                metrics.flatness_assessment if output_format == "full" and has_metrics else None
            ),
            "hessianTraceEstimate": metrics.hessian_trace_estimate if output_format == "full" else None,
            "topHessianEigenvalue": metrics.top_hessian_eigenvalue if output_format == "full" else None,
            "hessianConditionProxy": metrics.hessian_condition_proxy if output_format == "full" else None,
            "gradientSNR": metrics.gradient_snr,
            "snrAssessment": metrics.snr_assessment if output_format == "full" and has_metrics else None,
            "gradientVariance": metrics.gradient_variance if output_format == "full" else None,
            "effectiveStepRatio": metrics.effective_step_ratio if output_format == "full" else None,
            "activeLayers": metrics.active_layers,
            "perLayerGradientNorms": (
                metrics.per_layer_gradient_norms if output_format == "full" else None
            ),
            "circuitBreakerSeverity": metrics.circuit_breaker_severity,
            "circuitBreakerTripped": metrics.circuit_breaker_tripped,
            "refusalDistance": metrics.refusal_distance,
            "interpretation": interpretation,
            "thresholds": thresholds,
            "nextActions": [
                "mc_geometry_training_history for trend analysis",
                "mc_safety_circuit_breaker for detailed safety evaluation",
                "mc_safety_persona_drift for alignment monitoring",
            ],
        }

    def training_history_payload(self, job_id: str) -> dict:
        history = self.get_history(job_id)

        flatness_history = [
            {"step": step, "value": value} for step, value in history.flatness_history
        ]
        snr_history = [{"step": step, "value": value} for step, value in history.snr_history]
        divergence_history = [
            {"step": step, "value": value} for step, value in history.divergence_history
        ]

        if history.entries:
            interpretation = (
                f"History contains {len(history.entries)} entries from step "
                f"{history.entries[0].step} to {history.entries[-1].step}"
            )
        else:
            interpretation = (
                "No history available. Geometric metrics are captured in real-time during training. "
                "Use mc_geometry_training_status for current metrics."
            )

        return {
            "_schema": "mc.geometry.training_history.v1",
            "jobId": job_id,
            "startStep": history.entries[0].step if history.entries else 0,
            "endStep": history.entries[-1].step if history.entries else 0,
            "sampleCount": len(history.entries),
            "flatnessHistory": flatness_history or None,
            "snrHistory": snr_history or None,
            "circuitBreakerHistory": None,
            "parameterDivergenceHistory": divergence_history or None,
            "trendAnalysis": None,
            "interpretation": interpretation,
            "nextActions": [
                "mc_geometry_training_status for current metrics",
                "mc_job_detail for full job information",
            ],
        }
