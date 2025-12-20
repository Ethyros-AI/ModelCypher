"""Dashboard service for metrics and export functionality.

Provides dashboard metrics in Prometheus format and export capabilities
for Grafana integration and monitoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DashboardExportResult:
    """Result of dashboard export."""

    format: str
    export_path: str | None
    content: str
    exported_at: str
    metrics_count: int


class DashboardService:
    """Service for dashboard metrics and export.

    Provides metrics in Prometheus format for Grafana integration
    and supports various export formats for monitoring dashboards.
    """

    def __init__(self) -> None:
        """Initialize dashboard service."""
        self._metrics: dict[str, Any] = {}

    def metrics(self) -> str:
        """Return current metrics in Prometheus format.

        Returns:
            String containing metrics in Prometheus exposition format
        """
        lines = [
            "# HELP modelcypher_jobs_total Total number of training jobs",
            "# TYPE modelcypher_jobs_total counter",
            "modelcypher_jobs_total 0",
            "",
            "# HELP modelcypher_jobs_active Number of active training jobs",
            "# TYPE modelcypher_jobs_active gauge",
            "modelcypher_jobs_active 0",
            "",
            "# HELP modelcypher_models_registered Number of registered models",
            "# TYPE modelcypher_models_registered gauge",
            "modelcypher_models_registered 0",
            "",
            "# HELP modelcypher_checkpoints_total Total number of checkpoints",
            "# TYPE modelcypher_checkpoints_total counter",
            "modelcypher_checkpoints_total 0",
            "",
            "# HELP modelcypher_gpu_memory_bytes GPU memory usage in bytes",
            "# TYPE modelcypher_gpu_memory_bytes gauge",
            "modelcypher_gpu_memory_bytes 0",
            "",
            "# HELP modelcypher_gpu_memory_total_bytes Total GPU memory in bytes",
            "# TYPE modelcypher_gpu_memory_total_bytes gauge",
            "modelcypher_gpu_memory_total_bytes 0",
            "",
            "# HELP modelcypher_training_loss Current training loss",
            "# TYPE modelcypher_training_loss gauge",
            "modelcypher_training_loss 0",
            "",
            "# HELP modelcypher_training_step Current training step",
            "# TYPE modelcypher_training_step gauge",
            "modelcypher_training_step 0",
            "",
            "# HELP modelcypher_tokens_per_second Training throughput",
            "# TYPE modelcypher_tokens_per_second gauge",
            "modelcypher_tokens_per_second 0",
            "",
            "# HELP modelcypher_geometry_flatness Geometry flatness score",
            "# TYPE modelcypher_geometry_flatness gauge",
            "modelcypher_geometry_flatness 0",
            "",
            "# HELP modelcypher_geometry_snr Gradient signal-to-noise ratio",
            "# TYPE modelcypher_geometry_snr gauge",
            "modelcypher_geometry_snr 0",
            "",
            "# HELP modelcypher_circuit_breaker_severity Circuit breaker severity",
            "# TYPE modelcypher_circuit_breaker_severity gauge",
            "modelcypher_circuit_breaker_severity 0",
            "",
            "# HELP modelcypher_info ModelCypher version info",
            "# TYPE modelcypher_info gauge",
            'modelcypher_info{version="1.0.0"} 1',
        ]

        return "\n".join(lines)

    def export(
        self,
        format: str,
        output_path: str | None = None,
    ) -> DashboardExportResult:
        """Export dashboard data in specified format.

        Args:
            format: Export format (prometheus, json, csv)
            output_path: Optional path to write export

        Returns:
            DashboardExportResult with export content

        Raises:
            ValueError: If format is not supported
        """
        supported_formats = {"prometheus", "json", "csv"}
        if format.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported format: {format}. Supported: {supported_formats}"
            )

        exported_at = datetime.now(timezone.utc).isoformat()

        if format.lower() == "prometheus":
            content = self.metrics()
            metrics_count = content.count("# TYPE")
        elif format.lower() == "json":
            content = self._export_json()
            metrics_count = len(self._get_metrics_dict())
        else:  # csv
            content = self._export_csv()
            metrics_count = len(self._get_metrics_dict())

        if output_path:
            from pathlib import Path

            Path(output_path).write_text(content, encoding="utf-8")
            logger.info("Exported dashboard to %s", output_path)

        return DashboardExportResult(
            format=format.lower(),
            export_path=output_path,
            content=content,
            exported_at=exported_at,
            metrics_count=metrics_count,
        )

    def _get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics as a dictionary."""
        return {
            "jobs_total": 0,
            "jobs_active": 0,
            "models_registered": 0,
            "checkpoints_total": 0,
            "gpu_memory_bytes": 0,
            "gpu_memory_total_bytes": 0,
            "training_loss": 0,
            "training_step": 0,
            "tokens_per_second": 0,
            "geometry_flatness": 0,
            "geometry_snr": 0,
            "circuit_breaker_severity": 0,
            "version": "1.0.0",
        }

    def _export_json(self) -> str:
        """Export metrics as JSON."""
        import json

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self._get_metrics_dict(),
        }
        return json.dumps(data, indent=2)

    def _export_csv(self) -> str:
        """Export metrics as CSV."""
        lines = ["metric,value"]
        for key, value in self._get_metrics_dict().items():
            lines.append(f"{key},{value}")
        return "\n".join(lines)
