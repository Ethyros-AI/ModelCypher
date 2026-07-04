from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

from modelcypher.core.use_cases.export_service import ExportRequest


class ExportOrchestrationError(RuntimeError):
    """Raised when the isolated export worker fails."""


@dataclass(frozen=True)
class ExportWorkerResult:
    payload: dict


class ExportOrchestrator:
    """Run export work in a fresh Python process."""

    def __init__(self, python_executable: str | None = None) -> None:
        self._python_executable = python_executable or sys.executable

    def run(
        self,
        request: ExportRequest,
        *,
        timeout_seconds: int = 3600,
    ) -> ExportWorkerResult:
        cmd = [
            self._python_executable,
            "-m",
            "modelcypher.cli.export_worker",
            "--model",
            str(request.model_path),
            "--adapter",
            str(request.adapter_path),
            "--output",
            str(request.output_path),
            "--target",
            request.target_kind.value,
            "--quantization-bits",
            str(request.quantization_bits),
            "--quantization-group-size",
            str(request.quantization_group_size),
            "--quantization-mode",
            request.quantization_mode,
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or "isolated export worker failed"
            raise ExportOrchestrationError(detail)
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ExportOrchestrationError("isolated export worker returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ExportOrchestrationError("isolated export worker returned invalid payload")
        return ExportWorkerResult(payload=payload)
