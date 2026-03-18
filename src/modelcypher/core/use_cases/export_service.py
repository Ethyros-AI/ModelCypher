from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from modelcypher.core.use_cases.model_capability_manifest import (
    ModelCapabilityManifestResolver,
)

if TYPE_CHECKING:
    from modelcypher.core.use_cases.quantization_service import QuantizationService
    from modelcypher.ports.exporter import Exporter


class ExportTargetKind(str, Enum):
    """Explicit deployment targets for trained adapters."""

    ADAPTER = "adapter"
    MERGED_FP16 = "merged_fp16"
    DEPLOYMENT_QUANTIZED = "deployment_quantized"


@dataclass(frozen=True)
class ExportRequest:
    model_path: Path
    adapter_path: Path
    output_path: Path
    target_kind: ExportTargetKind
    quantization_bits: int = 4
    quantization_group_size: int = 64
    quantization_mode: str = "nf4"


@dataclass(frozen=True)
class ExportOutcome:
    target_kind: str
    output_path: str
    capability_manifest: dict[str, Any]
    quantization: dict[str, Any] | None = None
    artifacts: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target_kind": self.target_kind,
            "output_path": self.output_path,
            "capability_manifest": dict(self.capability_manifest),
        }
        if self.quantization is not None:
            payload["quantization"] = dict(self.quantization)
        if self.artifacts is not None:
            payload["artifacts"] = list(self.artifacts)
        return payload


class ExportService:
    """Export adapters into explicit deployment targets."""

    def __init__(
        self,
        exporter: "Exporter",
        quantization_service: "QuantizationService",
        capability_resolver: ModelCapabilityManifestResolver | None = None,
    ) -> None:
        self._exporter = exporter
        self._quantization_service = quantization_service
        self._capability_resolver = capability_resolver or ModelCapabilityManifestResolver()

    def export(self, request: ExportRequest) -> ExportOutcome:
        manifest = self._capability_resolver.resolve(request.model_path)
        export_support = manifest.export_support
        if not export_support.get(request.target_kind.value, False):
            raise ValueError(
                f"{request.target_kind.value} is not supported for {request.model_path}"
            )

        artifacts: list[str] = []
        quantization: dict[str, Any] | None = None

        if request.target_kind == ExportTargetKind.ADAPTER:
            payload = self._exporter.export(
                model_path=str(request.model_path),
                adapter_path=str(request.adapter_path),
                output_path=str(request.output_path),
                target_kind=request.target_kind.value,
            )
            artifacts = [str(request.output_path)]
            output_path = str(request.output_path)
        elif request.target_kind == ExportTargetKind.MERGED_FP16:
            payload = self._exporter.export(
                model_path=str(request.model_path),
                adapter_path=str(request.adapter_path),
                output_path=str(request.output_path),
                target_kind=request.target_kind.value,
            )
            artifacts = [str(request.output_path)]
            output_path = str(request.output_path)
        else:
            with TemporaryDirectory(
                prefix=f"{request.adapter_path.name}-merged-",
            ) as tmp_dir:
                merged_dir = Path(tmp_dir) / "merged_fp16"
                self._exporter.export(
                    model_path=str(request.model_path),
                    adapter_path=str(request.adapter_path),
                    output_path=str(merged_dir),
                    target_kind=ExportTargetKind.MERGED_FP16.value,
                )
                quantized = self._quantization_service.quantize_model(
                    model_path=merged_dir,
                    output_dir=request.output_path,
                    bits=request.quantization_bits,
                    group_size=request.quantization_group_size,
                    mode=request.quantization_mode,
                    overwrite=True,
                )
                quantization = quantized.to_dict()
                payload = {"output_path": str(request.output_path)}
                artifacts = [str(request.output_path)]
                output_path = str(request.output_path)

        return ExportOutcome(
            target_kind=request.target_kind.value,
            output_path=output_path,
            capability_manifest=manifest.to_dict(),
            quantization=quantization,
            artifacts=artifacts or [str(payload.get("output_path", request.output_path))],
        )
