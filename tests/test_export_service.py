from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from modelcypher.core.use_cases.export_service import (
    ExportRequest,
    ExportService,
    ExportTargetKind,
)
from modelcypher.core.use_cases.model_capability_manifest import (
    ModelCapabilityManifest,
)


class _FakeExporter:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    def export(
        self,
        *,
        model_path: str,
        adapter_path: str | None,
        output_path: str,
        target_kind: str,
    ) -> dict:
        self.calls.append(
            {
                "model_path": model_path,
                "adapter_path": adapter_path,
                "output_path": output_path,
                "target_kind": target_kind,
            }
        )
        return {"output_path": output_path, "target_kind": target_kind}


@dataclass(frozen=True)
class _QuantizedResult:
    output_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {"outputDir": str(self.output_dir)}


class _FakeQuantizationService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def quantize_model(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _QuantizedResult(output_dir=Path(kwargs["output_dir"]))


class _FixedResolver:
    def __init__(self, manifest: ModelCapabilityManifest) -> None:
        self._manifest = manifest

    def resolve(self, _model_path: str | Path) -> ModelCapabilityManifest:
        return self._manifest


def test_export_service_uses_explicit_adapter_target(tmp_path: Path) -> None:
    exporter = _FakeExporter()
    quantization = _FakeQuantizationService()
    manifest = ModelCapabilityManifest(
        model_path=str(tmp_path / "model"),
        export_support={
            "adapter": True,
            "merged_fp16": True,
            "deployment_quantized": True,
        },
    )
    service = ExportService(
        exporter=exporter,
        quantization_service=quantization,
        capability_resolver=_FixedResolver(manifest),
    )

    outcome = service.export(
        ExportRequest(
            model_path=tmp_path / "model",
            adapter_path=tmp_path / "adapter",
            output_path=tmp_path / "exported-adapter",
            target_kind=ExportTargetKind.ADAPTER,
        )
    )

    assert outcome.target_kind == "adapter"
    assert exporter.calls[0]["target_kind"] == "adapter"
    assert quantization.calls == []


def test_export_service_quantized_target_runs_merge_then_quantize(tmp_path: Path) -> None:
    exporter = _FakeExporter()
    quantization = _FakeQuantizationService()
    manifest = ModelCapabilityManifest(
        model_path=str(tmp_path / "model"),
        export_support={
            "adapter": True,
            "merged_fp16": True,
            "deployment_quantized": True,
        },
    )
    service = ExportService(
        exporter=exporter,
        quantization_service=quantization,
        capability_resolver=_FixedResolver(manifest),
    )

    outcome = service.export(
        ExportRequest(
            model_path=tmp_path / "model",
            adapter_path=tmp_path / "adapter",
            output_path=tmp_path / "deployment",
            target_kind=ExportTargetKind.DEPLOYMENT_QUANTIZED,
        )
    )

    assert exporter.calls[0]["target_kind"] == "merged_fp16"
    assert quantization.calls[0]["output_dir"] == tmp_path / "deployment"
    assert outcome.quantization is not None


def test_export_service_fails_closed_when_manifest_disables_target(tmp_path: Path) -> None:
    service = ExportService(
        exporter=_FakeExporter(),
        quantization_service=_FakeQuantizationService(),
        capability_resolver=_FixedResolver(
            ModelCapabilityManifest(
                model_path=str(tmp_path / "model"),
                export_support={
                    "adapter": True,
                    "merged_fp16": True,
                    "deployment_quantized": False,
                },
            )
        ),
    )

    with pytest.raises(ValueError, match="deployment_quantized"):
        service.export(
            ExportRequest(
                model_path=tmp_path / "model",
                adapter_path=tmp_path / "adapter",
                output_path=tmp_path / "deployment",
                target_kind=ExportTargetKind.DEPLOYMENT_QUANTIZED,
            )
        )
