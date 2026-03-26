from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelcypher.core.use_cases.observation_bundle_report_service import (
    ObservationBundleReportService,
)


def _write_bundle_fixture(
    tmp_path: Path,
    *,
    manifest: dict | None = None,
    summary: dict | None = None,
    variants: list[dict] | None = None,
    layer_rows: list[dict] | None = None,
    comparisons: list[dict] | None = None,
) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    manifest_payload = manifest or {
        "bundleVersion": "mc.analyze.bundle.v1",
        "workflow": "family",
        "promptFamilyManifest": {"name": "fixture_family"},
    }
    summary_payload = summary or {
        "bundleVersion": "mc.analyze.bundle.v1",
        "workflow": "family",
        "targetCount": 1,
        "variantCount": 2,
        "comparisonCount": 1,
        "errorCount": 1,
        "spaces": ["embedding", "hidden"],
        "meanPromptTokenCount": 5.0,
        "meanResponseTokenCount": 7.0,
        "meanEntropy": 0.3,
        "meanGeodesicDeviation": 0.2,
        "meanCurvature": 0.4,
    }
    variant_rows = variants or [
        {
            "targetLabel": "base",
            "caseId": "case1",
            "variantId": "control",
            "summaryMetrics": {
                "meanEntropy": 0.2,
                "meanGeodesicDeviation": 0.1,
                "meanCurvature": 0.3,
            },
        },
        {
            "targetLabel": "base",
            "caseId": "case1",
            "variantId": "all_caps",
            "summaryMetrics": {
                "meanEntropy": 0.5,
                "meanGeodesicDeviation": 0.4,
                "meanCurvature": 0.7,
            },
            "errors": ["entropy:fixture failure"],
        },
    ]
    layer_metric_rows = layer_rows or [
        {
            "targetLabel": "base",
            "caseId": "case1",
            "variantId": "control",
            "layer": -1,
            "space": "embedding",
            "vectorNorm": 0.5,
        },
        {
            "targetLabel": "base",
            "caseId": "case1",
            "variantId": "all_caps",
            "layer": 0,
            "space": "hidden",
            "vectorNorm": 1.0,
        },
        {
            "targetLabel": "base",
            "caseId": "case1",
            "variantId": "all_caps",
            "layer": 1,
            "space": "hidden",
            "vectorNorm": 2.0,
        },
    ]
    comparison_rows = comparisons if comparisons is not None else [
        {
            "mode": "within_target",
            "caseId": "case1",
            "from": "control",
            "to": "all_caps",
            "targetLabel": "base",
            "scalarDeltas": {
                "meanEntropy": 0.3,
                "meanCurvature": 0.4,
            },
            "perLayerDeltas": {
                "entropy": [{"layer": 1, "delta": 0.6}],
                "flowMeanCurvature": [{"layer": 0, "delta": -0.8}],
            },
        }
    ]

    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest_payload) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "summary.json").write_text(
        json.dumps(summary_payload) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "variants.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in variant_rows),
        encoding="utf-8",
    )
    (bundle_dir / "layer_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in layer_metric_rows),
        encoding="utf-8",
    )
    (bundle_dir / "comparisons.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in comparison_rows),
        encoding="utf-8",
    )
    (bundle_dir / "REPORT.md").write_text("# existing report\n", encoding="utf-8")
    return bundle_dir


def test_build_computes_sections_from_rows(tmp_path: Path) -> None:
    bundle_dir = _write_bundle_fixture(tmp_path)
    service = ObservationBundleReportService()
    result = service.load(bundle_dir)

    observed_spaces = result.sections["observedSpaces"]
    assert {row["space"] for row in observed_spaces} == {"embedding", "hidden"}

    top_scalar = result.sections["topScalarShifts"][0]
    assert top_scalar["metric"] == "meanCurvature"
    assert top_scalar["delta"] == pytest.approx(0.4)

    top_layer = result.sections["topLayerShifts"][0]
    assert top_layer["layer"] == 0
    assert top_layer["metric"] == "flowMeanCurvature"
    assert top_layer["delta"] == pytest.approx(-0.8)

    error_sample = result.sections["errorSamples"][0]
    assert error_sample["variantId"] == "all_caps"
    assert error_sample["errors"] == ["entropy:fixture failure"]


def test_load_rejects_missing_required_file(tmp_path: Path) -> None:
    bundle_dir = _write_bundle_fixture(tmp_path)
    (bundle_dir / "summary.json").unlink()

    service = ObservationBundleReportService()
    with pytest.raises(ValueError, match="missing required files"):
        service.load(bundle_dir)


def test_load_rejects_malformed_json(tmp_path: Path) -> None:
    bundle_dir = _write_bundle_fixture(tmp_path)
    (bundle_dir / "summary.json").write_text("{not-json\n", encoding="utf-8")

    service = ObservationBundleReportService()
    with pytest.raises(ValueError, match="Malformed JSON in summary.json"):
        service.load(bundle_dir)


def test_load_rejects_malformed_jsonl(tmp_path: Path) -> None:
    bundle_dir = _write_bundle_fixture(tmp_path)
    (bundle_dir / "comparisons.jsonl").write_text("{bad-line\n", encoding="utf-8")

    service = ObservationBundleReportService()
    with pytest.raises(ValueError, match="Malformed JSONL in comparisons.jsonl"):
        service.load(bundle_dir)


def test_load_handles_empty_comparisons_file(tmp_path: Path) -> None:
    bundle_dir = _write_bundle_fixture(tmp_path, comparisons=[])
    service = ObservationBundleReportService()

    result = service.load(bundle_dir)

    assert result.summary["comparisonCount"] == 0
    assert result.sections["topScalarShifts"] == []
    assert result.sections["topLayerShifts"] == []
    assert "## Comparisons" not in result.markdown
