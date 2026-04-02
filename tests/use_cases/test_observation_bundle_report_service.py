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


def _write_measurement_atlas_fixture(
    tmp_path: Path,
    *,
    schema_version: str = "mc.measurement_atlas.run_manifest.v2",
    include_report: bool = True,
    include_locus_comparisons: bool = True,
    legacy_embedding_shift: bool = False,
) -> Path:
    bundle_dir = tmp_path / "atlas-bundle"
    bundle_dir.mkdir()
    long_prompt = (
        "ALPHA\\nBETA\n\n"
        "This prompt keeps going with extra context about casing and reasoning. "
        "This prompt keeps going with extra context about casing and reasoning. "
        "This prompt keeps going with extra context about casing and reasoning."
    )
    long_generated = (
        "SUPPORTED\\n\\n"
        "This generated explanation keeps going across lines and should be clipped for atlas previews. "
        "This generated explanation keeps going across lines and should be clipped for atlas previews. "
        "This generated explanation keeps going across lines and should be clipped for atlas previews."
    )

    if schema_version.endswith(".v2"):
        frozen_surfaces = {
            "requestedLiveSpaces": ["hidden"],
            "observedLiveSpaces": ["hidden"],
            "requestedReplaySpaces": ["hidden", "embedding"],
            "observedReplaySpaces": ["hidden", "embedding"],
        }
    else:
        frozen_surfaces = {
            "liveSpaces": ["hidden"],
            "replaySpaces": ["hidden", "embedding", "intermediate"],
        }

    manifest_payload = {
        "schema": schema_version,
        "runId": "atlas-run",
        "linkedBlocker": "A1",
        "frozenSurfaces": frozen_surfaces,
    }
    summary_payload = {
        "runId": "atlas-run",
        "linkedBlocker": "A1",
        "studyCount": 1,
        "variantCount": 2,
        "comparisonCount": 1,
        "onsetEventCount": 2,
        "errorCount": 0,
        "spaces": ["embedding", "hidden"],
        "modes": ["live", "replay"],
    }
    variant_rows = [
        {
            "studyId": "measurement_atlas_casing",
            "caseId": "case1",
            "variantId": "control",
            "promptText": "alpha beta",
            "generatedText": "SUPPORTED",
            "errors": [],
        },
        {
            "studyId": "measurement_atlas_casing",
            "caseId": "case1",
            "variantId": "all_caps",
            "promptText": long_prompt,
            "generatedText": long_generated,
            "errors": [],
        },
    ]
    embedding_control_row = {
        "studyId": "measurement_atlas_casing",
        "caseId": "case1",
        "variantId": "control",
        "mode": "replay",
        "region": "response",
        "space": "embedding",
        "tokenCount": 1,
        "meanGeodesicDeviation": 0.1,
        "meanCurvature": 0.1,
        "maxCurvature": 0.2,
        "meanSpectralEntropy": 0.3,
        "meanEffectiveRank": 1.0,
        "meanIntrinsicDimension": 2.0,
        "meanPathLengthRatio": 1.1,
        "meanEntropy": None,
    }
    embedding_variant_row = dict(embedding_control_row)
    embedding_variant_row["variantId"] = "all_caps"
    if legacy_embedding_shift:
        embedding_control_row["peakLayer"] = -1
        embedding_control_row["firstBendLayer"] = -1
        embedding_variant_row["peakLayer"] = -1
        embedding_variant_row["firstBendLayer"] = -1
    else:
        embedding_control_row["peakLayer"] = None
        embedding_control_row["peakLocus"] = "embedding"
        embedding_control_row["firstBendLayer"] = None
        embedding_control_row["firstBendLocus"] = "embedding"
        embedding_variant_row["peakLayer"] = None
        embedding_variant_row["peakLocus"] = "embedding"
        embedding_variant_row["firstBendLayer"] = None
        embedding_variant_row["firstBendLocus"] = "embedding"

    sequence_metrics = [
        embedding_control_row,
        {
            "studyId": "measurement_atlas_casing",
            "caseId": "case1",
            "variantId": "control",
            "mode": "replay",
            "region": "response",
            "space": "hidden",
            "tokenCount": 1,
            "peakLayer": 1,
            "peakLocus": "layer:1",
            "firstBendLayer": None,
            "firstBendLocus": None,
            "meanEntropy": 0.2,
            "meanSpectralEntropy": 0.3,
            "meanEffectiveRank": 1.2,
            "meanIntrinsicDimension": 2.2,
            "meanCurvature": 0.4,
            "maxCurvature": 0.6,
            "meanGeodesicDeviation": 0.3,
            "meanPathLengthRatio": 1.2,
        },
        embedding_variant_row,
        {
            "studyId": "measurement_atlas_casing",
            "caseId": "case1",
            "variantId": "all_caps",
            "mode": "replay",
            "region": "response",
            "space": "hidden",
            "tokenCount": 1,
            "peakLayer": 3,
            "peakLocus": "layer:3",
            "firstBendLayer": 2,
            "firstBendLocus": "layer:2",
            "meanEntropy": 0.5,
            "meanSpectralEntropy": 0.5,
            "meanEffectiveRank": 1.5,
            "meanIntrinsicDimension": 2.5,
            "meanCurvature": 0.8,
            "maxCurvature": 1.0,
            "meanGeodesicDeviation": 0.7,
            "meanPathLengthRatio": 1.5,
        },
    ]
    comparison_payload = {
        "studyId": "measurement_atlas_casing",
        "caseId": "case1",
        "from": "control",
        "to": "all_caps",
        "alignmentMode": "step_index_min_prefix",
        "sharedStepCount": 1,
        "sequenceLengthDeltas": [],
        "scalarDeltas": {
            "replay.response.hidden.meanGeodesicDeviation": 0.4,
        },
        "sequenceDeltas": [
            {
                "mode": "replay",
                "region": "response",
                "space": "hidden",
                "metric": "meanGeodesicDeviation",
                "baselineValue": 0.3,
                "variantValue": 0.7,
                "delta": 0.4,
            },
            {
                "mode": "replay",
                "region": "response",
                "space": "embedding",
                "metric": "meanGeodesicDeviation",
                "baselineValue": 0.1,
                "variantValue": 0.1,
                "delta": 0.0,
            },
            {
                "mode": "replay",
                "region": "response",
                "space": "hidden",
                "metric": "meanPathLengthRatio",
                "baselineValue": 1.2,
                "variantValue": 99.2,
                "delta": 98.0,
            },
        ],
        "liveGeneratedFirstDivergenceStep": 1,
        "replayResponseFirstDivergenceStep": 1,
        "firstGeneratedShiftAgreement": True,
    }
    if include_locus_comparisons:
        comparison_payload["locusComparisons"] = [
            {
                "mode": "replay",
                "region": "response",
                "space": "hidden",
                "metric": "peak",
                "baselineLocus": "layer:1",
                "variantLocus": "layer:3",
                "changed": True,
            }
        ]

    (bundle_dir / "run_manifest.json").write_text(
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
    (bundle_dir / "sequence_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sequence_metrics),
        encoding="utf-8",
    )
    (bundle_dir / "step_metrics.jsonl").write_text(
        json.dumps({"studyId": "measurement_atlas_casing", "stepIndex": 0}) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "space_step_metrics.jsonl").write_text(
        json.dumps({"studyId": "measurement_atlas_casing", "space": "hidden", "stepIndex": 0}) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "comparisons.jsonl").write_text(
        json.dumps(comparison_payload) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "onset_events.jsonl").write_text(
        "".join(
            [
                json.dumps(
                    {
                        "studyId": "measurement_atlas_casing",
                        "caseId": "case1",
                        "variantId": "all_caps",
                        "eventType": "grounded_label_onset",
                        "mode": "live",
                        "region": "generated",
                        "stepIndex": 2,
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "studyId": "measurement_atlas_casing",
                        "caseId": "case1",
                        "variantId": "all_caps",
                        "baselineVariantId": "control",
                        "eventType": "first_divergence",
                        "mode": "replay",
                        "region": "response",
                        "stepIndex": 1,
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )
    if include_report:
        (bundle_dir / "REPORT.md").write_text("# retained atlas report\n", encoding="utf-8")
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


def test_load_measurement_atlas_bundle_builds_expected_sections(tmp_path: Path) -> None:
    bundle_dir = _write_measurement_atlas_fixture(tmp_path)
    service = ObservationBundleReportService()

    result = service.load(bundle_dir)

    assert result.summary["workflow"] == "measurement_atlas"
    assert result.sections["surfaces"]["observedReplaySpaces"] == ["hidden", "embedding"]
    assert result.sections["studySummaries"][0]["studyId"] == "measurement_atlas_casing"
    assert result.sections["topSequenceShifts"][0]["metric"] == "meanGeodesicDeviation"
    assert all(
        row["metric"] == "meanGeodesicDeviation"
        for row in result.sections["topSequenceShifts"]
    )
    assert result.sections["locusChanges"][0]["baselineLocus"] == "layer:1"
    example = result.sections["exampleComparisons"][0]
    assert "promptText" not in example
    assert "generatedText" not in example
    assert example["promptPreview"].startswith("ALPHA BETA This prompt keeps going")
    assert "\n" not in example["promptPreview"]
    assert "\\n" not in example["promptPreview"]
    assert "\\n" not in example["generatedPreview"]
    assert example["promptPreview"].endswith("...")
    assert example["generatedPreview"].endswith("...")
    assert example["promptCharCount"] > len(example["promptPreview"])
    assert example["generatedCharCount"] > len(example["generatedPreview"])
    assert result.sections["onsetSamples"][0]["eventType"] == "grounded_label_onset"
    assert "# Measurement Atlas Bundle" in result.markdown
    assert "## Largest Geodesic Shifts" in result.markdown
    assert "## Largest Sequence Shifts" not in result.markdown
    assert "biggest movement was in" in result.markdown
    assert "meanPathLengthRatio" not in result.markdown
    assert "prompt='ALPHA" not in result.markdown
    assert "prompt=`ALPHA BETA" in result.markdown
    assert "\\n\\n" not in result.markdown


def test_load_measurement_atlas_rejects_missing_required_file(tmp_path: Path) -> None:
    bundle_dir = _write_measurement_atlas_fixture(tmp_path)
    (bundle_dir / "onset_events.jsonl").unlink()
    service = ObservationBundleReportService()

    with pytest.raises(ValueError, match="missing required files"):
        service.load(bundle_dir)


def test_load_measurement_atlas_rejects_malformed_json(tmp_path: Path) -> None:
    bundle_dir = _write_measurement_atlas_fixture(tmp_path)
    (bundle_dir / "run_manifest.json").write_text("{not-json\n", encoding="utf-8")
    service = ObservationBundleReportService()

    with pytest.raises(ValueError, match="Malformed JSON in run_manifest.json"):
        service.load(bundle_dir)


def test_load_measurement_atlas_rejects_malformed_jsonl(tmp_path: Path) -> None:
    bundle_dir = _write_measurement_atlas_fixture(tmp_path)
    (bundle_dir / "comparisons.jsonl").write_text("{bad-line\n", encoding="utf-8")
    service = ObservationBundleReportService()

    with pytest.raises(ValueError, match="Malformed JSONL in comparisons.jsonl"):
        service.load(bundle_dir)


def test_load_measurement_atlas_normalizes_v1_surfaces_without_observed_claims(
    tmp_path: Path,
) -> None:
    bundle_dir = _write_measurement_atlas_fixture(
        tmp_path,
        schema_version="mc.measurement_atlas.run_manifest.v1",
    )
    service = ObservationBundleReportService()

    result = service.load(bundle_dir)

    surfaces = result.sections["surfaces"]
    assert surfaces["requestedLiveSpaces"] == ["hidden"]
    assert surfaces["requestedReplaySpaces"] == ["hidden", "embedding", "intermediate"]
    assert surfaces["observedLiveSpaces"] is None
    assert surfaces["observedReplaySpaces"] is None


def test_load_measurement_atlas_normalizes_v2_surfaces_with_observed_claims(
    tmp_path: Path,
) -> None:
    bundle_dir = _write_measurement_atlas_fixture(tmp_path)
    service = ObservationBundleReportService()

    result = service.load(bundle_dir)

    surfaces = result.sections["surfaces"]
    assert surfaces["requestedLiveSpaces"] == ["hidden"]
    assert surfaces["observedLiveSpaces"] == ["hidden"]
    assert surfaces["requestedReplaySpaces"] == ["hidden", "embedding"]
    assert surfaces["observedReplaySpaces"] == ["hidden", "embedding"]


def test_load_measurement_atlas_legacy_embedding_rows_render_embedding_without_sentinel(
    tmp_path: Path,
) -> None:
    bundle_dir = _write_measurement_atlas_fixture(
        tmp_path,
        include_locus_comparisons=False,
        legacy_embedding_shift=True,
    )
    service = ObservationBundleReportService()

    result = service.load(bundle_dir)

    assert result.sections["studySummaries"][0]["earliestShiftLocus"] == "embedding"
    assert "-1" not in result.markdown
