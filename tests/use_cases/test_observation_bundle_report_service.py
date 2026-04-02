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


def _write_pipeline_validation_fixture(
    tmp_path: Path,
    *,
    bundle_name: str = "pipeline-validation",
    include_failures: bool = True,
    include_contract_tags: bool = False,
    include_report: bool = True,
) -> Path:
    bundle_dir = tmp_path / bundle_name
    bundle_dir.mkdir()
    scale_dir = bundle_dir / "350M"
    scale_dir.mkdir()

    verdict_payload = {
        "timestamp": "2026-04-02T00:00:00+00:00",
        "git_hash": "deadbeef",
        "trials_per_model": 5 if include_failures else 20,
        "controller_mode": "mass_behavioral_probe",
        "optimizer_research_mode": "adamw_matched_trace",
        "benchmark_suite": "quick",
        "scales": ["350M"],
        "all_pass": not include_failures,
        "all_structural_pass": True,
        "all_inference_pass": not include_failures,
        "per_scale": {
            "350M": {
                "all_passed": not include_failures,
                "pass_count": 3 if include_failures else 20,
                "fail_count": 2 if include_failures else 0,
                "structural_pass_count": 5 if include_failures else 20,
                "structural_fail_count": 0,
                "inference_pass_count": 3 if include_failures else 20,
                "inference_fail_count": 2 if include_failures else 0,
                "phase5_inference_enabled": True,
                "error": None,
            }
        },
    }
    summary_payload = {
        "family": bundle_name,
        "status": "canonical",
        "retained_artifacts": [
            f"results/{bundle_name}/verdict.json",
            f"results/{bundle_name}/350M/result.json",
        ],
        "aggregate_verdict": verdict_payload,
        "per_scale_summary": {
            "350M": {
                "pass_count": 3 if include_failures else 20,
                "fail_count": 2 if include_failures else 0,
                "structural_pass_count": 5 if include_failures else 20,
                "structural_fail_count": 0,
                "inference_pass_count": 3 if include_failures else 20,
                "inference_fail_count": 2 if include_failures else 0,
                "phase5_probe_count": 10,
                "phase5_probe_seed": 3475334679,
                "mean_loss_delta": 0.9840814639514399 if include_failures else 0.7889143830883715,
                "min_loss_delta": 0.5715736496235642 if include_failures else 0.5232124283767878,
                "mean_perplexity_delta": 12.567903220520218 if include_failures else 10.887178869354896,
                "min_perplexity_delta": 8.874624862846403 if include_failures else 8.30431945054324,
            }
        },
        "worst_case_trial_diagnostics": {
            "lowest_min_cka": {
                "scale": "350M",
                "trial_index": 0,
                "seed": 4231027559,
                "min_cka": 0.9323521445735921 if include_failures else 0.9315346048482969,
                "min_cka_layer": 15,
                "mean_cka": 0.9885425762893757,
            },
            "max_blindness_ratio": {
                "scale": "350M",
                "trial_index": 1 if include_failures else 18,
                "seed": 4231027560 if include_failures else 4231027577,
                "cka_blindness_ratio": 17.356273235045485 if include_failures else 55.03113202131845,
                "cka_blindness_worst_layer": 7,
                "inference_min_cka": 0.3333333333333333 if include_failures else 0.91,
            },
            "min_behavioral_preserved_null_access_fraction": {
                "scale": "350M",
                "trial_index": 1,
                "seed": 4231027560,
                "fraction": 0.005462362115171986 if include_failures else 0.004797473250492749,
                "layer": 8,
            },
            "largest_loss_delta": {
                "scale": "350M",
                "trial_index": 0,
                "seed": 4231027559,
                "loss_delta": 1.1657124188826196 if include_failures else 1.0990587783853212,
            },
            "largest_perplexity_delta": {
                "scale": "350M",
                "trial_index": 0,
                "seed": 4231027559,
                "perplexity_delta": 14.030564000954609 if include_failures else 13.592625490301305,
            },
        },
        "retained_failure_cases": (
            [
                {
                    "scale": "350M",
                    "trial_index": 0,
                    "seed": 4231027559,
                    "failure_modes": ["online_eval_degraded", "fourgram_degenerated"],
                    "cooccurrence_class": "cka_shift_and_inference_degraded",
                    "stop_reason": "certificate (epoch=10)",
                    "loss_delta": 1.1657124188826196,
                    "perplexity_delta": 14.030564000954609,
                    "min_cka": 0.9323521445735921,
                    "min_cka_layer": 15,
                    "online_eval_delta_correct": -1,
                    "max_4gram_repeat_delta": 0.11228338863836185,
                    "null_access_min_behavioral_preserved_fraction": 0.006602507612182352,
                    "null_access_min_behavioral_preserved_layer": 8,
                    "cka_blindness_ratio": 14.165508425614297,
                    "cka_blindness_worst_layer": 7,
                    "margin_mean_delta": 0.6687500000000001,
                }
            ]
            if include_failures
            else []
        ),
        "deleted_raw_artifacts": [f"results/{bundle_name}/350M/phase5_artifacts"],
        "deleted_phase5_adapter_payload": {
            "trial_count": 5 if include_failures else 20,
            "adapter_safetensors_total_mb": 192.79 if include_failures else 771.18,
        },
    }
    if include_contract_tags:
        verdict_payload["workflow"] = "pipeline_validation"
        verdict_payload["schema"] = "mc.pipeline_validation.family.v1"
        summary_payload["workflow"] = "pipeline_validation"
        summary_payload["schema"] = "mc.pipeline_validation.family.v1"

    result_payload = {
        "model_path": "/models/LFM2-350M",
        "dataset_path": "/data/train.jsonl",
        "eval_dataset_path": "/data/eval.jsonl",
        "trials_requested": 5 if include_failures else 20,
        "phase5_inference_enabled": True,
        "phase5_probe_count": 10,
        "phase5_probe_seed": 3475334679,
        "pass_count": 3 if include_failures else 20,
        "fail_count": 2 if include_failures else 0,
        "structural_pass_count": 5 if include_failures else 20,
        "structural_fail_count": 0,
        "inference_pass_count": 3 if include_failures else 20,
        "inference_fail_count": 2 if include_failures else 0,
        "all_passed": not include_failures,
        "min_loss_delta": 0.5715736496235642 if include_failures else 0.5232124283767878,
        "mean_loss_delta": 0.9840814639514399 if include_failures else 0.7889143830883715,
        "min_perplexity_delta": 8.874624862846403 if include_failures else 8.30431945054324,
        "mean_perplexity_delta": 12.567903220520218 if include_failures else 10.887178869354896,
        "trial_results": [
            {
                "trial_index": 0,
                "seed": 4231027559,
                "loss_delta": 1.1657124188826196 if include_failures else 1.0990587783853212,
                "perplexity_delta": 14.030564000954609 if include_failures else 13.592625490301305,
                "min_cka": 0.9323521445735921 if include_failures else 0.9377733084427855,
                "min_cka_layer": 15,
                "max_ngram_repeat_delta": 0.11228338863836185 if include_failures else -0.08935629587803506,
            }
        ],
        "counterexamples": summary_payload["retained_failure_cases"],
    }

    (bundle_dir / "verdict.json").write_text(
        json.dumps(verdict_payload) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "summary.json").write_text(
        json.dumps(summary_payload) + "\n",
        encoding="utf-8",
    )
    (scale_dir / "result.json").write_text(
        json.dumps(result_payload) + "\n",
        encoding="utf-8",
    )
    if include_report:
        (bundle_dir / "REPORT.md").write_text("# retained pipeline validation report\n", encoding="utf-8")
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


def test_load_pipeline_validation_bundle_builds_expected_sections(tmp_path: Path) -> None:
    bundle_dir = _write_pipeline_validation_fixture(tmp_path)
    service = ObservationBundleReportService()

    result = service.load(bundle_dir)

    assert result.summary["workflow"] == "pipeline_validation"
    assert result.summary["family"] == "pipeline-validation"
    assert result.summary["schema"] is None
    assert result.summary["allPass"] is False
    assert result.summary["allStructuralPass"] is True
    assert result.summary["allInferencePass"] is False
    assert result.manifest["workflow"] == "pipeline_validation"
    assert "aggregateVerdict" in result.sections
    assert "perScaleSummaries" in result.sections
    assert "worstCaseDiagnostics" in result.sections
    assert "failureCases" in result.sections
    assert "retention" in result.sections
    assert result.sections["aggregateVerdict"]["controllerMode"] == "mass_behavioral_probe"
    per_scale = result.sections["perScaleSummaries"][0]
    assert per_scale["scale"] == "350M"
    assert per_scale["inferenceFailCount"] == 2
    worst_case = result.sections["worstCaseDiagnostics"]
    assert worst_case["lowestMinCka"]["minCka"] == pytest.approx(0.9323521445735921)
    assert worst_case["maxBlindnessRatio"]["ckaBlindnessWorstLayer"] == 7
    failure_case = result.sections["failureCases"][0]
    assert failure_case["cooccurrenceClass"] == "cka_shift_and_inference_degraded"
    assert failure_case["maxNgramRepeatDelta"] == pytest.approx(0.11228338863836185)
    assert "# Pipeline Validation Family" in result.markdown
    assert "## Per-Scale Summary" in result.markdown
    assert "## Failure Cases" in result.markdown
    assert "Structural verdict: `PASS`" in result.markdown
    assert "Inference verdict: `FAIL`" in result.markdown


def test_load_pipeline_validation_all_pass_bundle_omits_failure_cases_markdown(
    tmp_path: Path,
) -> None:
    bundle_dir = _write_pipeline_validation_fixture(
        tmp_path,
        bundle_name="pipeline-validation-blindness",
        include_failures=False,
    )
    service = ObservationBundleReportService()

    result = service.load(bundle_dir)

    assert result.summary["workflow"] == "pipeline_validation"
    assert result.summary["allPass"] is True
    assert result.sections["failureCases"] == []
    assert "## Failure Cases" not in result.markdown
    assert "Composite verdict: `PASS`" in result.markdown


def test_load_pipeline_validation_rejects_missing_verdict_file(tmp_path: Path) -> None:
    bundle_dir = _write_pipeline_validation_fixture(tmp_path)
    (bundle_dir / "verdict.json").unlink()
    service = ObservationBundleReportService()

    with pytest.raises(ValueError, match="missing required files"):
        service.load(bundle_dir)


def test_load_pipeline_validation_rejects_missing_summary_file(tmp_path: Path) -> None:
    bundle_dir = _write_pipeline_validation_fixture(tmp_path)
    (bundle_dir / "summary.json").unlink()
    service = ObservationBundleReportService()

    with pytest.raises(ValueError, match="missing required files"):
        service.load(bundle_dir)


def test_load_pipeline_validation_rejects_missing_per_scale_result_file(tmp_path: Path) -> None:
    bundle_dir = _write_pipeline_validation_fixture(tmp_path)
    (bundle_dir / "350M" / "result.json").unlink()
    service = ObservationBundleReportService()

    with pytest.raises(ValueError, match="missing required files"):
        service.load(bundle_dir)


def test_load_pipeline_validation_rejects_malformed_summary_json(tmp_path: Path) -> None:
    bundle_dir = _write_pipeline_validation_fixture(tmp_path)
    (bundle_dir / "summary.json").write_text("{not-json\n", encoding="utf-8")
    service = ObservationBundleReportService()

    with pytest.raises(ValueError, match="Malformed JSON in summary.json"):
        service.load(bundle_dir)


def test_load_pipeline_validation_rejects_malformed_verdict_json(tmp_path: Path) -> None:
    bundle_dir = _write_pipeline_validation_fixture(tmp_path)
    (bundle_dir / "verdict.json").write_text("{not-json\n", encoding="utf-8")
    service = ObservationBundleReportService()

    with pytest.raises(ValueError, match="Malformed JSON in verdict.json"):
        service.load(bundle_dir)


def test_load_pipeline_validation_rejects_malformed_result_json(tmp_path: Path) -> None:
    bundle_dir = _write_pipeline_validation_fixture(tmp_path)
    (bundle_dir / "350M" / "result.json").write_text("{not-json\n", encoding="utf-8")
    service = ObservationBundleReportService()

    with pytest.raises(ValueError, match="Malformed JSON in result.json"):
        service.load(bundle_dir)


def test_load_pipeline_validation_infers_single_scale_for_legacy_summary_rows(
    tmp_path: Path,
) -> None:
    bundle_dir = _write_pipeline_validation_fixture(tmp_path)
    summary_path = bundle_dir / "summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload["worst_case_trial_diagnostics"]["lowest_min_cka"].pop("scale", None)
    summary_payload["retained_failure_cases"][0].pop("scale", None)
    summary_path.write_text(json.dumps(summary_payload) + "\n", encoding="utf-8")
    service = ObservationBundleReportService()

    result = service.load(bundle_dir)

    assert result.sections["worstCaseDiagnostics"]["lowestMinCka"]["scale"] == "350M"
    assert result.sections["failureCases"][0]["scale"] == "350M"
