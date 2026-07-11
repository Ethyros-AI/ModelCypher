from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelcypher.core.use_cases.observation_identity import validate_observation_identity

REQUIRED_OBSERVATION_BUNDLE_FILES = (
    "manifest.json",
    "summary.json",
    "variants.jsonl",
    "layer_metrics.jsonl",
    "comparisons.jsonl",
)
REQUIRED_MEASUREMENT_ATLAS_FILES = (
    "run_manifest.json",
    "summary.json",
    "variants.jsonl",
    "sequence_metrics.jsonl",
    "step_metrics.jsonl",
    "space_step_metrics.jsonl",
    "comparisons.jsonl",
    "onset_events.jsonl",
)
REQUIRED_PIPELINE_VALIDATION_FILES = (
    "verdict.json",
    "summary.json",
)
MEASUREMENT_ATLAS_MANIFEST_PREFIX = "mc.measurement_atlas.run_manifest.v"
OBSERVATION_BUNDLE_VERSION_PREFIX = "mc.analyze.bundle.v"
PIPELINE_VALIDATION_SCHEMA_PREFIX = "mc.pipeline_validation.family.v"


@dataclass(frozen=True)
class ObservationBundleReportResult:
    """Structured read-side view of a report bundle."""

    bundle_dir: str
    manifest: dict[str, Any]
    summary: dict[str, Any]
    sections: dict[str, Any]
    files: dict[str, str]
    markdown: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.summary.get("workflow"),
            "bundleDir": self.bundle_dir,
            "manifest": self.manifest,
            "summary": self.summary,
            "sections": self.sections,
            "files": self.files,
            "markdown": self.markdown,
        }


class ObservationBundleReportService:
    """Shared reporting logic for observation bundles and retained report families."""

    def load(self, bundle_dir: str | Path) -> ObservationBundleReportResult:
        resolved_dir = Path(bundle_dir).expanduser().resolve()
        if not resolved_dir.exists():
            raise ValueError(f"Report bundle directory not found: {resolved_dir}")
        if not resolved_dir.is_dir():
            raise ValueError(f"Report bundle path is not a directory: {resolved_dir}")

        bundle_family = self._detect_bundle_family(resolved_dir)
        if bundle_family == "measurement_atlas_family":
            return self._load_measurement_atlas_family_bundle(resolved_dir)
        if bundle_family == "measurement_atlas":
            return self._load_measurement_atlas_bundle(resolved_dir)
        if bundle_family == "pipeline_validation":
            return self._load_pipeline_validation_bundle(resolved_dir)
        return self._load_observation_bundle(resolved_dir)

    def _load_observation_bundle(
        self,
        bundle_dir: Path,
    ) -> ObservationBundleReportResult:
        files = self._resolve_observation_files(bundle_dir)
        manifest = self._load_json(Path(files["manifest"]))
        if manifest.get("bundleVersion") == "mc.analyze.bundle.v2":
            validate_observation_identity(manifest)
        summary = self._load_json(Path(files["summary"]))
        variant_rows = self._load_jsonl(Path(files["variants"]))
        layer_rows = self._load_jsonl(Path(files["layerMetrics"]))
        comparisons = self._load_jsonl(Path(files["comparisons"]))

        return self.build(
            bundle_dir=bundle_dir,
            manifest=manifest,
            summary=summary,
            variant_rows=variant_rows,
            layer_rows=layer_rows,
            comparisons=comparisons,
            files=files,
        )

    def _load_measurement_atlas_bundle(
        self,
        bundle_dir: Path,
    ) -> ObservationBundleReportResult:
        files = self._resolve_measurement_atlas_files(bundle_dir)
        manifest = self._load_json(Path(files["runManifest"]))
        summary = self._load_json(Path(files["summary"]))
        variant_rows = self._load_jsonl(Path(files["variants"]))
        sequence_metrics = self._load_jsonl(Path(files["sequenceMetrics"]))
        step_metrics = self._load_jsonl(Path(files["stepMetrics"]))
        space_step_metrics = self._load_jsonl(Path(files["spaceStepMetrics"]))
        comparisons = self._load_jsonl(Path(files["comparisons"]))
        onset_events = self._load_jsonl(Path(files["onsetEvents"]))
        normalized_summary = self._normalize_measurement_atlas_summary(
            summary=summary,
            manifest=manifest,
            variant_rows=variant_rows,
            sequence_metrics=sequence_metrics,
            step_metrics=step_metrics,
            space_step_metrics=space_step_metrics,
            comparisons=comparisons,
            onset_events=onset_events,
        )
        sections = {
            "surfaces": self._atlas_surface_summary(manifest),
            "studySummaries": self._atlas_study_summaries(
                sequence_metrics=sequence_metrics,
                comparisons=comparisons,
                onset_events=onset_events,
            ),
            "topSequenceShifts": self._atlas_top_sequence_shifts(comparisons),
            "locusChanges": self._atlas_locus_changes(
                comparisons=comparisons,
                sequence_metrics=sequence_metrics,
            ),
            "onsetSamples": self._atlas_onset_samples(onset_events),
            "exampleComparisons": self._atlas_example_comparisons(
                variant_rows=variant_rows,
                comparisons=comparisons,
            ),
        }
        markdown = self._build_measurement_atlas_markdown(
            summary=normalized_summary,
            sections=sections,
        )
        return ObservationBundleReportResult(
            bundle_dir=str(bundle_dir.expanduser().resolve()),
            manifest=manifest,
            summary=normalized_summary,
            sections=sections,
            files=files,
            markdown=markdown,
        )

    def _load_measurement_atlas_family_bundle(
        self,
        bundle_dir: Path,
    ) -> ObservationBundleReportResult:
        files = self._resolve_measurement_atlas_family_files(bundle_dir)
        markdown = Path(files["report"]).read_text(encoding="utf-8")
        runs = self._measurement_atlas_family_runs(bundle_dir)
        run_rows: list[dict[str, Any]] = []
        per_run_summary_files: dict[str, str] = {}
        for run_dir in runs:
            manifest_path = run_dir / "run_manifest.json"
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                raise ValueError(
                    "Measurement-atlas family run is missing required summary.json: "
                    f"{run_dir}"
                )
            manifest = self._load_json(manifest_path)
            summary = self._load_json(summary_path)
            run_id = str(manifest.get("runId") or summary.get("runId") or run_dir.name)
            per_run_summary_files[run_id] = str(summary_path.resolve())
            run_rows.append(
                {
                    "runId": run_id,
                    "directory": str(run_dir.resolve()),
                    "schema": manifest.get("schema"),
                    "linkedBlocker": manifest.get("linkedBlocker"),
                    "studyCount": summary.get("studyCount"),
                    "variantCount": summary.get("variantCount"),
                    "comparisonCount": summary.get("comparisonCount"),
                    "onsetEventCount": summary.get("onsetEventCount"),
                    "errorCount": summary.get("errorCount"),
                    "surfaces": self._atlas_surface_summary(manifest),
                }
            )

        summary = {
            "workflow": "measurement_atlas_family",
            "family": bundle_dir.name,
            "runCount": len(run_rows),
            "linkedBlockers": sorted(
                {
                    str(row["linkedBlocker"])
                    for row in run_rows
                    if row.get("linkedBlocker") is not None
                }
            ),
            "errorCount": sum(
                int(row["errorCount"])
                for row in run_rows
                if row.get("errorCount") is not None
            ),
        }
        sections = {
            "runs": run_rows,
        }
        manifest = {
            "workflow": "measurement_atlas_family",
            "family": bundle_dir.name,
            "runCount": len(run_rows),
            "report": files["report"],
            "perRunSummaryFiles": per_run_summary_files,
        }
        return ObservationBundleReportResult(
            bundle_dir=str(bundle_dir.expanduser().resolve()),
            manifest=manifest,
            summary=summary,
            sections=sections,
            files=files,
            markdown=markdown,
        )

    def _load_pipeline_validation_bundle(
        self,
        bundle_dir: Path,
    ) -> ObservationBundleReportResult:
        files = self._resolve_pipeline_validation_files(bundle_dir)
        verdict = self._load_json(Path(files["verdict"]))
        summary = self._load_json(Path(files["summary"]))
        per_scale_results = {
            key.split("result:", 1)[1]: self._load_json(Path(path))
            for key, path in files.items()
            if key.startswith("result:")
        }
        normalized_summary = self._normalize_pipeline_validation_summary(
            summary=summary,
            verdict=verdict,
            per_scale_results=per_scale_results,
            bundle_dir=bundle_dir,
        )
        sections = {
            "aggregateVerdict": self._pipeline_validation_aggregate_verdict(
                summary=summary,
                verdict=verdict,
                normalized_summary=normalized_summary,
            ),
            "perScaleSummaries": self._pipeline_validation_per_scale_summaries(
                summary=summary,
                verdict=verdict,
                per_scale_results=per_scale_results,
                normalized_summary=normalized_summary,
            ),
            "worstCaseDiagnostics": self._pipeline_validation_worst_case_diagnostics(
                summary=summary,
                per_scale_results=per_scale_results,
            ),
            "failureCases": self._pipeline_validation_failure_cases(
                summary=summary,
                per_scale_results=per_scale_results,
            ),
            "retention": self._pipeline_validation_retention(summary=summary),
        }
        manifest = self._pipeline_validation_manifest(
            normalized_summary=normalized_summary,
            summary=summary,
            verdict=verdict,
            files=files,
        )
        markdown = self._build_pipeline_validation_markdown(
            summary=normalized_summary,
            sections=sections,
        )
        return ObservationBundleReportResult(
            bundle_dir=str(bundle_dir.expanduser().resolve()),
            manifest=manifest,
            summary=normalized_summary,
            sections=sections,
            files=files,
            markdown=markdown,
        )

    def _detect_bundle_family(self, bundle_dir: Path) -> str:
        atlas_manifest_path = bundle_dir / "run_manifest.json"
        observation_manifest_path = bundle_dir / "manifest.json"
        verdict_path = bundle_dir / "verdict.json"
        summary_path = bundle_dir / "summary.json"

        if atlas_manifest_path.exists():
            manifest = self._load_json(atlas_manifest_path)
            schema = str(manifest.get("schema", "")).strip()
            if schema.startswith(MEASUREMENT_ATLAS_MANIFEST_PREFIX):
                return "measurement_atlas"

        if self._measurement_atlas_family_paths(bundle_dir):
            return "measurement_atlas_family"

        if observation_manifest_path.exists():
            manifest = self._load_json(observation_manifest_path)
            bundle_version = str(manifest.get("bundleVersion", "")).strip()
            if bundle_version.startswith(OBSERVATION_BUNDLE_VERSION_PREFIX):
                return "observation"

        if (
            verdict_path.exists()
            or summary_path.exists()
            or self._pipeline_validation_result_paths(bundle_dir)
        ):
            return "pipeline_validation"

        if atlas_manifest_path.exists():
            return "measurement_atlas"
        if observation_manifest_path.exists():
            return "observation"

        raise ValueError(
            "Report bundle directory must contain manifest.json, run_manifest.json, "
            "a measurement-atlas family REPORT.md with atlas run subdirectories, "
            "or pipeline-validation files (verdict.json, summary.json, <scale>/result.json): "
            f"{bundle_dir}"
        )

    def build(
        self,
        *,
        bundle_dir: str | Path,
        manifest: dict[str, Any],
        summary: dict[str, Any],
        variant_rows: list[dict[str, Any]],
        layer_rows: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        files: dict[str, str],
    ) -> ObservationBundleReportResult:
        normalized_summary = self._normalize_summary(
            summary=summary,
            variant_rows=variant_rows,
            layer_rows=layer_rows,
            comparisons=comparisons,
            manifest=manifest,
        )
        sections = {
            "measurementIdentity": self._measurement_identity(manifest),
            "observedSpaces": self._space_summary(layer_rows),
            "topScalarShifts": self._top_scalar_shifts(comparisons),
            "topLayerShifts": self._top_layer_shifts(comparisons),
            "errorSamples": self._error_samples(variant_rows),
            "variantSamples": self._variant_samples(variant_rows),
            "comparisonSamples": self._comparison_samples(comparisons),
        }
        markdown = self._build_markdown(
            summary=normalized_summary,
            sections=sections,
        )
        return ObservationBundleReportResult(
            bundle_dir=str(Path(bundle_dir).expanduser().resolve()),
            manifest=manifest,
            summary=normalized_summary,
            sections=sections,
            files=files,
            markdown=markdown,
        )

    @staticmethod
    def _resolve_observation_files(bundle_dir: Path) -> dict[str, str]:
        file_map = {
            "manifest": bundle_dir / "manifest.json",
            "summary": bundle_dir / "summary.json",
            "report": bundle_dir / "REPORT.md",
            "variants": bundle_dir / "variants.jsonl",
            "layerMetrics": bundle_dir / "layer_metrics.jsonl",
            "comparisons": bundle_dir / "comparisons.jsonl",
        }
        missing = [
            file_path.name
            for key, file_path in file_map.items()
            if key != "report" and not file_path.exists()
        ]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "Report bundle is missing required files for observation bundles: "
                f"{missing_text}"
            )
        return {key: str(path.resolve()) for key, path in file_map.items() if path.exists()}

    @staticmethod
    def _resolve_measurement_atlas_files(bundle_dir: Path) -> dict[str, str]:
        file_map = {
            "runManifest": bundle_dir / "run_manifest.json",
            "summary": bundle_dir / "summary.json",
            "report": bundle_dir / "REPORT.md",
            "variants": bundle_dir / "variants.jsonl",
            "sequenceMetrics": bundle_dir / "sequence_metrics.jsonl",
            "stepMetrics": bundle_dir / "step_metrics.jsonl",
            "spaceStepMetrics": bundle_dir / "space_step_metrics.jsonl",
            "comparisons": bundle_dir / "comparisons.jsonl",
            "onsetEvents": bundle_dir / "onset_events.jsonl",
        }
        missing = [
            file_path.name
            for key, file_path in file_map.items()
            if key != "report" and not file_path.exists()
        ]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "Report bundle is missing required files for measurement-atlas bundles: "
                f"{missing_text}"
            )
        return {key: str(path.resolve()) for key, path in file_map.items() if path.exists()}

    @classmethod
    def _resolve_measurement_atlas_family_files(cls, bundle_dir: Path) -> dict[str, str]:
        report_path = bundle_dir / "REPORT.md"
        if not report_path.exists():
            raise ValueError(
                "Report bundle is missing required files for measurement-atlas families: "
                "REPORT.md"
            )
        runs = cls._measurement_atlas_family_runs(bundle_dir)
        if not runs:
            raise ValueError(
                "Report bundle is missing required measurement-atlas run directories: "
                "*/run_manifest.json"
            )
        files = {"report": str(report_path.resolve())}
        for run_dir in runs:
            run_key = run_dir.name
            files[f"runManifest:{run_key}"] = str((run_dir / "run_manifest.json").resolve())
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                files[f"summary:{run_key}"] = str(summary_path.resolve())
        return files

    @classmethod
    def _measurement_atlas_family_paths(cls, bundle_dir: Path) -> list[Path]:
        if not (bundle_dir / "REPORT.md").exists():
            return []
        return [
            child
            for child in sorted(bundle_dir.iterdir(), key=lambda path: path.name)
            if child.is_dir() and (child / "run_manifest.json").exists()
        ]

    @classmethod
    def _measurement_atlas_family_runs(cls, bundle_dir: Path) -> list[Path]:
        runs: list[Path] = []
        for child in cls._measurement_atlas_family_paths(bundle_dir):
            manifest = cls._load_json(child / "run_manifest.json")
            schema = str(manifest.get("schema", "")).strip()
            if schema.startswith(MEASUREMENT_ATLAS_MANIFEST_PREFIX):
                runs.append(child)
        return runs

    @staticmethod
    def _pipeline_validation_result_paths(bundle_dir: Path) -> dict[str, Path]:
        return {
            child.name: child / "result.json"
            for child in sorted(bundle_dir.iterdir(), key=lambda path: path.name)
            if child.is_dir() and (child / "result.json").exists()
        }

    @classmethod
    def _resolve_pipeline_validation_files(cls, bundle_dir: Path) -> dict[str, str]:
        file_map = {
            "verdict": bundle_dir / "verdict.json",
            "summary": bundle_dir / "summary.json",
            "report": bundle_dir / "REPORT.md",
        }
        result_paths = cls._pipeline_validation_result_paths(bundle_dir)
        missing = [
            file_name
            for file_name in REQUIRED_PIPELINE_VALIDATION_FILES
            if not (bundle_dir / file_name).exists()
        ]
        if not result_paths:
            missing.append("<scale>/result.json")
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "Report bundle is missing required files for pipeline-validation families: "
                f"{missing_text}"
            )
        resolved = {key: str(path.resolve()) for key, path in file_map.items() if path.exists()}
        for scale, path in result_paths.items():
            resolved[f"result:{scale}"] = str(path.resolve())
        return resolved

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path.name}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name} must contain a JSON object.")
        return payload

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL in {path.name} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path.name} must contain JSON objects on each non-empty line."
                )
            rows.append(payload)
        return rows

    @staticmethod
    def _normalize_summary(
        *,
        summary: dict[str, Any],
        variant_rows: list[dict[str, Any]],
        layer_rows: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(summary)
        normalized["variantCount"] = len(variant_rows)
        normalized["layerMetricCount"] = len(layer_rows)
        normalized["comparisonCount"] = len(comparisons)
        normalized["errorCount"] = sum(len(row.get("errors", [])) for row in variant_rows)
        normalized.setdefault(
            "spaces",
            sorted(
                {
                    str(row.get("space")).strip()
                    for row in layer_rows
                    if str(row.get("space", "")).strip()
                }
            ),
        )
        prompt_manifest = manifest.get("promptFamilyManifest")
        if isinstance(prompt_manifest, dict):
            normalized.setdefault("manifestName", prompt_manifest.get("name"))
        return normalized

    @staticmethod
    def _measurement_identity(manifest: dict[str, Any]) -> dict[str, Any]:
        context = manifest.get("contextState", {})
        precision = manifest.get("precisionState", {})
        operator = manifest.get("measurementOperator", {})
        digest = context.get("promptFamilyDigest", {})
        backend = precision.get("backend", {})
        targets = precision.get("targets", [])
        return {
            "contextDigest": digest.get("value"),
            "promptFamilyName": context.get("promptFamilyName"),
            "precisionBackend": backend.get("type"),
            "allTargetPrecisionsDeclared": precision.get("allTargetsDeclared"),
            "targetPrecisionDeclarationsMatch": precision.get("declarationsMatch"),
            "targetDeclarations": [
                {
                    "targetLabel": row.get("targetLabel"),
                    "declared": row.get("declared"),
                }
                for row in targets
                if isinstance(row, dict)
            ],
            "measurementOperatorId": operator.get("id"),
        }

    @staticmethod
    def _space_summary(layer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary_by_space: dict[str, dict[str, Any]] = {}
        for row in layer_rows:
            space = str(row.get("space", "")).strip()
            if not space:
                continue
            bucket = summary_by_space.setdefault(
                space,
                {
                    "space": space,
                    "rowCount": 0,
                    "layers": set(),
                    "vectorNorms": [],
                },
            )
            bucket["rowCount"] += 1
            layer = row.get("layer")
            if layer is not None:
                bucket["layers"].add(int(layer))
            vector_norm = row.get("vectorNorm")
            if vector_norm is not None:
                bucket["vectorNorms"].append(float(vector_norm))

        results = []
        for space, bucket in sorted(summary_by_space.items()):
            vector_norms = bucket["vectorNorms"]
            mean_vector_norm = (
                sum(vector_norms) / len(vector_norms) if vector_norms else None
            )
            results.append(
                {
                    "space": space,
                    "rowCount": bucket["rowCount"],
                    "layerCount": len(bucket["layers"]),
                    "meanVectorNorm": mean_vector_norm,
                }
            )
        return results

    def _top_scalar_shifts(
        self,
        comparisons: list[dict[str, Any]],
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for comparison in comparisons:
            for metric, delta in comparison.get("scalarDeltas", {}).items():
                if delta is None:
                    continue
                rows.append(
                    {
                        "label": self._comparison_label(comparison),
                        "metric": metric,
                        "delta": float(delta),
                    }
                )
        rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
        return rows[:limit]

    def _top_layer_shifts(
        self,
        comparisons: list[dict[str, Any]],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for comparison in comparisons:
            for metric, deltas in comparison.get("perLayerDeltas", {}).items():
                for delta_row in deltas:
                    delta = delta_row.get("delta")
                    layer = delta_row.get("layer")
                    if delta is None or layer is None:
                        continue
                    rows.append(
                        {
                            "label": self._comparison_label(comparison),
                            "metric": metric,
                            "layer": int(layer),
                            "delta": float(delta),
                        }
                    )
        rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
        return rows[:limit]

    @staticmethod
    def _error_samples(
        variant_rows: list[dict[str, Any]],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        rows = []
        for row in variant_rows:
            errors = row.get("errors")
            if not errors:
                continue
            rows.append(
                {
                    "targetLabel": row.get("targetLabel"),
                    "caseId": row.get("caseId"),
                    "variantId": row.get("variantId"),
                    "errors": list(errors),
                }
            )
        return rows[:limit]

    @staticmethod
    def _variant_samples(
        variant_rows: list[dict[str, Any]],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        rows = []
        for row in variant_rows[:limit]:
            summary_metrics = row.get("summaryMetrics", {})
            rows.append(
                {
                    "targetLabel": row.get("targetLabel"),
                    "caseId": row.get("caseId"),
                    "variantId": row.get("variantId"),
                    "meanEntropy": summary_metrics.get("meanEntropy"),
                    "meanGeodesicDeviation": summary_metrics.get("meanGeodesicDeviation"),
                    "meanCurvature": summary_metrics.get("meanCurvature"),
                }
            )
        return rows

    @staticmethod
    def _comparison_samples(
        comparisons: list[dict[str, Any]],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return comparisons[:limit]

    def _comparison_label(self, comparison: dict[str, Any]) -> str:
        parts = [
            str(comparison.get("mode", "comparison")),
            str(comparison.get("from", "?")),
            "->",
            str(comparison.get("to", "?")),
            "/",
            str(comparison.get("caseId", "?")),
        ]
        variant_id = comparison.get("variantId")
        if variant_id:
            parts.extend(["/", str(variant_id)])
        target_label = comparison.get("targetLabel")
        if target_label:
            parts.extend(["/", str(target_label)])
        return " ".join(parts)

    def _build_markdown(
        self,
        *,
        summary: dict[str, Any],
        sections: dict[str, Any],
    ) -> str:
        lines = [
            "# Observation Bundle",
            "",
            f"- Workflow: `{summary.get('workflow')}`",
            f"- Targets: {summary.get('targetCount')}",
            f"- Variants: {summary.get('variantCount')}",
            f"- Comparisons: {summary.get('comparisonCount')}",
            f"- Measurement errors: {summary.get('errorCount')}",
            f"- Spaces: {', '.join(summary.get('spaces', []))}",
        ]

        identity = sections.get("measurementIdentity", {})
        if identity.get("measurementOperatorId"):
            lines.extend(
                [
                    "",
                    "## Measurement Identity",
                    "",
                    f"- Context digest: `{identity.get('contextDigest')}`",
                    f"- Prompt family: `{identity.get('promptFamilyName')}`",
                    f"- Operator: `{identity.get('measurementOperatorId')}`",
                    f"- Backend: `{identity.get('precisionBackend')}`",
                    "- All target precisions declared: "
                    f"`{identity.get('allTargetPrecisionsDeclared')}`",
                    "- Target precision declarations match: "
                    f"`{identity.get('targetPrecisionDeclarationsMatch')}`",
                ]
            )

        lines.extend(
            [
                "",
                "## Means",
                "",
                f"- Mean prompt tokens: {summary.get('meanPromptTokenCount')}",
                f"- Mean response tokens: {summary.get('meanResponseTokenCount')}",
                f"- Mean entropy: {summary.get('meanEntropy')}",
                f"- Mean geodesic deviation: {summary.get('meanGeodesicDeviation')}",
                f"- Mean curvature: {summary.get('meanCurvature')}",
            ]
        )

        observed_spaces = sections.get("observedSpaces", [])
        if observed_spaces:
            lines.extend(["", "## Observed Spaces", ""])
            for row in observed_spaces:
                lines.append(
                    f"- `{row['space']}`: rows={row['rowCount']}, "
                    f"layers={row['layerCount']}, mean_vector_norm={row['meanVectorNorm']}"
                )

        top_scalar_shifts = sections.get("topScalarShifts", [])
        if top_scalar_shifts:
            lines.extend(["", "## Largest Scalar Shifts", ""])
            for row in top_scalar_shifts:
                lines.append(
                    f"- `{row['label']}`: `{row['metric']}` delta={row['delta']}"
                )

        top_layer_shifts = sections.get("topLayerShifts", [])
        if top_layer_shifts:
            lines.extend(["", "## Most Shifted Layers", ""])
            for row in top_layer_shifts:
                lines.append(
                    f"- `{row['label']}`: layer={row['layer']}, "
                    f"`{row['metric']}` delta={row['delta']}"
                )

        variant_samples = sections.get("variantSamples", [])
        if variant_samples:
            lines.extend(["", "## Variants", ""])
            for row in variant_samples:
                lines.append(
                    f"- `{row['targetLabel']}` / `{row['caseId']}` / `{row['variantId']}`: "
                    f"entropy={row['meanEntropy']}, "
                    f"geodesic={row['meanGeodesicDeviation']}, "
                    f"curvature={row['meanCurvature']}"
                )

        comparison_samples = sections.get("comparisonSamples", [])
        if comparison_samples:
            lines.extend(["", "## Comparisons", ""])
            for row in comparison_samples:
                deltas = row.get("scalarDeltas", {})
                lines.append(
                    f"- `{row.get('from')}` -> `{row.get('to')}` on `{row.get('caseId')}`: "
                    f"entropy_delta={deltas.get('meanEntropy')}, "
                    f"geodesic_delta={deltas.get('meanGeodesicDeviation')}, "
                    f"curvature_delta={deltas.get('meanCurvature')}"
                )

        error_samples = sections.get("errorSamples", [])
        if error_samples:
            lines.extend(["", "## Measurement Errors", ""])
            for row in error_samples:
                lines.append(
                    f"- `{row['targetLabel']}` / `{row['caseId']}` / `{row['variantId']}`: "
                    + "; ".join(row["errors"])
                )

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _normalize_pipeline_validation_summary(
        *,
        summary: dict[str, Any],
        verdict: dict[str, Any],
        per_scale_results: dict[str, dict[str, Any]],
        bundle_dir: Path,
    ) -> dict[str, Any]:
        aggregate = summary.get("aggregate_verdict")
        if not isinstance(aggregate, dict):
            aggregate = verdict
        scales = aggregate.get("scales")
        if not isinstance(scales, list) or not scales:
            scales = sorted(per_scale_results.keys())

        normalized = dict(summary)
        normalized["workflow"] = "pipeline_validation"
        normalized["schema"] = (
            str(summary.get("schema")).strip()
            if summary.get("schema") is not None
            else (
                str(verdict.get("schema")).strip()
                if verdict.get("schema") is not None
                else None
            )
        )
        normalized["family"] = str(summary.get("family") or bundle_dir.name)
        normalized["status"] = str(summary.get("status") or "n/a")
        normalized["scales"] = list(scales)
        normalized["trialsPerModel"] = aggregate.get("trials_per_model")
        normalized["allPass"] = aggregate.get("all_pass")
        normalized["allStructuralPass"] = aggregate.get("all_structural_pass")
        normalized["allInferencePass"] = aggregate.get("all_inference_pass")
        normalized["timestamp"] = aggregate.get("timestamp")
        normalized["gitHash"] = aggregate.get("git_hash")
        normalized["controllerMode"] = aggregate.get("controller_mode")
        normalized["optimizerResearchMode"] = aggregate.get("optimizer_research_mode")
        normalized["benchmarkSuite"] = aggregate.get("benchmark_suite")
        return normalized

    def _pipeline_validation_aggregate_verdict(
        self,
        *,
        summary: dict[str, Any],
        verdict: dict[str, Any],
        normalized_summary: dict[str, Any],
    ) -> dict[str, Any]:
        aggregate = summary.get("aggregate_verdict")
        if not isinstance(aggregate, dict):
            aggregate = verdict
        return {
            "timestamp": aggregate.get("timestamp") or normalized_summary.get("timestamp"),
            "gitHash": aggregate.get("git_hash") or normalized_summary.get("gitHash"),
            "trialsPerModel": (
                aggregate.get("trials_per_model") or normalized_summary.get("trialsPerModel")
            ),
            "controllerMode": (
                aggregate.get("controller_mode") or normalized_summary.get("controllerMode")
            ),
            "optimizerResearchMode": (
                aggregate.get("optimizer_research_mode")
                or normalized_summary.get("optimizerResearchMode")
            ),
            "benchmarkSuite": (
                aggregate.get("benchmark_suite") or normalized_summary.get("benchmarkSuite")
            ),
            "scales": list(
                aggregate.get("scales")
                or normalized_summary.get("scales")
                or []
            ),
            "allPass": aggregate.get("all_pass"),
            "allStructuralPass": aggregate.get("all_structural_pass"),
            "allInferencePass": aggregate.get("all_inference_pass"),
        }

    @staticmethod
    def _pipeline_validation_per_scale_summaries(
        *,
        summary: dict[str, Any],
        verdict: dict[str, Any],
        per_scale_results: dict[str, dict[str, Any]],
        normalized_summary: dict[str, Any],
    ) -> list[dict[str, Any]]:
        summary_by_scale = summary.get("per_scale_summary", {})
        if not isinstance(summary_by_scale, dict):
            summary_by_scale = {}
        verdict_by_scale = verdict.get("per_scale", {})
        if not isinstance(verdict_by_scale, dict):
            verdict_by_scale = {}

        rows: list[dict[str, Any]] = []
        for scale in normalized_summary.get("scales", []):
            summary_row = summary_by_scale.get(scale, {})
            if not isinstance(summary_row, dict):
                summary_row = {}
            verdict_row = verdict_by_scale.get(scale, {})
            if not isinstance(verdict_row, dict):
                verdict_row = {}
            result_row = per_scale_results.get(scale, {})
            rows.append(
                {
                    "scale": scale,
                    "allPassed": verdict_row.get(
                        "all_passed",
                        result_row.get("all_passed"),
                    ),
                    "passCount": verdict_row.get("pass_count", result_row.get("pass_count")),
                    "failCount": verdict_row.get("fail_count", result_row.get("fail_count")),
                    "structuralPassCount": verdict_row.get(
                        "structural_pass_count",
                        result_row.get("structural_pass_count", summary_row.get("structural_pass_count")),
                    ),
                    "structuralFailCount": verdict_row.get(
                        "structural_fail_count",
                        result_row.get("structural_fail_count", summary_row.get("structural_fail_count")),
                    ),
                    "inferencePassCount": verdict_row.get(
                        "inference_pass_count",
                        result_row.get("inference_pass_count", summary_row.get("inference_pass_count")),
                    ),
                    "inferenceFailCount": verdict_row.get(
                        "inference_fail_count",
                        result_row.get("inference_fail_count", summary_row.get("inference_fail_count")),
                    ),
                    "phase5InferenceEnabled": verdict_row.get(
                        "phase5_inference_enabled",
                        result_row.get("phase5_inference_enabled"),
                    ),
                    "phase5ProbeCount": result_row.get(
                        "phase5_probe_count",
                        summary_row.get("phase5_probe_count"),
                    ),
                    "phase5ProbeSeed": result_row.get(
                        "phase5_probe_seed",
                        summary_row.get("phase5_probe_seed"),
                    ),
                    "meanLossDelta": result_row.get(
                        "mean_loss_delta",
                        summary_row.get("mean_loss_delta"),
                    ),
                    "minLossDelta": result_row.get(
                        "min_loss_delta",
                        summary_row.get("min_loss_delta"),
                    ),
                    "meanPerplexityDelta": result_row.get(
                        "mean_perplexity_delta",
                        summary_row.get("mean_perplexity_delta"),
                    ),
                    "minPerplexityDelta": result_row.get(
                        "min_perplexity_delta",
                        summary_row.get("min_perplexity_delta"),
                    ),
                    "modelPath": result_row.get("model_path"),
                    "datasetPath": result_row.get("dataset_path"),
                    "evalDatasetPath": result_row.get("eval_dataset_path"),
                    "trialsRequested": result_row.get("trials_requested"),
                    "error": verdict_row.get("error", result_row.get("error")),
                }
            )
        return rows

    def _pipeline_validation_worst_case_diagnostics(
        self,
        *,
        summary: dict[str, Any],
        per_scale_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        default_scale = next(iter(per_scale_results)) if len(per_scale_results) == 1 else None
        diagnostics = summary.get("worst_case_trial_diagnostics")
        if isinstance(diagnostics, dict) and diagnostics:
            normalized = {
                "lowestMinCka": self._pipeline_validation_metric_row(
                    diagnostics.get("lowest_min_cka"),
                    value_key="min_cka",
                    layer_key="min_cka_layer",
                    output_value_key="minCka",
                    output_layer_key="minCkaLayer",
                ),
                "maxBlindnessRatio": self._pipeline_validation_metric_row(
                    diagnostics.get("max_blindness_ratio"),
                    value_key="cka_blindness_ratio",
                    layer_key="cka_blindness_worst_layer",
                    output_value_key="ckaBlindnessRatio",
                    output_layer_key="ckaBlindnessWorstLayer",
                    extra_keys=("inference_min_cka",),
                ),
                "minBehavioralPreservedNullAccessFraction": self._pipeline_validation_metric_row(
                    diagnostics.get("min_behavioral_preserved_null_access_fraction"),
                    value_key="fraction",
                    layer_key="layer",
                    output_value_key="fraction",
                    output_layer_key="layer",
                ),
                "largestLossDelta": self._pipeline_validation_metric_row(
                    diagnostics.get("largest_loss_delta"),
                    value_key="loss_delta",
                    layer_key=None,
                    output_value_key="lossDelta",
                    output_layer_key=None,
                ),
                "largestPerplexityDelta": self._pipeline_validation_metric_row(
                    diagnostics.get("largest_perplexity_delta"),
                    value_key="perplexity_delta",
                    layer_key=None,
                    output_value_key="perplexityDelta",
                    output_layer_key=None,
                ),
            }
            for row in normalized.values():
                if isinstance(row, dict) and row.get("scale") is None and default_scale is not None:
                    row["scale"] = default_scale
            optional = self._pipeline_validation_optional_diagnostic_rows(per_scale_results)
            normalized.update(optional)
            return normalized

        return self._pipeline_validation_fallback_worst_case_diagnostics(per_scale_results)

    def _pipeline_validation_failure_cases(
        self,
        *,
        summary: dict[str, Any],
        per_scale_results: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        default_scale = next(iter(per_scale_results)) if len(per_scale_results) == 1 else None
        summary_cases = summary.get("retained_failure_cases")
        if isinstance(summary_cases, list):
            return [
                self._pipeline_validation_failure_case_row(case, scale=default_scale)
                for case in summary_cases
            ]

        rows: list[dict[str, Any]] = []
        for scale, result in per_scale_results.items():
            counterexamples = result.get("counterexamples", [])
            if not isinstance(counterexamples, list):
                continue
            for case in counterexamples:
                if not isinstance(case, dict):
                    continue
                rows.append(self._pipeline_validation_failure_case_row(case, scale=scale))
        return rows

    @staticmethod
    def _pipeline_validation_retention(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": summary.get("status"),
            "retainedArtifacts": list(summary.get("retained_artifacts", [])),
            "deletedRawArtifacts": list(summary.get("deleted_raw_artifacts", [])),
            "deletedPhase5AdapterPayload": summary.get("deleted_phase5_adapter_payload"),
        }

    @staticmethod
    def _pipeline_validation_manifest(
        *,
        normalized_summary: dict[str, Any],
        summary: dict[str, Any],
        verdict: dict[str, Any],
        files: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "workflow": "pipeline_validation",
            "schema": normalized_summary.get("schema"),
            "family": normalized_summary.get("family"),
            "status": normalized_summary.get("status"),
            "scales": list(normalized_summary.get("scales", [])),
            "summarySchema": summary.get("schema"),
            "verdictSchema": verdict.get("schema"),
            "retainedArtifacts": list(summary.get("retained_artifacts", [])),
            "perScaleResultFiles": {
                key.split("result:", 1)[1]: path
                for key, path in files.items()
                if key.startswith("result:")
            },
        }

    def _build_pipeline_validation_markdown(
        self,
        *,
        summary: dict[str, Any],
        sections: dict[str, Any],
    ) -> str:
        aggregate_verdict = sections.get("aggregateVerdict", {})
        per_scale_summaries = sections.get("perScaleSummaries", [])
        worst_case_diagnostics = sections.get("worstCaseDiagnostics", {})
        failure_cases = sections.get("failureCases", [])
        retention = sections.get("retention", {})

        lines = [
            "# Pipeline Validation Family",
            "",
            f"- Workflow: `{summary.get('workflow')}`",
            f"- Family: `{summary.get('family')}`",
            f"- Status: `{summary.get('status')}`",
            f"- Trials per model: {summary.get('trialsPerModel')}",
            f"- Scales: {', '.join(summary.get('scales', []))}",
            f"- Structural verdict: `{self._pipeline_validation_verdict_text(summary.get('allStructuralPass'))}`",
            f"- Inference verdict: `{self._pipeline_validation_verdict_text(summary.get('allInferencePass'))}`",
            f"- Composite verdict: `{self._pipeline_validation_verdict_text(summary.get('allPass'))}`",
        ]

        if aggregate_verdict.get("controllerMode") or aggregate_verdict.get("optimizerResearchMode"):
            lines.extend(
                [
                    f"- Controller mode: `{aggregate_verdict.get('controllerMode') or 'n/a'}`",
                    f"- Optimizer mode: `{aggregate_verdict.get('optimizerResearchMode') or 'n/a'}`",
                ]
            )
        if aggregate_verdict.get("benchmarkSuite") is not None:
            lines.append(f"- Benchmark suite: `{aggregate_verdict.get('benchmarkSuite') or 'none'}`")

        if per_scale_summaries:
            lines.extend(
                [
                    "",
                    "## Per-Scale Summary",
                    "",
                    "| Scale | Structural | Inference | Composite | Structural pass/fail | Inference pass/fail | Composite pass/fail | Mean loss delta | Mean perplexity delta |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for row in per_scale_summaries:
                structural_status = self._pipeline_validation_verdict_text(
                    row.get("structuralFailCount") == 0 if row.get("structuralFailCount") is not None else None
                )
                inference_status = self._pipeline_validation_verdict_text(
                    row.get("inferenceFailCount") == 0 if row.get("inferenceFailCount") is not None else None
                )
                composite_status = self._pipeline_validation_verdict_text(row.get("allPassed"))
                lines.append(
                    f"| {row.get('scale')} | {structural_status} | {inference_status} | {composite_status} "
                    f"| {row.get('structuralPassCount')}/{row.get('structuralFailCount')} "
                    f"| {row.get('inferencePassCount')}/{row.get('inferenceFailCount')} "
                    f"| {row.get('passCount')}/{row.get('failCount')} "
                    f"| {self._pipeline_validation_number_text(row.get('meanLossDelta'))} "
                    f"| {self._pipeline_validation_number_text(row.get('meanPerplexityDelta'))} |"
                )

        if worst_case_diagnostics:
            lines.extend(["", "## Worst-Case Diagnostics", ""])
            for label, row in (
                ("Lowest min CKA", worst_case_diagnostics.get("lowestMinCka")),
                ("Max blindness ratio", worst_case_diagnostics.get("maxBlindnessRatio")),
                (
                    "Minimum null-access preserved fraction",
                    worst_case_diagnostics.get("minBehavioralPreservedNullAccessFraction"),
                ),
                ("Largest loss delta", worst_case_diagnostics.get("largestLossDelta")),
                (
                    "Largest perplexity delta",
                    worst_case_diagnostics.get("largestPerplexityDelta"),
                ),
                ("Largest repeat delta", worst_case_diagnostics.get("largestRepeatDelta")),
                ("Largest margin mean delta", worst_case_diagnostics.get("largestMarginMeanDelta")),
            ):
                if not row:
                    continue
                lines.append(f"- {label}: {self._pipeline_validation_diagnostic_text(row)}")

        if failure_cases:
            lines.extend(["", "## Failure Cases", ""])
            for row in failure_cases:
                lines.append(
                    f"- `{row.get('scale') or 'n/a'}` trial `{row.get('trialIndex')}` / seed `{row.get('seed')}`: "
                    f"failure_modes={', '.join(row.get('failureModes', [])) or 'n/a'}; "
                    f"cooccurrence={row.get('cooccurrenceClass') or 'n/a'}; "
                    f"stop_reason={row.get('stopReason') or 'n/a'}; "
                    f"min_cka={self._pipeline_validation_number_text(row.get('minCka'))} "
                    f"(layer={self._pipeline_validation_number_text(row.get('minCkaLayer'))}); "
                    f"blindness={self._pipeline_validation_number_text(row.get('ckaBlindnessRatio'))} "
                    f"(layer={self._pipeline_validation_number_text(row.get('ckaBlindnessWorstLayer'))}); "
                    f"null_access={self._pipeline_validation_number_text(row.get('nullAccessMinBehavioralPreservedFraction'))} "
                    f"(layer={self._pipeline_validation_number_text(row.get('nullAccessMinBehavioralPreservedLayer'))}); "
                    f"online_eval_delta={self._pipeline_validation_number_text(row.get('onlineEvalDeltaCorrect'))}; "
                    f"repeat_delta={self._pipeline_validation_number_text(row.get('maxNgramRepeatDelta'))}"
                )

        lines.extend(["", "## Retention", ""])
        retained_artifacts = retention.get("retainedArtifacts") or []
        deleted_raw_artifacts = retention.get("deletedRawArtifacts") or []
        if retained_artifacts:
            lines.append(f"- Retained artifacts: {', '.join(retained_artifacts)}")
        if deleted_raw_artifacts:
            lines.append(f"- Deleted raw artifacts: {', '.join(deleted_raw_artifacts)}")
        deleted_payload = retention.get("deletedPhase5AdapterPayload")
        if isinstance(deleted_payload, dict) and deleted_payload:
            lines.append(
                "- Deleted phase-5 adapter payload: "
                f"trials={deleted_payload.get('trial_count')}, "
                f"total_mb={self._pipeline_validation_number_text(deleted_payload.get('adapter_safetensors_total_mb'))}"
            )

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _pipeline_validation_metric_row(
        row: Any,
        *,
        value_key: str,
        layer_key: str | None,
        output_value_key: str,
        output_layer_key: str | None,
        extra_keys: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        normalized = {
            "scale": row.get("scale"),
            "trialIndex": row.get("trial_index"),
            "seed": row.get("seed"),
            output_value_key: row.get(value_key),
        }
        if layer_key and output_layer_key:
            normalized[output_layer_key] = row.get(layer_key)
        for key in extra_keys:
            camel_key = {
                "inference_min_cka": "inferenceMinCka",
            }.get(key, key)
            normalized[camel_key] = row.get(key)
        return normalized

    def _pipeline_validation_optional_diagnostic_rows(
        self,
        per_scale_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        repeat_rows = self._pipeline_validation_trials_with_metric(
            per_scale_results=per_scale_results,
            metric_key="max_ngram_repeat_delta",
        )
        margin_rows = self._pipeline_validation_counterexamples_with_metric(
            per_scale_results=per_scale_results,
            metric_key="margin_mean_delta",
        )
        diagnostics: dict[str, Any] = {}
        if repeat_rows:
            best = max(repeat_rows, key=lambda row: abs(float(row["value"])))
            diagnostics["largestRepeatDelta"] = {
                "scale": best.get("scale"),
                "trialIndex": best.get("trialIndex"),
                "seed": best.get("seed"),
                "maxNgramRepeatDelta": best.get("value"),
            }
        if margin_rows:
            best = max(margin_rows, key=lambda row: abs(float(row["value"])))
            diagnostics["largestMarginMeanDelta"] = {
                "scale": best.get("scale"),
                "trialIndex": best.get("trialIndex"),
                "seed": best.get("seed"),
                "marginMeanDelta": best.get("value"),
            }
        return diagnostics

    def _pipeline_validation_fallback_worst_case_diagnostics(
        self,
        per_scale_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}

        min_cka_rows = self._pipeline_validation_trials_with_metric(
            per_scale_results=per_scale_results,
            metric_key="min_cka",
            layer_key="min_cka_layer",
        )
        if min_cka_rows:
            best = min(min_cka_rows, key=lambda row: float(row["value"]))
            diagnostics["lowestMinCka"] = {
                "scale": best.get("scale"),
                "trialIndex": best.get("trialIndex"),
                "seed": best.get("seed"),
                "minCka": best.get("value"),
                "minCkaLayer": best.get("layer"),
            }

        blindness_rows = self._pipeline_validation_counterexamples_with_metric(
            per_scale_results=per_scale_results,
            metric_key="cka_blindness_ratio",
            layer_key="cka_blindness_worst_layer",
        )
        if blindness_rows:
            best = max(blindness_rows, key=lambda row: float(row["value"]))
            diagnostics["maxBlindnessRatio"] = {
                "scale": best.get("scale"),
                "trialIndex": best.get("trialIndex"),
                "seed": best.get("seed"),
                "ckaBlindnessRatio": best.get("value"),
                "ckaBlindnessWorstLayer": best.get("layer"),
            }

        null_access_rows = self._pipeline_validation_counterexamples_with_metric(
            per_scale_results=per_scale_results,
            metric_key="null_access_min_behavioral_preserved_fraction",
            layer_key="null_access_min_behavioral_preserved_layer",
        )
        if null_access_rows:
            best = min(null_access_rows, key=lambda row: float(row["value"]))
            diagnostics["minBehavioralPreservedNullAccessFraction"] = {
                "scale": best.get("scale"),
                "trialIndex": best.get("trialIndex"),
                "seed": best.get("seed"),
                "fraction": best.get("value"),
                "layer": best.get("layer"),
            }

        loss_rows = self._pipeline_validation_trials_with_metric(
            per_scale_results=per_scale_results,
            metric_key="loss_delta",
        )
        if loss_rows:
            best = max(loss_rows, key=lambda row: float(row["value"]))
            diagnostics["largestLossDelta"] = {
                "scale": best.get("scale"),
                "trialIndex": best.get("trialIndex"),
                "seed": best.get("seed"),
                "lossDelta": best.get("value"),
            }

        perplexity_rows = self._pipeline_validation_trials_with_metric(
            per_scale_results=per_scale_results,
            metric_key="perplexity_delta",
        )
        if perplexity_rows:
            best = max(perplexity_rows, key=lambda row: float(row["value"]))
            diagnostics["largestPerplexityDelta"] = {
                "scale": best.get("scale"),
                "trialIndex": best.get("trialIndex"),
                "seed": best.get("seed"),
                "perplexityDelta": best.get("value"),
            }

        diagnostics.update(self._pipeline_validation_optional_diagnostic_rows(per_scale_results))
        return diagnostics

    @staticmethod
    def _pipeline_validation_trials_with_metric(
        *,
        per_scale_results: dict[str, dict[str, Any]],
        metric_key: str,
        layer_key: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scale, result in per_scale_results.items():
            trial_results = result.get("trial_results", [])
            if not isinstance(trial_results, list):
                continue
            for trial in trial_results:
                if not isinstance(trial, dict) or trial.get(metric_key) is None:
                    continue
                row = {
                    "scale": scale,
                    "trialIndex": trial.get("trial_index"),
                    "seed": trial.get("seed"),
                    "value": trial.get(metric_key),
                }
                if layer_key:
                    row["layer"] = trial.get(layer_key)
                rows.append(row)
        return rows

    @staticmethod
    def _pipeline_validation_counterexamples_with_metric(
        *,
        per_scale_results: dict[str, dict[str, Any]],
        metric_key: str,
        layer_key: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scale, result in per_scale_results.items():
            counterexamples = result.get("counterexamples", [])
            if not isinstance(counterexamples, list):
                continue
            for counterexample in counterexamples:
                if not isinstance(counterexample, dict) or counterexample.get(metric_key) is None:
                    continue
                row = {
                    "scale": scale,
                    "trialIndex": counterexample.get("trial_index"),
                    "seed": counterexample.get("seed"),
                    "value": counterexample.get(metric_key),
                }
                if layer_key:
                    row["layer"] = counterexample.get(layer_key)
                rows.append(row)
        return rows

    @staticmethod
    def _pipeline_validation_failure_case_row(
        case: dict[str, Any],
        *,
        scale: str | None = None,
    ) -> dict[str, Any]:
        return {
            "scale": case.get("scale", scale),
            "trialIndex": case.get("trial_index"),
            "seed": case.get("seed"),
            "failureModes": list(case.get("failure_modes", [])),
            "cooccurrenceClass": case.get("cooccurrence_class"),
            "stopReason": case.get("stop_reason"),
            "lossDelta": case.get("loss_delta"),
            "perplexityDelta": case.get("perplexity_delta"),
            "minCka": case.get("min_cka"),
            "minCkaLayer": case.get("min_cka_layer"),
            "onlineEvalDeltaCorrect": case.get("online_eval_delta_correct"),
            "maxNgramRepeatDelta": case.get("max_4gram_repeat_delta", case.get("max_ngram_repeat_delta")),
            "nullAccessMinBehavioralPreservedFraction": case.get(
                "null_access_min_behavioral_preserved_fraction"
            ),
            "nullAccessMinBehavioralPreservedLayer": case.get(
                "null_access_min_behavioral_preserved_layer"
            ),
            "ckaBlindnessRatio": case.get("cka_blindness_ratio"),
            "ckaBlindnessWorstLayer": case.get("cka_blindness_worst_layer"),
            "marginMeanDelta": case.get("margin_mean_delta"),
            "degenerationMaxNgramRepeat": case.get("degeneration_max_ngram_repeat"),
            "degenerationMeanNgramRepeat": case.get("degeneration_mean_ngram_repeat"),
            "rssFinalCosine": case.get("rss_final_cosine"),
            "rssFinalSpearman": case.get("rss_final_spearman"),
            "dimNullRecruitmentFromBaseline": case.get("dim_null_recruitment_from_baseline"),
            "inferenceMinCka": case.get("inference_min_cka"),
            "inferenceMinCkaLayer": case.get("inference_min_cka_layer"),
            "nullObservabilityMaxConditionNumber": case.get(
                "null_observability_max_condition_number"
            ),
            "nullObservabilityMaxConditionLayer": case.get(
                "null_observability_max_condition_layer"
            ),
            "modeConnectivityBarrier": case.get("mode_connectivity_barrier"),
            "modeConnectivityNormalizedBarrier": case.get(
                "mode_connectivity_normalized_barrier"
            ),
            "modeConnectivityMethod": case.get("mode_connectivity_method"),
            "moeRouterStability": case.get("moe_router_stability"),
            "onlineEvalFirstPreDegradedEpoch": case.get("online_eval_first_pre_degraded_epoch"),
            "onlineEvalFirstPostDegradedEpoch": case.get("online_eval_first_post_degraded_epoch"),
        }

    @staticmethod
    def _pipeline_validation_verdict_text(value: Any) -> str:
        if value is True:
            return "PASS"
        if value is False:
            return "FAIL"
        return "n/a"

    @staticmethod
    def _pipeline_validation_number_text(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    def _pipeline_validation_diagnostic_text(self, row: dict[str, Any]) -> str:
        parts = []
        if row.get("scale") is not None:
            parts.append(f"scale={row['scale']}")
        if row.get("trialIndex") is not None:
            parts.append(f"trial={self._pipeline_validation_number_text(row['trialIndex'])}")
        if row.get("seed") is not None:
            parts.append(f"seed={self._pipeline_validation_number_text(row['seed'])}")
        for key in (
            "minCka",
            "minCkaLayer",
            "ckaBlindnessRatio",
            "ckaBlindnessWorstLayer",
            "fraction",
            "layer",
            "lossDelta",
            "perplexityDelta",
            "maxNgramRepeatDelta",
            "marginMeanDelta",
            "inferenceMinCka",
        ):
            if row.get(key) is not None:
                parts.append(f"{key}={self._pipeline_validation_number_text(row[key])}")
        return ", ".join(parts)

    @staticmethod
    def _normalize_measurement_atlas_summary(
        *,
        summary: dict[str, Any],
        manifest: dict[str, Any],
        variant_rows: list[dict[str, Any]],
        sequence_metrics: list[dict[str, Any]],
        step_metrics: list[dict[str, Any]],
        space_step_metrics: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        onset_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized = dict(summary)
        normalized["workflow"] = "measurement_atlas"
        normalized["variantCount"] = len(variant_rows)
        normalized["sequenceMetricCount"] = len(sequence_metrics)
        normalized["stepMetricCount"] = len(step_metrics)
        normalized["spaceStepMetricCount"] = len(space_step_metrics)
        normalized["comparisonCount"] = len(comparisons)
        normalized["onsetEventCount"] = len(onset_events)
        normalized["errorCount"] = sum(len(row.get("errors", [])) for row in variant_rows)
        normalized.setdefault(
            "spaces",
            sorted(
                {
                    str(row.get("space", "")).strip()
                    for row in sequence_metrics
                    if str(row.get("space", "")).strip()
                }
            ),
        )
        normalized.setdefault(
            "modes",
            sorted(
                {
                    str(row.get("mode", "")).strip()
                    for row in sequence_metrics
                    if str(row.get("mode", "")).strip()
                }
            ),
        )
        normalized.setdefault(
            "studies",
            sorted(
                {
                    str(row.get("studyId", "")).strip()
                    for row in variant_rows
                    if str(row.get("studyId", "")).strip()
                }
            ),
        )
        normalized.setdefault("studyCount", len(normalized.get("studies", [])))
        if isinstance(manifest.get("linkedBlocker"), str):
            normalized.setdefault("linkedBlocker", manifest["linkedBlocker"])
        return normalized

    @staticmethod
    def _atlas_surface_summary(manifest: dict[str, Any]) -> dict[str, Any]:
        frozen_surfaces = manifest.get("frozenSurfaces", {})
        schema = str(manifest.get("schema", "")).strip()
        if schema.endswith(".v2"):
            requested_live = list(frozen_surfaces.get("requestedLiveSpaces", []))
            observed_live = list(frozen_surfaces.get("observedLiveSpaces", []))
            requested_replay = list(frozen_surfaces.get("requestedReplaySpaces", []))
            observed_replay = list(frozen_surfaces.get("observedReplaySpaces", []))
        else:
            requested_live = list(frozen_surfaces.get("liveSpaces", []))
            observed_live = None
            requested_replay = list(frozen_surfaces.get("replaySpaces", []))
            observed_replay = None
        return {
            "schema": schema,
            "requestedLiveSpaces": requested_live,
            "observedLiveSpaces": observed_live,
            "requestedReplaySpaces": requested_replay,
            "observedReplaySpaces": observed_replay,
        }

    def _atlas_study_summaries(
        self,
        *,
        sequence_metrics: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        onset_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        study_ids = sorted(
            {
                str(row.get("studyId", "")).strip()
                for row in sequence_metrics
                if str(row.get("studyId", "")).strip()
            }
        )
        summaries: list[dict[str, Any]] = []
        for study_id in study_ids:
            study_comparisons = [
                comparison for comparison in comparisons if comparison.get("studyId") == study_id
            ]
            study_sequence_metrics = [
                row for row in sequence_metrics if row.get("studyId") == study_id
            ]
            study_onsets = [event for event in onset_events if event.get("studyId") == study_id]
            grounded_events = [
                event
                for event in study_onsets
                if event.get("eventType") == "grounded_label_onset"
            ]
            earliest_grounded = min(
                (int(event["stepIndex"]) for event in grounded_events if event.get("stepIndex") is not None),
                default=None,
            )
            summaries.append(
                {
                    "studyId": study_id,
                    "regionMovedMost": self._atlas_top_delta_axis(
                        study_comparisons,
                        axis="region",
                    ),
                    "spaceMovedMost": self._atlas_top_delta_axis(
                        study_comparisons,
                        axis="space",
                    ),
                    "earliestDivergenceStep": self._atlas_earliest_divergence(study_comparisons),
                    "earliestShiftLocus": self._atlas_earliest_shift_locus(study_sequence_metrics),
                    "liveReplayAgreement": self._atlas_agreement_summary(study_comparisons),
                    "groundedOnsetCount": len(grounded_events),
                    "earliestGroundedOnsetStep": earliest_grounded,
                }
            )
        return summaries

    def _atlas_top_sequence_shifts(
        self,
        comparisons: list[dict[str, Any]],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for comparison in comparisons:
            for row in comparison.get("sequenceDeltas", []):
                if row.get("metric") != "meanGeodesicDeviation":
                    continue
                delta = row.get("delta")
                if delta is None:
                    continue
                rows.append(
                    {
                        "studyId": comparison.get("studyId"),
                        "caseId": comparison.get("caseId"),
                        "from": comparison.get("from"),
                        "to": comparison.get("to"),
                        "mode": row.get("mode"),
                        "region": row.get("region"),
                        "space": row.get("space"),
                        "metric": row.get("metric"),
                        "delta": float(delta),
                    }
                )
        rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
        return rows[:limit]

    def _atlas_locus_changes(
        self,
        *,
        comparisons: list[dict[str, Any]],
        sequence_metrics: list[dict[str, Any]],
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for comparison in comparisons:
            for row in comparison.get("locusComparisons", []):
                if not row.get("changed"):
                    continue
                rows.append(
                    {
                        "studyId": comparison.get("studyId"),
                        "caseId": comparison.get("caseId"),
                        "from": comparison.get("from"),
                        "to": comparison.get("to"),
                        "mode": row.get("mode"),
                        "region": row.get("region"),
                        "space": row.get("space"),
                        "metric": row.get("metric"),
                        "baselineLocus": row.get("baselineLocus"),
                        "variantLocus": row.get("variantLocus"),
                        "changed": True,
                    }
                )
        if rows:
            return rows[:limit]
        return self._atlas_legacy_locus_changes(
            comparisons=comparisons,
            sequence_metrics=sequence_metrics,
            limit=limit,
        )

    @staticmethod
    def _atlas_onset_samples(
        onset_events: list[dict[str, Any]],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        event_priority = {
            "grounded_label_onset": 0,
            "first_divergence": 1,
        }
        rows = sorted(
            onset_events,
            key=lambda row: (
                event_priority.get(str(row.get("eventType", "")).strip(), 99),
                str(row.get("studyId", "")),
                str(row.get("caseId", "")),
                str(row.get("variantId", "")),
                int(row.get("stepIndex", -1)),
            ),
        )
        return rows[:limit]

    def _atlas_example_comparisons(
        self,
        *,
        variant_rows: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        variant_index = {
            (
                row.get("studyId"),
                row.get("caseId"),
                row.get("variantId"),
            ): row
            for row in variant_rows
        }
        scored: list[tuple[float, dict[str, Any]]] = []
        for comparison in comparisons:
            comparison_rows = comparison.get("sequenceDeltas", [])
            delta = max(
                (
                    abs(float(row["delta"]))
                    for row in comparison_rows
                    if row.get("metric") == "meanGeodesicDeviation"
                    and row.get("delta") is not None
                ),
                default=0.0,
            )
            variant_row = variant_index.get(
                (
                    comparison.get("studyId"),
                    comparison.get("caseId"),
                    comparison.get("to"),
                )
            )
            if variant_row is None:
                continue
            scored.append(
                (
                    delta,
                    {
                        "studyId": comparison.get("studyId"),
                        "caseId": comparison.get("caseId"),
                        "from": comparison.get("from"),
                        "to": comparison.get("to"),
                        "promptPreview": self._atlas_preview_text(variant_row.get("promptText")),
                        "generatedPreview": self._atlas_preview_text(
                            variant_row.get("generatedText")
                        ),
                        "promptCharCount": len(str(variant_row.get("promptText", ""))),
                        "generatedCharCount": len(str(variant_row.get("generatedText", ""))),
                        "liveGeneratedFirstDivergenceStep": comparison.get(
                            "liveGeneratedFirstDivergenceStep"
                        ),
                        "replayResponseFirstDivergenceStep": comparison.get(
                            "replayResponseFirstDivergenceStep"
                        ),
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _score, row in scored[:limit]]

    def _build_measurement_atlas_markdown(
        self,
        *,
        summary: dict[str, Any],
        sections: dict[str, Any],
    ) -> str:
        surfaces = sections.get("surfaces", {})
        study_summaries = sections.get("studySummaries", [])
        top_sequence_shifts = sections.get("topSequenceShifts", [])
        locus_changes = sections.get("locusChanges", [])
        onset_samples = sections.get("onsetSamples", [])
        example_comparisons = sections.get("exampleComparisons", [])

        lines = [
            "# Measurement Atlas Bundle",
            "",
            f"- Workflow: `{summary.get('workflow')}`",
            f"- Linked blocker: `{summary.get('linkedBlocker')}`",
            f"- Studies: {summary.get('studyCount')}",
            f"- Variants: {summary.get('variantCount')}",
            f"- Comparisons: {summary.get('comparisonCount')}",
            f"- Onset events: {summary.get('onsetEventCount')}",
            f"- Measurement errors: {summary.get('errorCount')}",
            f"- Spaces: {', '.join(summary.get('spaces', []))}",
        ]

        lines.extend(
            [
                "",
                "## Surfaces",
                "",
                f"- Requested live spaces: {self._atlas_space_list_text(surfaces.get('requestedLiveSpaces'))}",
                f"- Observed live spaces: {self._atlas_space_list_text(surfaces.get('observedLiveSpaces'))}",
                f"- Requested replay spaces: {self._atlas_space_list_text(surfaces.get('requestedReplaySpaces'))}",
                f"- Observed replay spaces: {self._atlas_space_list_text(surfaces.get('observedReplaySpaces'))}",
            ]
        )

        if study_summaries:
            lines.extend(["", "## Study Summaries", ""])
            for row in study_summaries:
                lines.append(
                    f"- `{row['studyId']}`: "
                    f"biggest movement was in `{row['regionMovedMost'] or 'n/a'}` / "
                    f"`{row['spaceMovedMost'] or 'n/a'}`; "
                    f"earliest divergence step was `{row['earliestDivergenceStep'] if row['earliestDivergenceStep'] is not None else 'n/a'}`; "
                    f"earliest shift locus was `{row['earliestShiftLocus'] or 'n/a'}`; "
                    f"live/replay agreement was `{row['liveReplayAgreement']}`; "
                    f"grounded onsets=`{row['groundedOnsetCount']}` "
                    f"(earliest=`{row['earliestGroundedOnsetStep'] if row['earliestGroundedOnsetStep'] is not None else 'n/a'}`)"
                )

        if top_sequence_shifts:
            lines.extend(["", "## Largest Geodesic Shifts", ""])
            for row in top_sequence_shifts:
                lines.append(
                    f"- `{row['studyId']}` / `{row['caseId']}` / `{row['from']}` -> `{row['to']}`: "
                    f"{row['mode']}.{row['region']}.{row['space']}.{row['metric']} delta={row['delta']}"
                )

        if locus_changes:
            lines.extend(["", "## Locus Changes", ""])
            for row in locus_changes:
                lines.append(
                    f"- `{row['studyId']}` / `{row['caseId']}` / `{row['from']}` -> `{row['to']}`: "
                    f"{row['mode']}.{row['region']}.{row['space']}.{row['metric']} "
                    f"{row['baselineLocus'] or 'n/a'} -> {row['variantLocus'] or 'n/a'}"
                )

        if onset_samples:
            lines.extend(["", "## Onset Samples", ""])
            for row in onset_samples:
                lines.append(
                    f"- `{row.get('studyId')}` / `{row.get('caseId')}` / `{row.get('variantId')}`: "
                    f"{row.get('eventType')} {row.get('mode')}.{row.get('region')} step={row.get('stepIndex')}"
                )

        if example_comparisons:
            lines.extend(["", "## Example Comparisons", ""])
            for row in example_comparisons:
                lines.append(
                    f"- `{row['studyId']}` / `{row['caseId']}` / `{row['from']}` -> `{row['to']}`: "
                    f"prompt=`{self._atlas_markdown_preview(row['promptPreview'])}` "
                    f"({row['promptCharCount']} chars) "
                    f"generated=`{self._atlas_markdown_preview(row['generatedPreview'])}` "
                    f"({row['generatedCharCount']} chars) "
                    f"live_divergence={row['liveGeneratedFirstDivergenceStep']} "
                    f"replay_divergence={row['replayResponseFirstDivergenceStep']}"
                )

        lines.append("")
        return "\n".join(lines)

    def _atlas_legacy_locus_changes(
        self,
        *,
        comparisons: list[dict[str, Any]],
        sequence_metrics: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        sequence_index = {
            (
                row.get("studyId"),
                row.get("caseId"),
                row.get("variantId"),
                row.get("mode"),
                row.get("region"),
                row.get("space"),
            ): row
            for row in sequence_metrics
        }
        rows: list[dict[str, Any]] = []
        for comparison in comparisons:
            for mode, region, space in self._atlas_comparison_axes(comparison):
                baseline_row = sequence_index.get(
                    (
                        comparison.get("studyId"),
                        comparison.get("caseId"),
                        comparison.get("from"),
                        mode,
                        region,
                        space,
                    )
                )
                variant_row = sequence_index.get(
                    (
                        comparison.get("studyId"),
                        comparison.get("caseId"),
                        comparison.get("to"),
                        mode,
                        region,
                        space,
                    )
                )
                if baseline_row is None or variant_row is None:
                    continue
                for metric, layer_key, locus_key in (
                    ("peak", "peakLayer", "peakLocus"),
                    ("firstBend", "firstBendLayer", "firstBendLocus"),
                ):
                    baseline_locus = self._atlas_row_locus(
                        baseline_row,
                        layer_key=layer_key,
                        locus_key=locus_key,
                    )
                    variant_locus = self._atlas_row_locus(
                        variant_row,
                        layer_key=layer_key,
                        locus_key=locus_key,
                    )
                    if baseline_locus is None and variant_locus is None:
                        continue
                    if baseline_locus == variant_locus:
                        continue
                    rows.append(
                        {
                            "studyId": comparison.get("studyId"),
                            "caseId": comparison.get("caseId"),
                            "from": comparison.get("from"),
                            "to": comparison.get("to"),
                            "mode": mode,
                            "region": region,
                            "space": space,
                            "metric": metric,
                            "baselineLocus": baseline_locus,
                            "variantLocus": variant_locus,
                            "changed": True,
                        }
                    )
        return rows[:limit]

    @staticmethod
    def _atlas_comparison_axes(comparison: dict[str, Any]) -> set[tuple[str, str, str]]:
        axes: set[tuple[str, str, str]] = set()
        for row in comparison.get("sequenceDeltas", []):
            mode = str(row.get("mode", "")).strip()
            region = str(row.get("region", "")).strip()
            space = str(row.get("space", "")).strip()
            if mode and region and space:
                axes.add((mode, region, space))
        return axes

    @staticmethod
    def _atlas_top_delta_axis(
        comparisons: list[dict[str, Any]],
        *,
        axis: str,
    ) -> str | None:
        scores: dict[str, list[float]] = {}
        for comparison in comparisons:
            for row in comparison.get("sequenceDeltas", []):
                if row.get("metric") != "meanGeodesicDeviation":
                    continue
                key = str(row.get(axis, "")).strip()
                delta = row.get("delta")
                if not key or delta is None:
                    continue
                scores.setdefault(key, []).append(abs(float(delta)))
        ranked = [
            (key, sum(values) / len(values))
            for key, values in scores.items()
            if values
        ]
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[0][0]

    @staticmethod
    def _atlas_earliest_divergence(comparisons: list[dict[str, Any]]) -> int | None:
        candidates = [
            int(step)
            for comparison in comparisons
            for step in (
                comparison.get("liveGeneratedFirstDivergenceStep"),
                comparison.get("replayResponseFirstDivergenceStep"),
            )
            if step is not None
        ]
        return min(candidates) if candidates else None

    def _atlas_earliest_shift_locus(
        self,
        sequence_metrics: list[dict[str, Any]],
    ) -> str | None:
        best: tuple[int, str] | None = None
        for row in sequence_metrics:
            if row.get("variantId") == "control":
                continue
            candidate = self._atlas_row_locus_candidate(row)
            if candidate is not None and (best is None or candidate[0] < best[0]):
                best = candidate
        return best[1] if best is not None else None

    @staticmethod
    def _atlas_row_locus(
        row: dict[str, Any],
        *,
        layer_key: str,
        locus_key: str,
    ) -> str | None:
        raw_locus = row.get(locus_key)
        if isinstance(raw_locus, str):
            locus = raw_locus.strip()
            if locus:
                return locus
        layer_value = row.get(layer_key)
        if layer_value is None:
            return None
        layer_index = int(layer_value)
        space = str(row.get("space", "")).strip()
        if space == "embedding" or layer_index < 0:
            return "embedding"
        return f"layer:{layer_index}"

    def _atlas_row_locus_candidate(self, row: dict[str, Any]) -> tuple[int, str] | None:
        for layer_key, locus_key in (
            ("firstBendLayer", "firstBendLocus"),
            ("peakLayer", "peakLocus"),
        ):
            locus = self._atlas_row_locus(
                row,
                layer_key=layer_key,
                locus_key=locus_key,
            )
            if not locus:
                continue
            if locus == "embedding":
                return (-1, "embedding")
            if locus.startswith("layer:"):
                layer_index = int(locus.split(":", 1)[1])
                return (layer_index, f"layer {layer_index}")
        return None

    @staticmethod
    def _atlas_agreement_summary(comparisons: list[dict[str, Any]]) -> str:
        comparable = [
            comparison
            for comparison in comparisons
            if comparison.get("liveGeneratedFirstDivergenceStep") is not None
            and comparison.get("replayResponseFirstDivergenceStep") is not None
        ]
        if not comparable:
            return "n/a"
        agreed = sum(1 for comparison in comparable if comparison.get("firstGeneratedShiftAgreement"))
        return f"{agreed}/{len(comparable)}"

    @staticmethod
    def _atlas_space_list_text(spaces: list[str] | None) -> str:
        if spaces is None:
            return "n/a"
        if not spaces:
            return "(none)"
        return ", ".join(spaces)

    @staticmethod
    def _atlas_preview_text(
        text: Any,
        *,
        limit: int = 160,
    ) -> str:
        normalized = str(text or "")
        for escaped_whitespace in ("\\n", "\\r", "\\t"):
            normalized = normalized.replace(escaped_whitespace, " ")
        normalized = " ".join(normalized.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    @staticmethod
    def _atlas_markdown_preview(text: str) -> str:
        return text.replace("`", "'")
