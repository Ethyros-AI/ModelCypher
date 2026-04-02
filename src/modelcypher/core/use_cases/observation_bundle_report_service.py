from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
MEASUREMENT_ATLAS_MANIFEST_PREFIX = "mc.measurement_atlas.run_manifest.v"
OBSERVATION_BUNDLE_VERSION_PREFIX = "mc.analyze.bundle.v"


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
    """Shared reporting logic for observation bundles and atlas artifacts."""

    def load(self, bundle_dir: str | Path) -> ObservationBundleReportResult:
        resolved_dir = Path(bundle_dir).expanduser().resolve()
        if not resolved_dir.exists():
            raise ValueError(f"Report bundle directory not found: {resolved_dir}")
        if not resolved_dir.is_dir():
            raise ValueError(f"Report bundle path is not a directory: {resolved_dir}")

        bundle_family = self._detect_bundle_family(resolved_dir)
        if bundle_family == "measurement_atlas":
            return self._load_measurement_atlas_bundle(resolved_dir)
        return self._load_observation_bundle(resolved_dir)

    def _load_observation_bundle(
        self,
        bundle_dir: Path,
    ) -> ObservationBundleReportResult:
        files = self._resolve_observation_files(bundle_dir)
        manifest = self._load_json(Path(files["manifest"]))
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

    def _detect_bundle_family(self, bundle_dir: Path) -> str:
        atlas_manifest_path = bundle_dir / "run_manifest.json"
        observation_manifest_path = bundle_dir / "manifest.json"

        if atlas_manifest_path.exists():
            manifest = self._load_json(atlas_manifest_path)
            schema = str(manifest.get("schema", "")).strip()
            if schema.startswith(MEASUREMENT_ATLAS_MANIFEST_PREFIX):
                return "measurement_atlas"

        if observation_manifest_path.exists():
            manifest = self._load_json(observation_manifest_path)
            bundle_version = str(manifest.get("bundleVersion", "")).strip()
            if bundle_version.startswith(OBSERVATION_BUNDLE_VERSION_PREFIX):
                return "observation"

        if atlas_manifest_path.exists():
            return "measurement_atlas"
        if observation_manifest_path.exists():
            return "observation"

        raise ValueError(
            "Report bundle directory must contain manifest.json or run_manifest.json: "
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
            "",
            "## Means",
            "",
            f"- Mean prompt tokens: {summary.get('meanPromptTokenCount')}",
            f"- Mean response tokens: {summary.get('meanResponseTokenCount')}",
            f"- Mean entropy: {summary.get('meanEntropy')}",
            f"- Mean geodesic deviation: {summary.get('meanGeodesicDeviation')}",
            f"- Mean curvature: {summary.get('meanCurvature')}",
        ]

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
                    f"prompt={row['promptPreview']!r} ({row['promptCharCount']} chars) "
                    f"generated={row['generatedPreview']!r} ({row['generatedCharCount']} chars) "
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
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."
