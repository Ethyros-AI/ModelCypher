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


@dataclass(frozen=True)
class ObservationBundleReportResult:
    """Structured read-side view of an observation bundle."""

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
    """Shared reporting logic for observation-bundle write and read paths."""

    def load(self, bundle_dir: str | Path) -> ObservationBundleReportResult:
        resolved_dir = Path(bundle_dir).expanduser().resolve()
        if not resolved_dir.exists():
            raise ValueError(f"Observation bundle directory not found: {resolved_dir}")
        if not resolved_dir.is_dir():
            raise ValueError(f"Observation bundle path is not a directory: {resolved_dir}")

        files = self._resolve_files(resolved_dir)
        manifest = self._load_json(Path(files["manifest"]))
        summary = self._load_json(Path(files["summary"]))
        variant_rows = self._load_jsonl(Path(files["variants"]))
        layer_rows = self._load_jsonl(Path(files["layerMetrics"]))
        comparisons = self._load_jsonl(Path(files["comparisons"]))

        return self.build(
            bundle_dir=resolved_dir,
            manifest=manifest,
            summary=summary,
            variant_rows=variant_rows,
            layer_rows=layer_rows,
            comparisons=comparisons,
            files=files,
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
    def _resolve_files(bundle_dir: Path) -> dict[str, str]:
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
                f"Observation bundle is missing required files: {missing_text}"
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
