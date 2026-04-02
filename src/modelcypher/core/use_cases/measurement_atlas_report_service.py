from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.use_cases.generation_trace_service import (
    GenerationTraceResult,
    compute_first_divergence_step,
    detect_grounded_label_onset,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


LEDGER_HEADER = (
    "run_id\t"
    "timestamp_utc\t"
    "commit\t"
    "status\t"
    "linked_blocker\t"
    "claim\t"
    "mutable_surface\t"
    "frozen_surfaces\t"
    "command\t"
    "primary_observable\t"
    "artifact_dir\t"
    "next_falsifier"
)


@dataclass(frozen=True)
class MeasurementAtlasExecution:
    """One executed prompt variant in a measurement-atlas run."""

    study_id: str
    case_id: str
    variant_id: str
    prompt_text: str
    comparison_to: str | None
    tags: tuple[str, ...]
    annotations: dict[str, Any]
    trace: GenerationTraceResult
    tokenizer: Any | None = None


@dataclass(frozen=True)
class MeasurementAtlasBuildResult:
    """Structured measurement-atlas artifact payload."""

    variant_rows: list[dict[str, Any]]
    sequence_metrics: list[dict[str, Any]]
    step_metrics: list[dict[str, Any]]
    space_step_metrics: list[dict[str, Any]]
    comparisons: list[dict[str, Any]]
    onset_events: list[dict[str, Any]]
    summary: dict[str, Any]
    report_markdown: str
    ledger_header: str
    ledger_row: str


class MeasurementAtlasReportService:
    """Build reportable artifact rows for measurement-atlas runs."""

    def __init__(self, *, backend: "Backend") -> None:
        self._backend = backend

    def build(
        self,
        *,
        run_id: str,
        timestamp_utc: str,
        commit: str,
        linked_blocker: str,
        claim: str,
        mutable_surface: str,
        frozen_surfaces: str,
        command: str,
        primary_observable: str,
        artifact_dir: str | Path,
        next_falsifier: str,
        executions: list[MeasurementAtlasExecution],
    ) -> MeasurementAtlasBuildResult:
        variant_rows = [self._variant_row(execution) for execution in executions]
        sequence_metrics = self._flatten_sequence_metrics(executions)
        step_metrics = self._flatten_step_metrics(executions)
        space_step_metrics = self._flatten_space_step_metrics(executions)
        comparisons, onset_events = self._build_comparisons_and_onsets(
            executions=executions,
            sequence_metrics=sequence_metrics,
        )
        summary = self._build_summary(
            run_id=run_id,
            linked_blocker=linked_blocker,
            executions=executions,
            variant_rows=variant_rows,
            sequence_metrics=sequence_metrics,
            step_metrics=step_metrics,
            space_step_metrics=space_step_metrics,
            comparisons=comparisons,
            onset_events=onset_events,
        )
        report_markdown = self._build_markdown(
            summary=summary,
            executions=executions,
            comparisons=comparisons,
            onset_events=onset_events,
        )
        ledger_row = "\t".join(
            [
                run_id,
                timestamp_utc,
                commit,
                "completed",
                linked_blocker,
                claim,
                mutable_surface,
                frozen_surfaces,
                command,
                primary_observable,
                str(Path(artifact_dir).expanduser().resolve()),
                next_falsifier,
            ]
        )
        return MeasurementAtlasBuildResult(
            variant_rows=variant_rows,
            sequence_metrics=sequence_metrics,
            step_metrics=step_metrics,
            space_step_metrics=space_step_metrics,
            comparisons=comparisons,
            onset_events=onset_events,
            summary=summary,
            report_markdown=report_markdown,
            ledger_header=LEDGER_HEADER,
            ledger_row=ledger_row,
        )

    def _variant_row(self, execution: MeasurementAtlasExecution) -> dict[str, Any]:
        return {
            "studyId": execution.study_id,
            "caseId": execution.case_id,
            "variantId": execution.variant_id,
            "comparisonTo": execution.comparison_to,
            "tags": list(execution.tags),
            "annotations": execution.annotations,
            "promptText": execution.prompt_text,
            "generatedText": execution.trace.generated_text,
            "promptTokenCount": len(execution.trace.prompt_token_ids),
            "responseTokenCount": len(execution.trace.response_token_ids),
            "fullTokenCount": len(execution.trace.full_token_ids),
            "decode": execution.trace.decode,
            "errors": list(execution.trace.errors),
        }

    def _flatten_sequence_metrics(
        self,
        executions: list[MeasurementAtlasExecution],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for execution in executions:
            for row in execution.trace.sequence_metrics:
                rows.append(
                    {
                        "studyId": execution.study_id,
                        "caseId": execution.case_id,
                        "variantId": execution.variant_id,
                        **row,
                    }
                )
        return rows

    def _flatten_step_metrics(
        self,
        executions: list[MeasurementAtlasExecution],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for execution in executions:
            for row in execution.trace.step_metrics:
                rows.append(
                    {
                        "studyId": execution.study_id,
                        "caseId": execution.case_id,
                        "variantId": execution.variant_id,
                        **row,
                    }
                )
        return rows

    def _flatten_space_step_metrics(
        self,
        executions: list[MeasurementAtlasExecution],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for execution in executions:
            for row in execution.trace.space_step_metrics:
                rows.append(
                    {
                        "studyId": execution.study_id,
                        "caseId": execution.case_id,
                        "variantId": execution.variant_id,
                        **row,
                    }
                )
        return rows

    def _build_comparisons_and_onsets(
        self,
        *,
        executions: list[MeasurementAtlasExecution],
        sequence_metrics: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        comparisons: list[dict[str, Any]] = []
        onset_events: list[dict[str, Any]] = []
        grouped = self._group_executions(executions)
        sequence_index = {
            (
                row["studyId"],
                row["caseId"],
                row["variantId"],
                row["mode"],
                row["region"],
                row["space"],
            ): row
            for row in sequence_metrics
        }
        for (study_id, case_id), case_executions in sorted(grouped.items()):
            ordered = list(case_executions)
            by_variant_id = {execution.variant_id: execution for execution in ordered}
            default_baseline = self._default_baseline_variant(ordered)
            for execution in ordered:
                baseline_variant_id = execution.comparison_to or default_baseline.variant_id
                if baseline_variant_id == execution.variant_id:
                    continue
                baseline = by_variant_id.get(baseline_variant_id)
                if baseline is None:
                    continue

                sequence_deltas: list[dict[str, Any]] = []
                scalar_deltas: dict[str, float] = {}
                locus_comparisons: list[dict[str, Any]] = []
                sequence_length_deltas: list[dict[str, Any]] = []
                for stream in execution.trace.token_streams:
                    baseline_stream = self._token_stream(
                        baseline.trace,
                        mode=stream.mode,
                        region=stream.region,
                    )
                    if baseline_stream is None:
                        continue
                    shared_step_count = min(
                        len(baseline_stream.token_ids),
                        len(stream.token_ids),
                    )
                    sequence_length_deltas.append(
                        {
                            "mode": stream.mode,
                            "region": stream.region,
                            "baselineTokenCount": len(baseline_stream.token_ids),
                            "variantTokenCount": len(stream.token_ids),
                            "sharedStepCount": shared_step_count,
                            "tokenCountDelta": len(stream.token_ids) - len(baseline_stream.token_ids),
                        }
                    )

                for key, baseline_row in sequence_index.items():
                    row_study, row_case, row_variant, mode, region, space = key
                    if row_study != study_id or row_case != case_id or row_variant != baseline.variant_id:
                        continue
                    comparison_row = sequence_index.get(
                        (study_id, case_id, execution.variant_id, mode, region, space)
                    )
                    if comparison_row is None:
                        continue
                    for metric in (
                        "meanEntropy",
                        "meanSpectralEntropy",
                        "meanEffectiveRank",
                        "meanIntrinsicDimension",
                        "meanCurvature",
                        "maxCurvature",
                        "meanGeodesicDeviation",
                        "meanPathLengthRatio",
                    ):
                        baseline_value = baseline_row.get(metric)
                        variant_value = comparison_row.get(metric)
                        if baseline_value is None or variant_value is None:
                            continue
                        delta = float(variant_value) - float(baseline_value)
                        metric_key = f"{mode}.{region}.{space}.{metric}"
                        scalar_deltas[metric_key] = delta
                        sequence_deltas.append(
                            {
                                "mode": mode,
                                "region": region,
                                "space": space,
                                "metric": metric,
                                "baselineValue": baseline_value,
                                "variantValue": variant_value,
                                "delta": delta,
                            }
                        )
                    for metric, layer_key, locus_key in (
                        ("peak", "peakLayer", "peakLocus"),
                        ("firstBend", "firstBendLayer", "firstBendLocus"),
                    ):
                        baseline_layer_value = self._numeric_layer_value(
                            baseline_row,
                            layer_key=layer_key,
                            locus_key=locus_key,
                        )
                        variant_layer_value = self._numeric_layer_value(
                            comparison_row,
                            layer_key=layer_key,
                            locus_key=locus_key,
                        )
                        if baseline_layer_value is not None and variant_layer_value is not None:
                            delta = float(variant_layer_value) - float(baseline_layer_value)
                            metric_key = f"{mode}.{region}.{space}.{layer_key}"
                            scalar_deltas[metric_key] = delta
                            sequence_deltas.append(
                                {
                                    "mode": mode,
                                    "region": region,
                                    "space": space,
                                    "metric": layer_key,
                                    "baselineValue": baseline_layer_value,
                                    "variantValue": variant_layer_value,
                                    "delta": delta,
                                }
                            )
                            continue

                        baseline_locus = self._row_locus(
                            baseline_row,
                            layer_key=layer_key,
                            locus_key=locus_key,
                        )
                        variant_locus = self._row_locus(
                            comparison_row,
                            layer_key=layer_key,
                            locus_key=locus_key,
                        )
                        if baseline_locus is None and variant_locus is None:
                            continue
                        locus_comparisons.append(
                            {
                                "mode": mode,
                                "region": region,
                                "space": space,
                                "metric": metric,
                                "baselineLocus": baseline_locus,
                                "variantLocus": variant_locus,
                                "changed": baseline_locus != variant_locus,
                            }
                        )

                live_divergence = self._first_divergence_for(
                    baseline.trace,
                    execution.trace,
                    mode="live",
                    region="generated",
                )
                replay_divergence = self._first_divergence_for(
                    baseline.trace,
                    execution.trace,
                    mode="replay",
                    region="response",
                )
                comparisons.append(
                    {
                        "studyId": study_id,
                        "caseId": case_id,
                        "from": baseline.variant_id,
                        "to": execution.variant_id,
                        "alignmentMode": "step_index_min_prefix",
                        "sharedStepCount": self._shared_step_count(
                            baseline.trace,
                            execution.trace,
                            mode="live",
                            region="generated",
                        ),
                        "sequenceLengthDeltas": sequence_length_deltas,
                        "scalarDeltas": scalar_deltas,
                        "sequenceDeltas": sequence_deltas,
                        "locusComparisons": locus_comparisons,
                        "liveGeneratedFirstDivergenceStep": live_divergence,
                        "replayResponseFirstDivergenceStep": replay_divergence,
                        "firstGeneratedShiftAgreement": (
                            live_divergence is not None
                            and replay_divergence is not None
                            and live_divergence == replay_divergence
                        ),
                    }
                )
                if live_divergence is not None:
                    onset_events.append(
                        {
                            "studyId": study_id,
                            "caseId": case_id,
                            "variantId": execution.variant_id,
                            "baselineVariantId": baseline.variant_id,
                            "eventType": "first_divergence",
                            "mode": "live",
                            "region": "generated",
                            "stepIndex": live_divergence,
                        }
                    )
                if replay_divergence is not None:
                    onset_events.append(
                        {
                            "studyId": study_id,
                            "caseId": case_id,
                            "variantId": execution.variant_id,
                            "baselineVariantId": baseline.variant_id,
                            "eventType": "first_divergence",
                            "mode": "replay",
                            "region": "response",
                            "stepIndex": replay_divergence,
                        }
                    )
                onset_events.extend(self._grounded_label_events(execution))
        return comparisons, onset_events

    def _build_summary(
        self,
        *,
        run_id: str,
        linked_blocker: str,
        executions: list[MeasurementAtlasExecution],
        variant_rows: list[dict[str, Any]],
        sequence_metrics: list[dict[str, Any]],
        step_metrics: list[dict[str, Any]],
        space_step_metrics: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        onset_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "runId": run_id,
            "linkedBlocker": linked_blocker,
            "studyCount": len({execution.study_id for execution in executions}),
            "variantCount": len(variant_rows),
            "sequenceMetricCount": len(sequence_metrics),
            "stepMetricCount": len(step_metrics),
            "spaceStepMetricCount": len(space_step_metrics),
            "comparisonCount": len(comparisons),
            "onsetEventCount": len(onset_events),
            "errorCount": sum(len(row.get("errors", [])) for row in variant_rows),
            "studies": sorted({execution.study_id for execution in executions}),
            "spaces": sorted(
                {
                    row["space"]
                    for row in sequence_metrics
                    if str(row.get("space", "")).strip()
                }
            ),
            "modes": sorted(
                {
                    row["mode"]
                    for row in sequence_metrics
                    if str(row.get("mode", "")).strip()
                }
            ),
        }

    def _build_markdown(
        self,
        *,
        summary: dict[str, Any],
        executions: list[MeasurementAtlasExecution],
        comparisons: list[dict[str, Any]],
        onset_events: list[dict[str, Any]],
    ) -> str:
        grouped_executions = self._group_executions(executions)
        comparisons_by_study: dict[str, list[dict[str, Any]]] = {}
        for comparison in comparisons:
            comparisons_by_study.setdefault(comparison["studyId"], []).append(comparison)
        onset_by_study: dict[str, list[dict[str, Any]]] = {}
        for event in onset_events:
            onset_by_study.setdefault(event["studyId"], []).append(event)

        lines = [
            "# Measurement Atlas Report",
            "",
            f"- Linked blocker: `{summary['linkedBlocker']}`",
            f"- Studies: {summary['studyCount']}",
            f"- Variants: {summary['variantCount']}",
            f"- Comparisons: {summary['comparisonCount']}",
            f"- Onset events: {summary['onsetEventCount']}",
            f"- Errors: {summary['errorCount']}",
            "",
        ]

        for study_id in summary["studies"]:
            study_comparisons = comparisons_by_study.get(study_id, [])
            study_onsets = onset_by_study.get(study_id, [])
            study_executions = [
                execution
                for (row_study, _case_id), case_rows in grouped_executions.items()
                if row_study == study_id
                for execution in case_rows
            ]
            region_move = self._top_delta_axis(study_comparisons, axis="region")
            space_move = self._top_delta_axis(study_comparisons, axis="space")
            earliest_divergence = self._earliest_divergence(study_comparisons)
            earliest_locus = self._earliest_shift_locus(study_executions)
            agreement = self._agreement_summary(study_comparisons)
            grounded_events = [
                event for event in study_onsets if event["eventType"] == "grounded_label_onset"
            ]
            earliest_grounded = min(
                (event["stepIndex"] for event in grounded_events),
                default=None,
            )

            lines.extend(
                [
                    f"## Study: `{study_id}`",
                    "",
                    f"- Region moved most: `{region_move or 'n/a'}`",
                    f"- Space moved most: `{space_move or 'n/a'}`",
                    f"- Earliest divergence step: `{earliest_divergence if earliest_divergence is not None else 'n/a'}`",
                    f"- Earliest high-curvature/high-deviation locus: `{earliest_locus if earliest_locus is not None else 'n/a'}`",
                    f"- Live/replay first generated-token agreement: `{agreement}`",
                    f"- Grounded hallucination onsets: `{len(grounded_events)}`",
                    f"- Earliest grounded onset step: `{earliest_grounded if earliest_grounded is not None else 'n/a'}`",
                    "",
                    "### Top Qualitative Examples",
                    "",
                ]
            )
            top_examples = self._top_examples(study_executions, study_comparisons)
            for example in top_examples:
                lines.append(
                    f"- `{example['caseId']}/{example['variantId']}` vs `{example['baselineVariantId']}`: "
                    f"prompt=`{example['promptText']}` generated=`{example['generatedText']}` "
                    f"live_divergence=`{example['liveDivergence']}` replay_divergence=`{example['replayDivergence']}`"
                )
            if not top_examples:
                lines.append("- No comparison examples were available.")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _group_executions(
        executions: list[MeasurementAtlasExecution],
    ) -> dict[tuple[str, str], list[MeasurementAtlasExecution]]:
        grouped: dict[tuple[str, str], list[MeasurementAtlasExecution]] = {}
        for execution in executions:
            grouped.setdefault((execution.study_id, execution.case_id), []).append(execution)
        return grouped

    @staticmethod
    def _default_baseline_variant(
        executions: list[MeasurementAtlasExecution],
    ) -> MeasurementAtlasExecution:
        for execution in executions:
            if execution.variant_id == "control":
                return execution
        return executions[0]

    @staticmethod
    def _token_stream(
        trace: GenerationTraceResult,
        *,
        mode: str,
        region: str,
    ):
        for stream in trace.token_streams:
            if stream.mode == mode and stream.region == region:
                return stream
        return None

    def _first_divergence_for(
        self,
        baseline_trace: GenerationTraceResult,
        variant_trace: GenerationTraceResult,
        *,
        mode: str,
        region: str,
    ) -> int | None:
        baseline_stream = self._token_stream(baseline_trace, mode=mode, region=region)
        variant_stream = self._token_stream(variant_trace, mode=mode, region=region)
        if baseline_stream is None or variant_stream is None:
            return None
        return compute_first_divergence_step(
            baseline_stream.token_ids,
            variant_stream.token_ids,
        )

    def _shared_step_count(
        self,
        baseline_trace: GenerationTraceResult,
        variant_trace: GenerationTraceResult,
        *,
        mode: str,
        region: str,
    ) -> int:
        baseline_stream = self._token_stream(baseline_trace, mode=mode, region=region)
        variant_stream = self._token_stream(variant_trace, mode=mode, region=region)
        if baseline_stream is None or variant_stream is None:
            return 0
        return min(len(baseline_stream.token_ids), len(variant_stream.token_ids))

    def _grounded_label_events(
        self,
        execution: MeasurementAtlasExecution,
    ) -> list[dict[str, Any]]:
        annotations = execution.annotations or {}
        expected_label = str(annotations.get("expected_label", "")).strip()
        allowed_aliases = annotations.get("allowed_label_aliases", [])
        if not expected_label and not allowed_aliases:
            return []
        allowed_labels = [expected_label] if expected_label else []
        allowed_labels.extend(
            alias.strip()
            for alias in allowed_aliases
            if isinstance(alias, str) and alias.strip()
        )
        if not allowed_labels:
            return []
        live_stream = self._token_stream(execution.trace, mode="live", region="generated")
        if live_stream is None or not live_stream.token_ids or execution.tokenizer is None:
            return []
        allowed_label_token_ids = [
            tuple(self._backend.encode_tokens(execution.tokenizer, label))
            for label in allowed_labels
        ]
        onset_step, reason = detect_grounded_label_onset(
            generated_token_ids=live_stream.token_ids,
            allowed_label_token_ids=allowed_label_token_ids,
        )
        if onset_step is None:
            return []
        return [
            {
                "studyId": execution.study_id,
                "caseId": execution.case_id,
                "variantId": execution.variant_id,
                "eventType": "grounded_label_onset",
                "mode": "live",
                "region": "generated",
                "stepIndex": onset_step,
                "reason": reason,
                "allowedLabels": allowed_labels,
            }
        ]

    @staticmethod
    def _top_delta_axis(
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
                if not key:
                    continue
                scores.setdefault(key, []).append(abs(float(row["delta"])))
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
    def _earliest_divergence(comparisons: list[dict[str, Any]]) -> int | None:
        candidates = [
            step
            for comparison in comparisons
            for step in (
                comparison.get("liveGeneratedFirstDivergenceStep"),
                comparison.get("replayResponseFirstDivergenceStep"),
            )
            if step is not None
        ]
        return min(candidates) if candidates else None

    @staticmethod
    def _earliest_shift_locus(executions: list[MeasurementAtlasExecution]) -> str | None:
        best: tuple[int, str] | None = None
        for execution in executions:
            if execution.variant_id == "control":
                continue
            for row in execution.trace.sequence_metrics:
                candidate = MeasurementAtlasReportService._row_locus_candidate(row)
                if candidate is not None and (best is None or candidate[0] < best[0]):
                    best = candidate
        return best[1] if best is not None else None

    @staticmethod
    def _numeric_layer_value(
        row: dict[str, Any],
        *,
        layer_key: str,
        locus_key: str,
    ) -> int | None:
        layer_value = row.get(layer_key)
        if layer_value is None:
            return None
        locus = MeasurementAtlasReportService._row_locus(
            row,
            layer_key=layer_key,
            locus_key=locus_key,
        )
        if locus is None or not locus.startswith("layer:"):
            return None
        return int(layer_value)

    @staticmethod
    def _row_locus(
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

    @staticmethod
    def _row_locus_candidate(row: dict[str, Any]) -> tuple[int, str] | None:
        for layer_key, locus_key in (
            ("firstBendLayer", "firstBendLocus"),
            ("peakLayer", "peakLocus"),
        ):
            locus = MeasurementAtlasReportService._row_locus(
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
    def _agreement_summary(comparisons: list[dict[str, Any]]) -> str:
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
    def _top_examples(
        executions: list[MeasurementAtlasExecution],
        comparisons: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        execution_index = {
            (execution.case_id, execution.variant_id): execution
            for execution in executions
        }
        scored: list[tuple[float, dict[str, Any]]] = []
        for comparison in comparisons:
            delta = max(
                (
                    abs(float(row["delta"]))
                    for row in comparison.get("sequenceDeltas", [])
                    if row.get("metric") == "meanGeodesicDeviation"
                ),
                default=0.0,
            )
            execution = execution_index.get((comparison["caseId"], comparison["to"]))
            if execution is None:
                continue
            scored.append(
                (
                    delta,
                    {
                        "caseId": comparison["caseId"],
                        "variantId": comparison["to"],
                        "baselineVariantId": comparison["from"],
                        "promptText": execution.prompt_text[:80],
                        "generatedText": execution.trace.generated_text[:80],
                        "liveDivergence": comparison.get("liveGeneratedFirstDivergenceStep"),
                        "replayDivergence": comparison.get("replayResponseFirstDivergenceStep"),
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _score, row in scored[:3]]
