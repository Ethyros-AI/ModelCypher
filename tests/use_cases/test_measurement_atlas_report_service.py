from __future__ import annotations

from modelcypher.core.use_cases.generation_trace_service import (
    GenerationTraceResult,
    GenerationTraceTokenStream,
)
from modelcypher.core.use_cases.measurement_atlas_report_service import (
    MeasurementAtlasExecution,
    MeasurementAtlasReportService,
)


class _StubTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        token_ids: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            token_ids.append(self._token_to_id[token])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._id_to_token[token_id] for token_id in token_ids if token_id in self._id_to_token)


class _StubBackend:
    def encode_tokens(self, tokenizer: _StubTokenizer, text: str) -> list[int]:
        return tokenizer.encode(text)


def _trace(
    tokenizer: _StubTokenizer,
    *,
    prompt: str,
    generated: str,
    live_generated: str,
    first_bend_layer: int,
    include_embedding_shift: bool = False,
    legacy_embedding_shift: bool = False,
) -> GenerationTraceResult:
    prompt_ids = tuple(tokenizer.encode(prompt))
    response_ids = tuple(tokenizer.encode(generated))
    full_ids = tuple(tokenizer.encode(f"{prompt} {generated}".strip()))
    live_ids = tuple(tokenizer.encode(live_generated))
    sequence_metrics = [
        {
            "mode": "replay",
            "region": "response",
            "space": "hidden",
            "tokenCount": len(response_ids),
            "meanEntropy": 0.1,
            "meanSpectralEntropy": 0.2,
            "meanEffectiveRank": 1.0,
            "meanIntrinsicDimension": 2.0,
            "meanCurvature": 0.3,
            "maxCurvature": 0.5,
            "meanGeodesicDeviation": 0.4,
            "meanPathLengthRatio": 1.1,
            "peakLayer": 2,
            "peakLocus": "layer:2",
            "firstBendLayer": first_bend_layer,
            "firstBendLocus": f"layer:{first_bend_layer}",
        },
        {
            "mode": "live",
            "region": "generated",
            "space": "hidden",
            "tokenCount": len(live_ids),
            "meanEntropy": None,
            "meanSpectralEntropy": 0.25,
            "meanEffectiveRank": 1.1,
            "meanIntrinsicDimension": 2.1,
            "meanCurvature": 0.35,
            "maxCurvature": 0.55,
            "meanGeodesicDeviation": 0.45,
            "meanPathLengthRatio": 1.15,
            "peakLayer": 3,
            "peakLocus": "layer:3",
            "firstBendLayer": first_bend_layer,
            "firstBendLocus": f"layer:{first_bend_layer}",
        },
    ]
    if include_embedding_shift:
        embedding_row = {
            "mode": "replay",
            "region": "response",
            "space": "embedding",
            "tokenCount": len(response_ids),
            "meanEntropy": 0.1,
            "meanSpectralEntropy": 0.2,
            "meanEffectiveRank": 1.0,
            "meanIntrinsicDimension": 2.0,
            "meanCurvature": 0.3,
            "maxCurvature": 0.5,
            "meanGeodesicDeviation": 0.4,
            "meanPathLengthRatio": 1.1,
        }
        if legacy_embedding_shift:
            embedding_row["peakLayer"] = -1
            embedding_row["firstBendLayer"] = -1
        else:
            embedding_row["peakLayer"] = None
            embedding_row["peakLocus"] = "embedding"
            embedding_row["firstBendLayer"] = None
            embedding_row["firstBendLocus"] = "embedding"
        sequence_metrics.append(embedding_row)

    return GenerationTraceResult(
        prompt_text=prompt,
        generated_text=generated,
        prompt_token_ids=prompt_ids,
        response_token_ids=response_ids,
        full_token_ids=full_ids,
        live_generated_token_ids=live_ids,
        token_streams=(
            GenerationTraceTokenStream(
                mode="replay",
                region="response",
                token_ids=response_ids,
                token_texts=tuple(generated.split()),
            ),
            GenerationTraceTokenStream(
                mode="live",
                region="generated",
                token_ids=live_ids,
                token_texts=tuple(live_generated.split()),
            ),
        ),
        sequence_metrics=tuple(sequence_metrics),
        step_metrics=(),
        space_step_metrics=(),
        decode={"policy": "greedy"},
        errors=(),
    )


def test_build_reports_alignment_and_shared_step_counts() -> None:
    tokenizer = _StubTokenizer()
    service = MeasurementAtlasReportService(backend=_StubBackend())
    executions = [
        MeasurementAtlasExecution(
            study_id="measurement_atlas_casing",
            case_id="case1",
            variant_id="control",
            prompt_text="alpha beta",
            comparison_to=None,
            tags=(),
            annotations={"study_role": "control"},
            trace=_trace(
                tokenizer,
                prompt="alpha beta",
                generated="SUPPORTED because",
                live_generated="SUPPORTED because",
                first_bend_layer=4,
            ),
            tokenizer=tokenizer,
        ),
        MeasurementAtlasExecution(
            study_id="measurement_atlas_casing",
            case_id="case1",
            variant_id="all_caps",
            prompt_text="ALPHA BETA",
            comparison_to="control",
            tags=("caps",),
            annotations={"study_role": "perturbation"},
            trace=_trace(
                tokenizer,
                prompt="ALPHA BETA",
                generated="SUPPORTED",
                live_generated="SUPPORTED",
                first_bend_layer=2,
            ),
            tokenizer=tokenizer,
        ),
    ]

    result = service.build(
        run_id="atlas-run",
        timestamp_utc="2026-03-26T18:00:00Z",
        commit="deadbeef",
        linked_blocker="A1",
        claim="trace perturbation geometry",
        mutable_surface="measurement_atlas",
        frozen_surfaces="decode=greedy",
        command="poetry run python scripts/run_measurement_atlas.py ...",
        primary_observable="meanGeodesicDeviation",
        artifact_dir="/tmp/atlas-run",
        next_falsifier="check live vs replay agreement",
        executions=executions,
    )

    comparison = result.comparisons[0]
    assert comparison["alignmentMode"] == "step_index_min_prefix"
    assert comparison["sharedStepCount"] == 1
    assert any(row["tokenCountDelta"] != 0 for row in comparison["sequenceLengthDeltas"])
    assert "replay.response.hidden.meanGeodesicDeviation" in comparison["scalarDeltas"]


def test_grounded_label_onset_is_emitted_when_prefix_leaves_allowed_label() -> None:
    tokenizer = _StubTokenizer()
    service = MeasurementAtlasReportService(backend=_StubBackend())
    executions = [
        MeasurementAtlasExecution(
            study_id="measurement_atlas_grounded_hallucination",
            case_id="case1",
            variant_id="control",
            prompt_text="question",
            comparison_to=None,
            tags=(),
            annotations={"study_role": "supported", "expected_label": "SUPPORTED"},
            trace=_trace(
                tokenizer,
                prompt="question",
                generated="SUPPORTED because",
                live_generated="SUPPORTED because",
                first_bend_layer=3,
            ),
            tokenizer=tokenizer,
        ),
        MeasurementAtlasExecution(
            study_id="measurement_atlas_grounded_hallucination",
            case_id="case1",
            variant_id="bad_label",
            prompt_text="question",
            comparison_to="control",
            tags=(),
            annotations={
                "study_role": "unsupported",
                "expected_label": "SUPPORTED",
                "allowed_label_aliases": ["SUPPORTED"],
            },
            trace=_trace(
                tokenizer,
                prompt="question",
                generated="UNSUPPORTED answer",
                live_generated="UNSUPPORTED answer",
                first_bend_layer=2,
            ),
            tokenizer=tokenizer,
        ),
    ]

    result = service.build(
        run_id="atlas-run",
        timestamp_utc="2026-03-26T18:00:00Z",
        commit="deadbeef",
        linked_blocker="A1",
        claim="trace perturbation geometry",
        mutable_surface="measurement_atlas",
        frozen_surfaces="decode=greedy",
        command="poetry run python scripts/run_measurement_atlas.py ...",
        primary_observable="meanGeodesicDeviation",
        artifact_dir="/tmp/atlas-run",
        next_falsifier="check live vs replay agreement",
        executions=executions,
    )

    grounded = [event for event in result.onset_events if event["eventType"] == "grounded_label_onset"]
    assert grounded
    assert grounded[0]["stepIndex"] == 0


def test_grounded_label_onset_is_absent_when_label_stays_on_prefix() -> None:
    tokenizer = _StubTokenizer()
    service = MeasurementAtlasReportService(backend=_StubBackend())
    execution = MeasurementAtlasExecution(
        study_id="measurement_atlas_grounded_hallucination",
        case_id="case1",
        variant_id="supported",
        prompt_text="question",
        comparison_to=None,
        tags=(),
        annotations={
            "study_role": "supported",
            "expected_label": "SUPPORTED",
            "allowed_label_aliases": ["SUPPORTED"],
        },
        trace=_trace(
            tokenizer,
            prompt="question",
            generated="SUPPORTED because",
            live_generated="SUPPORTED because",
            first_bend_layer=1,
        ),
        tokenizer=tokenizer,
    )

    result = service.build(
        run_id="atlas-run",
        timestamp_utc="2026-03-26T18:00:00Z",
        commit="deadbeef",
        linked_blocker="A1",
        claim="trace perturbation geometry",
        mutable_surface="measurement_atlas",
        frozen_surfaces="decode=greedy",
        command="poetry run python scripts/run_measurement_atlas.py ...",
        primary_observable="meanGeodesicDeviation",
        artifact_dir="/tmp/atlas-run",
        next_falsifier="check live vs replay agreement",
        executions=[execution],
    )

    grounded = [event for event in result.onset_events if event["eventType"] == "grounded_label_onset"]
    assert grounded == []
    assert "## Study: `measurement_atlas_grounded_hallucination`" in result.report_markdown


def test_report_uses_embedding_label_instead_of_raw_negative_layer() -> None:
    tokenizer = _StubTokenizer()
    service = MeasurementAtlasReportService(backend=_StubBackend())
    executions = [
        MeasurementAtlasExecution(
            study_id="measurement_atlas_casing",
            case_id="case1",
            variant_id="control",
            prompt_text="alpha beta",
            comparison_to=None,
            tags=(),
            annotations={"study_role": "control"},
            trace=_trace(
                tokenizer,
                prompt="alpha beta",
                generated="SUPPORTED because",
                live_generated="SUPPORTED because",
                first_bend_layer=4,
            ),
            tokenizer=tokenizer,
        ),
        MeasurementAtlasExecution(
            study_id="measurement_atlas_casing",
            case_id="case1",
            variant_id="all_caps",
            prompt_text="ALPHA BETA",
            comparison_to="control",
            tags=("caps",),
            annotations={"study_role": "perturbation"},
            trace=_trace(
                tokenizer,
                prompt="ALPHA BETA",
                generated="SUPPORTED",
                live_generated="SUPPORTED",
                first_bend_layer=2,
                include_embedding_shift=True,
                legacy_embedding_shift=True,
            ),
            tokenizer=tokenizer,
        ),
    ]

    result = service.build(
        run_id="atlas-run",
        timestamp_utc="2026-03-26T18:00:00Z",
        commit="deadbeef",
        linked_blocker="A1",
        claim="trace perturbation geometry",
        mutable_surface="measurement_atlas",
        frozen_surfaces="decode=greedy",
        command="poetry run python scripts/run_measurement_atlas.py ...",
        primary_observable="meanGeodesicDeviation",
        artifact_dir="/tmp/atlas-run",
        next_falsifier="check live vs replay agreement",
        executions=executions,
    )

    assert "Earliest high-curvature/high-deviation locus: `embedding`" in result.report_markdown
    assert "`-1`" not in result.report_markdown


def test_comparisons_emit_locus_comparisons_and_skip_numeric_embedding_deltas() -> None:
    tokenizer = _StubTokenizer()
    service = MeasurementAtlasReportService(backend=_StubBackend())
    executions = [
        MeasurementAtlasExecution(
            study_id="measurement_atlas_casing",
            case_id="case1",
            variant_id="control",
            prompt_text="alpha beta",
            comparison_to=None,
            tags=(),
            annotations={"study_role": "control"},
            trace=_trace(
                tokenizer,
                prompt="alpha beta",
                generated="SUPPORTED because",
                live_generated="SUPPORTED because",
                first_bend_layer=4,
                include_embedding_shift=True,
            ),
            tokenizer=tokenizer,
        ),
        MeasurementAtlasExecution(
            study_id="measurement_atlas_casing",
            case_id="case1",
            variant_id="all_caps",
            prompt_text="ALPHA BETA",
            comparison_to="control",
            tags=("caps",),
            annotations={"study_role": "perturbation"},
            trace=_trace(
                tokenizer,
                prompt="ALPHA BETA",
                generated="SUPPORTED",
                live_generated="SUPPORTED",
                first_bend_layer=2,
                include_embedding_shift=True,
            ),
            tokenizer=tokenizer,
        ),
    ]

    result = service.build(
        run_id="atlas-run",
        timestamp_utc="2026-03-26T18:00:00Z",
        commit="deadbeef",
        linked_blocker="A1",
        claim="trace perturbation geometry",
        mutable_surface="measurement_atlas",
        frozen_surfaces="decode=greedy",
        command="poetry run python scripts/run_measurement_atlas.py ...",
        primary_observable="meanGeodesicDeviation",
        artifact_dir="/tmp/atlas-run",
        next_falsifier="check live vs replay agreement",
        executions=executions,
    )

    comparison = result.comparisons[0]
    assert all(
        row["metric"] not in {"peakLayer", "firstBendLayer"} or row["space"] != "embedding"
        for row in comparison["sequenceDeltas"]
    )
    assert not any(
        row["baselineLocus"] == "None" or row["variantLocus"] == "None"
        for row in comparison["locusComparisons"]
    )
    assert not any(
        key.endswith(".embedding.peakLayer") or key.endswith(".embedding.firstBendLayer")
        for key in comparison["scalarDeltas"]
    )
    assert {
        (row["space"], row["metric"], row["baselineLocus"], row["variantLocus"], row["changed"])
        for row in comparison["locusComparisons"]
    } == {
        ("embedding", "peak", "embedding", "embedding", False),
        ("embedding", "firstBend", "embedding", "embedding", False),
    }
