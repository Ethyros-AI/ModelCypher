from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from modelcypher.core.use_cases.generation_trace_service import (
    GenerationTraceService,
    LiveGenerationTraceResult,
    LiveTraceStep,
)
from modelcypher.core.use_cases.observation_service import PromptFamilyManifest


class _StubTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_token_id = 1

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        token_ids: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = self._next_token_id
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
                self._next_token_id += 1
            token_ids.append(self._token_to_id[token])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._id_to_token[token_id] for token_id in token_ids if token_id in self._id_to_token)


class _BosTokenizer(_StubTokenizer):
    bos_token_id = 1

    def __init__(self) -> None:
        super().__init__()
        self._token_to_id["<|startoftext|>"] = self.bos_token_id
        self._id_to_token[self.bos_token_id] = "<|startoftext|>"
        self._next_token_id = 2

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [self.bos_token_id, *super().encode(text)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(
            self._id_to_token[token_id]
            for token_id in token_ids
            if token_id in self._id_to_token and token_id != self.bos_token_id
        )


class _StubBackend:
    def encode_tokens(self, tokenizer: _StubTokenizer, text: str) -> list[int]:
        return tokenizer.encode(text)

    def decode_tokens(self, tokenizer: _StubTokenizer, token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids)

    def array(self, data, dtype=None):
        _ = dtype
        return np.array(data, dtype=float)

    def stack(self, arrays: list[np.ndarray], axis: int = 0):
        return np.stack(arrays, axis=axis)


class _StubModelLoader:
    def generate(self, _model, _tokenizer, _prompt: str, max_tokens: int = 128, **_kwargs):
        return f"fallback_response_{max_tokens}"


class _StubActivationProvider:
    def __init__(self, tokenizer: _StubTokenizer) -> None:
        self._tokenizer = tokenizer

    def collect_trajectory_batch(
        self,
        _model,
        _tokenizer,
        texts: list[str],
        token_ids_batch: list[list[int] | None] | None = None,
    ):
        text_lengths = [
            len(token_ids_batch[index]) if token_ids_batch and token_ids_batch[index] is not None else len(self._tokenizer.encode(text))
            for index, text in enumerate(texts)
        ]
        total_tokens = sum(text_lengths)
        layers = {
            0: np.array([[index + 1.0, index + 1.5] for index in range(total_tokens)], dtype=float),
            1: np.array([[index + 2.0, index + 2.5] for index in range(total_tokens)], dtype=float),
        }
        embedding = np.array([[index + 0.5, index + 0.75] for index in range(total_tokens)], dtype=float)
        return SimpleNamespace(
            positions=layers,
            embedding_positions=embedding,
            intermediate_positions={0: embedding + 0.1},
            q_positions={0: embedding + 0.2},
            k_positions={0: embedding + 0.3},
            v_positions={0: embedding + 0.4},
            gate_positions={0: embedding + 0.5},
            text_lengths=text_lengths,
            total_tokens=total_tokens,
        )


class _CheapTraceService(GenerationTraceService):
    def _summarize_region(self, *, model, tokenizer, mode, region, region_text, space_positions):
        _ = (model, tokenizer, region_text)
        sequence_metrics: list[dict[str, float | int | str | None]] = []
        space_step_metrics: list[dict[str, float | int | str | None]] = []
        for space, mapping in sorted(space_positions.items()):
            if not mapping:
                continue
            token_count = self._first_dim(next(iter(mapping.values())))
            peak_layer_value, peak_locus = self._normalize_layer_metric(
                space=space,
                layer_value=max(mapping.keys()),
            )
            first_bend_layer_value, first_bend_locus = self._normalize_layer_metric(
                space=space,
                layer_value=min(mapping.keys()),
            )
            sequence_metrics.append(
                {
                    "mode": mode,
                    "region": region,
                    "space": space,
                    "tokenCount": token_count,
                    "meanEntropy": 0.1 if mode == "replay" else None,
                    "meanSpectralEntropy": 0.2,
                    "meanEffectiveRank": 1.2,
                    "meanIntrinsicDimension": 2.2,
                    "meanCurvature": 0.3,
                    "maxCurvature": 0.4,
                    "meanGeodesicDeviation": 0.5,
                    "meanPathLengthRatio": 1.1,
                    "peakLayer": peak_layer_value,
                    "peakLocus": peak_locus,
                    "firstBendLayer": first_bend_layer_value,
                    "firstBendLocus": first_bend_locus,
                }
            )
            for layer_idx, positions in mapping.items():
                for step_index in range(self._first_dim(positions)):
                    space_step_metrics.append(
                        {
                            "mode": mode,
                            "region": region,
                            "space": space,
                            "layer": layer_idx,
                            "stepIndex": step_index,
                            "vectorNorm": float(step_index + 1),
                            "euclideanStepLength": None if step_index == 0 else float(step_index),
                            "geodesicStepLength": None if step_index == 0 else float(step_index) + 0.1,
                            "stepDeviation": None if step_index == 0 else 0.1,
                        }
                    )
        return sequence_metrics, space_step_metrics, []


class _ReplayVisibleSpaceTraceService(_CheapTraceService):
    def _summarize_region(self, *, model, tokenizer, mode, region, region_text, space_positions):
        filtered_positions = {
            space: mapping
            for space, mapping in space_positions.items()
            if space in {"hidden", "embedding"}
        }
        return super()._summarize_region(
            model=model,
            tokenizer=tokenizer,
            mode=mode,
            region=region,
            region_text=region_text,
            space_positions=filtered_positions,
        )


def _live_trace_runner(_model, tokenizer: _StubTokenizer, prompt: str, _max_tokens: int):
    prompt_ids = tuple(tokenizer.encode(prompt))
    generated_ids = tuple(tokenizer.encode("SUPPORTED because"))
    steps = (
        LiveTraceStep(
            step_index=0,
            token_id=generated_ids[0],
            token_text="SUPPORTED",
            hidden_by_layer={0: np.array([1.0, 1.5]), 1: np.array([2.0, 2.5])},
            logit_entropy=0.2,
            logit_margin=0.9,
        ),
        LiveTraceStep(
            step_index=1,
            token_id=generated_ids[1],
            token_text="because",
            hidden_by_layer={0: np.array([1.1, 1.6]), 1: np.array([2.1, 2.6])},
            logit_entropy=0.3,
            logit_margin=0.8,
        ),
    )
    return LiveGenerationTraceResult(
        prompt_token_ids=prompt_ids,
        generated_token_ids=generated_ids,
        generated_text="SUPPORTED because",
        stop_reason="eos",
        steps=steps,
    )


def _service() -> tuple[_CheapTraceService, _StubTokenizer]:
    tokenizer = _StubTokenizer()
    backend = _StubBackend()
    service = _CheapTraceService(
        backend=backend,
        model_loader=_StubModelLoader(),
        activation_provider=_StubActivationProvider(tokenizer),
        live_trace_runner=_live_trace_runner,
    )
    return service, tokenizer


def _bos_live_trace_runner(_model, tokenizer: _BosTokenizer, prompt: str, _max_tokens: int):
    prompt_ids = tuple(tokenizer.encode(prompt))
    generated_ids = tuple(tokenizer.encode("SUPPORTED because")[1:])
    steps = (
        LiveTraceStep(
            step_index=0,
            token_id=generated_ids[0],
            token_text="SUPPORTED",
            hidden_by_layer={0: np.array([1.0, 1.5]), 1: np.array([2.0, 2.5])},
            logit_entropy=0.2,
            logit_margin=0.9,
        ),
        LiveTraceStep(
            step_index=1,
            token_id=generated_ids[1],
            token_text="because",
            hidden_by_layer={0: np.array([1.1, 1.6]), 1: np.array([2.1, 2.6])},
            logit_entropy=0.3,
            logit_margin=0.8,
        ),
    )
    return LiveGenerationTraceResult(
        prompt_token_ids=prompt_ids,
        generated_token_ids=generated_ids,
        generated_text="SUPPORTED because",
        stop_reason="eos",
        steps=steps,
    )


def _bos_service() -> tuple[_CheapTraceService, _BosTokenizer]:
    tokenizer = _BosTokenizer()
    backend = _StubBackend()
    service = _CheapTraceService(
        backend=backend,
        model_loader=_StubModelLoader(),
        activation_provider=_StubActivationProvider(tokenizer),
        live_trace_runner=_bos_live_trace_runner,
    )
    return service, tokenizer


def _replay_visible_space_service() -> tuple[_ReplayVisibleSpaceTraceService, _StubTokenizer]:
    tokenizer = _StubTokenizer()
    backend = _StubBackend()
    service = _ReplayVisibleSpaceTraceService(
        backend=backend,
        model_loader=_StubModelLoader(),
        activation_provider=_StubActivationProvider(tokenizer),
        live_trace_runner=_live_trace_runner,
    )
    return service, tokenizer


def test_prompt_family_manifest_supports_v2_annotations_round_trip() -> None:
    manifest = PromptFamilyManifest.from_data(
        {
            "schema": "mc.analyze.prompt_family.v2",
            "name": "measurement_atlas_casing",
            "variants": [
                {
                    "case_id": "case1",
                    "variant_id": "control",
                    "text": "hello world",
                    "annotations": {
                        "study_role": "control",
                        "perturbation_type": "casing",
                    },
                }
            ],
        }
    )

    assert manifest.schema_version == "mc.analyze.prompt_family.v2"
    assert manifest.variants[0].annotations["study_role"] == "control"
    assert manifest.to_dict()["schema"] == "mc.analyze.prompt_family.v2"


def test_trace_variant_emits_replay_and_live_regions_with_boundaries() -> None:
    service, tokenizer = _service()
    result = service.trace_variant(
        model={"model": "stub"},
        tokenizer=tokenizer,
        prompt="alpha beta",
        spaces=("hidden", "embedding"),
        max_tokens=4,
    )

    stream_keys = {(stream.mode, stream.region) for stream in result.token_streams}
    assert ("replay", "prompt") in stream_keys
    assert ("replay", "response") in stream_keys
    assert ("replay", "full") in stream_keys
    assert ("live", "generated") in stream_keys

    full_rows = [
        row for row in result.step_metrics
        if row["mode"] == "replay" and row["region"] == "full"
    ]
    full_stream = next(
        stream
        for stream in result.token_streams
        if stream.mode == "replay" and stream.region == "full"
    )

    assert result.full_token_ids == result.prompt_token_ids + result.response_token_ids
    assert full_stream.token_ids == result.full_token_ids
    assert full_stream.prompt_boundary_index == len(result.prompt_token_ids)
    assert [row["isPromptToken"] for row in full_rows[:2]] == [True, True]
    assert any(row["isResponseToken"] for row in full_rows[2:])

    regions = {(row["mode"], row["region"]) for row in result.sequence_metrics}
    assert ("replay", "prompt") in regions
    assert ("replay", "response") in regions
    assert ("replay", "full") in regions
    assert ("live", "generated") in regions
    assert result.decode["liveSpaces"] == ["hidden"]
    assert result.decode["replaySpaces"] == ["hidden", "embedding"]


def test_live_trace_rows_include_logit_stats_and_hidden_space_metrics() -> None:
    service, tokenizer = _service()
    result = service.trace_variant(
        model={"model": "stub"},
        tokenizer=tokenizer,
        prompt="alpha beta",
        spaces=("hidden", "embedding"),
        max_tokens=4,
    )

    live_rows = [
        row for row in result.step_metrics
        if row["mode"] == "live" and row["region"] == "generated"
    ]
    assert len(live_rows) == 2
    assert live_rows[0]["logitEntropy"] == 0.2
    assert live_rows[0]["logitMargin"] == 0.9

    live_space_rows = [
        row for row in result.space_step_metrics
        if row["mode"] == "live" and row["region"] == "generated"
    ]
    assert live_space_rows
    assert {row["space"] for row in live_space_rows} == {"hidden"}


def test_trace_variant_preserves_full_region_split_without_leading_whitespace() -> None:
    service, tokenizer = _service()
    result = service.trace_variant(
        model={"model": "stub"},
        tokenizer=tokenizer,
        prompt="alpha beta",
        spaces=("hidden", "embedding"),
        max_tokens=4,
    )

    full_stream = next(
        stream
        for stream in result.token_streams
        if stream.mode == "replay" and stream.region == "full"
    )

    assert not result.generated_text.startswith(" ")
    assert full_stream.token_texts == ("alpha", "beta", "SUPPORTED", "because")
    assert full_stream.token_ids == result.prompt_token_ids + result.response_token_ids


def test_trace_variant_uses_continuation_tokens_for_replay_response_with_bos_tokenizer() -> None:
    service, tokenizer = _bos_service()
    result = service.trace_variant(
        model={"model": "stub"},
        tokenizer=tokenizer,
        prompt="alpha beta",
        spaces=("hidden", "embedding"),
        max_tokens=4,
    )

    response_stream = next(
        stream
        for stream in result.token_streams
        if stream.mode == "replay" and stream.region == "response"
    )
    full_stream = next(
        stream
        for stream in result.token_streams
        if stream.mode == "replay" and stream.region == "full"
    )
    response_rows = [
        row for row in result.step_metrics
        if row["mode"] == "replay" and row["region"] == "response"
    ]

    assert result.live_generated_token_ids == result.response_token_ids
    assert response_stream.token_ids == result.live_generated_token_ids
    assert response_rows[0]["tokenText"] == "SUPPORTED"
    assert full_stream.token_ids == result.prompt_token_ids + result.live_generated_token_ids
    assert "<|startoftext|>" not in full_stream.token_texts[full_stream.prompt_boundary_index :]


def test_trace_variant_decode_spaces_reflect_observed_rows_not_requested_enum() -> None:
    service, tokenizer = _replay_visible_space_service()
    result = service.trace_variant(
        model={"model": "stub"},
        tokenizer=tokenizer,
        prompt="alpha beta",
        spaces=("hidden", "embedding", "q", "k"),
        max_tokens=4,
    )

    assert result.decode["liveSpaces"] == ["hidden"]
    assert result.decode["replaySpaces"] == ["hidden", "embedding"]


def test_trace_variant_emits_explicit_locus_fields_without_embedding_sentinels() -> None:
    service, tokenizer = _replay_visible_space_service()
    result = service.trace_variant(
        model={"model": "stub"},
        tokenizer=tokenizer,
        prompt="alpha beta",
        spaces=("hidden", "embedding"),
        max_tokens=4,
    )

    embedding_row = next(
        row
        for row in result.sequence_metrics
        if row["mode"] == "replay" and row["region"] == "full" and row["space"] == "embedding"
    )
    hidden_row = next(
        row
        for row in result.sequence_metrics
        if row["mode"] == "replay" and row["region"] == "full" and row["space"] == "hidden"
    )

    assert embedding_row["peakLayer"] is None
    assert embedding_row["firstBendLayer"] is None
    assert embedding_row["peakLocus"] == "embedding"
    assert embedding_row["firstBendLocus"] == "embedding"
    assert hidden_row["peakLayer"] == 1
    assert hidden_row["firstBendLayer"] == 0
    assert hidden_row["peakLocus"] == "layer:1"
    assert hidden_row["firstBendLocus"] == "layer:0"
