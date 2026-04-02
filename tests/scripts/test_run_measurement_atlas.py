from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from modelcypher.core.use_cases.generation_trace_service import (
    GenerationTraceResult,
    GenerationTraceTokenStream,
)
from modelcypher.core.use_cases.measurement_atlas_report_service import (
    MeasurementAtlasReportService,
)


def _load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "run_measurement_atlas.py"
    spec = importlib.util.spec_from_file_location(
        "run_measurement_atlas_script",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


class _StubModelLoader:
    def load_model(self, _model_path: str):
        return {"model": "stub"}, _StubTokenizer()


class _StubTraceService:
    def trace_variant(self, *, model, tokenizer, prompt, spaces, max_tokens):
        _ = (model, spaces, max_tokens)
        prompt_ids = tuple(tokenizer.encode(prompt))
        response_ids = tuple(tokenizer.encode("SUPPORTED because"))
        full_ids = tuple(tokenizer.encode(f"{prompt} SUPPORTED because"))
        return GenerationTraceResult(
            prompt_text=prompt,
            generated_text="SUPPORTED because",
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
            full_token_ids=full_ids,
            live_generated_token_ids=response_ids,
            token_streams=(
                GenerationTraceTokenStream(
                    mode="replay",
                    region="response",
                    token_ids=response_ids,
                    token_texts=("SUPPORTED", "because"),
                ),
                GenerationTraceTokenStream(
                    mode="live",
                    region="generated",
                    token_ids=response_ids,
                    token_texts=("SUPPORTED", "because"),
                ),
            ),
            sequence_metrics=(
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
                    "maxCurvature": 0.4,
                    "meanGeodesicDeviation": 0.5,
                    "meanPathLengthRatio": 1.1,
                    "peakLayer": 3,
                    "peakLocus": "layer:3",
                    "firstBendLayer": 2,
                    "firstBendLocus": "layer:2",
                },
            ),
            step_metrics=(
                {
                    "mode": "live",
                    "region": "generated",
                    "globalStepIndex": len(prompt_ids),
                    "regionStepIndex": 0,
                    "tokenId": response_ids[0],
                    "tokenText": "SUPPORTED",
                    "isPromptToken": False,
                    "isResponseToken": True,
                    "logitEntropy": 0.2,
                    "logitMargin": 0.9,
                },
            ),
            space_step_metrics=(
                {
                    "mode": "live",
                    "region": "generated",
                    "space": "hidden",
                    "layer": 0,
                    "stepIndex": 0,
                    "vectorNorm": 1.0,
                    "euclideanStepLength": None,
                    "geodesicStepLength": None,
                    "stepDeviation": None,
                },
            ),
            decode={
                "policy": "greedy",
                "liveSpaces": ["hidden"],
                "replaySpaces": ["hidden"],
            },
            errors=(),
        )


def test_help_exposes_research_script_surface() -> None:
    script = _load_script_module()
    help_text = script.build_parser().format_help()

    assert "--model" in help_text
    assert "--manifest" in help_text
    assert "measurement atlas" in help_text.lower()
    assert "mc analyze trace" not in help_text


def test_fixture_run_writes_required_artifact_family(tmp_path: Path) -> None:
    script = _load_script_module()
    manifest_path = tmp_path / "study.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "mc.analyze.prompt_family.v2",
                "name": "measurement_atlas_casing",
                "variants": [
                    {
                        "case_id": "case1",
                        "variant_id": "control",
                        "text": "alpha beta",
                        "annotations": {"study_role": "control", "expected_label": "SUPPORTED"},
                    },
                    {
                        "case_id": "case1",
                        "variant_id": "all_caps",
                        "text": "ALPHA BETA",
                        "comparison_to": "control",
                        "annotations": {"study_role": "perturbation", "expected_label": "SUPPORTED"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = script.run_measurement_atlas(
        model_path="/tmp/model",
        manifest_paths=[str(manifest_path)],
        output_root=str(tmp_path / "results"),
        max_tokens=8,
        backend=_StubBackend(),
        activation_provider=object(),
        model_loader=_StubModelLoader(),
        trace_service=_StubTraceService(),
        report_service=MeasurementAtlasReportService(backend=_StubBackend()),
        timestamp_utc="2026-03-26T18:00:00Z",
        commit="deadbeef",
    )

    required = {
        "run_manifest.json",
        "summary.json",
        "REPORT.md",
        "ledger.tsv",
        "variants.jsonl",
        "sequence_metrics.jsonl",
        "step_metrics.jsonl",
        "space_step_metrics.jsonl",
        "comparisons.jsonl",
        "onset_events.jsonl",
    }
    assert required.issubset({path.name for path in output_dir.iterdir()})

    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "## Study: `measurement_atlas_casing`" in report
    assert "Region moved most" in report
    assert "Grounded hallucination onsets" in report

    ledger_lines = (output_dir / "ledger.tsv").read_text(encoding="utf-8").splitlines()
    assert "linked_blocker" in ledger_lines[0]
    assert "mutable_surface" in ledger_lines[0]
    assert "command" in ledger_lines[0]
    assert "artifact_dir" in ledger_lines[0]
    assert "next_falsifier" in ledger_lines[0]

    sequence_row = json.loads(
        (output_dir / "sequence_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert sequence_row["peakLayer"] == 3
    assert sequence_row["peakLocus"] == "layer:3"
    assert sequence_row["firstBendLayer"] == 2
    assert sequence_row["firstBendLocus"] == "layer:2"

    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["schema"] == "mc.measurement_atlas.run_manifest.v2"
    frozen_surfaces = run_manifest["frozenSurfaces"]
    assert frozen_surfaces["requestedReplaySpaces"] == ["hidden", "embedding"]
    assert frozen_surfaces["observedReplaySpaces"] == ["hidden"]
    assert frozen_surfaces["requestedLiveSpaces"] == ["hidden"]
    assert frozen_surfaces["observedLiveSpaces"] == ["hidden"]
    assert "replaySpaces" not in frozen_surfaces
    assert "liveSpaces" not in frozen_surfaces
