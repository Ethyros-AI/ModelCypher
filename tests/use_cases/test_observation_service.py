from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from modelcypher.core.use_cases.observation_service import (
    ObservationService,
    ObservationTarget,
    PromptFamilyManifest,
)


class _StubBackend:
    def encode_tokens(self, _tokenizer, text: str) -> list[str]:
        return text.split()

    def eval(self, *_values) -> None:
        return None

    def tolist(self, value):
        return value


class _StubModelLoader:
    def load_model(self, model_path: str, adapter_path: str | None = None):
        return {"model_path": model_path, "adapter_path": adapter_path}, {"tokenizer": "stub"}

    def generate(self, _model, _tokenizer, prompt: str, max_tokens: int = 128, **_kwargs) -> str:
        return f"response:{prompt}:{max_tokens}"


class _StubActivationProvider:
    def collect_trajectory_batch(self, _model, _tokenizer, texts: list[str]):
        text = texts[0]
        token_count = max(3, len(text.split()))
        return SimpleNamespace(
            positions={
                0: [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]][:token_count],
                1: [[1.0, 1.0], [0.5, 1.0], [0.0, 1.5]][:token_count],
            },
            total_tokens=token_count,
        )

    def collect_hidden_activations(self, _model, _tokenizer, _text: str):
        return {
            0: [1.0, 2.0],
            1: [2.0, 3.0],
        }

    def collect_embedding_activations(self, _model, _tokenizer, _text: str):
        return [0.5, 0.25]

    def collect_intermediate_activations(self, _model, _tokenizer, _text: str):
        return {0: [0.1, 0.2], 1: [0.3, 0.4]}

    def collect_gate_activations_batch(self, _model, _tokenizer, _texts: list[str]):
        return [{0: [0.9, 0.8], 1: [0.7, 0.6]}]

    def collect_attention_activations(self, _model, _tokenizer, _text: str):
        q = {0: [0.1, 0.1], 1: [0.1, 0.2]}
        k = {0: [0.2, 0.2], 1: [0.2, 0.3]}
        v = {0: [0.3, 0.3], 1: [0.3, 0.4]}
        return q, k, v


class _StubGeometryService:
    def analyze_reasoning_flow(self, positions: dict[int, list[list[float]]]):
        return [
            SimpleNamespace(
                layer_idx=layer_idx,
                metrics=SimpleNamespace(
                    mean_curvature=0.1 + layer_idx,
                    max_curvature=0.2 + layer_idx,
                    smoothness=0.9 - (0.1 * layer_idx),
                    directness=0.8 - (0.1 * layer_idx),
                ),
            )
            for layer_idx in positions
        ]

    def compute_layer_entropy(self, _positions, layer_idx: int):
        return SimpleNamespace(
            spectral_entropy=0.5 + layer_idx,
            effective_rank=1.5 + layer_idx,
            intrinsic_dimension=2.5 + layer_idx,
        )


class _StubChainService:
    def analyze_chain(self, _model, _tokenizer, _probe_texts: list[str]):
        return SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    phase=SimpleNamespace(value="highway"),
                    attn_fraction=0.25,
                ),
                SimpleNamespace(
                    layer_idx=1,
                    phase=SimpleNamespace(value="processing"),
                    attn_fraction=0.5,
                ),
            ]
        )


class _StubGeodesicService:
    def measure_layer_profile(self, _model, _tokenizer, _text: str):
        return SimpleNamespace(
            layer_profiles=[
                SimpleNamespace(layer=0, mean_deviation=0.05, path_length_ratio=1.05),
                SimpleNamespace(layer=1, mean_deviation=0.1, path_length_ratio=1.1),
            ]
        )


class _StubBehavioralAnalyzer:
    def analyze_entropy_trajectory(self, _model, _tokenizer, _probe_texts: list[str]):
        return SimpleNamespace(
            layer_indices=(0, 1),
            layer_entropies=(0.2, 0.4),
            slope=0.2,
            peak_layer_fraction=1.0,
        )


def _service() -> ObservationService:
    backend = _StubBackend()
    activation_provider = _StubActivationProvider()
    return ObservationService(
        backend=backend,
        model_loader=_StubModelLoader(),
        activation_provider=activation_provider,
        geometry_service_factory=lambda: _StubGeometryService(),
        chain_service_factory=lambda: _StubChainService(),
        geodesic_service_factory=lambda: _StubGeodesicService(),
        behavioral_analyzer_factory=lambda: _StubBehavioralAnalyzer(),
    )


def test_prompt_family_manifest_parses_flat_rows() -> None:
    manifest = PromptFamilyManifest.from_data(
        {
            "name": "caps-study",
            "variants": [
                {
                    "case_id": "logic_1",
                    "variant_id": "control",
                    "text": "hello world",
                },
                {
                    "case_id": "logic_1",
                    "variant_id": "all_caps",
                    "text": "HELLO WORLD",
                    "comparison_to": "control",
                    "tags": ["caps"],
                },
            ],
        }
    )

    assert manifest.name == "caps-study"
    assert len(manifest.variants) == 2
    grouped = manifest.grouped_variants()
    assert list(grouped.keys()) == ["logic_1"]
    assert grouped["logic_1"][1].comparison_to == "control"


def test_family_bundle_writes_manifest_summary_and_comparisons(tmp_path: Path) -> None:
    manifest = PromptFamilyManifest.from_data(
        {
            "name": "minimal_pairs",
            "variants": [
                {"case_id": "case1", "variant_id": "control", "text": "hello world"},
                {
                    "case_id": "case1",
                    "variant_id": "all_caps",
                    "text": "HELLO WORLD",
                    "comparison_to": "control",
                },
            ],
        }
    )

    result = _service().family(
        target=ObservationTarget(label="base", model="/tmp/model"),
        manifest=manifest,
        output_dir=str(tmp_path / "bundle"),
        spaces=("hidden", "embedding"),
        max_tokens=9,
    )

    bundle_dir = Path(result.output_dir)
    assert bundle_dir.exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "summary.json").exists()
    assert (bundle_dir / "REPORT.md").exists()
    assert (bundle_dir / "variants.jsonl").exists()
    assert (bundle_dir / "layer_metrics.jsonl").exists()
    assert (bundle_dir / "comparisons.jsonl").exists()

    summary = json.loads((bundle_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["workflow"] == "family"
    assert summary["variantCount"] == 2
    assert summary["comparisonCount"] == 1

    comparisons = [
        json.loads(line)
        for line in (bundle_dir / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert comparisons[0]["mode"] == "within_target"
    assert comparisons[0]["from"] == "control"
    assert comparisons[0]["to"] == "all_caps"


def test_compare_bundle_writes_between_target_comparisons(tmp_path: Path) -> None:
    manifest = PromptFamilyManifest.from_data(
        {
            "name": "checkpoint_compare",
            "variants": [
                {"case_id": "case1", "variant_id": "control", "text": "hello world"},
            ],
        }
    )

    result = _service().compare(
        left=ObservationTarget(label="base", model="/tmp/base"),
        right=ObservationTarget(label="adapter", model="/tmp/base", adapter="/tmp/adapter"),
        manifest=manifest,
        output_dir=str(tmp_path / "compare-bundle"),
        spaces=("hidden",),
        max_tokens=7,
    )

    comparisons = [
        json.loads(line)
        for line in (Path(result.output_dir) / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["mode"] == "between_targets" for row in comparisons)
    between = next(row for row in comparisons if row["mode"] == "between_targets")
    assert between["from"] == "base"
    assert between["to"] == "adapter"
