# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from modelcypher.core.domain.atlas.unified_atlas import AtlasProbe, AtlasSource
from modelcypher.core.domain.domains import AtlasDomain


def _load_script_module(name: str, relative_path: str) -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module(
    "r2_execution_plan_scan_script",
    "scripts/r2_execution_plan_scan.py",
)


class _FakeBackend:
    def get_num_layers(self, model) -> int:
        return len(model.model.layers)

    def finfo(self):
        return SimpleNamespace(eps=1e-7)


class _FakeModelLoader:
    def load_model(self, _model_path: str):
        model = SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=object(),
                layers=[object() for _ in range(4)],
            )
        )
        tokenizer = object()
        return model, tokenizer


def _make_services():
    return SCRIPT.ScanServices(
        backend=_FakeBackend(),
        activation_provider=SimpleNamespace(),
        verification_depth_service=SimpleNamespace(),
        benchmark_service=SimpleNamespace(),
        model_loader=_FakeModelLoader(),
    )


def _make_probe(probe_id: str, depth: int, text: str) -> AtlasProbe:
    return AtlasProbe(
        id=probe_id,
        source=AtlasSource.SEQUENCE_INVARIANT,
        domain=AtlasDomain.LINGUISTIC,
        name=f"probe-{probe_id}",
        description=text,
        cross_domain_weight=1.0,
        category_name="tests",
        support_texts=(text,),
        verification_depth=depth,
    )


def _fake_stage1_row(candidate, base_rank: int = 10) -> dict[str, object]:
    repeated = candidate.repeated_block()
    ordering_score = 0 if repeated is None else (repeated["start"] * 10 + repeated["end"])
    return {
        "planKey": candidate.key,
        "plan": candidate.plan.to_dict(),
        "repeatedBlock": repeated,
        "stage1": {
            "layerProfiles": [],
            "canonicalTrajectoryRank": base_rank if repeated is None else base_rank - ordering_score,
            "canonicalIntrinsicDimension": 5.0 if repeated is None else 5.0 - ordering_score / 100.0,
            "canonicalNullRank": 0,
            "canonicalConditionNumber": 1.0 if repeated is None else 1.0 + ordering_score / 1000.0,
            "normSummary": {
                "meanLayerNorms": [],
                "consecutiveNormJumps": [],
                "maxConsecutiveNormJump": 0.0 if repeated is None else ordering_score / 1000.0,
            },
        },
        "stage2": None,
    }


def test_run_scan_emits_artifacts_and_freezes_probe_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fake",
                "architectures": ["FakeLM"],
                "torch_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )

    probes = [
        _make_probe("p0", 1, "alpha"),
        _make_probe("p1", 2, "beta"),
        _make_probe("p2", 3, "gamma"),
    ]
    monkeypatch.setattr(SCRIPT, "load_all_probes", lambda: list(probes))
    monkeypatch.setattr(
        SCRIPT,
        "compute_readout_effective_rank",
        lambda model, backend: 8.0,
    )
    monkeypatch.setattr(
        SCRIPT,
        "evaluate_stage1_candidate",
        lambda **kwargs: _fake_stage1_row(kwargs["candidate"]),
    )

    def _fake_stage2_candidate(
        *,
        candidate,
        identity_positions_by_layer,
        **_: object,
    ):
        repeated = candidate.repeated_block()
        score = 0 if repeated is None else repeated["start"] + repeated["end"]
        return (
            {
                "verificationDepthProfile": {"levels": [1, 2, 3]},
                "inferenceCkaVsIdentity": (
                    None
                    if identity_positions_by_layer is None
                    else {
                        "perLayer": {"0": max(0.0, 0.95 - score / 20.0)},
                        "mean": max(0.0, 0.95 - score / 20.0),
                        "min": max(0.0, 0.90 - score / 20.0),
                    }
                ),
                "behavioralQuickSuite": {
                    "suite": "quick",
                    "suiteBenchmarks": list(SCRIPT.BENCHMARK_SUITE),
                    "overallAccuracy": 0.5 if repeated is None else 0.5 - score / 100.0,
                    "correct": 3,
                    "total": 6,
                    "readoutEffectiveRank": 8.0,
                    "degenerationMeanRepetitionRate": 0.0,
                    "degenerationMaxRepetitionRate": 0.0,
                    "benchmarks": [],
                },
            },
            {layer_idx: float(layer_idx) for layer_idx in range(candidate.plan.base_layer_count)},
        )

    monkeypatch.setattr(SCRIPT, "evaluate_stage2_candidate", _fake_stage2_candidate)

    output_dir = tmp_path / "scan"
    config = SCRIPT.ScanConfig(
        model_path=model_dir,
        output_dir=output_dir,
        top_k=2,
        max_probes=2,
        behavior_limit_per_benchmark=5,
        max_tokens=32,
    )
    services = _make_services()

    first_summary = SCRIPT.run_scan(config, services=services)

    for artifact_name in (
        "REPORT.md",
        "summary.json",
        "run_manifest.json",
        "probe_manifest.json",
        "ledger.jsonl",
        "variant_results.jsonl",
    ):
        assert (output_dir / artifact_name).exists()

    probe_manifest_before = (output_dir / "probe_manifest.json").read_text(encoding="utf-8")
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    variant_rows = (output_dir / "variant_results.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert first_summary["stage1PlanCount"] == len(variant_rows)
    assert summary_payload["stage2PlanKeys"][0] == "identity"
    assert summary_payload["selected_probe_count"] == 2

    monkeypatch.setattr(
        SCRIPT,
        "load_all_probes",
        lambda: pytest.fail("Frozen probe manifest should be reused on rerun"),
    )
    second_summary = SCRIPT.run_scan(config, services=services)

    assert second_summary["selected_probe_count"] == 2
    assert (output_dir / "probe_manifest.json").read_text(encoding="utf-8") == probe_manifest_before


def test_select_stage2_plan_keys_includes_identity_and_bottom_controls() -> None:
    rows = [
        {"planKey": "identity"},
        {"planKey": "rys:0:1"},
        {"planKey": "rys:0:2"},
        {"planKey": "rys:1:3"},
        {"planKey": "rys:2:4"},
    ]

    selected = SCRIPT.select_stage2_plan_keys(rows, top_k=1)

    assert selected == ["identity", "rys:0:1", "rys:1:3", "rys:2:4"]
