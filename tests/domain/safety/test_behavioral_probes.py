# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import modelcypher.core.domain.safety.behavioral_probes as probes_mod
from modelcypher.core.domain.safety.adapter_safety_models import (
    AdapterSafetyTier,
    AdapterSafetyTrigger,
)
from modelcypher.core.domain.safety.adapter_safety_probe import (
    AdapterSafetyProbe,
    ProbeContext,
    SafetyProbeResult,
)


class _Embedder:
    pass


class _Hook:
    def __init__(self, responses: dict[str, str], fail_prompts: set[str] | None = None):
        self.responses = responses
        self.fail_prompts = fail_prompts or set()

    async def generate(self, prompt: str) -> str:
        if prompt in self.fail_prompts:
            raise RuntimeError("boom")
        return self.responses.get(prompt, "")


@dataclass(frozen=True)
class _Probe:
    probe_id: str
    support_texts: tuple[str, ...]
    description: str
    name: str


def _context(*, hook=None, embedder=None) -> ProbeContext:
    return ProbeContext(
        adapter_path=Path("/tmp/adapter"),
        tier=AdapterSafetyTier.STANDARD,
        trigger=AdapterSafetyTrigger.MANUAL_RESCAN,
        inference_hook=hook,
        embedder=embedder,
    )


def test_distance_threshold_and_geodesic_helpers(any_backend) -> None:
    b = any_backend

    assert probes_mod._distance_threshold([]) == 0.0
    assert probes_mod._distance_threshold([1.0, 1.0, 1.0]) == float("inf")

    threshold = probes_mod._distance_threshold([1.0, 1.2, 5.0])
    assert threshold >= 1.0

    assert probes_mod._geodesic_min_distance([], b.array([0.0, 0.0])) == 0.0

    d = probes_mod._geodesic_min_distance(
        anchor_points=[[0.0, 0.0], [2.0, 0.0]],
        query=[1.0, 0.0],
    )
    assert d >= 0.0


def test_anchor_embedding_paths(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(probes_mod, "get_default_backend", lambda: b)
    monkeypatch.setattr(
        probes_mod,
        "get_or_compute_embeddings_sync",
        lambda *_args, **_kwargs: b.array([]),
    )
    assert probes_mod._anchor_embedding(_Embedder(), ("a", "b")) == []

    monkeypatch.setattr(
        probes_mod,
        "get_or_compute_embeddings_sync",
        lambda *_args, **_kwargs: b.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    anchor = probes_mod._anchor_embedding(_Embedder(), ("a", "b"))
    assert len(anchor) == 2
    assert anchor[0] == pytest.approx(anchor[1], abs=1e-6)


async def test_semantic_drift_probe_missing_context_and_outliers(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(probes_mod, "get_default_backend", lambda: b)

    probe = probes_mod.SemanticDriftProbe(
        probes=[
            _Probe("p1", ("prompt1",), "d1", "n1"),
            _Probe("p2", ("prompt2",), "d2", "n2"),
        ]
    )

    missing = await probe.evaluate(_context(hook=None, embedder=None))
    assert missing.finding_counts["missing_inference"] == 1
    assert missing.finding_counts["missing_embedder"] == 1

    monkeypatch.setattr(probes_mod, "_anchor_embedding", lambda *_args, **_kwargs: [0.0, 0.0])
    monkeypatch.setattr(
        probes_mod,
        "get_or_compute_embeddings_sync",
        lambda *_args, **_kwargs: b.array([[0.0, 0.0]]),
    )
    distances = iter([0.1, 3.0])
    monkeypatch.setattr(probes_mod, "_geodesic_min_distance", lambda *_args, **_kwargs: next(distances))

    result = await probe.evaluate(
        _context(
            hook=_Hook({"prompt1": "r1", "prompt2": "r2"}),
            embedder=_Embedder(),
        )
    )
    assert result.finding_counts["probes_tested"] == 2
    assert result.finding_counts["outlier_probes"] == 1
    assert any("p2" in finding for finding in result.findings)


async def test_canary_probe_and_runner(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(probes_mod, "get_default_backend", lambda: b)

    canary = probes_mod.CanaryQAProbe()
    missing = await canary.evaluate(_context(hook=None, embedder=None))
    assert missing.finding_counts["missing_inference"] == 1

    monkeypatch.setattr(probes_mod, "_anchor_embedding", lambda *_args, **_kwargs: [0.0, 0.0])
    monkeypatch.setattr(
        probes_mod,
        "get_or_compute_embeddings_sync",
        lambda *_args, **_kwargs: b.array([[0.0, 0.0]]),
    )
    monkeypatch.setattr(probes_mod, "_geodesic_min_distance", lambda *_args, **_kwargs: 0.5)

    result = await canary.evaluate(
        _context(
            hook=_Hook(
                {q.prompt: "ok" for q in probes_mod.CanaryQAProbe.CANARY_QUESTIONS},
                fail_prompts={probes_mod.CanaryQAProbe.CANARY_QUESTIONS[0].prompt},
            ),
            embedder=_Embedder(),
        )
    )
    assert result.finding_counts["questions_tested"] == len(probes_mod.CanaryQAProbe.CANARY_QUESTIONS) - 1
    assert any("canary_inference_error" in finding for finding in result.findings)

    class _SkippingProbe(AdapterSafetyProbe):
        @property
        def name(self) -> str:
            return "skipper"

        @property
        def version(self) -> str:
            return "v"

        @property
        def supported_tiers(self):
            return frozenset({AdapterSafetyTier.FULL})

        async def evaluate(self, context: ProbeContext) -> SafetyProbeResult:
            return SafetyProbeResult(probe_name=self.name, probe_version=self.version)

    class _FailingProbe(AdapterSafetyProbe):
        @property
        def name(self) -> str:
            return "failing"

        @property
        def version(self) -> str:
            return "v"

        @property
        def supported_tiers(self):
            return frozenset({AdapterSafetyTier.STANDARD})

        async def evaluate(self, context: ProbeContext) -> SafetyProbeResult:
            raise RuntimeError("probe-crash")

    runner = probes_mod.ProbeRunner()
    composite = await runner.run([_SkippingProbe(), _FailingProbe()], _context(hook=_Hook({}), embedder=_Embedder()))
    assert composite.total_probes == 1
    assert composite.findings_probe_count == 1
    assert composite.aggregate_finding_counts["execution_errors"] == 1

