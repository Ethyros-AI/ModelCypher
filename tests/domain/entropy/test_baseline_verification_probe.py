from __future__ import annotations

import logging

import pytest

import modelcypher.core.domain.entropy.baseline_verification_probe as probe_mod
from modelcypher.core.domain.entropy.baseline_verification_probe import (
    AdapterEntropyBaseline,
    BaselineComparison,
    BaselineVerificationProbe,
    DeltaSample,
    adversarial_prompts,
    default_test_prompts,
)


def _baseline(
    *,
    mean: float,
    std: float,
    max_value: float,
    min_value: float,
    base: str = "base-model",
    count: int = 10,
    conditions: str | None = None,
) -> AdapterEntropyBaseline:
    return AdapterEntropyBaseline(
        delta_mean=mean,
        delta_std_dev=std,
        delta_max=max_value,
        delta_min=min_value,
        base_model_id=base,
        sample_count=count,
        test_conditions=conditions,
    )


def test_adapter_entropy_baseline_roundtrip_and_optional_field() -> None:
    base_without_conditions = _baseline(
        mean=1.0,
        std=0.5,
        max_value=2.0,
        min_value=0.0,
        conditions=None,
    )
    payload = base_without_conditions.to_dict()
    assert payload == {
        "deltaMean": 1.0,
        "deltaStdDev": 0.5,
        "deltaMax": 2.0,
        "deltaMin": 0.0,
        "baseModelId": "base-model",
        "sampleCount": 10,
    }
    assert "testConditions" not in payload

    base_with_conditions = _baseline(
        mean=2.0,
        std=0.25,
        max_value=3.0,
        min_value=1.5,
        conditions="lab-run",
    )
    payload2 = base_with_conditions.to_dict()
    assert payload2["testConditions"] == "lab-run"

    restored = AdapterEntropyBaseline.from_dict(payload2)
    assert isinstance(restored, probe_mod.EntropyBaseline)
    assert restored.delta_mean == pytest.approx(2.0, abs=1e-8)
    assert restored.delta_std_dev == pytest.approx(0.25, abs=1e-8)
    assert restored.delta_max == pytest.approx(3.0, abs=1e-8)
    assert restored.delta_min == pytest.approx(1.5, abs=1e-8)
    assert restored.base_model_id == "base-model"
    assert restored.sample_count == 10
    assert restored.test_conditions == "lab-run"


def test_baseline_comparison_from_baselines(any_backend, monkeypatch) -> None:
    monkeypatch.setattr(probe_mod, "get_default_backend", lambda: any_backend)

    declared = _baseline(mean=1.0, std=0.5, max_value=3.0, min_value=0.0)
    observed = _baseline(mean=2.0, std=0.25, max_value=2.5, min_value=-0.5)

    comparison = BaselineComparison.from_baselines(observed, declared)
    assert comparison.mean_z_score == pytest.approx(2.0, abs=1e-6)
    assert comparison.std_dev_ratio == pytest.approx(0.5, abs=1e-6)
    assert comparison.max_deviation == pytest.approx(0.5, abs=1e-6)
    assert comparison.min_deviation == pytest.approx(0.5, abs=1e-6)
    assert comparison.declared_range == pytest.approx(3.0, abs=1e-6)
    assert comparison.observed_range == pytest.approx(3.0, abs=1e-6)

    declared_zero_std = _baseline(mean=1.0, std=0.0, max_value=3.0, min_value=0.0)
    comparison_zero_std = BaselineComparison.from_baselines(observed, declared_zero_std)
    assert comparison_zero_std.mean_z_score > 0.0
    assert comparison_zero_std.std_dev_ratio > 0.0


def test_quick_verify_sync_with_precollected_samples(any_backend, monkeypatch) -> None:
    monkeypatch.setattr(probe_mod, "get_default_backend", lambda: any_backend)

    probe = BaselineVerificationProbe(test_prompts=("p1", "p2"))
    declared = _baseline(mean=1.0, std=0.5, max_value=3.0, min_value=0.0)
    samples = [
        DeltaSample(token_index=0, delta=1.0, anomaly_score=0.2),
        DeltaSample(token_index=1, delta=2.0, anomaly_score=0.5),
        DeltaSample(token_index=2, delta=3.0, anomaly_score=0.1),
    ]

    result = probe.quick_verify_sync(
        declared_baseline=declared,
        observed_samples=samples,
        adapter_path="/tmp/adapter",
        base_model_path="/tmp/base",
    )

    assert result.total_samples == 3
    assert result.prompt_results == ()
    assert result.verification_duration == pytest.approx(0.0, abs=1e-8)

    observed = result.observed_baseline
    assert observed.delta_mean == pytest.approx(2.0, abs=1e-6)
    assert observed.delta_std_dev == pytest.approx(1.0, abs=1e-6)
    assert observed.delta_max == pytest.approx(3.0, abs=1e-6)
    assert observed.delta_min == pytest.approx(1.0, abs=1e-6)
    assert observed.base_model_id == "/tmp/base"

    assert result.comparison.mean_z_score == pytest.approx(2.0, abs=1e-6)


async def test_verify_async_path_with_hook(any_backend, monkeypatch) -> None:
    monkeypatch.setattr(probe_mod, "get_default_backend", lambda: any_backend)

    prompts = (
        "safe prompt",
        adversarial_prompts()[0],
    )
    probe = BaselineVerificationProbe(test_prompts=prompts)
    declared = _baseline(mean=2.0, std=1.0, max_value=3.0, min_value=1.0)

    async def inference_hook(prompt: str) -> list[DeltaSample]:
        if prompt == "safe prompt":
            return [
                DeltaSample(0, 1.0, 0.1),
                DeltaSample(1, 2.0, 0.2),
            ]
        return [DeltaSample(0, 3.0, 0.9)]

    result = await probe.verify(
        adapter_path="/tmp/adapter",
        base_model_path="/tmp/base",
        declared_baseline=declared,
        inference_hook=inference_hook,
    )

    assert len(result.prompt_results) == 2
    assert result.total_samples == 3

    first, second = result.prompt_results
    assert first.token_count == 2
    assert first.avg_delta == pytest.approx(1.5, abs=1e-6)
    assert first.max_anomaly_score == pytest.approx(0.2, abs=1e-6)
    assert first.is_adversarial is False

    assert second.token_count == 1
    assert second.avg_delta == pytest.approx(3.0, abs=1e-6)
    assert second.max_anomaly_score == pytest.approx(0.9, abs=1e-6)
    assert second.is_adversarial is True


async def test_verify_async_none_and_failing_hook(any_backend, monkeypatch, caplog) -> None:
    monkeypatch.setattr(probe_mod, "get_default_backend", lambda: any_backend)

    prompts = ("first", adversarial_prompts()[1])
    probe = BaselineVerificationProbe(test_prompts=prompts)
    declared = _baseline(mean=0.0, std=1.0, max_value=1.0, min_value=-1.0)

    none_result = await probe.verify(
        adapter_path="/tmp/adapter",
        base_model_path="/tmp/base",
        declared_baseline=declared,
        inference_hook=None,
    )
    assert none_result.total_samples == 0
    assert all(pr.token_count == 0 for pr in none_result.prompt_results)
    assert all(pr.avg_delta == pytest.approx(0.0, abs=1e-8) for pr in none_result.prompt_results)
    assert none_result.observed_baseline.base_model_id == "unknown"

    async def flaky_hook(prompt: str) -> list[DeltaSample]:
        if prompt == "first":
            raise RuntimeError("boom")
        return [DeltaSample(0, 0.7, 0.4)]

    with caplog.at_level(logging.WARNING):
        flaky_result = await probe.verify(
            adapter_path="/tmp/adapter",
            base_model_path="/tmp/base",
            declared_baseline=declared,
            inference_hook=flaky_hook,
        )

    assert flaky_result.total_samples == 1
    assert flaky_result.prompt_results[0].token_count == 0
    assert flaky_result.prompt_results[1].token_count == 1
    assert any("inference failed" in rec.message for rec in caplog.records)


def test_default_and_adversarial_prompt_sets_do_not_overlap() -> None:
    default_prompts = default_test_prompts()
    adversarial = adversarial_prompts()

    assert default_prompts
    assert adversarial
    assert set(default_prompts).isdisjoint(set(adversarial))


def test_compute_observed_baseline_edge_cases(any_backend, monkeypatch) -> None:
    monkeypatch.setattr(probe_mod, "get_default_backend", lambda: any_backend)

    probe = BaselineVerificationProbe(test_prompts=("p1", "p2", "p3"))

    empty = probe._compute_observed_baseline([], "base-id")
    assert empty.delta_mean == pytest.approx(0.0, abs=1e-8)
    assert empty.delta_std_dev == pytest.approx(0.0, abs=1e-8)
    assert empty.delta_max == pytest.approx(0.0, abs=1e-8)
    assert empty.delta_min == pytest.approx(0.0, abs=1e-8)
    assert empty.base_model_id == "unknown"
    assert empty.sample_count == 0

    single = probe._compute_observed_baseline([DeltaSample(0, 1.5, 0.1)], "base-id")
    assert single.delta_mean == pytest.approx(1.5, abs=1e-6)
    assert single.delta_std_dev == pytest.approx(0.0, abs=1e-8)
    assert single.delta_max == pytest.approx(1.5, abs=1e-6)
    assert single.delta_min == pytest.approx(1.5, abs=1e-6)
    assert single.base_model_id == "base-id"

    multi = probe._compute_observed_baseline(
        [
            DeltaSample(0, 1.0, 0.0),
            DeltaSample(1, 2.0, 0.0),
            DeltaSample(2, 3.0, 0.0),
        ],
        "base-id",
    )
    assert multi.delta_mean == pytest.approx(2.0, abs=1e-6)
    assert multi.delta_std_dev == pytest.approx(1.0, abs=1e-6)
    assert multi.delta_max == pytest.approx(3.0, abs=1e-6)
    assert multi.delta_min == pytest.approx(1.0, abs=1e-6)
