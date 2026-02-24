from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

import pytest

from modelcypher.core.domain.entropy.entropy_delta_sample import (
    BaselineDistribution,
    EntropyDeltaSample,
    EntropyDeltaSessionResult,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_sample(
    *,
    base_entropy: float = 5.0,
    adapter_entropy: float = 3.0,
    base_logit_variance: float = 2.0,
    adapter_logit_variance: float = 1.0,
    base_top_token: int = 100,
    adapter_top_token: int = 100,
    latency_ms: float = 0.0,
) -> EntropyDeltaSample:
    """Create an EntropyDeltaSample via the factory method."""
    return EntropyDeltaSample.create(
        token_index=0,
        generated_token=42,
        base_entropy=base_entropy,
        base_logit_variance=base_logit_variance,
        base_top_token=base_top_token,
        adapter_entropy=adapter_entropy,
        adapter_logit_variance=adapter_logit_variance,
        adapter_top_token=adapter_top_token,
        latency_ms=latency_ms,
    )


# =============================================================================
# BaselineDistribution
# =============================================================================


class TestBaselineDistributionFromSamples:
    """Verify BaselineDistribution.from_samples classmethod."""

    def test_from_samples_valid_data(self) -> None:
        bd = BaselineDistribution.from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
        assert bd.mean == pytest.approx(3.0)
        # Population std: sqrt(2.0)
        assert bd.std == pytest.approx(2.0**0.5, abs=1e-4)

    def test_from_samples_single_value(self) -> None:
        bd = BaselineDistribution.from_samples([7.0])
        assert bd.mean == pytest.approx(7.0)
        assert bd.std == pytest.approx(0.0)

    def test_from_samples_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            BaselineDistribution.from_samples([])

    def test_from_samples_identical_values(self) -> None:
        bd = BaselineDistribution.from_samples([4.0, 4.0, 4.0])
        assert bd.mean == pytest.approx(4.0)
        assert bd.std == pytest.approx(0.0)


class TestBaselineDistributionZScore:
    """Verify BaselineDistribution.z_score method."""

    def test_z_score_normal_distribution(self) -> None:
        bd = BaselineDistribution(mean=10.0, std=2.0)
        # (14 - 10) / 2 = 2.0
        assert bd.z_score(14.0) == pytest.approx(2.0)

    def test_z_score_negative(self) -> None:
        bd = BaselineDistribution(mean=10.0, std=2.0)
        # (6 - 10) / 2 = -2.0
        assert bd.z_score(6.0) == pytest.approx(-2.0)

    def test_z_score_at_mean_is_zero(self) -> None:
        bd = BaselineDistribution(mean=5.0, std=1.0)
        assert bd.z_score(5.0) == pytest.approx(0.0)

    def test_z_score_near_zero_std_returns_zero_when_at_mean(self) -> None:
        bd = BaselineDistribution(mean=5.0, std=0.0)
        # std < eps, value == mean -> within eps -> returns 0.0
        assert bd.z_score(5.0) == pytest.approx(0.0)

    def test_z_score_near_zero_std_returns_inf_when_away_from_mean(self) -> None:
        bd = BaselineDistribution(mean=5.0, std=0.0)
        # std < eps, value != mean -> returns inf
        result = bd.z_score(10.0)
        assert result == float("inf")


# =============================================================================
# EntropyDeltaSample
# =============================================================================


class TestEntropyDeltaSampleCreate:
    """Verify the static factory generates a valid UUID."""

    def test_create_generates_uuid(self) -> None:
        sample = _make_sample()
        assert isinstance(sample.id, UUID)

    def test_create_generates_unique_uuids(self) -> None:
        s1 = _make_sample()
        s2 = _make_sample()
        assert s1.id != s2.id

    def test_create_stores_all_fields(self) -> None:
        sample = EntropyDeltaSample.create(
            token_index=5,
            generated_token=99,
            base_entropy=6.0,
            base_logit_variance=3.0,
            base_top_token=200,
            adapter_entropy=4.0,
            adapter_logit_variance=2.0,
            adapter_top_token=201,
            latency_ms=1.5,
            source="test",
        )
        assert sample.token_index == 5
        assert sample.generated_token == 99
        assert sample.base_entropy == 6.0
        assert sample.adapter_entropy == 4.0
        assert sample.latency_ms == 1.5
        assert sample.source == "test"


class TestEntropyDeltaSampleDelta:
    """Verify delta property (base_entropy - adapter_entropy)."""

    def test_delta_positive_base_higher(self) -> None:
        sample = _make_sample(base_entropy=8.0, adapter_entropy=3.0)
        assert sample.delta == pytest.approx(5.0)

    def test_delta_negative_adapter_higher(self) -> None:
        sample = _make_sample(base_entropy=2.0, adapter_entropy=7.0)
        assert sample.delta == pytest.approx(-5.0)

    def test_delta_zero_equal(self) -> None:
        sample = _make_sample(base_entropy=4.0, adapter_entropy=4.0)
        assert sample.delta == pytest.approx(0.0)


class TestEntropyDeltaSampleTopTokenDisagreement:
    """Verify top_token_disagreement property."""

    def test_tokens_differ(self) -> None:
        sample = _make_sample(base_top_token=100, adapter_top_token=200)
        assert sample.top_token_disagreement is True

    def test_tokens_match(self) -> None:
        sample = _make_sample(base_top_token=100, adapter_top_token=100)
        assert sample.top_token_disagreement is False


class TestEntropyDeltaSampleAnomalyScore:
    """Verify anomaly_score property: max(delta,0) / max(base_entropy, eps)."""

    def test_positive_delta(self) -> None:
        sample = _make_sample(base_entropy=10.0, adapter_entropy=4.0)
        # delta=6.0, anomaly_score = 6.0 / 10.0 = 0.6
        assert sample.anomaly_score == pytest.approx(0.6)

    def test_negative_delta_clamped_to_zero(self) -> None:
        sample = _make_sample(base_entropy=3.0, adapter_entropy=8.0)
        # delta=-5.0, max(delta,0)=0.0 -> anomaly_score = 0.0
        assert sample.anomaly_score == pytest.approx(0.0)

    def test_zero_delta(self) -> None:
        sample = _make_sample(base_entropy=5.0, adapter_entropy=5.0)
        assert sample.anomaly_score == pytest.approx(0.0)

    def test_near_zero_base_entropy(self) -> None:
        sample = _make_sample(base_entropy=0.0, adapter_entropy=0.0)
        # delta=0.0, max(0,0)=0 -> anomaly_score = 0/eps = 0.0
        assert sample.anomaly_score == pytest.approx(0.0)

    def test_near_zero_base_entropy_positive_delta(self) -> None:
        # base_entropy ~0 but adapter_entropy < 0 would be unusual,
        # so use small positive base and even smaller adapter
        sample = _make_sample(base_entropy=1e-12, adapter_entropy=0.0)
        # delta = 1e-12, denom = eps (since base_entropy < eps)
        # Result will be a small positive number, not inf
        score = sample.anomaly_score
        assert score >= 0.0


class TestEntropyDeltaSampleVarianceDelta:
    """Verify variance_delta property (base - adapter logit variance)."""

    def test_variance_delta_computed(self) -> None:
        sample = _make_sample(base_logit_variance=5.0, adapter_logit_variance=2.0)
        assert sample.variance_delta == pytest.approx(3.0)

    def test_variance_delta_negative(self) -> None:
        sample = _make_sample(base_logit_variance=1.0, adapter_logit_variance=4.0)
        assert sample.variance_delta == pytest.approx(-3.0)


# =============================================================================
# EntropyDeltaSessionResult
# =============================================================================


def _make_session_result(
    *,
    samples: list[EntropyDeltaSample] | None = None,
    duration_seconds: float = 10.0,
) -> EntropyDeltaSessionResult:
    """Create an EntropyDeltaSessionResult for testing."""
    from uuid import uuid4

    start = datetime(2026, 1, 1, 12, 0, 0)
    end = start + timedelta(seconds=duration_seconds)
    return EntropyDeltaSessionResult(
        session_id=uuid4(),
        correlation_id=uuid4(),
        session_start=start,
        session_end=end,
        total_tokens=100,
        anomaly_count=5,
        max_anomaly_score=0.8,
        avg_delta=0.3,
        disagreement_rate=0.1,
        samples=samples or [],
    )


class TestEntropyDeltaSessionResultDuration:
    """Verify duration property returns seconds."""

    def test_duration_ten_seconds(self) -> None:
        result = _make_session_result(duration_seconds=10.0)
        assert result.duration == pytest.approx(10.0)

    def test_duration_fractional(self) -> None:
        result = _make_session_result(duration_seconds=0.5)
        assert result.duration == pytest.approx(0.5)

    def test_duration_zero(self) -> None:
        result = _make_session_result(duration_seconds=0.0)
        assert result.duration == pytest.approx(0.0)


class TestEntropyDeltaSessionResultAvgLatency:
    """Verify avg_latency_ms property."""

    def test_no_samples_returns_zero(self) -> None:
        result = _make_session_result(samples=[])
        assert result.avg_latency_ms == pytest.approx(0.0)

    def test_with_samples(self) -> None:
        s1 = _make_sample(latency_ms=10.0)
        s2 = _make_sample(latency_ms=20.0)
        s3 = _make_sample(latency_ms=30.0)
        result = _make_session_result(samples=[s1, s2, s3])
        assert result.avg_latency_ms == pytest.approx(20.0)

    def test_single_sample(self) -> None:
        s = _make_sample(latency_ms=7.5)
        result = _make_session_result(samples=[s])
        assert result.avg_latency_ms == pytest.approx(7.5)


class TestEntropyDeltaSessionResultSecurityZScore:
    """Verify security_z_score delegates to baseline.z_score(max_anomaly_score)."""

    def test_security_z_score(self) -> None:
        result = _make_session_result()
        baseline = BaselineDistribution(mean=0.3, std=0.2)
        # (0.8 - 0.3) / 0.2 = 2.5
        assert result.security_z_score(baseline) == pytest.approx(2.5)
