"""Tests for SidecarSafetyPolicy threshold derivation and consent adjustment."""

from datetime import datetime, timedelta, timezone

import pytest

from modelcypher.core.domain.safety.sidecar.sidecar_safety_policy import (
    SidecarSafetyPolicy,
    SidecarSafetyThresholds,
)
from modelcypher.core.domain.safety.sidecar.session_control_state import (
    ConsentGrant,
    ScenarioMode,
    SessionControlState,
)


# ---------------------------------------------------------------------------
# SidecarSafetyThresholds
# ---------------------------------------------------------------------------

class TestSidecarSafetyThresholds:
    """Frozen threshold container."""

    def test_to_dict(self):
        t = SidecarSafetyThresholds(horror_hard=0.1, horror_soft=0.2, sentinel_soft=0.3)
        d = t.to_dict()
        assert d == {"horror_hard": 0.1, "horror_soft": 0.2, "sentinel_soft": 0.3}

    def test_to_dict_sentinel_none(self):
        t = SidecarSafetyThresholds(horror_hard=0.1, horror_soft=0.2, sentinel_soft=None)
        d = t.to_dict()
        assert d["sentinel_soft"] is None

    def test_frozen(self):
        t = SidecarSafetyThresholds(horror_hard=0.1, horror_soft=0.2, sentinel_soft=None)
        with pytest.raises(AttributeError):
            t.horror_hard = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SidecarSafetyPolicy construction
# ---------------------------------------------------------------------------

class TestSidecarSafetyPolicyConstruction:
    """Factory methods and validation."""

    def test_requires_explicit_percentiles(self):
        with pytest.raises(TypeError):
            SidecarSafetyPolicy()  # type: ignore[call-arg]

    def test_from_baseline_valid(self):
        measurements = [0.5, 1.0, 1.5, 2.0, 3.0]
        policy = SidecarSafetyPolicy.from_baseline(
            measurements, hard_percentile=1.0, soft_percentile=5.0
        )
        assert policy.baseline_kl_measurements == measurements

    def test_from_baseline_copies_list(self):
        measurements = [1.0, 2.0]
        policy = SidecarSafetyPolicy.from_baseline(
            measurements, hard_percentile=1.0, soft_percentile=5.0
        )
        measurements.append(999.0)
        assert 999.0 not in policy.baseline_kl_measurements

    def test_from_baseline_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            SidecarSafetyPolicy.from_baseline(
                [], hard_percentile=1.0, soft_percentile=5.0
            )

    def test_from_baseline_requires_explicit_percentiles(self):
        with pytest.raises(TypeError):
            SidecarSafetyPolicy.from_baseline([1.0, 2.0, 3.0])  # type: ignore[call-arg]

    def test_from_baseline_custom_percentiles(self):
        policy = SidecarSafetyPolicy.from_baseline(
            [1.0, 2.0, 3.0], hard_percentile=2.0, soft_percentile=10.0
        )
        assert policy.hard_percentile == 2.0
        assert policy.soft_percentile == 10.0


# ---------------------------------------------------------------------------
# thresholds -- no data
# ---------------------------------------------------------------------------

class TestThresholdsNoData:
    """Thresholds with no baseline and no observed KL return most conservative."""

    def test_no_data_returns_zero(self):
        policy = SidecarSafetyPolicy(hard_percentile=1.0, soft_percentile=5.0)
        t = policy.thresholds()
        assert t.horror_hard == 0.0
        assert t.horror_soft == 0.0

    def test_no_data_sentinel_none(self):
        policy = SidecarSafetyPolicy(hard_percentile=1.0, soft_percentile=5.0)
        t = policy.thresholds()
        assert t.sentinel_soft is None


# ---------------------------------------------------------------------------
# thresholds -- with baseline data
# ---------------------------------------------------------------------------

class TestThresholdsWithBaseline:
    """Percentile-based threshold computation."""

    def test_computed_from_baseline(self):
        # 100 values from 0.0 to 9.9
        measurements = [i * 0.1 for i in range(100)]
        policy = SidecarSafetyPolicy.from_baseline(
            measurements, hard_percentile=1.0, soft_percentile=5.0
        )
        t = policy.thresholds()
        # hard: index = int(1.0 * 100 / 100) = 1 -> sorted_values[1] = 0.1
        assert t.horror_hard == pytest.approx(0.1)
        # soft: index = int(5.0 * 100 / 100) = 5 -> sorted_values[5] = 0.5
        assert t.horror_soft == pytest.approx(0.5)

    def test_sentinel_soft_equals_horror_soft_with_data(self):
        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]
        policy = SidecarSafetyPolicy.from_baseline(
            measurements, hard_percentile=1.0, soft_percentile=5.0
        )
        t = policy.thresholds()
        assert t.sentinel_soft == t.horror_soft

    def test_soft_ge_hard_enforced(self):
        """If percentile math makes soft < hard, soft is bumped to hard."""
        # All identical values -- both percentiles yield same value
        measurements = [1.0] * 20
        policy = SidecarSafetyPolicy.from_baseline(
            measurements, hard_percentile=5.0, soft_percentile=1.0
        )
        t = policy.thresholds()
        assert t.horror_soft >= t.horror_hard

    def test_single_measurement(self):
        """One measurement: both percentiles clamp to index 0."""
        policy = SidecarSafetyPolicy.from_baseline(
            [42.0], hard_percentile=1.0, soft_percentile=5.0
        )
        t = policy.thresholds()
        assert t.horror_hard == 42.0
        assert t.horror_soft == 42.0


# ---------------------------------------------------------------------------
# thresholds -- observed_kl fallback
# ---------------------------------------------------------------------------

class TestThresholdsObservedKL:
    """When no baseline, observed_kl used for online estimation."""

    def test_observed_kl_used_when_no_baseline(self):
        policy = SidecarSafetyPolicy(hard_percentile=1.0, soft_percentile=5.0)
        observed = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        t = policy.thresholds(observed_kl=observed)
        # Should produce non-zero thresholds
        assert t.horror_hard > 0.0

    def test_baseline_takes_precedence_over_observed(self):
        policy = SidecarSafetyPolicy.from_baseline(
            [100.0, 200.0, 300.0], hard_percentile=1.0, soft_percentile=5.0
        )
        t_with = policy.thresholds(observed_kl=[0.001])
        t_without = policy.thresholds()
        # Baseline overrides observed -- thresholds should be the same
        assert t_with.horror_hard == t_without.horror_hard
        assert t_with.horror_soft == t_without.horror_soft


# ---------------------------------------------------------------------------
# thresholds -- consent adjustment
# ---------------------------------------------------------------------------

class TestThresholdsConsent:
    """Consent-gated soft threshold relaxation."""

    @pytest.fixture
    def baseline_policy(self):
        measurements = [i * 0.1 for i in range(100)]
        policy = SidecarSafetyPolicy.from_baseline(
            measurements, hard_percentile=1.0, soft_percentile=5.0
        )
        policy.consent_soft_multiplier = 0.5
        return policy

    @pytest.fixture
    def now(self):
        return datetime.now(timezone.utc)

    @pytest.fixture
    def active_horror_control(self, now):
        return SessionControlState(
            scenario=ScenarioMode.HORROR,
            consent=ConsentGrant(
                is_granted=True,
                expires_at=now + timedelta(hours=1),
            ),
        )

    def test_consent_horror_applies_multiplier(self, baseline_policy, now, active_horror_control):
        t_no_consent = baseline_policy.thresholds()
        t_consent = baseline_policy.thresholds(control=active_horror_control, now=now)
        # Soft threshold should be halved (default multiplier 0.5)
        assert t_consent.horror_soft == pytest.approx(
            t_no_consent.horror_soft * 0.5
        )
        # Hard threshold NEVER changes
        assert t_consent.horror_hard == t_no_consent.horror_hard

    def test_consent_roleplay_applies_multiplier(self, baseline_policy, now):
        control = SessionControlState(
            scenario=ScenarioMode.ROLEPLAY,
            consent=ConsentGrant(is_granted=True, expires_at=now + timedelta(hours=1)),
        )
        t_no = baseline_policy.thresholds()
        t_yes = baseline_policy.thresholds(control=control, now=now)
        assert t_yes.horror_soft == pytest.approx(t_no.horror_soft * 0.5)

    def test_consent_fiction_applies_multiplier(self, baseline_policy, now):
        control = SessionControlState(
            scenario=ScenarioMode.FICTION,
            consent=ConsentGrant(is_granted=True, expires_at=now + timedelta(hours=1)),
        )
        t_no = baseline_policy.thresholds()
        t_yes = baseline_policy.thresholds(control=control, now=now)
        assert t_yes.horror_soft == pytest.approx(t_no.horror_soft * 0.5)

    def test_no_consent_returns_normal(self, baseline_policy, now):
        control = SessionControlState(
            scenario=ScenarioMode.HORROR,
            consent=ConsentGrant(is_granted=False),
        )
        t_no_control = baseline_policy.thresholds()
        t_with = baseline_policy.thresholds(control=control, now=now)
        assert t_with.horror_soft == pytest.approx(t_no_control.horror_soft)

    def test_default_scenario_no_multiplier(self, baseline_policy, now):
        """DEFAULT scenario never triggers soft relaxation even with consent."""
        control = SessionControlState(
            scenario=ScenarioMode.DEFAULT,
            consent=ConsentGrant(is_granted=True, expires_at=now + timedelta(hours=1)),
        )
        t_no = baseline_policy.thresholds()
        t_yes = baseline_policy.thresholds(control=control, now=now)
        assert t_yes.horror_soft == pytest.approx(t_no.horror_soft)

    def test_therapy_scenario_no_multiplier(self, baseline_policy, now):
        """THERAPY scenario is not in the consent-adjustable set."""
        control = SessionControlState(
            scenario=ScenarioMode.THERAPY,
            consent=ConsentGrant(is_granted=True, expires_at=now + timedelta(hours=1)),
        )
        t_no = baseline_policy.thresholds()
        t_yes = baseline_policy.thresholds(control=control, now=now)
        assert t_yes.horror_soft == pytest.approx(t_no.horror_soft)

    def test_sentinel_soft_also_adjusted(self, baseline_policy, now, active_horror_control):
        t_no = baseline_policy.thresholds()
        t_yes = baseline_policy.thresholds(control=active_horror_control, now=now)
        assert t_yes.sentinel_soft is not None
        assert t_no.sentinel_soft is not None
        assert t_yes.sentinel_soft == pytest.approx(
            t_no.sentinel_soft * 0.5
        )

    def test_expired_consent_no_multiplier(self, baseline_policy, now):
        control = SessionControlState(
            scenario=ScenarioMode.HORROR,
            consent=ConsentGrant(
                is_granted=True,
                expires_at=now - timedelta(hours=1),  # expired
            ),
        )
        t_no = baseline_policy.thresholds()
        t_yes = baseline_policy.thresholds(control=control, now=now)
        assert t_yes.horror_soft == pytest.approx(t_no.horror_soft)


# ---------------------------------------------------------------------------
# _compute_percentile edge cases
# ---------------------------------------------------------------------------

class TestComputePercentile:
    """Internal percentile function edge cases."""

    @pytest.fixture
    def policy(self):
        return SidecarSafetyPolicy(hard_percentile=1.0, soft_percentile=5.0)

    def test_empty_list_returns_zero(self, policy):
        assert policy._compute_percentile([], 50.0) == 0.0

    def test_single_element(self, policy):
        assert policy._compute_percentile([5.0], 50.0) == 5.0

    def test_percentile_zero(self, policy):
        assert policy._compute_percentile([1.0, 2.0, 3.0], 0.0) == 1.0

    def test_percentile_100(self, policy):
        # idx = int(100 * 3 / 100) = 3, clamped to 2
        assert policy._compute_percentile([1.0, 2.0, 3.0], 100.0) == 3.0

    def test_unsorted_input_is_sorted(self, policy):
        # Should sort internally: [1, 3, 5]
        result = policy._compute_percentile([5.0, 1.0, 3.0], 50.0)
        # idx = int(50 * 3 / 100) = 1 -> sorted_values[1] = 3.0
        assert result == 3.0


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    """Policy survives serialization and deserialization."""

    def test_no_baseline_round_trip(self):
        policy = SidecarSafetyPolicy(hard_percentile=1.0, soft_percentile=5.0)
        d = policy.to_dict()
        restored = SidecarSafetyPolicy.from_dict(d)
        assert restored.baseline_kl_measurements == []
        assert restored.hard_percentile == 1.0
        assert restored.soft_percentile == 5.0
        assert restored.relax_soft_thresholds_under_consent is True
        assert restored.consent_soft_multiplier is None

    def test_with_baseline_round_trip(self):
        measurements = [0.1, 0.5, 1.0, 2.0]
        policy = SidecarSafetyPolicy.from_baseline(
            measurements, hard_percentile=2.0, soft_percentile=8.0
        )
        policy.relax_soft_thresholds_under_consent = False
        policy.consent_soft_multiplier = 0.75
        d = policy.to_dict()
        restored = SidecarSafetyPolicy.from_dict(d)
        assert restored.baseline_kl_measurements == measurements
        assert restored.hard_percentile == 2.0
        assert restored.soft_percentile == 8.0
        assert restored.relax_soft_thresholds_under_consent is False
        assert restored.consent_soft_multiplier == 0.75

    def test_from_dict_missing_percentiles_raises(self):
        with pytest.raises(ValueError, match="hard_percentile and soft_percentile must be explicitly"):
            SidecarSafetyPolicy.from_dict({})

    def test_from_dict_missing_hard_percentile_raises(self):
        with pytest.raises(ValueError, match="hard_percentile and soft_percentile must be explicitly"):
            SidecarSafetyPolicy.from_dict({"soft_percentile": 5.0})

    def test_thresholds_match_after_round_trip(self):
        measurements = [i * 0.1 for i in range(50)]
        policy = SidecarSafetyPolicy.from_baseline(
            measurements, hard_percentile=1.0, soft_percentile=5.0
        )
        restored = SidecarSafetyPolicy.from_dict(policy.to_dict())
        t_orig = policy.thresholds()
        t_restored = restored.thresholds()
        assert t_orig.horror_hard == t_restored.horror_hard
        assert t_orig.horror_soft == t_restored.horror_soft


# ---------------------------------------------------------------------------
# from_kl_distributions (G5 geometric)
# ---------------------------------------------------------------------------

class TestFromKLDistributions:
    """Geometric threshold derivation from safe/attack KL distributions."""

    def test_well_separated_kl(self):
        """Safe KL [2,5] vs attack KL [0.1,0.5] → boundary between them."""
        safe_kl = [2.0 + i * 0.15 for i in range(20)]  # 2.0..4.85
        attack_kl = [0.1 + i * 0.02 for i in range(20)]  # 0.1..0.48
        policy = SidecarSafetyPolicy.from_kl_distributions(
            safe_kl, attack_kl, seed=42
        )
        t = policy.thresholds()
        # Hard should be between attack max and safe min
        assert t.horror_hard > 0.0
        assert t.horror_soft >= t.horror_hard

    def test_consent_requires_explicit_multiplier(self):
        """from_kl_distributions without multiplier → error on consent thresholds."""
        safe_kl = [2.0, 3.0, 4.0, 5.0, 6.0]
        attack_kl = [0.1, 0.2, 0.3, 0.4, 0.5]
        policy = SidecarSafetyPolicy.from_kl_distributions(
            safe_kl, attack_kl, seed=42
        )
        assert policy.consent_soft_multiplier is None

        now = datetime.now(timezone.utc)
        control = SessionControlState(
            scenario=ScenarioMode.HORROR,
            consent=ConsentGrant(
                is_granted=True,
                expires_at=now + timedelta(hours=1),
            ),
        )
        with pytest.raises(ValueError, match="consent_soft_multiplier must be explicitly"):
            policy.thresholds(control=control, now=now)

    def test_with_explicit_multiplier(self):
        """from_kl_distributions with multiplier → consent thresholds work."""
        safe_kl = [2.0, 3.0, 4.0, 5.0, 6.0]
        attack_kl = [0.1, 0.2, 0.3, 0.4, 0.5]
        policy = SidecarSafetyPolicy.from_kl_distributions(
            safe_kl, attack_kl, seed=42,
            consent_soft_multiplier=0.5,
        )
        assert policy.consent_soft_multiplier == 0.5

        now = datetime.now(timezone.utc)
        control = SessionControlState(
            scenario=ScenarioMode.HORROR,
            consent=ConsentGrant(
                is_granted=True,
                expires_at=now + timedelta(hours=1),
            ),
        )
        t_no = policy.thresholds()
        t_yes = policy.thresholds(control=control, now=now)
        assert t_yes.horror_soft == pytest.approx(t_no.horror_soft * 0.5)

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="Both groups need >= 2"):
            SidecarSafetyPolicy.from_kl_distributions([1.0], [2.0, 3.0])

    def test_no_consent_without_multiplier_works(self):
        """Without consent, None multiplier is fine."""
        safe_kl = [2.0, 3.0, 4.0, 5.0, 6.0]
        attack_kl = [0.1, 0.2, 0.3, 0.4, 0.5]
        policy = SidecarSafetyPolicy.from_kl_distributions(
            safe_kl, attack_kl, seed=42
        )
        t = policy.thresholds()  # No control → no consent check
        assert isinstance(t.horror_hard, float)
        assert isinstance(t.horror_soft, float)
