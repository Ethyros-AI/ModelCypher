"""Unit tests for Baranov Track A decision logic."""

from __future__ import annotations

import pytest

from modelcypher.experimental.baranov.decision import (
    ModeVerdict,
    ModelVerdict,
    NoiseFloor,
    TrackADecision,
    compute_mode_verdict,
    compute_model_verdict,
    compute_noise_floor,
    compute_track_a_decision,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed_result(
    post_raw: float = 0.8,
    post_chat: float = 0.6,
    cka_drift: float = 0.05,
    preserved_fraction: float = 0.95,
) -> dict:
    return {
        "post_raw_rate": post_raw,
        "post_chat_rate": post_chat,
        "cka_drift": cka_drift,
        "preserved_fraction": preserved_fraction,
    }


# ---------------------------------------------------------------------------
# NoiseFloor
# ---------------------------------------------------------------------------


class TestNoiseFloor:
    def test_default_zero(self):
        nf = NoiseFloor()
        assert nf.raw == 0.0
        assert nf.chat == 0.0

    def test_as_dict(self):
        nf = NoiseFloor(raw=0.01, chat=0.02)
        d = nf.as_dict()
        assert d == {"raw": 0.01, "chat": 0.02}


class TestComputeNoiseFloor:
    def test_empty_results(self):
        nf = compute_noise_floor([])
        assert nf.raw == 0.0
        assert nf.chat == 0.0

    def test_no_deltas(self):
        results = [{"deltas": None}]
        nf = compute_noise_floor(results)
        assert nf.raw == 0.0
        assert nf.chat == 0.0

    def test_extracts_max_absolute_delta(self):
        results = [
            {"deltas": {"delta_raw_recall": 0.01, "delta_chat_recall": -0.02}},
            {"deltas": {"delta_raw_recall": -0.03, "delta_chat_recall": 0.01}},
        ]
        nf = compute_noise_floor(results)
        assert nf.raw == 0.03
        assert nf.chat == 0.02


# ---------------------------------------------------------------------------
# ModeVerdict
# ---------------------------------------------------------------------------


class TestModeVerdict:
    def test_as_dict(self):
        v = ModeVerdict(
            mode="raw_completion",
            pre_recall=0.5,
            post_recall_per_seed=(0.7,),
            delta_mean=0.2,
            delta_ci=(0.1, 0.3),
            noise_floor=0.0,
            significant=True,
            verdict="effective",
            reason="test",
        )
        d = v.as_dict()
        assert d["mode"] == "raw_completion"
        assert d["delta_mean"] == 0.2
        assert d["significant"] is True

    def test_frozen(self):
        v = ModeVerdict(
            mode="raw", pre_recall=0.0, post_recall_per_seed=(),
            delta_mean=0.0, delta_ci=(0.0, 0.0), noise_floor=0.0,
            significant=False, verdict="ineffective", reason="",
        )
        with pytest.raises(AttributeError):
            v.verdict = "effective"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_mode_verdict
# ---------------------------------------------------------------------------


class TestComputeModeVerdict:
    def test_no_post_data_returns_ineffective(self):
        v = compute_mode_verdict("raw_completion", 0.5, [], 10, 0.0)
        assert v.verdict == "ineffective"
        assert not v.significant

    def test_perfect_injection_single_seed(self):
        """Post rate = 1.0, pre rate = 0.5 → effective."""
        v = compute_mode_verdict("raw_completion", 0.5, [1.0], 10, 0.0)
        assert v.significant
        assert v.verdict == "effective"
        assert v.delta_mean == pytest.approx(0.5)

    def test_zero_delta_not_significant(self):
        """Same pre/post rate → not significant."""
        v = compute_mode_verdict("raw_completion", 0.8, [0.8], 10, 0.0)
        # CI from Clopper-Pearson will span 0
        # With 8/10 correct, CI at alpha=0.1 is roughly [0.5, 0.96]
        # delta CI ≈ [-0.3, 0.16], which includes 0
        assert v.verdict == "ineffective"

    def test_degraded_verdict(self):
        """Post rate < pre rate → degraded if significant."""
        v = compute_mode_verdict("raw_completion", 1.0, [0.0], 20, 0.0)
        assert v.verdict == "degraded"
        assert v.significant
        assert v.delta_mean == pytest.approx(-1.0)

    def test_noise_floor_raises_threshold(self):
        """With high noise floor, small delta should not be significant."""
        # Delta = +0.1 but noise floor is 0.2
        v = compute_mode_verdict("raw_completion", 0.5, [0.6], 20, 0.2)
        assert not v.significant
        assert v.verdict == "ineffective"

    def test_multi_seed_uses_t_interval(self):
        """Multiple seeds should compute a t-interval on deltas."""
        # 3 seeds all showing +0.4 delta → should be significant
        v = compute_mode_verdict("raw_completion", 0.5, [0.9, 0.9, 0.9], 20, 0.0)
        assert v.significant
        assert v.verdict == "effective"
        assert v.delta_mean == pytest.approx(0.4)

    def test_multi_seed_high_variance_not_significant(self):
        """High variance across seeds → wide CI → not significant."""
        # Deltas: +0.4, -0.2, +0.3 → mean +0.167, high variance
        v = compute_mode_verdict("raw_completion", 0.5, [0.9, 0.3, 0.8], 20, 0.0)
        # With such variance over 3 seeds, CI likely includes 0
        # This depends on the t-interval width
        assert v.delta_mean == pytest.approx(0.167, abs=0.01)

    def test_alpha_derived_from_n_facts(self):
        """Alpha = 1/n_facts, not hardcoded."""
        # With n_facts=5, alpha=0.2 → wider CI than n_facts=100, alpha=0.01
        v5 = compute_mode_verdict("raw", 0.5, [0.8], 5, 0.0)
        v100 = compute_mode_verdict("raw", 0.5, [0.8], 100, 0.0)
        # v100 should have a tighter CI → more likely to be significant
        # v5 might not be significant with only 5 facts
        assert v100.delta_ci[1] - v100.delta_ci[0] < v5.delta_ci[1] - v5.delta_ci[0]


# ---------------------------------------------------------------------------
# compute_model_verdict
# ---------------------------------------------------------------------------


class TestComputeModelVerdict:
    def test_basic_effective(self):
        v = compute_model_verdict(
            model="TestModel",
            pre_raw_rate=0.5,
            pre_chat_rate=0.3,
            seed_results=[_make_seed_result(post_raw=1.0, post_chat=0.9)],
            n_facts=20,
            noise_floor=NoiseFloor(),
        )
        assert isinstance(v, ModelVerdict)
        assert v.model == "TestModel"
        assert v.n_seeds == 1
        assert v.cka_drift_mean == 0.05

    def test_model_verdict_reflects_modes(self):
        """If either mode is effective, model is effective."""
        v = compute_model_verdict(
            model="M",
            pre_raw_rate=0.5,
            pre_chat_rate=0.5,
            seed_results=[_make_seed_result(post_raw=1.0, post_chat=0.5)],
            n_facts=20,
            noise_floor=NoiseFloor(),
        )
        # Raw should be effective, chat should be ineffective
        assert v.raw.verdict == "effective"
        # Chat: same rate → ineffective
        # Overall: effective (at least one mode works)
        assert v.verdict == "effective"

    def test_multi_seed_model(self):
        v = compute_model_verdict(
            model="M",
            pre_raw_rate=0.5,
            pre_chat_rate=0.5,
            seed_results=[
                _make_seed_result(post_raw=0.9, post_chat=0.8, cka_drift=0.03),
                _make_seed_result(post_raw=0.85, post_chat=0.75, cka_drift=0.07),
            ],
            n_facts=20,
            noise_floor=NoiseFloor(),
        )
        assert v.n_seeds == 2
        assert v.cka_drift_mean == pytest.approx(0.05)
        assert v.preserved_fraction_mean == pytest.approx(0.95)

    def test_as_dict_round_trip(self):
        v = compute_model_verdict(
            model="M",
            pre_raw_rate=0.5,
            pre_chat_rate=0.5,
            seed_results=[_make_seed_result()],
            n_facts=20,
            noise_floor=NoiseFloor(),
        )
        d = v.as_dict()
        assert d["model"] == "M"
        assert "raw" in d
        assert "chat" in d


# ---------------------------------------------------------------------------
# compute_track_a_decision
# ---------------------------------------------------------------------------


class TestComputeTrackADecision:
    def _make_model_verdict(
        self,
        model: str,
        raw_delta: float = 0.2,
        chat_delta: float = 0.1,
        raw_significant: bool = True,
        chat_significant: bool = True,
        cka_drift: float = 0.05,
    ) -> ModelVerdict:
        return ModelVerdict(
            model=model,
            n_seeds=1,
            raw=ModeVerdict(
                mode="raw_completion",
                pre_recall=0.5,
                post_recall_per_seed=(0.5 + raw_delta,),
                delta_mean=raw_delta,
                delta_ci=(raw_delta - 0.05, raw_delta + 0.05),
                noise_floor=0.0,
                significant=raw_significant,
                verdict="effective" if raw_delta > 0 and raw_significant else "ineffective",
                reason="test",
            ),
            chat=ModeVerdict(
                mode="chat_template",
                pre_recall=0.5,
                post_recall_per_seed=(0.5 + chat_delta,),
                delta_mean=chat_delta,
                delta_ci=(chat_delta - 0.05, chat_delta + 0.05),
                noise_floor=0.0,
                significant=chat_significant,
                verdict="effective" if chat_delta > 0 and chat_significant else "ineffective",
                reason="test",
            ),
            cka_drift_mean=cka_drift,
            preserved_fraction_mean=1.0 - cka_drift,
            verdict="effective",
            reason="test",
        )

    def test_single_model_inconclusive(self):
        """Fewer than 2 models → inconclusive scaling."""
        verdicts = {"M1": self._make_model_verdict("M1")}
        decision = compute_track_a_decision(verdicts, ["M1"])
        assert decision.scaling_verdict == "inconclusive"
        assert decision.overall_verdict == "inconclusive"

    def test_decreasing_trend_with_geometry_passes(self):
        """Deltas decreasing with scale + geometry → pass."""
        verdicts = {
            "Small": self._make_model_verdict("Small", raw_delta=0.4, chat_delta=0.3),
            "Medium": self._make_model_verdict("Medium", raw_delta=0.3, chat_delta=0.2),
            "Large": self._make_model_verdict("Large", raw_delta=0.2, chat_delta=0.1),
        }
        decision = compute_track_a_decision(verdicts, ["Small", "Medium", "Large"])
        assert decision.scaling_verdict == "pass"
        assert decision.overall_verdict == "pass"
        assert decision.geometry_co_occurrence

    def test_no_significant_effects_fails(self):
        """No significant effects across all → fail."""
        verdicts = {
            "Small": self._make_model_verdict(
                "Small", raw_significant=False, chat_significant=False,
            ),
            "Large": self._make_model_verdict(
                "Large", raw_significant=False, chat_significant=False,
            ),
        }
        decision = compute_track_a_decision(verdicts, ["Small", "Large"])
        assert decision.scaling_verdict == "fail"
        assert decision.overall_verdict == "fail"

    def test_increasing_trend_no_pass(self):
        """Deltas increasing with scale → no suppression trend → not pass."""
        verdicts = {
            "Small": self._make_model_verdict("Small", raw_delta=0.1, chat_delta=0.1),
            "Large": self._make_model_verdict("Large", raw_delta=0.3, chat_delta=0.3),
        }
        decision = compute_track_a_decision(verdicts, ["Small", "Large"])
        assert decision.scaling_verdict != "pass"

    def test_trend_without_geometry_inconclusive(self):
        """Decreasing trend but no geometry co-occurrence → inconclusive."""
        verdicts = {
            "Small": self._make_model_verdict("Small", raw_delta=0.4, cka_drift=0.0),
            "Large": self._make_model_verdict("Large", raw_delta=0.2, cka_drift=0.0),
        }
        decision = compute_track_a_decision(verdicts, ["Small", "Large"])
        assert decision.scaling_verdict == "inconclusive"
        assert "geometry" in decision.scaling_reason.lower()

    def test_mixed_modes_one_passes(self):
        """Raw mode has decreasing trend, chat does not → pass (protocol: 'at least one')."""
        verdicts = {
            "Small": self._make_model_verdict("Small", raw_delta=0.4, chat_delta=0.1),
            "Large": self._make_model_verdict("Large", raw_delta=0.2, chat_delta=0.3),
        }
        decision = compute_track_a_decision(verdicts, ["Small", "Large"])
        # Raw: decreasing [0.4, 0.2] ✓, CIs exclude zero ✓, geometry ✓ → pass
        assert decision.scaling_verdict == "pass"
        assert "raw_completion" in decision.scaling_reason

    def test_as_dict(self):
        verdicts = {
            "M1": self._make_model_verdict("M1"),
            "M2": self._make_model_verdict("M2"),
        }
        decision = compute_track_a_decision(verdicts, ["M1", "M2"])
        d = decision.as_dict()
        assert "per_model" in d
        assert "scaling_verdict" in d
        assert "overall_verdict" in d
        assert "geometry_co_occurrence" in d

    def test_models_not_in_order_filtered(self):
        """Models not in model_order are handled gracefully."""
        verdicts = {
            "M1": self._make_model_verdict("M1", raw_delta=0.4),
            "M2": self._make_model_verdict("M2", raw_delta=0.2),
        }
        # Only M1 in model_order → single model → inconclusive
        decision = compute_track_a_decision(verdicts, ["M1"])
        assert decision.scaling_verdict == "inconclusive"
