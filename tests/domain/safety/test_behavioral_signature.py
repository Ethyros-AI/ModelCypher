"""Tests for behavioral signature dataclasses and their computed properties."""

import math

import pytest

from modelcypher.core.domain.safety.behavioral_signature import (
    BehavioralSignature,
    CapabilityPreservationResult,
    EntropyAnalysisResult,
    EntropyTrajectoryResult,
    PersonaStabilityResult,
    RefusalBoundaryResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_nan_signature() -> BehavioralSignature:
    """Signature with all NaN defaults -- no signal data available."""
    return BehavioralSignature(
        refusal_geodesic_distance=float("nan"),
        refusal_trajectory_slope=float("nan"),
        factual_sensitivity=float("nan"),
        persona_cka_to_baseline=float("nan"),
        identity_layer_consistency=float("nan"),
        entropy_z_score=float("nan"),
        probe_count=0,
        layer_indices_analyzed=(),
    )


def _all_real_signature() -> BehavioralSignature:
    """Signature with valid data in every signal category."""
    return BehavioralSignature(
        refusal_geodesic_distance=0.5,
        refusal_trajectory_slope=-0.1,
        factual_sensitivity=0.94,
        persona_cka_to_baseline=0.98,
        identity_layer_consistency=0.95,
        entropy_z_score=1.2,
        probe_count=10,
        layer_indices_analyzed=(0, 1, 2, 3),
        mean_entropy=3.5,
        vocab_size=32000,
        trajectory_path_ratio=1.1,
        trajectory_mean_curvature=0.05,
        trajectory_return_cka=0.8,
        trajectory_effective_rank=4.2,
        entropy_trajectory_slope=0.02,
        entropy_peak_layer_fraction=0.6,
        entropy_monotonicity=0.9,
        entropy_early_late_ratio=1.3,
    )


# ---------------------------------------------------------------------------
# BehavioralSignature -- has_* properties
# ---------------------------------------------------------------------------

class TestBehavioralSignatureAvailability:
    """has_* flags and signal_availability fraction."""

    def test_all_nan_has_no_data(self):
        sig = _all_nan_signature()
        assert sig.has_refusal_data is False
        assert sig.has_capability_data is False
        assert sig.has_persona_data is False
        assert sig.has_entropy_data is False
        assert sig.has_trajectory_data is False
        assert sig.has_entropy_trajectory_data is False

    def test_all_nan_signal_availability_zero(self):
        sig = _all_nan_signature()
        assert sig.signal_availability == 0.0

    def test_all_real_has_all_data(self):
        sig = _all_real_signature()
        assert sig.has_refusal_data is True
        assert sig.has_capability_data is True
        assert sig.has_persona_data is True
        assert sig.has_entropy_data is True
        assert sig.has_trajectory_data is True
        assert sig.has_entropy_trajectory_data is True

    def test_all_real_signal_availability_one(self):
        sig = _all_real_signature()
        assert sig.signal_availability == 1.0

    def test_partial_two_of_four(self):
        """Refusal + entropy present, capability + persona missing."""
        sig = BehavioralSignature(
            refusal_geodesic_distance=0.3,
            refusal_trajectory_slope=-0.05,
            factual_sensitivity=float("nan"),
            persona_cka_to_baseline=float("nan"),
            identity_layer_consistency=float("nan"),
            entropy_z_score=2.0,
            probe_count=5,
            layer_indices_analyzed=(0, 1),
        )
        assert sig.has_refusal_data is True
        assert sig.has_capability_data is False
        assert sig.has_persona_data is False
        assert sig.has_entropy_data is True
        assert sig.signal_availability == pytest.approx(0.5)

    def test_partial_three_of_four(self):
        """All except entropy."""
        sig = BehavioralSignature(
            refusal_geodesic_distance=0.3,
            refusal_trajectory_slope=-0.05,
            factual_sensitivity=0.9,
            persona_cka_to_baseline=0.85,
            identity_layer_consistency=0.9,
            entropy_z_score=float("nan"),
            probe_count=8,
            layer_indices_analyzed=(0, 1, 2),
        )
        assert sig.signal_availability == pytest.approx(0.75)

    def test_has_trajectory_data_requires_path_ratio(self):
        sig = _all_nan_signature()
        assert sig.has_trajectory_data is False
        sig2 = BehavioralSignature(
            refusal_geodesic_distance=float("nan"),
            refusal_trajectory_slope=float("nan"),
            factual_sensitivity=float("nan"),
            persona_cka_to_baseline=float("nan"),
            identity_layer_consistency=float("nan"),
            entropy_z_score=float("nan"),
            probe_count=0,
            layer_indices_analyzed=(),
            trajectory_path_ratio=1.5,
        )
        assert sig2.has_trajectory_data is True

    def test_has_entropy_trajectory_data_requires_slope(self):
        sig = _all_nan_signature()
        assert sig.has_entropy_trajectory_data is False
        sig2 = BehavioralSignature(
            refusal_geodesic_distance=float("nan"),
            refusal_trajectory_slope=float("nan"),
            factual_sensitivity=float("nan"),
            persona_cka_to_baseline=float("nan"),
            identity_layer_consistency=float("nan"),
            entropy_z_score=float("nan"),
            probe_count=0,
            layer_indices_analyzed=(),
            entropy_trajectory_slope=0.01,
        )
        assert sig2.has_entropy_trajectory_data is True


# ---------------------------------------------------------------------------
# BehavioralSignature -- as_dict
# ---------------------------------------------------------------------------

class TestBehavioralSignatureAsDict:
    """Serialization preserves values including NaN."""

    def test_all_nan_preserves_nan(self):
        d = _all_nan_signature().as_dict()
        assert math.isnan(d["refusal_geodesic_distance"])
        assert math.isnan(d["factual_sensitivity"])
        assert math.isnan(d["persona_cka_to_baseline"])
        assert math.isnan(d["entropy_z_score"])
        assert math.isnan(d["trajectory_path_ratio"])
        assert math.isnan(d["entropy_trajectory_slope"])

    def test_all_real_preserves_values(self):
        sig = _all_real_signature()
        d = sig.as_dict()
        assert d["refusal_geodesic_distance"] == 0.5
        assert d["factual_sensitivity"] == 0.94
        assert d["persona_cka_to_baseline"] == 0.98
        assert d["entropy_z_score"] == 1.2
        assert d["probe_count"] == 10
        assert d["layer_indices_analyzed"] == [0, 1, 2, 3]

    def test_as_dict_includes_all_expected_keys(self):
        d = _all_nan_signature().as_dict()
        expected_keys = {
            "refusal_geodesic_distance",
            "refusal_trajectory_slope",
            "factual_sensitivity",
            "persona_cka_to_baseline",
            "identity_layer_consistency",
            "entropy_z_score",
            "mean_entropy",
            "vocab_size",
            "probe_count",
            "layer_indices_analyzed",
            "trajectory_path_ratio",
            "trajectory_mean_curvature",
            "trajectory_return_cka",
            "trajectory_effective_rank",
            "entropy_trajectory_slope",
            "entropy_peak_layer_fraction",
            "entropy_monotonicity",
            "entropy_early_late_ratio",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# EntropyTrajectoryResult -- normalized_trajectory
# ---------------------------------------------------------------------------

class TestEntropyTrajectoryResultNormalized:
    """normalized_trajectory divides by max_possible_entropy."""

    def test_valid_normalization(self):
        result = EntropyTrajectoryResult(
            layer_entropies=(2.0, 4.0, 6.0),
            layer_indices=(0, 1, 2),
            slope=2.0,
            peak_layer_fraction=1.0,
            monotonicity=1.0,
            early_late_ratio=0.5,
            vocab_size=100,
            max_possible_entropy=10.0,
            probe_count=5,
        )
        normed = result.normalized_trajectory
        assert normed == pytest.approx((0.2, 0.4, 0.6))

    def test_max_possible_entropy_zero_returns_zeros(self):
        result = EntropyTrajectoryResult(
            layer_entropies=(2.0, 4.0, 6.0),
            layer_indices=(0, 1, 2),
            slope=2.0,
            peak_layer_fraction=1.0,
            monotonicity=1.0,
            early_late_ratio=0.5,
            vocab_size=1,
            max_possible_entropy=0.0,
            probe_count=5,
        )
        normed = result.normalized_trajectory
        assert normed == (0.0, 0.0, 0.0)

    def test_max_possible_entropy_negative_returns_zeros(self):
        result = EntropyTrajectoryResult(
            layer_entropies=(1.0, 2.0),
            layer_indices=(0, 1),
            slope=1.0,
            peak_layer_fraction=0.5,
            monotonicity=1.0,
            early_late_ratio=0.5,
            vocab_size=0,
            max_possible_entropy=-1.0,
            probe_count=3,
        )
        normed = result.normalized_trajectory
        assert normed == (0.0, 0.0)

    def test_empty_layer_entropies(self):
        result = EntropyTrajectoryResult(
            layer_entropies=(),
            layer_indices=(),
            slope=0.0,
            peak_layer_fraction=0.0,
            monotonicity=0.0,
            early_late_ratio=0.0,
            vocab_size=100,
            max_possible_entropy=10.0,
            probe_count=0,
        )
        assert result.normalized_trajectory == ()


class TestEntropyTrajectoryResultAsDict:
    """as_dict serialization."""

    def test_as_dict_keys(self):
        result = EntropyTrajectoryResult(
            layer_entropies=(1.0, 2.0),
            layer_indices=(0, 1),
            slope=0.5,
            peak_layer_fraction=0.5,
            monotonicity=0.8,
            early_late_ratio=1.2,
            vocab_size=32000,
            max_possible_entropy=10.37,
            probe_count=4,
        )
        d = result.as_dict()
        assert d["layer_entropies"] == [1.0, 2.0]
        assert d["layer_indices"] == [0, 1]
        assert d["slope"] == 0.5
        assert d["vocab_size"] == 32000
        assert d["probe_count"] == 4

    def test_as_dict_all_keys_present(self):
        result = EntropyTrajectoryResult(
            layer_entropies=(1.0,),
            layer_indices=(0,),
            slope=0.0,
            peak_layer_fraction=0.0,
            monotonicity=0.0,
            early_late_ratio=0.0,
            vocab_size=100,
            max_possible_entropy=4.6,
            probe_count=1,
        )
        expected_keys = {
            "layer_entropies",
            "layer_indices",
            "slope",
            "peak_layer_fraction",
            "monotonicity",
            "early_late_ratio",
            "vocab_size",
            "max_possible_entropy",
            "probe_count",
        }
        assert set(result.as_dict().keys()) == expected_keys


# ---------------------------------------------------------------------------
# Result dataclasses -- instantiation
# ---------------------------------------------------------------------------

class TestResultDataclasses:
    """All result dataclasses instantiate and hold values correctly."""

    def test_refusal_boundary_result(self):
        r = RefusalBoundaryResult(
            min_distance=0.1,
            mean_distance=0.5,
            distances=(0.1, 0.5, 0.9),
            anchor_count=3,
        )
        assert r.min_distance == 0.1
        assert r.mean_distance == 0.5
        assert r.distances == (0.1, 0.5, 0.9)
        assert r.anchor_count == 3

    def test_capability_preservation_result(self):
        r = CapabilityPreservationResult(
            mean_sensitivity=0.94,
            sensitivities=(0.9, 0.94, 0.98),
            pair_count=3,
        )
        assert r.mean_sensitivity == 0.94
        assert r.pair_count == 3

    def test_persona_stability_result(self):
        r = PersonaStabilityResult(
            cka_to_baseline=0.99,
            layer_consistency=0.97,
            layer_cka_values=(0.95, 0.97, 0.99),
            layers_analyzed=(0, 6, 12),
        )
        assert r.cka_to_baseline == 0.99
        assert r.layers_analyzed == (0, 6, 12)

    def test_entropy_analysis_result(self):
        r = EntropyAnalysisResult(
            mean_entropy=3.5,
            z_score=1.2,
            entropies=(3.0, 3.5, 4.0),
            probe_count=3,
            vocab_size=32000,
        )
        assert r.mean_entropy == 3.5
        assert r.vocab_size == 32000

    def test_all_result_dataclasses_are_frozen(self):
        r = RefusalBoundaryResult(
            min_distance=0.1, mean_distance=0.5, distances=(), anchor_count=0
        )
        with pytest.raises(AttributeError):
            r.min_distance = 0.2  # type: ignore[misc]

    def test_behavioral_signature_is_frozen(self):
        sig = _all_nan_signature()
        with pytest.raises(AttributeError):
            sig.probe_count = 99  # type: ignore[misc]
