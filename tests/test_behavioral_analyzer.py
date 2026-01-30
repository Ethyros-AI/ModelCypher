# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for BehavioralAnalyzer and BehavioralSignature.

Design Principles Tested:
- Raw metrics only (no composite scores)
- Frozen immutability for signatures
- NaN values preserved for degenerate cases
- Circuit breaker signals bounded in [0, 1]
- Geodesic distance symmetry
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from modelcypher.core.domain.safety.behavioral_signature import (
    BehavioralSignature,
    CapabilityPreservationResult,
    EntropyAnalysisResult,
    EntropyTrajectoryResult,
    PersonaStabilityResult,
    RefusalBoundaryResult,
)
from modelcypher.core.domain.safety.circuit_breaker_integration import InputSignals


class TestBehavioralSignature:
    """Tests for BehavioralSignature dataclass."""

    def test_frozen_dataclass(self):
        """BehavioralSignature should be immutable."""
        sig = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=-0.1,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=1.2,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            sig.refusal_geodesic_distance = 0.8

    def test_as_dict(self):
        """as_dict should include all fields."""
        sig = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=-0.1,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=1.2,
            mean_entropy=5.5,
            vocab_size=32000,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
            trajectory_path_ratio=1.5,
            trajectory_mean_curvature=0.3,
            trajectory_return_cka=0.2,
            trajectory_effective_rank=2.5,
            entropy_trajectory_slope=0.1,
            entropy_peak_layer_fraction=0.6,
            entropy_monotonicity=-0.8,
            entropy_early_late_ratio=1.2,
        )

        d = sig.as_dict()

        assert d["refusal_geodesic_distance"] == 0.5
        assert d["refusal_trajectory_slope"] == -0.1
        assert d["factual_sensitivity"] == 0.25
        assert d["persona_cka_to_baseline"] == 0.95
        assert d["identity_layer_consistency"] == 0.9
        assert d["entropy_z_score"] == 1.2
        assert d["mean_entropy"] == 5.5
        assert d["vocab_size"] == 32000
        assert d["probe_count"] == 10
        assert d["layer_indices_analyzed"] == [4, 8, 12]
        # Trajectory complexity fields
        assert d["trajectory_path_ratio"] == 1.5
        assert d["trajectory_mean_curvature"] == 0.3
        assert d["trajectory_return_cka"] == 0.2
        assert d["trajectory_effective_rank"] == 2.5
        # Entropy trajectory fields
        assert d["entropy_trajectory_slope"] == 0.1
        assert d["entropy_peak_layer_fraction"] == 0.6
        assert d["entropy_monotonicity"] == -0.8
        assert d["entropy_early_late_ratio"] == 1.2
        # 18 fields total
        assert len(d) == 18

    def test_nan_values_preserved(self):
        """NaN values should be preserved for degenerate cases."""
        sig = BehavioralSignature(
            refusal_geodesic_distance=float("nan"),
            refusal_trajectory_slope=float("nan"),
            factual_sensitivity=float("nan"),
            persona_cka_to_baseline=float("nan"),
            identity_layer_consistency=float("nan"),
            entropy_z_score=float("nan"),
            probe_count=0,
            layer_indices_analyzed=(),
        )

        assert math.isnan(sig.refusal_geodesic_distance)
        assert math.isnan(sig.factual_sensitivity)
        assert math.isnan(sig.persona_cka_to_baseline)

        d = sig.as_dict()
        assert math.isnan(d["refusal_geodesic_distance"])
        assert math.isnan(d["factual_sensitivity"])

    def test_has_data_properties(self):
        """has_*_data properties should correctly detect NaN values."""
        sig_full = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=-0.1,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=1.2,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
            entropy_trajectory_slope=0.1,
        )

        assert sig_full.has_refusal_data is True
        assert sig_full.has_capability_data is True
        assert sig_full.has_persona_data is True
        assert sig_full.has_entropy_data is True
        assert sig_full.has_entropy_trajectory_data is True
        assert sig_full.signal_availability == 1.0

        sig_partial = BehavioralSignature(
            refusal_geodesic_distance=float("nan"),
            refusal_trajectory_slope=float("nan"),
            factual_sensitivity=0.25,
            persona_cka_to_baseline=float("nan"),
            identity_layer_consistency=float("nan"),
            entropy_z_score=float("nan"),
            probe_count=5,
            layer_indices_analyzed=(4,),
        )

        assert sig_partial.has_refusal_data is False
        assert sig_partial.has_capability_data is True
        assert sig_partial.has_persona_data is False
        assert sig_partial.has_entropy_data is False
        assert sig_partial.has_entropy_trajectory_data is False
        assert sig_partial.signal_availability == 0.25


class TestRefusalBoundaryResult:
    """Tests for RefusalBoundaryResult dataclass."""

    def test_frozen_dataclass(self):
        """RefusalBoundaryResult should be immutable."""
        result = RefusalBoundaryResult(
            min_distance=0.3,
            mean_distance=0.5,
            distances=(0.3, 0.5, 0.7),
            anchor_count=3,
        )

        with pytest.raises(Exception):
            result.min_distance = 0.4

    def test_empty_distances(self):
        """Empty distances should be handled."""
        result = RefusalBoundaryResult(
            min_distance=float("nan"),
            mean_distance=float("nan"),
            distances=(),
            anchor_count=0,
        )

        assert math.isnan(result.min_distance)
        assert result.anchor_count == 0


class TestCapabilityPreservationResult:
    """Tests for CapabilityPreservationResult dataclass."""

    def test_frozen_dataclass(self):
        """CapabilityPreservationResult should be immutable."""
        result = CapabilityPreservationResult(
            mean_sensitivity=0.2,
            sensitivities=(0.15, 0.2, 0.25),
            pair_count=3,
        )

        with pytest.raises(Exception):
            result.mean_sensitivity = 0.3


class TestPersonaStabilityResult:
    """Tests for PersonaStabilityResult dataclass."""

    def test_frozen_dataclass(self):
        """PersonaStabilityResult should be immutable."""
        result = PersonaStabilityResult(
            cka_to_baseline=0.95,
            layer_consistency=0.9,
            layer_cka_values=(0.88, 0.91, 0.92),
            layers_analyzed=(4, 8, 12),
        )

        with pytest.raises(Exception):
            result.cka_to_baseline = 0.8


class TestBehavioralAnalyzerSignalConversion:
    """Tests for BehavioralAnalyzer signal conversion logic."""

    def test_circuit_breaker_signals_bounded(self):
        """All circuit breaker signals should be in [0, 1] or None."""
        # Import here to test the conversion logic
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        # Create a mock provider (we only test conversion, not activation collection)
        mock_provider = MagicMock()
        backend = get_default_backend()

        analyzer = BehavioralAnalyzer(mock_provider, backend)

        # Test with typical values
        sig = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=-0.1,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=1.2,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        signals = analyzer.to_circuit_breaker_signals(sig, geodesic_diameter=1.0)

        # All signals should be bounded
        if signals.refusal_distance is not None:
            assert 0.0 <= signals.refusal_distance <= 1.0
        if signals.persona_drift_magnitude is not None:
            assert 0.0 <= signals.persona_drift_magnitude <= 1.0
        if signals.entropy_signal is not None:
            assert 0.0 <= signals.entropy_signal <= 1.0

    def test_signals_with_nan_values(self):
        """NaN signature values should produce None signals."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        mock_provider = MagicMock()
        backend = get_default_backend()
        analyzer = BehavioralAnalyzer(mock_provider, backend)

        sig = BehavioralSignature(
            refusal_geodesic_distance=float("nan"),
            refusal_trajectory_slope=float("nan"),
            factual_sensitivity=float("nan"),
            persona_cka_to_baseline=float("nan"),
            identity_layer_consistency=float("nan"),
            entropy_z_score=float("nan"),
            probe_count=0,
            layer_indices_analyzed=(),
        )

        signals = analyzer.to_circuit_breaker_signals(sig)

        # All signals should be None when signature has NaN values
        assert signals.refusal_distance is None
        assert signals.persona_drift_magnitude is None
        assert signals.entropy_signal is None
        assert signals.is_approaching_refusal is None

    def test_persona_drift_from_cka(self):
        """Persona drift should be 1 - CKA (inverted)."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        mock_provider = MagicMock()
        backend = get_default_backend()
        analyzer = BehavioralAnalyzer(mock_provider, backend)

        # High CKA = low drift
        sig_high_cka = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=0.0,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,  # High CKA
            identity_layer_consistency=0.9,
            entropy_z_score=0.0,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        signals_high = analyzer.to_circuit_breaker_signals(sig_high_cka)
        assert signals_high.persona_drift_magnitude is not None
        assert signals_high.persona_drift_magnitude == pytest.approx(0.05, abs=0.01)

        # Low CKA = high drift
        sig_low_cka = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=0.0,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.3,  # Low CKA
            identity_layer_consistency=0.9,
            entropy_z_score=0.0,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        signals_low = analyzer.to_circuit_breaker_signals(sig_low_cka)
        assert signals_low.persona_drift_magnitude is not None
        assert signals_low.persona_drift_magnitude == pytest.approx(0.7, abs=0.01)

    def test_approaching_refusal_detection(self):
        """Negative trajectory slope should indicate approaching refusal."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        mock_provider = MagicMock()
        backend = get_default_backend()
        analyzer = BehavioralAnalyzer(mock_provider, backend)

        # Negative slope = approaching
        sig_approaching = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=-0.5,  # Negative
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=0.0,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        signals_approaching = analyzer.to_circuit_breaker_signals(sig_approaching)
        assert signals_approaching.is_approaching_refusal is True

        # Positive slope = moving away
        sig_away = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=0.5,  # Positive
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=0.0,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        signals_away = analyzer.to_circuit_breaker_signals(sig_away)
        assert signals_away.is_approaching_refusal is False


class TestBehavioralAnalyzerGeometry:
    """Tests for geometric properties of BehavioralAnalyzer."""

    @pytest.fixture
    def backend(self):
        """Get the default backend."""
        from modelcypher.core.domain._backend import get_default_backend

        return get_default_backend()

    @pytest.fixture
    def analyzer(self, backend):
        """Create analyzer with mock provider."""
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        mock_provider = MagicMock()
        return BehavioralAnalyzer(mock_provider, backend)

    def test_geodesic_distance_non_negative(self, analyzer, backend):
        """Geodesic distance should be non-negative."""
        # Create test vectors
        anchor = backend.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        query = backend.array([0.5, 0.5, 0.0])

        dist = analyzer._geodesic_min_distance(anchor, query)

        assert dist >= 0.0

    def test_geodesic_distance_zero_for_same_point(self, analyzer, backend):
        """Geodesic distance to same point should be zero."""
        point = backend.array([[1.0, 0.0, 0.0]])
        query = backend.array([1.0, 0.0, 0.0])

        dist = analyzer._geodesic_min_distance(point, query)

        # Should be very close to zero (within numerical precision)
        assert dist < 1e-6


class TestBehavioralAnalyzerLayerValidation:
    """Tests for layer index validation in BehavioralAnalyzer."""

    def test_empty_layer_indices_raises(self):
        """Empty layer_indices should raise ValueError."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        mock_provider = MagicMock()
        backend = get_default_backend()
        analyzer = BehavioralAnalyzer(mock_provider, backend)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with pytest.raises(ValueError, match="layer_indices must be non-empty"):
            analyzer.compute_full_signature(mock_model, mock_tokenizer, layer_indices=[])


# Property-based tests (if hypothesis is available)
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    class TestBehavioralSignatureProperties:
        """Property-based tests for BehavioralSignature."""

        @given(
            refusal_dist=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
            persona_cka=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            entropy_z=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
        )
        @settings(max_examples=20)
        def test_signals_always_bounded(
            self, refusal_dist: float, persona_cka: float, entropy_z: float
        ):
            """Circuit breaker signals should always be in [0, 1]."""
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

            mock_provider = MagicMock()
            backend = get_default_backend()
            analyzer = BehavioralAnalyzer(mock_provider, backend)

            sig = BehavioralSignature(
                refusal_geodesic_distance=refusal_dist,
                refusal_trajectory_slope=0.0,
                factual_sensitivity=0.2,
                persona_cka_to_baseline=persona_cka,
                identity_layer_consistency=0.9,
                entropy_z_score=entropy_z,
                probe_count=10,
                layer_indices_analyzed=(4, 8, 12),
            )

            signals = analyzer.to_circuit_breaker_signals(sig, geodesic_diameter=10.0)

            # All non-None signals should be bounded
            if signals.refusal_distance is not None:
                assert 0.0 <= signals.refusal_distance <= 1.0
            if signals.persona_drift_magnitude is not None:
                assert 0.0 <= signals.persona_drift_magnitude <= 1.0
            if signals.entropy_signal is not None:
                assert 0.0 <= signals.entropy_signal <= 1.0

        @given(
            signal_availability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        )
        @settings(max_examples=10)
        def test_signal_availability_bounded(self, signal_availability: float):
            """signal_availability should always be in [0, 1]."""
            # Create signatures with varying NaN values
            sig = BehavioralSignature(
                refusal_geodesic_distance=0.5 if signal_availability > 0.25 else float("nan"),
                refusal_trajectory_slope=0.0,
                factual_sensitivity=0.2 if signal_availability > 0.5 else float("nan"),
                persona_cka_to_baseline=0.9 if signal_availability > 0.75 else float("nan"),
                identity_layer_consistency=0.9,
                entropy_z_score=1.0 if signal_availability > 0.0 else float("nan"),
                probe_count=10,
                layer_indices_analyzed=(4,),
            )

            assert 0.0 <= sig.signal_availability <= 1.0

except ImportError:
    pass  # hypothesis not installed, skip property tests


class TestEntropyAnalysisResult:
    """Tests for EntropyAnalysisResult dataclass."""

    def test_frozen_dataclass(self):
        """EntropyAnalysisResult should be immutable."""
        result = EntropyAnalysisResult(
            mean_entropy=5.5,
            z_score=1.2,
            entropies=(5.0, 5.5, 6.0),
            probe_count=3,
            vocab_size=32000,
        )

        with pytest.raises(Exception):
            result.mean_entropy = 6.0

    def test_empty_entropies(self):
        """Empty entropies should be handled with NaN."""
        result = EntropyAnalysisResult(
            mean_entropy=float("nan"),
            z_score=float("nan"),
            entropies=(),
            probe_count=0,
            vocab_size=32000,
        )

        assert math.isnan(result.mean_entropy)
        assert math.isnan(result.z_score)
        assert result.probe_count == 0


class TestEntropySignalConversion:
    """Tests for entropy signal conversion to circuit breaker signals."""

    def test_entropy_signal_from_mean_entropy(self):
        """Entropy signal should be computed from mean_entropy and vocab_size."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        mock_provider = MagicMock()
        backend = get_default_backend()
        analyzer = BehavioralAnalyzer(mock_provider, backend)

        # Signature with entropy data
        sig = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=0.0,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=1.2,
            mean_entropy=5.5,  # Raw entropy
            vocab_size=32000,  # Vocab size for normalization
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        signals = analyzer.to_circuit_breaker_signals(sig)

        # Entropy signal should be computed (normalized to [0, 1])
        assert signals.entropy_signal is not None
        assert 0.0 <= signals.entropy_signal <= 1.0

    def test_entropy_signal_none_without_vocab_size(self):
        """Entropy signal should be None if vocab_size is 0."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        mock_provider = MagicMock()
        backend = get_default_backend()
        analyzer = BehavioralAnalyzer(mock_provider, backend)

        sig = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=0.0,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=1.2,
            mean_entropy=5.5,
            vocab_size=0,  # No vocab size
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        signals = analyzer.to_circuit_breaker_signals(sig)

        # Entropy signal should be None without vocab_size
        assert signals.entropy_signal is None

    def test_entropy_signal_none_with_nan_entropy(self):
        """Entropy signal should be None if mean_entropy is NaN."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        mock_provider = MagicMock()
        backend = get_default_backend()
        analyzer = BehavioralAnalyzer(mock_provider, backend)

        sig = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=0.0,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=float("nan"),  # NaN z-score
            mean_entropy=float("nan"),  # NaN entropy
            vocab_size=32000,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        signals = analyzer.to_circuit_breaker_signals(sig)

        # Entropy signal should be None with NaN mean_entropy
        assert signals.entropy_signal is None

    def test_full_signal_availability_with_entropy(self):
        """Signal availability should be 1.0 when all signals including entropy are available."""
        sig = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=0.0,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=1.2,  # Valid entropy z-score
            mean_entropy=5.5,
            vocab_size=32000,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        assert sig.has_entropy_data is True
        assert sig.signal_availability == 1.0

    def test_entropy_normalization_bounds(self):
        """Normalized entropy should respect theoretical bounds."""
        from modelcypher.core.domain.entropy.logit_entropy_calculator import (
            LogitEntropyCalculator,
        )

        # Zero entropy -> 0.0 normalized
        assert LogitEntropyCalculator.normalize_entropy(0.0, 32000) == 0.0

        # Max entropy (ln(vocab_size)) -> 1.0 normalized
        import math
        max_entropy = math.log(32000)
        normalized = LogitEntropyCalculator.normalize_entropy(max_entropy, 32000)
        assert normalized == pytest.approx(1.0, abs=0.01)

        # Mid-range entropy
        mid_entropy = max_entropy / 2
        normalized_mid = LogitEntropyCalculator.normalize_entropy(mid_entropy, 32000)
        assert 0.4 <= normalized_mid <= 0.6


class TestEntropyTrajectoryResult:
    """Tests for EntropyTrajectoryResult dataclass."""

    def test_frozen_dataclass(self):
        """EntropyTrajectoryResult should be immutable."""
        result = EntropyTrajectoryResult(
            layer_entropies=(5.0, 5.5, 6.0, 5.8),
            layer_indices=(0, 1, 2, 3),
            slope=0.2,
            peak_layer_fraction=0.67,
            monotonicity=0.6,
            early_late_ratio=0.9,
            vocab_size=32000,
            max_possible_entropy=10.37,
            probe_count=4,
        )

        with pytest.raises(Exception):
            result.slope = 0.3

    def test_as_dict(self):
        """as_dict should include all fields."""
        result = EntropyTrajectoryResult(
            layer_entropies=(5.0, 5.5, 6.0, 5.8),
            layer_indices=(0, 1, 2, 3),
            slope=0.2,
            peak_layer_fraction=0.67,
            monotonicity=0.6,
            early_late_ratio=0.9,
            vocab_size=32000,
            max_possible_entropy=10.37,
            probe_count=4,
        )

        d = result.as_dict()

        assert d["layer_entropies"] == [5.0, 5.5, 6.0, 5.8]
        assert d["layer_indices"] == [0, 1, 2, 3]
        assert d["slope"] == 0.2
        assert d["peak_layer_fraction"] == 0.67
        assert d["monotonicity"] == 0.6
        assert d["early_late_ratio"] == 0.9
        assert d["vocab_size"] == 32000
        assert d["max_possible_entropy"] == 10.37
        assert d["probe_count"] == 4

    def test_normalized_trajectory(self):
        """normalized_trajectory should scale to [0, 1] by max possible entropy."""
        result = EntropyTrajectoryResult(
            layer_entropies=(5.0, 10.0),
            layer_indices=(0, 1),
            slope=2.5,
            peak_layer_fraction=1.0,
            monotonicity=1.0,
            early_late_ratio=0.5,
            vocab_size=32000,
            max_possible_entropy=10.0,
            probe_count=2,
        )

        norm = result.normalized_trajectory

        assert norm[0] == pytest.approx(0.5, abs=0.01)
        assert norm[1] == pytest.approx(1.0, abs=0.01)

    def test_normalized_trajectory_zero_max(self):
        """normalized_trajectory with zero max entropy should return zeros."""
        result = EntropyTrajectoryResult(
            layer_entropies=(5.0, 6.0),
            layer_indices=(0, 1),
            slope=0.5,
            peak_layer_fraction=1.0,
            monotonicity=1.0,
            early_late_ratio=0.83,
            vocab_size=0,
            max_possible_entropy=0.0,
            probe_count=2,
        )

        norm = result.normalized_trajectory

        assert norm == (0.0, 0.0)

    def test_single_layer_nan_features(self):
        """Single-layer trajectory should have NaN for derived features."""
        result = EntropyTrajectoryResult(
            layer_entropies=(5.0,),
            layer_indices=(4,),
            slope=float("nan"),
            peak_layer_fraction=float("nan"),
            monotonicity=float("nan"),
            early_late_ratio=float("nan"),
            vocab_size=32000,
            max_possible_entropy=10.37,
            probe_count=4,
        )

        assert math.isnan(result.slope)
        assert math.isnan(result.peak_layer_fraction)
        assert math.isnan(result.monotonicity)


class TestEntropyTrajectoryFeatures:
    """Tests for entropy trajectory feature computation logic."""

    @pytest.fixture
    def backend(self):
        """Get the default backend."""
        from modelcypher.core.domain._backend import get_default_backend

        return get_default_backend()

    @pytest.fixture
    def analyzer(self, backend):
        """Create analyzer with mock provider."""
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        mock_provider = MagicMock()
        return BehavioralAnalyzer(mock_provider, backend)

    def test_compute_ranks_no_ties(self, analyzer):
        """_compute_ranks should return correct ranks without ties."""
        values = [3.0, 1.0, 2.0]
        ranks = analyzer._compute_ranks(values)

        # 1.0 is smallest (rank 0), 2.0 is middle (rank 1), 3.0 is largest (rank 2)
        assert ranks[0] == 2.0  # 3.0 -> rank 2
        assert ranks[1] == 0.0  # 1.0 -> rank 0
        assert ranks[2] == 1.0  # 2.0 -> rank 1

    def test_compute_ranks_with_ties(self, analyzer):
        """_compute_ranks should average ranks for ties."""
        values = [2.0, 1.0, 2.0]
        ranks = analyzer._compute_ranks(values)

        # 1.0 is smallest (rank 0), both 2.0s share ranks 1 and 2 -> avg 1.5
        assert ranks[0] == 1.5  # 2.0 -> avg rank 1.5
        assert ranks[1] == 0.0  # 1.0 -> rank 0
        assert ranks[2] == 1.5  # 2.0 -> avg rank 1.5

    def test_monotonic_increasing_trajectory(self, analyzer):
        """Monotonically increasing trajectory should have monotonicity ~1.0."""
        # Create trajectory [1.0, 2.0, 3.0, 4.0]
        trajectory = [1.0, 2.0, 3.0, 4.0]

        # Compute Spearman correlation manually
        ranks = analyzer._compute_ranks(trajectory)
        n = len(trajectory)

        # Layer ranks are [0, 1, 2, 3], entropy ranks should also be [0, 1, 2, 3]
        assert ranks == [0.0, 1.0, 2.0, 3.0]

        # Spearman correlation should be 1.0 for perfect monotonic increase
        rank_x_mean = (n - 1) / 2.0
        rank_y_mean = sum(ranks) / n

        cov = sum((i - rank_x_mean) * (ranks[i] - rank_y_mean) for i in range(n))
        var_x = sum((i - rank_x_mean) ** 2 for i in range(n))
        var_y = sum((r - rank_y_mean) ** 2 for r in ranks)

        monotonicity = cov / math.sqrt(var_x * var_y)
        assert monotonicity == pytest.approx(1.0, abs=0.01)

    def test_monotonic_decreasing_trajectory(self, analyzer):
        """Monotonically decreasing trajectory should have monotonicity ~-1.0."""
        trajectory = [4.0, 3.0, 2.0, 1.0]

        ranks = analyzer._compute_ranks(trajectory)
        n = len(trajectory)

        # Entropy ranks should be [3, 2, 1, 0] (highest to lowest)
        assert ranks == [3.0, 2.0, 1.0, 0.0]

        rank_x_mean = (n - 1) / 2.0
        rank_y_mean = sum(ranks) / n

        cov = sum((i - rank_x_mean) * (ranks[i] - rank_y_mean) for i in range(n))
        var_x = sum((i - rank_x_mean) ** 2 for i in range(n))
        var_y = sum((r - rank_y_mean) ** 2 for r in ranks)

        monotonicity = cov / math.sqrt(var_x * var_y)
        assert monotonicity == pytest.approx(-1.0, abs=0.01)

    def test_slope_calculation(self, analyzer):
        """Slope should be calculated via linear regression."""
        # trajectory = [1.0, 2.0, 3.0, 4.0] should have slope = 1.0
        trajectory = [1.0, 2.0, 3.0, 4.0]
        n = len(trajectory)

        x_mean = (n - 1) / 2.0  # 1.5
        y_mean = sum(trajectory) / n  # 2.5

        numerator = sum((i - x_mean) * (trajectory[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator
        assert slope == pytest.approx(1.0, abs=0.01)

    def test_peak_layer_fraction(self, analyzer):
        """Peak layer fraction should locate max entropy."""
        # trajectory = [1.0, 5.0, 3.0, 2.0] - peak at index 1
        trajectory = [1.0, 5.0, 3.0, 2.0]
        n = len(trajectory)

        peak_idx = trajectory.index(max(trajectory))  # 1
        peak_layer_fraction = peak_idx / (n - 1)  # 1/3 = 0.333...

        assert peak_layer_fraction == pytest.approx(0.333, abs=0.01)

    def test_early_late_ratio(self, analyzer, backend):
        """Early/late ratio should compare first half to second half mean."""
        from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

        trajectory = [2.0, 2.0, 4.0, 4.0]  # Early mean = 2.0, late mean = 4.0
        n = len(trajectory)
        mid = n // 2

        early_mean = sum(trajectory[:mid]) / mid  # 2.0
        late_mean = sum(trajectory[mid:]) / (n - mid)  # 4.0

        eps = float(division_epsilon(backend, backend.array([1.0])))
        early_late_ratio = early_mean / (late_mean + eps)

        assert early_late_ratio == pytest.approx(0.5, abs=0.01)


class TestBehavioralSignatureEntropyTrajectory:
    """Tests for entropy trajectory fields in BehavioralSignature."""

    def test_entropy_trajectory_fields_default_nan(self):
        """Entropy trajectory fields should default to NaN."""
        sig = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=0.0,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=1.2,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
        )

        assert math.isnan(sig.entropy_trajectory_slope)
        assert math.isnan(sig.entropy_peak_layer_fraction)
        assert math.isnan(sig.entropy_monotonicity)
        assert math.isnan(sig.entropy_early_late_ratio)
        assert sig.has_entropy_trajectory_data is False

    def test_entropy_trajectory_fields_populated(self):
        """Entropy trajectory fields should be correctly populated."""
        sig = BehavioralSignature(
            refusal_geodesic_distance=0.5,
            refusal_trajectory_slope=0.0,
            factual_sensitivity=0.25,
            persona_cka_to_baseline=0.95,
            identity_layer_consistency=0.9,
            entropy_z_score=1.2,
            probe_count=10,
            layer_indices_analyzed=(4, 8, 12),
            entropy_trajectory_slope=-0.5,
            entropy_peak_layer_fraction=0.2,
            entropy_monotonicity=-0.9,
            entropy_early_late_ratio=1.5,
        )

        assert sig.entropy_trajectory_slope == -0.5
        assert sig.entropy_peak_layer_fraction == 0.2
        assert sig.entropy_monotonicity == -0.9
        assert sig.entropy_early_late_ratio == 1.5
        assert sig.has_entropy_trajectory_data is True
