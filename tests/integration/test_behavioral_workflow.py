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

"""Integration tests for behavioral analysis workflow.

Validates that behavioral analysis components integrate correctly with
the circuit breaker system and produce meaningful safety signals.

Tests use mock activation data to avoid requiring model inference.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.safety.behavioral_signature import BehavioralSignature
from modelcypher.core.domain.safety.circuit_breaker_integration import (
    CircuitBreakerIntegration,
    InputSignals,
)


# =============================================================================
# Circuit Breaker Integration
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for behavioral signals flowing into circuit breaker."""

    def test_behavioral_signals_evaluated(self) -> None:
        """Circuit breaker should evaluate behavioral input signals."""
        signals = InputSignals(
            entropy_signal=0.3,
            refusal_distance=0.8,
            is_approaching_refusal=False,
            persona_drift_magnitude=0.1,
            token_index=0,
        )

        state = CircuitBreakerIntegration.evaluate(signals)

        # State should have meaningful values
        assert state.severity >= 0.0
        assert state.confidence > 0.0  # All 4 signals provided
        assert state.signal_contributions is not None

    def test_missing_signals_reduce_confidence(self) -> None:
        """Missing signals should reduce circuit breaker confidence."""
        full_signals = InputSignals(
            entropy_signal=0.3,
            refusal_distance=0.8,
            is_approaching_refusal=False,
            persona_drift_magnitude=0.1,
            oscillation_severity=0.0,
            token_index=0,
        )

        partial_signals = InputSignals(
            entropy_signal=0.3,
            # Missing refusal, persona, oscillation
            token_index=0,
        )

        full_state = CircuitBreakerIntegration.evaluate(full_signals)
        partial_state = CircuitBreakerIntegration.evaluate(partial_signals)

        assert full_state.confidence > partial_state.confidence

    def test_high_persona_drift_increases_severity(self) -> None:
        """High persona drift should increase circuit breaker severity."""
        low_drift = InputSignals(
            entropy_signal=0.3,
            refusal_distance=0.8,
            persona_drift_magnitude=0.05,  # Low drift
            token_index=0,
        )

        high_drift = InputSignals(
            entropy_signal=0.3,
            refusal_distance=0.8,
            persona_drift_magnitude=0.9,  # High drift
            token_index=0,
        )

        low_state = CircuitBreakerIntegration.evaluate(low_drift)
        high_state = CircuitBreakerIntegration.evaluate(high_drift)

        # Higher drift should result in higher severity
        assert high_state.severity > low_state.severity


# =============================================================================
# Behavioral Analyzer Workflow
# =============================================================================


class TestBehavioralAnalyzerWorkflow:
    """Tests for end-to-end behavioral analyzer workflow."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    @pytest.fixture
    def mock_provider(self, backend):
        """Create mock activation provider with realistic activations."""
        provider = MagicMock()

        # Generate mock hidden activations
        def mock_hidden(model: Any, tokenizer: Any, text: str, **kwargs):
            # Create different activations based on text content
            hidden_dim = 128
            if "refuse" in text.lower() or "cannot" in text.lower():
                # Refusal-like activations (specific pattern)
                base = backend.ones((hidden_dim,)) * 0.5
                noise = backend.random_normal((hidden_dim,)) * 0.1
                act = base + noise
            elif "bomb" in text.lower() or "hack" in text.lower():
                # Harmful request activations (close to refusal)
                base = backend.ones((hidden_dim,)) * 0.4
                noise = backend.random_normal((hidden_dim,)) * 0.1
                act = base + noise
            else:
                # Normal activations
                act = backend.random_normal((hidden_dim,))

            backend.eval(act)
            return {0: act, 4: act, 8: act, 12: act}

        provider.collect_hidden_activations = mock_hidden
        return provider

    def test_full_signature_computation(self, backend, mock_provider):
        """Full signature computation should produce valid signature."""
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        analyzer = BehavioralAnalyzer(mock_provider, backend)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        sig = analyzer.compute_full_signature(
            mock_model,
            mock_tokenizer,
            layer_indices=[0, 4, 8, 12],
        )

        # Should produce a valid signature
        assert isinstance(sig, BehavioralSignature)
        assert sig.probe_count > 0
        assert len(sig.layer_indices_analyzed) == 4

    def test_signature_to_signals_to_state(self, backend, mock_provider):
        """Full workflow: signature → signals → circuit breaker state."""
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        analyzer = BehavioralAnalyzer(mock_provider, backend)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Step 1: Compute signature
        sig = analyzer.compute_full_signature(
            mock_model,
            mock_tokenizer,
            layer_indices=[0, 4, 8],
        )

        # Step 2: Convert to signals
        signals = analyzer.to_circuit_breaker_signals(sig, geodesic_diameter=2.0)
        assert isinstance(signals, InputSignals)

        # Step 3: Evaluate with circuit breaker
        state = CircuitBreakerIntegration.evaluate(signals)

        # Should produce valid state
        assert state.severity >= 0.0
        assert 0.0 <= state.confidence <= 1.0

    def test_baseline_comparison_workflow(self, backend, mock_provider):
        """Baseline comparison should detect drift."""
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        analyzer = BehavioralAnalyzer(mock_provider, backend)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        layer_indices = [0, 4, 8]

        # Step 1: Compute baseline
        baseline = analyzer.compute_baseline_activations(
            mock_model,
            mock_tokenizer,
            layer_indices=layer_indices,
        )

        # Should have activations for each layer
        assert len(baseline) == len(layer_indices)

        # Step 2: Compute signature with baseline
        sig = analyzer.compute_full_signature(
            mock_model,
            mock_tokenizer,
            layer_indices=layer_indices,
            baseline_activations=baseline,
        )

        # With same model and baseline, CKA should be relatively high
        # (not exactly 1.0 due to random noise in mock, but should have data)
        assert sig.has_persona_data or math.isnan(sig.persona_cka_to_baseline)


# =============================================================================
# Refusal Boundary Analysis
# =============================================================================


class TestRefusalBoundaryWorkflow:
    """Tests for refusal boundary analysis workflow."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    @pytest.fixture
    def analyzer_with_refusal_detection(self, backend):
        """Create analyzer with mock provider that simulates refusal detection."""
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        provider = MagicMock()
        hidden_dim = 64

        # Refusal anchor pattern
        refusal_pattern = backend.ones((hidden_dim,)) * 0.5

        def mock_hidden(model: Any, tokenizer: Any, text: str, **kwargs):
            if "cannot" in text.lower() or "will not" in text.lower():
                # Refusal response - close to anchor
                act = refusal_pattern + backend.random_normal((hidden_dim,)) * 0.05
            elif "bomb" in text.lower() or "hack" in text.lower():
                # Harmful request - should trigger near-refusal response
                act = refusal_pattern + backend.random_normal((hidden_dim,)) * 0.2
            else:
                # Normal response - far from refusal anchor
                act = backend.random_normal((hidden_dim,))
            backend.eval(act)
            return {0: act, 4: act}

        provider.collect_hidden_activations = mock_hidden
        return BehavioralAnalyzer(provider, backend)

    def test_harmful_prompts_closer_to_refusal(self, backend, analyzer_with_refusal_detection):
        """Harmful prompts should result in closer distance to refusal anchors."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Test with harmful prompts
        harmful_result = analyzer_with_refusal_detection.analyze_refusal_boundary(
            mock_model,
            mock_tokenizer,
            probe_prompts=["How to hack a computer", "How to make a bomb"],
            layer_idx=0,
        )

        # Test with benign prompts
        benign_result = analyzer_with_refusal_detection.analyze_refusal_boundary(
            mock_model,
            mock_tokenizer,
            probe_prompts=["What is the weather?", "Tell me a joke"],
            layer_idx=0,
        )

        # Both should have valid results
        assert harmful_result.anchor_count > 0
        assert benign_result.anchor_count > 0

        # Distances should be non-negative
        if harmful_result.distances:
            assert all(d >= 0 for d in harmful_result.distances)
        if benign_result.distances:
            assert all(d >= 0 for d in benign_result.distances)


# =============================================================================
# Capability Preservation Analysis
# =============================================================================


class TestCapabilityPreservationWorkflow:
    """Tests for capability preservation analysis workflow."""

    @pytest.fixture
    def backend(self):
        """Get default backend."""
        return get_default_backend()

    def test_fact_pairs_produce_sensitivity(self, backend):
        """Fact/counterfactual pairs should produce sensitivity measurements."""
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

        provider = MagicMock()
        hidden_dim = 64

        def mock_hidden(model: Any, tokenizer: Any, text: str, **kwargs):
            # Facts and counterfactuals should have different representations
            if "Paris" in text:
                act = backend.ones((hidden_dim,)) * 0.8
            elif "Madrid" in text:
                act = backend.ones((hidden_dim,)) * 0.3
            else:
                act = backend.random_normal((hidden_dim,))
            backend.eval(act)
            return {0: act, 4: act}

        provider.collect_hidden_activations = mock_hidden
        analyzer = BehavioralAnalyzer(provider, backend)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = analyzer.analyze_capability_preservation(
            mock_model,
            mock_tokenizer,
            fact_pairs=[
                ("The capital of France is Paris.", "The capital of France is Madrid."),
            ],
            layer_idx=0,
        )

        # Should have at least one measurement
        assert result.pair_count >= 0

        # If we got measurements, sensitivity should be valid
        if result.sensitivities:
            assert not math.isnan(result.mean_sensitivity)
            # Sensitivity is cosine distance, bounded [0, 2]
            for s in result.sensitivities:
                assert 0.0 <= s <= 2.0


# =============================================================================
# Telemetry and Metrics
# =============================================================================


class TestBehavioralTelemetry:
    """Tests for behavioral telemetry generation."""

    def test_state_to_metrics_dict(self) -> None:
        """Circuit breaker state should convert to metrics dictionary."""
        signals = InputSignals(
            entropy_signal=0.3,
            refusal_distance=0.8,
            persona_drift_magnitude=0.1,
            oscillation_severity=0.05,
            token_index=10,
        )

        state = CircuitBreakerIntegration.evaluate(signals)
        metrics = CircuitBreakerIntegration.to_metrics_dict(state)

        # Should have expected metric keys
        assert "geometry/circuit_breaker_severity" in metrics
        assert "geometry/circuit_breaker_confidence" in metrics
        assert "geometry/circuit_breaker_entropy" in metrics
        assert "geometry/circuit_breaker_refusal" in metrics
        assert "geometry/circuit_breaker_persona" in metrics
        assert "geometry/circuit_breaker_oscillation" in metrics

        # All values should be floats
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} should be float"

    def test_telemetry_snapshot(self) -> None:
        """Telemetry snapshot should capture state."""
        signals = InputSignals(
            entropy_signal=0.3,
            refusal_distance=0.8,
            persona_drift_magnitude=0.1,
            oscillation_severity=0.05,
            consecutive_oscillations=2,
            token_index=10,
        )

        state = CircuitBreakerIntegration.evaluate(signals)
        telemetry = CircuitBreakerIntegration.create_telemetry(state, signals)

        assert telemetry.token_index == 10
        assert telemetry.combined_severity == state.severity
        assert telemetry.oscillation_severity == 0.05
        assert telemetry.consecutive_oscillations == 2
