# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for stage_validate contracts.

Covers:
  V1 — ValidateResult always returned; all 7 metric keys always present.
  V2 — Model-dependent stages skip gracefully when prerequisites are absent.
  V3 — Numerical stability: no-data path and populated path with layer data.
  V4 — Spectral analysis: skipped on empty weights; populated with shared keys.
  V5 — Circuit breaker and behavioral probes are always populated.
"""

from __future__ import annotations

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.merge.stages.validate import (
    ValidateResult,
    stage_validate,
)

_SEVEN_METRIC_KEYS = {
    "numerical_stability",
    "content_safety",
    "behavioral_probes",
    "circuit_breaker",
    "ridge_resistance",
    "spectral_analysis",
    "tangent_alignment",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_call(b, **kwargs):
    """stage_validate with empty weight dicts and all optionals None."""
    defaults = dict(
        merged_weights={},
        source_weights={},
        target_weights={},
        layer_confidences={},
        layer_indices=[],
        hidden_dim=32,
        backend=b,
    )
    defaults.update(kwargs)
    return stage_validate(**defaults)


def _layer_weights(b):
    """Return weight dicts with a matching layers.0. key for numerical stability."""
    W = b.random_normal((8, 8))
    d = {"layers.0.weight": W}
    return d, d, d  # merged, source, target


# ---------------------------------------------------------------------------
# V1: ValidateResult always returned with all 7 metric keys
# ---------------------------------------------------------------------------

class TestAlwaysReturnsAllMetrics:
    def test_returns_validate_result(self):
        b = get_default_backend()
        result = _minimal_call(b)
        assert isinstance(result, ValidateResult)

    def test_all_seven_metric_keys_present(self):
        b = get_default_backend()
        result = _minimal_call(b)
        assert _SEVEN_METRIC_KEYS.issubset(result.metrics.keys())


# ---------------------------------------------------------------------------
# V2: Graceful skips when prerequisites absent
# ---------------------------------------------------------------------------

class TestGracefulSkips:
    def test_content_safety_skips_without_model(self):
        """target_model=None → content_safety skipped with reason no_model."""
        b = get_default_backend()
        result = _minimal_call(b)  # target_model defaults to None
        cs = result.metrics["content_safety"]
        assert cs.get("skipped") is True
        assert cs.get("reason") == "no_model"

    def test_ridge_resistance_skips_without_model_path(self):
        """merged_model_path=None → ridge_resistance skipped."""
        b = get_default_backend()
        result = _minimal_call(b)  # merged_model_path defaults to None
        rr = result.metrics["ridge_resistance"]
        assert rr.get("skipped") is True
        assert rr.get("reason") == "no_model_path"

    def test_tangent_alignment_skips_without_activation_fn(self):
        """collect_activations_fn=None → tangent_alignment skipped."""
        b = get_default_backend()
        result = _minimal_call(b)  # collect_activations_fn defaults to None
        ta = result.metrics["tangent_alignment"]
        assert ta.get("skipped") is True


# ---------------------------------------------------------------------------
# V3: Numerical stability paths
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_no_layer_diagnostics_when_no_matching_keys(self):
        """layer_indices=[0] but weight dicts empty → no_layer_diagnostics note."""
        b = get_default_backend()
        result = _minimal_call(b, layer_indices=[0], layer_confidences={0: 0.8})
        ns = result.metrics["numerical_stability"]
        # No weights keyed "layers.0.*" in empty dicts → no diagnostics collected
        assert ns.get("note") == "no_layer_diagnostics"

    def test_numerical_stability_populated_with_layer_data(self):
        """Weights keyed layers.0.weight + matching confidence → full profile populated."""
        b = get_default_backend()
        merged, source, target = _layer_weights(b)
        result = stage_validate(
            merged_weights=merged,
            source_weights=source,
            target_weights=target,
            layer_confidences={0: 0.8},
            layer_indices=[0],
            hidden_dim=8,
            backend=b,
        )
        ns = result.metrics["numerical_stability"]
        assert "mean_interference" in ns
        assert "mean_importance" in ns


# ---------------------------------------------------------------------------
# V4: Spectral analysis paths
# ---------------------------------------------------------------------------

class TestSpectralAnalysis:
    def test_spectral_skipped_on_empty_weights(self):
        """All weight dicts empty → spectral_analysis skipped."""
        b = get_default_backend()
        result = _minimal_call(b)
        sa = result.metrics["spectral_analysis"]
        assert sa == {"skipped": True, "reason": "no_weights_analyzed"}

    def test_spectral_populated_with_shared_key(self):
        """Same key in all three weight dicts → spectral analysis runs."""
        b = get_default_backend()
        W = b.random_normal((8, 8))
        weights = {"model.weight": W}
        result = stage_validate(
            merged_weights=weights,
            source_weights=weights,
            target_weights=weights,
            layer_confidences={},
            layer_indices=[],
            hidden_dim=32,
            backend=b,
        )
        sa = result.metrics["spectral_analysis"]
        assert "skipped" not in sa
        assert sa["total_weights"] >= 1


# ---------------------------------------------------------------------------
# V5: Circuit breaker and behavioral probes always present
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_circuit_breaker_always_populated(self):
        """circuit_breaker_result is not None; metrics has both signal keys."""
        b = get_default_backend()
        result = _minimal_call(b)
        assert result.circuit_breaker_result is not None
        cb = result.metrics["circuit_breaker"]
        assert "refusal_score" in cb
        assert "persona_drift_magnitude" in cb

    def test_behavioral_probes_always_has_probes_run(self):
        """metrics['behavioral_probes'] always has probes_run key."""
        b = get_default_backend()
        result = _minimal_call(b)
        assert "probes_run" in result.metrics["behavioral_probes"]
