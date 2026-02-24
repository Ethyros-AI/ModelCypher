# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for domain_signal_profile module.

Covers LayerSignal and DomainSignalProfile dataclasses including
the create() factory, to_dict()/from_dict() serialization, and round-trip.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from modelcypher.core.domain.geometry.domain_signal_profile import (
    DomainSignalProfile,
    LayerSignal,
)

# ---------------------------------------------------------------------------
# LayerSignal
# ---------------------------------------------------------------------------


class TestLayerSignal:
    """Tests for the LayerSignal frozen dataclass."""

    def test_default_all_none(self):
        sig = LayerSignal()
        assert sig.sparsity is None
        assert sig.gradient_variance is None
        assert sig.gradient_snr is None
        assert sig.mean_gradient_norm is None
        assert sig.gradient_sample_count is None
        assert sig.spatial_coherence is None
        assert sig.social_coherence is None
        assert sig.temporal_coherence is None
        assert sig.moral_coherence is None

    def test_partial_fields(self):
        sig = LayerSignal(sparsity=0.5, gradient_snr=2.0)
        assert sig.sparsity == 0.5
        assert sig.gradient_snr == 2.0
        assert sig.gradient_variance is None

    def test_all_fields_set(self):
        sig = LayerSignal(
            sparsity=0.1,
            gradient_variance=0.02,
            gradient_snr=5.0,
            mean_gradient_norm=0.3,
            gradient_sample_count=10,
            spatial_coherence=0.9,
            social_coherence=0.7,
            temporal_coherence=0.8,
            moral_coherence=0.6,
        )
        assert sig.sparsity == 0.1
        assert sig.gradient_variance == 0.02
        assert sig.gradient_snr == 5.0
        assert sig.mean_gradient_norm == 0.3
        assert sig.gradient_sample_count == 10
        assert sig.spatial_coherence == 0.9
        assert sig.social_coherence == 0.7
        assert sig.temporal_coherence == 0.8
        assert sig.moral_coherence == 0.6

    def test_frozen(self):
        sig = LayerSignal(sparsity=0.5)
        with pytest.raises(AttributeError):
            sig.sparsity = 0.9  # type: ignore[misc]

    def test_coherence_fields_only(self):
        sig = LayerSignal(
            spatial_coherence=0.8,
            social_coherence=0.6,
            temporal_coherence=0.7,
            moral_coherence=0.5,
        )
        assert sig.sparsity is None
        assert sig.spatial_coherence == 0.8


# ---------------------------------------------------------------------------
# DomainSignalProfile
# ---------------------------------------------------------------------------


def _make_profile(**overrides) -> DomainSignalProfile:
    """Helper to create a DomainSignalProfile with sane defaults."""
    defaults = dict(
        layer_signals={
            0: LayerSignal(sparsity=0.1, gradient_snr=5.0),
            1: LayerSignal(sparsity=0.3, gradient_variance=0.05),
        },
        model_id="test-model-123",
        domain="code",
        baseline_domain="baseline",
        total_layers=12,
        prompt_count=10,
        max_tokens_per_prompt=128,
        generated_at=datetime(2025, 6, 15, 12, 0, 0),
        notes="test profile",
    )
    defaults.update(overrides)
    return DomainSignalProfile(**defaults)


class TestDomainSignalProfile:
    """Tests for DomainSignalProfile dataclass and serialization."""

    def test_instantiation(self):
        p = _make_profile()
        assert p.model_id == "test-model-123"
        assert p.domain == "code"
        assert p.baseline_domain == "baseline"
        assert p.total_layers == 12
        assert p.prompt_count == 10
        assert p.max_tokens_per_prompt == 128
        assert p.notes == "test profile"
        assert len(p.layer_signals) == 2
        assert 0 in p.layer_signals
        assert 1 in p.layer_signals

    def test_frozen(self):
        p = _make_profile()
        with pytest.raises(AttributeError):
            p.model_id = "changed"  # type: ignore[misc]

    def test_create_factory(self):
        p = DomainSignalProfile.create(
            layer_signals={0: LayerSignal(sparsity=0.2)},
            model_id="factory-test",
            domain="creative",
            baseline_domain="general",
            total_layers=24,
            prompt_count=5,
            max_tokens_per_prompt=64,
            notes="from create",
        )
        assert p.model_id == "factory-test"
        assert p.domain == "creative"
        assert p.total_layers == 24
        assert p.notes == "from create"
        # generated_at should be set automatically
        assert isinstance(p.generated_at, datetime)

    def test_create_factory_no_notes(self):
        p = DomainSignalProfile.create(
            layer_signals={},
            model_id="m",
            domain="d",
            baseline_domain="b",
            total_layers=1,
            prompt_count=1,
            max_tokens_per_prompt=16,
        )
        assert p.notes is None

    def test_to_dict_keys(self):
        p = _make_profile()
        d = p.to_dict()
        expected_keys = {
            "layerSignals",
            "modelId",
            "domain",
            "baselineDomain",
            "totalLayers",
            "promptCount",
            "maxTokensPerPrompt",
            "generatedAt",
            "notes",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_layer_signal_keys(self):
        p = _make_profile()
        d = p.to_dict()
        ls = d["layerSignals"]
        # Keys are stringified layer indices
        assert "0" in ls
        assert "1" in ls
        # Each layer signal has camelCase keys
        layer0 = ls["0"]
        assert "sparsity" in layer0
        assert "gradientVariance" in layer0
        assert "gradientSNR" in layer0
        assert "meanGradientNorm" in layer0
        assert "gradientSampleCount" in layer0
        assert "spatialCoherence" in layer0
        assert "socialCoherence" in layer0
        assert "temporalCoherence" in layer0
        assert "moralCoherence" in layer0

    def test_to_dict_values(self):
        p = _make_profile()
        d = p.to_dict()
        assert d["modelId"] == "test-model-123"
        assert d["domain"] == "code"
        assert d["baselineDomain"] == "baseline"
        assert d["totalLayers"] == 12
        assert d["promptCount"] == 10
        assert d["maxTokensPerPrompt"] == 128
        assert d["notes"] == "test profile"
        # Layer 0 should have sparsity=0.1 and gradient_snr=5.0
        assert d["layerSignals"]["0"]["sparsity"] == 0.1
        assert d["layerSignals"]["0"]["gradientSNR"] == 5.0
        assert d["layerSignals"]["0"]["gradientVariance"] is None

    def test_from_dict_basic(self):
        d = {
            "layerSignals": {
                "0": {
                    "sparsity": 0.4,
                    "gradientVariance": None,
                    "gradientSNR": None,
                    "meanGradientNorm": None,
                    "gradientSampleCount": None,
                    "spatialCoherence": None,
                    "socialCoherence": None,
                    "temporalCoherence": None,
                    "moralCoherence": None,
                },
            },
            "modelId": "from-dict-model",
            "domain": "math",
            "baselineDomain": "general",
            "totalLayers": 6,
            "promptCount": 3,
            "maxTokensPerPrompt": 32,
            "generatedAt": "2025-06-15T12:00:00",
            "notes": None,
        }
        p = DomainSignalProfile.from_dict(d)
        assert p.model_id == "from-dict-model"
        assert p.domain == "math"
        assert p.total_layers == 6
        assert 0 in p.layer_signals
        assert p.layer_signals[0].sparsity == 0.4
        assert p.layer_signals[0].gradient_variance is None
        assert p.notes is None

    def test_round_trip(self):
        original = _make_profile()
        d = original.to_dict()
        restored = DomainSignalProfile.from_dict(d)

        assert restored.model_id == original.model_id
        assert restored.domain == original.domain
        assert restored.baseline_domain == original.baseline_domain
        assert restored.total_layers == original.total_layers
        assert restored.prompt_count == original.prompt_count
        assert restored.max_tokens_per_prompt == original.max_tokens_per_prompt
        assert restored.notes == original.notes
        assert set(restored.layer_signals.keys()) == set(original.layer_signals.keys())

        for layer_idx in original.layer_signals:
            orig_sig = original.layer_signals[layer_idx]
            rest_sig = restored.layer_signals[layer_idx]
            assert rest_sig.sparsity == orig_sig.sparsity
            assert rest_sig.gradient_variance == orig_sig.gradient_variance
            assert rest_sig.gradient_snr == orig_sig.gradient_snr
            assert rest_sig.mean_gradient_norm == orig_sig.mean_gradient_norm
            assert rest_sig.gradient_sample_count == orig_sig.gradient_sample_count
            assert rest_sig.spatial_coherence == orig_sig.spatial_coherence
            assert rest_sig.social_coherence == orig_sig.social_coherence
            assert rest_sig.temporal_coherence == orig_sig.temporal_coherence
            assert rest_sig.moral_coherence == orig_sig.moral_coherence

    def test_round_trip_with_all_coherence_fields(self):
        profile = DomainSignalProfile.create(
            layer_signals={
                0: LayerSignal(
                    sparsity=0.2,
                    gradient_variance=0.01,
                    gradient_snr=10.0,
                    mean_gradient_norm=0.5,
                    gradient_sample_count=20,
                    spatial_coherence=0.9,
                    social_coherence=0.7,
                    temporal_coherence=0.8,
                    moral_coherence=0.6,
                ),
            },
            model_id="coherence-test",
            domain="social",
            baseline_domain="baseline",
            total_layers=1,
            prompt_count=5,
            max_tokens_per_prompt=64,
        )
        d = profile.to_dict()
        restored = DomainSignalProfile.from_dict(d)
        sig = restored.layer_signals[0]
        assert sig.spatial_coherence == 0.9
        assert sig.social_coherence == 0.7
        assert sig.temporal_coherence == 0.8
        assert sig.moral_coherence == 0.6

    def test_round_trip_empty_signals(self):
        profile = _make_profile(layer_signals={})
        d = profile.to_dict()
        restored = DomainSignalProfile.from_dict(d)
        assert len(restored.layer_signals) == 0

    def test_from_dict_missing_optional_fields(self):
        """from_dict should handle missing optional keys gracefully."""
        d = {
            "layerSignals": {
                "0": {"sparsity": 0.5},
            },
            "modelId": "sparse-dict",
            "domain": "code",
            "baselineDomain": "base",
            "totalLayers": 2,
            "promptCount": 1,
            "maxTokensPerPrompt": 16,
            "generatedAt": "2025-01-01T00:00:00",
        }
        p = DomainSignalProfile.from_dict(d)
        assert p.layer_signals[0].sparsity == 0.5
        assert p.layer_signals[0].gradient_snr is None
        assert p.notes is None

    def test_generated_at_preserved_in_round_trip(self):
        ts = datetime(2025, 3, 14, 15, 9, 26)
        profile = _make_profile(generated_at=ts)
        d = profile.to_dict()
        restored = DomainSignalProfile.from_dict(d)
        assert restored.generated_at == ts
