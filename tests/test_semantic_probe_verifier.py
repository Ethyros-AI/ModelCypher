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

"""Tests for semantic_probe_verifier.py.

These tests validate the semantic probe verification system for LoRA transfer.
We test:
1. SemanticProbe data structure validation
2. SemanticProbeResult and SemanticDriftResult properties
3. KL divergence computation via LogitDivergenceCalculator
4. Threshold-based pass/fail logic
5. Probe loading from JSON
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.logit_divergence_calculator import (
    LogitDivergenceCalculator,
)
from modelcypher.core.domain.geometry.semantic_probe_verifier import (
    KL_DIVERGENCE_THRESHOLD,
    SemanticDriftResult,
    SemanticProbe,
    SemanticProbeResult,
    SemanticProbeVerifier,
    get_default_probes,
    load_semantic_probes,
)


class TestSemanticProbe:
    """Tests for SemanticProbe data structure."""

    def test_valid_probe_creation(self):
        """Valid probe should be created successfully."""
        probe = SemanticProbe(
            id="test-probe",
            context="The capital of France is",
            candidates=("Paris", "London", "Berlin"),
            correct_index=0,
            domain="geography",
        )

        assert probe.id == "test-probe"
        assert probe.context == "The capital of France is"
        assert probe.candidates == ("Paris", "London", "Berlin")
        assert probe.correct_index == 0
        assert probe.domain == "geography"

    def test_probe_defaults(self):
        """Probe should have sensible defaults."""
        probe = SemanticProbe(
            id="minimal",
            context="Test",
            candidates=("A", "B"),
        )

        assert probe.correct_index == 0
        assert probe.domain == "general"

    def test_probe_empty_candidates_fails(self):
        """Probe with no candidates should raise."""
        with pytest.raises(ValueError, match="at least one candidate"):
            SemanticProbe(
                id="empty",
                context="Test",
                candidates=(),
            )

    def test_probe_invalid_correct_index_fails(self):
        """Probe with out-of-range correct_index should raise."""
        with pytest.raises(ValueError, match="out of range"):
            SemanticProbe(
                id="bad-index",
                context="Test",
                candidates=("A", "B"),
                correct_index=5,  # Only 0, 1 valid
            )

    def test_probe_negative_correct_index_fails(self):
        """Probe with negative correct_index should raise."""
        with pytest.raises(ValueError, match="out of range"):
            SemanticProbe(
                id="negative",
                context="Test",
                candidates=("A", "B"),
                correct_index=-1,
            )


class TestSemanticProbeResult:
    """Tests for SemanticProbeResult data structure."""

    def test_top_prediction_preserved_same(self):
        """Same top predictions should be preserved."""
        result = SemanticProbeResult(
            probe_id="test",
            kl_divergence=0.1,
            source_top_idx=0,
            target_top_idx=0,
            source_correct_rank=1,
            target_correct_rank=1,
            rank_preserved=True,
            passed=True,
        )

        assert result.top_prediction_preserved is True

    def test_top_prediction_preserved_different(self):
        """Different top predictions should not be preserved."""
        result = SemanticProbeResult(
            probe_id="test",
            kl_divergence=0.5,
            source_top_idx=0,
            target_top_idx=1,
            source_correct_rank=1,
            target_correct_rank=2,
            rank_preserved=False,
            passed=True,
        )

        assert result.top_prediction_preserved is False


class TestSemanticDriftResult:
    """Tests for SemanticDriftResult aggregate metrics."""

    def test_passed_below_threshold(self):
        """Result should pass when mean KL below threshold."""
        result = SemanticDriftResult(
            mean_kl_divergence=0.1,  # Well below ln(2) ≈ 0.693
            max_kl_divergence=0.2,
            probes_passed=10,
            probes_total=10,
            rank_preservation_rate=1.0,
            top_prediction_rate=1.0,
            probe_results=[],
        )

        assert result.passed is True

    def test_passed_above_threshold(self):
        """Result should fail when mean KL above threshold."""
        result = SemanticDriftResult(
            mean_kl_divergence=1.0,  # Above ln(2) ≈ 0.693
            max_kl_divergence=1.5,
            probes_passed=5,
            probes_total=10,
            rank_preservation_rate=0.5,
            top_prediction_rate=0.5,
            probe_results=[],
        )

        assert result.passed is False

    def test_passed_at_threshold(self):
        """Result at exactly threshold should fail (strict inequality)."""
        result = SemanticDriftResult(
            mean_kl_divergence=KL_DIVERGENCE_THRESHOLD,
            max_kl_divergence=KL_DIVERGENCE_THRESHOLD,
            probes_passed=5,
            probes_total=10,
            rank_preservation_rate=0.5,
            top_prediction_rate=0.5,
            probe_results=[],
        )

        assert result.passed is False

    def test_pass_rate_calculation(self):
        """Pass rate should be correctly computed."""
        result = SemanticDriftResult(
            mean_kl_divergence=0.5,
            max_kl_divergence=0.8,
            probes_passed=7,
            probes_total=10,
            rank_preservation_rate=0.7,
            top_prediction_rate=0.7,
            probe_results=[],
        )

        assert result.pass_rate == 0.7

    def test_pass_rate_zero_probes(self):
        """Pass rate with zero probes should be 0."""
        result = SemanticDriftResult(
            mean_kl_divergence=0.0,
            max_kl_divergence=0.0,
            probes_passed=0,
            probes_total=0,
            rank_preservation_rate=0.0,
            top_prediction_rate=0.0,
            probe_results=[],
        )

        assert result.pass_rate == 0.0

    def test_to_dict_contains_key_fields(self):
        """to_dict should contain all important fields."""
        result = SemanticDriftResult(
            mean_kl_divergence=0.3,
            max_kl_divergence=0.5,
            probes_passed=8,
            probes_total=10,
            rank_preservation_rate=0.8,
            top_prediction_rate=0.9,
            probe_results=[],
        )

        d = result.to_dict()

        assert "mean_kl_divergence" in d
        assert "max_kl_divergence" in d
        assert "probes_passed" in d
        assert "probes_total" in d
        assert "pass_rate" in d
        assert "passed" in d
        assert "threshold" in d
        assert d["threshold"] == KL_DIVERGENCE_THRESHOLD


class TestKLThreshold:
    """Tests for the KL divergence threshold constant."""

    def test_threshold_is_ln2(self):
        """Threshold should be ln(2)."""
        assert abs(KL_DIVERGENCE_THRESHOLD - math.log(2)) < 1e-10

    def test_threshold_approximately_0693(self):
        """Threshold should be approximately 0.693."""
        assert abs(KL_DIVERGENCE_THRESHOLD - 0.693) < 0.001


class TestKLDivergenceCalculation:
    """Tests for KL divergence computation via LogitDivergenceCalculator."""

    def setup_method(self):
        self.backend = get_default_backend()
        self.calc = LogitDivergenceCalculator(self.backend)

    def test_identical_distributions_zero_kl(self):
        """Identical distributions should have KL = 0."""
        b = self.backend
        logits = b.array([1.0, 2.0, 3.0, 4.0])
        b.eval(logits)

        kl = self.calc.kl_divergence(logits, logits)

        assert abs(kl) < 1e-6

    def test_different_distributions_positive_kl(self):
        """Different distributions should have positive KL."""
        b = self.backend
        logits1 = b.array([1.0, 2.0, 3.0, 4.0])
        logits2 = b.array([4.0, 3.0, 2.0, 1.0])  # Reversed
        b.eval(logits1, logits2)

        kl = self.calc.kl_divergence(logits1, logits2)

        assert kl > 0

    def test_stable_softmax_sums_to_one(self):
        """Softmax should produce valid probability distribution."""
        b = self.backend
        logits = b.array([1.0, 2.0, 3.0, 4.0])
        b.eval(logits)

        probs = self.calc.stable_softmax(logits)
        b.eval(probs)

        total = float(b.to_scalar(b.sum(probs)))
        assert abs(total - 1.0) < 1e-6


class TestProbeLoading:
    """Tests for loading probes from JSON."""

    def test_load_semantic_probes_from_file(self):
        """Should load probes from valid JSON file."""
        probe_data = {
            "probes": [
                {
                    "id": "test-1",
                    "context": "Context 1",
                    "candidates": ["A", "B", "C"],
                    "correct_index": 0,
                    "domain": "test",
                },
                {
                    "id": "test-2",
                    "context": "Context 2",
                    "candidates": ["X", "Y"],
                    "correct_index": 1,
                    "domain": "test",
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(probe_data, f)
            f.flush()

            probes = load_semantic_probes(Path(f.name))

        assert len(probes) == 2
        assert probes[0].id == "test-1"
        assert probes[0].candidates == ("A", "B", "C")
        assert probes[1].id == "test-2"
        assert probes[1].correct_index == 1

    def test_get_default_probes(self):
        """Default probes should be available."""
        probes = get_default_probes()

        assert len(probes) > 0
        assert all(isinstance(p, SemanticProbe) for p in probes)

        # Check for expected domains
        domains = {p.domain for p in probes}
        assert "geography" in domains
        assert "common-sense" in domains
        assert "arithmetic" in domains


class TestSemanticProbeVerifier:
    """Tests for the SemanticProbeVerifier class."""

    def setup_method(self):
        self.backend = get_default_backend()

    def test_verifier_initialization(self):
        """Verifier should initialize without tokenizer."""
        verifier = SemanticProbeVerifier(backend=self.backend, tokenizer=None)

        assert verifier is not None

    def test_verify_empty_probes_returns_empty_result(self):
        """Verification with no probes should return empty result."""
        verifier = SemanticProbeVerifier(backend=self.backend, tokenizer=None)

        result = verifier.verify_transfer(
            source_model=None,
            source_adapter=None,
            target_model=None,
            target_adapter=None,
            probes=[],
        )

        assert result.probes_total == 0
        assert result.probes_passed == 0
        assert result.mean_kl_divergence == 0.0

    def test_compute_rank_correct(self):
        """_compute_rank should correctly find rank of target index."""
        verifier = SemanticProbeVerifier(backend=self.backend, tokenizer=None)
        b = self.backend

        # Probabilities: index 2 is highest, then 0, then 1
        probs = b.array([0.3, 0.1, 0.6])
        b.eval(probs)

        # Index 2 should be rank 1 (highest)
        rank2 = verifier._compute_rank(probs, 2)
        assert rank2 == 1

        # Index 0 should be rank 2
        rank0 = verifier._compute_rank(probs, 0)
        assert rank0 == 2

        # Index 1 should be rank 3 (lowest)
        rank1 = verifier._compute_rank(probs, 1)
        assert rank1 == 3


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_with_mock_logits(self):
        """Test full pipeline with mock logit data."""
        backend = get_default_backend()
        calc = LogitDivergenceCalculator(backend)

        # Simulate source producing strong preference for index 0
        source_logits = backend.array([10.0, 1.0, 1.0, 1.0])
        # Simulate target with similar but slightly shifted preference
        target_logits = backend.array([9.5, 1.2, 1.1, 1.0])
        backend.eval(source_logits, target_logits)

        # Compute KL
        kl = calc.kl_divergence(source_logits, target_logits)

        # Should be low (similar distributions)
        assert kl < 0.5

        # Simulate a bad transfer (completely different preference)
        bad_target_logits = backend.array([1.0, 1.0, 10.0, 1.0])
        backend.eval(bad_target_logits)

        bad_kl = calc.kl_divergence(source_logits, bad_target_logits)

        # Should be high (very different distributions)
        assert bad_kl > 1.0

    def test_probe_result_aggregation(self):
        """Test that SemanticDriftResult correctly aggregates probe results."""
        # Create some probe results
        results = [
            SemanticProbeResult(
                probe_id="p1",
                kl_divergence=0.1,
                source_top_idx=0,
                target_top_idx=0,
                source_correct_rank=1,
                target_correct_rank=1,
                rank_preserved=True,
                passed=True,
            ),
            SemanticProbeResult(
                probe_id="p2",
                kl_divergence=0.5,
                source_top_idx=0,
                target_top_idx=1,
                source_correct_rank=1,
                target_correct_rank=2,
                rank_preserved=False,
                passed=True,
            ),
            SemanticProbeResult(
                probe_id="p3",
                kl_divergence=1.0,
                source_top_idx=0,
                target_top_idx=2,
                source_correct_rank=1,
                target_correct_rank=3,
                rank_preserved=False,
                passed=False,
            ),
        ]

        # Manually compute expected values
        kl_values = [0.1, 0.5, 1.0]
        mean_kl = sum(kl_values) / len(kl_values)
        max_kl = max(kl_values)

        drift_result = SemanticDriftResult(
            mean_kl_divergence=mean_kl,
            max_kl_divergence=max_kl,
            probes_passed=2,  # Only p1 and p2 passed
            probes_total=3,
            rank_preservation_rate=1 / 3,  # Only p1 preserved rank
            top_prediction_rate=1 / 3,  # Only p1 preserved top prediction
            probe_results=results,
        )

        assert abs(drift_result.mean_kl_divergence - mean_kl) < 1e-6
        assert drift_result.max_kl_divergence == 1.0
        assert drift_result.pass_rate == 2 / 3
        # Mean KL ≈ 0.533 < 0.693, so overall should pass
        assert drift_result.passed is True
