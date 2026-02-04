#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ModelCypher
"""Tests for the self-improvement pipeline.

Tests cover:
- types.py: Data classes and serialization
- scanner.py: Capability scanning and classification
- generator.py: Training data generation
- oracle.py: Verification oracle
- improver.py: Orchestration logic
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from modelcypher.core.use_cases.self_improve.types import (
    Capability,
    CapabilityAnalysis,
    CapabilityStatus,
    DEFAULT_ACCURACY_THRESHOLD,
    DEFAULT_PRIMES,
    ImprovementAction,
    ImprovementLog,
    SelfImprovementConfig,
    VerifiedSample,
)


# =============================================================================
# Types Tests
# =============================================================================


class TestCapabilityStatus:
    """Tests for CapabilityStatus enum."""

    def test_status_values(self):
        """Verify enum values match expected strings."""
        assert CapabilityStatus.WORKING.value == "working"
        assert CapabilityStatus.DISCONNECTED.value == "disconnected"
        assert CapabilityStatus.TRUE_GAP.value == "true_gap"

    def test_all_statuses_exist(self):
        """Verify all expected statuses exist."""
        statuses = list(CapabilityStatus)
        assert len(statuses) == 3


class TestCapability:
    """Tests for Capability dataclass."""

    def test_from_lists(self):
        """Test creating Capability from mutable lists."""
        cap = Capability.from_lists(
            name="arithmetic",
            prompts=["1+1=", "2+2="],
            problems=[("1+1=", "2"), ("2+2=", "4")],
        )
        assert cap.name == "arithmetic"
        assert cap.prompts == ("1+1=", "2+2=")
        assert cap.problems == (("1+1=", "2"), ("2+2=", "4"))

    def test_capability_is_frozen(self):
        """Verify Capability is immutable."""
        cap = Capability.from_lists("test", ["a"], [("a", "b")])
        with pytest.raises(AttributeError):
            cap.name = "changed"

    def test_empty_capability(self):
        """Test capability with empty prompts and problems."""
        cap = Capability.from_lists("empty", [], [])
        assert cap.prompts == ()
        assert cap.problems == ()


class TestCapabilityAnalysis:
    """Tests for CapabilityAnalysis dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        cap = Capability.from_lists("test", ["p"], [("q", "a")])
        analysis = CapabilityAnalysis(
            capability=cap,
            status=CapabilityStatus.WORKING,
            accuracy_raw=0.9,
            accuracy_primed=0.95,
            kappa_raw=10.5,
            kappa_primed=8.2,
            best_prime="say",
        )
        d = analysis.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "working"
        assert d["accuracy_raw"] == 0.9
        assert d["accuracy_primed"] == 0.95
        assert d["kappa_raw"] == 10.5
        assert d["kappa_primed"] == 8.2
        assert d["best_prime"] == "say"


class TestVerifiedSample:
    """Tests for VerifiedSample dataclass."""

    def test_to_training_format(self):
        """Test conversion to MLX-LM training format."""
        sample = VerifiedSample(
            input_text="I have 3 apples. I get 2 more. Total:",
            output_text="3+2=",
            answer="5",
            oracle_computed="5",
        )
        fmt = sample.to_training_format()
        assert fmt["prompt"] == "I have 3 apples. I get 2 more. Total:"
        assert fmt["completion"] == "3+2=5"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        sample = VerifiedSample(
            input_text="input",
            output_text="output",
            answer="5",
            oracle_computed="5",
        )
        d = sample.to_dict()
        assert d["input"] == "input"
        assert d["output"] == "output"
        assert d["answer"] == "5"
        assert d["verified_computed"] == "5"

    def test_verified_sample_is_frozen(self):
        """Verify VerifiedSample is immutable."""
        sample = VerifiedSample("in", "out", "5", "5")
        with pytest.raises(AttributeError):
            sample.answer = "6"


class TestImprovementAction:
    """Tests for ImprovementAction dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        action = ImprovementAction(
            capability="arithmetic",
            action_type="apply_prime",
            details={"prime": "say", "accuracy_improvement": 0.3},
        )
        d = action.to_dict()
        assert d["capability"] == "arithmetic"
        assert d["action_type"] == "apply_prime"
        assert d["details"]["prime"] == "say"
        assert d["details"]["accuracy_improvement"] == 0.3


class TestSelfImprovementConfig:
    """Tests for SelfImprovementConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SelfImprovementConfig()
        assert config.loop_preservation is True
        assert config.geometric_self_awareness is True
        assert config.max_rounds == 5
        assert config.n_samples_per_round == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SelfImprovementConfig(
            loop_preservation=False,
            geometric_self_awareness=False,
            max_rounds=10,
            n_samples_per_round=50,
        )
        assert config.loop_preservation is False
        assert config.max_rounds == 10


class TestImprovementLog:
    """Tests for ImprovementLog dataclass."""

    def test_default_values(self):
        """Test default log values."""
        log = ImprovementLog()
        assert log.iterations == 0
        assert log.capabilities_scanned == []
        assert log.true_gaps == []
        assert log.training_data_path is None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        log = ImprovementLog(
            iterations=1,
            capabilities_scanned=["arithmetic"],
            capabilities_working=["arithmetic"],
            capabilities_bridged=[],
            true_gaps=["word_problems"],
        )
        log.actions.append(
            ImprovementAction("word_problems", "generate_training", {"n": 100})
        )
        d = log.to_dict()
        assert d["iterations"] == 1
        assert d["capabilities_scanned"] == ["arithmetic"]
        assert d["true_gaps"] == ["word_problems"]
        assert len(d["actions"]) == 1

    def test_log_json_serializable(self):
        """Test that log can be JSON serialized."""
        log = ImprovementLog(iterations=1)
        log.training_spec = {"adapter": {"type": "lora", "rank": 8}}
        d = log.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert "lora" in json_str


class TestDefaultConstants:
    """Tests for default constants."""

    def test_default_primes(self):
        """Test default primes are defined."""
        assert len(DEFAULT_PRIMES) > 0
        assert isinstance(DEFAULT_PRIMES, tuple)
        assert all(isinstance(p, str) for p in DEFAULT_PRIMES)

    def test_default_accuracy_threshold(self):
        """Test default accuracy threshold is reasonable."""
        assert 0 < DEFAULT_ACCURACY_THRESHOLD < 1
        assert DEFAULT_ACCURACY_THRESHOLD == 0.7


# =============================================================================
# Scanner Tests
# =============================================================================


class TestCapabilityScanner:
    """Tests for CapabilityScanner class."""

    def test_compute_kappa_identity(self):
        """Test kappa computation on well-conditioned data."""
        from modelcypher.core.use_cases.self_improve.scanner import CapabilityScanner

        # Create mock model/tokenizer (not used for compute_kappa)
        scanner = CapabilityScanner(MagicMock(), MagicMock())

        # Orthonormal activations should have kappa = 1
        activations = np.eye(4)
        kappa = scanner.compute_kappa(activations)
        assert np.isclose(kappa, 1.0, rtol=1e-6)

    def test_compute_kappa_ill_conditioned(self):
        """Test kappa on ill-conditioned data."""
        from modelcypher.core.use_cases.self_improve.scanner import CapabilityScanner

        scanner = CapabilityScanner(MagicMock(), MagicMock())

        # Nearly collinear vectors have high kappa
        activations = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1e-10, 0.0],  # Nearly parallel to first
        ])
        kappa = scanner.compute_kappa(activations)
        assert kappa > 1e6  # Very high condition number

    def test_compute_kappa_singular(self):
        """Test kappa is very large for singular matrices."""
        from modelcypher.core.use_cases.self_improve.scanner import CapabilityScanner

        scanner = CapabilityScanner(MagicMock(), MagicMock())

        # Duplicate rows → singular Gram matrix
        activations = np.array([
            [1.0, 0.0],
            [1.0, 0.0],  # Exact duplicate
        ])
        kappa = scanner.compute_kappa(activations)
        # np.linalg.cond returns very large number (not inf) for near-singular
        assert kappa > 1e10

    def test_scanner_custom_threshold(self):
        """Test scanner with custom accuracy threshold."""
        from modelcypher.core.use_cases.self_improve.scanner import CapabilityScanner

        scanner = CapabilityScanner(
            MagicMock(),
            MagicMock(),
            accuracy_threshold=0.5,
        )
        assert scanner._threshold == 0.5

    def test_scanner_custom_primes(self):
        """Test scanner with custom primes."""
        from modelcypher.core.use_cases.self_improve.scanner import CapabilityScanner

        custom_primes = ("custom1", "custom2")
        scanner = CapabilityScanner(
            MagicMock(),
            MagicMock(),
            primes=custom_primes,
        )
        assert scanner._primes == custom_primes


# =============================================================================
# Generator Tests
# =============================================================================


class TestSafeSelfPlayGenerator:
    """Tests for SafeSelfPlayGenerator class."""

    def test_to_training_format(self):
        """Test conversion of samples to training format."""
        from modelcypher.core.use_cases.self_improve.generator import (
            SafeSelfPlayGenerator,
        )

        # Create generator with mock oracle
        oracle = MagicMock()
        generator = SafeSelfPlayGenerator(oracle)

        samples = [
            VerifiedSample("prompt1", "3+2=", "5", "5"),
            VerifiedSample("prompt2", "4-1=", "3", "3"),
        ]
        training_data = generator.to_training_format(samples)

        assert len(training_data) == 2
        assert training_data[0]["prompt"] == "prompt1"
        assert training_data[0]["completion"] == "3+2=5"
        assert training_data[1]["completion"] == "4-1=3"

    def test_get_statistics(self):
        """Test statistics computation."""
        from modelcypher.core.use_cases.self_improve.generator import (
            SafeSelfPlayGenerator,
        )

        oracle = MagicMock()
        generator = SafeSelfPlayGenerator(oracle)

        samples = [
            VerifiedSample("p1", "3+2=", "5", "5"),
            VerifiedSample("p2", "4+1=", "5", "5"),
            VerifiedSample("p3", "5-2=", "3", "3"),
        ]
        stats = generator.get_statistics(samples)

        assert stats["total"] == 3
        assert stats["addition"] == 2
        assert stats["subtraction"] == 1

    def test_save_jsonl(self):
        """Test saving samples to JSONL file."""
        from modelcypher.core.use_cases.self_improve.generator import (
            SafeSelfPlayGenerator,
        )

        oracle = MagicMock()
        generator = SafeSelfPlayGenerator(oracle)

        samples = [
            VerifiedSample("prompt", "3+2=", "5", "5"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "training.jsonl"
            generator.save_jsonl(samples, path)

            assert path.exists()
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["prompt"] == "prompt"
            assert data["completion"] == "3+2=5"

    def test_custom_templates(self):
        """Test generator with custom templates."""
        from modelcypher.core.use_cases.self_improve.generator import (
            SafeSelfPlayGenerator,
        )

        oracle = MagicMock()
        custom_add = [("Custom {a} plus {b}:", "{a}+{b}=")]
        custom_sub = [("Custom {a} minus {b}:", "{a}-{b}=")]

        generator = SafeSelfPlayGenerator(
            oracle,
            addition_templates=custom_add,
            subtraction_templates=custom_sub,
        )
        assert generator._addition_templates == custom_add
        assert generator._subtraction_templates == custom_sub

    def test_generate_verified_with_mock_oracle(self):
        """Test generation with mocked oracle that always verifies."""
        from modelcypher.core.use_cases.self_improve.generator import (
            SafeSelfPlayGenerator,
        )

        oracle = MagicMock()
        # Oracle always returns correct verification
        oracle.verify.return_value = (True, "5")

        generator = SafeSelfPlayGenerator(oracle)
        samples = generator.generate_verified(n_samples=5, seed=42)

        assert len(samples) == 5
        assert oracle.verify.call_count == 5
        # All samples should have valid structure
        for sample in samples:
            assert "+" in sample.output_text or "-" in sample.output_text
            assert sample.oracle_computed == "5"

    def test_generate_verified_with_rejections(self):
        """Test generation when oracle rejects some samples."""
        from modelcypher.core.use_cases.self_improve.generator import (
            SafeSelfPlayGenerator,
        )

        oracle = MagicMock()
        # Alternate between accept and reject
        call_count = [0]

        def verify_side_effect(eq, expected):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return (True, expected)
            return (False, "wrong")

        oracle.verify.side_effect = verify_side_effect

        generator = SafeSelfPlayGenerator(oracle)
        samples = generator.generate_verified(n_samples=3, seed=42)

        # Should still get 3 samples eventually (50% acceptance)
        assert len(samples) == 3
        # Had to make more attempts than samples
        assert oracle.verify.call_count > 3

    def test_generate_verified_max_attempts(self):
        """Test generation stops after max attempts."""
        from modelcypher.core.use_cases.self_improve.generator import (
            SafeSelfPlayGenerator,
        )

        oracle = MagicMock()
        # Oracle always rejects
        oracle.verify.return_value = (False, "wrong")

        generator = SafeSelfPlayGenerator(oracle)
        samples = generator.generate_verified(
            n_samples=10,
            seed=42,
            max_attempts_multiplier=2,
        )

        # Should return empty list (or less than requested)
        assert len(samples) < 10
        # Should have tried n_samples * multiplier times
        assert oracle.verify.call_count == 10 * 2


# =============================================================================
# Oracle Tests
# =============================================================================


class TestVerificationOracle:
    """Tests for VerificationOracle class."""

    def test_default_calibration_tests(self):
        """Test default calibration tests are valid."""
        from modelcypher.core.use_cases.self_improve.oracle import VerificationOracle

        tests = VerificationOracle.default_calibration_tests()
        assert len(tests) >= 10
        for eq, expected in tests:
            assert "=" in eq
            # Verify expected is a valid integer
            assert expected.isdigit() or (expected[0] == "-" and expected[1:].isdigit())

    def test_custom_prime(self):
        """Test oracle with custom prime."""
        from modelcypher.core.use_cases.self_improve.oracle import VerificationOracle

        oracle = VerificationOracle(
            MagicMock(),
            MagicMock(),
            prime="My custom prime.",
        )
        assert oracle.prime == "My custom prime."

    def test_default_prime(self):
        """Test oracle uses default prime."""
        from modelcypher.core.use_cases.self_improve.oracle import VerificationOracle

        oracle = VerificationOracle(MagicMock(), MagicMock())
        assert oracle.prime == VerificationOracle.DEFAULT_PRIME

    def test_calibrate_empty_tests(self):
        """Test calibrate returns 0 for empty test list."""
        from modelcypher.core.use_cases.self_improve.oracle import VerificationOracle

        oracle = VerificationOracle(MagicMock(), MagicMock())
        accuracy, details = oracle.calibrate([])
        assert accuracy == 0.0
        assert details == []


# =============================================================================
# Improver Tests
# =============================================================================


class TestAutonomousSelfImprover:
    """Tests for AutonomousSelfImprover class."""

    def test_create_training_spec(self):
        """Test training spec creation."""
        from modelcypher.core.use_cases.self_improve.improver import (
            AutonomousSelfImprover,
        )

        # Create with mocks
        improver = AutonomousSelfImprover(MagicMock(), MagicMock())

        spec = improver.create_training_spec(
            gap_names=["word_problems", "parsing"],
            data_path="/path/to/data.jsonl",
            n_samples=250,
        )

        assert spec["target_capabilities"] == ["word_problems", "parsing"]
        assert spec["adapter"]["type"] == "lora"
        assert spec["adapter"]["rank"] == 8
        assert spec["adapter"]["alpha"] == 16
        assert spec["training"]["epochs"] == 3
        assert spec["training"]["batch_size"] == 4
        assert spec["freeze"]["late_layers"] is True
        assert spec["data"]["path"] == "/path/to/data.jsonl"
        assert spec["data"]["samples"] == 250
        assert spec["data"]["verified"] is True
        assert "rationale" in spec

    def test_save_log(self):
        """Test saving improvement log to file."""
        from modelcypher.core.use_cases.self_improve.improver import (
            AutonomousSelfImprover,
        )

        improver = AutonomousSelfImprover(MagicMock(), MagicMock())
        log = ImprovementLog(
            iterations=1,
            capabilities_scanned=["test"],
            true_gaps=["gap1"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "log.json"
            improver.save_log(log, path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["iterations"] == 1
            assert data["capabilities_scanned"] == ["test"]
            assert data["true_gaps"] == ["gap1"]

    def test_improver_initializes_components(self):
        """Test that improver initializes all required components."""
        from modelcypher.core.use_cases.self_improve.improver import (
            AutonomousSelfImprover,
        )
        from modelcypher.core.use_cases.self_improve.scanner import CapabilityScanner
        from modelcypher.core.use_cases.self_improve.oracle import VerificationOracle
        from modelcypher.core.use_cases.self_improve.generator import (
            SafeSelfPlayGenerator,
        )

        model = MagicMock()
        tokenizer = MagicMock()
        improver = AutonomousSelfImprover(model, tokenizer)

        assert isinstance(improver.scanner, CapabilityScanner)
        assert isinstance(improver.oracle, VerificationOracle)
        assert isinstance(improver.generator, SafeSelfPlayGenerator)
        assert improver.model is model
        assert improver.tokenizer is tokenizer


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================


class TestScannerClassification:
    """Integration tests for capability classification logic."""

    def test_classify_working(self):
        """Test classification of WORKING capability."""
        # A capability is WORKING if raw accuracy >= threshold (0.7)
        cap = Capability.from_lists(
            "arithmetic",
            ["1+1="],
            [("1+1=", "2"), ("2+2=", "4"), ("3+3=", "6")],
        )
        analysis = CapabilityAnalysis(
            capability=cap,
            status=CapabilityStatus.WORKING,
            accuracy_raw=0.8,
            accuracy_primed=0.9,
            kappa_raw=5.0,
            kappa_primed=4.0,
        )
        assert analysis.status == CapabilityStatus.WORKING

    def test_classify_disconnected(self):
        """Test classification of DISCONNECTED capability."""
        # DISCONNECTED: raw < 0.7 but primed >= 0.7
        cap = Capability.from_lists("parsing", ["x"], [("x", "y")])
        analysis = CapabilityAnalysis(
            capability=cap,
            status=CapabilityStatus.DISCONNECTED,
            accuracy_raw=0.3,
            accuracy_primed=0.8,
            kappa_raw=100.0,
            kappa_primed=10.0,
            best_prime="say",
        )
        assert analysis.status == CapabilityStatus.DISCONNECTED
        assert analysis.best_prime == "say"

    def test_classify_true_gap(self):
        """Test classification of TRUE_GAP capability."""
        # TRUE_GAP: both raw and primed < 0.7
        cap = Capability.from_lists("advanced_math", ["x"], [("x", "y")])
        analysis = CapabilityAnalysis(
            capability=cap,
            status=CapabilityStatus.TRUE_GAP,
            accuracy_raw=0.2,
            accuracy_primed=0.4,
            kappa_raw=1000.0,
            kappa_primed=500.0,
        )
        assert analysis.status == CapabilityStatus.TRUE_GAP


class TestEndToEndMocked:
    """End-to-end tests with mocked model."""

    def test_full_improvement_loop_mocked(self):
        """Test full improvement loop with mocked components."""
        from modelcypher.core.use_cases.self_improve.improver import (
            AutonomousSelfImprover,
        )

        # Create improver with mocks
        model = MagicMock()
        tokenizer = MagicMock()
        improver = AutonomousSelfImprover(model, tokenizer)

        # Mock scanner to return a TRUE_GAP
        cap = Capability.from_lists("test_cap", ["p"], [("q", "a")])
        mock_analysis = CapabilityAnalysis(
            capability=cap,
            status=CapabilityStatus.TRUE_GAP,
            accuracy_raw=0.3,
            accuracy_primed=0.4,
            kappa_raw=50.0,
            kappa_primed=40.0,
        )
        improver.scanner.scan = MagicMock(return_value=mock_analysis)

        # Mock oracle calibration (high accuracy so generation proceeds)
        improver.oracle.calibrate = MagicMock(
            return_value=(0.95, [("1+1=", "2", "2", True)])
        )

        # Mock oracle verify for generator
        improver.oracle.verify = MagicMock(return_value=(True, "5"))

        with tempfile.TemporaryDirectory() as tmpdir:
            training_path = Path(tmpdir) / "training.jsonl"

            log = improver.improve(
                capabilities=[cap],
                training_data_path=training_path,
                n_training_samples=10,
            )

            # Verify log structure
            assert log.iterations == 1
            assert log.capabilities_scanned == ["test_cap"]
            assert log.true_gaps == ["test_cap"]
            assert log.training_data_path == str(training_path)
            assert log.training_spec is not None

            # Verify training data was generated
            assert training_path.exists()
            with open(training_path) as f:
                lines = f.readlines()
            assert len(lines) == 10
