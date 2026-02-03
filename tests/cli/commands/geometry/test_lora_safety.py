# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Tests for LoRA Safety CLI commands

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


# Fixtures paths
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"
PROMPTS_FILE = FIXTURES_DIR / "lora_safety_prompts.json"
PROBLEMS_FILE = FIXTURES_DIR / "lora_safety_problems.jsonl"


class TestLoRASafetyHelp:
    """Test lora-safety command help."""

    def test_lora_safety_help(self):
        """lora-safety --help should show usage information."""
        result = runner.invoke(app, ["geometry", "lora-safety", "--help"])
        assert result.exit_code == 0
        assert "recommend" in result.output
        assert "check-barrier" in result.output
        assert "score-curriculum" in result.output

    def test_recommend_help(self):
        """recommend --help should show Fisher-guided targeting info."""
        result = runner.invoke(app, ["geometry", "lora-safety", "recommend", "--help"])
        assert result.exit_code == 0
        assert "Fisher" in result.output
        assert "--prompts" in result.output
        assert "--top-k" in result.output

    def test_check_barrier_help(self):
        """check-barrier --help should show safety level info."""
        result = runner.invoke(app, ["geometry", "lora-safety", "check-barrier", "--help"])
        assert result.exit_code == 0
        assert "barrier" in result.output.lower()
        assert "SAFE" in result.output
        assert "CAUTION" in result.output
        assert "WARNING" in result.output

    def test_score_curriculum_help(self):
        """score-curriculum --help should show Goldilocks info."""
        result = runner.invoke(app, ["geometry", "lora-safety", "score-curriculum", "--help"])
        assert result.exit_code == 0
        assert "Goldilocks" in result.output
        assert "--problems" in result.output
        assert "--reference" in result.output


class TestLoRASafetyMissingArgs:
    """Test error handling for missing arguments."""

    def test_recommend_missing_model(self):
        """recommend without model should error."""
        result = runner.invoke(
            app,
            ["geometry", "lora-safety", "recommend", "--prompts", str(PROMPTS_FILE)],
        )
        assert result.exit_code != 0

    def test_recommend_missing_prompts(self):
        """recommend without --prompts should error."""
        result = runner.invoke(
            app,
            ["geometry", "lora-safety", "recommend", "/path/to/model"],
        )
        assert result.exit_code != 0
        assert "prompts" in result.output.lower()

    def test_check_barrier_missing_target(self):
        """check-barrier without target should error."""
        result = runner.invoke(
            app,
            [
                "geometry", "lora-safety", "check-barrier",
                "/path/to/base",
                "--prompts", str(PROMPTS_FILE),
            ],
        )
        assert result.exit_code != 0

    def test_score_curriculum_missing_problems(self):
        """score-curriculum without --problems should error."""
        result = runner.invoke(
            app,
            ["geometry", "lora-safety", "score-curriculum", "/path/to/model"],
        )
        assert result.exit_code != 0
        assert "problems" in result.output.lower()


class TestGoldilocksQuality:
    """Tests for the Goldilocks quality domain module."""

    def test_goldilocks_quality_result_fields(self):
        """Test that GoldilocksQualityResult has expected fields."""
        from modelcypher.core.domain.geometry.goldilocks_quality import GoldilocksQualityResult

        result = GoldilocksQualityResult(
            quality_score=0.85,
            cka_similarity=0.89,
            barrier_height=0.045,
            fisher_mean=0.001,
            cka_goldilocks=0.95,
            barrier_score=1.0,
            fisher_learning=0.9,
            quality_level="high",
        )

        assert result.quality_score == 0.85
        assert result.quality_level == "high"
        assert result.cka_goldilocks == 0.95

    def test_cka_goldilocks_score_peaks_at_0_9(self):
        """Test that CKA Goldilocks score peaks at 0.9."""
        from modelcypher.core.domain.geometry.goldilocks_quality import (
            _compute_cka_goldilocks_score,
        )

        # Peak at 0.9
        assert _compute_cka_goldilocks_score(0.90) == 1.0

        # Drops off both sides
        assert _compute_cka_goldilocks_score(0.99) < 1.0
        assert _compute_cka_goldilocks_score(0.80) < 1.0

    def test_cka_goldilocks_score_symmetric(self):
        """Test that CKA Goldilocks is roughly symmetric around 0.9."""
        from modelcypher.core.domain.geometry.goldilocks_quality import (
            _compute_cka_goldilocks_score,
        )

        score_high = _compute_cka_goldilocks_score(0.95)
        score_low = _compute_cka_goldilocks_score(0.85)
        assert abs(score_high - score_low) < 0.01

    def test_barrier_score_optimal_zone(self):
        """Test that barrier score is optimal in 0.02-0.10 range."""
        from modelcypher.core.domain.geometry.goldilocks_quality import (
            _compute_barrier_score,
        )

        # Optimal zone
        assert _compute_barrier_score(0.02) == 1.0
        assert _compute_barrier_score(0.05) == 1.0
        assert _compute_barrier_score(0.10) == 1.0

    def test_barrier_score_below_optimal(self):
        """Test that barrier score is reduced below 0.02."""
        from modelcypher.core.domain.geometry.goldilocks_quality import (
            _compute_barrier_score,
        )

        # Below optimal (too easy)
        assert _compute_barrier_score(0.01) < 1.0
        assert _compute_barrier_score(0.005) < 0.5

    def test_barrier_score_above_optimal(self):
        """Test that barrier score is reduced above 0.10."""
        from modelcypher.core.domain.geometry.goldilocks_quality import (
            _compute_barrier_score,
        )

        # Above optimal (too hard)
        assert _compute_barrier_score(0.15) < 1.0

    def test_fisher_learning_score_inverse(self):
        """Test that Fisher learning score is inverse of Fisher mean."""
        from modelcypher.core.domain.geometry.goldilocks_quality import (
            _compute_fisher_learning_score,
        )

        # Low Fisher = high learning opportunity
        assert _compute_fisher_learning_score(0.0) == 1.0
        assert _compute_fisher_learning_score(0.001) >= 0.9

    def test_fisher_learning_score_decreases(self):
        """Test that Fisher learning score decreases with higher Fisher."""
        from modelcypher.core.domain.geometry.goldilocks_quality import (
            _compute_fisher_learning_score,
        )

        # Higher Fisher = lower learning opportunity
        score_low = _compute_fisher_learning_score(0.001)
        score_high = _compute_fisher_learning_score(0.005)
        assert score_high < score_low


class TestLoRASafetyService:
    """Tests for the LoRA Safety Service."""

    def test_service_instantiation(self):
        """Test that LoRASafetyService can be instantiated."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        service = LoRASafetyService()
        assert service is not None

    def test_service_has_required_methods(self):
        """Test that service has all required methods."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        service = LoRASafetyService()
        assert hasattr(service, "recommend_target_modules")
        assert hasattr(service, "check_barrier_safety")
        assert hasattr(service, "score_curriculum")

    def test_service_constants(self):
        """Test that service has expected safety thresholds."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        assert LoRASafetyService.BARRIER_SAFE == 0.01
        assert LoRASafetyService.BARRIER_CAUTION == 0.03
        assert LoRASafetyService.FISHER_EXCELLENT < LoRASafetyService.FISHER_GOOD
        assert LoRASafetyService.FISHER_GOOD < LoRASafetyService.FISHER_ACCEPTABLE
