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

"""Tests for the Geometric Self-Study Sandbox."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.sandbox.feedback_formatter import (
    EntropyPattern,
    GeometricFeedback,
    classify_entropy_pattern,
    format_feedback_text,
    format_geometric_feedback,
)
from modelcypher.core.domain.sandbox.curriculum import (
    Curriculum,
    CurriculumExample,
    CurriculumLevel,
    get_builtin_curriculum,
)


class TestFeedbackFormatter:
    """Tests for the feedback formatter."""

    def test_format_geometric_feedback_aligned(self):
        """Test feedback formatting for aligned geometry."""
        feedback = format_geometric_feedback(
            comp_phi=1.0,
            peak_layer=8.0,
            n_layers=16,
            expansion_rate=0.5,
            compression_rate=0.5,
        )

        assert feedback.comp_phi == 1.0
        assert feedback.peak_layer == 8.0
        assert feedback.n_layers == 16
        assert feedback.peak_layer_fraction == 0.5
        assert "ALIGNED" in feedback.interpretation or "optimal" in feedback.interpretation.lower()

    def test_format_geometric_feedback_under(self):
        """Test feedback formatting for under-expanded geometry."""
        feedback = format_geometric_feedback(
            comp_phi=0.6,
            peak_layer=14.0,
            n_layers=16,
            expansion_rate=0.1,
            compression_rate=0.05,
        )

        assert feedback.comp_phi == 0.6
        assert "shallow" in feedback.interpretation.lower() or "narrow" in feedback.interpretation.lower()

    def test_format_geometric_feedback_over(self):
        """Test feedback formatting for over-expanded geometry."""
        feedback = format_geometric_feedback(
            comp_phi=1.6,
            peak_layer=4.0,
            n_layers=16,
            expansion_rate=0.8,
            compression_rate=0.2,
        )

        assert feedback.comp_phi == 1.6
        assert "focus" in feedback.interpretation.lower() or "unfocused" in feedback.interpretation.lower()

    def test_format_feedback_text(self):
        """Test text formatting of feedback."""
        feedback = GeometricFeedback(
            comp_phi=0.98,
            peak_layer=8.0,
            n_layers=16,
            entropy_pattern=EntropyPattern.EXPAND_COMPRESS,
            expansion_rate=0.5,
            compression_rate=0.5,
            interpretation="Processing geometry is optimal.",
        )

        text = format_feedback_text(feedback)

        assert "=== GEOMETRIC FEEDBACK ===" in text
        assert "comp/phi:" in text
        assert "0.98" in text
        assert "peak_layer:" in text
        assert "8.0/16" in text
        assert "entropy_pattern:" in text
        assert "===========================" in text


class TestEntropyPatternClassification:
    """Tests for entropy pattern classification."""

    def test_classify_flat_pattern(self):
        """Test classification of flat entropy trajectory."""
        trajectory = [1.0, 1.01, 0.99, 1.0, 1.02, 0.98]
        pattern = classify_entropy_pattern(trajectory)
        assert pattern == EntropyPattern.FLAT

    def test_classify_expand_compress(self):
        """Test classification of expand-compress pattern."""
        trajectory = [1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0]
        pattern = classify_entropy_pattern(trajectory)
        assert pattern == EntropyPattern.EXPAND_COMPRESS

    def test_classify_monotonic_increase(self):
        """Test classification of monotonically increasing pattern."""
        trajectory = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        pattern = classify_entropy_pattern(trajectory)
        assert pattern == EntropyPattern.MONOTONIC_INCREASE

    def test_classify_monotonic_decrease(self):
        """Test classification of monotonically decreasing pattern."""
        trajectory = [3.5, 3.0, 2.5, 2.0, 1.5, 1.0]
        pattern = classify_entropy_pattern(trajectory)
        assert pattern == EntropyPattern.MONOTONIC_DECREASE

    def test_short_trajectory_returns_flat(self):
        """Test that short trajectories return flat."""
        pattern = classify_entropy_pattern([1.0, 2.0])
        assert pattern == EntropyPattern.FLAT


class TestCurriculum:
    """Tests for curriculum loading."""

    def test_curriculum_example_from_dict(self):
        """Test creating curriculum example from dict."""
        data = {
            "prompt": "Test prompt",
            "expected_answer": "Test answer",
            "level": 2,
            "geometry_hint": "Test hint",
        }
        example = CurriculumExample.from_dict(data)

        assert example.prompt == "Test prompt"
        assert example.expected_answer == "Test answer"
        assert example.level == CurriculumLevel.PREDICTION
        assert example.geometry_hint == "Test hint"

    def test_curriculum_example_to_dict(self):
        """Test converting curriculum example to dict."""
        example = CurriculumExample(
            prompt="Test",
            expected_answer="Answer",
            level=CurriculumLevel.SELECTION,
            approaches=["A", "B"],
        )
        data = example.to_dict()

        assert data["prompt"] == "Test"
        assert data["expected_answer"] == "Answer"
        assert data["level"] == 3
        assert data["approaches"] == ["A", "B"]

    def test_get_builtin_curriculum(self):
        """Test loading built-in curriculum."""
        curriculum = get_builtin_curriculum()

        assert len(curriculum) > 0
        assert curriculum.name is not None
        assert len(curriculum.levels) == 4  # 4 levels

    def test_curriculum_level_organization(self):
        """Test that curriculum organizes examples by level."""
        examples = [
            CurriculumExample(prompt="L1", level=CurriculumLevel.OBSERVATION),
            CurriculumExample(prompt="L2", level=CurriculumLevel.PREDICTION),
            CurriculumExample(prompt="L3", level=CurriculumLevel.SELECTION),
            CurriculumExample(prompt="L1b", level=CurriculumLevel.OBSERVATION),
        ]
        curriculum = Curriculum(name="test", description="test", examples=examples)

        assert len(curriculum.get_level(CurriculumLevel.OBSERVATION)) == 2
        assert len(curriculum.get_level(CurriculumLevel.PREDICTION)) == 1
        assert len(curriculum.get_level(CurriculumLevel.SELECTION)) == 1
        assert len(curriculum.get_level(CurriculumLevel.CORRECTION)) == 0


class TestGeometricSelfStudyExamples:
    """Tests for geometric self-study training examples."""

    def test_examples_load(self):
        """Test that geometric examples load correctly."""
        from modelcypher.core.domain.training.self_reflection import (
            get_geometric_self_study_examples,
        )

        examples = get_geometric_self_study_examples()

        assert len(examples) > 0
        assert all(hasattr(ex, "full_text") for ex in examples)
        assert all(hasattr(ex, "geometry_note") for ex in examples)

    def test_examples_have_required_fields(self):
        """Test that examples have all required fields."""
        from modelcypher.core.domain.training.self_reflection import (
            get_geometric_self_study_examples,
        )

        examples = get_geometric_self_study_examples()

        for ex in examples:
            assert ex.prompt is not None
            assert ex.completion is not None
            assert ex.geometry_note is not None
            assert len(ex.full_text) > 0

    def test_geometry_check_pattern_present(self):
        """Test that GEOMETRY_CHECK pattern is present in examples."""
        from modelcypher.core.domain.training.self_reflection import (
            get_geometric_self_study_examples,
        )

        examples = get_geometric_self_study_examples()
        has_geometry_check = any("GEOMETRY_CHECK" in ex.completion for ex in examples)

        assert has_geometry_check, "Should have examples with GEOMETRY_CHECK pattern"
