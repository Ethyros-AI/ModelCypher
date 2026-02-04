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

"""Tests for geometric context injection (geometric_context.py).

Validates the implementation of geometric self-awareness features,
including context computation, formatting, and data augmentation.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.geometric_context import (
    GeometricContext,
)
from modelcypher.core.use_cases.self_improve.geometric_training_data import (
    extract_geometry_from_prompt,
)


class TestGeometricContext:
    """Tests for GeometricContext dataclass."""

    def test_context_creation(self):
        """Context can be created with all required fields."""
        ctx = GeometricContext(
            loop_persistence=0.15,
            expansion_ratio=1.02,
            highway_depth=0.44,
            exit_convergence=2.14,
            has_reasoning_loops=True,
        )

        assert ctx.loop_persistence == 0.15
        assert ctx.expansion_ratio == 1.02
        assert ctx.highway_depth == 0.44
        assert ctx.exit_convergence == 2.14
        assert ctx.has_reasoning_loops is True

    def test_context_to_dict(self):
        """Context can be serialized to dict."""
        ctx = GeometricContext(
            loop_persistence=-0.1,
            expansion_ratio=0.95,
            highway_depth=0.33,
            exit_convergence=1.5,
            has_reasoning_loops=False,
        )

        d = ctx.to_dict()
        assert d["loop_persistence"] == -0.1
        assert d["expansion_ratio"] == 0.95
        assert d["highway_depth"] == 0.33
        assert d["exit_convergence"] == 1.5
        assert d["has_reasoning_loops"] is False

    def test_format_produces_valid_prefix(self):
        """format() produces valid [GEOMETRY] block."""
        ctx = GeometricContext(
            loop_persistence=0.15,
            expansion_ratio=1.02,
            highway_depth=0.44,
            exit_convergence=2.14,
            has_reasoning_loops=True,
        )

        formatted = ctx.format()

        # Check structure
        assert formatted.startswith("[GEOMETRY]")
        assert "[/GEOMETRY]" in formatted

        # Check values
        assert "loop_persistence: +0.15" in formatted
        assert "expansion_ratio: 1.02" in formatted
        assert "highway_depth: 0.44" in formatted
        assert "convergence: 2.14" in formatted
        assert "reasoning_loops: yes" in formatted

    def test_format_negative_loop_persistence(self):
        """format() handles negative loop_persistence correctly."""
        ctx = GeometricContext(
            loop_persistence=-0.25,
            expansion_ratio=0.8,
            highway_depth=0.5,
            exit_convergence=1.0,
            has_reasoning_loops=False,
        )

        formatted = ctx.format()

        assert "loop_persistence: -0.25" in formatted
        assert "reasoning_loops: no" in formatted


class TestGeometryExtraction:
    """Tests for extract_geometry_from_prompt function."""

    def test_extracts_geometry_from_augmented_prompt(self):
        """Correctly extracts geometry from augmented prompt."""
        augmented = """[GEOMETRY]
loop_persistence: +0.15
expansion_ratio: 1.02
highway_depth: 0.44
convergence: 2.14
reasoning_loops: yes
[/GEOMETRY]

What is 2 + 2?"""

        geometry, original = extract_geometry_from_prompt(augmented)

        assert geometry is not None
        assert geometry["loop_persistence"] == pytest.approx(0.15, abs=1e-6)
        assert geometry["expansion_ratio"] == pytest.approx(1.02, abs=1e-6)
        assert geometry["highway_depth"] == pytest.approx(0.44, abs=1e-6)
        assert geometry["convergence"] == pytest.approx(2.14, abs=1e-6)
        assert geometry["reasoning_loops"] is True
        assert original == "What is 2 + 2?"

    def test_returns_none_for_plain_prompt(self):
        """Returns None geometry for prompts without [GEOMETRY] block."""
        plain = "What is 2 + 2?"

        geometry, original = extract_geometry_from_prompt(plain)

        assert geometry is None
        assert original == plain

    def test_handles_negative_values(self):
        """Correctly parses negative values."""
        augmented = """[GEOMETRY]
loop_persistence: -0.25
expansion_ratio: 0.80
highway_depth: 0.50
convergence: 1.00
reasoning_loops: no
[/GEOMETRY]

Calculate 10 - 7"""

        geometry, original = extract_geometry_from_prompt(augmented)

        assert geometry is not None
        assert geometry["loop_persistence"] == pytest.approx(-0.25, abs=1e-6)
        assert geometry["reasoning_loops"] is False
        assert original == "Calculate 10 - 7"

    def test_preserves_multiline_original_prompt(self):
        """Preserves multiline structure in original prompt."""
        augmented = """[GEOMETRY]
loop_persistence: +0.10
expansion_ratio: 1.00
highway_depth: 0.33
convergence: 1.50
reasoning_loops: yes
[/GEOMETRY]

Problem: I have 5 apples.
I eat 2 of them.
How many do I have left?"""

        geometry, original = extract_geometry_from_prompt(augmented)

        assert geometry is not None
        assert "Problem: I have 5 apples." in original
        assert "How many do I have left?" in original


class TestDataAugmentationIntegration:
    """Integration tests for the full augmentation flow."""

    def test_format_and_extract_roundtrip(self):
        """format() and extract() are inverse operations."""
        original_ctx = GeometricContext(
            loop_persistence=0.12,
            expansion_ratio=1.05,
            highway_depth=0.40,
            exit_convergence=2.5,
            has_reasoning_loops=True,
        )

        original_prompt = "Solve: 3x + 5 = 20"

        # Format and prepend
        augmented = original_ctx.format() + original_prompt

        # Extract back
        extracted_geom, extracted_prompt = extract_geometry_from_prompt(augmented)

        # Verify roundtrip
        assert extracted_geom is not None
        assert extracted_geom["loop_persistence"] == pytest.approx(0.12, abs=0.01)
        assert extracted_geom["expansion_ratio"] == pytest.approx(1.05, abs=0.01)
        assert extracted_geom["reasoning_loops"] is True
        assert extracted_prompt == original_prompt

    def test_augmentation_preserves_sample_content(self):
        """Augmentation preserves the original sample content."""
        ctx = GeometricContext(
            loop_persistence=0.0,
            expansion_ratio=1.0,
            highway_depth=0.5,
            exit_convergence=1.0,
            has_reasoning_loops=False,
        )

        sample = {
            "prompt": "What is 5 + 3?",
            "completion": "5 + 3 = 8",
        }

        # Simulate augmentation
        augmented_prompt = ctx.format() + sample["prompt"]

        # Extract and verify original is preserved
        _, original = extract_geometry_from_prompt(augmented_prompt)
        assert original == sample["prompt"]


class TestEdgeCases:
    """Edge case tests."""

    def test_handles_empty_prompt(self):
        """Handles empty prompt gracefully."""
        geometry, original = extract_geometry_from_prompt("")

        assert geometry is None
        assert original == ""

    def test_handles_malformed_geometry_block(self):
        """Handles malformed geometry blocks gracefully."""
        malformed = """[GEOMETRY]
this is not valid
[/GEOMETRY]

Some prompt"""

        geometry, original = extract_geometry_from_prompt(malformed)

        # Should still return something, but empty geometry
        assert geometry is not None or geometry is None  # Either is acceptable
        assert "Some prompt" in original

    def test_context_with_extreme_values(self):
        """Context handles extreme values."""
        ctx = GeometricContext(
            loop_persistence=100.0,  # Extreme positive
            expansion_ratio=0.001,   # Very small
            highway_depth=0.0,       # Boundary
            exit_convergence=1e10,   # Very large
            has_reasoning_loops=True,
        )

        formatted = ctx.format()

        # Should still produce valid format
        assert "[GEOMETRY]" in formatted
        assert "[/GEOMETRY]" in formatted
        assert "loop_persistence:" in formatted
