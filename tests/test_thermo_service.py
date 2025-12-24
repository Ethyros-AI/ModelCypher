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

"""Property tests for ThermoService."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.use_cases.thermo_service import (
    DETECT_PRESETS,
    ThermoDetectResult,
    ThermoMeasureResult,
    ThermoService,
)


# **Feature: cli-mcp-parity, Property 2: Thermo detect returns valid classification**
# **Validates: Requirements 1.5**
@given(
    prompt=st.text(min_size=1, max_size=200),
    preset=st.sampled_from(["default", "strict", "sensitive", "quick"]),
)
@settings(max_examples=100, deadline=None)
def test_thermo_detect_returns_valid_classification(prompt: str, preset: str):
    """Property 2: For any prompt and model, detect() returns a classification in
    {"safe", "unsafe", "ambiguous"} with risk_level in [0, 3] and confidence in [0, 1].
    """
    service = ThermoService()
    # Use a dummy model path since we're using simulated entropy
    result = service.detect(prompt, "/tmp/model", preset)

    # Verify result is correct type
    assert isinstance(result, ThermoDetectResult)

    # Verify classification is valid
    assert result.classification in {"safe", "unsafe", "ambiguous"}

    # Verify risk_level is in [0, 3]
    assert 0 <= result.risk_level <= 3

    # Verify confidence is in [0, 1]
    assert 0.0 <= result.confidence <= 1.0

    # Verify entropy values are non-negative
    assert result.baseline_entropy >= 0.0
    assert result.intensity_entropy >= 0.0

    # Verify processing_time is non-negative
    assert result.processing_time >= 0.0

    # Verify prompt is preserved
    assert result.prompt == prompt


# **Feature: cli-mcp-parity, Property 3: Thermo detect-batch preserves count**
# **Validates: Requirements 1.6**
@given(
    prompts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20),
    preset=st.sampled_from(["default", "strict", "sensitive", "quick"]),
)
@settings(max_examples=100, deadline=None)
def test_thermo_detect_batch_preserves_count(prompts: list[str], preset: str):
    """Property 3: For any prompts file with N prompts, detect_batch() returns exactly N results."""
    service = ThermoService()

    # Create a temporary file with prompts
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(prompts, f)
        prompts_file = f.name

    try:
        results = service.detect_batch(prompts_file, "/tmp/model", preset)

        # Verify count is preserved
        assert len(results) == len(prompts)

        # Verify each result is valid
        for i, result in enumerate(results):
            assert isinstance(result, ThermoDetectResult)
            assert result.classification in {"safe", "unsafe", "ambiguous"}
            assert 0 <= result.risk_level <= 3
            assert 0.0 <= result.confidence <= 1.0
            assert result.prompt == prompts[i]
    finally:
        Path(prompts_file).unlink()


@given(
    prompts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10),
)
@settings(max_examples=50, deadline=None)
def test_thermo_detect_batch_newline_format(prompts: list[str]):
    """Test detect_batch with newline-separated format."""
    # Filter out prompts with newlines or carriage returns since they'd break the format
    prompts = [p.replace("\n", " ").replace("\r", " ").strip() for p in prompts if p.strip()]
    if not prompts:
        return  # Skip if all prompts were empty

    service = ThermoService()

    # Create a temporary file with newline-separated prompts
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(prompts))
        prompts_file = f.name

    try:
        results = service.detect_batch(prompts_file, "/tmp/model")

        # Verify count is preserved
        assert len(results) == len(prompts)
    finally:
        Path(prompts_file).unlink()


def test_thermo_measure_returns_statistics():
    """Test that measure returns valid statistics."""
    service = ThermoService()
    result = service.measure("Test prompt", "/tmp/model")

    assert isinstance(result, ThermoMeasureResult)
    assert result.base_prompt == "Test prompt"
    assert len(result.measurements) > 0
    assert result.statistics.mean_entropy >= 0.0
    assert result.statistics.std_entropy >= 0.0
    assert result.statistics.min_entropy >= 0.0
    assert result.statistics.max_entropy >= 0.0
    assert result.statistics.min_entropy <= result.statistics.max_entropy


def test_thermo_detect_presets_exist():
    """Test that all presets are properly configured."""
    for preset_name, config in DETECT_PRESETS.items():
        assert "threshold_safe" in config
        assert "threshold_unsafe" in config
        assert "modifiers" in config
        assert config["threshold_safe"] < config["threshold_unsafe"]
        assert len(config["modifiers"]) > 0
