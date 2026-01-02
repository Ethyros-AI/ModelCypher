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

from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.use_cases.thermo_service import (
    ThermoDetectResult,
    ThermoMeasureResult,
    ThermoService,
)


# **Feature: cli-mcp-parity, Property 2: Thermo detect returns raw measurements**
# **Validates: Requirements 1.5**
@given(
    prompt=st.text(min_size=1, max_size=200),
)
@settings(max_examples=100, deadline=None)
def test_thermo_detect_returns_raw_measurements(prompt: str):
    """Property 2: For any prompt, detect() returns raw entropy measurements.

    ThermoDetectResult contains:
    - baseline_entropy: entropy of original prompt
    - intensity_entropy: entropy of modified prompt
    - delta_h: difference in entropy
    - processing_time: time to process

    No interpretation or classification is provided - caller decides meaning.
    """
    service = ThermoService()
    # Use a dummy model path since we're using simulated entropy
    result = service.detect(prompt, "/tmp/model")

    # Verify result is correct type
    assert isinstance(result, ThermoDetectResult)

    # Verify entropy values are non-negative
    assert result.baseline_entropy >= 0.0
    assert result.intensity_entropy >= 0.0

    # delta_h can be positive or negative (entropy increase or decrease)
    assert isinstance(result.delta_h, float)

    # Verify processing_time is non-negative
    assert result.processing_time >= 0.0

    # Verify prompt is preserved
    assert result.prompt == prompt


# **Feature: cli-mcp-parity, Property 3: Thermo detect-batch preserves count**
# **Validates: Requirements 1.6**
@given(
    prompts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20),
)
@settings(max_examples=100, deadline=None)
def test_thermo_detect_batch_preserves_count(prompts: list[str]):
    """Property 3: For any prompts file with N prompts, detect_batch() returns exactly N results."""
    service = ThermoService()

    # Create a temporary file with prompts
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(prompts, f)
        prompts_file = f.name

    try:
        results = service.detect_batch(prompts_file, "/tmp/model")

        # Verify count is preserved
        assert len(results) == len(prompts)

        # Verify each result is valid raw measurement
        for i, result in enumerate(results):
            assert isinstance(result, ThermoDetectResult)
            assert result.baseline_entropy >= 0.0
            assert result.intensity_entropy >= 0.0
            assert isinstance(result.delta_h, float)
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


def test_thermo_detect_returns_consistent_delta():
    """Test that delta_h is consistent with baseline and intensity entropy."""
    service = ThermoService()
    result = service.detect("Test prompt", "/tmp/model")

    # delta_h should be intensity_entropy - baseline_entropy
    # (within floating point tolerance)
    expected_delta = result.intensity_entropy - result.baseline_entropy
    assert abs(result.delta_h - expected_delta) < 1e-6
