"""Property tests for LocalInferenceEngine."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.adapters.local_inference import (
    InferenceSuiteResult,
    LocalInferenceEngine,
)


# **Feature: cli-mcp-parity, Property 6: Inference suite preserves prompt count**
# **Validates: Requirements 8.2**
@given(
    prompts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20),
)
@settings(max_examples=100, deadline=None)
def test_inference_suite_preserves_prompt_count(prompts: list[str]):
    """Property 6: For any suite file with N prompts, suite() returns exactly N case results."""
    engine = LocalInferenceEngine()

    # Create a temporary model directory (the engine validates model path exists)
    with tempfile.TemporaryDirectory() as model_dir:
        # Create a temporary suite file with prompts as JSON array
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(prompts, f)
            suite_file = f.name

        try:
            result = engine.suite(
                model=model_dir,
                suite_file=suite_file,
            )

            # Verify result is correct type
            assert isinstance(result, InferenceSuiteResult)

            # Verify count is preserved - this is the core property
            assert len(result.cases) == len(prompts)
            assert result.total_cases == len(prompts)

            # Verify each case has required fields
            for i, case in enumerate(result.cases):
                assert case.name is not None
                assert case.prompt is not None
                assert case.duration >= 0.0
                assert case.token_count >= 0

        finally:
            Path(suite_file).unlink()


@given(
    prompts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10),
)
@settings(max_examples=50, deadline=None)
def test_inference_suite_txt_format_preserves_count(prompts: list[str]):
    """Test suite with plain text format (one prompt per line)."""
    # Filter out prompts with newlines since they'd break the format
    prompts = [p.replace("\n", " ").replace("\r", " ").strip() for p in prompts if p.strip()]
    if not prompts:
        return  # Skip if all prompts were empty

    engine = LocalInferenceEngine()

    with tempfile.TemporaryDirectory() as model_dir:
        # Create a temporary suite file with newline-separated prompts
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(prompts))
            suite_file = f.name

        try:
            result = engine.suite(
                model=model_dir,
                suite_file=suite_file,
            )

            # Verify count is preserved
            assert len(result.cases) == len(prompts)
            assert result.total_cases == len(prompts)

        finally:
            Path(suite_file).unlink()


@given(
    test_configs=st.lists(
        st.fixed_dictionaries({
            "name": st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
            "prompt": st.text(min_size=1, max_size=100),
        }),
        min_size=1,
        max_size=15,
    ),
)
@settings(max_examples=50, deadline=None)
def test_inference_suite_config_format_preserves_count(test_configs: list[dict]):
    """Test suite with structured config format containing tests."""
    engine = LocalInferenceEngine()

    with tempfile.TemporaryDirectory() as model_dir:
        # Create a suite config with tests
        config = {
            "name": "test_suite",
            "tests": test_configs,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            suite_file = f.name

        try:
            result = engine.suite(
                model=model_dir,
                suite_file=suite_file,
            )

            # Verify count is preserved
            assert len(result.cases) == len(test_configs)
            assert result.total_cases == len(test_configs)

        finally:
            Path(suite_file).unlink()
