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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.use_cases.thermo_service import (
    ThermoDetectResult,
    ThermoMeasureResult,
    ThermoService,
)


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


class DummyTokenizer:
    def __init__(self, vocab_size: int = 16, model_max_length: int = 32) -> None:
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.eos_token_id = vocab_size - 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not text.strip():
            return [0] if add_special_tokens else []
        tokens = []
        for part in text.split():
            token_id = sum(ord(ch) for ch in part) % (self.vocab_size - 1)
            tokens.append(token_id)
        return tokens or ([0] if add_special_tokens else [])

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"<t{token_id}>" for token_id in token_ids)


class DummyModel:
    def __init__(self, backend, vocab_size: int) -> None:
        self._backend = backend
        self._vocab_size = vocab_size

    def __call__(self, input_ids):
        seq_len = int(input_ids.shape[1])
        vocab = self._backend.arange(self._vocab_size)
        vocab = vocab + 0.0
        logits = self._backend.tile(vocab, (seq_len, 1))
        return self._backend.expand_dims(logits, axis=0)


class DummyModelLoader:
    def __init__(self, model, tokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def load_model_for_training(self, model_path, lora_config=None, adapter_path=None):
        return self._model, self._tokenizer


def _make_service() -> ThermoService:
    backend = get_default_backend()
    tokenizer = DummyTokenizer()
    model = DummyModel(backend, tokenizer.vocab_size)
    loader = DummyModelLoader(model, tokenizer)
    return ThermoService(model_loader=loader)


# **Feature: cli-parity, Property 2: Thermo detect returns raw measurements**
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
    service = _make_service()
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = service.detect(prompt, tmp_dir)

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


# **Feature: cli-parity, Property 3: Thermo detect-batch preserves count**
# **Validates: Requirements 1.6**
@given(
    prompts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20),
)
@settings(max_examples=100, deadline=None)
def test_thermo_detect_batch_preserves_count(prompts: list[str]):
    """Property 3: For any prompts file with N prompts, detect_batch() returns exactly N results."""
    service = _make_service()

    # Create a temporary file with prompts
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(prompts, f)
        prompts_file = f.name

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = service.detect_batch(prompts_file, tmp_dir)

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

    service = _make_service()

    # Create a temporary file with newline-separated prompts
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(prompts))
        prompts_file = f.name

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = service.detect_batch(prompts_file, tmp_dir)

        # Verify count is preserved
        assert len(results) == len(prompts)
    finally:
        Path(prompts_file).unlink()


def test_thermo_measure_returns_statistics():
    """Test that measure returns valid statistics."""
    service = _make_service()
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = service.measure("Test prompt", tmp_dir)

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
    service = _make_service()
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = service.detect("Test prompt", tmp_dir)

    # delta_h should be intensity_entropy - baseline_entropy
    # (within floating point tolerance)
    expected_delta = result.intensity_entropy - result.baseline_entropy
    eps = _eps(result.delta_h, expected_delta)
    assert abs(result.delta_h - expected_delta) <= eps
