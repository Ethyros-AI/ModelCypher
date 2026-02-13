# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.exceptions import CheckpointError
from modelcypher.core.domain.training.types import (
    ComputePrecision,
    FineTuneType,
    LoRASettings,
    PreflightResult,
    TrainingStatus,
)


class TestLoRASettings:
    """Tests for LoRASettings.scale — geometry-derived, not a hyperparameter."""

    def test_scale_normal(self):
        """rank=8, alpha=16 -> scale=2.0."""
        s = LoRASettings(rank=8, alpha=16.0, dropout=0.0, target_modules=["q_proj"])
        assert s.scale == pytest.approx(2.0)

    def test_scale_rank_zero_uses_max_clamp(self):
        """rank=0 -> max(0,1)=1 -> scale=alpha. Prevents division by zero."""
        s = LoRASettings(rank=0, alpha=4.0, dropout=0.0, target_modules=["q_proj"])
        assert s.scale == pytest.approx(4.0)

    def test_scale_rank_one(self):
        """rank=1 -> scale=alpha/1=alpha."""
        s = LoRASettings(rank=1, alpha=0.5, dropout=0.0, target_modules=[])
        assert s.scale == pytest.approx(0.5)

    def test_scale_large_rank_small_alpha(self):
        """rank=64, alpha=32 -> 0.5 (typical high-rank adapter)."""
        s = LoRASettings(rank=64, alpha=32.0, dropout=0.0, target_modules=["v_proj"])
        assert s.scale == pytest.approx(0.5)

    def test_default_fine_tune_type_is_lora(self):
        """Default is LORA, not DORA."""
        s = LoRASettings(rank=8, alpha=16.0, dropout=0.0, target_modules=[])
        assert s.fine_tune_type == FineTuneType.LORA

    def test_dora_type_settable(self):
        """DORA fine-tune type can be set explicitly."""
        s = LoRASettings(
            rank=8, alpha=16.0, dropout=0.0, target_modules=[],
            fine_tune_type=FineTuneType.DORA,
        )
        assert s.fine_tune_type == FineTuneType.DORA


class TestComputePrecision:
    """Tests for ComputePrecision str enum."""

    def test_usable_as_string(self):
        """str enum values work in string comparisons (JSON serialization)."""
        assert ComputePrecision.FLOAT32 == "float32"
        assert ComputePrecision.FLOAT16 == "float16"
        assert ComputePrecision.BFLOAT16 == "bfloat16"

    def test_all_values_distinct(self):
        """No duplicate precision values."""
        values = [m.value for m in ComputePrecision]
        assert len(values) == len(set(values))


class TestTrainingStatus:
    """Tests for TrainingStatus str enum."""

    def test_usable_as_string(self):
        """Status values can be compared as strings."""
        assert TrainingStatus.running == "running"
        assert TrainingStatus.failed == "failed"

    def test_terminal_states_present(self):
        """Completed, failed, cancelled must all exist (state machine requires them)."""
        names = {m.name for m in TrainingStatus}
        assert {"completed", "failed", "cancelled"}.issubset(names)


class TestCheckpointError:
    """Tests for CheckpointError exception."""

    def test_catchable(self):
        """CheckpointError can be raised and caught with pattern match."""
        with pytest.raises(CheckpointError, match="disk full"):
            raise CheckpointError("disk full")

    def test_message_preserved(self):
        """Exception message is accessible via str()."""
        try:
            raise CheckpointError("write failed: permission denied")
        except CheckpointError as e:
            assert "permission denied" in str(e)


class TestPreflightResult:
    """Tests for PreflightResult — protects against OOM at runtime."""

    def test_insufficient_vram(self):
        """can_proceed=False when estimated > available."""
        result = PreflightResult(
            predicted_batch_size=4,
            estimated_vram_bytes=16_000_000_000,
            available_vram_bytes=8_000_000_000,
            can_proceed=False,
        )
        assert result.can_proceed is False
        assert result.estimated_vram_bytes > result.available_vram_bytes

    def test_sufficient_vram(self):
        """can_proceed=True when estimated < available."""
        result = PreflightResult(
            predicted_batch_size=2,
            estimated_vram_bytes=4_000_000_000,
            available_vram_bytes=16_000_000_000,
            can_proceed=True,
        )
        assert result.can_proceed is True
