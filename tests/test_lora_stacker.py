# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Unit tests for LoRA stacker module.

Tests cumulative geometry tracking, merge decisions, and state persistence.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from modelcypher.core.use_cases.self_improve.lora_stacker import (
    AdapterInfo,
    LoRAStacker,
    MergeResult,
    StackedLoRAState,
    StackerPolicy,
    StackResult,
)


@pytest.fixture
def test_policy() -> StackerPolicy:
    """Create a test policy with explicit thresholds."""
    return StackerPolicy(
        barrier_merge_threshold=0.03,
        cka_drift_threshold=0.1,
        max_adapters=5,
        convergence_ratio_threshold=1.0,
        convergence_barrier_multiplier=0.5,
    )


class TestAdapterInfo:
    """Test AdapterInfo dataclass."""

    def test_to_dict_roundtrip(self, tmp_path: Path) -> None:
        """Test serialization roundtrip."""
        adapter_path = tmp_path / "adapter1"
        adapter_path.mkdir()

        info = AdapterInfo(
            path=adapter_path,
            added_at="2026-02-02T19:00:00",
            barrier_contribution=0.008,
            cka_from_base=0.95,
            difficulty_level=1,
            training_samples=100,
            target_modules=["q_proj", "v_proj"],
        )

        data = info.to_dict()
        restored = AdapterInfo.from_dict(data)

        assert restored.path == adapter_path
        assert restored.barrier_contribution == 0.008
        assert restored.cka_from_base == 0.95
        assert restored.target_modules == ["q_proj", "v_proj"]


class TestStackedLoRAState:
    """Test StackedLoRAState."""

    def test_empty_state_no_merge(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Empty state should not recommend merge."""
        state = StackedLoRAState(
            base_model_path=tmp_path / "model",
            policy=test_policy,
        )

        assert state.n_adapters == 0
        assert not state.should_merge
        assert state.merge_reason == "none"

    def test_barrier_triggers_merge(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Cumulative barrier exceeding threshold triggers merge."""
        state = StackedLoRAState(
            base_model_path=tmp_path / "model",
            policy=test_policy,
            cumulative_barrier=test_policy.barrier_merge_threshold + 0.01,
        )

        assert state.should_merge
        assert "barrier_exceeded" in state.merge_reason

    def test_cka_drift_triggers_merge(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """CKA drift exceeding threshold triggers merge."""
        state = StackedLoRAState(
            base_model_path=tmp_path / "model",
            policy=test_policy,
            cumulative_cka_drift=test_policy.cka_drift_threshold + 0.01,
        )

        assert state.should_merge
        assert "cka_drift_exceeded" in state.merge_reason

    def test_adapter_count_triggers_merge(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Too many adapters triggers merge."""
        adapters = [
            AdapterInfo(
                path=tmp_path / f"adapter{i}",
                added_at=f"2026-02-02T19:0{i}:00",
                barrier_contribution=0.001,
                cka_from_base=0.99,
                difficulty_level=i,
            )
            for i in range(test_policy.max_adapters)
        ]

        state = StackedLoRAState(
            base_model_path=tmp_path / "model",
            policy=test_policy,
            adapters=adapters,
        )

        assert state.should_merge
        assert "adapter_count" in state.merge_reason

    def test_save_load_roundtrip(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Test state persistence."""
        state = StackedLoRAState(
            base_model_path=tmp_path / "model",
            policy=test_policy,
            cumulative_barrier=0.02,
            current_difficulty=3,
        )

        state_file = tmp_path / "state.json"
        state.save(state_file)

        loaded = StackedLoRAState.load(state_file)

        assert loaded.base_model_path == tmp_path / "model"
        assert loaded.cumulative_barrier == 0.02
        assert loaded.current_difficulty == 3
        assert loaded.policy.barrier_merge_threshold == test_policy.barrier_merge_threshold


class TestLoRAStacker:
    """Test LoRAStacker."""

    def test_new_stacker_empty(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """New stacker starts empty."""
        stacker = LoRAStacker(tmp_path / "model", policy=test_policy)

        assert stacker.state.n_adapters == 0
        assert stacker.state.cumulative_barrier == 0.0

    def test_add_adapter_updates_state(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Adding adapter updates cumulative metrics."""
        base_path = tmp_path / "model"
        base_path.mkdir()
        adapter_path = tmp_path / "adapter1"
        adapter_path.mkdir()

        stacker = LoRAStacker(base_path, policy=test_policy)
        result = stacker.add_adapter(
            adapter_path=adapter_path,
            barrier=0.008,
            cka_from_base=0.95,
            difficulty_level=1,
        )

        assert result.success
        assert result.cumulative_barrier == pytest.approx(0.008)
        assert result.cumulative_cka_drift == pytest.approx(0.05)  # 1 - 0.95
        assert stacker.state.n_adapters == 1

    def test_add_nonexistent_adapter_fails(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Adding nonexistent adapter fails gracefully."""
        stacker = LoRAStacker(tmp_path / "model", policy=test_policy)
        result = stacker.add_adapter(
            adapter_path=tmp_path / "nonexistent",
            barrier=0.008,
            cka_from_base=0.95,
            difficulty_level=1,
        )

        assert not result.success
        assert "does not exist" in result.message

    def test_barriers_accumulate(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Barrier contributions accumulate additively."""
        base_path = tmp_path / "model"
        base_path.mkdir()

        stacker = LoRAStacker(base_path, policy=test_policy)

        for i in range(3):
            adapter_path = tmp_path / f"adapter{i}"
            adapter_path.mkdir()
            stacker.add_adapter(
                adapter_path=adapter_path,
                barrier=0.01,
                cka_from_base=0.98,
                difficulty_level=i,
            )

        assert stacker.state.cumulative_barrier == pytest.approx(0.03)
        assert stacker.state.n_adapters == 3

    def test_cka_drift_uses_max(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """CKA drift uses maximum (worst case)."""
        base_path = tmp_path / "model"
        base_path.mkdir()

        stacker = LoRAStacker(base_path, policy=test_policy)

        # First adapter: small drift
        adapter1 = tmp_path / "adapter1"
        adapter1.mkdir()
        stacker.add_adapter(
            adapter_path=adapter1,
            barrier=0.005,
            cka_from_base=0.98,  # drift = 0.02
            difficulty_level=1,
        )

        # Second adapter: larger drift
        adapter2 = tmp_path / "adapter2"
        adapter2.mkdir()
        stacker.add_adapter(
            adapter_path=adapter2,
            barrier=0.005,
            cka_from_base=0.92,  # drift = 0.08
            difficulty_level=2,
        )

        assert stacker.state.cumulative_cka_drift == pytest.approx(0.08)

    def test_should_merge_triggers_correctly(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Merge recommendation triggers at correct thresholds."""
        base_path = tmp_path / "model"
        base_path.mkdir()

        stacker = LoRAStacker(base_path, policy=test_policy)

        # Add adapters until barrier threshold exceeded
        for i in range(4):
            adapter_path = tmp_path / f"adapter{i}"
            adapter_path.mkdir()
            result = stacker.add_adapter(
                adapter_path=adapter_path,
                barrier=0.01,
                cka_from_base=0.99,
                difficulty_level=i,
            )

        # 4 * 0.01 = 0.04 > barrier_merge_threshold (0.03)
        assert result.should_merge
        assert "barrier_exceeded" in result.merge_reason

    def test_get_adapter_paths(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Get ordered list of adapter paths."""
        base_path = tmp_path / "model"
        base_path.mkdir()

        stacker = LoRAStacker(base_path, policy=test_policy)
        expected_paths = []

        for i in range(3):
            adapter_path = tmp_path / f"adapter{i}"
            adapter_path.mkdir()
            expected_paths.append(adapter_path)
            stacker.add_adapter(
                adapter_path=adapter_path,
                barrier=0.005,
                cka_from_base=0.98,
                difficulty_level=i,
            )

        assert stacker.get_adapter_paths() == expected_paths

    def test_save_and_resume_state(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Test state persistence and resumption."""
        base_path = tmp_path / "model"
        base_path.mkdir()
        state_file = tmp_path / "state.json"

        # Create stacker and add adapter
        stacker1 = LoRAStacker(base_path, policy=test_policy)
        adapter_path = tmp_path / "adapter1"
        adapter_path.mkdir()
        stacker1.add_adapter(
            adapter_path=adapter_path,
            barrier=0.012,
            cka_from_base=0.94,
            difficulty_level=2,
        )
        stacker1.save_state(state_file)

        # Resume from saved state
        stacker2 = LoRAStacker(base_path, policy=test_policy, state_path=state_file)

        assert stacker2.state.n_adapters == 1
        assert stacker2.state.cumulative_barrier == pytest.approx(0.012)
        assert stacker2.state.current_difficulty == 2

    def test_get_status(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Test status reporting."""
        base_path = tmp_path / "model"
        base_path.mkdir()

        stacker = LoRAStacker(base_path, policy=test_policy)
        status = stacker.get_status()

        assert status["n_adapters"] == 0
        assert status["should_merge"] is False
        assert "policy" in status
        assert status["policy"]["barrier_merge"] == test_policy.barrier_merge_threshold


class TestMergeEmpty:
    """Test merge with empty stack."""

    def test_merge_empty_fails(self, tmp_path: Path, test_policy: StackerPolicy) -> None:
        """Merging empty stack fails gracefully."""
        stacker = LoRAStacker(tmp_path / "model", policy=test_policy)
        result = stacker.merge_stack(tmp_path / "merged")

        assert not result.success
        assert result.adapters_merged == 0
        assert "No adapters" in result.message
