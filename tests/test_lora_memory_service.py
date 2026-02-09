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

"""Tests for LoRA memory service training orchestration."""

from __future__ import annotations

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.lora_memory_store import TrainStepResult
from modelcypher.core.use_cases.lora_memory_service import LoRAMemoryService


class _DummyStore:
    def __init__(self, buffer_size: int = 16) -> None:
        self.buffer_size = buffer_size
        self.last_batch_size = 0
        self.last_learning_rate = 0.0
        self.saved = False

    def derive_critical_batch_size(self):
        return 4, {
            "critical_batch_size": 4,
            "max_layer_critical": 3.2,
            "min_layer_snr": 0.31,
            "layer_count": 1,
            "layers": {"layer_0.q_proj.weight": {"critical_batch": 3.2}},
        }

    def derive_learning_rate(self) -> float:
        return 0.01

    def sqrt_eps(self) -> float:
        return 1e-6

    def train_step(self, batch_size: int, learning_rate: float) -> TrainStepResult:
        self.last_batch_size = batch_size
        self.last_learning_rate = learning_rate
        return TrainStepResult(loss=1.0, samples_used=batch_size, gradient_norm=0.1)

    def compute_spectral_budget_ratios(self) -> list[float]:
        return []

    def save(self) -> None:
        self.saved = True


def test_train_uses_critical_batch_size_when_not_provided(tmp_path):
    service = LoRAMemoryService(backend=get_default_backend(), base_dir=tmp_path)
    dummy = _DummyStore()
    service._stores["agent"] = dummy

    result = service.train("agent", max_steps=1)

    assert dummy.last_batch_size == 4
    assert result.resolved_batch_size == 4
    assert result.resolved_critical_batch_size == 4
    assert result.critical_batch_measurements["layer_count"] == 1
    assert dummy.saved is True


def test_train_preserves_explicit_batch_size_override(tmp_path):
    service = LoRAMemoryService(backend=get_default_backend(), base_dir=tmp_path)
    dummy = _DummyStore()
    service._stores["agent"] = dummy

    result = service.train("agent", max_steps=1, batch_size=6)

    assert dummy.last_batch_size == 6
    assert result.resolved_batch_size == 6
    # Still reported for observability even when caller overrides batch size.
    assert result.resolved_critical_batch_size == 4
