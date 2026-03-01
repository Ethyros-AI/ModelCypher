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

"""Delegation contract tests for ActivationProviderAdapter.

Verifies that the adapter delegates intermediate, gate, and probe batch
collection to the backend instead of falling back to hidden activations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, sentinel

import pytest

from modelcypher.adapters.activation_provider import ActivationProviderAdapter
from modelcypher.ports.activation_provider import ProbeActivationBatch


HIDDEN_DIM = 960
INTERMEDIATE_DIM = 2560
N_LAYERS = 4


class _ArrayStub:
    """Minimal array stub with shape attribute."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.ndim = len(shape)

    def __repr__(self) -> str:
        return f"ArrayStub(shape={self.shape})"


@dataclass
class _CallLog:
    """Tracks which backend methods were called."""

    calls: list[str] = field(default_factory=list)


def _make_hidden(n_layers: int = N_LAYERS) -> dict[int, _ArrayStub]:
    return {i: _ArrayStub((HIDDEN_DIM,)) for i in range(n_layers)}


def _make_intermediate(n_layers: int = N_LAYERS) -> dict[int, _ArrayStub]:
    return {i: _ArrayStub((INTERMEDIATE_DIM,)) for i in range(n_layers)}


def _make_gate(n_layers: int = N_LAYERS) -> dict[int, _ArrayStub]:
    return {i: _ArrayStub((INTERMEDIATE_DIM,)) for i in range(n_layers)}


def _make_embedding() -> _ArrayStub:
    return _ArrayStub((HIDDEN_DIM,))


def _make_mock_backend(call_log: _CallLog) -> MagicMock:
    """Create a mock backend that returns correctly-shaped stubs."""
    backend = MagicMock()

    def log_and_return(method_name, return_value):
        def side_effect(*args, **kwargs):
            call_log.calls.append(method_name)
            return return_value

        return side_effect

    # Single-text methods
    backend.collect_intermediate_activations.side_effect = log_and_return(
        "collect_intermediate_activations", _make_intermediate()
    )
    backend.collect_hidden_activations.side_effect = log_and_return(
        "collect_hidden_activations", _make_hidden()
    )
    backend.collect_embedding_activations.side_effect = log_and_return(
        "collect_embedding_activations", _make_embedding()
    )

    # Batch methods
    backend.collect_probe_activations_batch.side_effect = log_and_return(
        "collect_probe_activations_batch",
        ProbeActivationBatch(
            hidden=[_make_hidden()],
            intermediate=[_make_intermediate()],
            gate=[_make_gate()],
            embedding=[_make_embedding()],
        ),
    )
    backend.collect_intermediate_activations_batch.side_effect = log_and_return(
        "collect_intermediate_activations_batch", [_make_intermediate()]
    )
    backend.collect_gate_activations_batch.side_effect = log_and_return(
        "collect_gate_activations_batch", [_make_gate()]
    )
    backend.collect_routing_decisions.side_effect = log_and_return(
        "collect_routing_decisions", {0: _ArrayStub((4, 2))}
    )

    # eval is a no-op
    backend.eval.return_value = None
    backend.array.return_value = _ArrayStub((1, 10))

    return backend


class TestDelegation:
    """Verify adapter delegates to backend instead of falling back to hidden."""

    def setup_method(self):
        self.call_log = _CallLog()
        self.backend = _make_mock_backend(self.call_log)
        self.adapter = ActivationProviderAdapter(self.backend)

    def test_collect_intermediate_delegates(self):
        result = self.adapter.collect_intermediate_activations(
            sentinel.model, sentinel.tokenizer, "test text"
        )
        assert "collect_intermediate_activations" in self.call_log.calls
        assert "collect_hidden_activations" not in self.call_log.calls

    def test_collect_probe_batch_delegates(self):
        result = self.adapter.collect_probe_activations_batch(
            sentinel.model, sentinel.tokenizer, ["test text"]
        )
        assert "collect_probe_activations_batch" in self.call_log.calls
        assert "collect_hidden_activations" not in self.call_log.calls

    def test_collect_intermediate_batch_delegates(self):
        result = self.adapter.collect_intermediate_activations_batch(
            sentinel.model, sentinel.tokenizer, ["test text"]
        )
        assert "collect_intermediate_activations_batch" in self.call_log.calls
        assert "collect_hidden_activations" not in self.call_log.calls

    def test_collect_gate_batch_delegates(self):
        result = self.adapter.collect_gate_activations_batch(
            sentinel.model, sentinel.tokenizer, ["test text"]
        )
        assert "collect_gate_activations_batch" in self.call_log.calls
        assert "collect_hidden_activations" not in self.call_log.calls

    def test_collect_routing_decisions_delegates(self):
        result = self.adapter.collect_routing_decisions(
            sentinel.model, sentinel.tokenizer, ["test text"]
        )
        assert "collect_routing_decisions" in self.call_log.calls
        assert "collect_hidden_activations" not in self.call_log.calls


class TestShapeContract:
    """Verify adapter returns correct dimensions (intermediate_dim != hidden_dim)."""

    def setup_method(self):
        self.call_log = _CallLog()
        self.backend = _make_mock_backend(self.call_log)
        self.adapter = ActivationProviderAdapter(self.backend)

    def test_intermediate_returns_intermediate_dim(self):
        result = self.adapter.collect_intermediate_activations(
            sentinel.model, sentinel.tokenizer, "test"
        )
        for layer_idx, act in result.items():
            assert act.shape[-1] == INTERMEDIATE_DIM, (
                f"Layer {layer_idx}: expected intermediate_dim={INTERMEDIATE_DIM}, "
                f"got {act.shape[-1]}"
            )

    def test_intermediate_dim_differs_from_hidden(self):
        result = self.adapter.collect_intermediate_activations(
            sentinel.model, sentinel.tokenizer, "test"
        )
        for layer_idx, act in result.items():
            assert act.shape[-1] != HIDDEN_DIM, (
                f"Layer {layer_idx}: intermediate should NOT have hidden_dim={HIDDEN_DIM}"
            )


class TestProbeBatchContract:
    """Verify probe batch returns correct shapes for all activation types."""

    def setup_method(self):
        self.call_log = _CallLog()
        self.backend = _make_mock_backend(self.call_log)
        self.adapter = ActivationProviderAdapter(self.backend)

    def test_probe_batch_intermediate_has_intermediate_dim(self):
        batch = self.adapter.collect_probe_activations_batch(
            sentinel.model, sentinel.tokenizer, ["test"]
        )
        assert isinstance(batch, ProbeActivationBatch)
        for text_acts in batch.intermediate:
            for layer_idx, act in text_acts.items():
                assert act.shape[-1] == INTERMEDIATE_DIM, (
                    f"Layer {layer_idx}: probe batch intermediate should have "
                    f"dim={INTERMEDIATE_DIM}, got {act.shape[-1]}"
                )

    def test_probe_batch_gate_has_intermediate_dim(self):
        batch = self.adapter.collect_probe_activations_batch(
            sentinel.model, sentinel.tokenizer, ["test"]
        )
        for text_acts in batch.gate:
            for layer_idx, act in text_acts.items():
                assert act.shape[-1] == INTERMEDIATE_DIM, (
                    f"Layer {layer_idx}: probe batch gate should have "
                    f"dim={INTERMEDIATE_DIM}, got {act.shape[-1]}"
                )

    def test_probe_batch_hidden_has_hidden_dim(self):
        batch = self.adapter.collect_probe_activations_batch(
            sentinel.model, sentinel.tokenizer, ["test"]
        )
        for text_acts in batch.hidden:
            for layer_idx, act in text_acts.items():
                assert act.shape[-1] == HIDDEN_DIM, (
                    f"Layer {layer_idx}: probe batch hidden should have "
                    f"dim={HIDDEN_DIM}, got {act.shape[-1]}"
                )
