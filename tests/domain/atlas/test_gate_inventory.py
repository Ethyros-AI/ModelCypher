# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import modelcypher.core.domain.atlas.gate_inventory as gate_inventory

# ---------------------------------------------------------------------------
# ComputationalGate frozen dataclass
# ---------------------------------------------------------------------------


class TestComputationalGate:
    """Tests for the ComputationalGate frozen dataclass."""

    def test_instantiation(self) -> None:
        gate = gate_inventory.ComputationalGate(
            id="computational_gate:1",
            name="LITERAL",
            description="Literal computation",
            examples=("x = 1",),
            polyglot_examples=("x = 1", "let x = 1"),
        )
        assert gate.id == "computational_gate:1"
        assert gate.name == "LITERAL"
        assert gate.description == "Literal computation"
        assert gate.examples == ("x = 1",)
        assert gate.polyglot_examples == ("x = 1", "let x = 1")

    def test_field_access(self) -> None:
        gate = gate_inventory.ComputationalGate(
            id="computational_gate:42",
            name="BRANCH",
            description="Conditional branching",
            examples=("if x > 0:", "switch (x)"),
            polyglot_examples=("if x > 0:", "if (x > 0)"),
        )
        assert gate.id == "computational_gate:42"
        assert gate.name == "BRANCH"
        assert isinstance(gate.examples, tuple)
        assert isinstance(gate.polyglot_examples, tuple)
        assert len(gate.examples) == 2
        assert len(gate.polyglot_examples) == 2

    def test_frozen_enforcement(self) -> None:
        gate = gate_inventory.ComputationalGate(
            id="computational_gate:1",
            name="LITERAL",
            description="Literal computation",
            examples=("x = 1",),
            polyglot_examples=("x = 1",),
        )
        with pytest.raises(FrozenInstanceError):
            gate.id = "modified"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            gate.name = "CHANGED"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            gate.description = "changed"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            gate.examples = ()  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            gate.polyglot_examples = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ComputationalGateInventory
# ---------------------------------------------------------------------------


class TestComputationalGateInventory:
    """Tests for the ComputationalGateInventory class methods."""

    def setup_method(self) -> None:
        gate_inventory.ComputationalGateInventory.clear_cache()

    def teardown_method(self) -> None:
        gate_inventory.ComputationalGateInventory.clear_cache()

    def test_all_gates_returns_list(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        assert isinstance(gates, list)

    def test_all_gates_non_empty(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        assert len(gates) > 0, "Expected at least one computational gate"

    def test_all_gates_returns_computational_gate_instances(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        for g in gates:
            assert isinstance(g, gate_inventory.ComputationalGate)

    def test_gate_count_positive(self) -> None:
        count = gate_inventory.ComputationalGateInventory.gate_count()
        assert isinstance(count, int)
        assert count > 0

    def test_gate_count_matches_all_gates(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        count = gate_inventory.ComputationalGateInventory.gate_count()
        assert count == len(gates)

    def test_clear_cache(self) -> None:
        # Load gates to populate cache
        gates_first = gate_inventory.ComputationalGateInventory.all_gates()
        # Clear cache
        gate_inventory.ComputationalGateInventory.clear_cache()
        # Reload after clear -- should still return same data
        gates_second = gate_inventory.ComputationalGateInventory.all_gates()
        assert len(gates_first) == len(gates_second)
        # Verify data equality
        for g1, g2 in zip(gates_first, gates_second):
            assert g1.id == g2.id
            assert g1.name == g2.name

    def test_clear_cache_resets_internal_state(self) -> None:
        # Load to populate
        gate_inventory.ComputationalGateInventory.all_gates()
        assert gate_inventory.ComputationalGateInventory._cached is not None
        # Clear
        gate_inventory.ComputationalGateInventory.clear_cache()
        assert gate_inventory.ComputationalGateInventory._cached is None

    def test_all_gates_returns_fresh_list(self) -> None:
        """Verify all_gates returns a copy, not the cached reference."""
        gates_a = gate_inventory.ComputationalGateInventory.all_gates()
        gates_b = gate_inventory.ComputationalGateInventory.all_gates()
        assert gates_a is not gates_b
        assert gates_a == gates_b


# ---------------------------------------------------------------------------
# Gate field validation
# ---------------------------------------------------------------------------


class TestGateFieldValidation:
    """Validate that all loaded gates have proper field values."""

    def setup_method(self) -> None:
        gate_inventory.ComputationalGateInventory.clear_cache()

    def teardown_method(self) -> None:
        gate_inventory.ComputationalGateInventory.clear_cache()

    def test_all_gate_ids_non_empty(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        for g in gates:
            assert g.id, f"Gate has empty id: {g}"
            assert isinstance(g.id, str)

    def test_all_gate_ids_prefixed(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        for g in gates:
            assert g.id.startswith("computational_gate:"), (
                f"Gate id does not start with 'computational_gate:': {g.id}"
            )

    def test_all_gate_names_non_empty(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        for g in gates:
            assert g.name, f"Gate has empty name: {g}"
            assert isinstance(g.name, str)

    def test_all_gate_descriptions_non_empty(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        for g in gates:
            assert g.description, f"Gate has empty description: {g}"
            assert isinstance(g.description, str)

    def test_all_gate_examples_are_tuples_of_str(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        for g in gates:
            assert isinstance(g.examples, tuple), f"examples not a tuple: {g.id}"
            for ex in g.examples:
                assert isinstance(ex, str), f"example not a str in {g.id}: {ex}"

    def test_all_gate_polyglot_examples_are_tuples_of_str(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        for g in gates:
            assert isinstance(g.polyglot_examples, tuple), (
                f"polyglot_examples not a tuple: {g.id}"
            )
            for ex in g.polyglot_examples:
                assert isinstance(ex, str), (
                    f"polyglot_example not a str in {g.id}: {ex}"
                )

    def test_all_gate_ids_unique(self) -> None:
        gates = gate_inventory.ComputationalGateInventory.all_gates()
        ids = [g.id for g in gates]
        assert len(ids) == len(set(ids)), "Duplicate gate ids found"
