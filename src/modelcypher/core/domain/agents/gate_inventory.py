# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Gate Inventory - Computational gate concepts for path detection.

Provides ComputationalGate dataclass satisfying ComputationalGateProtocol
for use in gate-based path detection and code pattern analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to computational.json (data/probes relative to package root)
_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "probes"


@dataclass(frozen=True)
class ComputationalGate:
    """
    Computational gate concept for path detection.

    Satisfies ComputationalGateProtocol with:
    - id: Unique identifier (e.g., "computational_gate:1")
    - name: Gate name (e.g., "LITERAL")
    - description: Brief description
    - examples: Example code snippets
    - polyglot_examples: Examples in multiple languages
    """

    id: str
    name: str
    description: str
    examples: tuple[str, ...]
    polyglot_examples: tuple[str, ...]


def _load_gates_from_json() -> list[ComputationalGate]:
    """Load computational gates from JSON data file."""
    filepath = _DATA_DIR / "computational.json"
    if not filepath.exists():
        logger.warning("computational.json not found at %s", filepath)
        return []

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        gates = []
        for probe in data.get("probes", []):
            probe_id = probe.get("id", "")
            if not probe_id.startswith("computational_gate:"):
                continue

            support_texts = tuple(probe.get("support_texts", []))
            gates.append(
                ComputationalGate(
                    id=probe_id,
                    name=probe.get("name", ""),
                    description=probe.get("description", ""),
                    examples=support_texts,
                    polyglot_examples=support_texts,  # Use same texts for polyglot
                )
            )

        logger.debug("Loaded %d computational gates from %s", len(gates), filepath)
        return gates
    except Exception as e:
        logger.warning("Failed to load computational gates: %s", e)
        return []


class ComputationalGateInventory:
    """Inventory of computational gates for path detection."""

    _cached: list[ComputationalGate] | None = None

    @classmethod
    def all_gates(cls) -> list[ComputationalGate]:
        """Get all computational gates."""
        if cls._cached is None:
            cls._cached = _load_gates_from_json()
        return list(cls._cached)

    @classmethod
    def gate_count(cls) -> int:
        """Get count of gates."""
        return len(cls.all_gates())

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the gate cache."""
        cls._cached = None
