"""Edit applicator protocol for Baranov replication.

EXPERIMENTAL: Not validated for production use.

Defines the ``EditApplicator`` protocol for applying fact edits to model
weights.  No concrete implementation is provided in this patchset --
the Woodbury-equivalent MEMIT path is deferred to patchset 2.

Integration tests mock this interface.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from modelcypher.experimental.baranov.models import EditState, FactTriple


@runtime_checkable
class EditApplicator(Protocol):
    """Protocol for applying fact edits to model weights.

    Concrete implementations (Woodbury-equivalent MEMIT path, naive
    direct-injection, etc.) are deferred to future patchsets.  This
    protocol defines the contract.
    """

    def apply_edit(
        self,
        facts: list[FactTriple],
        layer_ids: list[int],
        model: Any,
    ) -> EditState:
        """Apply a batch of facts as a weight edit.

        Returns an ``EditState`` with status ``applied`` on success
        or ``failed`` on error.
        """
        ...

    def rollback_edit(
        self,
        edit_state: EditState,
        model: Any,
    ) -> EditState:
        """Rollback a previously applied edit.

        Returns an ``EditState`` with status ``rolled_back``.
        """
        ...


__all__ = ["EditApplicator"]
