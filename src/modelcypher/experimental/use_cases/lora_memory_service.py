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

"""LoRA Memory Service - Orchestration layer for two-tier memory.

This service manages the lifecycle of LoRA memory stores:

1. **Create/Load**: Get or create stores for agents
2. **Status**: Monitor buffer sizes, training progress
3. **Train**: Run LoRA training steps on accumulated data
4. **Merge**: Consolidate LoRA to base weights
5. **Export**: Save LoRA adapters for sharing

The service abstracts the store management and provides a clean API
for the CLI and other callers.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  CLI                                                        │
    │  mc learn lora-status --agent <id>                          │
    │  mc learn lora-train --agent <id> --steps 100               │
    │  mc learn merge-lora --agent <id> --output <path>           │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  LoRAMemoryService (this module)                            │
    │  - Manages store lifecycle                                  │
    │  - Orchestrates training and merge                          │
    │  - Provides status monitoring                               │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  LoRAMemoryStore (domain)                                   │
    │  - Accumulate events                                        │
    │  - Train LoRA weights                                       │
    │  - Merge to base                                            │
    └─────────────────────────────────────────────────────────────┘

Usage:
    service = LoRAMemoryService()

    # Get or create store for an agent
    store = service.get_or_create_store(
        agent_id="agent-001",
        base_model_path="/path/to/model",
    )

    # Check status
    status = service.status("agent-001")
    print(f"Buffer size: {status.buffer_size}")

    # Train LoRA from accumulated events
    train_result = service.train("agent-001", max_steps=100)
    print(f"Final loss: {train_result.final_loss}")

    # Merge to base weights
    merge_result = service.merge_to_base("agent-001", model, tracker)
    print(f"Preserved: {merge_result.preserved_fraction:.1%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any


from modelcypher.experimental.continual.lora_memory_store import (
    LORA_MEMORY_BASE_DIR,
    LoRAMemoryStore,
    MergeResult,
    TrainStepResult,
    get_or_create_store,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.null_space_tracker import NullSpaceTracker
    from modelcypher.ports.backend import Backend

logger = logging.getLogger("modelcypher.lora_memory_service")


class LoRAMemoryStatus(str, Enum):
    """Status of a LoRA memory store."""

    EMPTY = "empty"  # No events accumulated
    ACCUMULATING = "accumulating"  # Events in buffer, no LoRA trained yet
    TRAINED = "trained"  # LoRA weights trained, ready to merge
    MERGED = "merged"  # Recently merged, buffer cleared


@dataclass
class StoreStatus:
    """Detailed status of a LoRA memory store.

    Attributes
    ----------
    agent_id : str
        Agent identifier.
    status : LoRAMemoryStatus
        Current store status.
    buffer_size : int
        Number of events in accumulation buffer.
    lora_weights_count : int
        Number of trained LoRA weight pairs.
    event_count : int
        Total events ever accumulated.
    train_steps : int
        Total training steps performed.
    merge_count : int
        Number of merges performed.
    learned_regions : int
        Number of unique regions learned.
    last_updated : str
        ISO timestamp of last update.
    base_model_path : str
        Path to the base model.
    rank : int
        LoRA rank.
    alpha : float
        LoRA alpha.
    """

    agent_id: str
    status: LoRAMemoryStatus
    buffer_size: int
    lora_weights_count: int
    event_count: int
    train_steps: int
    merge_count: int
    learned_regions: int
    last_updated: str
    base_model_path: str
    rank: int
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "buffer_size": self.buffer_size,
            "lora_weights_count": self.lora_weights_count,
            "event_count": self.event_count,
            "train_steps": self.train_steps,
            "merge_count": self.merge_count,
            "learned_regions": self.learned_regions,
            "last_updated": self.last_updated,
            "base_model_path": self.base_model_path,
            "rank": self.rank,
            "alpha": self.alpha,
        }


@dataclass
class TrainResult:
    """Result of a training session.

    Attributes
    ----------
    steps_completed : int
        Number of training steps completed.
    final_loss : float
        Loss at the end of training.
    initial_loss : float
        Loss at the start of training.
    samples_processed : int
        Total samples processed across all steps.
    converged : bool
        Whether training converged (loss below threshold).
    step_results : list[TrainStepResult]
        Per-step results.
    """

    steps_completed: int
    final_loss: float
    initial_loss: float
    samples_processed: int
    converged: bool
    step_results: list[TrainStepResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "steps_completed": self.steps_completed,
            "final_loss": self.final_loss,
            "initial_loss": self.initial_loss,
            "samples_processed": self.samples_processed,
            "converged": self.converged,
            "step_results": [r.to_dict() for r in self.step_results],
        }


@dataclass
class ExportResult:
    """Result of exporting LoRA adapters.

    Attributes
    ----------
    success : bool
        Whether export succeeded.
    output_path : str
        Path to exported files.
    parameter_count : int
        Number of parameters exported.
    file_size_bytes : int
        Size of exported files.
    error : str | None
        Error message if export failed.
    """

    success: bool
    output_path: str
    parameter_count: int
    file_size_bytes: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "success": self.success,
            "output_path": self.output_path,
            "parameter_count": self.parameter_count,
            "file_size_bytes": self.file_size_bytes,
            "error": self.error,
        }


class LoRAMemoryService:
    """Service layer for LoRA memory management.

    Orchestrates the lifecycle of LoRA memory stores:
    - Create/load stores
    - Monitor status
    - Run training
    - Merge to base
    - Export adapters

    Parameters
    ----------
    base_dir : Path, optional
        Base directory for stores. Defaults to ~/.modelcypher/lora_memory/
    backend : Backend, optional
        Compute backend.
    """

    def __init__(
        self,
        backend: "Backend",
        base_dir: Path | None = None,
    ) -> None:
        self._base_dir = base_dir or LORA_MEMORY_BASE_DIR
        self._backend = backend
        self._stores: dict[str, LoRAMemoryStore] = {}

    def get_or_create_store(
        self,
        agent_id: str,
        base_model_path: str | Path,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> LoRAMemoryStore:
        """Get or create a LoRA memory store for an agent.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier.
        base_model_path : str | Path
            Path to the base model.
        rank : int
            LoRA rank.
        alpha : float
            LoRA alpha.

        Returns
        -------
        LoRAMemoryStore
            The store instance.
        """
        if agent_id in self._stores:
            return self._stores[agent_id]

        store = get_or_create_store(
            agent_id=agent_id,
            base_model_path=base_model_path,
            rank=rank,
            alpha=alpha,
        )
        self._stores[agent_id] = store

        logger.info(
            "Created/loaded LoRA memory store: agent=%s, buffer=%d",
            agent_id,
            store.buffer_size,
        )

        return store

    def status(self, agent_id: str) -> StoreStatus | None:
        """Get status of a LoRA memory store.

        Parameters
        ----------
        agent_id : str
            Agent identifier.

        Returns
        -------
        StoreStatus | None
            Status if store exists, None otherwise.
        """
        store = self._stores.get(agent_id)
        if store is None:
            # Try to load from disk
            store_dir = self._base_dir / agent_id
            if not store_dir.exists():
                return None
            # Can't load without base_model_path - return minimal status
            return None

        metadata = store.metadata
        stats = store.get_stats()

        # Determine status
        if stats["buffer_size"] == 0 and stats["lora_weights_count"] == 0:
            status = LoRAMemoryStatus.EMPTY
        elif stats["lora_weights_count"] == 0:
            status = LoRAMemoryStatus.ACCUMULATING
        elif metadata.merge_count > 0 and stats["buffer_size"] == 0:
            status = LoRAMemoryStatus.MERGED
        else:
            status = LoRAMemoryStatus.TRAINED

        return StoreStatus(
            agent_id=agent_id,
            status=status,
            buffer_size=stats["buffer_size"],
            lora_weights_count=stats["lora_weights_count"],
            event_count=stats["event_count"],
            train_steps=stats["train_steps"],
            merge_count=stats["merge_count"],
            learned_regions=stats["learned_regions"],
            last_updated=metadata.updated_at,
            base_model_path=metadata.base_model_path,
            rank=metadata.rank,
            alpha=metadata.alpha,
        )

    def train(
        self,
        agent_id: str,
        max_steps: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        convergence_threshold: float = 0.01,
    ) -> TrainResult:
        """Run LoRA training on accumulated events.

        Parameters
        ----------
        agent_id : str
            Agent identifier.
        max_steps : int
            Maximum training steps.
        batch_size : int
            Batch size per step.
        learning_rate : float
            Learning rate.
        convergence_threshold : float
            Loss threshold for early stopping.

        Returns
        -------
        TrainResult
            Training statistics.
        """
        store = self._stores.get(agent_id)
        if store is None:
            return TrainResult(
                steps_completed=0,
                final_loss=0.0,
                initial_loss=0.0,
                samples_processed=0,
                converged=False,
            )

        step_results: list[TrainStepResult] = []
        initial_loss = 0.0
        total_samples = 0

        for step in range(max_steps):
            result = store.train_step(
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            step_results.append(result)
            total_samples += result.samples_used

            if step == 0:
                initial_loss = result.loss

            # Check convergence
            if result.loss < convergence_threshold:
                logger.info(
                    "Training converged at step %d with loss %.4f",
                    step + 1,
                    result.loss,
                )
                break

            # No more samples
            if result.samples_used == 0:
                break

        final_loss = step_results[-1].loss if step_results else 0.0
        converged = final_loss < convergence_threshold

        # Save store state
        store.save()

        return TrainResult(
            steps_completed=len(step_results),
            final_loss=final_loss,
            initial_loss=initial_loss,
            samples_processed=total_samples,
            converged=converged,
            step_results=step_results,
        )

    def merge_to_base(
        self,
        agent_id: str,
        model: Any,
        null_space_tracker: "NullSpaceTracker | None" = None,
        save_merged: bool = False,
        output_path: str | Path | None = None,
    ) -> MergeResult:
        """Merge LoRA weights into base model.

        Parameters
        ----------
        agent_id : str
            Agent identifier.
        model : Any
            Model to merge into.
        null_space_tracker : NullSpaceTracker, optional
            Tracker for null-space projection.
        save_merged : bool
            Whether to save the merged model.
        output_path : str | Path, optional
            Output path for merged model.

        Returns
        -------
        MergeResult
            Merge statistics.
        """
        store = self._stores.get(agent_id)
        if store is None:
            return MergeResult(
                success=False,
                layers_merged=0,
                preserved_fraction=0.0,
                timestamp=datetime.now().isoformat(),
                error="Store not found",
            )

        result = store.merge_to_base(model, null_space_tracker)

        if result.success and save_merged and output_path:
            self._save_merged_model(model, output_path)

        return result

    def reset_lora(self, agent_id: str) -> bool:
        """Reset LoRA weights and buffer after merge.

        Parameters
        ----------
        agent_id : str
            Agent identifier.

        Returns
        -------
        bool
            True if reset succeeded.
        """
        store = self._stores.get(agent_id)
        if store is None:
            return False

        store.reset_lora()
        return True

    def export_lora(
        self,
        agent_id: str,
        output_path: str | Path,
    ) -> ExportResult:
        """Export LoRA adapters to files.

        Parameters
        ----------
        agent_id : str
            Agent identifier.
        output_path : str | Path
            Output directory.

        Returns
        -------
        ExportResult
            Export statistics.
        """
        store = self._stores.get(agent_id)
        if store is None:
            return ExportResult(
                success=False,
                output_path="",
                parameter_count=0,
                file_size_bytes=0,
                error="Store not found",
            )

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save store state (includes LoRA weights)
            store.save()

            # Copy to output location
            import shutil

            store_dir = store._store_dir

            # Copy relevant files
            files_copied = []
            total_size = 0

            for filename in ["metadata.json", "lora_weights.safetensors"]:
                src = store_dir / filename
                if src.exists():
                    dst = output_path / filename
                    shutil.copy(src, dst)
                    files_copied.append(dst)
                    total_size += dst.stat().st_size

            # Count parameters
            param_count = 0
            if store._lora_weights:
                for lora_a, lora_b in store._lora_weights.values():
                    param_count += lora_a.size + lora_b.size

            return ExportResult(
                success=True,
                output_path=str(output_path),
                parameter_count=param_count,
                file_size_bytes=total_size,
            )

        except Exception as e:
            logger.error("Export failed: %s", e)
            return ExportResult(
                success=False,
                output_path=str(output_path),
                parameter_count=0,
                file_size_bytes=0,
                error=str(e),
            )

    def _save_merged_model(self, model: Any, output_path: str | Path) -> None:
        """Save merged model to disk."""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save weights
            weights = dict(model.parameters())
            self._backend.save_safetensors(str(output_path / "model.safetensors"), weights)

            logger.info("Saved merged model to %s", output_path)

        except Exception as e:
            logger.error("Failed to save merged model: %s", e)

    def list_agents(self) -> list[str]:
        """List all agents with stores.

        Returns
        -------
        list[str]
            Agent IDs.
        """
        agents = list(self._stores.keys())

        # Also check disk
        if self._base_dir.exists():
            for agent_dir in self._base_dir.iterdir():
                if agent_dir.is_dir() and agent_dir.name not in agents:
                    agents.append(agent_dir.name)

        return sorted(agents)

    def delete_store(self, agent_id: str) -> bool:
        """Delete a LoRA memory store.

        Parameters
        ----------
        agent_id : str
            Agent identifier.

        Returns
        -------
        bool
            True if deletion succeeded.
        """
        # Remove from memory
        if agent_id in self._stores:
            del self._stores[agent_id]

        # Remove from disk
        store_dir = self._base_dir / agent_id
        if store_dir.exists():
            import shutil

            shutil.rmtree(store_dir)
            logger.info("Deleted store: %s", agent_id)
            return True

        return False


def create_lora_memory_service(
    base_dir: Path | None = None,
) -> LoRAMemoryService:
    """Create a LoRA memory service.

    Parameters
    ----------
    base_dir : Path, optional
        Base directory for stores.

    Returns
    -------
    LoRAMemoryService
        Configured service.
    """
    return LoRAMemoryService(base_dir=base_dir)


__all__ = [
    "LoRAMemoryStatus",
    "StoreStatus",
    "TrainResult",
    "ExportResult",
    "LoRAMemoryService",
    "create_lora_memory_service",
]
