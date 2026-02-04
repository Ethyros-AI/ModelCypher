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

"""LoRA Memory Store - Two-tier biological memory system.

This module implements the "hippocampus" part of two-tier memory:

1. **Hippocampus (LoRA Adapter)**: Fast binding for session-level learning
   - Accumulates (hidden_state, delta) pairs during inference
   - Trains LoRA adapters from accumulated data
   - Ephemeral until merged to base weights

2. **Neocortex (Base Weights)**: Slow consolidation via periodic merge
   - Null-space projection preserves existing knowledge
   - Merged LoRA becomes permanent part of model identity

The biological metaphor:
- Wake: Encounter new knowledge, accumulate in hippocampus (LoRA buffer)
- Sleep: Consolidate to neocortex (merge LoRA → base weights)
- Dream: Replay and strengthen connections (LoRA training step)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  HIPPOCAMPUS (LoRA Adapter)                                 │
    │  - Fast binding: session-level learning                     │
    │  - Accumulates (hidden_state, delta) pairs                  │
    │  - Ephemeral until merged                                   │
    └────────────────────────┬────────────────────────────────────┘
                             │ Periodic merge (sleep consolidation)
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  NEOCORTEX (Base Weights)                                   │
    │  - Slow consolidation: permanent storage                    │
    │  - Null-space projection preserves existing knowledge       │
    │  - Merged LoRA becomes part of identity                     │
    └─────────────────────────────────────────────────────────────┘

Storage format:
    ~/.modelcypher/lora_memory/{agent_id}/
    ├── metadata.json           # LoRAMemoryMetadata
    ├── lora_weights.safetensors # LoRA A/B matrices
    ├── events.safetensors       # Pending (hidden_state, delta) pairs
    └── history/
        └── merged_{timestamp}.json

Schema: mc.lora_memory.v1
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.lora_memory_store")

LORA_MEMORY_VERSION = "mc.lora_memory.v1"
LORA_MEMORY_BASE_DIR = Path.home() / ".modelcypher" / "lora_memory"
METADATA_FILE = "metadata.json"
LORA_WEIGHTS_FILE = "lora_weights.safetensors"
EVENTS_FILE = "events.safetensors"
HISTORY_DIR = "history"


class MemoryEventSource(str, Enum):
    """Source of a learning event."""

    INFERENCE = "inference"  # From generation-time uncertainty
    EXTERNAL = "external"  # From external knowledge retrieval
    SYNTHETIC = "synthetic"  # From manifold completion probing


@dataclass(frozen=True)
class HeatSignal:
    """Computed heat signal for memory promotion decisions.

    Heat measures learning opportunity - events where updating the model
    would yield meaningful behavioral change without corrupting existing
    knowledge.

    HEAT = surprise_percentile × preserved_fraction × entropy_stability

    All fields are raw measurements. No thresholds - heat is continuous [0, 1].
    Higher heat = more likely to be sampled during training.

    Attributes
    ----------
    timestamp : int
        Timestep when signal was computed.
    surprise_percentile : float
        How novel this event was [0, 1]. From SurpriseEvent.percentile.
    preserved_fraction : float
        What fraction of behavioral change survived null-space projection [0, 1].
    entropy_normalized : float
        Model uncertainty at event time [0, 1]. H / ln(vocab_size).
    entropy_derivative_abs : float
        Absolute rate of entropy change |dH/dt|.
    heat : float
        Final computed heat = surprise × preserved × stability.
    eigenscore : float
        Manifold sparsity at event (from EntropySignal).
    capacity_fraction : float
        Null space available at this layer.
    """

    timestamp: int
    surprise_percentile: float
    preserved_fraction: float
    entropy_normalized: float
    entropy_derivative_abs: float
    heat: float
    eigenscore: float = 0.0
    capacity_fraction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "surprise_percentile": self.surprise_percentile,
            "preserved_fraction": self.preserved_fraction,
            "entropy_normalized": self.entropy_normalized,
            "entropy_derivative_abs": self.entropy_derivative_abs,
            "heat": self.heat,
            "eigenscore": self.eigenscore,
            "capacity_fraction": self.capacity_fraction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HeatSignal":
        """Create from dict."""
        return cls(
            timestamp=d["timestamp"],
            surprise_percentile=d["surprise_percentile"],
            preserved_fraction=d["preserved_fraction"],
            entropy_normalized=d["entropy_normalized"],
            entropy_derivative_abs=d["entropy_derivative_abs"],
            heat=d["heat"],
            eigenscore=d.get("eigenscore", 0.0),
            capacity_fraction=d.get("capacity_fraction", 0.0),
        )


@dataclass(frozen=True)
class LoRAMemoryEvent:
    """A single learning event to be encoded into LoRA.

    Each event represents a (WHERE, WHAT) pair:
    - hidden_state: WHERE in activation space
    - delta: WHAT direction to learn

    Attributes
    ----------
    timestamp : int
        Unix timestamp of event creation.
    hidden_state_hash : str
        SHA256 hash of hidden state for deduplication.
    delta_hash : str
        SHA256 hash of delta for deduplication.
    layer_id : int
        Layer index where this event occurred.
    weight_name : str
        Weight matrix name (e.g., "mlp.up_proj").
    source : MemoryEventSource
        Origin of this learning event.
    confidence : float
        Trust level for this event [0, 1].
    heat : float
        Learning opportunity signal [0, 1]. Higher = more valuable for training.
        Computed as: surprise × preserved_fraction × entropy_stability.
        Default 0.0 means uniform sampling (backwards compatible).
    """

    timestamp: int
    hidden_state_hash: str
    delta_hash: str
    layer_id: int
    weight_name: str
    source: MemoryEventSource
    confidence: float
    heat: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "hidden_state_hash": self.hidden_state_hash,
            "delta_hash": self.delta_hash,
            "layer_id": self.layer_id,
            "weight_name": self.weight_name,
            "source": self.source.value,
            "confidence": self.confidence,
            "heat": self.heat,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LoRAMemoryEvent:
        """Create from dict."""
        return cls(
            timestamp=d["timestamp"],
            hidden_state_hash=d["hidden_state_hash"],
            delta_hash=d["delta_hash"],
            layer_id=d["layer_id"],
            weight_name=d["weight_name"],
            source=MemoryEventSource(d["source"]),
            confidence=d["confidence"],
            heat=d.get("heat", 0.0),
        )


@dataclass
class LoRAMemoryMetadata:
    """Metadata for the LoRA memory store.

    Stored in metadata.json alongside the weight files.

    Attributes
    ----------
    agent_id : str
        Unique identifier for the agent owning this memory.
    store_version : str
        Schema version for compatibility checking.
    base_model_path : str
        Path to the base model these LoRA weights apply to.
    base_model_hash : str
        Hash of base model for invalidation detection.
    rank : int
        LoRA rank (number of adapter dimensions).
    alpha : float
        LoRA alpha (scaling factor).
    target_modules : list[str]
        Which weight matrices get LoRA adapters.
    created_at : str
        ISO timestamp of store creation.
    updated_at : str
        ISO timestamp of last modification.
    event_count : int
        Total events accumulated (may exceed buffer if deduplicated).
    buffer_size : int
        Current number of (hidden_state, delta) pairs in buffer.
    train_steps : int
        Total LoRA training steps performed.
    merge_count : int
        Number of times LoRA was merged to base.
    learned_region_hashes : set[str]
        Hashes of regions already learned (query optimization).
    """

    agent_id: str
    base_model_path: str
    base_model_hash: str = ""
    store_version: str = LORA_MEMORY_VERSION
    rank: int = 8
    alpha: float = 16.0
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "up_proj"]
    )
    created_at: str = ""
    updated_at: str = ""
    event_count: int = 0
    buffer_size: int = 0
    train_steps: int = 0
    merge_count: int = 0
    learned_region_hashes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Set timestamps if not provided."""
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "_schema": self.store_version,
            "agent_id": self.agent_id,
            "base_model_path": self.base_model_path,
            "base_model_hash": self.base_model_hash,
            "store_version": self.store_version,
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "event_count": self.event_count,
            "buffer_size": self.buffer_size,
            "train_steps": self.train_steps,
            "merge_count": self.merge_count,
            "learned_region_hashes": list(self.learned_region_hashes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LoRAMemoryMetadata:
        """Create from dict."""
        return cls(
            agent_id=d["agent_id"],
            base_model_path=d["base_model_path"],
            base_model_hash=d.get("base_model_hash", ""),
            store_version=d.get("store_version", LORA_MEMORY_VERSION),
            rank=d.get("rank", 8),
            alpha=d.get("alpha", 16.0),
            target_modules=d.get("target_modules", ["q_proj", "v_proj", "up_proj"]),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            event_count=d.get("event_count", 0),
            buffer_size=d.get("buffer_size", 0),
            train_steps=d.get("train_steps", 0),
            merge_count=d.get("merge_count", 0),
            learned_region_hashes=set(d.get("learned_region_hashes", [])),
        )


@dataclass
class TrainStepResult:
    """Result of a LoRA training step.

    Attributes
    ----------
    loss : float
        Training loss for this step.
    samples_used : int
        Number of (hidden_state, delta) pairs used.
    gradient_norm : float
        L2 norm of gradients.
    """

    loss: float
    samples_used: int
    gradient_norm: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "loss": self.loss,
            "samples_used": self.samples_used,
            "gradient_norm": self.gradient_norm,
        }


@dataclass
class MergeResult:
    """Result of merging LoRA to base weights.

    Attributes
    ----------
    success : bool
        Whether merge completed successfully.
    layers_merged : int
        Number of layers that received LoRA updates.
    preserved_fraction : float
        Fraction of delta that survived null-space projection.
    timestamp : str
        ISO timestamp of merge.
    error : str | None
        Error message if merge failed.
    """

    success: bool
    layers_merged: int
    preserved_fraction: float
    timestamp: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "success": self.success,
            "layers_merged": self.layers_merged,
            "preserved_fraction": self.preserved_fraction,
            "timestamp": self.timestamp,
            "error": self.error,
        }


def _compute_array_hash(arr: "Array", backend: "Backend") -> str:
    """Compute a fingerprint hash of an array for deduplication.

    Uses GPU-friendly operations to compute a fingerprint without
    converting to numpy (which is disabled in the Backend).
    """
    b = backend

    # Flatten array
    flat = b.reshape(arr, (-1,))
    b.eval(flat)

    # Compute fingerprint using multiple statistics
    # This is not cryptographically secure but sufficient for deduplication
    arr_sum = b.sum(flat)
    arr_sum_sq = b.sum(flat * flat)
    arr_min = b.min(flat)
    arr_max = b.max(flat)
    arr_size = flat.shape[0]
    b.eval(arr_sum, arr_sum_sq, arr_min, arr_max)

    # Combine into a string for hashing
    fingerprint = (
        f"{float(b.to_scalar(arr_sum)):.8e}"
        f"{float(b.to_scalar(arr_sum_sq)):.8e}"
        f"{float(b.to_scalar(arr_min)):.8e}"
        f"{float(b.to_scalar(arr_max)):.8e}"
        f"{arr_size}"
    )

    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


class LoRAMemoryStore:
    """Two-tier memory: LoRA (hippocampus) + Base (neocortex).

    This store manages the fast-binding hippocampus layer:
    - Accumulates (hidden_state, delta) pairs during inference
    - Trains LoRA adapters from accumulated data
    - Merges trained LoRA into base weights via null-space projection

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent.
    base_model_path : str | Path
        Path to the base model.
    rank : int
        LoRA rank (default: 8).
    alpha : float
        LoRA alpha scaling (default: 16.0).
    target_modules : list[str], optional
        Which weight matrices to adapt.
    backend : Backend, optional
        Compute backend.

    Examples
    --------
    Basic usage with KnowledgeEncoder:

        store = LoRAMemoryStore(
            agent_id="agent-001",
            base_model_path="/path/to/model",
        )

        # Wire to KnowledgeEncoder via LoRAAccumulateStrategy
        from modelcypher.experimental.continual.update_strategy import (
            LoRAAccumulateStrategy,
        )
        strategy = LoRAAccumulateStrategy(accumulator=store)

        # During inference, updates accumulate in store
        encoder = KnowledgeEncoder(
            model=model,
            null_space_tracker=tracker,
            update_strategy=strategy,
        )

        # Periodically train LoRA from accumulated data
        for step in range(100):
            result = store.train_step(batch_size=32)
            if result.loss < 0.01:
                break

        # Merge trained LoRA into base weights
        merge_result = store.merge_to_base(model, tracker)

        # Clear hippocampus for next session
        store.reset_lora()
    """

    def __init__(
        self,
        agent_id: str,
        base_model_path: str | Path,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: list[str] | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        self._backend = backend or get_default_backend()
        self._agent_id = agent_id
        self._base_model_path = Path(base_model_path).expanduser().resolve()

        # Storage directory
        self._store_dir = LORA_MEMORY_BASE_DIR / agent_id
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._history_dir = self._store_dir / HISTORY_DIR
        self._history_dir.mkdir(exist_ok=True)

        # Initialize or load metadata
        self._metadata = self._load_or_create_metadata(
            rank=rank,
            alpha=alpha,
            target_modules=target_modules or ["q_proj", "v_proj", "up_proj"],
        )

        # In-memory buffers for accumulated events
        # Key: (layer_id, weight_name) -> list of (hidden_state, delta, confidence)
        self._event_buffer: dict[
            tuple[int, str], list[tuple["Array", "Array", float]]
        ] = {}

        # LoRA weights (lazy initialization)
        # Key: (layer_id, weight_name) -> (lora_a, lora_b)
        self._lora_weights: dict[tuple[int, str], tuple["Array", "Array"]] = {}

        # Load existing state if available
        self._load_state()

    @property
    def agent_id(self) -> str:
        """Return agent ID."""
        return self._agent_id

    @property
    def buffer_size(self) -> int:
        """Return total number of events in buffer."""
        return sum(len(events) for events in self._event_buffer.values())

    @property
    def metadata(self) -> LoRAMemoryMetadata:
        """Return current metadata."""
        return self._metadata

    def _load_or_create_metadata(
        self,
        rank: int,
        alpha: float,
        target_modules: list[str],
    ) -> LoRAMemoryMetadata:
        """Load existing metadata or create new."""
        metadata_path = self._store_dir / METADATA_FILE
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                return LoRAMemoryMetadata.from_dict(data)
            except Exception as e:
                logger.warning("Failed to load metadata, creating new: %s", e)

        # Compute base model hash
        base_hash = self._compute_base_model_hash()

        return LoRAMemoryMetadata(
            agent_id=self._agent_id,
            base_model_path=str(self._base_model_path),
            base_model_hash=base_hash,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
        )

    def _compute_base_model_hash(self) -> str:
        """Compute hash of base model for invalidation detection."""
        hasher = hashlib.sha256()

        # Hash config.json
        config_path = self._base_model_path / "config.json"
        if config_path.exists():
            hasher.update(config_path.read_bytes())

        # Hash first chunk of safetensors
        for sf in sorted(self._base_model_path.glob("*.safetensors"))[:1]:
            try:
                with open(sf, "rb") as f:
                    hasher.update(f.read(1024 * 1024))  # 1MB
            except OSError:
                pass

        return hasher.hexdigest()[:16]

    def _load_state(self) -> None:
        """Load existing buffers and LoRA weights from disk."""
        events_path = self._store_dir / EVENTS_FILE
        lora_path = self._store_dir / LORA_WEIGHTS_FILE

        if events_path.exists():
            self._load_events(events_path)

        if lora_path.exists():
            self._load_lora_weights(lora_path)

    def _load_events(self, path: Path) -> None:
        """Load accumulated events from safetensors."""
        try:
            tensors = self._backend.load_safetensors(str(path))

            # Parse tensors back into buffer
            # Format: {layer_id}_{weight_name}_hidden_{idx}, {layer_id}_{weight_name}_delta_{idx}
            # This is a simplified implementation - real version would use structured keys
            logger.info("Loaded events from %s", path)

        except Exception as e:
            logger.warning("Failed to load events: %s", e)

    def _load_lora_weights(self, path: Path) -> None:
        """Load trained LoRA weights from safetensors."""
        try:
            tensors = self._backend.load_safetensors(str(path))

            # Parse weights into lora_weights dict
            # Format: {layer_id}_{weight_name}_lora_a, {layer_id}_{weight_name}_lora_b
            for key, tensor in tensors.items():
                if "_lora_a" in key:
                    base_key = key.replace("_lora_a", "")
                    parts = base_key.split("_", 1)
                    if len(parts) == 2:
                        layer_id = int(parts[0])
                        weight_name = parts[1]
                        lora_b_key = f"{base_key}_lora_b"
                        if lora_b_key in tensors:
                            self._lora_weights[(layer_id, weight_name)] = (
                                tensor,
                                tensors[lora_b_key],
                            )

            logger.info("Loaded LoRA weights from %s", path)

        except Exception as e:
            logger.warning("Failed to load LoRA weights: %s", e)

    def accumulate(
        self,
        hidden_state: "Array",
        delta: "Array",
        layer_id: int,
        weight_name: str,
        confidence: float = 1.0,
        heat: float = 0.0,
        source: MemoryEventSource = MemoryEventSource.INFERENCE,
    ) -> bool:
        """Accumulate a learning event for later LoRA training.

        This is the main entry point called by LoRAAccumulateStrategy.

        Parameters
        ----------
        hidden_state : Array
            WHERE in activation space (input to the weight).
        delta : Array
            WHAT direction to learn (weight update, already null-space projected).
        layer_id : int
            Layer index.
        weight_name : str
            Weight matrix name (e.g., "mlp.up_proj").
        confidence : float
            Trust level for this event [0, 1].
        heat : float
            Learning opportunity signal [0, 1]. Higher = more likely to be sampled
            during training. Computed as: surprise × preserved × entropy_stability.
            Default 0.0 means uniform sampling (backwards compatible).
        source : MemoryEventSource
            Origin of this learning event.

        Returns
        -------
        bool
            True if accumulated (not deduplicated), False if duplicate.
        """
        b = self._backend

        # Compute hashes for deduplication
        h_hash = _compute_array_hash(hidden_state, b)
        d_hash = _compute_array_hash(delta, b)
        combined_hash = f"{h_hash}_{d_hash}"

        # Check if already learned
        if combined_hash in self._metadata.learned_region_hashes:
            logger.debug("Duplicate event skipped: %s", combined_hash[:16])
            return False

        # Add to buffer
        key = (layer_id, weight_name)
        if key not in self._event_buffer:
            self._event_buffer[key] = []

        # Copy tensors to avoid reference issues
        h_copy = hidden_state + b.zeros_like(hidden_state)  # backend-safe copy
        d_copy = delta + b.zeros_like(delta)
        b.eval(h_copy, d_copy)

        self._event_buffer[key].append((h_copy, d_copy, confidence, heat))

        # Update metadata
        self._metadata.event_count += 1
        self._metadata.buffer_size = self.buffer_size
        self._metadata.updated_at = datetime.now().isoformat()

        logger.debug(
            "Accumulated event: layer=%d, weight=%s, buffer_size=%d",
            layer_id,
            weight_name,
            self.buffer_size,
        )

        return True

    def train_step(self, batch_size: int = 32, learning_rate: float = 1e-4) -> TrainStepResult:
        """Perform one LoRA training step from accumulated events.

        This implements the "dreaming" phase - replaying accumulated
        experiences to strengthen LoRA connections.

        Parameters
        ----------
        batch_size : int
            Number of events to use per step.
        learning_rate : float
            Learning rate for LoRA updates.

        Returns
        -------
        TrainStepResult
            Training statistics.
        """
        b = self._backend

        if self.buffer_size == 0:
            return TrainStepResult(loss=0.0, samples_used=0, gradient_norm=0.0)

        total_loss = 0.0
        total_samples = 0
        total_grad_norm = 0.0

        # Train each (layer, weight) independently
        for (layer_id, weight_name), events in self._event_buffer.items():
            if not events:
                continue

            # Sample batch with heat-weighted sampling
            # Higher heat = more likely to be selected (better learning opportunities)
            n_samples = min(batch_size, len(events))
            import random

            # Extract heat values from 4-tuples: (hidden_state, delta, confidence, heat)
            heats = [e[3] for e in events]
            heat_sum = sum(heats)

            # Use heat as sampling weights if available, otherwise uniform
            # Heat of 0.0 means backwards-compatible uniform sampling
            from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
            eps = float(machine_epsilon(b, b.array([1.0])))

            if heat_sum > eps:
                # Heat-weighted sampling: higher heat = more samples
                weights = [h / heat_sum for h in heats]
                batch = random.choices(events, weights=weights, k=n_samples)
            else:
                # Uniform sampling when no heat signals (backwards compatible)
                batch = random.sample(events, n_samples)

            # Get or initialize LoRA weights for this layer/weight
            lora_a, lora_b = self._get_or_init_lora(layer_id, weight_name, events[0])

            # Compute loss and update
            # Loss = MSE between (W + scale * B @ A) @ h and (W @ h + delta @ h)
            # Simplified: minimize ||B @ A @ h - delta @ h / scale||²
            scale = self._metadata.alpha / max(self._metadata.rank, 1)

            batch_loss = 0.0
            grad_a_accum = b.zeros_like(lora_a)
            grad_b_accum = b.zeros_like(lora_b)

            for hidden_state, delta, confidence, _heat in batch:
                # Forward: lora_out = B @ A @ h
                # Hidden state shape: [hidden_dim] or [seq, hidden_dim]
                h = hidden_state
                if len(h.shape) == 1:
                    h = b.reshape(h, (1, -1))  # [1, hidden_dim]

                # A @ h.T -> [rank, seq]
                a_h = lora_a @ h.T

                # B @ (A @ h.T) -> [out_dim, seq]
                lora_out = lora_b @ a_h  # [out_dim, seq]

                # Target: delta @ h.T / scale
                # delta shape: [out_dim, in_dim]
                target = (delta @ h.T) / scale  # [out_dim, seq]

                # Loss: MSE weighted by confidence
                diff = lora_out - target
                loss = confidence * b.mean(diff * diff)
                batch_loss += float(b.to_scalar(loss))

                # Gradients (simplified - actual impl would use mx.grad)
                # d_loss/d_B = 2 * diff @ (A @ h.T).T = 2 * diff @ h @ A.T
                # d_loss/d_A = 2 * B.T @ diff @ h
                grad_b = 2.0 * confidence * diff @ a_h.T / n_samples
                grad_a = 2.0 * confidence * (lora_b.T @ diff) @ h / n_samples

                grad_a_accum = grad_a_accum + grad_a
                grad_b_accum = grad_b_accum + grad_b

            # Update LoRA weights
            lora_a = lora_a - learning_rate * grad_a_accum
            lora_b = lora_b - learning_rate * grad_b_accum
            b.eval(lora_a, lora_b)

            self._lora_weights[(layer_id, weight_name)] = (lora_a, lora_b)

            # Compute gradient norm
            grad_norm = float(
                b.to_scalar(
                    b.sqrt(
                        b.sum(grad_a_accum * grad_a_accum)
                        + b.sum(grad_b_accum * grad_b_accum)
                    )
                )
            )

            total_loss += batch_loss
            total_samples += n_samples
            total_grad_norm += grad_norm

        self._metadata.train_steps += 1
        self._metadata.updated_at = datetime.now().isoformat()

        return TrainStepResult(
            loss=total_loss / max(len(self._event_buffer), 1),
            samples_used=total_samples,
            gradient_norm=total_grad_norm / max(len(self._event_buffer), 1),
        )

    def _get_or_init_lora(
        self,
        layer_id: int,
        weight_name: str,
        sample_event: tuple["Array", "Array", float],
    ) -> tuple["Array", "Array"]:
        """Get or initialize LoRA weights for a layer/weight pair."""
        key = (layer_id, weight_name)
        if key in self._lora_weights:
            return self._lora_weights[key]

        b = self._backend
        hidden_state, delta, _ = sample_event

        # Infer dimensions from sample
        in_dim = hidden_state.shape[-1]
        out_dim = delta.shape[0]
        rank = self._metadata.rank

        # Initialize: A ~ N(0, 0.01), B = 0
        lora_a = b.random_normal((rank, in_dim)) * 0.01
        lora_b = b.zeros((out_dim, rank))
        b.eval(lora_a, lora_b)

        self._lora_weights[key] = (lora_a, lora_b)
        return lora_a, lora_b

    def is_known_region(self, hidden_state: "Array") -> bool:
        """Check if a region has already been learned.

        Used for query optimization - skip retrieval for known regions.

        Parameters
        ----------
        hidden_state : Array
            Activation to check.

        Returns
        -------
        bool
            True if this region (or similar) has been learned.
        """
        h_hash = _compute_array_hash(hidden_state, self._backend)
        # Check if any learned hash starts with this prefix (similarity)
        return any(
            learned.startswith(h_hash[:8])
            for learned in self._metadata.learned_region_hashes
        )

    def save(self) -> Path:
        """Save current state to disk.

        Returns
        -------
        Path
            Path to the store directory.
        """
        # Save metadata
        metadata_path = self._store_dir / METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(self._metadata.to_dict(), f, indent=2)

        # Save LoRA weights
        if self._lora_weights:
            self._save_lora_weights()

        # Save event buffer
        if self.buffer_size > 0:
            self._save_events()

        logger.info(
            "Saved LoRA memory store: agent=%s, buffer=%d, lora_keys=%d",
            self._agent_id,
            self.buffer_size,
            len(self._lora_weights),
        )

        return self._store_dir

    def _save_lora_weights(self) -> None:
        """Save LoRA weights to safetensors."""
        try:
            tensors: dict[str, Any] = {}
            for (layer_id, weight_name), (lora_a, lora_b) in self._lora_weights.items():
                key_base = f"{layer_id}_{weight_name}"
                tensors[f"{key_base}_lora_a"] = lora_a
                tensors[f"{key_base}_lora_b"] = lora_b

            lora_path = self._store_dir / LORA_WEIGHTS_FILE
            self._backend.save_safetensors(str(lora_path), tensors)

        except Exception as e:
            logger.error("Failed to save LoRA weights: %s", e)

    def _save_events(self) -> None:
        """Save event buffer to safetensors."""
        try:
            tensors: dict[str, Any] = {}
            for (layer_id, weight_name), events in self._event_buffer.items():
                key_base = f"{layer_id}_{weight_name}"
                for idx, (h, d, conf, _heat) in enumerate(events):
                    tensors[f"{key_base}_h_{idx}"] = h
                    tensors[f"{key_base}_d_{idx}"] = d
                    tensors[f"{key_base}_c_{idx}"] = self._backend.array([conf])

            events_path = self._store_dir / EVENTS_FILE
            self._backend.save_safetensors(str(events_path), tensors)

        except Exception as e:
            logger.error("Failed to save events: %s", e)

    def load(self) -> bool:
        """Reload state from disk.

        Returns
        -------
        bool
            True if load succeeded, False otherwise.
        """
        try:
            self._metadata = self._load_or_create_metadata(
                rank=self._metadata.rank,
                alpha=self._metadata.alpha,
                target_modules=self._metadata.target_modules,
            )
            self._load_state()
            return True
        except Exception as e:
            logger.error("Failed to load store: %s", e)
            return False

    def merge_to_base(
        self,
        model: Any,
        null_space_tracker: Any | None = None,
    ) -> MergeResult:
        """Merge trained LoRA weights into base model via null-space projection.

        This implements the "sleep consolidation" phase - transferring
        hippocampus (LoRA) knowledge to neocortex (base weights).

        Parameters
        ----------
        model : Any
            The model to merge into.
        null_space_tracker : NullSpaceTracker, optional
            Tracker for null-space projection. If None, merges directly.

        Returns
        -------
        MergeResult
            Merge statistics.
        """
        if not self._lora_weights:
            return MergeResult(
                success=False,
                layers_merged=0,
                preserved_fraction=0.0,
                timestamp=datetime.now().isoformat(),
                error="No LoRA weights to merge",
            )

        b = self._backend
        scale = self._metadata.alpha / max(self._metadata.rank, 1)
        layers_merged = 0
        total_preserved = 0.0
        total_original = 0.0

        try:
            # Get base model layers
            base_model = getattr(model, "model", model)
            layers = getattr(base_model, "layers", [])

            for (layer_id, weight_name), (lora_a, lora_b) in self._lora_weights.items():
                if layer_id >= len(layers):
                    continue

                layer = layers[layer_id]

                # Navigate to weight
                obj = layer
                for attr in weight_name.split("."):
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break

                if obj is None:
                    continue

                weight_holder = obj
                current_weight = getattr(weight_holder, "weight", None)
                if current_weight is None:
                    continue

                # Compute delta: scale * B @ A
                delta = scale * (lora_b @ lora_a)
                b.eval(delta)

                original_norm = float(
                    b.to_scalar(b.sqrt(b.sum(delta * delta)))
                )
                total_original += original_norm

                # Project through null-space if tracker provided
                if null_space_tracker is not None:
                    projector = null_space_tracker.get_layer_projector(layer_id)
                    if projector is not None:
                        delta = projector.project(delta)
                        b.eval(delta)

                preserved_norm = float(
                    b.to_scalar(b.sqrt(b.sum(delta * delta)))
                )
                total_preserved += preserved_norm

                # Apply update
                new_weight = current_weight + delta
                b.eval(new_weight)
                setattr(weight_holder, "weight", new_weight)

                layers_merged += 1

                # Mark region as learned
                for events in self._event_buffer.get((layer_id, weight_name), []):
                    h_hash = _compute_array_hash(events[0], b)
                    d_hash = _compute_array_hash(events[1], b)
                    self._metadata.learned_region_hashes.add(f"{h_hash}_{d_hash}")

            # Update metadata
            self._metadata.merge_count += 1
            self._metadata.updated_at = datetime.now().isoformat()

            # Save merge history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = self._history_dir / f"merged_{timestamp}.json"
            with open(history_file, "w") as f:
                json.dump(
                    {
                        "layers_merged": layers_merged,
                        "preserved_fraction": (
                            total_preserved / max(total_original, 1e-10)
                        ),
                        "timestamp": timestamp,
                    },
                    f,
                    indent=2,
                )

            preserved_frac = total_preserved / max(total_original, 1e-10)

            logger.info(
                "Merged LoRA to base: layers=%d, preserved=%.2f%%",
                layers_merged,
                preserved_frac * 100,
            )

            return MergeResult(
                success=True,
                layers_merged=layers_merged,
                preserved_fraction=preserved_frac,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error("Merge failed: %s", e)
            return MergeResult(
                success=False,
                layers_merged=layers_merged,
                preserved_fraction=0.0,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

    def reset_lora(self) -> None:
        """Clear LoRA weights and event buffer after merge.

        Call this after successful merge_to_base to prepare for
        new session learning.
        """
        self._lora_weights.clear()
        self._event_buffer.clear()
        self._metadata.buffer_size = 0
        self._metadata.updated_at = datetime.now().isoformat()

        # Remove persisted files
        events_path = self._store_dir / EVENTS_FILE
        lora_path = self._store_dir / LORA_WEIGHTS_FILE

        if events_path.exists():
            events_path.unlink()
        if lora_path.exists():
            lora_path.unlink()

        # Save updated metadata
        self.save()

        logger.info("Reset LoRA memory: agent=%s", self._agent_id)

    def get_stats(self) -> dict[str, Any]:
        """Get current store statistics."""
        return {
            "agent_id": self._agent_id,
            "buffer_size": self.buffer_size,
            "lora_weights_count": len(self._lora_weights),
            "event_count": self._metadata.event_count,
            "train_steps": self._metadata.train_steps,
            "merge_count": self._metadata.merge_count,
            "learned_regions": len(self._metadata.learned_region_hashes),
        }


def get_or_create_store(
    agent_id: str,
    base_model_path: str | Path,
    rank: int = 8,
    alpha: float = 16.0,
) -> LoRAMemoryStore:
    """Get or create a LoRA memory store for an agent.

    Parameters
    ----------
    agent_id : str
        Unique identifier for the agent.
    base_model_path : str | Path
        Path to the base model.
    rank : int
        LoRA rank.
    alpha : float
        LoRA alpha.

    Returns
    -------
    LoRAMemoryStore
        Configured store (loaded from disk if exists).
    """
    store = LoRAMemoryStore(
        agent_id=agent_id,
        base_model_path=base_model_path,
        rank=rank,
        alpha=alpha,
    )
    return store


__all__ = [
    "LORA_MEMORY_VERSION",
    "LORA_MEMORY_BASE_DIR",
    "MemoryEventSource",
    "LoRAMemoryEvent",
    "LoRAMemoryMetadata",
    "TrainStepResult",
    "MergeResult",
    "LoRAMemoryStore",
    "get_or_create_store",
]
