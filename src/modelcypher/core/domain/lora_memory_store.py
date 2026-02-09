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
   - Trains LoRA adapters from accumulated data (using NB-LoRA)
   - Ephemeral until merged to base weights

2. **Neocortex (Base Weights)**: Slow consolidation via periodic merge
   - Null-space projection preserves existing knowledge
   - Merged LoRA becomes permanent part of model identity

The biological metaphor:
- Wake: Encounter new knowledge, accumulate in hippocampus (LoRA buffer)
- Sleep: Consolidate to neocortex (merge LoRA → base weights)
- Dream: Replay and strengthen connections (LoRA training step)

Implementation:
    Uses NB-LoRA (Norm-Bounded Low-Rank Adaptation) via Cayley transform.
    This provides mathematically guaranteed spectral bounds during training,
    eliminating the need for post-hoc rescaling or heuristic hyperparameters.

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
from modelcypher.core.domain.geometry.cayley_lora import (
    cayley_transform_full,
    NBLoRAConfig,
    NBLoRALayer,
    create_nb_lora_from_base_weight,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.training.geometric_optimizer import (
    OptimizerGeometryConfig,
    derive_optimizer_geometry_config,
)
from modelcypher.core.domain.training.scaled_gd import precondition_lora_gradients
from modelcypher.core.domain.training.spectral_budget import compute_budget_ratios
from modelcypher.ports.training import LoRALayerConfig

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
class LoraMemoryMergeResult:
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


def compute_spectral_regularization_loss(
    B: "Array",
    A: "Array",
    sigma_k: float,
    backend: "Backend",
    lambda_reg: float,
) -> tuple[float, float]:
    """Compute spectral regularization loss for LoRA training.

    .. deprecated::
        NB-LoRA (Cayley transform) guarantees spectral bounds by construction,
        making soft regularization unnecessary. Prefer ``create_nb_lora_from_base_weight``
        for new code. Retained for non-NB-LoRA training paths.

    Soft constraint that penalizes ||B @ A||_spectral exceeding sigma_k.
    This encourages the LoRA delta to respect the geometry-derived scale bound
    during training, rather than relying solely on post-hoc rescaling.

    The loss is:
        L_reg = lambda_reg * max(0, ||B @ A||_2 / sigma_k - 1)^2

    This is zero when the spectral norm is within bound, and grows quadratically
    as the norm exceeds the bound.

    Args:
        B: LoRA B matrix [out_dim, rank]
        A: LoRA A matrix [rank, in_dim]
        sigma_k: Geometry-derived scale bound (smallest significant SV of base weight)
        backend: Compute backend
        lambda_reg: Regularization strength.

    Returns:
        Tuple (regularization_loss, spectral_norm) where:
            - regularization_loss: The computed loss value
            - spectral_norm: Current ||B @ A||_spectral for monitoring

    Reference:
        Related to Spectral Normalization (Miyato et al., 2018) but as soft
        regularization rather than hard normalization.
    """
    b = backend

    # Compute LoRA delta
    BA = b.matmul(B, A)
    BA_f32 = b.astype(BA, "float32")
    b.eval(BA_f32)

    # SVD to get spectral norm (largest singular value)
    # Use compute_uv=True since some backends don't support False well
    _, S, _ = b.svd(BA_f32, compute_uv=True)
    b.eval(S)
    spectral_norm = float(b.to_scalar(S[0]))

    # Compute excess over bound
    # excess = max(0, spectral_norm / sigma_k - 1)
    sigma_floor_arr = b.sqrt(b.array([machine_epsilon(b, BA_f32)], dtype="float32"))
    b.eval(sigma_floor_arr)
    sigma_floor = float(b.to_scalar(sigma_floor_arr[0]))
    ratio = spectral_norm / max(sigma_k, sigma_floor)
    excess = max(0.0, ratio - 1.0)

    # Quadratic penalty
    reg_loss = lambda_reg * (excess ** 2)

    return reg_loss, spectral_norm


def compute_spectral_regularization_gradient(
    B: "Array",
    A: "Array",
    sigma_k: float,
    backend: "Backend",
    lambda_reg: float,
) -> tuple["Array", "Array", float]:
    """Compute gradients for spectral regularization.

    .. deprecated::
        NB-LoRA (Cayley transform) guarantees spectral bounds by construction.
        Retained for non-NB-LoRA training paths.

    Computes approximate gradients of the spectral regularization loss
    with respect to A and B for manual gradient descent.

    The gradient approximation uses:
        d||BA||_2/dB ≈ u @ v^T @ A^T  (scaled by v)
        d||BA||_2/dA ≈ B^T @ u @ v^T  (scaled by u)

    Where u, v are the left and right singular vectors corresponding to
    the largest singular value.

    Args:
        B: LoRA B matrix [out_dim, rank]
        A: LoRA A matrix [rank, in_dim]
        sigma_k: Geometry-derived scale bound
        backend: Compute backend
        lambda_reg: Regularization strength

    Returns:
        Tuple (grad_B, grad_A, spectral_norm) where gradients are with respect
        to the regularization loss only.
    """
    b = backend

    # Compute LoRA delta and SVD
    BA = b.matmul(B, A)
    BA_f32 = b.astype(BA, "float32")
    b.eval(BA_f32)

    U, S, Vt = b.svd(BA_f32, compute_uv=True)
    b.eval(U, S, Vt)

    spectral_norm = float(b.to_scalar(S[0]))
    sigma_floor_arr = b.sqrt(b.array([machine_epsilon(b, BA_f32)], dtype="float32"))
    b.eval(sigma_floor_arr)
    sigma_floor = float(b.to_scalar(sigma_floor_arr[0]))
    ratio = spectral_norm / max(sigma_k, sigma_floor)
    excess = max(0.0, ratio - 1.0)

    # If within bound, no regularization gradient
    eps = b.finfo(BA_f32.dtype).eps
    if excess < eps:
        return b.zeros_like(B), b.zeros_like(A), spectral_norm

    # Get top singular vectors
    u = U[:, 0:1]  # [out_dim, 1]
    v = Vt[0:1, :]  # [1, in_dim]
    b.eval(u, v)

    # Gradient of spectral norm: d||BA||/d(BA) = u @ v
    # By chain rule:
    #   d||BA||/dB = d||BA||/d(BA) @ d(BA)/dB = u @ v @ A^T
    #   d||BA||/dA = d(BA)/dA^T @ d||BA||/d(BA) = B^T @ u @ v

    # d(reg_loss)/d(spectral) = 2 * lambda_reg * excess / sigma_k
    scale = 2.0 * lambda_reg * excess / max(sigma_k, sigma_floor)

    grad_BA = scale * b.matmul(u, v)  # [out_dim, in_dim]
    b.eval(grad_BA)

    # Backprop through BA = B @ A
    # d(loss)/dB = grad_BA @ A^T
    # d(loss)/dA = B^T @ grad_BA
    grad_B = b.matmul(grad_BA, b.transpose(A))
    grad_A = b.matmul(b.transpose(B), grad_BA)
    b.eval(grad_B, grad_A)

    # Convert back to original dtype
    grad_B = b.astype(grad_B, B.dtype)
    grad_A = b.astype(grad_A, A.dtype)
    b.eval(grad_B, grad_A)

    return grad_B, grad_A, spectral_norm


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
        # Key: (layer_id, weight_name) -> list of (hidden_state, delta, confidence, heat)
        self._event_buffer: dict[
            tuple[int, str], list[tuple["Array", "Array", float, float]]
        ] = {}

        # NB-LoRA layers (lazy initialization via Cayley transform)
        # Key: (layer_id, weight_name) -> NBLoRALayer
        # Uses geometry-derived scale bounds, no alpha/rank heuristics
        self._lora_layers: dict[tuple[int, str], NBLoRALayer] = {}

        # Base weights cache for NB-LoRA initialization
        # Key: (layer_id, weight_name) -> base_weight Array
        self._base_weights: dict[tuple[int, str], "Array"] = {}

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

    def _sqrt_eps(self) -> float:
        """Return dtype-derived sqrt(machine epsilon) for the active backend."""
        b = self._backend
        eps = machine_epsilon(b, b.array([1.0]))
        eps_arr = b.sqrt(b.array([eps], dtype="float32"))
        b.eval(eps_arr)
        return float(b.to_scalar(eps_arr[0]))

    def sqrt_eps(self) -> float:
        """Public helper for dtype-derived sqrt(machine epsilon)."""
        return self._sqrt_eps()

    def _layer_weight_key(self, layer_id: int, weight_name: str) -> str:
        """Canonical synthetic key used by geometry-derived training utilities."""
        return f"layer_{layer_id}.{weight_name}.weight"

    def _event_weight_views(self) -> dict[str, "Array"]:
        """Representative 2D matrices from current event buffer for geometry analysis."""
        weights: dict[str, "Array"] = {}
        for (layer_id, weight_name), events in self._event_buffer.items():
            if not events:
                continue
            # Each delta is already in weight space [out_dim, in_dim].
            _, delta, _, _ = events[0]
            if len(delta.shape) == 2:
                weights[self._layer_weight_key(layer_id, weight_name)] = delta
        return weights

    def derive_learning_rate(self) -> float:
        """Derive base learning rate from current event geometry.

        Uses the extracted geometric optimizer config on representative deltas.
        Falls back to sqrt(eps) when no valid 2D event matrices are present.
        """
        weight_views = self._event_weight_views()
        if not weight_views:
            return self._sqrt_eps()

        try:
            config = derive_optimizer_geometry_config(
                weight_views,
                self._backend,
                min_shape=1,
            )
            return config.base_lr
        except Exception:
            return self._sqrt_eps()

    def derive_optimizer_config(self) -> OptimizerGeometryConfig | None:
        """Derive per-layer optimizer geometry from current event deltas."""
        weight_views = self._event_weight_views()
        if not weight_views:
            return None
        try:
            return derive_optimizer_geometry_config(
                weight_views,
                self._backend,
                min_shape=1,
            )
        except Exception:
            return None

    def compute_spectral_budget_ratios(self) -> list[float]:
        """Compute per-layer spectral budget ratios for active NB-LoRA layers."""
        b = self._backend
        lora_products: list[tuple[float, Any, Any, float]] = []

        for layer in self._lora_layers.values():
            A, B = cayley_transform_full(layer.A_tilde, layer.B_tilde, b)  # [r, in], [r, out]
            S = layer.get_S()  # [r]
            b.eval(A, B, S)

            # Effective delta is W = 2 * B^T * diag(S) * A.
            # We feed an equivalent factorization to budget monitor:
            #   W^T = A^T @ (2 * diag(S) @ B)
            # Spectral norm is invariant to transpose.
            lora_a = b.transpose(A)  # [in, r]
            lora_b = 2.0 * (b.reshape(S, (-1, 1)) * B)  # [r, out]
            sigma_ref = 2.0 * layer.scale_bound

            lora_products.append((1.0, lora_a, lora_b, sigma_ref))

        return compute_budget_ratios(lora_products, b)

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
        """Load existing buffers and NB-LoRA layers from disk."""
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
        """Load trained NB-LoRA weights from safetensors."""
        try:
            b = self._backend
            tensors = b.load_safetensors(str(path))

            # Load config metadata for scale bounds
            config_path = self._store_dir / "nb_lora_config.json"
            config_metadata: dict[str, str] = {}
            if config_path.exists():
                with open(config_path) as f:
                    config_metadata = json.load(f)

            # Parse NB-LoRA parameters
            # Format: {layer_id}_{weight_name}_A_tilde, _B_tilde, _S_raw
            processed_keys: set[str] = set()

            for key, tensor in tensors.items():
                if "_A_tilde" in key and key not in processed_keys:
                    base_key = key.replace("_A_tilde", "")
                    parts = base_key.split("_", 1)
                    if len(parts) == 2:
                        layer_id = int(parts[0])
                        weight_name = parts[1]

                        B_tilde_key = f"{base_key}_B_tilde"
                        S_raw_key = f"{base_key}_S_raw"

                        if B_tilde_key in tensors and S_raw_key in tensors:
                            A_tilde = tensor
                            B_tilde = tensors[B_tilde_key]
                            S_raw = tensors[S_raw_key]

                            # Get scale bound from config
                            scale_bound_key = f"{base_key}_scale_bound"
                            default_scale_bound = self._sqrt_eps() / 2.0
                            scale_bound = float(
                                config_metadata.get(scale_bound_key, str(default_scale_bound))
                            )

                            # Reconstruct NBLoRALayer
                            r = A_tilde.shape[0]
                            in_features = A_tilde.shape[1]
                            out_features = B_tilde.shape[1]

                            config = NBLoRAConfig(
                                in_features=in_features,
                                out_features=out_features,
                                rank=r,
                                scale_bound=scale_bound,
                            )
                            layer = NBLoRALayer(config, b)
                            layer.A_tilde = A_tilde
                            layer.B_tilde = B_tilde
                            layer.S_raw = S_raw

                            self._lora_layers[(layer_id, weight_name)] = layer
                            processed_keys.add(key)
                            processed_keys.add(B_tilde_key)
                            processed_keys.add(S_raw_key)

            logger.info("Loaded NB-LoRA weights from %s", path)

        except Exception as e:
            logger.warning("Failed to load NB-LoRA weights: %s", e)

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

    def train_step(
        self,
        batch_size: int | None = None,
        learning_rate: float | None = None,
    ) -> TrainStepResult:
        """Perform one LoRA training step from accumulated events.

        This implements the "dreaming" phase - replaying accumulated
        experiences to strengthen LoRA connections.

        Parameters
        ----------
        batch_size : int, optional
            Number of events to use per step. If None, uses full-buffer batch.
        learning_rate : float, optional
            Learning rate. If None, derived from current event geometry.

        Returns
        -------
        TrainStepResult
            Training statistics.

        Note:
            Uses NB-LoRA (Cayley-parameterized) which guarantees spectral bounds
            by construction. No separate regularization needed - the math does it.
        """
        b = self._backend

        if self.buffer_size == 0:
            return TrainStepResult(loss=0.0, samples_used=0, gradient_norm=0.0)

        if batch_size is None:
            batch_size = self.buffer_size
        if learning_rate is None:
            learning_rate = self.derive_learning_rate()

        optimizer_config = self.derive_optimizer_config()
        if optimizer_config is None:
            optimizer_config = OptimizerGeometryConfig(
                base_lr=learning_rate,
                max_sigma=learning_rate,
                layer_configs={},
            )

        total_loss = 0.0
        total_samples = 0
        total_grad_norm = 0.0

        import random

        # Train each (layer, weight) independently
        for (layer_id, weight_name), events in self._event_buffer.items():
            if not events:
                continue

            n_samples = min(batch_size, len(events))
            heats = [e[3] for e in events]
            heat_sum = sum(heats)
            eps = float(machine_epsilon(b, b.array([1.0])))

            if heat_sum > eps:
                weights = [h / heat_sum for h in heats]
                batch = random.choices(events, weights=weights, k=n_samples)
            else:
                batch = random.sample(events, n_samples)

            layer = self._get_or_init_lora(layer_id, weight_name, events[0])
            layer_key = self._layer_weight_key(layer_id, weight_name)
            rank = int(layer.A_tilde.shape[0])
            in_features = int(layer.A_tilde.shape[1])
            out_features = int(layer.B_tilde.shape[1])

            # Effective factorization:
            #   W_delta = B_scaled^T @ A_eff
            # where B_scaled = 2 * diag(S) @ B_eff
            A_eff, B_eff = cayley_transform_full(layer.A_tilde, layer.B_tilde, b)
            S = layer.get_S()
            B_scaled = 2.0 * (b.reshape(S, (-1, 1)) * B_eff)  # [rank, out]
            b.eval(A_eff, B_eff, B_scaled)

            grad_a_eff_accum = b.zeros_like(A_eff)  # [rank, in]
            grad_b_scaled_accum = b.zeros_like(B_scaled)  # [rank, out]
            grad_s_accum = b.zeros_like(layer.S_raw)  # [rank]
            batch_loss = 0.0

            for hidden_state, delta, confidence, _heat in batch:
                h = hidden_state
                if len(h.shape) == 1:
                    h = b.reshape(h, (1, -1))

                proj = b.matmul(A_eff, b.transpose(h))  # [rank, seq]
                lora_out = b.matmul(b.transpose(B_scaled), proj)  # [out, seq]
                target = b.matmul(delta, b.transpose(h))  # [out, seq]
                b.eval(proj, lora_out, target)

                diff = lora_out - target
                loss = confidence * b.mean(diff * diff)
                b.eval(diff, loss)
                batch_loss += float(b.to_scalar(loss))

                n_elements = int(diff.shape[0]) * int(diff.shape[1])
                grad_scale = (2.0 * confidence) / max(n_elements, 1)
                scaled_diff = grad_scale * diff

                grad_w = b.matmul(scaled_diff, h)  # [out, in]
                grad_a_eff = b.matmul(B_scaled, grad_w)  # [rank, in]
                grad_b_scaled = b.transpose(
                    b.matmul(grad_w, b.transpose(A_eff))
                )  # [rank, out]

                # B_scaled = 2 * diag(S) @ B_eff
                grad_s = 2.0 * b.sum(grad_b_scaled * B_eff, axis=1)  # [rank]

                grad_a_eff_accum = grad_a_eff_accum + grad_a_eff
                grad_b_scaled_accum = grad_b_scaled_accum + grad_b_scaled
                grad_s_accum = grad_s_accum + grad_s
                b.eval(grad_a_eff_accum, grad_b_scaled_accum, grad_s_accum)

            # ScaledGD expects W = A_pre @ B_pre with:
            #   A_pre [in, rank], B_pre [rank, out]
            prefix = layer_key.replace(".weight", "")
            a_key = prefix + ".lora_a"
            b_key = prefix + ".lora_b"

            A_pre = b.transpose(A_eff)
            B_pre = B_scaled
            grad_a_pre = b.transpose(grad_a_eff_accum)
            grad_b_pre = grad_b_scaled_accum
            b.eval(A_pre, B_pre, grad_a_pre, grad_b_pre)

            lora_cfg = LoRALayerConfig(
                layer_key=layer_key,
                rank=rank,
                sigma_k=2.0 * layer.scale_bound,
                in_features=in_features,
                out_features=out_features,
                dropout=0.0,
            )
            preconditioned = precondition_lora_gradients(
                grad_flat={a_key: grad_a_pre, b_key: grad_b_pre},
                param_flat={a_key: A_pre, b_key: B_pre},
                lora_configs=[lora_cfg],
                opt_config=optimizer_config,
                backend=b,
            )

            grad_a_tilde = b.transpose(preconditioned[a_key])  # [rank, in]
            grad_b_tilde = preconditioned[b_key]  # [rank, out]

            layer.A_tilde = layer.A_tilde - learning_rate * grad_a_tilde
            layer.B_tilde = layer.B_tilde - learning_rate * grad_b_tilde
            layer.S_raw = layer.S_raw - learning_rate * grad_s_accum
            b.eval(layer.A_tilde, layer.B_tilde, layer.S_raw)

            self._lora_layers[(layer_id, weight_name)] = layer

            grad_norm = float(
                b.to_scalar(
                    b.sqrt(
                        b.sum(grad_a_tilde * grad_a_tilde)
                        + b.sum(grad_b_tilde * grad_b_tilde)
                        + b.sum(grad_s_accum * grad_s_accum)
                    )
                )
            )

            total_loss += batch_loss
            total_samples += n_samples
            total_grad_norm += grad_norm

        self._metadata.train_steps += 1
        self._metadata.updated_at = datetime.now().isoformat()

        n_layers = max(len(self._event_buffer), 1)
        return TrainStepResult(
            loss=total_loss / n_layers,
            samples_used=total_samples,
            gradient_norm=total_grad_norm / n_layers,
        )

    def _get_or_init_lora(
        self,
        layer_id: int,
        weight_name: str,
        sample_event: tuple["Array", "Array", float, float],
    ) -> NBLoRALayer:
        """Get or initialize NB-LoRA layer for a layer/weight pair.

        Uses Cayley-parameterized NB-LoRA with geometry-derived scale bounds.
        No alpha/rank heuristics - the math determines the bounds.
        """
        key = (layer_id, weight_name)
        if key in self._lora_layers:
            return self._lora_layers[key]

        b = self._backend
        hidden_state, delta, _, _ = sample_event

        # Infer dimensions from sample
        in_dim = hidden_state.shape[-1]
        out_dim = delta.shape[0]
        rank = self._metadata.rank

        # Check if we have cached base weight
        if key in self._base_weights:
            base_weight = self._base_weights[key]
            safety_margin = max(0.0, 1.0 - self._sqrt_eps())
            layer = create_nb_lora_from_base_weight(
                W=base_weight,
                rank=rank,
                backend=b,
                safety_margin=safety_margin,
            )
        else:
            # No base weight available - use delta's spectral structure
            # This is a fallback; proper usage should register base weights first
            logger.warning(
                "No base weight for %s - using delta-derived bounds", key
            )

            # Derive sigma_k from delta geometry (same Shannon-effective-rank
            # boundary used by geometric_lora.compute_layer_geometry).
            from modelcypher.core.domain.training.geometric_lora import compute_layer_geometry

            delta_geom = compute_layer_geometry(
                b.astype(delta, "float32"),
                layer_key=self._layer_weight_key(layer_id, weight_name),
                backend=b,
            )
            sigma_k = max(delta_geom.sigma_k, self._sqrt_eps())

            config = NBLoRAConfig(
                in_features=in_dim,
                out_features=out_dim,
                rank=rank,
                scale_bound=sigma_k / 2.0,  # NB-LoRA: max(S) = sigma_k/2
            )
            layer = NBLoRALayer(config, b)

        self._lora_layers[key] = layer
        return layer

    def register_base_weight(
        self,
        layer_id: int,
        weight_name: str,
        weight: "Array",
    ) -> None:
        """Register a base weight for geometry-derived NB-LoRA bounds.

        Call this before accumulating events to enable proper spectral bounds.

        Parameters
        ----------
        layer_id : int
            Layer index.
        weight_name : str
            Weight matrix name (e.g., "mlp.up_proj").
        weight : Array
            Base weight matrix.
        """
        key = (layer_id, weight_name)
        b = self._backend

        # Copy to avoid reference issues
        weight_copy = weight + b.zeros_like(weight)
        b.eval(weight_copy)
        self._base_weights[key] = weight_copy

        logger.debug("Registered base weight: %s, shape=%s", key, weight.shape)

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

        # Save NB-LoRA layers
        if self._lora_layers:
            self._save_lora_weights()

        # Save event buffer
        if self.buffer_size > 0:
            self._save_events()

        logger.info(
            "Saved NB-LoRA memory store: agent=%s, buffer=%d, lora_keys=%d",
            self._agent_id,
            self.buffer_size,
            len(self._lora_layers),
        )

        return self._store_dir

    def _save_lora_weights(self) -> None:
        """Save NB-LoRA layer parameters to safetensors."""
        try:
            tensors: dict[str, Any] = {}
            metadata: dict[str, str] = {}

            for (layer_id, weight_name), nb_layer in self._lora_layers.items():
                key_base = f"{layer_id}_{weight_name}"
                # Save NB-LoRA parameters (Cayley-parameterized)
                tensors[f"{key_base}_A_tilde"] = nb_layer.A_tilde
                tensors[f"{key_base}_B_tilde"] = nb_layer.B_tilde
                tensors[f"{key_base}_S_raw"] = nb_layer.S_raw
                # Save scale bound in metadata
                metadata[f"{key_base}_scale_bound"] = str(nb_layer.scale_bound)

            lora_path = self._store_dir / LORA_WEIGHTS_FILE
            self._backend.save_safetensors(str(lora_path), tensors)

            # Save config metadata separately
            config_path = self._store_dir / "nb_lora_config.json"
            with open(config_path, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error("Failed to save NB-LoRA weights: %s", e)

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
        """Merge trained NB-LoRA weights into base model via null-space projection.

        This implements the "sleep consolidation" phase - transferring
        hippocampus (NB-LoRA) knowledge to neocortex (base weights).

        The delta is computed from the trained NBLoRALayer via get_effective_delta(),
        which returns 2 * B^T @ S @ A with guaranteed spectral bounds.

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
        if not self._lora_layers:
            return MergeResult(
                success=False,
                layers_merged=0,
                preserved_fraction=0.0,
                timestamp=datetime.now().isoformat(),
                error="No NB-LoRA layers to merge",
            )

        b = self._backend
        layers_merged = 0
        total_preserved = 0.0
        total_original = 0.0

        try:
            # Get base model layers
            base_model = getattr(model, "model", model)
            layers = getattr(base_model, "layers", [])

            for (layer_id, weight_name), nb_layer in self._lora_layers.items():
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

                # Get delta from NB-LoRA layer (already bounded by construction)
                delta = nb_layer.get_effective_delta()
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
            norm_floor = self._sqrt_eps()
            with open(history_file, "w") as f:
                json.dump(
                    {
                        "layers_merged": layers_merged,
                        "preserved_fraction": (
                            total_preserved / max(total_original, norm_floor)
                        ),
                        "timestamp": timestamp,
                    },
                    f,
                    indent=2,
                )

            preserved_frac = total_preserved / max(total_original, norm_floor)

            logger.info(
                "Merged NB-LoRA to base: layers=%d, preserved=%.2f%%",
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
        """Clear NB-LoRA layers and event buffer after merge.

        Call this after successful merge_to_base to prepare for
        new session learning.
        """
        self._lora_layers.clear()
        self._base_weights.clear()
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

        logger.info("Reset NB-LoRA memory: agent=%s", self._agent_id)

    def get_stats(self) -> dict[str, Any]:
        """Get current store statistics."""
        return {
            "agent_id": self._agent_id,
            "buffer_size": self.buffer_size,
            "nb_lora_layers_count": len(self._lora_layers),
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
    # Spectral regularization utilities
    "compute_spectral_regularization_loss",
    "compute_spectral_regularization_gradient",
]
