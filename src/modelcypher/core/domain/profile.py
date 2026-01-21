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

"""Geometric Profile for Model Merging.

This module provides the core data structure for model profiling:
**Profile once, merge many.**

A GeometricProfile contains everything needed to merge a model:
- Layer activations (the raw activation matrices from probing)
- Trajectory ranks (the geometric ceiling per layer)
- Gram condition numbers (numerical stability per layer)
- Signal ranks (Marchenko-Pastur signal rank per layer)

The profile is computed ONCE per model and stored alongside the weights.
Merge operations load profiles and compute alignment from cached activations.

Storage format:
    .modelcypher/
    ├── profile.json           # Metadata, dimensions, convergence metrics
    └── activations.safetensors # Layer activations [n_probes, hidden_dim]

Schema: mc.geometric_profile.v2
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

PROFILE_VERSION = "mc.geometric_profile.v2"
PROFILE_DIR_NAME = ".modelcypher"


@dataclass
class ProfileActivations:
    """All activation types stored in a geometric profile.

    This is the return type for load_activations() and provides
    structured access to all activation types needed for merging.
    """

    # Hidden layer activations (positions on manifold)
    hidden: dict[int, "Array"] = field(default_factory=dict)

    # MLP intermediate activations (for down_proj null-space)
    intermediate: dict[int, "Array"] = field(default_factory=dict)

    # Gate activations (pre-SiLU gating signals)
    gate: dict[int, "Array"] = field(default_factory=dict)

    # Attention key activations (for K projection alignment)
    k: dict[int, "Array"] = field(default_factory=dict)

    # Embedding layer activations
    embedding: "Array | None" = None

    # === MEAN-POOLED ACTIVATIONS for density analysis ===
    # These are per-probe mean-pooled vectors (1 per probe, not per token)
    # Used by density stage to compute per-concept knowledge density
    # Ensures profile-based merges produce identical results to probe-based
    mean_pooled: dict[int, "Array"] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if all activation types needed for merge are present."""
        return bool(
            self.hidden
            and self.intermediate
            and self.gate
            and self.embedding is not None
        )


PROFILE_METADATA_FILE = "profile.json"
PROFILE_ACTIVATIONS_FILE = "activations.safetensors"


def compute_weights_hash(model_path: str | Path, max_bytes: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of model weights for invalidation.

    Hashes the first max_bytes of safetensors files + config.json content.
    This allows quick detection if model weights changed since profiling.

    Args:
        model_path: Path to model directory
        max_bytes: Maximum bytes to read from weight files (default 1MB)

    Returns:
        SHA256 hex digest of weights + config
    """
    model_path = Path(model_path).expanduser().resolve()
    hasher = hashlib.sha256()

    # Hash config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        hasher.update(config_path.read_bytes())

    # Hash first max_bytes of safetensors files (sorted for determinism)
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    bytes_read = 0
    for sf_path in safetensor_files:
        if bytes_read >= max_bytes:
            break
        try:
            with open(sf_path, "rb") as f:
                chunk = f.read(max_bytes - bytes_read)
                hasher.update(chunk)
                bytes_read += len(chunk)
        except OSError:
            continue

    return hasher.hexdigest()


@dataclass
class ConvergenceMetrics:
    """Metrics tracking how the profile converged during computation.

    These are diagnostic - they tell us whether the profile is complete
    and trustworthy for merge operations.
    """

    # Final activation rank achieved per layer
    final_rank: dict[int, int] = field(default_factory=dict)

    # Trajectory rank (geometric ceiling) per layer
    trajectory_rank: dict[int, int] = field(default_factory=dict)

    # Whether each layer achieved its geometric ceiling
    ceiling_achieved: dict[int, bool] = field(default_factory=dict)

    # CKA stability (change in CKA over last N probes)
    # When this drops below sqrt(eps), geometry is converged
    cka_delta: float = 0.0

    # Total probes processed
    probes_processed: int = 0

    # Probes that failed to process
    probes_failed: int = 0

    # Augmentation rounds (null-space token discovery)
    augmentation_rounds: int = 0

    # Total augmentation probes generated
    augmentation_probes: int = 0

    # === TRAJECTORY-BASED SATURATION METRICS ===
    # Batches until rank saturated per layer (0 = not saturated)
    batches_to_saturation: dict[int, int] = field(default_factory=dict)

    # Total batches processed during manifold mapping
    total_batches: int = 0

    # Whether all layers reached rank saturation
    all_layers_saturated: bool = False

    # Atlas domains covered during profiling
    domains_covered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "final_rank": {str(k): v for k, v in self.final_rank.items()},
            "trajectory_rank": {str(k): v for k, v in self.trajectory_rank.items()},
            "ceiling_achieved": {str(k): v for k, v in self.ceiling_achieved.items()},
            "cka_delta": self.cka_delta,
            "probes_processed": self.probes_processed,
            "probes_failed": self.probes_failed,
            "augmentation_rounds": self.augmentation_rounds,
            "augmentation_probes": self.augmentation_probes,
            "batches_to_saturation": {str(k): v for k, v in self.batches_to_saturation.items()},
            "total_batches": self.total_batches,
            "all_layers_saturated": self.all_layers_saturated,
            "domains_covered": self.domains_covered,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConvergenceMetrics:
        """Create from dict."""
        return cls(
            final_rank={int(k): v for k, v in d.get("final_rank", {}).items()},
            trajectory_rank={int(k): v for k, v in d.get("trajectory_rank", {}).items()},
            ceiling_achieved={int(k): v for k, v in d.get("ceiling_achieved", {}).items()},
            cka_delta=d.get("cka_delta", 0.0),
            probes_processed=d.get("probes_processed", 0),
            probes_failed=d.get("probes_failed", 0),
            augmentation_rounds=d.get("augmentation_rounds", 0),
            augmentation_probes=d.get("augmentation_probes", 0),
            batches_to_saturation={int(k): v for k, v in d.get("batches_to_saturation", {}).items()},
            total_batches=d.get("total_batches", 0),
            all_layers_saturated=d.get("all_layers_saturated", False),
            domains_covered=d.get("domains_covered", []),
        )


@dataclass
class LayerGeometricProfile:
    """Geometric profile of a single layer.

    These are RAW MEASUREMENTS. No interpretation, no thresholds.
    Downstream code (merge, analysis) uses these directly.
    """

    layer_idx: int

    # Numerical rank of activation matrix (# of singular values > sqrt(eps))
    activation_rank: int = 0

    # Trajectory rank - the geometric ceiling (from trajectory analysis)
    # This is the intrinsic manifold dimension - activation_rank cannot exceed this
    trajectory_rank: int = 0

    # Gram matrix condition number (numerical stability of alignment)
    gram_condition: float = 0.0

    # Marchenko-Pastur signal rank (# of eigenvalues above noise floor)
    signal_rank: int = 0

    # Hidden dimension for this layer
    hidden_dim: int = 0

    # Number of probes used
    n_probes: int = 0

    # Null space dimension (hidden_dim - trajectory_rank)
    null_rank: int = 0

    # === STRUCTURAL CAPACITY (from weight matrix ranks) ===
    # These tell us what the layer CAN do, independent of what probes activate
    weight_rank_o_proj: int = 0  # Rank of attention output projection
    weight_rank_down_proj: int = 0  # Rank of MLP down projection
    structural_capacity: int = 0  # Min of weight ranks (true ceiling)

    # === TRAJECTORY-BASED SAMPLING METRICS ===
    # Total samples (positions + velocities) used for this layer
    trajectory_samples: int = 0

    # Position samples (raw token positions in activation space)
    position_samples: int = 0

    # Velocity samples (first differences, manifold tangent vectors)
    velocity_samples: int = 0

    # Atlas domains that contributed samples to this layer
    domains_sampled: list[str] = field(default_factory=list)

    # Batches until this layer's rank saturated (0 if not saturated)
    batches_to_saturation: int = 0

    # Whether this layer reached rank saturation
    saturated: bool = False

    @property
    def is_probe_limited(self) -> bool:
        """True if probes didn't fully activate the structural capacity."""
        return self.structural_capacity > 0 and self.activation_rank < self.structural_capacity

    @property
    def unused_capacity(self) -> int:
        """Dimensions the model CAN use but probes didn't activate."""
        if self.structural_capacity > 0:
            return max(0, self.structural_capacity - self.activation_rank)
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "layer_idx": self.layer_idx,
            "activation_rank": self.activation_rank,
            "trajectory_rank": self.trajectory_rank,
            "gram_condition": self.gram_condition,
            "signal_rank": self.signal_rank,
            "hidden_dim": self.hidden_dim,
            "n_probes": self.n_probes,
            "null_rank": self.null_rank,
            "weight_rank_o_proj": self.weight_rank_o_proj,
            "weight_rank_down_proj": self.weight_rank_down_proj,
            "structural_capacity": self.structural_capacity,
            "trajectory_samples": self.trajectory_samples,
            "position_samples": self.position_samples,
            "velocity_samples": self.velocity_samples,
            "domains_sampled": self.domains_sampled,
            "batches_to_saturation": self.batches_to_saturation,
            "saturated": self.saturated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LayerGeometricProfile:
        """Create from dict."""
        return cls(
            layer_idx=d["layer_idx"],
            activation_rank=d.get("activation_rank", 0),
            trajectory_rank=d.get("trajectory_rank", 0),
            gram_condition=d.get("gram_condition", 0.0),
            signal_rank=d.get("signal_rank", 0),
            hidden_dim=d.get("hidden_dim", 0),
            n_probes=d.get("n_probes", 0),
            null_rank=d.get("null_rank", 0),
            weight_rank_o_proj=d.get("weight_rank_o_proj", 0),
            weight_rank_down_proj=d.get("weight_rank_down_proj", 0),
            structural_capacity=d.get("structural_capacity", 0),
            trajectory_samples=d.get("trajectory_samples", 0),
            position_samples=d.get("position_samples", 0),
            velocity_samples=d.get("velocity_samples", 0),
            domains_sampled=d.get("domains_sampled", []),
            batches_to_saturation=d.get("batches_to_saturation", 0),
            saturated=d.get("saturated", False),
        )


@dataclass
class GeometricProfile:
    """Complete geometric profile of a model for merge operations.

    This is THE canonical representation of a model's geometry.
    Everything needed for merge is derived from this profile.

    The profile contains:
    1. Metadata (model path, hash, version, probe info)
    2. Per-layer geometry (ranks, condition numbers, dimensions)
    3. Convergence metrics (how the profile was computed)

    Layer activations are stored separately in activations.safetensors
    to keep the JSON metadata small and the activations GPU-loadable.
    """

    # === IDENTITY ===
    model_path: str
    profile_version: str = PROFILE_VERSION
    created_at: str = ""
    weights_hash: str = ""  # For invalidation detection

    # === PROBE CORPUS ===
    probe_count: int = 0
    probe_ids: list[str] = field(default_factory=list)
    probe_domains: list[str] = field(default_factory=list)

    # === MODEL DIMENSIONS ===
    hidden_dim: int = 0
    intermediate_dim: int = 0
    num_layers: int = 0
    vocab_size: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0

    # === PER-LAYER GEOMETRY ===
    layer_profiles: dict[int, LayerGeometricProfile] = field(default_factory=dict)

    # === EMBEDDING LAYER ===
    embedding_rank: int = 0
    embedding_gram_condition: float = 0.0
    embedding_n_probes: int = 0

    # === CONVERGENCE ===
    convergence: ConvergenceMetrics = field(default_factory=ConvergenceMetrics)

    # === ACTIVATION STORAGE ===
    # Path to activations.safetensors (relative to profile.json)
    activations_file: str = PROFILE_ACTIVATIONS_FILE

    # Whether activations are available (profile may exist without them)
    has_activations: bool = False

    # === V2: EXTENDED ACTIVATION TYPES FOR FULL MERGE SUPPORT ===
    # These flags indicate which activation types are stored in the profile.
    # For profile-based merging, ALL of these should be True.
    has_hidden: bool = False       # Hidden layer activations (positions)
    has_intermediate: bool = False  # MLP intermediate activations (for down_proj null-space)
    has_gate: bool = False          # Gate activations (pre-SiLU gating signals)
    has_k: bool = False             # K (key) activations for attention alignment
    has_embedding: bool = False     # Embedding layer activations

    def is_complete_for_merge(self) -> bool:
        """Check if this profile has all activation types needed for merging.

        Returns:
            True if profile can be used for full profile-based merging.
        """
        return (
            self.has_activations
            and self.has_hidden
            and self.has_intermediate
            and self.has_gate
            and self.has_embedding
        )

    def __post_init__(self) -> None:
        """Set created_at if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def is_valid_for(self, model_path: str) -> bool:
        """Check if this profile is valid for the given model path.

        A profile is valid if:
        1. The weights hash matches (model hasn't changed)
        2. The profile version is compatible
        """
        if self.profile_version != PROFILE_VERSION:
            logger.warning(
                "Profile version mismatch: %s != %s",
                self.profile_version,
                PROFILE_VERSION,
            )
            return False

        current_hash = compute_weights_hash(model_path)
        if current_hash != self.weights_hash:
            logger.warning(
                "Weights hash mismatch: profile=%s, current=%s",
                self.weights_hash[:16],
                current_hash[:16],
            )
            return False

        return True

    def get_layer_profile(self, layer_idx: int) -> LayerGeometricProfile | None:
        """Get the geometric profile for a specific layer."""
        return self.layer_profiles.get(layer_idx)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "_schema": self.profile_version,
            "model_path": self.model_path,
            "profile_version": self.profile_version,
            "created_at": self.created_at,
            "weights_hash": self.weights_hash,
            "probe_count": self.probe_count,
            "probe_ids": self.probe_ids,
            "probe_domains": self.probe_domains,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "layer_profiles": {
                str(k): v.to_dict() for k, v in self.layer_profiles.items()
            },
            "embedding_rank": self.embedding_rank,
            "embedding_gram_condition": self.embedding_gram_condition,
            "embedding_n_probes": self.embedding_n_probes,
            "convergence": self.convergence.to_dict(),
            "activations_file": self.activations_file,
            "has_activations": self.has_activations,
            # V2 fields
            "has_hidden": self.has_hidden,
            "has_intermediate": self.has_intermediate,
            "has_gate": self.has_gate,
            "has_k": self.has_k,
            "has_embedding": self.has_embedding,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeometricProfile:
        """Create from dict."""
        layer_profiles = {}
        for k, v in d.get("layer_profiles", {}).items():
            layer_profiles[int(k)] = LayerGeometricProfile.from_dict(v)

        convergence_data = d.get("convergence", {})
        convergence = ConvergenceMetrics.from_dict(convergence_data)

        return cls(
            model_path=d["model_path"],
            profile_version=d.get("profile_version", PROFILE_VERSION),
            created_at=d.get("created_at", ""),
            weights_hash=d.get("weights_hash", ""),
            probe_count=d.get("probe_count", 0),
            probe_ids=d.get("probe_ids", []),
            probe_domains=d.get("probe_domains", []),
            hidden_dim=d.get("hidden_dim", 0),
            intermediate_dim=d.get("intermediate_dim", 0),
            num_layers=d.get("num_layers", 0),
            vocab_size=d.get("vocab_size", 0),
            num_attention_heads=d.get("num_attention_heads", 0),
            num_kv_heads=d.get("num_kv_heads", 0),
            layer_profiles=layer_profiles,
            embedding_rank=d.get("embedding_rank", 0),
            embedding_gram_condition=d.get("embedding_gram_condition", 0.0),
            embedding_n_probes=d.get("embedding_n_probes", 0),
            convergence=convergence,
            activations_file=d.get("activations_file", PROFILE_ACTIVATIONS_FILE),
            has_activations=d.get("has_activations", False),
            has_hidden=d.get("has_hidden", False),
            has_intermediate=d.get("has_intermediate", False),
            has_gate=d.get("has_gate", False),
            has_k=d.get("has_k", False),
            has_embedding=d.get("has_embedding", False),
        )

    def save(self, profile_dir: str | Path) -> Path:
        """Save profile metadata to JSON.

        Args:
            profile_dir: Directory to save profile.json
                        (activations.safetensors saved separately)

        Returns:
            Path to saved profile.json
        """
        profile_dir = Path(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = profile_dir / PROFILE_METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info("Saved geometric profile to %s", metadata_path)
        return metadata_path

    @classmethod
    def load(cls, profile_dir: str | Path) -> GeometricProfile:
        """Load profile metadata from JSON.

        Args:
            profile_dir: Directory containing profile.json

        Returns:
            GeometricProfile instance
        """
        profile_dir = Path(profile_dir)
        metadata_path = profile_dir / PROFILE_METADATA_FILE

        with open(metadata_path) as f:
            data = json.load(f)

        profile = cls.from_dict(data)

        # Check if activations file exists
        activations_path = profile_dir / profile.activations_file
        profile.has_activations = activations_path.exists()

        return profile


class GeometricProfileStore:
    """Store for loading and saving geometric profiles.

    Profiles are stored in:
    1. Model sidecar: {model_path}/.modelcypher/profile.json
    2. Central cache: ~/.modelcypher/profiles/geometric/{model_id}/profile.json

    The sidecar is preferred (travels with the model).
    The central cache is fallback (for read-only model directories).
    """

    def __init__(self, central_dir: str | Path | None = None) -> None:
        """Initialize the store.

        Args:
            central_dir: Central profile directory.
                        Defaults to ~/.modelcypher/profiles/geometric/
        """
        if central_dir is None:
            central_dir = Path.home() / ".modelcypher" / "profiles" / "geometric"
        self._central_dir = Path(central_dir)
        self._cache: dict[str, GeometricProfile] = {}

    def profile_dir_for_model(self, model_path: str | Path) -> Path:
        """Get the sidecar profile directory for a model.

        This is the preferred location: {model_path}/.modelcypher/
        """
        model_path = Path(model_path).expanduser().resolve()
        return model_path / PROFILE_DIR_NAME

    def central_profile_dir(self, model_path: str | Path) -> Path:
        """Get the central cache directory for a model.

        Falls back here if sidecar can't be written.
        Uses hash of model path for uniqueness.
        """
        model_path = Path(model_path).expanduser().resolve()
        path_hash = hashlib.sha256(str(model_path).encode()).hexdigest()[:16]
        return self._central_dir / path_hash

    def load(self, model_path: str | Path) -> GeometricProfile | None:
        """Load profile for a model if it exists.

        Checks sidecar first, then central cache.
        Returns None if no valid profile exists.

        Args:
            model_path: Path to model directory

        Returns:
            GeometricProfile if found and valid, None otherwise
        """
        model_path = Path(model_path).expanduser().resolve()
        model_key = str(model_path)

        # Check memory cache
        if model_key in self._cache:
            profile = self._cache[model_key]
            if profile.is_valid_for(model_key):
                return profile
            # Invalid, remove from cache
            del self._cache[model_key]

        # Check sidecar
        sidecar_dir = self.profile_dir_for_model(model_path)
        if (sidecar_dir / PROFILE_METADATA_FILE).exists():
            try:
                profile = GeometricProfile.load(sidecar_dir)
                if profile.is_valid_for(model_key):
                    self._cache[model_key] = profile
                    return profile
                logger.info("Sidecar profile invalid for %s, re-profile needed", model_path)
            except Exception as e:
                logger.warning("Failed to load sidecar profile: %s", e)

        # Check central cache
        central_dir = self.central_profile_dir(model_path)
        if (central_dir / PROFILE_METADATA_FILE).exists():
            try:
                profile = GeometricProfile.load(central_dir)
                if profile.is_valid_for(model_key):
                    self._cache[model_key] = profile
                    return profile
                logger.info("Central profile invalid for %s, re-profile needed", model_path)
            except Exception as e:
                logger.warning("Failed to load central profile: %s", e)

        return None

    def save(
        self,
        profile: GeometricProfile,
        model_path: str | Path,
        prefer_sidecar: bool = True,
    ) -> Path:
        """Save profile for a model.

        Attempts to save to sidecar first, falls back to central cache.

        Args:
            profile: Profile to save
            model_path: Path to model directory
            prefer_sidecar: If True, try sidecar first (default)

        Returns:
            Path to saved profile directory
        """
        model_path = Path(model_path).expanduser().resolve()
        model_key = str(model_path)

        # Update model_path in profile
        profile.model_path = model_key

        # Try sidecar first
        if prefer_sidecar:
            sidecar_dir = self.profile_dir_for_model(model_path)
            try:
                profile.save(sidecar_dir)
                self._cache[model_key] = profile
                return sidecar_dir
            except PermissionError:
                logger.info(
                    "Cannot write sidecar for %s, using central cache",
                    model_path,
                )

        # Fall back to central cache
        central_dir = self.central_profile_dir(model_path)
        profile.save(central_dir)
        self._cache[model_key] = profile
        return central_dir

    def exists(self, model_path: str | Path) -> bool:
        """Check if a valid profile exists for a model."""
        return self.load(model_path) is not None

    def invalidate(self, model_path: str | Path) -> None:
        """Remove cached profile for a model."""
        model_path = Path(model_path).expanduser().resolve()
        model_key = str(model_path)
        if model_key in self._cache:
            del self._cache[model_key]


def save_activations(
    activations: dict[int, "Array"],
    embedding_activations: "Array | None",
    profile_dir: str | Path,
    backend: "Backend",
    *,
    intermediate_activations: dict[int, "Array"] | None = None,
    gate_activations: dict[int, "Array"] | None = None,
    k_activations: dict[int, "Array"] | None = None,
    mean_pooled_activations: dict[int, "Array"] | None = None,
) -> Path:
    """Save layer activations to safetensors file.

    Format:
        - hidden_{idx}: [n_probes, hidden_dim] float32 array (layer hidden states)
        - intermediate_{idx}: [n_probes, intermediate_dim] float32 array (MLP intermediate)
        - gate_{idx}: [n_probes, intermediate_dim] float32 array (pre-SiLU gate)
        - k_{idx}: [n_probes, head_dim] float32 array (attention keys)
        - embedding: [n_probes, hidden_dim] float32 array (if provided)

    Args:
        activations: Dict mapping layer_idx -> hidden activation array
        embedding_activations: Embedding layer activations (optional)
        profile_dir: Directory to save activations.safetensors
        backend: Compute backend for array conversion
        intermediate_activations: Dict mapping layer_idx -> MLP intermediate activations
        gate_activations: Dict mapping layer_idx -> gate activations (pre-SiLU)
        k_activations: Dict mapping layer_idx -> attention key activations

    Returns:
        Path to saved activations file
    """
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    activations_path = profile_dir / PROFILE_ACTIVATIONS_FILE

    tensors: dict[str, Any] = {}

    def _save_activation_dict(
        act_dict: dict[int, "Array"], prefix: str
    ) -> None:
        """Helper to save a dict of activations with a key prefix."""
        for layer_idx, acts in act_dict.items():
            # Ensure stacked if list
            if isinstance(acts, list):
                stacked = backend.stack(acts, axis=0)
            else:
                stacked = acts
            backend.eval(stacked)

            # Convert to float32 for storage
            stacked_f32 = backend.astype(stacked, "float32")
            backend.eval(stacked_f32)

            key = f"{prefix}_{layer_idx}"
            tensors[key] = stacked_f32

    # Save hidden layer activations
    _save_activation_dict(activations, "hidden")

    # Save intermediate activations (MLP)
    if intermediate_activations:
        _save_activation_dict(intermediate_activations, "intermediate")

    # Save gate activations (pre-SiLU)
    if gate_activations:
        _save_activation_dict(gate_activations, "gate")

    # Save K (key) activations for attention alignment
    if k_activations:
        _save_activation_dict(k_activations, "k")

    # Save mean-pooled activations for density analysis (per-probe, not per-token)
    if mean_pooled_activations:
        _save_activation_dict(mean_pooled_activations, "mean_pooled")

    # Save embedding activations
    if embedding_activations is not None:
        if isinstance(embedding_activations, list):
            emb_stacked = backend.stack(embedding_activations, axis=0)
        else:
            emb_stacked = embedding_activations
        backend.eval(emb_stacked)
        emb_f32 = backend.astype(emb_stacked, "float32")
        backend.eval(emb_f32)
        tensors["embedding"] = emb_f32

    # Save using backend's safetensors method
    backend.save_safetensors(str(activations_path), tensors)

    logger.info(
        "Saved %d layer activations to %s",
        len(activations) + (1 if embedding_activations is not None else 0),
        activations_path,
    )
    return activations_path


def load_activations(
    profile_dir: str | Path,
    backend: "Backend",
) -> ProfileActivations:
    """Load layer activations from safetensors file.

    Args:
        profile_dir: Directory containing activations.safetensors
        backend: Compute backend for array conversion

    Returns:
        ProfileActivations containing all stored activation types
    """
    try:
        import mlx.core as mx
        from mlx.utils import load

        HAS_MLX = True
    except ImportError:
        HAS_MLX = False

    profile_dir = Path(profile_dir)
    activations_path = profile_dir / PROFILE_ACTIVATIONS_FILE

    if not activations_path.exists():
        raise FileNotFoundError(f"Activations file not found: {activations_path}")

    # Load tensors
    if HAS_MLX:
        tensors = load(str(activations_path))
    else:
        try:
            from safetensors.numpy import load_file

            np_tensors = load_file(str(activations_path))
            tensors = {k: backend.array(v) for k, v in np_tensors.items()}
        except ImportError:
            raise RuntimeError(
                "Neither mlx nor safetensors.numpy available for loading activations"
            )

    # Parse all activation types
    result = ProfileActivations()

    for key, tensor in tensors.items():
        if key == "embedding":
            result.embedding = tensor
        elif key.startswith("hidden_"):
            layer_idx = int(key.split("_")[1])
            result.hidden[layer_idx] = tensor
        elif key.startswith("intermediate_"):
            layer_idx = int(key.split("_")[1])
            result.intermediate[layer_idx] = tensor
        elif key.startswith("gate_"):
            layer_idx = int(key.split("_")[1])
            result.gate[layer_idx] = tensor
        elif key.startswith("k_"):
            layer_idx = int(key.split("_")[1])
            result.k[layer_idx] = tensor
        elif key.startswith("mean_pooled_"):
            layer_idx = int(key.split("_")[2])
            result.mean_pooled[layer_idx] = tensor
        elif key.startswith("layer_"):
            # Legacy format: layer_{idx} -> hidden_{idx}
            layer_idx = int(key.split("_")[1])
            result.hidden[layer_idx] = tensor

    total_count = (
        len(result.hidden)
        + len(result.intermediate)
        + len(result.gate)
        + len(result.k)
        + len(result.mean_pooled)
        + (1 if result.embedding is not None else 0)
    )
    logger.info(
        "Loaded %d activation tensors from %s (hidden=%d, intermediate=%d, gate=%d, k=%d, mean_pooled=%d, embedding=%s)",
        total_count,
        activations_path,
        len(result.hidden),
        len(result.intermediate),
        len(result.gate),
        len(result.k),
        len(result.mean_pooled),
        result.embedding is not None,
    )

    return result


__all__ = [
    "PROFILE_VERSION",
    "PROFILE_DIR_NAME",
    "PROFILE_METADATA_FILE",
    "PROFILE_ACTIVATIONS_FILE",
    "ProfileActivations",
    "compute_weights_hash",
    "ConvergenceMetrics",
    "LayerGeometricProfile",
    "GeometricProfile",
    "GeometricProfileStore",
    "save_activations",
    "load_activations",
]
