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

"""Unified ModelProfile - The transparent black box.

This module provides a single, unified schema that captures everything needed
to understand a model's high-dimensional geometry. It unifies 18+ existing
profile types into one complete picture.

A ModelProfile answers: "What does this model look like on the inside?"

For any two models, the unified profile enables:
- Alignment assessment: How similar is their geometry?
- Alignment planning: What transformations are needed to merge?
- Capability mapping: Where does each model store what knowledge?
- Transfer prediction: What will survive a merge?

Schema: mc.model_profile.v1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import is_inf, is_nan

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.curvature_profile import (
        CurvatureProfile,
    )
    from modelcypher.core.domain.geometry.knowledge_density import ModelDensityProfile
    from modelcypher.core.domain.geometry.topological_fingerprint import Fingerprint
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "mc.model_profile.v1"


class ProfileSection(Enum):
    """Sections of a ModelProfile for partial computation."""

    IDENTITY = "identity"  # Always fast - just config.json
    GEOMETRY = "geometry"  # Curvature, intrinsic dimension
    TOPOLOGY = "topology"  # Persistent homology (expensive)
    SEMANTIC = "semantic"  # Semantic primes (requires inference)
    DENSITY = "density"  # Knowledge density (expensive)
    ENTROPY = "entropy"  # Entropy measurements


@dataclass
class ManifoldRegion:
    """A region of the manifold with consistent properties.

    Contains only raw measurements. The mean_entropy value is the raw
    measurement - callers interpret relative to baselines.
    """

    start_position: float  # 0.0-1.0 relative position
    end_position: float
    mean_entropy: float


@dataclass
class LayerProfile:
    """Complete geometric profile for a single transformer layer.

    Captures curvature, intrinsic dimension, entropy, topology, and stability
    metrics for a single layer. These are the building blocks of the full
    model profile.
    """

    layer_idx: int
    layer_name: str = ""  # e.g., "layers.0.self_attn"

    # === CURVATURE ===
    sectional_curvature_mean: float = 0.0
    sectional_curvature_std: float = 0.0
    ollivier_ricci_mean: float = 0.0
    ollivier_ricci_std: float = 0.0
    dominant_curvature_sign: str = "unknown"  # "positive", "negative", "flat", "mixed"

    # === INTRINSIC DIMENSION ===
    intrinsic_dimension: float = 0.0
    intrinsic_dimension_uncertainty: float = 0.0
    intrinsic_dimension_method: str = "mle"  # "mle", "correlation", "twonn"

    # === ENTROPY ===
    shannon_entropy: float | None = None
    renyi_entropy_alpha2: float | None = None

    # === TOPOLOGY ===
    betti_0: int | None = None  # Connected components
    betti_1: int | None = None  # Holes
    max_persistence: float | None = None

    # === STABILITY ===
    gradient_norm: float | None = None
    condition_number: float | None = None

    # === MANIFOLD STRUCTURE ===
    manifold_regions: list[ManifoldRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        _b = get_default_backend()

        def safe_float(v: float | None) -> float | None:
            if v is None:
                return None
            if is_nan(v, _b) or is_inf(v, _b):
                return None
            return v

        return {
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name,
            # Curvature
            "sectional_curvature_mean": safe_float(self.sectional_curvature_mean),
            "sectional_curvature_std": safe_float(self.sectional_curvature_std),
            "ollivier_ricci_mean": safe_float(self.ollivier_ricci_mean),
            "ollivier_ricci_std": safe_float(self.ollivier_ricci_std),
            "dominant_curvature_sign": self.dominant_curvature_sign,
            # Intrinsic dimension
            "intrinsic_dimension": safe_float(self.intrinsic_dimension),
            "intrinsic_dimension_uncertainty": safe_float(
                self.intrinsic_dimension_uncertainty
            ),
            "intrinsic_dimension_method": self.intrinsic_dimension_method,
            # Entropy
            "shannon_entropy": safe_float(self.shannon_entropy),
            "renyi_entropy_alpha2": safe_float(self.renyi_entropy_alpha2),
            # Topology
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "max_persistence": safe_float(self.max_persistence),
            # Stability
            "gradient_norm": safe_float(self.gradient_norm),
            "condition_number": safe_float(self.condition_number),
            # Manifold structure
            "manifold_regions": [
                {
                    "start_position": r.start_position,
                    "end_position": r.end_position,
                    "mean_entropy": r.mean_entropy,
                }
                for r in self.manifold_regions
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LayerProfile:
        """Create from dictionary."""

        def safe_get(key: str, default: float = 0.0) -> float:
            val = d.get(key, default)
            return default if val is None else val

        regions = [
            ManifoldRegion(
                start_position=r["start_position"],
                end_position=r["end_position"],
                mean_entropy=r["mean_entropy"],
            )
            for r in d.get("manifold_regions", [])
        ]

        return cls(
            layer_idx=d["layer_idx"],
            layer_name=d.get("layer_name", ""),
            sectional_curvature_mean=safe_get("sectional_curvature_mean", 0.0),
            sectional_curvature_std=safe_get("sectional_curvature_std", 0.0),
            ollivier_ricci_mean=safe_get("ollivier_ricci_mean", 0.0),
            ollivier_ricci_std=safe_get("ollivier_ricci_std", 0.0),
            dominant_curvature_sign=d.get("dominant_curvature_sign", "unknown"),
            intrinsic_dimension=safe_get("intrinsic_dimension", 0.0),
            intrinsic_dimension_uncertainty=safe_get(
                "intrinsic_dimension_uncertainty", 0.0
            ),
            intrinsic_dimension_method=d.get("intrinsic_dimension_method", "mle"),
            shannon_entropy=d.get("shannon_entropy"),
            renyi_entropy_alpha2=d.get("renyi_entropy_alpha2"),
            betti_0=d.get("betti_0"),
            betti_1=d.get("betti_1"),
            max_persistence=d.get("max_persistence"),
            gradient_norm=d.get("gradient_norm"),
            condition_number=d.get("condition_number"),
            manifold_regions=regions,
        )


@dataclass
class TopologySummary:
    """Summary of topological fingerprint for storage in ModelProfile."""

    component_count: int = 1
    cycle_count: int = 0
    average_persistence: float = 0.0
    max_persistence: float = 0.0
    persistence_entropy: float = 0.0
    betti_numbers: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_count": self.component_count,
            "cycle_count": self.cycle_count,
            "average_persistence": self.average_persistence,
            "max_persistence": self.max_persistence,
            "persistence_entropy": self.persistence_entropy,
            "betti_numbers": self.betti_numbers,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TopologySummary:
        return cls(
            component_count=d.get("component_count", 1),
            cycle_count=d.get("cycle_count", 0),
            average_persistence=d.get("average_persistence", 0.0),
            max_persistence=d.get("max_persistence", 0.0),
            persistence_entropy=d.get("persistence_entropy", 0.0),
            betti_numbers=d.get("betti_numbers", {}),
        )

    @classmethod
    def from_fingerprint(cls, fingerprint: "Fingerprint") -> TopologySummary:
        """Create from a TopologicalFingerprint's Fingerprint result."""
        return cls(
            component_count=fingerprint.summary.component_count,
            cycle_count=fingerprint.summary.cycle_count,
            average_persistence=fingerprint.summary.average_persistence,
            max_persistence=fingerprint.summary.max_persistence,
            persistence_entropy=fingerprint.summary.persistence_entropy,
            betti_numbers=fingerprint.betti_numbers.copy(),
        )


@dataclass
class SemanticSignature:
    """Summary of semantic prime signature for storage in ModelProfile."""

    vector: list[float] = field(default_factory=list)  # 65-dimensional
    dominant_primes: list[str] = field(default_factory=list)  # Top 5 semantic primes

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": self.vector,
            "dominant_primes": self.dominant_primes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SemanticSignature:
        return cls(
            vector=d.get("vector", []),
            dominant_primes=d.get("dominant_primes", []),
        )


@dataclass
class DensitySummary:
    """Summary of knowledge density for storage in ModelProfile."""

    overall_density: float = 0.0
    domain_densities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_density": self.overall_density,
            "domain_densities": self.domain_densities,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DensitySummary:
        return cls(
            overall_density=d.get("overall_density", 0.0),
            domain_densities=d.get("domain_densities", {}),
        )

    @classmethod
    def from_density_profile(cls, profile: "ModelDensityProfile") -> DensitySummary:
        """Create from a ModelDensityProfile."""
        return cls(
            overall_density=profile.overall_density,
            domain_densities=dict(profile.domain_densities),
        )


@dataclass
class ModelProfile:
    """Complete geometric profile of a model - the transparent black box.

    This is the unified schema that combines all 18+ existing profile types
    into a single, complete picture of a model's geometry.

    All fields are optional except identity fields, allowing partial profiles
    to be built incrementally as different sections are computed.
    """

    # === IDENTITY (always required) ===
    model_path: str
    profile_version: str = SCHEMA_VERSION
    computed_at: str = ""

    # === ARCHITECTURE (from config.json) ===
    model_family: str = "unknown"  # "llama", "qwen", "mistral", "smollm"
    architecture: str = "unknown"  # "llama", "qwen2", "mistral", etc.
    parameter_count: int = 0
    hidden_dim: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    vocab_size: int = 0

    # === LAYER-LEVEL GEOMETRY ===
    layer_profiles: list[LayerProfile] = field(default_factory=list)

    # === GLOBAL CURVATURE (aggregated from layer profiles) ===
    global_sectional_mean: float = 0.0
    global_sectional_std: float = 0.0
    global_ollivier_ricci_mean: float = 0.0
    global_ollivier_ricci_std: float = 0.0
    global_intrinsic_dimension_mean: float = 0.0

    # === TOPOLOGY ===
    topology_summary: TopologySummary | None = None

    # === SEMANTIC STRUCTURE ===
    semantic_signature: SemanticSignature | None = None

    # === KNOWLEDGE DENSITY ===
    density_summary: DensitySummary | None = None

    # === DOMAIN-SPECIFIC METRICS ===
    # Maps domain (spatial, social, temporal, moral) to domain-specific metrics
    domain_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # === COMPUTED SECTIONS (track what's been computed) ===
    computed_sections: list[str] = field(default_factory=list)

    # === METADATA ===
    probe_corpus_hash: str = ""  # Which probes generated this profile
    backend_used: str = ""  # "mlx", "jax", etc.
    extraction_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set computed_at if not provided."""
        if not self.computed_at:
            self.computed_at = datetime.now().isoformat()

    def has_section(self, section: ProfileSection) -> bool:
        """Check if a section has been computed."""
        return section.value in self.computed_sections

    def add_section(self, section: ProfileSection) -> None:
        """Mark a section as computed."""
        if section.value not in self.computed_sections:
            self.computed_sections.append(section.value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        _b = get_default_backend()

        def safe_float(v: float) -> float | None:
            if is_nan(v, _b) or is_inf(v, _b):
                return None
            return v

        result: dict[str, Any] = {
            "_schema": self.profile_version,
            # Identity
            "model_path": self.model_path,
            "profile_version": self.profile_version,
            "computed_at": self.computed_at,
            # Architecture
            "model_family": self.model_family,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "vocab_size": self.vocab_size,
            # Layer profiles
            "layer_profiles": [lp.to_dict() for lp in self.layer_profiles],
            # Global curvature
            "global_sectional_mean": safe_float(self.global_sectional_mean),
            "global_sectional_std": safe_float(self.global_sectional_std),
            "global_ollivier_ricci_mean": safe_float(self.global_ollivier_ricci_mean),
            "global_ollivier_ricci_std": safe_float(self.global_ollivier_ricci_std),
            "global_intrinsic_dimension_mean": safe_float(
                self.global_intrinsic_dimension_mean
            ),
            # Topology
            "topology_summary": (
                self.topology_summary.to_dict() if self.topology_summary else None
            ),
            # Semantic
            "semantic_signature": (
                self.semantic_signature.to_dict() if self.semantic_signature else None
            ),
            # Density
            "density_summary": (
                self.density_summary.to_dict() if self.density_summary else None
            ),
            # Domain-specific metrics
            "domain_metrics": self.domain_metrics,
            # Computed sections
            "computed_sections": self.computed_sections,
            # Metadata
            "probe_corpus_hash": self.probe_corpus_hash,
            "backend_used": self.backend_used,
            "extraction_config": self.extraction_config,
        }

        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelProfile:
        """Create from dictionary."""

        def safe_get(key: str, default: float = 0.0) -> float:
            val = d.get(key, default)
            return default if val is None else val

        layer_profiles = [
            LayerProfile.from_dict(lp) for lp in d.get("layer_profiles", [])
        ]

        topology = None
        if d.get("topology_summary"):
            topology = TopologySummary.from_dict(d["topology_summary"])

        semantic = None
        if d.get("semantic_signature"):
            semantic = SemanticSignature.from_dict(d["semantic_signature"])

        density = None
        if d.get("density_summary"):
            density = DensitySummary.from_dict(d["density_summary"])

        return cls(
            model_path=d["model_path"],
            profile_version=d.get("profile_version", SCHEMA_VERSION),
            computed_at=d.get("computed_at", ""),
            model_family=d.get("model_family", "unknown"),
            architecture=d.get("architecture", "unknown"),
            parameter_count=d.get("parameter_count", 0),
            hidden_dim=d.get("hidden_dim", 0),
            num_layers=d.get("num_layers", 0),
            num_attention_heads=d.get("num_attention_heads", 0),
            vocab_size=d.get("vocab_size", 0),
            layer_profiles=layer_profiles,
            global_sectional_mean=safe_get("global_sectional_mean", 0.0),
            global_sectional_std=safe_get("global_sectional_std", 0.0),
            global_ollivier_ricci_mean=safe_get("global_ollivier_ricci_mean", 0.0),
            global_ollivier_ricci_std=safe_get("global_ollivier_ricci_std", 0.0),
            global_intrinsic_dimension_mean=safe_get(
                "global_intrinsic_dimension_mean", 0.0
            ),
            topology_summary=topology,
            semantic_signature=semantic,
            density_summary=density,
            domain_metrics=d.get("domain_metrics", {}),
            computed_sections=d.get("computed_sections", []),
            probe_corpus_hash=d.get("probe_corpus_hash", ""),
            backend_used=d.get("backend_used", ""),
            extraction_config=d.get("extraction_config", {}),
        )

    def save(self, path: str | Path) -> None:
        """Save profile to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved model profile to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ModelProfile:
        """Load profile from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_curvature_profile(
        cls, curvature: "CurvatureProfile"
    ) -> ModelProfile:
        """Create ModelProfile from an existing CurvatureProfile.

        This is the primary import path for converting existing curvature
        profile data to the unified format.
        """
        layer_profiles = []
        for lc in curvature.layer_curvatures:
            layer_profiles.append(
                LayerProfile(
                    layer_idx=lc.layer_idx,
                    sectional_curvature_mean=lc.sectional_mean,
                    sectional_curvature_std=lc.sectional_std,
                    ollivier_ricci_mean=lc.ollivier_ricci_mean,
                    ollivier_ricci_std=lc.ollivier_ricci_std,
                    dominant_curvature_sign=lc.dominant_sign,
                    intrinsic_dimension=lc.intrinsic_dimension,
                    intrinsic_dimension_uncertainty=lc.intrinsic_dimension_uncertainty,
                )
            )

        profile = cls(
            model_path=curvature.model_path,
            model_family=curvature.model_family,
            num_layers=curvature.total_layers,
            layer_profiles=layer_profiles,
            global_sectional_mean=curvature.global_sectional_mean,
            global_sectional_std=curvature.global_sectional_std,
            global_ollivier_ricci_mean=curvature.global_ollivier_ricci_mean,
            global_ollivier_ricci_std=curvature.global_ollivier_ricci_std,
            global_intrinsic_dimension_mean=curvature.global_intrinsic_dimension_mean,
            computed_sections=[ProfileSection.GEOMETRY.value],
            extraction_config=curvature.extraction_config,
        )

        if curvature.extraction_date:
            profile.computed_at = curvature.extraction_date

        return profile

    def merge_with(self, other: ModelProfile) -> ModelProfile:
        """Merge another profile into this one, filling in missing sections.

        This allows building up a complete profile incrementally.
        """
        # Start with a copy of self
        result = ModelProfile.from_dict(self.to_dict())

        # Merge layer profiles (prefer other if we have no layers)
        if not result.layer_profiles and other.layer_profiles:
            result.layer_profiles = [
                LayerProfile.from_dict(lp.to_dict()) for lp in other.layer_profiles
            ]
        elif result.layer_profiles and other.layer_profiles:
            # Merge per-layer data
            other_by_idx = {lp.layer_idx: lp for lp in other.layer_profiles}
            merged_layers = []
            for lp in result.layer_profiles:
                if lp.layer_idx in other_by_idx:
                    olp = other_by_idx[lp.layer_idx]
                    # Merge fields from other if ours are default/empty
                    merged = LayerProfile(
                        layer_idx=lp.layer_idx,
                        layer_name=lp.layer_name or olp.layer_name,
                        sectional_curvature_mean=(
                            lp.sectional_curvature_mean
                            if lp.sectional_curvature_mean != 0.0
                            else olp.sectional_curvature_mean
                        ),
                        sectional_curvature_std=(
                            lp.sectional_curvature_std
                            if lp.sectional_curvature_std != 0.0
                            else olp.sectional_curvature_std
                        ),
                        ollivier_ricci_mean=(
                            lp.ollivier_ricci_mean
                            if lp.ollivier_ricci_mean != 0.0
                            else olp.ollivier_ricci_mean
                        ),
                        ollivier_ricci_std=(
                            lp.ollivier_ricci_std
                            if lp.ollivier_ricci_std != 0.0
                            else olp.ollivier_ricci_std
                        ),
                        dominant_curvature_sign=(
                            lp.dominant_curvature_sign
                            if lp.dominant_curvature_sign != "unknown"
                            else olp.dominant_curvature_sign
                        ),
                        intrinsic_dimension=(
                            lp.intrinsic_dimension
                            if lp.intrinsic_dimension != 0.0
                            else olp.intrinsic_dimension
                        ),
                        intrinsic_dimension_uncertainty=(
                            lp.intrinsic_dimension_uncertainty
                            if lp.intrinsic_dimension_uncertainty != 0.0
                            else olp.intrinsic_dimension_uncertainty
                        ),
                        intrinsic_dimension_method=(
                            lp.intrinsic_dimension_method
                            if lp.intrinsic_dimension_method != "mle"
                            else olp.intrinsic_dimension_method
                        ),
                        shannon_entropy=(
                            lp.shannon_entropy
                            if lp.shannon_entropy is not None
                            else olp.shannon_entropy
                        ),
                        renyi_entropy_alpha2=(
                            lp.renyi_entropy_alpha2
                            if lp.renyi_entropy_alpha2 is not None
                            else olp.renyi_entropy_alpha2
                        ),
                        betti_0=lp.betti_0 if lp.betti_0 is not None else olp.betti_0,
                        betti_1=lp.betti_1 if lp.betti_1 is not None else olp.betti_1,
                        max_persistence=(
                            lp.max_persistence
                            if lp.max_persistence is not None
                            else olp.max_persistence
                        ),
                        gradient_norm=(
                            lp.gradient_norm
                            if lp.gradient_norm is not None
                            else olp.gradient_norm
                        ),
                        condition_number=(
                            lp.condition_number
                            if lp.condition_number is not None
                            else olp.condition_number
                        ),
                        manifold_regions=(
                            lp.manifold_regions if lp.manifold_regions else olp.manifold_regions
                        ),
                    )
                    merged_layers.append(merged)
                else:
                    merged_layers.append(lp)
            result.layer_profiles = merged_layers

        # Merge optional sections
        if result.topology_summary is None and other.topology_summary is not None:
            result.topology_summary = TopologySummary.from_dict(
                other.topology_summary.to_dict()
            )

        if result.semantic_signature is None and other.semantic_signature is not None:
            result.semantic_signature = SemanticSignature.from_dict(
                other.semantic_signature.to_dict()
            )

        if result.density_summary is None and other.density_summary is not None:
            result.density_summary = DensitySummary.from_dict(
                other.density_summary.to_dict()
            )

        # Merge domain metrics (add domains from other that we don't have)
        for domain, metrics in other.domain_metrics.items():
            if domain not in result.domain_metrics:
                result.domain_metrics[domain] = metrics.copy()

        # Merge computed sections
        for section in other.computed_sections:
            if section not in result.computed_sections:
                result.computed_sections.append(section)

        # Merge architecture info if missing
        if result.architecture == "unknown" and other.architecture != "unknown":
            result.architecture = other.architecture
        if result.model_family == "unknown" and other.model_family != "unknown":
            result.model_family = other.model_family
        if result.parameter_count == 0:
            result.parameter_count = other.parameter_count
        if result.hidden_dim == 0:
            result.hidden_dim = other.hidden_dim
        if result.num_layers == 0:
            result.num_layers = other.num_layers
        if result.num_attention_heads == 0:
            result.num_attention_heads = other.num_attention_heads
        if result.vocab_size == 0:
            result.vocab_size = other.vocab_size

        return result


class ProfileRepository:
    """Repository for loading and saving model geometry profiles.

    Profiles are stored as JSON files in a directory structure:
        ~/.modelcypher/profiles/{model_family}_{model_size}.json

    This replaces the old BaselineRepository with a unified profile system.
    """

    def __init__(self, profile_dir: str | Path | None = None):
        """Initialize the repository.

        Args:
            profile_dir: Directory containing profile JSON files.
                        Defaults to ~/.modelcypher/profiles/
        """
        if profile_dir is None:
            profile_dir = Path.home() / ".modelcypher" / "profiles"
        self._profile_dir = Path(profile_dir)
        self._cache: dict[str, ModelProfile] = {}

    @property
    def profile_dir(self) -> Path:
        """Get the profile directory path."""
        return self._profile_dir

    def get_profile(
        self, model_family: str, model_size: str
    ) -> ModelProfile | None:
        """Get a profile by family and size.

        Args:
            model_family: e.g., "qwen", "llama", "mistral"
            model_size: e.g., "0.5B", "3B", "7B"

        Returns:
            ModelProfile if found, None otherwise
        """
        cache_key = f"{model_family}_{model_size}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try to load from file
        filename = f"{model_family}_{model_size}.json"
        filepath = self._profile_dir / filename

        if filepath.exists():
            profile = ModelProfile.load(filepath)
            self._cache[cache_key] = profile
            return profile

        return None

    def get_profiles_for_family(self, model_family: str) -> list[ModelProfile]:
        """Get all profiles for a given model family."""
        profiles = []

        if not self._profile_dir.exists():
            return profiles

        for filepath in self._profile_dir.glob(f"{model_family}_*.json"):
            try:
                profile = ModelProfile.load(filepath)
                profiles.append(profile)
            except Exception as e:
                logger.warning("Failed to load profile %s: %s", filepath, e)

        return profiles

    def get_all_profiles(self) -> list[ModelProfile]:
        """Get all available profiles."""
        profiles = []

        if not self._profile_dir.exists():
            return profiles

        for filepath in self._profile_dir.glob("*.json"):
            try:
                profile = ModelProfile.load(filepath)
                profiles.append(profile)
            except Exception as e:
                logger.warning("Failed to load profile %s: %s", filepath, e)

        return profiles

    def save_profile(self, profile: ModelProfile) -> Path:
        """Save a profile to the repository.

        Returns the path where the profile was saved.
        """
        self._profile_dir.mkdir(parents=True, exist_ok=True)

        # Normalize family and size for filename
        family = profile.model_family.lower().replace(" ", "_")
        size = self._extract_size(profile)

        filename = f"{family}_{size}.json"
        filepath = self._profile_dir / filename

        profile.save(filepath)

        # Update cache
        cache_key = f"{family}_{size}"
        self._cache[cache_key] = profile

        return filepath

    def _extract_size(self, profile: ModelProfile) -> str:
        """Extract model size string from profile."""
        # Try to get from path name
        path = Path(profile.model_path)
        name = path.name.lower()

        for pattern in ["0.5b", "1b", "1.5b", "3b", "7b", "8b", "13b", "70b"]:
            if pattern in name:
                return pattern.upper()

        # Fall back to parameter count
        params = profile.parameter_count
        if params >= 60_000_000_000:
            return "70B"
        elif params >= 10_000_000_000:
            return "13B"
        elif params >= 6_000_000_000:
            return "7B"
        elif params >= 2_000_000_000:
            return "3B"
        elif params >= 1_000_000_000:
            return "1B"
        elif params >= 400_000_000:
            return "0.5B"
        else:
            return "unknown"

    def find_matching_profile(
        self, model_family: str, model_size: str
    ) -> ModelProfile | None:
        """Find the closest matching profile for a model.

        First tries exact match, then falls back to same family.
        """
        # Exact match
        profile = self.get_profile(model_family, model_size)
        if profile:
            return profile

        # Same family, any size
        family_profiles = self.get_profiles_for_family(model_family)
        if family_profiles:
            return family_profiles[0]

        return None


class ModelProfileExtractor:
    """Extract complete geometry profiles from models.

    This is the unified extraction system that replaces domain_geometry_baselines.
    It computes curvature, intrinsic dimension, and domain-specific metrics.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        model_loader: "ModelLoaderPort | None" = None,
        activation_provider: Any | None = None,
    ):
        from modelcypher.core.domain._backend import get_default_backend

        self._backend = backend or get_default_backend()
        self._model_loader = model_loader
        self._activation_provider = activation_provider

    def extract_profile(
        self,
        model_path: str,
        layers: list[int] | None = None,
    ) -> ModelProfile:
        """Extract a geometry profile from a model.

        Args:
            model_path: Path to the model directory
            layers: Specific layers to analyze (None = sample layers)

        Returns:
            ModelProfile with computed metrics

        Note:
            k for k-NN graph is computed from data, not user-specified.
            Domain metrics are computed from actual geometry, not placeholders.
        """
        from modelcypher.core.domain.geometry.manifold_curvature import (
            OllivierRicciConfig,
            OllivierRicciCurvature,
        )

        logger.info("Extracting geometry profile from %s", model_path)

        # Parse model info from path
        model_family, model_size = self._parse_model_info(model_path)

        # Get probes and collect activations
        probes = self._get_probes()
        logger.debug("Using %d probes for geometry extraction", len(probes))

        # Collect activations
        activations_by_layer = self._collect_activations(model_path, probes, layers)

        if not activations_by_layer:
            logger.warning("No activations collected from %s", model_path)
            return ModelProfile(
                model_path=model_path,
                model_family=model_family,
                extraction_config={"error": "no_activations"},
            )

        layer_profiles: list[LayerProfile] = []
        ricci_values: list[float] = []
        id_values: list[float] = []

        for layer_idx, activations in sorted(activations_by_layer.items()):
            try:
                # Compute intrinsic dimension FIRST - this determines k
                id_value, id_std = self._compute_intrinsic_dimension(activations)

                # k is derived from geometry: k = 2 * intrinsic_dimension (minimum for manifold connectivity)
                # No arbitrary fallbacks - if ID estimation fails, we need to know
                if id_value <= 0 or is_nan(id_value, self._backend):
                    raise ValueError(
                        f"Intrinsic dimension estimation failed for layer {layer_idx}: "
                        f"id_value={id_value}, std={id_std}. "
                        "This indicates degenerate or insufficient activation data. "
                        "Check that the layer has sufficient variance and enough samples."
                    )
                k = max(3, int(2 * id_value))

                # Compute Ollivier-Ricci curvature with computed k
                orc = OllivierRicciCurvature(
                    config=OllivierRicciConfig(k_neighbors=k),
                    backend=self._backend,
                )
                result = orc.compute(activations, k_neighbors=k)
                curvature = result.mean_edge_curvature

                # Skip NaN values
                if is_nan(curvature, self._backend):
                    logger.debug("Layer %d returned NaN curvature, skipping", layer_idx)
                    continue

                lp = LayerProfile(
                    layer_idx=layer_idx,
                    ollivier_ricci_mean=curvature,
                    ollivier_ricci_std=result.curvature_std,
                    intrinsic_dimension=id_value,
                    intrinsic_dimension_uncertainty=id_std,
                )
                layer_profiles.append(lp)
                ricci_values.append(curvature)
                if not is_nan(id_value, self._backend):
                    id_values.append(id_value)

            except Exception as e:
                logger.warning("Failed to compute metrics for layer %d: %s", layer_idx, e)
                continue

        if not layer_profiles:
            logger.warning("No valid layer profiles computed")
            return ModelProfile(
                model_path=model_path,
                model_family=model_family,
                extraction_config={"error": "no_valid_layers"},
            )

        # Compute global statistics
        b = self._backend
        ricci_arr = b.array(ricci_values)

        ricci_mean_arr = b.mean(ricci_arr)
        ricci_std_arr = b.std(ricci_arr)
        b.eval(ricci_mean_arr, ricci_std_arr)
        global_ricci_mean = float(b.to_scalar(ricci_mean_arr))
        global_ricci_std = float(b.to_scalar(ricci_std_arr))
        global_id_mean = float(sum(id_values) / len(id_values)) if id_values else 0.0

        return ModelProfile(
            model_path=model_path,
            model_family=model_family,
            layer_profiles=layer_profiles,
            global_ollivier_ricci_mean=global_ricci_mean,
            global_ollivier_ricci_std=global_ricci_std,
            global_intrinsic_dimension_mean=global_id_mean,
            computed_sections=[ProfileSection.GEOMETRY.value],
            backend_used=type(self._backend).__name__,
            extraction_config={
                "num_probes": len(probes),
                "layers_analyzed": len(layer_profiles),
            },
        )

    def _parse_model_info(self, model_path: str) -> tuple[str, str]:
        """Parse model family and size from path."""
        path = Path(model_path)
        name = path.name.lower()

        # Detect family
        if "qwen" in name:
            family = "qwen"
        elif "llama" in name:
            family = "llama"
        elif "mistral" in name:
            family = "mistral"
        elif "phi" in name:
            family = "phi"
        elif "gemma" in name:
            family = "gemma"
        elif "smol" in name:
            family = "smollm"
        else:
            family = "unknown"

        # Detect size
        size = "unknown"
        for pattern in ["0.5b", "1b", "1.5b", "3b", "7b", "8b", "13b", "70b"]:
            if pattern in name:
                size = pattern.upper()
                break

        return family, size

    def _get_probes(self) -> list[str]:
        """Get probe prompts for geometry measurement."""
        from modelcypher.core.domain.geometry.atlas_registry import get_atlas_probes

        probes = list(get_atlas_probes())
        if not probes:
            raise ValueError(
                "No atlas probes registered. Call "
                "modelcypher.core.use_cases.atlas_bootstrap.register_default_atlas_inventories() "
                "before extracting geometry profiles."
            )

        prompts: list[str] = []
        for probe in probes:
            prompts.append(f"The concept of {probe.name}.")
            for text in probe.support_texts:
                if text and len(text) > 3:
                    prompts.append(text)

        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        return unique

    def _collect_activations(
        self,
        model_path: str,
        probes: list[str],
        layers: list[int] | None,
    ) -> dict[int, "Array"]:
        """Collect activations from a model for given probes.

        Uses ActivationProvider for platform-agnostic activation collection.
        """
        if self._model_loader is None:
            raise RuntimeError(
                "ModelProfileExtractor requires a model_loader. "
                "Pass a ModelLoaderPort implementation to the constructor."
            )

        if self._activation_provider is None:
            raise RuntimeError(
                "ModelProfileExtractor requires an activation_provider. "
                "Pass an ActivationProvider implementation to the constructor."
            )

        model, tokenizer = self._model_loader.load_model_for_training(model_path)

        # Determine which layers to analyze
        if layers is None:
            # Get layer count from model
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                total_layers = len(model.model.layers)
            elif hasattr(model, "layers"):
                total_layers = len(model.layers)
            else:
                total_layers = 24
            layers = list(range(0, total_layers, max(1, total_layers // 8)))

        activations_by_layer: dict[int, list["Array"]] = {l: [] for l in layers}

        for probe in probes:
            try:
                # Collect all layer activations in one forward pass
                all_acts = self._activation_provider.collect_hidden_activations(
                    model, tokenizer, probe
                )

                # Filter to requested layers
                for layer_idx in layers:
                    if layer_idx in all_acts:
                        activations_by_layer[layer_idx].append(all_acts[layer_idx])
            except Exception as e:
                logger.debug("Failed to get activation for probe: %s", e)
                continue

        # Stack activations per layer
        result = {}
        b = self._backend
        for layer_idx, acts in activations_by_layer.items():
            if acts:
                stacked = b.stack(acts, axis=0)
                result[layer_idx] = stacked

        return result

    def _compute_intrinsic_dimension(
        self, activations: "Array"
    ) -> tuple[float, float]:
        """Compute intrinsic dimension of the activation manifold."""
        try:
            from modelcypher.core.domain.geometry.intrinsic_dimension import (
                IntrinsicDimension,
            )

            estimator = IntrinsicDimension(backend=self._backend)
            global_estimate = estimator.compute(activations)
            local_map = estimator.local_dimension_map(activations)

            return global_estimate.intrinsic_dimension, local_map.std_dimension
        except Exception as e:
            logger.debug("ID estimation failed: %s", e)
            return 0.0, 0.0

__all__ = [
    "ProfileSection",
    "ManifoldRegion",
    "LayerProfile",
    "TopologySummary",
    "SemanticSignature",
    "DensitySummary",
    "ModelProfile",
    "ProfileRepository",
    "ModelProfileExtractor",
    "SCHEMA_VERSION",
]
