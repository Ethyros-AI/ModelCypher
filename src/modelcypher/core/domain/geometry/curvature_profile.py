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

"""Curvature profiles for model family baselines.

This module provides infrastructure for:
1. Computing per-layer curvature profiles for models
2. Aggregating profiles into family baselines
3. Computing curvature compatibility for merge decisions

Key insight: Knowledge density profiles tell us WHAT is encoded (semantics),
while curvature profiles tell us HOW it's encoded (geometry/manifold shape).
These are complementary measurements for understanding model representations.

Schema: mc.geometry.research.curvature_profile.v1
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "mc.geometry.research.curvature_profile.v1"


@dataclass
class LayerCurvature:
    """Curvature measurements for a single layer."""

    layer_idx: int

    # Sectional curvature (Christoffel → Riemann tensor)
    sectional_mean: float = 0.0
    sectional_std: float = 0.0
    sectional_min: float = 0.0
    sectional_max: float = 0.0
    dominant_sign: str = "unknown"  # positive, negative, flat, mixed

    # Ollivier-Ricci curvature (optimal transport on k-NN graph)
    ollivier_ricci_mean: float = 0.0
    ollivier_ricci_std: float = 0.0

    # Intrinsic dimension
    intrinsic_dimension: float = 0.0
    intrinsic_dimension_uncertainty: float = 0.0

    # Manifold health classification (for reference only)
    manifold_health: str = "unknown"  # healthy, degenerate, collapsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "sectional_mean": self.sectional_mean,
            "sectional_std": self.sectional_std,
            "sectional_min": self.sectional_min,
            "sectional_max": self.sectional_max,
            "dominant_sign": self.dominant_sign,
            "ollivier_ricci_mean": self.ollivier_ricci_mean,
            "ollivier_ricci_std": self.ollivier_ricci_std,
            "intrinsic_dimension": self.intrinsic_dimension,
            "intrinsic_dimension_uncertainty": self.intrinsic_dimension_uncertainty,
            "manifold_health": self.manifold_health,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LayerCurvature:
        return cls(
            layer_idx=d["layer_idx"],
            sectional_mean=d.get("sectional_mean", 0.0),
            sectional_std=d.get("sectional_std", 0.0),
            sectional_min=d.get("sectional_min", 0.0),
            sectional_max=d.get("sectional_max", 0.0),
            dominant_sign=d.get("dominant_sign", "unknown"),
            ollivier_ricci_mean=d.get("ollivier_ricci_mean", 0.0),
            ollivier_ricci_std=d.get("ollivier_ricci_std", 0.0),
            intrinsic_dimension=d.get("intrinsic_dimension", 0.0),
            intrinsic_dimension_uncertainty=d.get("intrinsic_dimension_uncertainty", 0.0),
            manifold_health=d.get("manifold_health", "unknown"),
        )


@dataclass
class CurvatureProfile:
    """Complete curvature profile for a model.

    This captures the geometric "shape" of the model's representation space
    across all layers, enabling:
    - Comparison of geometric structure between models
    - Family baseline computation
    - Curvature compatibility for merge decisions
    """

    # Model identification
    model_path: str
    model_family: str  # qwen, llama, mistral, etc.
    model_size: str  # 0.5B, 3B, 7B, etc.

    # Per-layer curvature
    layer_curvatures: list[LayerCurvature] = field(default_factory=list)
    total_layers: int = 0

    # Global statistics (aggregated across layers)
    global_sectional_mean: float = 0.0
    global_sectional_std: float = 0.0
    global_ollivier_ricci_mean: float = 0.0
    global_ollivier_ricci_std: float = 0.0
    global_intrinsic_dimension_mean: float = 0.0

    # Metadata
    extraction_date: str = ""
    extraction_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "_schema": SCHEMA_VERSION,
            "model_path": self.model_path,
            "model_family": self.model_family,
            "model_size": self.model_size,
            "layer_curvatures": [lc.to_dict() for lc in self.layer_curvatures],
            "total_layers": self.total_layers,
            "global_sectional_mean": self.global_sectional_mean,
            "global_sectional_std": self.global_sectional_std,
            "global_ollivier_ricci_mean": self.global_ollivier_ricci_mean,
            "global_ollivier_ricci_std": self.global_ollivier_ricci_std,
            "global_intrinsic_dimension_mean": self.global_intrinsic_dimension_mean,
            "extraction_date": self.extraction_date,
            "extraction_config": self.extraction_config,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CurvatureProfile:
        return cls(
            model_path=d["model_path"],
            model_family=d["model_family"],
            model_size=d["model_size"],
            layer_curvatures=[LayerCurvature.from_dict(lc) for lc in d.get("layer_curvatures", [])],
            total_layers=d.get("total_layers", 0),
            global_sectional_mean=d.get("global_sectional_mean", 0.0),
            global_sectional_std=d.get("global_sectional_std", 0.0),
            global_ollivier_ricci_mean=d.get("global_ollivier_ricci_mean", 0.0),
            global_ollivier_ricci_std=d.get("global_ollivier_ricci_std", 0.0),
            global_intrinsic_dimension_mean=d.get("global_intrinsic_dimension_mean", 0.0),
            extraction_date=d.get("extraction_date", ""),
            extraction_config=d.get("extraction_config", {}),
        )

    def save(self, path: str | Path) -> None:
        """Save profile to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved curvature profile to {path}")

    @classmethod
    def load(cls, path: str | Path) -> CurvatureProfile:
        """Load profile from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class FamilyBaseline:
    """Aggregated curvature baseline for a model family.

    Built from multiple models in the same family (e.g., Qwen 0.5B, 3B, 7B).
    Enables z-score comparisons for new models in the family.
    """

    family: str  # qwen, llama, mistral, etc.

    # Per-layer statistics (indexed by relative layer position 0.0-1.0)
    # This allows comparison across models with different layer counts
    layer_positions: list[float] = field(default_factory=list)  # [0.0, 0.1, ..., 1.0]

    # Sectional curvature baseline per position
    sectional_mean_by_position: list[float] = field(default_factory=list)
    sectional_std_by_position: list[float] = field(default_factory=list)

    # Ollivier-Ricci baseline per position
    ollivier_ricci_mean_by_position: list[float] = field(default_factory=list)
    ollivier_ricci_std_by_position: list[float] = field(default_factory=list)

    # Intrinsic dimension baseline per position
    intrinsic_dimension_by_position: list[float] = field(default_factory=list)

    # Contributing models
    contributing_models: list[str] = field(default_factory=list)
    sample_count: int = 0

    # Metadata
    created_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "_schema": "mc.geometry.research.family_baseline.v1",
            "family": self.family,
            "layer_positions": self.layer_positions,
            "sectional_mean_by_position": self.sectional_mean_by_position,
            "sectional_std_by_position": self.sectional_std_by_position,
            "ollivier_ricci_mean_by_position": self.ollivier_ricci_mean_by_position,
            "ollivier_ricci_std_by_position": self.ollivier_ricci_std_by_position,
            "intrinsic_dimension_by_position": self.intrinsic_dimension_by_position,
            "contributing_models": self.contributing_models,
            "sample_count": self.sample_count,
            "created_date": self.created_date,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FamilyBaseline:
        return cls(
            family=d["family"],
            layer_positions=d.get("layer_positions", []),
            sectional_mean_by_position=d.get("sectional_mean_by_position", []),
            sectional_std_by_position=d.get("sectional_std_by_position", []),
            ollivier_ricci_mean_by_position=d.get("ollivier_ricci_mean_by_position", []),
            ollivier_ricci_std_by_position=d.get("ollivier_ricci_std_by_position", []),
            intrinsic_dimension_by_position=d.get("intrinsic_dimension_by_position", []),
            contributing_models=d.get("contributing_models", []),
            sample_count=d.get("sample_count", 0),
            created_date=d.get("created_date", ""),
        )

    def save(self, path: str | Path) -> None:
        """Save baseline to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved family baseline to {path}")

    @classmethod
    def load(cls, path: str | Path) -> FamilyBaseline:
        """Load baseline from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass(frozen=True)
class CurvatureCompatibility:
    """Curvature compatibility score between two models.

    Uses z-scores relative to family baseline, NOT absolute thresholds.
    Score of 1.0 = perfect match, 0.0 = 3σ or more divergence.
    """

    # Overall compatibility (0.0 - 1.0)
    score: float

    # Component scores
    sectional_compatibility: float
    ollivier_ricci_compatibility: float
    intrinsic_dimension_compatibility: float

    # Raw z-scores for transparency
    sectional_z_score: float
    ollivier_ricci_z_score: float
    intrinsic_dimension_z_score: float

    # Baseline used
    baseline_family: str
    baseline_model_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "sectional_compatibility": self.sectional_compatibility,
            "ollivier_ricci_compatibility": self.ollivier_ricci_compatibility,
            "intrinsic_dimension_compatibility": self.intrinsic_dimension_compatibility,
            "sectional_z_score": self.sectional_z_score,
            "ollivier_ricci_z_score": self.ollivier_ricci_z_score,
            "intrinsic_dimension_z_score": self.intrinsic_dimension_z_score,
            "baseline_family": self.baseline_family,
            "baseline_model_count": self.baseline_model_count,
        }


def compute_curvature_compatibility(
    source_profile: CurvatureProfile,
    target_profile: CurvatureProfile,
    baseline: FamilyBaseline | None = None,
) -> CurvatureCompatibility:
    """Compute curvature compatibility between two models.

    Uses z-scores relative to family baseline when available.
    Falls back to direct comparison when no baseline exists.

    Args:
        source_profile: Curvature profile of source model
        target_profile: Curvature profile of target model
        baseline: Optional family baseline for z-score computation

    Returns:
        CurvatureCompatibility with score and component details
    """
    # Direct comparison values
    sectional_diff = abs(
        source_profile.global_sectional_mean - target_profile.global_sectional_mean
    )
    ricci_diff = abs(
        source_profile.global_ollivier_ricci_mean - target_profile.global_ollivier_ricci_mean
    )
    dim_diff = abs(
        source_profile.global_intrinsic_dimension_mean
        - target_profile.global_intrinsic_dimension_mean
    )

    if baseline is not None and baseline.sample_count > 1:
        # Z-score computation relative to baseline
        baseline_sectional_std = _safe_mean(baseline.sectional_std_by_position) or 0.1
        baseline_ricci_std = _safe_mean(baseline.ollivier_ricci_std_by_position) or 0.1
        baseline_dim_std = 1.0  # Default for dimension

        sectional_z = sectional_diff / baseline_sectional_std
        ricci_z = ricci_diff / baseline_ricci_std
        dim_z = dim_diff / baseline_dim_std

        baseline_family = baseline.family
        baseline_model_count = baseline.sample_count
    else:
        # No baseline: use source profile std as reference
        sectional_z = sectional_diff / max(source_profile.global_sectional_std, 0.01)
        ricci_z = ricci_diff / max(source_profile.global_ollivier_ricci_std, 0.01)
        dim_z = dim_diff / 1.0

        baseline_family = "none"
        baseline_model_count = 0

    # Convert z-scores to compatibility (1.0 at z=0, 0.0 at z>=3)
    sectional_compat = max(0.0, 1.0 - sectional_z / 3.0)
    ricci_compat = max(0.0, 1.0 - ricci_z / 3.0)
    dim_compat = max(0.0, 1.0 - dim_z / 3.0)

    # Overall score: weighted average
    # Ollivier-Ricci is most important for manifold health
    overall = 0.5 * ricci_compat + 0.3 * sectional_compat + 0.2 * dim_compat

    return CurvatureCompatibility(
        score=overall,
        sectional_compatibility=sectional_compat,
        ollivier_ricci_compatibility=ricci_compat,
        intrinsic_dimension_compatibility=dim_compat,
        sectional_z_score=sectional_z,
        ollivier_ricci_z_score=ricci_z,
        intrinsic_dimension_z_score=dim_z,
        baseline_family=baseline_family,
        baseline_model_count=baseline_model_count,
    )


def build_family_baseline(
    profiles: list[CurvatureProfile],
    family: str,
    num_positions: int = 11,
) -> FamilyBaseline:
    """Build a family baseline from multiple curvature profiles.

    Aggregates per-layer curvature from models with potentially different
    layer counts by normalizing to relative positions (0.0 to 1.0).

    Args:
        profiles: List of CurvatureProfile from same family
        family: Family name (qwen, llama, etc.)
        num_positions: Number of positions to sample (default 11 = 0.0, 0.1, ..., 1.0)

    Returns:
        FamilyBaseline with aggregated statistics
    """
    if not profiles:
        return FamilyBaseline(
            family=family,
            layer_positions=[i / (num_positions - 1) for i in range(num_positions)],
            created_date=datetime.now().isoformat(),
        )

    positions = [i / (num_positions - 1) for i in range(num_positions)]

    # Collect values at each position across all models
    sectional_values_by_pos: list[list[float]] = [[] for _ in positions]
    ricci_values_by_pos: list[list[float]] = [[] for _ in positions]
    dim_values_by_pos: list[list[float]] = [[] for _ in positions]

    for profile in profiles:
        if not profile.layer_curvatures:
            continue

        num_layers = profile.total_layers or len(profile.layer_curvatures)
        if num_layers < 2:
            continue

        for pos_idx, pos in enumerate(positions):
            # Map position to layer index
            layer_idx = int(pos * (num_layers - 1))
            layer_idx = min(layer_idx, len(profile.layer_curvatures) - 1)

            lc = profile.layer_curvatures[layer_idx]

            if not math.isnan(lc.sectional_mean):
                sectional_values_by_pos[pos_idx].append(lc.sectional_mean)
            if not math.isnan(lc.ollivier_ricci_mean):
                ricci_values_by_pos[pos_idx].append(lc.ollivier_ricci_mean)
            if not math.isnan(lc.intrinsic_dimension):
                dim_values_by_pos[pos_idx].append(lc.intrinsic_dimension)

    # Compute mean and std at each position
    sectional_means = [_safe_mean(vals) for vals in sectional_values_by_pos]
    sectional_stds = [_safe_std(vals) for vals in sectional_values_by_pos]
    ricci_means = [_safe_mean(vals) for vals in ricci_values_by_pos]
    ricci_stds = [_safe_std(vals) for vals in ricci_values_by_pos]
    dim_means = [_safe_mean(vals) for vals in dim_values_by_pos]

    return FamilyBaseline(
        family=family,
        layer_positions=positions,
        sectional_mean_by_position=sectional_means,
        sectional_std_by_position=sectional_stds,
        ollivier_ricci_mean_by_position=ricci_means,
        ollivier_ricci_std_by_position=ricci_stds,
        intrinsic_dimension_by_position=dim_means,
        contributing_models=[p.model_path for p in profiles],
        sample_count=len(profiles),
        created_date=datetime.now().isoformat(),
    )


def _safe_mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float:
    """Compute std, returning 0.0 for empty or single-element lists."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def parse_model_info(model_path: str) -> tuple[str, str]:
    """Parse model family and size from path.

    Examples:
        /path/to/Qwen2.5-0.5B-Instruct-bf16 -> ("qwen", "0.5B")
        /path/to/Llama-3.2-3B-Instruct-4bit -> ("llama", "3B")
    """
    path = Path(model_path)
    name = path.name.lower()

    # Detect family
    if "qwen" in name:
        family = "qwen"
    elif "llama" in name:
        family = "llama"
    elif "mistral" in name or "mathstral" in name:
        family = "mistral"
    elif "phi" in name:
        family = "phi"
    elif "gemma" in name:
        family = "gemma"
    elif "smol" in name:
        family = "smollm"
    elif "granite" in name:
        family = "granite"
    else:
        family = "unknown"

    # Detect size
    size = "unknown"
    for pattern in ["0.5b", "0.6b", "1b", "1.2b", "1.5b", "3b", "7b", "8b", "13b", "70b", "72b"]:
        if pattern in name:
            size = pattern.upper()
            break

    return family, size
