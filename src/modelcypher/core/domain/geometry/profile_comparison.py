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

"""Profile comparison for alignment analysis.

Two profiles together tell the alignment story:
- What transformations are needed to merge?
- How aligned is the geometry?
- What capabilities might be at risk?

This module computes the comparison without making judgment calls.
The geometry speaks for itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    exp_scalar,
    log_scalar,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.model_profile import (
    LayerProfile,
    ModelProfile,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.curvature_profile import FamilyBaseline

logger = logging.getLogger(__name__)


@dataclass
class LayerComparison:
    """Comparison metrics for a single layer pair."""

    source_layer_idx: int
    target_layer_idx: int

    # Curvature differences
    sectional_curvature_diff: float = 0.0
    ollivier_ricci_diff: float = 0.0

    # Dimension differences
    intrinsic_dimension_diff: float = 0.0
    dimension_ratio: float = 1.0  # target/source

    # Entropy differences (if available)
    shannon_entropy_diff: float | None = None

    # Topology differences (if available)
    betti_0_diff: int | None = None
    betti_1_diff: int | None = None

    # Alignment metrics
    alignment_effort: float = 0.0  # 0-1, how much transformation needed
    curvature_sign_match: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_layer_idx": self.source_layer_idx,
            "target_layer_idx": self.target_layer_idx,
            "sectional_curvature_diff": self.sectional_curvature_diff,
            "ollivier_ricci_diff": self.ollivier_ricci_diff,
            "intrinsic_dimension_diff": self.intrinsic_dimension_diff,
            "dimension_ratio": self.dimension_ratio,
            "shannon_entropy_diff": self.shannon_entropy_diff,
            "betti_0_diff": self.betti_0_diff,
            "betti_1_diff": self.betti_1_diff,
            "alignment_effort": self.alignment_effort,
            "curvature_sign_match": self.curvature_sign_match,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LayerComparison:
        return cls(
            source_layer_idx=d["source_layer_idx"],
            target_layer_idx=d["target_layer_idx"],
            sectional_curvature_diff=d.get("sectional_curvature_diff", 0.0),
            ollivier_ricci_diff=d.get("ollivier_ricci_diff", 0.0),
            intrinsic_dimension_diff=d.get("intrinsic_dimension_diff", 0.0),
            dimension_ratio=d.get("dimension_ratio", 1.0),
            shannon_entropy_diff=d.get("shannon_entropy_diff"),
            betti_0_diff=d.get("betti_0_diff"),
            betti_1_diff=d.get("betti_1_diff"),
            alignment_effort=d.get("alignment_effort", 0.0),
            curvature_sign_match=d.get("curvature_sign_match", True),
        )


@dataclass
class ProfileComparison:
    """What the geometry says about aligning two models.

    This is the "alignment story" - computed metrics about how two models
    relate to each other geometrically. No interpretation strings,
    just raw measurements.
    """

    source_path: str
    target_path: str

    # === STRUCTURAL METRICS ===
    architecture_match: bool = False  # Same base architecture?
    hidden_dim_ratio: float = 1.0  # target/source
    layer_count_ratio: float = 1.0  # target/source
    vocab_overlap: float = 0.0  # Shared vocabulary percentage (if computed)

    # === GEOMETRIC METRICS ===
    # Curvature alignment (0-1 scale derived from z-score differences)
    curvature_alignment: float = 0.0

    # Ollivier-Ricci alignment
    ricci_alignment: float = 0.0

    # Intrinsic dimension alignment
    dimension_alignment: float = 0.0

    # Overall geometric alignment (weighted combination)
    overall_alignment: float = 0.0

    # === TOPOLOGY METRICS ===
    topology_similarity: float | None = None  # If topology summaries available

    # === SEMANTIC METRICS ===
    semantic_alignment: float | None = None  # Cosine similarity of semantic signatures

    # === LAYER CORRESPONDENCE ===
    layer_mapping: dict[int, int] = field(default_factory=dict)  # source -> target
    layer_comparisons: list[LayerComparison] = field(default_factory=list)

    # === ALIGNMENT SUMMARY ===
    total_alignment_effort: float = 0.0
    mean_alignment_effort: float = 0.0
    max_alignment_effort: float = 0.0

    # === BASELINE-RELATIVE Z-SCORES ===
    # These are only populated when a FamilyBaseline is provided
    sectional_z_score: float | None = None  # Mean z-score across layers
    ricci_z_score: float | None = None  # Mean z-score for Ollivier-Ricci
    dimension_z_score: float | None = None  # Mean z-score for intrinsic dimension
    baseline_family: str | None = None  # Family used for baseline
    baseline_model_count: int | None = None  # Number of models in baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            # Structural
            "architecture_match": self.architecture_match,
            "hidden_dim_ratio": self.hidden_dim_ratio,
            "layer_count_ratio": self.layer_count_ratio,
            "vocab_overlap": self.vocab_overlap,
            # Geometric
            "curvature_alignment": self.curvature_alignment,
            "ricci_alignment": self.ricci_alignment,
            "dimension_alignment": self.dimension_alignment,
            "overall_alignment": self.overall_alignment,
            # Topology
            "topology_similarity": self.topology_similarity,
            # Semantic
            "semantic_alignment": self.semantic_alignment,
            # Layer correspondence
            "layer_mapping": self.layer_mapping,
            "layer_comparisons": [lc.to_dict() for lc in self.layer_comparisons],
            # Alignment
            "total_alignment_effort": self.total_alignment_effort,
            "mean_alignment_effort": self.mean_alignment_effort,
            "max_alignment_effort": self.max_alignment_effort,
            # Baseline-relative z-scores
            "sectional_z_score": self.sectional_z_score,
            "ricci_z_score": self.ricci_z_score,
            "dimension_z_score": self.dimension_z_score,
            "baseline_family": self.baseline_family,
            "baseline_model_count": self.baseline_model_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProfileComparison:
        return cls(
            source_path=d["source_path"],
            target_path=d["target_path"],
            architecture_match=d.get("architecture_match", False),
            hidden_dim_ratio=d.get("hidden_dim_ratio", 1.0),
            layer_count_ratio=d.get("layer_count_ratio", 1.0),
            vocab_overlap=d.get("vocab_overlap", 0.0),
            curvature_alignment=d.get("curvature_alignment", 0.0),
            ricci_alignment=d.get("ricci_alignment", 0.0),
            dimension_alignment=d.get("dimension_alignment", 0.0),
            overall_alignment=d.get("overall_alignment", 0.0),
            topology_similarity=d.get("topology_similarity"),
            semantic_alignment=d.get("semantic_alignment"),
            layer_mapping=d.get("layer_mapping", {}),
            layer_comparisons=[
                LayerComparison.from_dict(lc)
                for lc in d.get("layer_comparisons", [])
            ],
            total_alignment_effort=d.get("total_alignment_effort", 0.0),
            mean_alignment_effort=d.get("mean_alignment_effort", 0.0),
            max_alignment_effort=d.get("max_alignment_effort", 0.0),
            sectional_z_score=d.get("sectional_z_score"),
            ricci_z_score=d.get("ricci_z_score"),
            dimension_z_score=d.get("dimension_z_score"),
            baseline_family=d.get("baseline_family"),
            baseline_model_count=d.get("baseline_model_count"),
        )


def compare_profiles(
    source: ModelProfile,
    target: ModelProfile,
    baseline: "FamilyBaseline | None" = None,
) -> ProfileComparison:
    """Compare two ModelProfiles and produce alignment analysis.

    This function computes geometric alignment metrics between two models
    without making value judgments. The geometry speaks for itself.

    Args:
        source: Source model profile
        target: Target model profile
        baseline: Optional family baseline for z-score computation

    Returns:
        ProfileComparison with all computed metrics
    """
    # === STRUCTURAL COMPARISON ===
    architecture_match = (
        source.architecture == target.architecture
        and source.architecture != "unknown"
    )

    hidden_dim_ratio = (
        target.hidden_dim / source.hidden_dim
        if source.hidden_dim > 0
        else 1.0
    )

    layer_count_ratio = (
        target.num_layers / source.num_layers
        if source.num_layers > 0
        else 1.0
    )

    # === LAYER CORRESPONDENCE ===
    # Map layers by relative position
    layer_mapping: dict[int, int] = {}
    layer_comparisons: list[LayerComparison] = []

    source_layers = {lp.layer_idx: lp for lp in source.layer_profiles}
    target_layers = {lp.layer_idx: lp for lp in target.layer_profiles}

    source_num_layers = source.num_layers or len(source_layers)
    target_num_layers = target.num_layers or len(target_layers)

    for src_idx, src_lp in source_layers.items():
        # Map by relative position
        src_position = src_idx / max(1, source_num_layers - 1)
        tgt_idx = round(src_position * max(1, target_num_layers - 1))

        layer_mapping[src_idx] = tgt_idx

        tgt_lp = target_layers.get(tgt_idx)
        if tgt_lp:
            comparison = _compare_layers(src_lp, tgt_lp)
            layer_comparisons.append(comparison)

    # === GEOMETRIC ALIGNMENT ===
    # Compute alignment scores from layer comparisons
    sectional_diffs = [
        abs(lc.sectional_curvature_diff) for lc in layer_comparisons
    ]
    ricci_diffs = [abs(lc.ollivier_ricci_diff) for lc in layer_comparisons]
    dim_diffs = [abs(lc.intrinsic_dimension_diff) for lc in layer_comparisons]

    # Convert differences to alignment (0-1 scale)
    # Use exponential decay: exp(-diff/scale) where scale is derived from data
    # Scale = median of differences (so "typical" difference gives ~0.5 alignment)
    backend = get_default_backend()
    eps = float(machine_epsilon(backend, backend.array([1.0])))

    def _data_scale(diffs: list[float]) -> float:
        """Derive scale from median of differences."""
        if not diffs:
            return 1.0
        sorted_d = sorted(diffs)
        mid = len(sorted_d) // 2
        median = sorted_d[mid] if len(sorted_d) % 2 else (sorted_d[mid - 1] + sorted_d[mid]) / 2.0
        return max(median, eps)  # Ensure non-zero scale

    curvature_alignment = _mean_alignment(sectional_diffs, scale=_data_scale(sectional_diffs))
    ricci_alignment = _mean_alignment(ricci_diffs, scale=_data_scale(ricci_diffs))
    dimension_alignment = _mean_alignment(dim_diffs, scale=_data_scale(dim_diffs))

    # Equal weights - let individual alignments speak for themselves
    overall_alignment = (ricci_alignment + curvature_alignment + dimension_alignment) / 3.0

    backend = get_default_backend()

    # === TOPOLOGY COMPARISON ===
    topology_similarity = None
    if source.topology_summary and target.topology_summary:
        # Compare Betti numbers and persistence
        src_topo = source.topology_summary
        tgt_topo = target.topology_summary

        betti_diff = (
            abs(src_topo.component_count - tgt_topo.component_count)
            + abs(src_topo.cycle_count - tgt_topo.cycle_count)
        )
        persist_diff = abs(src_topo.max_persistence - tgt_topo.max_persistence)
        scale_eps = division_epsilon(
            backend,
            backend.array([src_topo.max_persistence, tgt_topo.max_persistence]),
        )
        scale = max(src_topo.max_persistence, tgt_topo.max_persistence, scale_eps)

        topology_similarity = exp_scalar(-betti_diff, backend) * exp_scalar(-persist_diff / scale, backend)

    # === SEMANTIC COMPARISON ===
    semantic_alignment = None
    if source.semantic_signature and target.semantic_signature:
        src_vec = source.semantic_signature.vector
        tgt_vec = target.semantic_signature.vector
        if src_vec and tgt_vec and len(src_vec) == len(tgt_vec):
            # Cosine similarity
            dot = sum(a * b for a, b in zip(src_vec, tgt_vec))
            norm_src = sqrt_scalar(sum(a * a for a in src_vec), backend)
            norm_tgt = sqrt_scalar(sum(b * b for b in tgt_vec), backend)
            if norm_src > 0 and norm_tgt > 0:
                semantic_alignment = dot / (norm_src * norm_tgt)

    # === ALIGNMENT EFFORT ===
    efforts = [lc.alignment_effort for lc in layer_comparisons]
    total_effort = sum(efforts)
    mean_effort = total_effort / len(efforts) if efforts else 0.0
    max_effort = max(efforts) if efforts else 0.0

    # === BASELINE-RELATIVE Z-SCORES ===
    sectional_z_score = None
    ricci_z_score = None
    dimension_z_score = None
    baseline_family = None
    baseline_model_count = None

    if baseline is not None and baseline.sample_count > 0:
        baseline_family = baseline.family
        baseline_model_count = baseline.sample_count

        # Compute z-scores for differences relative to baseline
        # The baseline captures typical variation within a family
        # Z-score = (observed_diff - expected_diff) / std
        # For curvature differences, expected_diff ≈ 0 within same family

        sectional_z_scores = []
        ricci_z_scores = []
        dim_z_scores = []

        backend = get_default_backend()
        for lc in layer_comparisons:
            # Map layer to relative position for baseline lookup
            src_layers = source.num_layers or len(source.layer_profiles)
            pos = lc.source_layer_idx / max(1, src_layers - 1)

            # Find closest baseline position
            closest_idx = 0
            min_dist = float("inf")
            for i, bp in enumerate(baseline.layer_positions):
                dist = abs(bp - pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            # Sectional z-score
            if closest_idx < len(baseline.sectional_std_by_position):
                std = baseline.sectional_std_by_position[closest_idx]
                eps = division_epsilon(backend, backend.array([std]))
                if std > eps:
                    z = abs(lc.sectional_curvature_diff) / std
                    sectional_z_scores.append(z)

            # Ricci z-score
            if closest_idx < len(baseline.ollivier_ricci_std_by_position):
                std = baseline.ollivier_ricci_std_by_position[closest_idx]
                eps = division_epsilon(backend, backend.array([std]))
                if std > eps:
                    z = abs(lc.ollivier_ricci_diff) / std
                    ricci_z_scores.append(z)

            # Dimension z-score (using relative diff)
            if closest_idx < len(baseline.intrinsic_dimension_by_position):
                baseline_dim = baseline.intrinsic_dimension_by_position[closest_idx]
                eps = division_epsilon(backend, backend.array([baseline_dim]))
                if baseline_dim > eps:
                    # Use relative diff: |diff| / baseline_dim
                    z = abs(lc.intrinsic_dimension_diff) / baseline_dim
                    dim_z_scores.append(z)

        # Aggregate z-scores (mean across layers)
        if sectional_z_scores:
            sectional_z_score = sum(sectional_z_scores) / len(sectional_z_scores)
        if ricci_z_scores:
            ricci_z_score = sum(ricci_z_scores) / len(ricci_z_scores)
        if dim_z_scores:
            dimension_z_score = sum(dim_z_scores) / len(dim_z_scores)

    return ProfileComparison(
        source_path=source.model_path,
        target_path=target.model_path,
        architecture_match=architecture_match,
        hidden_dim_ratio=hidden_dim_ratio,
        layer_count_ratio=layer_count_ratio,
        curvature_alignment=curvature_alignment,
        ricci_alignment=ricci_alignment,
        dimension_alignment=dimension_alignment,
        overall_alignment=overall_alignment,
        topology_similarity=topology_similarity,
        semantic_alignment=semantic_alignment,
        layer_mapping=layer_mapping,
        layer_comparisons=layer_comparisons,
        total_alignment_effort=total_effort,
        mean_alignment_effort=mean_effort,
        max_alignment_effort=max_effort,
        sectional_z_score=sectional_z_score,
        ricci_z_score=ricci_z_score,
        dimension_z_score=dimension_z_score,
        baseline_family=baseline_family,
        baseline_model_count=baseline_model_count,
    )


def _compare_layers(source: LayerProfile, target: LayerProfile) -> LayerComparison:
    """Compare two layer profiles."""
    # Curvature differences
    sectional_diff = target.sectional_curvature_mean - source.sectional_curvature_mean
    ricci_diff = target.ollivier_ricci_mean - source.ollivier_ricci_mean

    # Dimension differences
    src_dim = max(1.0, source.intrinsic_dimension)
    tgt_dim = max(1.0, target.intrinsic_dimension)
    dim_diff = tgt_dim - src_dim
    dim_ratio = tgt_dim / src_dim

    # Entropy difference
    entropy_diff = None
    if source.shannon_entropy is not None and target.shannon_entropy is not None:
        entropy_diff = target.shannon_entropy - source.shannon_entropy

    # Topology differences
    betti_0_diff = None
    betti_1_diff = None
    if source.betti_0 is not None and target.betti_0 is not None:
        betti_0_diff = target.betti_0 - source.betti_0
    if source.betti_1 is not None and target.betti_1 is not None:
        betti_1_diff = target.betti_1 - source.betti_1

    # Curvature sign match
    sign_match = (
        source.dominant_curvature_sign == target.dominant_curvature_sign
        or source.dominant_curvature_sign == "unknown"
        or target.dominant_curvature_sign == "unknown"
    )

    # Alignment effort (0-1)
    # Higher effort = more transformation needed
    _b = get_default_backend()
    dim_effort = min(1.0, abs(log_scalar(dim_ratio, _b)) / log_scalar(2, _b))  # Double/half = 1.0
    curv_effort = min(1.0, abs(ricci_diff))

    # Weight: dimension matters more for projection, curvature for rotation
    alignment_effort = 0.4 * dim_effort + 0.6 * curv_effort

    return LayerComparison(
        source_layer_idx=source.layer_idx,
        target_layer_idx=target.layer_idx,
        sectional_curvature_diff=sectional_diff,
        ollivier_ricci_diff=ricci_diff,
        intrinsic_dimension_diff=dim_diff,
        dimension_ratio=dim_ratio,
        shannon_entropy_diff=entropy_diff,
        betti_0_diff=betti_0_diff,
        betti_1_diff=betti_1_diff,
        alignment_effort=alignment_effort,
        curvature_sign_match=sign_match,
    )


def _mean_alignment(diffs: list[float], scale: float = 1.0) -> float:
    """Convert differences to alignment score using exponential decay."""
    if not diffs:
        return 1.0

    # Exponential decay: exp(-diff/scale)
    _b = get_default_backend()
    compatibilities = [exp_scalar(-d / scale, _b) for d in diffs]
    return sum(compatibilities) / len(compatibilities)


__all__ = [
    "LayerComparison",
    "ProfileComparison",
    "compare_profiles",
]
