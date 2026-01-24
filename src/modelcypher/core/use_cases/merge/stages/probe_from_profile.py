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

"""Profile-based alignment for merge operations.

Provides alignment computation using pre-computed profile activations
instead of running probe inference.

Usage:
    from modelcypher.core.use_cases.merge.stages.probe_from_profile import (
        compute_alignment_from_profiles,
    )

    result = compute_alignment_from_profiles(
        source_profile_dir="/path/to/source/.modelcypher",
        target_profile_dir="/path/to/target/.modelcypher",
        backend=backend,
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.profile import (
    GeometricProfile,
    ProfileActivations,
    load_activations,
)
from modelcypher.core.use_cases.merge.stages.probe_alignment import (
    AlignmentResult,
    align_layers,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class ProfileAlignmentResult:
    """Result of profile-based alignment computation."""

    # Core alignment results (same as probe stage)
    feature_transforms: dict[int, Any]
    layer_mapping: dict[int, int]
    scale_ratios: dict[int, float]

    # Attention transforms (will be empty for profile-only alignment)
    attention_transforms: dict[int, Any] = field(default_factory=dict)
    k_transforms: dict[int, Any] = field(default_factory=dict)
    v_transforms: dict[int, Any] = field(default_factory=dict)

    # Intermediate/gate transforms (will be empty for profile-only alignment)
    intermediate_transforms: dict[int, Any] = field(default_factory=dict)
    gate_transforms: dict[int, Any] = field(default_factory=dict)

    # Embedding transform
    embedding_transform: Any | None = None

    # Activations (loaded from profiles)
    source_activations: dict[int, "Array"] = field(default_factory=dict)
    target_activations: dict[int, "Array"] = field(default_factory=dict)
    source_intermediate_activations: dict[int, "Array"] = field(default_factory=dict)
    target_intermediate_activations: dict[int, "Array"] = field(default_factory=dict)
    source_gate_activations: dict[int, "Array"] = field(default_factory=dict)
    target_gate_activations: dict[int, "Array"] = field(default_factory=dict)
    source_embedding_activations: "Array | None" = None
    target_embedding_activations: "Array | None" = None

    # Diagnostics
    layer_cka_scores: dict[int, float] = field(default_factory=dict)
    gram_condition_numbers: dict[int, float] = field(default_factory=dict)

    # Metrics for compatibility with probe stage
    probe_metrics: dict[str, Any] = field(default_factory=dict)
    probe_result: dict[str, Any] = field(default_factory=dict)

    # === PROBE METADATA for merge consistency ===
    # These ensure profile-based merges produce identical results to probe-based
    probe_ids: list[str] = field(default_factory=list)
    probe_domains: list[str] = field(default_factory=list)

    # Mean-pooled per-probe activations for density analysis
    source_mean_pooled: dict[int, "Array"] = field(default_factory=dict)
    target_mean_pooled: dict[int, "Array"] = field(default_factory=dict)

    # Injection layer computed during alignment (for single-point injection)
    injection_layer: int | None = None


def compute_alignment_from_profiles(
    source_profile_dir: str | Path,
    target_profile_dir: str | Path,
    backend: "Backend",
) -> ProfileAlignmentResult:
    """Compute alignment transforms from pre-computed profile activations.

    This is the core function for "profile once, merge many". Instead of
    running probe inference through both models (slow), we load cached
    activations from their profiles and compute alignment from those.

    Args:
        source_profile_dir: Directory containing source model's profile
        target_profile_dir: Directory containing target model's profile
        backend: Compute backend

    Returns:
        ProfileAlignmentResult with transforms and loaded activations

    Raises:
        FileNotFoundError: If profiles or activations don't exist
        RuntimeError: If alignment computation fails
    """
    source_profile_dir = Path(source_profile_dir)
    target_profile_dir = Path(target_profile_dir)

    logger.info("PROFILE ALIGNMENT: Loading profiles...")

    # Load profiles
    source_profile = GeometricProfile.load(source_profile_dir)
    target_profile = GeometricProfile.load(target_profile_dir)

    if not source_profile.has_activations:
        raise FileNotFoundError(
            f"Source profile has no activations: {source_profile_dir}"
        )
    if not target_profile.has_activations:
        raise FileNotFoundError(
            f"Target profile has no activations: {target_profile_dir}"
        )

    # Load activations
    logger.info("PROFILE ALIGNMENT: Loading source activations...")
    source_acts = load_activations(source_profile_dir, backend)

    logger.info("PROFILE ALIGNMENT: Loading target activations...")
    target_acts = load_activations(target_profile_dir, backend)

    logger.info(
        "PROFILE ALIGNMENT: Loaded source (hidden=%d, intermediate=%d, gate=%d), "
        "target (hidden=%d, intermediate=%d, gate=%d)",
        len(source_acts.hidden),
        len(source_acts.intermediate),
        len(source_acts.gate),
        len(target_acts.hidden),
        len(target_acts.intermediate),
        len(target_acts.gate),
    )

    # =========================================================================
    # BOTTLENECK-ONLY OPTIMIZATION: Identify bottleneck BEFORE alignment
    # =========================================================================
    # Alignment is O(n³) per layer. If we're only transferring the bottleneck,
    # we can skip aligning all other layers - massive speedup!
    #
    # Compute variance concentration from intermediate activations to identify
    # the bottleneck layer (highest variance concentration = most compressed).
    # =========================================================================
    bottleneck_layer: int | None = None
    layer_filter: list[int] | None = None

    if target_acts.intermediate:
        from modelcypher.core.domain.geometry.variance_concentration import (
            compute_variance_concentration,
            VarianceConcentrationResult,
        )

        logger.info("PROFILE ALIGNMENT: Computing variance concentration to identify bottleneck...")
        layer_variance: dict[int, VarianceConcentrationResult] = {}

        for layer_idx, acts in target_acts.intermediate.items():
            try:
                if isinstance(acts, list):
                    stacked = backend.stack(acts, axis=0)
                else:
                    stacked = acts
                backend.eval(stacked)

                var_result = compute_variance_concentration(stacked, backend)
                layer_variance[layer_idx] = var_result
            except Exception as exc:
                logger.debug("Variance computation failed for layer %d: %s", layer_idx, exc)
                continue

        if layer_variance:
            # The bottleneck is the layer with MAXIMUM variance concentration.
            # No threshold - the geometry tells us which layer is most compressed.
            best_layer = max(layer_variance.items(), key=lambda x: x[1].var_top1)
            bottleneck_layer = best_layer[0]

            # Also find TRANSMISSION layers (low var_top1, high effective_rank)
            # These are the ideal injection points (linear highway, massive null space)
            sorted_by_var = sorted(layer_variance.items(), key=lambda x: x[1].var_top1)
            n = len(sorted_by_var)
            median_var = sorted_by_var[n // 2][1].var_top1 if n > 0 else 0.5

            sorted_by_rank = sorted(layer_variance.items(), key=lambda x: x[1].effective_rank, reverse=True)
            median_rank = sorted_by_rank[n // 2][1].effective_rank if n > 0 else 50.0

            # Transmission = low var_top1 AND high effective_rank
            transmission_layers = [
                idx for idx, v in layer_variance.items()
                if v.var_top1 < median_var and v.effective_rank > median_rank
                and idx != 0  # Exclude embedding layer
            ]

            # Pick best injection layer: lowest var_top1 among transmission
            injection_layer = None
            if transmission_layers:
                injection_layer = min(transmission_layers, key=lambda x: layer_variance[x].var_top1)

            # Filter includes BOTH bottleneck (for scale ratios) AND injection layer
            layer_filter = [bottleneck_layer]
            if injection_layer is not None and injection_layer != bottleneck_layer:
                layer_filter.append(injection_layer)

            # Check for MC_INJECTION_LAYER override - MUST include in alignment
            injection_override = os.environ.get("MC_INJECTION_LAYER")
            if injection_override is not None:
                try:
                    override_layer = int(injection_override)
                    if override_layer not in layer_filter:
                        layer_filter.append(override_layer)
                        logger.info(
                            "PROFILE ALIGNMENT: Including override layer %d in alignment (MC_INJECTION_LAYER)",
                            override_layer,
                        )
                except ValueError:
                    logger.warning(
                        "PROFILE ALIGNMENT: Invalid MC_INJECTION_LAYER=%s, ignoring",
                        injection_override,
                    )

            logger.info(
                "PROFILE ALIGNMENT: BOTTLENECK = Layer %d (var_top1=%.1f%%, eff_rank=%.1f).",
                bottleneck_layer,
                best_layer[1].var_top1 * 100,
                best_layer[1].effective_rank,
            )
            if injection_layer is not None:
                inj_var = layer_variance[injection_layer]
                logger.info(
                    "PROFILE ALIGNMENT: INJECTION = Layer %d (var_top1=%.1f%%, eff_rank=%.1f). "
                    "Transmission layers: %s",
                    injection_layer,
                    inj_var.var_top1 * 100,
                    inj_var.effective_rank,
                    transmission_layers,
                )
            logger.info(
                "PROFILE ALIGNMENT: Aligning %d critical layers: %s",
                len(layer_filter), layer_filter,
            )
    else:
        logger.info("PROFILE ALIGNMENT: No intermediate activations - using full alignment")

    # Compute alignment using the same function as probe stage
    # Use layer_filter if bottleneck detected for massive speedup
    logger.info("PROFILE ALIGNMENT: Computing layer alignment...")
    alignment_result = align_layers(
        source_layer_activations=source_acts.hidden,
        target_layer_activations=target_acts.hidden,
        source_intermediate_activations=source_acts.intermediate or {},
        target_intermediate_activations=target_acts.intermediate or {},
        source_gate_activations=source_acts.gate or None,
        target_gate_activations=target_acts.gate or None,
        backend=backend,
        require_full_rank=False,  # Profiles may have partial coverage
        layer_filter=layer_filter,  # BOTTLENECK-ONLY: skip non-bottleneck layers
    )

    # Compute embedding alignment if both have embeddings
    embedding_transform = None
    if source_acts.embedding is not None and target_acts.embedding is not None:
        from modelcypher.core.domain.geometry.gram_aligner import GramAligner

        logger.info("PROFILE ALIGNMENT: Computing embedding alignment...")
        aligner = GramAligner(backend=backend)
        emb_result = aligner.find_perfect_alignment(source_acts.embedding, target_acts.embedding)
        embedding_transform = emb_result.feature_transform

    # Build metrics for compatibility with probe stage
    mean_cka = 0.0
    if alignment_result.layer_cka_scores:
        mean_cka = sum(alignment_result.layer_cka_scores.values()) / len(
            alignment_result.layer_cka_scores
        )

    probe_metrics = {
        "mean_cka": mean_cka,
        "min_cka": min(alignment_result.layer_cka_scores.values())
        if alignment_result.layer_cka_scores
        else 0.0,
        "converged_count": len(alignment_result.layer_mapping),
        "boundary_preserved_count": len(alignment_result.layer_mapping),
        "skipped_count": 0,
        "perfect_alignment": mean_cka >= 0.9999,
        "probe_failed": False,
        "from_profile": True,
        "source_profile": str(source_profile_dir),
        "target_profile": str(target_profile_dir),
        "layer_mapping": alignment_result.layer_mapping,
        "scale_ratios": alignment_result.scale_ratios,
    }

    # Get probe metadata from profiles for merge consistency
    # Both profiles should have the same probes if profiled with same atlas
    # Use target's probe_ids as the reference (what we're merging into)
    probe_ids = target_profile.probe_ids or source_profile.probe_ids or []
    probe_domains = target_profile.probe_domains or source_profile.probe_domains or []

    probe_result = {
        "confidences": {k: v for k, v in alignment_result.layer_cka_scores.items()},
        "intersection_map": None,
        # Include probe metadata for density stage
        "probe_ids": probe_ids,
        "probe_domains": probe_domains,
    }

    logger.info(
        "PROFILE ALIGNMENT: Complete. %d layers aligned, mean_cka=%.4f, %d probes",
        len(alignment_result.layer_mapping),
        mean_cka,
        len(probe_ids),
    )

    return ProfileAlignmentResult(
        feature_transforms=alignment_result.feature_transforms,
        layer_mapping=alignment_result.layer_mapping,
        scale_ratios=alignment_result.scale_ratios,
        attention_transforms=alignment_result.attention_transforms,
        k_transforms=alignment_result.k_transforms,
        v_transforms=alignment_result.v_transforms,
        intermediate_transforms=alignment_result.intermediate_transforms,
        gate_transforms=alignment_result.gate_transforms,
        embedding_transform=embedding_transform,
        source_activations=source_acts.hidden,
        target_activations=target_acts.hidden,
        source_intermediate_activations=source_acts.intermediate,
        target_intermediate_activations=target_acts.intermediate,
        source_gate_activations=source_acts.gate,
        target_gate_activations=target_acts.gate,
        source_embedding_activations=source_acts.embedding,
        target_embedding_activations=target_acts.embedding,
        layer_cka_scores=alignment_result.layer_cka_scores,
        gram_condition_numbers={},  # Diagnostics removed - condition numbers logged inline
        probe_metrics=probe_metrics,
        probe_result=probe_result,
        probe_ids=probe_ids,
        probe_domains=probe_domains,
        source_mean_pooled=source_acts.mean_pooled,
        target_mean_pooled=target_acts.mean_pooled,
        injection_layer=injection_layer,
    )


def check_profiles_available(
    source_path: str | Path,
    target_path: str | Path,
) -> tuple[bool, str | None, str | None]:
    """Check if valid profiles with activations exist for both models.

    Args:
        source_path: Path to source model
        target_path: Path to target model

    Returns:
        Tuple of (both_available, source_profile_dir, target_profile_dir)
        If both_available is False, the profile dirs will be None.
    """
    from modelcypher.core.domain.profile import (
        PROFILE_DIR_NAME,
        PROFILE_METADATA_FILE,
        GeometricProfileStore,
    )

    store = GeometricProfileStore()

    source_path = Path(source_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()

    # Check source
    source_profile = store.load(source_path)
    if source_profile is None or not source_profile.has_activations:
        logger.info("PROFILE CHECK: Source profile not available or missing activations")
        return False, None, None

    # Check target
    target_profile = store.load(target_path)
    if target_profile is None or not target_profile.has_activations:
        logger.info("PROFILE CHECK: Target profile not available or missing activations")
        return False, None, None

    # Determine profile directories
    source_profile_dir = store.profile_dir_for_model(source_path)
    if not (source_profile_dir / PROFILE_METADATA_FILE).exists():
        source_profile_dir = store.central_profile_dir(source_path)

    target_profile_dir = store.profile_dir_for_model(target_path)
    if not (target_profile_dir / PROFILE_METADATA_FILE).exists():
        target_profile_dir = store.central_profile_dir(target_path)

    logger.info(
        "PROFILE CHECK: Both profiles available with activations"
    )
    return True, str(source_profile_dir), str(target_profile_dir)


__all__ = [
    "ProfileAlignmentResult",
    "compute_alignment_from_profiles",
    "check_profiles_available",
]
