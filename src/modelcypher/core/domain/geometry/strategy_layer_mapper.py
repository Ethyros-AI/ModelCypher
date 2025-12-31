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

"""Strategy-based layer mapper supporting CRM and INVARIANT_COLLAPSE strategies.

Conceptual geometry is invariant across all models - knowledge occupies
fixed probability clouds in hyperspace. These strategies differ only in
how they measure this invariant structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.invariant_layer_mapper import (
        Config,
        CRMMappingConfig,
        InvariantLayerMapper,
        LayerMapping,
        LayerMappingStrategy,
        LayerProfile,
        ModelFingerprints,
        Summary,
    )

__all__ = [
    "LayerCategoryScores",
    "StrategyMappingResult",
    "StrategyLayerMapper",
    "compute_layer_alignment_confidence",
    "select_optimal_strategy",
]


@dataclass(frozen=True)
class LayerCategoryScores:
    """Per-layer scores broken down by matching category.

    Individual geometric measurements. No aggregation - consumers
    use the measurements they need directly.
    """

    layer_index: int
    activation_pattern: float
    invariant_coverage: float
    collapse_state: float  # 1.0 if not collapsed, 0.0 if collapsed
    triangulation: float
    cka_alignment: float


@dataclass(frozen=True)
class StrategyMappingResult:
    """Result of strategy-based layer mapping."""

    source_model: str
    target_model: str
    strategy: LayerMappingStrategy
    mappings: tuple["LayerMapping", ...]
    source_category_scores: tuple[LayerCategoryScores, ...]
    target_category_scores: tuple[LayerCategoryScores, ...]
    summary: "Summary"
    # CRM-specific metrics
    mean_cka_alignment: float | None = None
    cka_matrix: tuple[tuple[float, ...], ...] | None = None


class StrategyLayerMapper:
    """
    Strategy-based layer mapper supporting CRM and INVARIANT_COLLAPSE strategies.

    Conceptual geometry is invariant across all models - knowledge occupies
    fixed probability clouds in hyperspace. These strategies differ only in
    how they measure this invariant structure.

    CRM Strategy:
        Uses CKA (Centered Kernel Alignment) to find optimal layer mappings.
        Works well when raw activations are available for direct comparison.

    INVARIANT_COLLAPSE Strategy:
        Uses semantic invariant probes with collapse detection. Maps layers
        by invariant activation patterns while penalizing collapsed layer
        mismatches. Robust when only fingerprints are available.
    """

    @staticmethod
    def map_layers_with_strategy(
        source: "ModelFingerprints",
        target: "ModelFingerprints",
        config: "Config",
        source_activations: dict[int, list[list[float]]] | None = None,
        target_activations: dict[int, list[list[float]]] | None = None,
    ) -> StrategyMappingResult:
        """
        Map layers using the configured strategy.

        Args:
            source: Fingerprints for source model
            target: Fingerprints for target model
            config: Mapping configuration with strategy selection
            source_activations: Raw activations by layer for CRM (optional)
            target_activations: Raw activations by layer for CRM (optional)

        Returns:
            StrategyMappingResult with mappings and per-layer scores
        """
        from modelcypher.core.domain.geometry.invariant_layer_mapper import (
            LayerMappingStrategy,
        )

        if config.strategy == LayerMappingStrategy.CRM:
            return StrategyLayerMapper._map_with_crm(
                source, target, config, source_activations, target_activations
            )
        else:
            return StrategyLayerMapper._map_with_invariant_collapse(
                source, target, config, source_activations, target_activations
            )

    @staticmethod
    def _map_with_crm(
        source: "ModelFingerprints",
        target: "ModelFingerprints",
        config: "Config",
        source_activations: dict[int, list[list[float]]] | None = None,
        target_activations: dict[int, list[list[float]]] | None = None,
    ) -> StrategyMappingResult:
        """Map layers using CRM-based CKA alignment."""
        from modelcypher.core.domain.geometry.invariant_layer_mapper import (
            CRMMappingConfig,
            InvariantLayerMapper,
            LayerMapping,
            LayerMappingStrategy,
            Summary,
        )

        crm_cfg = config.crm_config or CRMMappingConfig()

        # Build CKA matrix between source and target layers
        source_samples = InvariantLayerMapper._sample_layers(
            source.layer_count, config.sample_layer_count
        )
        target_samples = InvariantLayerMapper._sample_layers(
            target.layer_count, config.sample_layer_count
        )

        cka_matrix: list[list[float]] = []

        if source_activations and target_activations:
            # Use provided activations for CKA computation
            cka_matrix = StrategyLayerMapper._compute_cka_matrix(
                source_samples, target_samples, source_activations, target_activations, crm_cfg
            )
        else:
            # Fall back to fingerprint-based similarity
            invariant_ids, _, _ = InvariantLayerMapper._get_invariants(config)
            source_profile = InvariantLayerMapper._build_profile(source, invariant_ids, config)
            target_profile = InvariantLayerMapper._build_profile(target, invariant_ids, config)

            cka_matrix = [[0.0] * len(target_samples) for _ in range(len(source_samples))]
            for i, src_layer in enumerate(source_samples):
                src_vec = source_profile.vectors.get(src_layer, [])
                for j, tgt_layer in enumerate(target_samples):
                    tgt_vec = target_profile.vectors.get(tgt_layer, [])
                    cka_matrix[i][j] = InvariantLayerMapper._cosine_similarity(src_vec, tgt_vec)

        # Find optimal alignment using Hungarian algorithm or greedy
        mappings = StrategyLayerMapper._align_with_cka(
            source_samples, target_samples, cka_matrix, config
        )

        # Build category scores
        source_scores = StrategyLayerMapper._build_category_scores_crm(
            source_samples, cka_matrix, True
        )
        target_scores = StrategyLayerMapper._build_category_scores_crm(
            target_samples, cka_matrix, False
        )

        # Compute mean CKA
        all_cka = [
            cka_matrix[i][j] for i in range(len(source_samples)) for j in range(len(target_samples))
        ]
        mean_cka = sum(all_cka) / len(all_cka) if all_cka else 0.0

        mapped_count = len(mappings)
        skipped_count = sum(1 for m in mappings if m.is_skipped)
        mean_sim = sum(m.similarity for m in mappings) / len(mappings) if mappings else 0.0
        valid_mappings = [m for m in mappings if not m.is_skipped]
        alignment_quality = (
            sum(m.similarity for m in valid_mappings) / len(valid_mappings)
            if valid_mappings
            else 0.0
        )

        summary = Summary(
            mapped_layers=mapped_count,
            skipped_layers=skipped_count,
            mean_similarity=mean_sim,
            alignment_quality=alignment_quality,
            source_collapsed_layers=0,
            target_collapsed_layers=0,
        )

        return StrategyMappingResult(
            source_model=source.model_id,
            target_model=target.model_id,
            strategy=LayerMappingStrategy.CRM,
            mappings=tuple(mappings),
            source_category_scores=tuple(source_scores),
            target_category_scores=tuple(target_scores),
            summary=summary,
            mean_cka_alignment=mean_cka,
            cka_matrix=tuple(tuple(row) for row in cka_matrix),
        )

    @staticmethod
    def _map_with_invariant_collapse(
        source: "ModelFingerprints",
        target: "ModelFingerprints",
        config: "Config",
        source_activations: dict[int, list[list[float]]] | None = None,
        target_activations: dict[int, list[list[float]]] | None = None,
    ) -> StrategyMappingResult:
        """Map layers using invariant-collapse strategy."""
        from modelcypher.core.domain.geometry.invariant_layer_mapper import (
            InvariantLayerMapper,
            LayerMappingStrategy,
        )

        # Use the existing InvariantLayerMapper for base mapping
        base_report = InvariantLayerMapper.map_layers(source, target, config)

        # Compute CKA matrix if both activations available
        cka_matrix: list[list[float]] | None = None
        mean_cka: float | None = None
        source_cka_scores: dict[int, float] = {}
        target_cka_scores: dict[int, float] = {}

        if source_activations and target_activations:
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka

            backend = get_default_backend()
            source_layers = sorted(source_activations.keys())
            target_layers = sorted(target_activations.keys())

            cka_matrix = [[0.0] * len(target_layers) for _ in range(len(source_layers))]
            for i, src_layer in enumerate(source_layers):
                src_act = backend.array(source_activations[src_layer], dtype=backend.float32)
                for j, tgt_layer in enumerate(target_layers):
                    tgt_act = backend.array(target_activations[tgt_layer], dtype=backend.float32)
                    # Ensure same sample count
                    min_samples = min(backend.shape(src_act)[0], backend.shape(tgt_act)[0])
                    if min_samples >= 2:
                        result = compute_cka(
                            src_act[:min_samples],
                            tgt_act[:min_samples],
                            estimator=HSICEstimator.AUTO,
                            feature_bias_correction=True,
                        )
                        if result.is_valid:
                            cka_matrix[i][j] = (
                                result.cka_corrected
                                if result.cka_corrected is not None
                                else result.cka
                            )
                        else:
                            cka_matrix[i][j] = 0.0

            # Compute per-layer max CKA (best alignment with any layer in other model)
            for i, src_layer in enumerate(source_layers):
                source_cka_scores[src_layer] = max(cka_matrix[i]) if cka_matrix[i] else 0.0
            for j, tgt_layer in enumerate(target_layers):
                col = [cka_matrix[i][j] for i in range(len(source_layers))]
                target_cka_scores[tgt_layer] = max(col) if col else 0.0

            all_cka = [
                cka_matrix[i][j]
                for i in range(len(source_layers))
                for j in range(len(target_layers))
            ]
            mean_cka = sum(all_cka) / len(all_cka) if all_cka else None

        # Compute per-layer category scores
        source_scores = StrategyLayerMapper._build_category_scores_invariant(
            base_report.source_profiles, source_cka_scores
        )
        target_scores = StrategyLayerMapper._build_category_scores_invariant(
            base_report.target_profiles, target_cka_scores
        )

        return StrategyMappingResult(
            source_model=source.model_id,
            target_model=target.model_id,
            strategy=LayerMappingStrategy.INVARIANT_COLLAPSE,
            mappings=base_report.mappings,
            source_category_scores=tuple(source_scores),
            target_category_scores=tuple(target_scores),
            summary=base_report.summary,
            mean_cka_alignment=mean_cka,
            cka_matrix=tuple(tuple(row) for row in cka_matrix) if cka_matrix else None,
        )

    @staticmethod
    def _compute_cka_matrix(
        source_layers: list[int],
        target_layers: list[int],
        source_activations: dict[int, list[list[float]]],
        target_activations: dict[int, list[list[float]]],
        crm_cfg: CRMMappingConfig,
    ) -> list[list[float]]:
        """Compute CKA similarity matrix between layers."""
        matrix = [[0.0] * len(target_layers) for _ in range(len(source_layers))]

        for i, src_layer in enumerate(source_layers):
            src_acts = source_activations.get(src_layer)
            if not src_acts:
                continue

            for j, tgt_layer in enumerate(target_layers):
                tgt_acts = target_activations.get(tgt_layer)
                if not tgt_acts:
                    continue

                # Compute CKA between activations
                cka = StrategyLayerMapper._compute_linear_cka(src_acts, tgt_acts, crm_cfg)
                matrix[i][j] = cka

        return matrix

    @staticmethod
    def _compute_linear_cka(
        x: list[list[float]],
        y: list[list[float]],
        crm_cfg: CRMMappingConfig,
    ) -> float:
        """Compute linear CKA between two activation matrices.

        Delegates to the canonical CKA implementation in cka.py.
        """
        from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka_from_lists

        return compute_cka_from_lists(
            x,
            y,
            estimator=HSICEstimator.AUTO,
            feature_bias_correction=True,
        )

    @staticmethod
    def _align_with_cka(
        source_layers: list[int],
        target_layers: list[int],
        cka_matrix: list[list[float]],
        config: "Config",
    ) -> list["LayerMapping"]:
        """Align layers using CKA matrix (greedy optimal assignment)."""
        from modelcypher.core.domain.geometry.invariant_layer_mapper import (
            CRMMappingConfig,
            LayerMapping,
        )

        crm_cfg = config.crm_config or CRMMappingConfig()
        mappings: list[LayerMapping] = []

        # Greedy assignment: for each source, find best available target
        used_targets: set[int] = set()

        for i, src_layer in enumerate(source_layers):
            best_j = -1
            best_cka = -1.0

            for j, tgt_layer in enumerate(target_layers):
                if j in used_targets:
                    continue
                if cka_matrix[i][j] > best_cka:
                    best_cka = cka_matrix[i][j]
                    best_j = j

            if best_j >= 0:
                used_targets.add(best_j)
                tgt_layer = target_layers[best_j]

                is_skipped = best_cka < crm_cfg.min_cka_score

                mappings.append(
                    LayerMapping(
                        source_layer=src_layer,
                        target_layer=tgt_layer,
                        similarity=best_cka,
                        is_skipped=is_skipped,
                    )
                )

        return mappings

    @staticmethod
    def _build_category_scores_crm(
        layers: list[int],
        cka_matrix: list[list[float]],
        is_source: bool,
    ) -> list[LayerCategoryScores]:
        """Build category scores for CRM strategy."""
        scores: list[LayerCategoryScores] = []

        for idx, layer in enumerate(layers):
            # For CRM, CKA alignment is the primary signal
            if is_source:
                row = cka_matrix[idx] if idx < len(cka_matrix) else []
                max_cka = max(row) if row else 0.0
                mean_cka = sum(row) / len(row) if row else 0.0
            else:
                col = [
                    cka_matrix[i][idx] if idx < len(cka_matrix[i]) else 0.0
                    for i in range(len(cka_matrix))
                ]
                max_cka = max(col) if col else 0.0
                mean_cka = sum(col) / len(col) if col else 0.0

            scores.append(
                LayerCategoryScores(
                    layer_index=layer,
                    activation_pattern=mean_cka,
                    invariant_coverage=0.0,  # Not used in CRM
                    collapse_state=1.0,  # Assumed not collapsed in CRM
                    triangulation=0.0,  # Not used in CRM
                    cka_alignment=max_cka,
                )
            )

        return scores

    @staticmethod
    def _build_category_scores_invariant(
        profiles: tuple["LayerProfile", ...],
        cka_scores: dict[int, float],
    ) -> list[LayerCategoryScores]:
        """Build category scores for invariant-collapse strategy.

        Returns individual geometric measurements. No weighted aggregation -
        consumers use the measurements they need directly.

        Args:
            profiles: Layer profiles from invariant analysis.
            cka_scores: Pre-computed CKA scores per layer index.
        """
        scores: list[LayerCategoryScores] = []

        for profile in profiles:
            activation_score = profile.strength
            coverage_score = profile.coverage
            collapse_score = 0.0 if profile.collapsed else 1.0

            tri_score = 0.0
            if profile.triangulation:
                # Use raw cross_domain_multiplier - it's already a dimensionless ratio
                tri_score = profile.triangulation.cross_domain_multiplier

            # Use pre-computed CKA score, or 0.0 if not available for this layer
            cka_score = cka_scores.get(profile.layer_index, 0.0)

            scores.append(
                LayerCategoryScores(
                    layer_index=profile.layer_index,
                    activation_pattern=activation_score,
                    invariant_coverage=coverage_score,
                    collapse_state=collapse_score,
                    triangulation=tri_score,
                    cka_alignment=cka_score,
                )
            )

        return scores


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_layer_alignment_confidence(
    mappings: tuple["LayerMapping", ...],
    config: "Config",
) -> float:
    """Compute overall confidence in layer alignment.

    Parameters
    ----------
    mappings : tuple of LayerMapping
        Layer mappings to evaluate.
    config : Config
        Configuration with thresholds.

    Returns
    -------
    float
        Confidence value between 0 and 1.
    """
    if not mappings:
        return 0.0

    valid_mappings = [m for m in mappings if not m.is_skipped]
    if not valid_mappings:
        return 0.0

    mean_similarity = sum(m.similarity for m in valid_mappings) / len(valid_mappings)

    # Factor in coverage (what fraction of mappings are valid)
    coverage = len(valid_mappings) / len(mappings)

    return mean_similarity * coverage


def select_optimal_strategy(
    source: "ModelFingerprints",
    target: "ModelFingerprints",
    has_activations: bool = False,
) -> "LayerMappingStrategy":
    """
    Select optimal layer mapping strategy based on model characteristics.

    Heuristics:
    - If raw activations available and models have similar depth: CRM
    - If models have very different depths: INVARIANT_COLLAPSE
    - If one model has many collapsed layers: INVARIANT_COLLAPSE
    """
    from modelcypher.core.domain.geometry.invariant_layer_mapper import (
        LayerMappingStrategy,
    )

    depth_ratio = min(source.layer_count, target.layer_count) / max(
        source.layer_count, target.layer_count, 1
    )

    # CRM works well when depths are similar and we have activations
    if has_activations and depth_ratio >= 0.7:
        return LayerMappingStrategy.CRM

    # Default to invariant-collapse for robustness
    return LayerMappingStrategy.INVARIANT_COLLAPSE
