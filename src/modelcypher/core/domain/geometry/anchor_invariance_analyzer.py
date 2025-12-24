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

"""
Anchor Invariance Analyzer.

Measures stability of semantic anchors across model pairs.
Anchors that maintain high cosine similarity across multiple model comparisons
are considered "invariant" - they represent stable semantic features.

Ported 1:1 from the reference Swift implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain.geometry.manifold_stitcher import (
    ModelFingerprints, ProbeSpace, output_layer_marker
)
from modelcypher.core.domain.geometry.metaphor_convergence_analyzer import (
    MetaphorConvergenceAnalyzer,
    DimensionAlignmentBuilder,
    AlignedDimension,
)
from modelcypher.core.domain.geometry.vector_math import SparseVectorMath


class AnchorInvarianceError(Exception):
    """Errors during anchor invariance analysis."""
    pass


class NoRunsError(AnchorInvarianceError):
    """No run inputs provided for analysis."""
    def __init__(self):
        super().__init__("No run inputs were provided for anchor invariance analysis.")


class NoAnchorsError(AnchorInvarianceError):
    """No anchors found for the given prefix."""
    def __init__(self, prefix: str):
        super().__init__(f"No anchors found for prefix {prefix}.")
        self.prefix = prefix


@dataclass
class RunInput:
    """Input for a single run (model pair comparison)."""
    id: str
    source: ModelFingerprints
    target: ModelFingerprints


@dataclass
class RunModels:
    """Model identifiers for a run."""
    model_a: str
    model_b: str


@dataclass
class RunResult:
    """Result from a single run."""
    id: str
    models: RunModels
    probe_space: ProbeSpace
    align_mode: MetaphorConvergenceAnalyzer.AlignMode
    anchor_means: dict[str, float]


@dataclass
class AnchorScore:
    """Stability score for a single anchor across all runs."""
    anchor_id: str
    prompt: str
    category: str
    family: str | None
    mean_cosine: float
    std_cosine: float
    min_cosine: float
    max_cosine: float
    stability_score: float  # mean - std (higher is more stable)
    run_count: int


@dataclass
class TopAnchor:
    """Summary of a top-performing anchor."""
    anchor_id: str
    mean_cosine: float
    stability_score: float


@dataclass
class Summary:
    """Summary statistics for the analysis."""
    anchor_count: int
    run_count: int
    overall_mean_cosine: float
    top_anchors: list[TopAnchor]


@dataclass
class Report:
    """Full anchor invariance analysis report."""
    align_mode: MetaphorConvergenceAnalyzer.AlignMode
    anchor_prefix: str
    holdout_prefixes: list[str]
    runs: list[RunResult]
    anchors: list[AnchorScore]
    summary: Summary


@dataclass
class AnchorVectorIndex:
    """Index of anchor vectors with metadata."""
    vectors: dict[str, dict[int, dict[int, float]]]  # anchor_id -> layer -> dim -> value
    prompts: dict[str, str]
    categories: dict[str, str]
    families: dict[str, str]


@dataclass
class LayerAlignment:
    """Layer alignment result."""
    aligned_pairs: list[MetaphorConvergenceAnalyzer.AlignmentPair]
    aligned_indices: list[int]


class AnchorInvarianceAnalyzer:
    """
    Analyzes anchor stability across model pairs.

    Takes multiple runs (each with source/target model fingerprints) and computes
    how consistently anchors maintain their semantic representation across
    different model comparisons.
    """

    @staticmethod
    def analyze(
        runs: list[RunInput],
        align_mode: MetaphorConvergenceAnalyzer.AlignMode = MetaphorConvergenceAnalyzer.AlignMode.NORMALIZED,
        anchor_prefix: str = "invariant:",
        anchor_family_allowlist: set[str] | None = None,
        holdout_prefixes: list[str] | None = None,
    ) -> Report:
        """
        Analyze anchor invariance across runs.

        Args:
            runs: List of RunInput, each containing source/target model fingerprints
            align_mode: How to align layers (LAYER = exact match, NORMALIZED = proportional)
            anchor_prefix: Prefix to filter anchors (e.g., "invariant:")
            anchor_family_allowlist: Optional set of families to include
            holdout_prefixes: Prefixes to exclude from dimension alignment

        Returns:
            Report with anchor scores and summary statistics
        """
        if not runs:
            raise NoRunsError()

        if holdout_prefixes is None:
            holdout_prefixes = []

        run_results: list[RunResult] = []
        anchor_values: dict[str, list[float]] = {}
        anchor_prompts: dict[str, str] = {}
        anchor_categories: dict[str, str] = {}
        anchor_families: dict[str, str] = {}

        for run in runs:
            if run.source.probe_space != run.target.probe_space:
                # Log warning but continue (parity with Swift)
                pass

            source_index = AnchorInvarianceAnalyzer._build_anchor_vectors(
                run.source, anchor_prefix, anchor_family_allowlist
            )
            target_index = AnchorInvarianceAnalyzer._build_anchor_vectors(
                run.target, anchor_prefix, anchor_family_allowlist
            )

            source_vectors = source_index.vectors
            target_vectors = target_index.vectors

            if not source_vectors or not target_vectors:
                raise NoAnchorsError(anchor_prefix)

            layer_alignment = AnchorInvarianceAnalyzer._build_layer_alignment(
                source_vectors=source_vectors,
                target_vectors=target_vectors,
                source_layer_count=run.source.layer_count,
                target_layer_count=run.target.layer_count,
                align_mode=align_mode,
            )

            dimension_alignment = DimensionAlignmentBuilder.build(
                source=run.source,
                target=run.target,
                aligned_pairs=layer_alignment.aligned_pairs,
                holdout_prefixes=holdout_prefixes,
            )

            run_means: dict[str, float] = {}
            anchor_ids = sorted(set(source_vectors.keys()) & set(target_vectors.keys()))
            pair_by_index = {p.index: p for p in layer_alignment.aligned_pairs}

            for anchor_id in anchor_ids:
                source_layers = source_vectors.get(anchor_id, {})
                target_layers = target_vectors.get(anchor_id, {})

                layer_cosines: list[float] = []
                for index in layer_alignment.aligned_indices:
                    pair = pair_by_index.get(index)
                    if pair is None:
                        continue

                    source_vec = source_layers.get(pair.source_layer)
                    target_vec = target_layers.get(pair.target_layer)
                    if source_vec is None or target_vec is None:
                        continue

                    mapping = dimension_alignment.by_layer.get(index, [])
                    mapped_source = AnchorInvarianceAnalyzer._apply_alignment(source_vec, mapping)

                    cosine = AnchorInvarianceAnalyzer._cosine_sparse(mapped_source, target_vec)
                    if cosine is not None:
                        layer_cosines.append(cosine)

                if not layer_cosines:
                    continue

                mean = sum(layer_cosines) / len(layer_cosines)
                run_means[anchor_id] = mean
                anchor_values.setdefault(anchor_id, []).append(mean)

            # Collect metadata
            for anchor_id, prompt in source_index.prompts.items():
                if anchor_id not in anchor_prompts:
                    anchor_prompts[anchor_id] = prompt
            for anchor_id, category in source_index.categories.items():
                if anchor_id not in anchor_categories:
                    anchor_categories[anchor_id] = category
            for anchor_id, family in source_index.families.items():
                if anchor_id not in anchor_families:
                    anchor_families[anchor_id] = family

            run_result = RunResult(
                id=run.id,
                models=RunModels(
                    model_a=run.source.model_id,
                    model_b=run.target.model_id,
                ),
                probe_space=run.source.probe_space,
                align_mode=align_mode,
                anchor_means=run_means,
            )
            run_results.append(run_result)

        # Compute anchor scores
        anchor_scores: list[AnchorScore] = []
        for anchor_id in sorted(anchor_values.keys()):
            values = anchor_values[anchor_id]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            min_val = min(values)
            max_val = max(values)
            stability_score = mean - std

            anchor_scores.append(AnchorScore(
                anchor_id=anchor_id,
                prompt=anchor_prompts.get(anchor_id, ""),
                category=anchor_categories.get(anchor_id, ""),
                family=anchor_families.get(anchor_id),
                mean_cosine=mean,
                std_cosine=std,
                min_cosine=min_val,
                max_cosine=max_val,
                stability_score=stability_score,
                run_count=len(values),
            ))

        # Sort by stability score descending, then by anchor_id for ties
        anchor_scores.sort(key=lambda a: (-a.stability_score, a.anchor_id))

        # Build summary
        overall_mean = (
            sum(a.mean_cosine for a in anchor_scores) / len(anchor_scores)
            if anchor_scores else 0.0
        )

        top_anchors = [
            TopAnchor(
                anchor_id=a.anchor_id,
                mean_cosine=a.mean_cosine,
                stability_score=a.stability_score,
            )
            for a in anchor_scores[:5]
        ]

        summary = Summary(
            anchor_count=len(anchor_scores),
            run_count=len(runs),
            overall_mean_cosine=overall_mean,
            top_anchors=top_anchors,
        )

        return Report(
            align_mode=align_mode,
            anchor_prefix=anchor_prefix,
            holdout_prefixes=holdout_prefixes,
            runs=run_results,
            anchors=anchor_scores,
            summary=summary,
        )

    @staticmethod
    def _build_anchor_vectors(
        fingerprints: ModelFingerprints,
        anchor_prefix: str,
        family_allowlist: set[str] | None,
    ) -> AnchorVectorIndex:
        """Build indexed vectors from fingerprints matching the anchor prefix."""
        vectors: dict[str, dict[int, dict[int, float]]] = {}
        prompts: dict[str, str] = {}
        categories: dict[str, str] = {}
        families: dict[str, str] = {}

        for fp in fingerprints.fingerprints:
            if not fp.prime_id.startswith(anchor_prefix):
                continue

            # Check family allowlist
            if family_allowlist is not None:
                family = AnchorInvarianceAnalyzer._extract_family(fp.prime_id, anchor_prefix)
                if family is not None and family not in family_allowlist:
                    continue

            layer_map: dict[int, dict[int, float]] = vectors.get(fp.prime_id, {})

            for layer, dims in fp.activated_dimensions.items():
                normalized_layer = AnchorInvarianceAnalyzer._normalize_layer_index(
                    layer, fingerprints.layer_count
                )
                sparse: dict[int, float] = {}
                for dim in dims:
                    sparse[dim.index] = float(dim.activation)
                if sparse:
                    layer_map[normalized_layer] = sparse

            if layer_map:
                vectors[fp.prime_id] = layer_map
                prompts[fp.prime_id] = fp.prime_text
                categories[fp.prime_id] = anchor_prefix.rstrip(":")

                family = AnchorInvarianceAnalyzer._extract_family(fp.prime_id, anchor_prefix)
                if family is not None:
                    families[fp.prime_id] = family

        return AnchorVectorIndex(
            vectors=vectors,
            prompts=prompts,
            categories=categories,
            families=families,
        )

    @staticmethod
    def _build_layer_alignment(
        source_vectors: dict[str, dict[int, dict[int, float]]],
        target_vectors: dict[str, dict[int, dict[int, float]]],
        source_layer_count: int,
        target_layer_count: int,
        align_mode: MetaphorConvergenceAnalyzer.AlignMode,
    ) -> LayerAlignment:
        """Build layer alignment between source and target."""
        source_layers = AnchorInvarianceAnalyzer._collect_layers(source_vectors)
        target_layers = AnchorInvarianceAnalyzer._collect_layers(target_vectors)

        aligned_pairs: list[MetaphorConvergenceAnalyzer.AlignmentPair] = []
        aligned_indices: list[int] = []

        if align_mode == MetaphorConvergenceAnalyzer.AlignMode.NORMALIZED:
            aligned_count = min(len(source_layers), len(target_layers))
            aligned_indices = list(range(aligned_count))
            denom_source = max(1, source_layer_count)
            denom_target = max(1, target_layer_count)

            for index in aligned_indices:
                source_index = AnchorInvarianceAnalyzer._scaled_index(
                    index, aligned_count, len(source_layers)
                )
                target_index = AnchorInvarianceAnalyzer._scaled_index(
                    index, aligned_count, len(target_layers)
                )

                if source_index >= len(source_layers) or target_index >= len(target_layers):
                    continue

                source_layer = source_layers[source_index]
                target_layer = target_layers[target_index]

                source_position = float(source_layer) / float(denom_source)
                target_position = float(target_layer) / float(denom_target)
                normalized_depth = (source_position + target_position) * 0.5

                aligned_pairs.append(MetaphorConvergenceAnalyzer.AlignmentPair(
                    index=index,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    normalized_depth=normalized_depth,
                ))
        else:
            # LAYER mode: exact layer matching
            layers = sorted(set(source_layers) & set(target_layers))
            aligned_indices = layers
            aligned_pairs = [
                MetaphorConvergenceAnalyzer.AlignmentPair(
                    index=layer,
                    source_layer=layer,
                    target_layer=layer,
                    normalized_depth=float(layer),
                )
                for layer in layers
            ]

        return LayerAlignment(aligned_pairs=aligned_pairs, aligned_indices=aligned_indices)

    @staticmethod
    def _extract_family(anchor_id: str, prefix: str) -> str | None:
        """Extract family from anchor ID (e.g., 'invariant:time_001' -> 'time')."""
        if prefix != "invariant:":
            return None
        if not anchor_id.startswith(prefix):
            return None
        trimmed = anchor_id[len(prefix):]
        underscore_idx = trimmed.find("_")
        if underscore_idx < 0:
            return None
        return trimmed[:underscore_idx]

    @staticmethod
    def _collect_layers(vectors: dict[str, dict[int, dict[int, float]]]) -> list[int]:
        """Collect all unique layers from vectors."""
        layers: set[int] = set()
        for layer_map in vectors.values():
            layers.update(layer_map.keys())
        return sorted(layers)

    @staticmethod
    def _normalize_layer_index(layer: int, layer_count: int) -> int:
        """Normalize output layer marker to layer count."""
        if layer == output_layer_marker:
            return layer_count
        return layer

    @staticmethod
    def _scaled_index(position: int, aligned_count: int, total_count: int) -> int:
        """Scale position index to total count."""
        if total_count <= 0:
            return 0
        if aligned_count <= 1:
            return 0
        fraction = float(position) / float(aligned_count - 1)
        scaled = int(round(fraction * float(total_count - 1)))
        return min(max(0, scaled), total_count - 1)

    @staticmethod
    def _apply_alignment(
        vector: dict[int, float],
        mapping: list[AlignedDimension],
    ) -> dict[int, float]:
        """Apply dimension alignment mapping to a sparse vector."""
        if not vector or not mapping:
            return {}
        mapped: dict[int, float] = {}
        for entry in mapping:
            if entry.source_dim in vector:
                value = vector[entry.source_dim]
                mapped[entry.target_dim] = mapped.get(entry.target_dim, 0.0) + value * entry.weight
        return mapped

    @staticmethod
    def _cosine_sparse(a: dict[int, float], b: dict[int, float]) -> float | None:
        """Compute cosine similarity between two sparse vectors."""
        return SparseVectorMath.cosine_similarity(a, b)
