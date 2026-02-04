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

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.atlas.unified_atlas import (
    AtlasSource,
    UnifiedAtlasInventory,
)
from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    sqrt_scalar,
    ulp_scalar,
)
from modelcypher.core.domain.geometry.cross_architecture_layer_matcher import (
    CrossArchitectureLayerMatcher,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    SharedSubspaceProjector,
)
from modelcypher.ports.inference import HiddenStateEngine
from modelcypher.utils.paths import ensure_dir, expand_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRMBuildSummary:
    model_path: str
    output_path: str
    layer_count: int
    hidden_dim: int
    anchor_count: int
    prime_count: int  # semantic primes (NSM)
    gate_count: int  # computational gates
    sequence_invariant_count: int = 0
    emotion_count: int = 0
    prime_number_count: int = 0  # actual prime numbers (2, 3, 5, 7...)


@dataclass(frozen=True)
class CRMCompareSummary:
    source_path: str
    target_path: str
    common_anchor_count: int
    mean_cka: float
    alignment_precision: float  # Numerical precision (1.0 = exact kernel alignment)
    aligned: bool
    layer_correspondence: list[dict[str, float | int]]
    cka_matrix: list[list[float]] | None


@dataclass(frozen=True)
class CRMSharedSubspaceSummary:
    source_path: str
    target_path: str
    shared_dimension: int
    alignment_error: float
    shared_variance_ratio: float
    top_correlation: float
    sample_count: int
    method: str
    has_shared_structure: bool
    layer_count: int
    mean_mapping_cka: float
    aligned: bool
    h2_validation: dict[str, float | bool | str]
    layer_metrics: list[dict[str, float | int | bool]]


@dataclass(frozen=True)
class KnowledgeDeltaLayerSummary:
    layer: int
    anchor_count: int
    coverage: float
    source_mean_norm: float
    target_mean_norm: float
    source_std_norm: float
    target_std_norm: float
    delta_mean_norm: float
    density_ratio: float
    graftable: bool


@dataclass(frozen=True)
class KnowledgeDeltaMaskSummary:
    source_path: str
    target_path: str
    common_anchor_count: int
    layer_count: int
    target_sparse_threshold: float
    source_dense_threshold: float
    density_ratio_threshold: float
    graft_layers: list[int]
    graft_mask_by_layer: dict[int, float]
    skipped_layers: list[int]
    layer_summaries: list[KnowledgeDeltaLayerSummary]


class ConceptResponseMatrixService:
    def __init__(self, engine: HiddenStateEngine | None = None) -> None:
        self.engine = engine
        self._anchor_prompt_cache: list[tuple[str, list[str]]] | None = None
        self._prompt_state_cache: dict[tuple[str, str | None, str], dict[int, list[float]]] = {}
        self._anchor_activation_cache: dict[tuple[str, str | None, str], dict[int, list[float]]] = {}

    def build(
        self,
        model_path: str,
        output_path: str,
        adapter: str | None = None,
    ) -> CRMBuildSummary:
        if self.engine is None:
            raise ValueError("Hidden-state engine required to build concept response matrices.")

        resolved_model = expand_path(model_path)
        if not resolved_model.exists():
            raise ValueError(f"Model path does not exist: {resolved_model}")
        if not resolved_model.is_dir():
            raise ValueError(f"Model path is not a directory: {resolved_model}")

        layer_count, hidden_dim = self._resolve_model_shape(resolved_model)
        anchor_entries = self._build_anchor_prompts()

        anchor_ids = [anchor_id for anchor_id, _ in anchor_entries]
        # Count by unified atlas source types
        prime_count = sum(1 for aid in anchor_ids if aid.startswith("semantic_prime:"))
        gate_count = sum(1 for aid in anchor_ids if aid.startswith("computational_gate:"))
        seq_count = sum(1 for aid in anchor_ids if aid.startswith("sequence_invariant:"))
        emotion_count = sum(1 for aid in anchor_ids if aid.startswith("emotion_concept:"))
        prime_number_count = sum(1 for aid in anchor_ids if aid.startswith("prime_number:"))

        crm = ConceptResponseMatrix(
            model_identifier=str(resolved_model),
            layer_count=layer_count,
            hidden_dim=hidden_dim,
            anchor_metadata=AnchorMetadata(
                total_count=len(anchor_ids),
                semantic_prime_count=prime_count,
                computational_gate_count=gate_count,
                anchor_ids=anchor_ids,
            ),
        )

        backend = get_default_backend()
        used_anchor_ids: list[str] = []
        for anchor_id, prompts in anchor_entries:
            if not prompts:
                continue
            anchor_key = (str(resolved_model), adapter, anchor_id)
            cached_avg = self._anchor_activation_cache.get(anchor_key)
            if cached_avg is not None:
                crm.record_activations(anchor_id, cached_avg)
                used_anchor_ids.append(anchor_id)
                continue
            layer_sums: dict[int, object] = {}
            layer_counts: dict[int, int] = {}
            for prompt in prompts:
                cache_key = (str(resolved_model), adapter, prompt)
                states = self._prompt_state_cache.get(cache_key)
                if states is None:
                    states = self.engine.capture_hidden_states(
                        model=str(resolved_model),
                        prompt=prompt,
                        adapter=adapter,
                    )
                    self._prompt_state_cache[cache_key] = states
                for layer, vector in states.items():
                    arr = backend.array(vector, dtype="float32")
                    arr = backend.reshape(arr, (-1,))
                    backend.eval(arr)  # Evaluate in-place
                    arr_shape = backend.shape(arr)
                    if arr_shape[0] != hidden_dim:
                        logger.warning(
                            "Hidden dim mismatch for %s layer %s: expected %s, got %s",
                            anchor_id,
                            layer,
                            hidden_dim,
                            arr_shape[0],
                        )
                    if layer not in layer_sums:
                        layer_sums[layer] = arr
                    else:
                        layer_sums[layer] = layer_sums[layer] + arr
                        backend.eval(layer_sums[layer])
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1

            if not layer_sums:
                continue
            averaged = {}
            for layer in layer_sums:
                avg_arr = layer_sums[layer] / float(layer_counts[layer])
                backend.eval(avg_arr)
                averaged[layer] = backend.tolist(avg_arr)
            crm.record_activations(anchor_id, averaged)
            used_anchor_ids.append(anchor_id)
            self._anchor_activation_cache[anchor_key] = averaged

        if used_anchor_ids:
            prime_count = sum(1 for aid in used_anchor_ids if aid.startswith("semantic_prime:"))
            gate_count = sum(1 for aid in used_anchor_ids if aid.startswith("computational_gate:"))
            seq_count = sum(1 for aid in used_anchor_ids if aid.startswith("sequence_invariant:"))
            emotion_count = sum(1 for aid in used_anchor_ids if aid.startswith("emotion_concept:"))
            prime_number_count = sum(1 for aid in used_anchor_ids if aid.startswith("prime_number:"))
            crm.anchor_metadata = AnchorMetadata(
                total_count=len(used_anchor_ids),
                semantic_prime_count=prime_count,
                computational_gate_count=gate_count,
                anchor_ids=used_anchor_ids,
            )

        output = expand_path(output_path)
        ensure_dir(output.parent)
        crm.save(str(output))

        return CRMBuildSummary(
            model_path=str(resolved_model),
            output_path=str(output),
            layer_count=layer_count,
            hidden_dim=hidden_dim,
            anchor_count=crm.anchor_metadata.total_count,
            prime_count=crm.anchor_metadata.semantic_prime_count,
            gate_count=crm.anchor_metadata.computational_gate_count,
            sequence_invariant_count=seq_count,
            emotion_count=emotion_count,
            prime_number_count=prime_number_count,
        )

    def compare(
        self,
        source_path: str,
        target_path: str,
        include_matrix: bool = False,
    ) -> CRMCompareSummary:
        source = ConceptResponseMatrix.load(str(expand_path(source_path)))
        target = ConceptResponseMatrix.load(str(expand_path(target_path)))
        report = source.compare(target)

        correspondence = [
            {
                "sourceLayer": match.source_layer,
                "targetLayer": match.target_layer,
                "cka": match.cka,
            }
            for match in report.layer_correspondence
        ]

        return CRMCompareSummary(
            source_path=str(expand_path(source_path)),
            target_path=str(expand_path(target_path)),
            common_anchor_count=report.common_anchor_count,
            mean_cka=report.mean_cka,
            alignment_precision=report.alignment_precision,
            aligned=report.is_perfect,
            layer_correspondence=correspondence,
            cka_matrix=report.cka_matrix if include_matrix else None,
        )

    def shared_subspace(
        self,
        source_path: str,
        target_path: str,
    ) -> CRMSharedSubspaceSummary:
        """Discover shared subspace between two CRMs.

        All subspace discovery parameters are derived from data at runtime.

        Args:
            source_path: Path to source CRM file.
            target_path: Path to target CRM file.

        Returns:
            Summary of shared subspace discovery.
        """
        source = ConceptResponseMatrix.load(str(expand_path(source_path)))
        target = ConceptResponseMatrix.load(str(expand_path(target_path)))

        matcher = CrossArchitectureLayerMatcher.find_correspondence(source, target)

        layer_metrics: list[dict[str, float | int | bool]] = []
        results = []
        for mapping in matcher.mappings:
            if mapping.is_skipped:
                continue
            result = SharedSubspaceProjector.discover(
                source,
                target,
                mapping.source_layer,
                target_layer=mapping.target_layer,
            )
            if result is None:
                continue
            top_corr = float(result.alignment_strengths[0]) if result.alignment_strengths else 0.0
            layer_metrics.append(
                {
                    "sourceLayer": mapping.source_layer,
                    "targetLayer": mapping.target_layer,
                    "cka": float(mapping.cka),
                    "sharedDimension": int(result.shared_dimension),
                    "alignmentError": float(result.alignment_error),
                    "sharedVarianceRatio": float(result.shared_variance_ratio),
                    "topCorrelation": top_corr,
                    "sampleCount": int(result.sample_count),
                    "hasSharedStructure": bool(result.has_shared_structure),
                }
            )
            results.append(result)

        if not results:
            raise ValueError("Shared subspace discovery failed for all layer mappings.")

        backend = get_default_backend()

        shared_dims = backend.array([res.shared_dimension for res in results])
        shared_dims = backend.astype(shared_dims, "float32")
        shared_dim = int(backend.mean(shared_dims)) if results else 0

        alignment_errors = backend.array([res.alignment_error for res in results])
        alignment_errors = backend.astype(alignment_errors, "float32")
        alignment_error = float(backend.mean(alignment_errors)) if results else 0.0

        variance_ratios = backend.array([res.shared_variance_ratio for res in results])
        variance_ratios = backend.astype(variance_ratios, "float32")
        shared_variance_ratio = float(backend.mean(variance_ratios)) if results else 0.0

        top_correlations = [res.alignment_strengths[0] for res in results if res.alignment_strengths]
        if top_correlations:
            top_corr_arr = backend.array(top_correlations)
            top_corr_arr = backend.astype(top_corr_arr, "float32")
            top_correlation = float(backend.mean(top_corr_arr))
        else:
            top_correlation = 0.0

        sample_counts = backend.array([res.sample_count for res in results])
        sample_counts = backend.astype(sample_counts, "float32")
        sample_count = int(backend.mean(sample_counts)) if results else 0

        method = results[0].method.value if results else "cca"
        has_shared_structure = all(res.has_shared_structure for res in results)

        h2_validation = {
            "meanCKA": matcher.h2_validation.mean_cka,
            "minCKA": matcher.h2_validation.min_cka,
            "maxCKA": matcher.h2_validation.max_cka,
            "positionCorrelation": matcher.h2_validation.position_correlation,
        }

        return CRMSharedSubspaceSummary(
            source_path=str(expand_path(source_path)),
            target_path=str(expand_path(target_path)),
            shared_dimension=shared_dim,
            alignment_error=alignment_error,
            shared_variance_ratio=shared_variance_ratio,
            top_correlation=top_correlation,
            sample_count=sample_count,
            method=method,
            has_shared_structure=has_shared_structure,
            layer_count=len(results),
            mean_mapping_cka=matcher.mean_cka,
            aligned=matcher.aligned,
            h2_validation=h2_validation,
            layer_metrics=layer_metrics,
        )

    def knowledge_delta_mask(
        self,
        source_path: str,
        target_path: str,
    ) -> KnowledgeDeltaMaskSummary:
        """Build a knowledge delta mask from two CRMs.

        The mask highlights layers where the source has higher activation
        density while the target appears sparse, using distribution-derived
        thresholds for activation norms.
        """
        source = ConceptResponseMatrix.load(str(expand_path(source_path)))
        target = ConceptResponseMatrix.load(str(expand_path(target_path)))
        backend = get_default_backend()

        common = source.common_anchor_ids(target)
        if not common:
            raise ValueError("No common anchors found between source and target CRMs.")

        layer_count = min(source.layer_count, target.layer_count)
        raw_layers: list[dict[str, float | int]] = []
        skipped_layers: list[int] = []

        for layer in range(layer_count):
            source_layer = source.activations.get(layer, {})
            target_layer = target.activations.get(layer, {})
            if not source_layer or not target_layer:
                skipped_layers.append(layer)
                continue

            source_norms: list[float] = []
            target_norms: list[float] = []
            for anchor_id in common:
                source_act = source_layer.get(anchor_id)
                target_act = target_layer.get(anchor_id)
                if source_act is None or target_act is None:
                    continue
                source_norms.append(float(source_act.norm))
                target_norms.append(float(target_act.norm))

            anchor_count = len(source_norms)
            if anchor_count == 0:
                skipped_layers.append(layer)
                continue

            source_mean, source_std = _mean_std(source_norms)
            target_mean, target_std = _mean_std(target_norms)
            delta_mean = source_mean - target_mean
            eps = division_epsilon(backend, backend.array([target_mean]))
            density_ratio = source_mean / (target_mean + eps)
            coverage = anchor_count / float(len(common)) if common else 0.0

            raw_layers.append(
                {
                    "layer": layer,
                    "anchor_count": anchor_count,
                    "coverage": coverage,
                    "source_mean_norm": source_mean,
                    "target_mean_norm": target_mean,
                    "source_std_norm": source_std,
                    "target_std_norm": target_std,
                    "delta_mean_norm": delta_mean,
                    "density_ratio": density_ratio,
                }
            )

        if not raw_layers:
            raise ValueError("No layers met the minimum anchor count for delta mask.")

        target_means = [float(entry["target_mean_norm"]) for entry in raw_layers]
        source_means = [float(entry["source_mean_norm"]) for entry in raw_layers]
        density_ratios = [float(entry["density_ratio"]) for entry in raw_layers]

        _b = get_default_backend()
        target_sparse_threshold = find_magnitude_gap_threshold(
            sorted(target_means), eps=ulp_scalar(1.0, _b)
        )
        source_dense_threshold = find_magnitude_gap_threshold(
            sorted(source_means), eps=ulp_scalar(1.0, _b)
        )
        density_ratio_threshold = find_magnitude_gap_threshold(
            sorted(density_ratios), eps=ulp_scalar(1.0, _b)
        )

        graft_layers: list[int] = []
        graft_mask_by_layer = {layer: 0.0 for layer in range(layer_count)}
        layer_summaries: list[KnowledgeDeltaLayerSummary] = []

        for entry in raw_layers:
            layer = int(entry["layer"])
            target_mean = float(entry["target_mean_norm"])
            source_mean = float(entry["source_mean_norm"])
            delta_mean = float(entry["delta_mean_norm"])
            density_ratio = float(entry["density_ratio"])

            is_sparse = target_mean <= target_sparse_threshold
            is_dense = source_mean >= source_dense_threshold
            ratio_ok = density_ratio >= density_ratio_threshold
            delta_ok = delta_mean > 0.0

            graftable = bool(is_sparse and is_dense and ratio_ok and delta_ok)
            if graftable:
                graft_layers.append(layer)
                graft_mask_by_layer[layer] = 1.0

            layer_summaries.append(
                KnowledgeDeltaLayerSummary(
                    layer=layer,
                    anchor_count=int(entry["anchor_count"]),
                    coverage=float(entry["coverage"]),
                    source_mean_norm=source_mean,
                    target_mean_norm=target_mean,
                    source_std_norm=float(entry["source_std_norm"]),
                    target_std_norm=float(entry["target_std_norm"]),
                    delta_mean_norm=delta_mean,
                    density_ratio=density_ratio,
                    graftable=graftable,
                )
            )

        return KnowledgeDeltaMaskSummary(
            source_path=str(expand_path(source_path)),
            target_path=str(expand_path(target_path)),
            common_anchor_count=len(common),
            layer_count=layer_count,
            target_sparse_threshold=target_sparse_threshold,
            source_dense_threshold=source_dense_threshold,
            density_ratio_threshold=density_ratio_threshold,
            graft_layers=graft_layers,
            graft_mask_by_layer=graft_mask_by_layer,
            skipped_layers=skipped_layers,
            layer_summaries=layer_summaries,
        )

    def _resolve_model_shape(self, model_path: Path) -> tuple[int, int]:
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"config.json not found in model directory: {model_path}")

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid config.json: {exc}") from exc

        layer_count = _first_int(config, ["num_hidden_layers", "n_layer", "num_layers", "n_layers"])
        hidden_dim = _first_int(config, ["hidden_size", "n_embd", "hidden_dim", "d_model"])

        if layer_count is None:
            raise ValueError("Unable to determine layer count from config.json")
        if hidden_dim is None:
            raise ValueError("Unable to determine hidden dimension from config.json")

        return int(layer_count), int(hidden_dim)

    def _build_anchor_prompts(self) -> list[tuple[str, list[str]]]:
        """Build anchor prompts from the unified atlas.

        Uses ALL probes from the unified atlas system (1000+ probes across
        all domains including prime numbers, emotions, sequences, etc.)
        """
        if self._anchor_prompt_cache is not None:
            return self._anchor_prompt_cache

        entries: list[tuple[str, list[str]]] = []
        probes = UnifiedAtlasInventory.all_probes()

        for probe in probes:
            # Build prompts from name, description, and support texts
            texts: list[str] = [probe.name]
            if probe.description:
                texts.append(probe.description)
            texts.extend(probe.support_texts)
            prompts = _dedupe_texts(texts)

            # Use probe_id which includes source prefix (e.g., "prime_number:prime_2")
            entries.append((probe.probe_id, prompts))

        self._anchor_prompt_cache = entries
        return entries


def _dedupe_texts(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        trimmed = text.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        unique.append(trimmed)
    return unique


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return float(mean), sqrt_scalar(max(0.0, variance), get_default_backend())


def _first_int(payload: dict, keys: list[str]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
