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

import numpy as np

from modelcypher.core.domain.agents.computational_gate_atlas import ComputationalGateInventory
from modelcypher.core.domain.agents.emotion_concept_atlas import (
    EmotionCategory,
    EmotionConceptInventory,
)
from modelcypher.core.domain.agents.semantic_prime_frames import SemanticPrimeFrames
from modelcypher.core.domain.agents.semantic_prime_multilingual import (
    SemanticPrimeMultilingualInventoryLoader,
)
from modelcypher.core.domain.agents.sequence_invariant_atlas import (
    SequenceFamily,
    SequenceInvariantInventory,
)
from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.geometry.cross_architecture_layer_matcher import (
    CrossArchitectureLayerMatcher,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    Config as SharedSubspaceConfig,
    SharedSubspaceProjector,
)
from modelcypher.ports.inference import HiddenStateEngine
from modelcypher.utils.paths import ensure_dir, expand_path


logger = logging.getLogger(__name__)
DEFAULT_MAX_PROMPTS_PER_ANCHOR = 3  # Balanced default; override for deeper sampling.
DEFAULT_MAX_POLYGLOT_TEXTS_PER_LANGUAGE = 2  # Mirrors AnchorExtractor defaults.


@dataclass(frozen=True)
class CRMBuildConfig:
    include_primes: bool = True
    include_gates: bool = True
    include_polyglot: bool = True
    include_sequence_invariants: bool = True
    include_emotions: bool = True
    sequence_families: frozenset[SequenceFamily] | None = None
    emotion_categories: frozenset[EmotionCategory] | None = None
    max_prompts_per_anchor: int = DEFAULT_MAX_PROMPTS_PER_ANCHOR
    max_polyglot_texts_per_language: int = DEFAULT_MAX_POLYGLOT_TEXTS_PER_LANGUAGE
    anchor_prefixes: list[str] | None = None
    max_anchors: int | None = None


@dataclass(frozen=True)
class CRMBuildSummary:
    model_path: str
    output_path: str
    layer_count: int
    hidden_dim: int
    anchor_count: int
    prime_count: int
    gate_count: int
    sequence_invariant_count: int = 0
    emotion_count: int = 0


@dataclass(frozen=True)
class CRMCompareSummary:
    source_path: str
    target_path: str
    common_anchor_count: int
    overall_alignment: float
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
    is_valid: bool
    layer_count: int
    alignment_quality: float
    h2_validation: dict[str, float | bool | str]
    layer_metrics: list[dict[str, float | int | bool]]


class ConceptResponseMatrixService:
    def __init__(self, engine: HiddenStateEngine | None = None) -> None:
        self.engine = engine

    def build(
        self,
        model_path: str,
        output_path: str,
        config: CRMBuildConfig | None = None,
        adapter: str | None = None,
    ) -> CRMBuildSummary:
        if self.engine is None:
            raise ValueError("Hidden-state engine required to build concept response matrices.")

        cfg = config or CRMBuildConfig()
        resolved_model = expand_path(model_path)
        if not resolved_model.exists():
            raise ValueError(f"Model path does not exist: {resolved_model}")
        if not resolved_model.is_dir():
            raise ValueError(f"Model path is not a directory: {resolved_model}")

        layer_count, hidden_dim = self._resolve_model_shape(resolved_model)
        anchor_entries = self._build_anchor_prompts(cfg)
        if cfg.max_anchors is not None:
            anchor_entries = anchor_entries[: max(0, cfg.max_anchors)]

        anchor_ids = [anchor_id for anchor_id, _ in anchor_entries]
        prime_count = sum(1 for anchor_id in anchor_ids if anchor_id.startswith("prime:"))
        gate_count = sum(1 for anchor_id in anchor_ids if anchor_id.startswith("gate:"))
        seq_count = sum(1 for anchor_id in anchor_ids if anchor_id.startswith("seq:"))
        emotion_count = sum(1 for anchor_id in anchor_ids if anchor_id.startswith("emotion:"))

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

        used_anchor_ids: list[str] = []
        for anchor_id, prompts in anchor_entries:
            if not prompts:
                continue
            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for prompt in prompts:
                states = self.engine.capture_hidden_states(
                    model=str(resolved_model),
                    prompt=prompt,
                    adapter=adapter,
                )
                for layer, vector in states.items():
                    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
                    if arr.shape[0] != hidden_dim:
                        logger.warning(
                            "Hidden dim mismatch for %s layer %s: expected %s, got %s",
                            anchor_id,
                            layer,
                            hidden_dim,
                            arr.shape[0],
                        )
                    layer_sums[layer] = layer_sums.get(layer, 0.0) + arr
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1

            if not layer_sums:
                continue
            averaged = {
                layer: (layer_sums[layer] / float(layer_counts[layer])).tolist()
                for layer in layer_sums
            }
            crm.record_activations(anchor_id, averaged)
            used_anchor_ids.append(anchor_id)

        if used_anchor_ids:
            prime_count = sum(1 for anchor_id in used_anchor_ids if anchor_id.startswith("prime:"))
            gate_count = sum(1 for anchor_id in used_anchor_ids if anchor_id.startswith("gate:"))
            seq_count = sum(1 for anchor_id in used_anchor_ids if anchor_id.startswith("seq:"))
            emotion_count = sum(1 for anchor_id in used_anchor_ids if anchor_id.startswith("emotion:"))
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
            overall_alignment=report.overall_alignment,
            layer_correspondence=correspondence,
            cka_matrix=report.cka_matrix if include_matrix else None,
        )

    def shared_subspace(
        self,
        source_path: str,
        target_path: str,
        config: SharedSubspaceConfig | None = None,
    ) -> CRMSharedSubspaceSummary:
        source = ConceptResponseMatrix.load(str(expand_path(source_path)))
        target = ConceptResponseMatrix.load(str(expand_path(target_path)))

        config = config or SharedSubspaceConfig()
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
                config=config,
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
                    "isValid": bool(result.is_valid),
                }
            )
            results.append(result)

        if not results:
            raise ValueError("Shared subspace discovery failed for all layer mappings.")

        shared_dim = int(np.mean([res.shared_dimension for res in results])) if results else 0
        alignment_error = float(np.mean([res.alignment_error for res in results])) if results else 0.0
        shared_variance_ratio = (
            float(np.mean([res.shared_variance_ratio for res in results])) if results else 0.0
        )
        top_correlation = (
            float(np.mean([res.alignment_strengths[0] for res in results if res.alignment_strengths]))
            if results
            else 0.0
        )
        sample_count = int(np.mean([res.sample_count for res in results])) if results else 0
        method = results[0].method.value if results else "cca"
        is_valid = all(res.is_valid for res in results)

        h2_validation = {
            "meanCKA": matcher.h2_validation.mean_cka,
            "highConfidenceProportion": matcher.h2_validation.high_confidence_proportion,
            "positionCorrelation": matcher.h2_validation.position_correlation,
            "isValidated": matcher.h2_validation.is_validated,
            "interpretation": matcher.h2_validation.interpretation,
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
            is_valid=is_valid,
            layer_count=len(results),
            alignment_quality=matcher.alignment_quality,
            h2_validation=h2_validation,
            layer_metrics=layer_metrics,
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

    def _build_anchor_prompts(self, config: CRMBuildConfig) -> list[tuple[str, list[str]]]:
        entries: list[tuple[str, list[str]]] = []
        normalized_prefixes = _normalize_prefixes(config.anchor_prefixes)

        if config.include_primes:
            entries.extend(self._prime_prompts(config))
        if config.include_gates:
            entries.extend(self._gate_prompts(config))
        if config.include_sequence_invariants:
            entries.extend(self._sequence_invariant_prompts(config))
        if config.include_emotions:
            entries.extend(self._emotion_prompts(config))

        if normalized_prefixes:
            entries = [
                (anchor_id, prompts)
                for anchor_id, prompts in entries
                if any(anchor_id.startswith(prefix) for prefix in normalized_prefixes)
            ]

        return entries

    def _prime_prompts(self, config: CRMBuildConfig) -> list[tuple[str, list[str]]]:
        primes = SemanticPrimeFrames.enriched()
        polyglot_by_id: dict[str, list[str]] = {}
        if config.include_polyglot:
            polyglot_by_id = _load_polyglot_prompts(
                [prime.id for prime in primes],
                max_per_language=config.max_polyglot_texts_per_language,
            )

        entries: list[tuple[str, list[str]]] = []
        for prime in primes:
            texts: list[str] = [prime.word]
            texts.extend(prime.frames)
            texts.extend(prime.exemplars)
            if prime.contrast:
                texts.append(prime.contrast)
            texts.extend(polyglot_by_id.get(prime.id, []))
            prompts = _limit_texts(texts, config.max_prompts_per_anchor)
            entries.append((f"prime:{prime.id}", prompts))
        return entries

    def _gate_prompts(self, config: CRMBuildConfig) -> list[tuple[str, list[str]]]:
        entries: list[tuple[str, list[str]]] = []
        for gate in ComputationalGateInventory.probe_gates():
            texts: list[str] = []
            gate_name = gate.name.lower().replace("_", " ")
            if gate.description:
                texts.append(f"{gate_name}: {gate.description}")
            texts.append(gate_name)
            texts.extend(gate.examples)
            texts.extend(gate.polyglot_examples)
            prompts = _limit_texts(texts, config.max_prompts_per_anchor)
            entries.append((f"gate:{gate.id}", prompts))
        return entries

    def _sequence_invariant_prompts(self, config: CRMBuildConfig) -> list[tuple[str, list[str]]]:
        entries: list[tuple[str, list[str]]] = []
        families = config.sequence_families
        probes = SequenceInvariantInventory.probes_for_families(
            set(families) if families else None
        )
        for probe in probes:
            texts: list[str] = [probe.name]
            if probe.description:
                texts.append(probe.description)
            texts.extend(probe.support_texts)
            prompts = _limit_texts(texts, config.max_prompts_per_anchor)
            entries.append((f"seq:{probe.id}", prompts))
        return entries

    def _emotion_prompts(self, config: CRMBuildConfig) -> list[tuple[str, list[str]]]:
        entries: list[tuple[str, list[str]]] = []
        emotions = EmotionConceptInventory.all_emotions()

        # Filter by category if specified
        if config.emotion_categories:
            emotions = [e for e in emotions if e.category in config.emotion_categories]

        for emotion in emotions:
            texts: list[str] = [emotion.name]
            if emotion.description:
                texts.append(f"{emotion.name}: {emotion.description}")
            texts.extend(emotion.support_texts)
            prompts = _limit_texts(texts, config.max_prompts_per_anchor)
            entries.append((f"emotion:{emotion.id}", prompts))

        # Also include dyads
        for dyad in EmotionConceptInventory.primary_dyads():
            texts: list[str] = [dyad.name]
            if dyad.description:
                texts.append(f"{dyad.name}: {dyad.description}")
            texts.extend(dyad.support_texts)
            prompts = _limit_texts(texts, config.max_prompts_per_anchor)
            entries.append((f"emotion:{dyad.id}", prompts))

        return entries


def _limit_texts(texts: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        trimmed = text.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        unique.append(trimmed)
        if len(unique) >= limit:
            break
    return unique


def _normalize_prefixes(prefixes: list[str] | None) -> list[str]:
    if not prefixes:
        return []
    normalized: list[str] = []
    for value in prefixes:
        trimmed = value.strip()
        if not trimmed:
            continue
        if not trimmed.endswith(":"):
            trimmed = f"{trimmed}:"
        normalized.append(trimmed)
    return normalized


def _load_polyglot_prompts(prime_ids: list[str], max_per_language: int) -> dict[str, list[str]]:
    try:
        inventory = SemanticPrimeMultilingualInventoryLoader.global_diverse()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to load multilingual primes: %s", exc)
        return {}

    prompts: dict[str, list[str]] = {}
    prime_set = set(prime_ids)
    for prime in inventory.primes:
        if prime.id not in prime_set:
            continue
        texts: list[str] = []
        for bucket in prime.languages:
            texts.extend(bucket.texts[: max(0, max_per_language)])
        prompts[prime.id] = _limit_texts(texts, limit=len(texts))
    return prompts


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
