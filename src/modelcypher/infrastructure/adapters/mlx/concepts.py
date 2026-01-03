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


import logging
import re

import mlx.core as mx

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    ceil_scalar,
    log2_scalar,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.vector_math import (
    geodesic_cosine_batch,
    geodesic_norms,
)
from modelcypher.core.domain.geometry.types import (
    ConceptComparisonResult,
    DetectedConcept,
    DetectionResult,
)
from modelcypher.ports.async_embeddings import EmbedderPort
from modelcypher.ports.concept_discovery import ConceptDiscoveryPort

logger = logging.getLogger(__name__)


def _load_unified_atlas_concepts() -> list[tuple[str, list[str]]]:
    """Load concepts from the UnifiedAtlas (multi-domain probe system).

    The UnifiedAtlas triangulates across:
    - Computational Gates (76): Programming concept primitives
    - Sequence Invariants (70): Mathematical anchors (Fibonacci, primes, logic)
    - Semantic Primes (65): Linguistic universals from Goddard & Wierzbicka (2014)
    - Emotion Concepts (32): Plutchik's wheel with VAD dimensions
    - Moral Concepts (30): Haidt's Moral Foundations Theory
    - Temporal Concepts (25): Arrow of time, duration, causality
    - Spatial Concepts (23): Vertical, lateral, depth, mass, furniture
    - Social Concepts (25): Power hierarchy, kinship, formality
    - Compositional (22): Semantic prime compositions
    - Philosophical (30): Ontology, epistemology, logic, modality, mereology
    - Conceptual Genealogy (29): Etymology and lineage anchors
    - Metaphor Invariants (14): Cross-cultural semantic anchors
    - Syntax Concepts (24): Syntax, morphology, word order, punctuation

    Returns:
        List of (concept_id, [support_texts]) tuples for embedding triangulation.
    """
    try:
        from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

        probes = UnifiedAtlasInventory.all_probes()
        concepts: list[tuple[str, list[str]]] = []

        for probe in probes:
            # Create concept ID with source prefix for domain clarity
            concept_id = f"{probe.source.value}:{probe.id}"

            # Build support texts from probe metadata
            support_texts: list[str] = []

            # Primary text: name and description
            support_texts.append(f"{probe.name}: {probe.description}")

            # Add probe support texts if available
            for text in probe.support_texts[:3]:  # Limit to 3 per probe
                support_texts.append(text)

            # Ensure at least 2 support texts for lower-variance centroid estimation
            if len(support_texts) < 2:
                support_texts.append(f"The concept of {probe.name}")

            concepts.append((concept_id, support_texts))

        logger.info(
            f"Loaded {len(concepts)} concepts from UnifiedAtlas "
            f"({UnifiedAtlasInventory.probe_count()})"
        )
        return concepts

    except ImportError as e:
        logger.warning(f"UnifiedAtlas not available, using fallback concepts: {e}")
        return _fallback_concepts()


def _fallback_concepts() -> list[tuple[str, list[str]]]:
    """Fallback concept inventory when UnifiedAtlas is unavailable.

    Provides essential mathematical and semantic anchors for basic operation.
    """
    return [
        # Mathematical invariants
        (
            "sequence:fibonacci",
            [
                "Fibonacci sequence: each term is the sum of the two previous terms",
                "0, 1, 1, 2, 3, 5, 8, 13, 21, 34",
                "F(n) = F(n-1) + F(n-2)",
                "Golden ratio phi = 1.618...",
            ],
        ),
        (
            "sequence:primes",
            [
                "Prime numbers: divisible only by 1 and themselves",
                "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
                "The atoms of arithmetic",
                "Fundamental theorem of arithmetic",
            ],
        ),
        # Logical invariants
        (
            "logic:modus_ponens",
            [
                "If A implies B and A is true, then B is true",
                "A -> B, A, therefore B",
                "Implication elimination rule",
            ],
        ),
        (
            "logic:de_morgan",
            [
                "not (A and B) == (not A) or (not B)",
                "Negation distributes over AND/OR with duality",
                "De Morgan's equivalences",
            ],
        ),
        # Semantic primes
        (
            "semantic:KNOW",
            [
                "To have knowledge of something",
                "know, knowledge, awareness",
                "I know that this is true",
            ],
        ),
        (
            "semantic:WANT",
            [
                "To desire or wish for something",
                "want, desire, wish",
                "I want this to happen",
            ],
        ),
        (
            "semantic:THINK",
            [
                "Mental cognition and reasoning",
                "think, thought, consider",
                "I think therefore I am",
            ],
        ),
        # Computational gates
        (
            "gate:CONDITIONAL",
            [
                "Controls execution flow based on boolean condition",
                "if (x > 10) then",
                "Branch based on condition",
            ],
        ),
        (
            "gate:ITERATION",
            [
                "Repeated execution of a block of code",
                "for item in collection",
                "Loop until condition met",
            ],
        ),
        # Emotion concepts
        (
            "emotion:joy",
            [
                "Joy: A feeling of great pleasure and happiness",
                "Laughing, smiling, and celebrating",
                "A warm, pleasant feeling",
            ],
        ),
        (
            "emotion:fear",
            [
                "Fear: An unpleasant emotion caused by threat of danger",
                "Heart pounding, muscles tense, ready to flee",
                "A sense of dread and vulnerability",
            ],
        ),
    ]


class MLXConceptAdapter(ConceptDiscoveryPort):
    """
    MLX-based implementation of ConceptDiscoveryPort.

    Uses sliding window embedding similarity against a multi-atlas concept
    inventory for cross-domain triangulation. The UnifiedAtlas provides
    probes across all atlas sources (computational gates, sequence invariants,
    semantic primes, emotions, moral foundations, temporal, spatial, social,
    compositional, philosophical, genealogy, metaphor, syntax, safety ethics).
    """

    def __init__(self, embedder: EmbedderPort, concepts: list[tuple[str, list[str]]] | None = None):
        self.embedder = embedder
        # Load from UnifiedAtlas if no custom concepts provided
        self.concepts = concepts if concepts is not None else _load_unified_atlas_concepts()
        self._concept_embeddings = None  # Cache

    async def _ensure_concepts(self):
        if self._concept_embeddings is not None:
            return

        # Embed concepts (prototypes)
        # We take the mean embedding of expressions
        prototypes = []
        for _, expressions in self.concepts:
            vecs = await self.embedder.embed(expressions)
            # vecs is [N, D]
            centroid = mx.mean(vecs, axis=0)  # [D]
            norm_arr = geodesic_norms(
                get_default_backend().reshape(centroid, (1, -1)),
                get_default_backend(),
            )
            get_default_backend().eval(norm_arr)
            norm_val = float(get_default_backend().to_scalar(norm_arr))
            if norm_val > 0.0:
                centroid = centroid / norm_val
            prototypes.append(centroid)

        self._concept_embeddings = mx.stack(prototypes)  # [C, D]

    async def detect_concepts(
        self, response: str, model_id: str, prompt_id: str
    ) -> DetectionResult:
        """Detect concepts in response text.

        All parameters are derived from the data:
        - Window sizes: derived from response length via log2/sqrt scales
        - Stride: derived from window size via sqrt scale
        - Threshold: Otsu thresholding on similarity distribution
        """
        await self._ensure_concepts()

        trimmed = response.strip()
        if not trimmed:
            return DetectionResult(model_id, prompt_id, response, [], 0.0, None)

        # Tokenize - extract (word, start_idx, end_idx) tuples
        words = []
        for m in re.finditer(r"\S+", trimmed):
            words.append((m.group(), m.start(), m.end()))

        if not words:
            return DetectionResult(model_id, prompt_id, response, [], 0.0, None)

        # Collect all candidate detections with similarities
        all_candidates: list[tuple[DetectedConcept, float]] = []

        window_sizes = self._derive_window_sizes(len(words))

        for window_size in window_sizes:
            step = self._derive_stride(window_size)

            for i in range(0, len(words), step):
                if i + window_size > len(words):
                    break

                end_i = min(i + window_size, len(words))
                if end_i <= i:
                    break

                window_words = words[i:end_i]
                start_char = window_words[0][1]
                end_char = window_words[-1][2]
                text_slice = trimmed[start_char:end_char]

                res = await self._detect_in_window(text_slice, start_char, end_char)
                if res:
                    all_candidates.append((res, res.similarity))

        if not all_candidates:
            return DetectionResult(model_id, prompt_id, response, [], 0.0, None)

        # Otsu thresholding: find optimal split in similarity distribution
        similarities = [s for _, s in all_candidates]
        threshold = self._otsu_threshold(similarities)

        # Filter by Otsu-derived threshold
        detections = [d for d, s in all_candidates if s >= threshold]

        # Deduplicate by concept + position
        detections.sort(key=lambda x: x.similarity, reverse=True)
        seen_spans: set[tuple[str, int]] = set()
        span_bucket = self._derive_span_bucket(len(trimmed))
        unique = []
        for d in detections:
            center = (d.character_span.start + d.character_span.stop) // span_bucket
            key = (d.concept_id, center)
            if key not in seen_spans:
                unique.append(d)
                seen_spans.add(key)

        unique.sort(key=lambda x: x.character_span.start)

        mean_similarity = 0.0
        if unique:
            backend = get_default_backend()
            scores = backend.array([d.similarity for d in unique])
            mean_score = backend.mean(scores)
            backend.eval(mean_score)
            mean_similarity = float(backend.to_scalar(mean_score))

        return DetectionResult(
            model_id=model_id,
            prompt_id=prompt_id,
            response_text=response,
            detected_concepts=unique,
            mean_similarity=mean_similarity,
            mean_cross_modal_similarity=None,
        )

    def _derive_window_sizes(self, word_count: int) -> list[int]:
        if word_count <= 0:
            return []
        backend = get_default_backend()
        count_arr = backend.array([float(word_count)])
        unit_arr = count_arr / count_arr
        backend.eval(unit_arr)
        unit = int(backend.to_scalar(unit_arr))

        log2_count = log2_scalar(float(word_count), backend)
        sqrt_count = sqrt_scalar(float(word_count), backend)
        size_short = max(unit, ceil_scalar(log2_count, backend))
        size_mid = max(size_short, ceil_scalar(sqrt_count, backend))

        denom = log2_count if log2_count > float(unit) else float(unit)
        size_long = max(size_mid, ceil_scalar(float(word_count) / denom, backend))

        sizes = {
            min(word_count, size_short),
            min(word_count, size_mid),
            min(word_count, size_long),
        }
        return sorted(size for size in sizes if size >= unit)

    def _derive_stride(self, window_size: int) -> int:
        if window_size <= 0:
            return 0
        backend = get_default_backend()
        size_arr = backend.array([float(window_size)])
        unit_arr = size_arr / size_arr
        backend.eval(unit_arr)
        unit = int(backend.to_scalar(unit_arr))
        stride = ceil_scalar(sqrt_scalar(float(window_size), backend), backend)
        return max(unit, stride)

    def _derive_span_bucket(self, text_len: int) -> int:
        if text_len <= 0:
            return 1
        backend = get_default_backend()
        len_arr = backend.array([float(text_len)])
        unit_arr = len_arr / len_arr
        backend.eval(unit_arr)
        unit = int(backend.to_scalar(unit_arr))
        bucket = ceil_scalar(sqrt_scalar(float(text_len), backend), backend)
        return max(unit, bucket)

    def _otsu_threshold(self, values: list[float]) -> float:
        """Compute Otsu's threshold for optimal bimodal split.

        Finds the threshold that minimizes intra-class variance (or equivalently,
        maximizes inter-class variance) between two classes.
        """
        if len(values) < 2:
            return 0.0

        backend = get_default_backend()
        vals = backend.array(values)
        sorted_vals = backend.sort(vals)
        n = backend.shape(sorted_vals)[0]
        if n < 2:
            return 0.0

        idx = backend.arange(1, n)
        cumsum = backend.cumsum(sorted_vals)
        sum0 = backend.take(cumsum, idx - 1)
        total = backend.sum(sorted_vals)

        idx_float = backend.astype(idx, sorted_vals.dtype)
        denom1 = float(n) - idx_float
        mean0 = sum0 / idx_float
        sum1 = total - sum0
        mean1 = sum1 / denom1

        w0 = idx_float / float(n)
        w1 = 1.0 - w0
        variance = w0 * w1 * (mean0 - mean1) ** 2

        best_idx_arr = backend.argmax(variance)
        backend.eval(best_idx_arr)
        best_idx = int(backend.to_scalar(best_idx_arr))
        threshold_index = backend.take(idx, backend.array([best_idx]))
        threshold_val = backend.take(sorted_vals, threshold_index)
        backend.eval(threshold_val)
        return float(backend.to_scalar(threshold_val))

    async def _detect_in_window(self, text: str, start: int, end: int) -> DetectedConcept | None:
        vec = await self.embedder.embed([text])  # [1, D]
        vec = vec[0]

        # Cosine sim against concept prototypes
        # concepts: [C, D]
        # vec: [D]
        sims = self._concept_embeddings @ vec  # [C]
        if isinstance(sims, mx.array):
            # Helper to get argmax and max
            idx = mx.argmax(sims).item()
            score = sims[idx].item()

            concept_id = self.concepts[idx][0]

            # Extract category from concept_id (format: "source:id")
            # e.g., "semantic_prime:KNOW" -> category="semantic_prime"
            # e.g., "computational_gate:3" -> category="computational_gate"
            if ":" in concept_id:
                category = concept_id.split(":")[0]
            else:
                category = "general"

            return DetectedConcept(
                concept_id=concept_id,
                category=category,
                similarity=score,
                character_span=slice(start, end),
                trigger_text=text,
            )
        return None

    async def compare_results(
        self, result_a: DetectionResult, result_b: DetectionResult
    ) -> ConceptComparisonResult:
        # Simple set intersection logic
        set_a = set(result_a.concept_sequence)
        set_b = set(result_b.concept_sequence)

        aligned = sorted(list(set_a.intersection(set_b)))
        unique_a = sorted(list(set_a - set_b))
        unique_b = sorted(list(set_b - set_a))

        return ConceptComparisonResult(
            model_a=result_a.model_id,
            model_b=result_b.model_id,
            concept_path_a=result_a.concept_sequence,
            concept_path_b=result_b.concept_sequence,
            cka=None,  # requires full activation history usually
            cosine_similarity=None,
            aligned_concepts=aligned,
            unique_to_a=unique_a,
            unique_to_b=unique_b,
        )
