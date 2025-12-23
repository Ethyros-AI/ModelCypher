
from typing import List, Any, Optional
import mlx.core as mx
import time
import re
import logging

from modelcypher.ports.concept_discovery import ConceptDiscoveryPort
from modelcypher.ports.async_embeddings import EmbedderPort
from modelcypher.core.domain.geometry.types import (
    ConceptConfiguration, DetectionResult, ConceptComparisonResult, DetectedConcept
)

logger = logging.getLogger(__name__)


def _load_unified_atlas_concepts() -> List[tuple[str, List[str]]]:
    """Load concepts from the UnifiedAtlas (321 probes across 7 atlas sources).

    The UnifiedAtlas triangulates across:
    - Computational Gates (76): Programming concept primitives
    - Sequence Invariants (68): Mathematical anchors (Fibonacci, primes, logic)
    - Semantic Primes (65): Linguistic universals from Goddard & Wierzbicka (2014)
    - Emotion Concepts (32): Plutchik's wheel with VAD dimensions
    - Moral Concepts (30): Haidt's Moral Foundations Theory
    - Temporal Concepts (25): Arrow of time, duration, causality
    - Social Concepts (25): Power hierarchy, kinship, formality

    Returns:
        List of (concept_id, [support_texts]) tuples for embedding triangulation.
    """
    try:
        from modelcypher.core.domain.agents.unified_atlas import (
            UnifiedAtlasInventory,
            AtlasSource,
        )

        probes = UnifiedAtlasInventory.all_probes()
        concepts: List[tuple[str, List[str]]] = []

        for probe in probes:
            # Create concept ID with source prefix for domain clarity
            concept_id = f"{probe.source.value}:{probe.id}"

            # Build support texts from probe metadata
            support_texts: List[str] = []

            # Primary text: name and description
            support_texts.append(f"{probe.name}: {probe.description}")

            # Add probe support texts if available
            for text in probe.support_texts[:3]:  # Limit to 3 per probe
                support_texts.append(text)

            # Ensure at least 2 support texts for better centroid estimation
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


def _fallback_concepts() -> List[tuple[str, List[str]]]:
    """Fallback concept inventory when UnifiedAtlas is unavailable.

    Provides essential mathematical and semantic anchors for basic operation.
    """
    return [
        # Mathematical invariants
        ("sequence:fibonacci", [
            "Fibonacci sequence: each term is the sum of the two previous terms",
            "0, 1, 1, 2, 3, 5, 8, 13, 21, 34",
            "F(n) = F(n-1) + F(n-2)",
            "Golden ratio phi = 1.618...",
        ]),
        ("sequence:primes", [
            "Prime numbers: divisible only by 1 and themselves",
            "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
            "The atoms of arithmetic",
            "Fundamental theorem of arithmetic",
        ]),
        # Logical invariants
        ("logic:modus_ponens", [
            "If A implies B and A is true, then B is true",
            "A -> B, A, therefore B",
            "Implication elimination rule",
        ]),
        ("logic:de_morgan", [
            "not (A and B) == (not A) or (not B)",
            "Negation distributes over AND/OR with duality",
            "De Morgan's equivalences",
        ]),
        # Semantic primes
        ("semantic:KNOW", [
            "To have knowledge of something",
            "know, knowledge, awareness",
            "I know that this is true",
        ]),
        ("semantic:WANT", [
            "To desire or wish for something",
            "want, desire, wish",
            "I want this to happen",
        ]),
        ("semantic:THINK", [
            "Mental cognition and reasoning",
            "think, thought, consider",
            "I think therefore I am",
        ]),
        # Computational gates
        ("gate:CONDITIONAL", [
            "Controls execution flow based on boolean condition",
            "if (x > 10) then",
            "Branch based on condition",
        ]),
        ("gate:ITERATION", [
            "Repeated execution of a block of code",
            "for item in collection",
            "Loop until condition met",
        ]),
        # Emotion concepts
        ("emotion:joy", [
            "Joy: A feeling of great pleasure and happiness",
            "Laughing, smiling, and celebrating",
            "A warm, pleasant feeling",
        ]),
        ("emotion:fear", [
            "Fear: An unpleasant emotion caused by threat of danger",
            "Heart pounding, muscles tense, ready to flee",
            "A sense of dread and vulnerability",
        ]),
    ]


class MLXConceptAdapter(ConceptDiscoveryPort):
    """
    MLX-based implementation of ConceptDiscoveryPort.

    Uses sliding window embedding similarity against a multi-atlas concept
    inventory for cross-domain triangulation. The UnifiedAtlas provides
    321 probes across 7 atlas sources: computational gates, sequence invariants,
    semantic primes, emotions, moral foundations, temporal concepts, and social concepts.
    """

    def __init__(self, embedder: EmbedderPort, concepts: Optional[List[tuple[str, List[str]]]] = None):
        self.embedder = embedder
        # Load from UnifiedAtlas if no custom concepts provided
        self.concepts = concepts if concepts is not None else _load_unified_atlas_concepts()
        self._concept_embeddings = None  # Cache
    
    async def _ensure_concepts(self):
        if self._concept_embeddings is not None: return
        
        # Embed concepts (prototypes)
        # We take the mean embedding of expressions
        prototypes = []
        for _, expressions in self.concepts:
            vecs = await self.embedder.embed(expressions)
            # vecs is [N, D]
            centroid = mx.mean(vecs, axis=0) # [D]
            centroid = centroid / mx.linalg.norm(centroid)
            prototypes.append(centroid)
            
        self._concept_embeddings = mx.stack(prototypes) # [C, D]
        
    async def detect_concepts(
        self,
        response: str,
        model_id: str,
        prompt_id: str,
        config: ConceptConfiguration
    ) -> DetectionResult:
        await self._ensure_concepts()
        
        trimmed = response.strip()
        if not trimmed:
            return DetectionResult(model_id, prompt_id, response, [], 0.0, None)
            
        # Tokenize (simple split for now)
        # We need byte offsets for spans.
        # Python string slicing is by CHAR count, Swift was NSRange/Range<String.Index>.
        # `types.py` uses `slice`.
        
        # We need a list of (word, start_idx, end_idx)
        # Simple regex tokenizer
        words = []
        for m in re.finditer(r'\S+', trimmed):
            words.append((m.group(), m.start(), m.end()))
            
        if not words:
            return DetectionResult(model_id, prompt_id, response, [], 0.0, None)
            
        detections = []
        
        # Multi-scale windows
        for window_size in config.window_sizes:
            step = max(1, config.stride)
            
            for i in range(0, len(words), step):
                # Check bounds
                if i + window_size > len(words) + step: 
                    # Allow one partial window at end? Swift logic: 
                    # while windowStart + windowSize <= words.count
                    # It was strict.
                    break
                    
                end_i = min(i + window_size, len(words))
                if end_i <= i: break
                
                window_words = words[i:end_i]
                start_char = window_words[0][1]
                end_char = window_words[-1][2]
                text_slice = trimmed[start_char:end_char]
                
                # Detect
                res = await self._detect_in_window(text_slice, start_char, end_char)
                if res:
                    # Filter threshold
                    if res.confidence >= config.detection_threshold:
                        detections.append(res)
                        
        # Deduplicate
        detections.sort(key=lambda x: x.confidence, reverse=True)
        unique = []
        # Simple interval overlap check or keep best per span?
        # Swift kept highest confidence per "span-concept" key.
        # And sorted by position.
        
        # Using simple binning for this port
        seen_spans = set()
        for d in detections:
            # unique key: concept + rough span center
            center = (d.character_span.start + d.character_span.stop) // 10 # 10 char resolution
            key = (d.concept_id, center)
            if key not in seen_spans:
                unique.append(d)
                seen_spans.add(key)
                
        unique.sort(key=lambda x: x.character_span.start)
        
        # Limit
        unique = unique[:config.max_concepts_per_response]
        
        mean_conf = 0.0
        if unique:
            mean_conf = sum(d.confidence for d in unique) / len(unique)
            
        return DetectionResult(
            model_id=model_id,
            prompt_id=prompt_id,
            response_text=response,
            detected_concepts=unique,
            mean_confidence=mean_conf,
            mean_cross_modal_confidence=None
        )

    async def _detect_in_window(self, text: str, start: int, end: int) -> Optional[DetectedConcept]:
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
                confidence=score,
                character_span=slice(start, end),
                trigger_text=text,
            )
        return None

    async def compare_results(
        self,
        result_a: DetectionResult,
        result_b: DetectionResult
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
            cka=None, # requires full activation history usually
            cosine_similarity=None, 
            aligned_concepts=aligned,
            unique_to_a=unique_a,
            unique_to_b=unique_b
        )
