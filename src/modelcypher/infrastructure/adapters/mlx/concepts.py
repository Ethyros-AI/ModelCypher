
from typing import List, Any, Optional
import mlx.core as mx
import time
import re

from modelcypher.core.ports.concepts import ConceptDiscoveryPort
from modelcypher.core.ports.embeddings import EmbedderPort
from modelcypher.core.domain.geometry.types import (
    ConceptConfiguration, DetectionResult, ConceptComparisonResult, DetectedConcept
)

class MLXConceptAdapter(ConceptDiscoveryPort):
    """
    MLX-based implementation of ConceptDiscoveryPort.
    Uses sliding window embedding similarity.
    """
    
    def __init__(self, embedder: EmbedderPort):
        self.embedder = embedder
        # TODO: Load a real Concept Inventory (Atlas).
        # For now, we seed with dummy concepts for verification.
        self.concepts = [
            ("RECURRENCE", ["recurrence", "cycle", "repeat", "loop"]),
            ("SYMMETRY", ["symmetry", "balance", "reflection", "equal"]),
            ("EMERGENCE", ["emergence", "arise", "complex", "system"]),
            ("AGENCY", ["I am", "I think", "my will", "autonomy"])
        ]
        self._concept_embeddings = None # Cache
    
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
        vec = await self.embedder.embed([text]) # [1, D]
        vec = vec[0]
        
        # Cosine sim against concept prototypes
        # concepts: [C, D]
        # vec: [D]
        sims = self._concept_embeddings @ vec # [C]
        if isinstance(sims, mx.array): # Should be
             # Helper to get argmax and max
             idx = mx.argmax(sims).item()
             score = sims[idx].item()
             
             concept_name = self.concepts[idx][0]
             
             return DetectedConcept(
                 concept_id=concept_name,
                 category="general",
                 confidence=score,
                 character_span=slice(start, end),
                 trigger_text=text
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
