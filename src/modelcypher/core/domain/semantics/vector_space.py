import mlx.core as mx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ConceptNode:
    id: str
    vector: mx.array
    metadata: Dict[str, str]

class ConceptVectorSpace:
    """
    Manages a high-dimensional vector space for semantic concepts.
    Provides storage and similarity search operations.
    """
    
    def __init__(self, dimension: int = 4096):
        self.dimension = dimension
        self.concepts: Dict[str, ConceptNode] = {}
        
    def add_concept(self, concept_id: str, vector: mx.array, metadata: Optional[Dict] = None):
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}")
            
        # Normalize on insertion for cosine similarity
        norm = mx.linalg.norm(vector)
        normalized = vector / (norm + 1e-6)
        
        self.concepts[concept_id] = ConceptNode(
            id=concept_id,
            vector=normalized,
            metadata=metadata or {}
        )
        
    def find_nearest_neighbors(self, query_vector: mx.array, k: int = 5) -> List[Tuple[str, float]]:
        if not self.concepts:
            return []
            
        # 1. Prepare Query
        q_norm = mx.linalg.norm(query_vector)
        q = query_vector / (q_norm + 1e-6)
        
        # 2. Stack Concept Vectors
        ids = list(self.concepts.keys())
        matrix = mx.stack([self.concepts[id].vector for id in ids])
        
        # 3. Compute Cosine Similarity (Dot product of normalized vectors)
        scores = q @ matrix.T
        
        # 4. Top K
        # MLX argpartition/topk
        # For small N, simple sort is fine
        indices = mx.argsort(scores, descending=True)[:k]
        
        results = []
        for idx in indices.tolist():
             results.append((ids[idx], scores[idx].item()))
             
        return results

    def arithmetics(self, positive: List[str], negative: List[str]) -> mx.array:
        """
        Performs vector arithmetic: sum(pos) - sum(neg)
        """
        result = mx.zeros((self.dimension,))
        
        for p in positive:
            if p in self.concepts:
                result = result + self.concepts[p].vector
                
        for n in negative:
            if n in self.concepts:
                result = result - self.concepts[n].vector
                
        return result
