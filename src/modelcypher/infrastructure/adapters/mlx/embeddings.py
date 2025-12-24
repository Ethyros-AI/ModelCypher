
from typing import Any
import mlx.core as mx
from modelcypher.ports.async_embeddings import EmbedderPort

class MockMLXEmbedder(EmbedderPort):
    """
    A mock embedder for testing/verification. 
    Returns random compatible vectors.
    """
    def __init__(self, dim: int = 768):
        self._dim = dim
        
    async def embed(self, texts: list[str]) -> Any:
        n = len(texts)
        if n == 0: return mx.zeros((0, self._dim))
        # Random vectors normalized
        vecs = mx.random.normal((n, self._dim))
        norms = mx.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms
        
    async def dimension(self) -> int:
        return self._dim
