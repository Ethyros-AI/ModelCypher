
from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class EmbedderPort(Protocol):
    """
    Interface for semantic text embedding.
    """
    async def embed(self, texts: list[str]) -> Any: 
        """
        Embeds a list of texts into a matrix of shape [N, D].
        Returns MLX array or list of lists.
        """
        ...
    
    async def dimension(self) -> int:
        """
        Returns the embedding dimension.
        """
        ...
