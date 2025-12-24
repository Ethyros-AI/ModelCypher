
from typing import Protocol, Any, AsyncGenerator, runtime_checkable
from modelcypher.core.domain.inference.types import (
    DualPathGeneratorConfiguration, SecurityScanMetrics,
    ComparisonEvent, ComparisonResult, ComparisonTimeouts,
    AdapterPoolConfiguration, AdapterSwapResult, AdapterPoolEntry
)
import uuid

@runtime_checkable
class InferenceEnginePort(Protocol):
    """
    Abstract interface for model inference and orchestration.
    """
    
    async def generate_dual_path(
        self,
        prompt: str,
        config: DualPathGeneratorConfiguration
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generates text while monitoring entropy dynamics between base and adapter.
        Yields chunks with tokens and safety metrics.
        """
        ... # Protocol methods can have yield

    async def compare_checkpoints(
        self,
        checkpoints: list[str],
        prompt: str,
        config: DualPathGeneratorConfiguration,
        timeouts: ComparisonTimeouts
    ) -> AsyncGenerator[ComparisonEvent, None]:
        """
        Orchestrates side-by-side comparison of multiple checkpoints.
        """
        ...
        
    # --- Adapter Pool Interface ---
    
    async def pool_register_model(self, model_id: str, load_fn: Any, unload_fn: Any) -> None:
        ...
        
    async def pool_preload_adapter(self, adapter_id: uuid.UUID, path: str, priority: int) -> None:
        ...
        
    async def pool_swap_adapter(self, to_adapter_id: uuid.UUID | None, model_id: str) -> AdapterSwapResult:
        ...
    
    async def pool_evict_adapter(self, adapter_id: uuid.UUID) -> None:
        ...
