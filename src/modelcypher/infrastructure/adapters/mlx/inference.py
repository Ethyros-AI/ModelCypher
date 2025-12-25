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


import uuid
from typing import Any, AsyncGenerator

from modelcypher.core.domain.inference.adapter_pool import AdapterPreloadPriority, MLXAdapterPool
from modelcypher.core.domain.inference.comparison import CheckpointComparisonCoordinator

# Import Implementations (Currently in core/domain/inference, acting as MLX internals)
from modelcypher.core.domain.inference.dual_path_mlx import DualPathGenerator
from modelcypher.core.domain.inference.types import (
    AdapterSwapResult,
    ComparisonEvent,
    ComparisonTimeouts,
    DualPathGeneratorConfiguration,
)
from modelcypher.ports.async_inference import InferenceEnginePort


class MLXInferenceAdapter(InferenceEnginePort):
    def __init__(self):
        self.adapter_pool = MLXAdapterPool()
        self.comparison_coordinator = CheckpointComparisonCoordinator()

    async def generate_dual_path(
        self, prompt: str, config: DualPathGeneratorConfiguration
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Map port types to implementation types
        from modelcypher.core.domain.inference.dual_path_mlx import (
            DualPathGeneratorConfiguration as ImplConfig,
        )

        impl_config = ImplConfig(
            base_model_path=config.base_model_path,
            adapter_path=config.adapter_path,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            max_kl_threshold=config.max_kl_threshold,
            burst_length_limit=config.burst_length_limit,
            accumulated_kl_limit=config.accumulated_kl_limit,
        )

        generator = DualPathGenerator(impl_config)
        async for chunk in generator.generate(prompt):
            yield chunk

    async def compare_checkpoints(
        self,
        checkpoints: list[str],
        prompt: str,
        config: DualPathGeneratorConfiguration,
        timeouts: ComparisonTimeouts,
    ) -> AsyncGenerator[ComparisonEvent, None]:
        # Map config
        from modelcypher.core.domain.inference.dual_path_mlx import (
            DualPathGeneratorConfiguration as ImplConfig,
        )

        impl_config = ImplConfig(
            base_model_path=checkpoints[0],  # Dummy base? Or allow None?
            adapter_path=None,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        # Note: ComparisonCoordinator takes list of checkpoints and config.
        # It handles iteration.

        async for evt in self.comparison_coordinator.compare(checkpoints, prompt, impl_config):
            # Map internal Event to Port Event
            # They should be structurally identical if I copied fields correctly.
            # But technically different classes.
            yield ComparisonEvent(
                type=evt.type,  # Enum mapping might be needed if enums differ reference
                index=evt.index,
                path=evt.path,
                text=evt.text,
                result=evt.result,  # Result object mapping?
                error=evt.error,
            )

    async def pool_register_model(self, model_id: str, load_fn: Any, unload_fn: Any) -> None:
        await self.adapter_pool.register_model(model_id, load_fn, unload_fn)

    async def pool_preload_adapter(self, adapter_id: uuid.UUID, path: str, priority: int) -> None:
        # Map int to Enum
        prio_enum = AdapterPreloadPriority.NORMAL
        if priority == 1:
            prio_enum = AdapterPreloadPriority.HIGH
        elif priority == 2:
            prio_enum = AdapterPreloadPriority.CRITICAL

        await self.adapter_pool.preload(adapter_id, path, prio_enum)

    async def pool_swap_adapter(
        self, to_adapter_id: uuid.UUID | None, model_id: str
    ) -> AdapterSwapResult:
        res = await self.adapter_pool.swap(to_adapter_id, model_id)
        # Map implementation Result to Domain Result
        return AdapterSwapResult(
            previous_adapter_id=res.previous_adapter_id,
            new_adapter_id=res.new_adapter_id,
            swap_duration_ms=res.swap_duration_ms,
            was_cache_hit=res.was_cache_hit,
        )

    async def pool_evict_adapter(self, adapter_id: uuid.UUID) -> None:
        await self.adapter_pool.evict(adapter_id)
