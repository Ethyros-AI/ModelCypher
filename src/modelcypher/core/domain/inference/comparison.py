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

import asyncio
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Protocol

from modelcypher.core.domain.inference._platform import (
    get_dual_path_config_class,
    get_dual_path_generator_class,
)

logger = logging.getLogger("modelcypher.comparison")


class ComparisonError(Exception):
    pass


@dataclass
class ComparisonResult:
    checkpoint_path: str
    response: str
    metrics: Any  # InferenceMetrics type placeholder


class EventType(Enum):
    PREFETCH_STARTED = "prefetch_started"
    PREFETCH_FINISHED = "prefetch_finished"
    PREFETCH_FAILED = "prefetch_failed"
    CHECKPOINT_STARTED = "checkpoint_started"
    TOKEN = "token"
    CHECKPOINT_FINISHED = "checkpoint_finished"
    CHECKPOINT_FAILED = "checkpoint_failed"


@dataclass
class ComparisonEvent:
    type: EventType
    index: int
    path: str | None = None
    text: str | None = None
    result: ComparisonResult | None = None
    error: str | None = None


class InferenceServiceProtocol(Protocol):
    # Abstract interface for what the coordinator needs
    async def load_model(self, path: str): ...
    def generate(self, prompt: str, **kwargs) -> AsyncGenerator[dict[str, Any], None]: ...


class CheckpointComparisonCoordinator:
    """
    Orchestrates side-by-side checkpoint comparison.
    Ported from CheckpointComparisonCoordinator.swift.
    """

    def __init__(self, inference_service: InferenceServiceProtocol | None = None):
        # We can inject a service, or use DualPathGenerator directly if that's the standard
        # For now, let's assume we create generators on fly or use a provided service.
        self.inference_service = inference_service
        self._lock = asyncio.Lock()

    async def compare(
        self,
        checkpoints: list[str],
        prompt: str,
        config: Any,  # Platform-specific configuration
    ) -> AsyncGenerator[ComparisonEvent, None]:
        uuid.uuid4()
        # Prefetch logic (stubbed as simple log for now, since python models might just load on demand)
        # In real Python implementation, we might warm up cache.

        async with self._lock:  # Exclusive lease
            for i, ckpt in enumerate(checkpoints):
                yield ComparisonEvent(EventType.PREFETCH_STARTED, i, ckpt)
                # simulate prefetch
                yield ComparisonEvent(EventType.PREFETCH_FINISHED, i, ckpt)

            for i, ckpt in enumerate(checkpoints):
                yield ComparisonEvent(EventType.CHECKPOINT_STARTED, i, ckpt)

                try:
                    # Dynamically configure generator for this checkpoint
                    # If we have a service, use it. If not, instantiate DualPathGenerator (or simple Generator)
                    # For strict 1:1, we should rely on injected service.
                    # But verifying imports is easier if we just use our existing DualPathGenerator class for now as a "Service".

                    # Create a specific config for this checkpoint
                    config_class = get_dual_path_config_class()
                    current_config = config_class(
                        base_model_path=ckpt,
                        adapter_path=None,  # Comparison usually implies base weights, or we'd need adapter config
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    )

                    # Instantiate generator (which loads model)
                    # In a real app, this load happens in the service layer with caching.
                    generator_class = get_dual_path_generator_class()
                    generator = generator_class(current_config)

                    response_text = ""
                    metrics = None

                    async for chunk in generator.generate(prompt):
                        if chunk["type"] == "token":
                            txt = chunk["text"]
                            response_text += txt
                            yield ComparisonEvent(EventType.TOKEN, i, text=txt)
                        elif chunk["type"] == "metrics":
                            metrics = chunk["metrics"]

                    result = ComparisonResult(ckpt, response_text, metrics)
                    yield ComparisonEvent(EventType.CHECKPOINT_FINISHED, i, result=result)

                except Exception as e:
                    logger.error(f"Checkpoint failed {ckpt}: {e}")
                    yield ComparisonEvent(EventType.CHECKPOINT_FAILED, i, path=ckpt, error=str(e))
