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

"""
Checkpoint Comparison Coordinator for side-by-side model evaluation.

Orchestrates parallel inference across multiple model checkpoints, enabling
real-time comparison of model outputs. Designed for evaluating training
progress, comparing merge candidates, or A/B testing model variants.

Architecture:
    The coordinator uses DualPathGenerator for inference, supporting both
    base models and LoRA-adapted variants. An async event stream provides
    token-by-token updates for streaming UI applications. An exclusive
    asyncio.Lock ensures only one comparison runs at a time, preventing
    resource contention on GPU memory.

    Flow:
        1. Acquire exclusive lease (asyncio.Lock)
        2. Prefetch/load each checkpoint sequentially
        3. Run inference on each checkpoint, streaming tokens
        4. Yield metrics and complete responses
        5. Release lease

Event Types:
    PREFETCH_STARTED: Model loading phase begins for a checkpoint.
    PREFETCH_FINISHED: Model successfully loaded and ready for inference.
    PREFETCH_FAILED: Model loading failed (e.g., corrupt weights, OOM).
    CHECKPOINT_STARTED: Inference begins on a checkpoint.
    TOKEN: Individual token generated. Use for streaming UI updates.
    CHECKPOINT_FINISHED: Complete response with inference metrics.
    CHECKPOINT_FAILED: Error during inference (e.g., generation failure).

Usage:
    coordinator = CheckpointComparisonCoordinator(generator_cls=MyGeneratorClass)

    async for event in coordinator.compare(
        checkpoints=["/path/to/base", "/path/to/merged"],
        prompt="Explain quantum entanglement.",
    ):
        if event.type == EventType.TOKEN:
            print(event.text, end="", flush=True)
        elif event.type == EventType.CHECKPOINT_FINISHED:
            print(f"\\n[{event.result.checkpoint_path}] Done")
            print(f"  Tokens/sec: {event.result.metrics.tokens_per_second:.1f}")

Note:
    This module was ported from CheckpointComparisonCoordinator.swift.
    The async generator pattern replaces Swift's AsyncStream, and the
    asyncio.Lock replaces the actor-based exclusive lease mechanism.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Protocol

logger = logging.getLogger("modelcypher.comparison")


class ComparisonError(Exception):
    """Errors raised by checkpoint comparison."""


@dataclass
class InferenceComparisonResult:
    checkpoint_path: str
    response: str
    metrics: Any  # InferenceMetrics type placeholder


class ComparisonEventType(Enum):
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
    result: InferenceComparisonResult | None = None
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

    def __init__(
        self,
        generator_cls: type | None = None,
        inference_service: InferenceServiceProtocol | None = None,
    ):
        if generator_cls is None and inference_service is None:
            raise ValueError("generator_cls or inference_service is required")
        self._generator_cls = generator_cls
        self._inference_service = inference_service
        self._lock = asyncio.Lock()

    async def compare(
        self,
        checkpoints: list[str],
        prompt: str,
    ) -> AsyncGenerator[ComparisonEvent, None]:
        prefetched_generators: dict[str, Any] = {}
        prefetch_errors: dict[str, str] = {}

        async with self._lock:  # Exclusive lease
            for i, ckpt in enumerate(checkpoints):
                yield ComparisonEvent(EventType.PREFETCH_STARTED, i, ckpt)
                try:
                    if self._inference_service is not None:
                        await self._inference_service.load_model(ckpt)
                    else:
                        if self._generator_cls is None:
                            raise ComparisonError("No generator class configured")
                        prefetched_generators[ckpt] = self._generator_cls(
                            base_model_path=ckpt,
                            adapter_path=None,
                        )
                    yield ComparisonEvent(EventType.PREFETCH_FINISHED, i, ckpt)
                except Exception as exc:
                    prefetch_errors[ckpt] = str(exc)
                    yield ComparisonEvent(
                        EventType.PREFETCH_FAILED,
                        i,
                        path=ckpt,
                        error=str(exc),
                    )

            for i, ckpt in enumerate(checkpoints):
                if ckpt in prefetch_errors:
                    yield ComparisonEvent(
                        EventType.CHECKPOINT_FAILED,
                        i,
                        path=ckpt,
                        error=prefetch_errors[ckpt],
                    )
                    continue
                yield ComparisonEvent(EventType.CHECKPOINT_STARTED, i, ckpt)

                try:
                    # Dynamically configure generator for this checkpoint
                    # If we have a service, use it. If not, instantiate DualPathGenerator (or simple Generator)
                    # For strict 1:1, we should rely on injected service.
                    # But verifying imports is easier if we just use our existing DualPathGenerator class for now as a "Service".

                    response_text = ""
                    metrics = None

                    if self._inference_service is not None:
                        await self._inference_service.load_model(ckpt)
                        async for chunk in self._inference_service.generate(
                            prompt,
                        ):
                            if chunk["type"] == "token":
                                txt = chunk["text"]
                                response_text += txt
                                yield ComparisonEvent(EventType.TOKEN, i, text=txt)
                            elif chunk["type"] == "metrics":
                                metrics = chunk["metrics"]
                    else:
                        if self._generator_cls is None:
                            raise ComparisonError("No generator class configured")
                        generator = prefetched_generators.get(ckpt)
                        if generator is None:
                            generator = self._generator_cls(
                                base_model_path=ckpt,
                                adapter_path=None,
                            )

                        async for chunk in generator.generate(prompt):
                            if chunk["type"] == "token":
                                txt = chunk["text"]
                                response_text += txt
                                yield ComparisonEvent(EventType.TOKEN, i, text=txt)
                            elif chunk["type"] == "metrics":
                                metrics = chunk["metrics"]

                    result = InferenceComparisonResult(ckpt, response_text, metrics)
                    yield ComparisonEvent(EventType.CHECKPOINT_FINISHED, i, result=result)

                except Exception as e:
                    logger.error(f"Checkpoint failed {ckpt}: {e}")
                    yield ComparisonEvent(EventType.CHECKPOINT_FAILED, i, path=ckpt, error=str(e))
