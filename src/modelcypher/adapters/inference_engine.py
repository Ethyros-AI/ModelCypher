# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Unified Inference Engine - ONE implementation using Backend protocol.

All tensor operations and model loading go through Backend.
No framework imports (mlx, torch, jax) in this file.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from modelcypher.ports.inference import HiddenStateEngine
from modelcypher.utils.locks import FileLock
from modelcypher.utils.model_context import resolve_context_limit
from modelcypher.utils.paths import get_modelcypher_home

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCacheEntry:
    """Cached model and tokenizer."""

    model: Any
    tokenizer: Any
    adapter_path: str | None


@dataclass
class GenerationResult:
    """Result of text generation."""

    text: str
    token_count: int
    tokens_per_second: float
    time_to_first_token: float | None
    total_duration: float
    stop_reason: str = "length"


@dataclass
class InferenceResult:
    """Result of a single inference run."""

    prompt: str
    response: str
    token_count: int
    tokens_per_second: float
    time_to_first_token: float | None
    total_duration: float
    model_path: str
    adapter_path: str | None = None
    stop_reason: str = "length"


@dataclass
class BatchInferResult:
    """Result of batched inference."""

    model_id: str
    prompts_file: str
    results: list[dict[str, Any]]
    total_prompts: int
    successful: int
    failed: int
    total_tokens: int
    total_duration: float
    average_tokens_per_second: float


class InferenceEngine(HiddenStateEngine):
    """Unified inference engine using Backend protocol.

    All framework operations delegate to self._backend.
    No framework imports in this file.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        base_path: Path | None = None,
    ) -> None:
        """Initialize inference engine.

        Args:
            backend: Backend for framework operations. Auto-detects if None.
            base_path: Base directory for locks and caches.
        """
        if backend is None:
            from modelcypher.core.domain._backend import get_default_backend

            backend = get_default_backend()
        self._backend = backend
        self.base_path = base_path or get_modelcypher_home()
        self.lock = FileLock(self.base_path / "inference.lock")
        self._model_cache: dict[tuple[str, str | None], ModelCacheEntry] = {}
        self._context_cache: dict[str, int] = {}

    def _validate_model_assets(self, model_path: Path) -> bool:
        """Validate model directory has required files."""
        config_path = model_path / "config.json"
        if config_path.exists():
            return True
        raise ValueError(f"config.json not found in model directory: {model_path}")

    def _load_model(self, model_path: Path, adapter: str | None) -> ModelCacheEntry:
        """Load model using Backend, with caching."""
        adapter_path = Path(adapter).expanduser().resolve() if adapter else None
        cache_key = (str(model_path), str(adapter_path) if adapter_path else None)

        cached = self._model_cache.get(cache_key)
        if cached is not None:
            return cached

        # Check for geometric LoRA adapter
        if adapter_path and self._is_geometric_lora_adapter(adapter_path):
            logger.info("Loading geometric LoRA adapter from %s", adapter_path)
            model, tokenizer = self._load_geometric_lora_adapter(
                model_path, adapter_path
            )
        else:
            # Use Backend for model loading
            model, tokenizer = self._backend.load_model(
                str(model_path),
                adapter_path=str(adapter_path) if adapter_path else None,
            )

        entry = ModelCacheEntry(
            model=model,
            tokenizer=tokenizer,
            adapter_path=str(adapter_path) if adapter_path else None,
        )
        self._model_cache[cache_key] = entry
        return entry

    def _is_geometric_lora_adapter(self, adapter_path: Path) -> bool:
        """Check if adapter is a geometric LoRA adapter."""
        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            return False
        try:
            with open(config_path) as f:
                config = json.load(f)
            return config.get("type") == "geometric_lora"
        except Exception:
            return False

    def _load_geometric_lora_adapter(
        self, model_path: Path, adapter_path: Path
    ) -> tuple[Any, Any]:
        """Load model with geometric LoRA adapter applied.

        Uses Backend for all tensor operations.
        """
        # Load base model
        model, tokenizer = self._backend.load_model(str(model_path))

        # Load adapter config
        config_path = adapter_path / "adapter_config.json"
        with open(config_path) as f:
            adapter_config = json.load(f)

        # Load adapter weights
        weights_path = adapter_path / "adapters.safetensors"
        if not weights_path.exists():
            weights_path = adapter_path / "adapter.safetensors"

        weights = self._backend.load_binary_weights(weights_path)

        # Apply adapter weights to model
        # This uses Backend's tree operations
        model_params = self._backend.tree_flatten(model.parameters())
        model_params_dict = dict(model_params)

        for key, value in weights.items():
            if key in model_params_dict:
                model_params_dict[key] = value

        # Unflatten and update model
        # Note: model.update() is framework-specific but models are loaded by Backend
        try:
            model.update(model_params_dict)
        except AttributeError:
            # Some model types may not have update()
            logger.warning("Model does not support update(), adapter may not be applied")

        return model, tokenizer

    def _derive_max_tokens(
        self, model_path: Path, prompt: str, tokenizer: Any
    ) -> int:
        """Derive max tokens available for generation."""
        context_limit = resolve_context_limit(
            model_path, tokenizer, self._context_cache
        )
        if context_limit is None:
            return 512  # Safe default

        prompt_tokens = len(self._backend.encode_tokens(tokenizer, prompt))
        available = context_limit - prompt_tokens - 10  # Small margin
        return max(min(available, 2048), 1)

    def infer(
        self,
        model: str,
        prompt: str,
        adapter: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run inference and return structured results.

        Args:
            model: Path to model directory.
            prompt: Input prompt.
            adapter: Optional path to adapter.
            max_tokens: Max tokens to generate. Auto-derived if None.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            Dict with response, token_count, tokens_per_second, etc.
        """
        model_path = Path(model).expanduser().resolve()
        self._validate_model_assets(model_path)

        entry = self._load_model(model_path, adapter)

        if max_tokens is None:
            max_tokens = self._derive_max_tokens(model_path, prompt, entry.tokenizer)

        start_time = time.time()

        # Use Backend for generation
        response = self._backend.generate(
            entry.model,
            entry.tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        duration = time.time() - start_time
        token_count = len(self._backend.encode_tokens(entry.tokenizer, response))
        tokens_per_second = token_count / max(duration, 0.001)

        return {
            "response": response,
            "token_count": token_count,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token": None,  # Not available without streaming
            "total_duration": duration,
            "model_path": str(model_path),
            "adapter_path": entry.adapter_path,
        }

    def capture_hidden_states(
        self,
        model: str,
        prompt: str,
        adapter: str | None = None,
        target_layers: set[int] | None = None,
    ) -> dict[int, list[float]]:
        """Return hidden states keyed by layer index.

        Uses Backend's activation collection.
        """
        model_path = Path(model).expanduser().resolve()
        self._validate_model_assets(model_path)

        entry = self._load_model(model_path, adapter)

        # Use Backend for hidden state collection
        hidden_states = self._backend.collect_hidden_activations(
            entry.model, entry.tokenizer, [prompt]
        )

        # hidden_states is list[dict[int, Array]] - one per input
        if not hidden_states:
            return {}

        result: dict[int, list[float]] = {}
        for layer_idx, activations in hidden_states[0].items():
            if target_layers is not None and layer_idx not in target_layers:
                continue
            # Convert to list for protocol compliance
            result[layer_idx] = self._backend.tolist(activations)

        return result

    def clear_cache(self) -> None:
        """Clear model cache to free memory."""
        self._model_cache.clear()
        self._context_cache.clear()


def get_inference_engine(backend: "Backend | None" = None) -> InferenceEngine:
    """Get an inference engine instance."""
    return InferenceEngine(backend)


def load_model_and_tokenizer(
    model_path: str | Path, adapter_path: str | Path | None = None
) -> tuple[Any, Any]:
    """Load a model and tokenizer using the default backend.

    Args:
        model_path: Path to model directory.
        adapter_path: Optional path to adapter.

    Returns:
        Tuple of (model, tokenizer).
    """
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    return backend.load_model(
        str(model_path),
        adapter_path=str(adapter_path) if adapter_path else None,
    )


__all__ = [
    "InferenceEngine",
    "InferenceResult",
    "GenerationResult",
    "BatchInferResult",
    "ModelCacheEntry",
    "get_inference_engine",
    "load_model_and_tokenizer",
]
