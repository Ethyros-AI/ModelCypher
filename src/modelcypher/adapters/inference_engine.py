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
No framework imports in this file.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.ports.inference import HiddenStateEngine
from modelcypher.utils.locks import FileLock, FileLockError
from modelcypher.utils.model_context import resolve_context_limit
from modelcypher.utils.paths import get_modelcypher_home

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)

GEOMETRIC_ADAPTER_WEIGHT_CANDIDATES = (
    "adapters.safetensors",
    "adapter_model.safetensors",
    "adapter.safetensors",
    "lora_weights.safetensors",
    "adapters.bin",
    "adapter_model.bin",
    "adapter_model.pt",
)


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
    stop_reason: str
    model: str
    adapter: str | None = None
    security: "SecurityScanSummary | None" = None


@dataclass
class SecurityScanSummary:
    """Summary of security scan results."""

    anomaly_count: int
    max_anomaly_score: float
    avg_delta: float
    disagreement_rate: float


@dataclass
class InferenceCaseResult:
    """Result of a single inference case in a suite."""

    name: str
    prompt: str
    response: str
    token_count: int
    duration: float
    passed: bool | None
    expected: str | list[str] | None
    error: str | None = None


@dataclass
class InferenceSuiteResult:
    """Result of inference suite execution."""

    model: str
    adapter: str | None
    suite: str
    cases: list[InferenceCaseResult]
    total_cases: int
    passed: int
    failed: int
    total_duration: float
    summary: dict[str, Any] = field(default_factory=dict)
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
        self.lock = FileLock(self.base_path / "training.lock")
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

        # Use Backend.load_model() for all adapter types. mlx_lm.load()
        # handles standard LoRA adapters (including geometric/PiSSA) natively
        # via adapter_config.json + lora_parameters.
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
            json.load(f)

        # Load adapter weights — search repo-supported filenames, dispatch
        # by extension to the correct backend loader.
        weights_path = None
        for candidate in GEOMETRIC_ADAPTER_WEIGHT_CANDIDATES:
            p = adapter_path / candidate
            if p.exists():
                weights_path = p
                break

        if weights_path is None:
            raise FileNotFoundError(
                f"No adapter weights found in {adapter_path}. "
                f"Searched: {', '.join(GEOMETRIC_ADAPTER_WEIGHT_CANDIDATES)}"
            )

        if weights_path.suffix == ".safetensors":
            weights = self._backend.load_safetensors(str(weights_path))
        else:
            weights = self._backend.load_binary_weights(str(weights_path))

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run inference and return structured results.

        Decoding is always greedy (argmax). The forward pass is a deterministic
        geometric map; sampling injects noise that obscures the model's actual
        trajectory. If the greedy path gives the wrong answer, fix the training.

        Args:
            model: Path to model directory.
            prompt: Input prompt.
            adapter: Optional path to adapter.
            max_tokens: Max tokens to generate. Auto-derived if None.
            **kwargs: Additional generation parameters.

        Returns:
            Dict with response, token_count, tokens_per_second, etc.
        """
        model_path = Path(model).expanduser().resolve()
        self._validate_model_assets(model_path)

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        try:
            entry = self._load_model(model_path, adapter)

            if max_tokens is None:
                max_tokens = self._derive_max_tokens(model_path, prompt, entry.tokenizer)

            start_time = time.time()

            # Greedy decoding — no temperature, no sampling noise
            response = self._backend.generate(
                entry.model,
                entry.tokenizer,
                prompt,
                max_tokens=max_tokens,
                **kwargs,
            )

            duration = time.time() - start_time
            token_count = len(self._backend.encode_tokens(entry.tokenizer, response))
            tokens_per_second = token_count / duration if duration > 0.0 else 0.0
            stop_reason = "length" if token_count >= max_tokens else "eos"

            return {
                "response": response,
                "token_count": token_count,
                "tokens_per_second": tokens_per_second,
                "time_to_first_token": None,
                "total_duration": duration,
                "stop_reason": stop_reason,
                "model": str(model_path),
                "adapter": entry.adapter_path,
            }
        finally:
            self.lock.release()

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

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        try:
            # Use Backend for hidden state collection
            hidden_states = self._backend.collect_hidden_activations(
                entry.model,
                entry.tokenizer,
                [prompt],
                layer_indices=sorted(target_layers) if target_layers else None,
            )

            if not hidden_states:
                return {}

            result: dict[int, list[float]] = {}
            for layer_idx, activations in hidden_states.items():
                if target_layers is not None and layer_idx not in target_layers:
                    continue
                # activations: [batch, seq, hidden] or [seq, hidden]
                if hasattr(activations, "ndim") and activations.ndim >= 2:
                    if activations.ndim >= 3:
                        layer_act = activations[0, -1, :]
                    else:
                        layer_act = activations[-1, :]
                else:
                    layer_act = activations
                self._backend.eval(layer_act)
                result[layer_idx] = self._backend.tolist(layer_act)

            return result
        finally:
            self.lock.release()

    def _perform_security_scan(
        self, prompt: str, response: str, model: str
    ) -> SecurityScanSummary:
        """Stub security scan for now (returns raw zeroed metrics)."""
        logger.warning(
            "Security scan not implemented in unified inference engine; "
            "returning zeroed metrics."
        )
        return SecurityScanSummary(
            anomaly_count=0,
            max_anomaly_score=0.0,
            avg_delta=0.0,
            disagreement_rate=0.0,
        )

    def run(
        self,
        model: str,
        prompt: str,
        adapter: str | None = None,
        security_scan: bool = False,
        max_tokens: int | None = None,
    ) -> InferenceResult:
        """Execute inference with optional adapter and security scanning."""
        model_path = Path(model).expanduser().resolve()
        self._validate_model_assets(model_path)

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        try:
            entry = self._load_model(model_path, adapter)
            if max_tokens is None:
                max_tokens = self._derive_max_tokens(model_path, prompt, entry.tokenizer)
            start_time = time.time()
            response = self._backend.generate(
                entry.model,
                entry.tokenizer,
                prompt,
                max_tokens=max_tokens,
            )
            duration = time.time() - start_time
            token_count = len(self._backend.encode_tokens(entry.tokenizer, response))
            tokens_per_second = token_count / duration if duration > 0.0 else 0.0
            stop_reason = "length" if token_count >= max_tokens else "eos"

            security_summary = None
            if security_scan:
                security_summary = self._perform_security_scan(prompt, response, model)

            return InferenceResult(
                prompt=prompt,
                response=response,
                token_count=token_count,
                tokens_per_second=tokens_per_second,
                time_to_first_token=None,
                total_duration=duration,
                stop_reason=stop_reason,
                model=str(model_path),
                adapter=entry.adapter_path,
                security=security_summary,
            )
        finally:
            self.lock.release()

    def suite(
        self,
        model: str,
        suite_file: str,
        adapter: str | None = None,
        security_scan: bool = False,
    ) -> InferenceSuiteResult:
        """Execute batched inference over a suite of prompts."""
        suite_path = Path(suite_file).expanduser().resolve()
        if not suite_path.exists():
            raise ValueError(f"Suite file does not exist: {suite_path}")

        prompts = self._load_suite_prompts(suite_path)
        cases: list[InferenceCaseResult] = []
        start_time = time.time()

        for idx, entry in enumerate(prompts):
            name = entry.get("name") or f"case_{idx + 1}"
            prompt = entry.get("prompt", "")
            expected = entry.get("expected")
            try:
                t0 = time.time()
                result = self.run(
                    model=model,
                    prompt=prompt,
                    adapter=adapter,
                    security_scan=security_scan,
                )
                duration = time.time() - t0
                response = result.response
                passed = None
                if expected is not None:
                    if isinstance(expected, list):
                        passed = any(e in response for e in expected)
                    else:
                        passed = str(expected) in response
                cases.append(
                    InferenceCaseResult(
                        name=name,
                        prompt=prompt,
                        response=response,
                        token_count=result.token_count,
                        duration=duration,
                        passed=passed,
                        expected=expected,
                    )
                )
            except Exception as exc:
                cases.append(
                    InferenceCaseResult(
                        name=name,
                        prompt=prompt,
                        response="",
                        token_count=0,
                        duration=0.0,
                        passed=False if expected is not None else None,
                        expected=expected,
                        error=str(exc),
                    )
                )

        total_duration = time.time() - start_time
        passed = sum(1 for c in cases if c.passed is True)
        failed = sum(1 for c in cases if c.passed is False)
        summary = {}
        if cases:
            scored = [c for c in cases if c.passed is not None]
            if scored:
                summary["pass_rate"] = passed / max(len(scored), 1)

        return InferenceSuiteResult(
            model=model,
            adapter=adapter,
            suite=str(suite_path),
            cases=cases,
            total_cases=len(cases),
            passed=passed,
            failed=failed,
            total_duration=total_duration,
            summary=summary,
        )

    def _load_suite_prompts(self, suite_path: Path) -> list[dict[str, Any]]:
        """Load prompts from a suite file (.txt, .json, .jsonl)."""
        content = suite_path.read_text(encoding="utf-8").strip()
        if not content:
            return []

        # JSON array
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, list):
            prompts: list[dict[str, Any]] = []
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    prompts.append(item)
                else:
                    prompts.append({"name": f"case_{idx + 1}", "prompt": str(item)})
            return prompts

        # JSONL or plain text
        prompts = []
        for idx, line in enumerate(content.splitlines()):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        prompts.append(payload)
                        continue
                except json.JSONDecodeError:
                    pass
            prompts.append({"name": f"case_{idx + 1}", "prompt": line})
        return prompts

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
    "InferenceSuiteResult",
    "get_inference_engine",
    "load_model_and_tokenizer",
]
