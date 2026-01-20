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

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from modelcypher.core.domain.entropy.hidden_state_extractor import (
    HiddenStateExtractor,
)
from modelcypher.core.use_cases.entropy_learning_bridge import (
    EntropyLearningBridge,
    BridgeFeedback,
)
from modelcypher.core.use_cases.entropy_monitor import (
    EntropyMonitor,
    EntropyMonitorConfig,
    EntropySignal,
    UncertaintyAction,
    UncertaintyMode,
    create_entropy_monitor,
)
from modelcypher.ports.inference import HiddenStateEngine
from modelcypher.utils.locks import FileLock, FileLockError
from modelcypher.utils.paths import get_modelcypher_home

logger = logging.getLogger(__name__)


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


@dataclass
class SuiteInferResult:
    """Result of inference suite execution."""

    model_id: str
    suite_config: str
    test_results: list[dict[str, Any]]
    total_tests: int
    passed: int
    failed: int
    total_duration: float
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityScanSummary:
    """Summary of security scan results.

    Raw measurements: max_anomaly_score, anomaly_count, avg_delta.
    """

    anomaly_count: int
    max_anomaly_score: float
    avg_delta: float
    disagreement_rate: float


@dataclass
class InferenceResult:
    """Result of a single inference run with optional adapter and security scan."""

    prompt: str
    response: str
    token_count: int
    tokens_per_second: float
    time_to_first_token: float | None
    total_duration: float
    stop_reason: str
    model: str
    adapter: str | None
    security: SecurityScanSummary | None


@dataclass
class EntropySignalSummary:
    """Summary of entropy signals from generation.

    Raw measurements from entropy monitoring during generation,
    including bridge stats from the consciousness loop.
    """

    mean_entropy: float
    max_entropy: float
    mean_eigenscore: float
    max_eigenscore: float
    uncertainty_events: int
    abstention_triggered: bool
    signals: list[dict[str, Any]] = field(default_factory=list)
    bridge_stats: dict[str, int] = field(default_factory=dict)
    sparsity_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EntropyAwareInferenceResult:
    """Result of entropy-aware inference with uncertainty tracking.

    Extends InferenceResult with entropy trajectory and uncertainty signals.
    """

    prompt: str
    response: str
    token_count: int
    tokens_per_second: float
    time_to_first_token: float | None
    total_duration: float
    stop_reason: str
    model: str
    adapter: str | None
    uncertainty_mode: str
    entropy_summary: EntropySignalSummary
    agent_lora_loaded: bool = False  # True if agent's LoRA was auto-loaded


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
    """Result of inference suite execution with adapter support."""

    model: str
    adapter: str | None
    suite: str
    cases: list[InferenceCaseResult]
    total_cases: int
    passed: int
    failed: int
    total_duration: float
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ModelCacheEntry:
    model: Any
    tokenizer: Any
    adapter_path: str | None


@dataclass(frozen=True)
class _GenerationResult:
    text: str
    token_count: int
    tokens_per_second: float
    time_to_first_token: float | None
    total_duration: float
    stop_reason: str


class _LayerWrapper:
    def __init__(
        self,
        layer: Any,
        layer_index: int,
        capture: Callable[[int, Any], None],
    ) -> None:
        self._layer = layer
        self._layer_index = layer_index
        self._capture = capture

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._layer(*args, **kwargs)
        self._capture(self._layer_index, output)
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


class _LayerCapture:
    def __init__(
        self,
        layers: list[Any],
        capture: Callable[[int, Any], None],
        target_layers: set[int] | None = None,
    ) -> None:
        self._layers = layers
        self._capture = capture
        self._target_layers = target_layers
        self._original: list[Any] | None = None

    def __enter__(self) -> None:
        self._original = list(self._layers)
        wrapped: list[Any] = []
        for idx, layer in enumerate(self._layers):
            if self._target_layers is not None and idx not in self._target_layers:
                wrapped.append(layer)
            else:
                wrapped.append(_LayerWrapper(layer, idx, self._capture))
        self._layers[:] = wrapped

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._original is not None:
            self._layers[:] = self._original


class LocalInferenceEngine(HiddenStateEngine):
    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path = base_path or get_modelcypher_home()
        self.lock = FileLock(self.base_path / "training.lock")
        self._model_cache: dict[tuple[str, str | None], _ModelCacheEntry] = {}
        self._model_context_cache: dict[str, int] = {}
        self._mx = None
        self._mlx_load = None
        self._mlx_stream_generate = None
        self._mlx_make_sampler = None

    def _validate_model_assets(self, model_path: Path) -> bool:
        config_path = model_path / "config.json"
        if config_path.exists():
            return True
        raise ValueError(f"config.json not found in model directory: {model_path}")

    def _ensure_mlx(self) -> None:
        if self._mx is not None:
            return
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise RuntimeError("mlx is required for local inference") from exc
        try:
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError("mlx-lm is required for local inference") from exc

        self._mx = mx
        self._mlx_load = load
        self._mlx_stream_generate = stream_generate
        self._mlx_make_sampler = make_sampler

    def _load_model(self, model_path: Path, adapter: str | None) -> _ModelCacheEntry:
        self._ensure_mlx()
        adapter_path = Path(adapter).expanduser().resolve() if adapter else None
        cache_key = (str(model_path), str(adapter_path) if adapter_path else None)
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            return cached

        model, tokenizer = self._mlx_load(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None,
        )
        entry = _ModelCacheEntry(
            model=model,
            tokenizer=tokenizer,
            adapter_path=str(adapter_path) if adapter_path else None,
        )
        self._model_cache[cache_key] = entry
        return entry

    def _build_sampler(self) -> Callable[[Any], Any]:
        self._ensure_mlx()
        return self._mlx_make_sampler(temp=0.0, top_p=1.0)

    def _derive_max_tokens(
        self,
        model_path: Path,
        prompt: str,
        tokenizer: Any,
    ) -> int:
        context_limit = self._resolve_context_limit(model_path, tokenizer)
        if context_limit is None:
            return 0
        token_ids = self._encode_prompt(tokenizer, prompt)
        available = context_limit - len(token_ids)
        return max(0, available)

    def _resolve_context_limit(self, model_path: Path, tokenizer: Any) -> int | None:
        cache_key = str(model_path)
        cached = self._model_context_cache.get(cache_key)
        if cached is not None:
            return cached

        candidates: list[int] = []
        for attr in (
            "model_max_length",
            "max_length",
            "max_seq_len",
            "max_sequence_length",
            "n_ctx",
            "context_length",
            "max_context_length",
        ):
            value = getattr(tokenizer, attr, None)
            if isinstance(value, (int, float)):
                int_value = int(value)
                if 0 < int_value < 10**7:
                    candidates.append(int_value)

        config_value = self._context_from_config(model_path)
        if config_value is not None:
            candidates.append(config_value)

        if not candidates:
            return None

        resolved = min(candidates)
        self._model_context_cache[cache_key] = resolved
        return resolved

    @staticmethod
    def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
        bos_token = getattr(tokenizer, "bos_token", None)
        add_special_tokens = bos_token is None or not prompt.startswith(bos_token or "")
        try:
            return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        except TypeError:
            return tokenizer.encode(prompt)

    @staticmethod
    def _context_from_config(model_path: Path) -> int | None:
        config_path = model_path / "config.json"
        if not config_path.exists():
            return None
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        for key in (
            "max_position_embeddings",
            "max_sequence_length",
            "max_seq_len",
            "max_seq_length",
            "context_length",
            "max_context_length",
            "n_ctx",
            "model_max_length",
            "seq_length",
        ):
            value = config.get(key)
            if isinstance(value, (int, float)):
                int_value = int(value)
                if int_value > 0:
                    return int_value
        return None

    def _generate_text_mlx(
        self,
        model_path: Path,
        prompt: str,
        adapter: str | None,
    ) -> _GenerationResult:
        entry = self._load_model(model_path, adapter)
        resolved_max_tokens = self._derive_max_tokens(model_path, prompt, entry.tokenizer)
        if resolved_max_tokens <= 0:
            return _GenerationResult(
                text="",
                token_count=0,
                tokens_per_second=0.0,
                time_to_first_token=None,
                total_duration=0.0,
                stop_reason="context",
            )
        sampler = self._build_sampler()
        start = time.time()
        first_token_time: float | None = None
        text = ""
        last_response = None
        generated_tokens: list[int] = []
        prefix: list[int] = []
        stop_reason_override: str | None = None

        for response in self._mlx_stream_generate(
            entry.model,
            entry.tokenizer,
            prompt,
            max_tokens=resolved_max_tokens,
            sampler=sampler,
        ):
            if first_token_time is None and response.generation_tokens > 0:
                first_token_time = time.time() - start
            text += response.text
            last_response = response
            generated_tokens.append(response.token)

            if len(generated_tokens) == 1:
                prefix.append(0)
            else:
                j = prefix[-1]
                while j > 0 and response.token != generated_tokens[j]:
                    j = prefix[j - 1]
                if response.token == generated_tokens[j]:
                    j += 1
                prefix.append(j)
                period = len(generated_tokens) - prefix[-1]
                if period > 0 and len(generated_tokens) % period == 0:
                    if len(generated_tokens) >= 2 * period:
                        stop_reason_override = "cycle"
                        break

        # Min duration (1μs) prevents div-by-zero, below timer resolution.
        duration = max(time.time() - start, 1e-6)
        if last_response is None:
            token_count = 0
            tokens_per_second = 0.0
            stop_reason = "stop"
        else:
            token_count = len(generated_tokens)
            tokens_per_second = (
                float(last_response.generation_tps)
                if last_response.generation_tps
                else float(token_count) / duration
            )
            stop_reason = stop_reason_override or last_response.finish_reason or "stop"

        return _GenerationResult(
            text=text,
            token_count=token_count,
            tokens_per_second=tokens_per_second,
            time_to_first_token=first_token_time,
            total_duration=duration,
            stop_reason=stop_reason,
        )

    def _generate(
        self,
        model_path: Path,
        prompt: str,
        adapter: str | None,
    ) -> _GenerationResult:
        self._validate_model_assets(model_path)
        return self._generate_text_mlx(
            model_path=model_path,
            prompt=prompt,
            adapter=adapter,
        )

    def infer(
        self,
        model: str,
        prompt: str,
    ) -> dict:
        model_path = Path(model).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        try:
            result = self._generate(
                model_path=model_path,
                prompt=prompt,
                adapter=None,
            )
            return {
                "modelId": str(model_path),
                "prompt": prompt,
                "response": result.text,
                "tokenCount": result.token_count,
                "tokensPerSecond": result.tokens_per_second,
                "timeToFirstToken": result.time_to_first_token,
                "totalDuration": result.total_duration,
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
        model_path = Path(model).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not self._validate_model_assets(model_path):
            raise ValueError("Hidden-state capture requires a full model directory.")

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        try:
            entry = self._load_model(model_path, adapter)
            mx = self._mx
            if mx is None:
                raise RuntimeError("MLX backend not available for hidden-state capture.")

            tokenizer = entry.tokenizer
            add_special_tokens = getattr(
                tokenizer, "bos_token", None
            ) is None or not prompt.startswith(getattr(tokenizer, "bos_token", "") or "")
            token_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
            if not token_ids:
                return {}

            tokens = mx.array(token_ids)
            token_index = len(token_ids) - 1

            base_model = getattr(entry.model, "model", entry.model)
            layers = getattr(base_model, "layers", None)
            if layers is None:
                raise RuntimeError("Model does not expose transformer layers for capture.")

            if target_layers is None:
                target_layers = set(range(len(layers)))

            extractor = HiddenStateExtractor(
                target_layers=target_layers,
                expected_hidden_dim=None,
            )
            extractor.start_session()

            def _capture(layer_index: int, hidden_state: Any) -> None:
                extractor.capture(hidden_state, layer=layer_index, token_index=token_index)

            with _LayerCapture(layers, _capture, target_layers=target_layers):
                _ = base_model(tokens[None, :])
                self._mx.eval(_)

            extractor.end_session()
            states = extractor.extracted_states()
            if not states:
                return {}
            self._mx.eval(*states.values())
            return {
                int(layer): state.astype(mx.float32).reshape(-1).tolist()
                for layer, state in states.items()
            }
        finally:
            self.lock.release()

    def run_batch(
        self,
        model: str,
        prompts_file: str,
    ) -> BatchInferResult:
        """Execute batched inference from a prompts file.

        Args:
            model: Model identifier or path
            prompts_file: Path to file containing prompts (one per line or JSONL)

        Returns:
            BatchInferResult with all inference results

        Raises:
            ValueError: If prompts file doesn't exist or is invalid
        """
        prompts_path = Path(prompts_file).expanduser().resolve()
        if not prompts_path.exists():
            raise ValueError(f"Prompts file does not exist: {prompts_path}")

        # Read prompts from file
        prompts = self._read_prompts(prompts_path)
        if not prompts:
            raise ValueError(f"No prompts found in file: {prompts_path}")

        logger.info("Running batch inference with %d prompts", len(prompts))

        results = []
        successful = 0
        failed = 0
        total_tokens = 0
        start_time = time.time()

        for i, prompt in enumerate(prompts):
            try:
                result = self.infer(model, prompt)
                results.append(
                    {
                        "index": i,
                        "prompt": prompt[:100],  # Truncate for response
                        "response": result["response"],
                        "tokenCount": result["tokenCount"],
                        "status": "success",
                    }
                )
                successful += 1
                total_tokens += result["tokenCount"]
            except Exception as exc:
                results.append(
                    {
                        "index": i,
                        "prompt": prompt[:100],
                        "error": str(exc),
                        "status": "failed",
                    }
                )
                failed += 1
                logger.warning("Batch inference failed for prompt %d: %s", i, exc)

        total_duration = time.time() - start_time
        avg_tps = total_tokens / max(total_duration, 1e-6)

        return BatchInferResult(
            model_id=model,
            prompts_file=str(prompts_path),
            results=results,
            total_prompts=len(prompts),
            successful=successful,
            failed=failed,
            total_tokens=total_tokens,
            total_duration=total_duration,
            average_tokens_per_second=avg_tps,
        )

    def run_suite(
        self,
        model: str,
        suite_config: str,
    ) -> SuiteInferResult:
        """Execute inference suite from a configuration file.

        Args:
            model: Model identifier or path
            suite_config: Path to suite configuration (JSON)

        Returns:
            SuiteInferResult with test results and summary

        Raises:
            ValueError: If suite config doesn't exist or is invalid
        """
        config_path = Path(suite_config).expanduser().resolve()
        if not config_path.exists():
            raise ValueError(f"Suite config does not exist: {config_path}")

        # Load suite configuration
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid suite config JSON: {exc}") from exc

        tests = config.get("tests", [])
        if not tests:
            raise ValueError("Suite config contains no tests")

        logger.info("Running inference suite with %d tests", len(tests))

        test_results = []
        passed = 0
        failed = 0
        start_time = time.time()

        for i, test in enumerate(tests):
            test_name = test.get("name", f"test_{i}")
            prompt = test.get("prompt", "")
            expected = test.get("expected", None)

            try:
                result = self.infer(model, prompt)
                response = result["response"]

                # Check if expected pattern is in response
                test_passed = True
                if expected:
                    if isinstance(expected, str):
                        test_passed = expected.lower() in response.lower()
                    elif isinstance(expected, list):
                        test_passed = any(exp.lower() in response.lower() for exp in expected)

                test_results.append(
                    {
                        "name": test_name,
                        "prompt": prompt[:100],
                        "response": response[:200],
                        "expected": expected,
                        "passed": test_passed,
                        "tokenCount": result["tokenCount"],
                        "duration": result["totalDuration"],
                    }
                )

                if test_passed:
                    passed += 1
                else:
                    failed += 1

            except Exception as exc:
                test_results.append(
                    {
                        "name": test_name,
                        "prompt": prompt[:100],
                        "error": str(exc),
                        "passed": False,
                    }
                )
                failed += 1
                logger.warning("Suite test %s failed: %s", test_name, exc)

        total_duration = time.time() - start_time

        return SuiteInferResult(
            model_id=model,
            suite_config=str(config_path),
            test_results=test_results,
            total_tests=len(tests),
            passed=passed,
            failed=failed,
            total_duration=total_duration,
            summary={
                "pass_rate": passed / max(len(tests), 1),
                "average_duration": total_duration / max(len(tests), 1),
                "suite_name": config.get("name", "unnamed"),
            },
        )

    def _read_prompts(self, path: Path) -> list[str]:
        """Read prompts from a file.

        Supports:
        - Plain text (one prompt per line)
        - JSONL with "prompt" field
        - JSON array of prompts
        """
        content = path.read_text(encoding="utf-8")

        # Try to parse as JSON array first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                prompts = []
                for item in data:
                    if isinstance(item, str):
                        prompts.append(item)
                    elif isinstance(item, dict):
                        if "prompt" in item:
                            prompts.append(item["prompt"])
                        elif "text" in item:
                            prompts.append(item["text"])
                return prompts
        except json.JSONDecodeError:
            pass

        # Fall back to line-by-line parsing
        lines = content.strip().split("\n")

        prompts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse as JSON
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    if "prompt" in data:
                        prompts.append(data["prompt"])
                    elif "text" in data:
                        prompts.append(data["text"])
                    else:
                        prompts.append(line)
                except json.JSONDecodeError:
                    prompts.append(line)
            else:
                prompts.append(line)

        return prompts

    def _load_adapter(self, adapter_path: str) -> dict[str, Any] | None:
        """Load adapter configuration from path.

        Args:
            adapter_path: Path to adapter directory

        Returns:
            Adapter configuration dict or None if not found
        """
        adapter_dir = Path(adapter_path).expanduser().resolve()
        if not adapter_dir.exists():
            raise ValueError(f"Adapter path does not exist: {adapter_dir}")

        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse adapter config: %s", exc)
                return None
        return None

    def _perform_security_scan(
        self,
        prompt: str,
        response: str,
        model: str,
    ) -> SecurityScanSummary:
        """Perform dual-path security analysis on inference.

        Args:
            prompt: The input prompt
            response: The model response
            model: Model path

        Returns:
            SecurityScanSummary with analysis results
        """
        # Simplified security scan implementation
        # In production, this would use the geometry safety service
        logger.warning(
            "Local security scan does not compute geometry-derived measurements. "
            "Returning zeroed metrics."
        )
        anomaly_score = 0.0
        anomaly_count = 0
        avg_delta = 0.0

        return SecurityScanSummary(
            anomaly_count=anomaly_count,
            max_anomaly_score=anomaly_score,
            avg_delta=avg_delta,
            disagreement_rate=0.0,
        )

    def run(
        self,
        model: str,
        prompt: str,
        adapter: str | None = None,
        security_scan: bool = False,
    ) -> InferenceResult:
        """Execute inference with optional adapter and security scanning.

        Args:
            model: Model identifier or path
            prompt: Input prompt
            adapter: Optional path to adapter directory
            security_scan: Whether to perform dual-path security analysis

        Returns:
            InferenceResult with metrics and optional security summary

        Raises:
            ValueError: If model or adapter path is invalid
            RuntimeError: If training is running
        """
        model_path = Path(model).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Load adapter if specified
        if adapter:
            self._load_adapter(adapter)
            logger.info("Loaded adapter from %s", adapter)

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        try:
            result = self._generate(
                model_path=model_path,
                prompt=prompt,
                adapter=adapter,
            )

            security_summary = None
            if security_scan:
                security_summary = self._perform_security_scan(prompt, result.text, model)

            return InferenceResult(
                prompt=prompt,
                response=result.text,
                token_count=result.token_count,
                tokens_per_second=result.tokens_per_second,
                time_to_first_token=result.time_to_first_token,
                total_duration=result.total_duration,
                stop_reason=result.stop_reason,
                model=str(model_path),
                adapter=adapter,
                security=security_summary,
            )
        finally:
            self.lock.release()

    def run_with_entropy(
        self,
        model: str,
        prompt: str,
        adapter: str | None = None,
        uncertainty_mode: str = "human_in_loop",
        entropy_threshold: float = 0.7,
        eigenscore_threshold: float = 0.6,
        max_tokens: int | None = None,
        agent_id: str | None = None,
    ) -> EntropyAwareInferenceResult:
        """Execute entropy-aware inference with real-time uncertainty monitoring.

        Generates text while tracking entropy and EigenScore at each token.
        Can stop generation early if uncertainty exceeds thresholds based on
        the configured uncertainty mode.

        When agent_id is provided, sparsity events and hidden states are saved
        to the agent's directory for later consolidation into LoRA memory.

        Args:
            model: Model identifier or path
            prompt: Input prompt
            adapter: Optional path to adapter directory
            uncertainty_mode: One of "butler", "autonomous", "human_in_loop"
            entropy_threshold: Normalized entropy threshold for uncertainty
            eigenscore_threshold: EigenScore threshold for sparse manifold detection
            max_tokens: Maximum tokens to generate (None = auto from context)
            agent_id: Optional agent identifier for LoRA memory. When provided,
                hidden states at sparse regions are retained and saved for
                later consolidation.

        Returns:
            EntropyAwareInferenceResult with entropy trajectory and uncertainty signals

        Raises:
            ValueError: If model path is invalid
            RuntimeError: If training is running
        """
        model_path = Path(model).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Auto-load agent's LoRA if available and no explicit adapter specified
        effective_adapter = adapter
        agent_lora_loaded = False
        if agent_id and not adapter:
            agent_lora_path = self._get_agent_lora_path(agent_id)
            if agent_lora_path:
                effective_adapter = str(agent_lora_path)
                agent_lora_loaded = True
                logger.info(
                    "Auto-loading agent '%s' LoRA from %s",
                    agent_id, agent_lora_path
                )

        if effective_adapter:
            self._load_adapter(effective_adapter)
            logger.info("Loaded adapter from %s", effective_adapter)

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        try:
            # Initialize entropy monitor
            mode = UncertaintyMode(uncertainty_mode)
            config = EntropyMonitorConfig(
                uncertainty_mode=mode,
                entropy_threshold=entropy_threshold,
                eigenscore_threshold=eigenscore_threshold,
            )
            monitor = EntropyMonitor(config=config)
            monitor.reset()

            # Load model (with auto-loaded agent LoRA if applicable)
            entry = self._load_model(model_path, effective_adapter)
            mx = self._mx

            # Determine max tokens
            resolved_max = max_tokens
            if resolved_max is None:
                resolved_max = self._derive_max_tokens(model_path, prompt, entry.tokenizer)
            if resolved_max <= 0:
                return EntropyAwareInferenceResult(
                    prompt=prompt,
                    response="",
                    token_count=0,
                    tokens_per_second=0.0,
                    time_to_first_token=None,
                    total_duration=0.0,
                    stop_reason="context",
                    model=str(model_path),
                    adapter=effective_adapter,
                    uncertainty_mode=uncertainty_mode,
                    entropy_summary=EntropySignalSummary(
                        mean_entropy=0.0,
                        max_entropy=0.0,
                        mean_eigenscore=0.0,
                        max_eigenscore=0.0,
                        uncertainty_events=0,
                        abstention_triggered=False,
                    ),
                    agent_lora_loaded=agent_lora_loaded,
                )

            # Tokenize prompt
            token_ids = self._encode_prompt(entry.tokenizer, prompt)
            tokens = mx.array([token_ids])

            # Get vocab size for entropy normalization
            vocab_size = getattr(entry.tokenizer, "vocab_size", 32000)
            monitor._config.vocab_size = vocab_size

            # Generation loop with entropy monitoring
            start_time = time.time()
            first_token_time: float | None = None
            generated_tokens: list[int] = []
            entropy_signals: list[EntropySignal] = []
            stop_reason = "length"
            abstention_triggered = False

            # Get base model for forward passes
            base_model = getattr(entry.model, "model", entry.model)

            # Set up hidden state capture for EigenScore
            # Target only the last layer - most informative for manifold sparsity
            layers = getattr(base_model, "layers", None)
            captured_hidden_state: Any = None

            if layers is not None:
                last_layer_idx = len(layers) - 1
                target_layers = {last_layer_idx}

                def _capture_hidden(layer_index: int, hidden_state: Any) -> None:
                    nonlocal captured_hidden_state
                    # Only capture the target layer, extract last token
                    if layer_index == last_layer_idx:
                        if hidden_state.ndim > 2:
                            captured_hidden_state = hidden_state[0, -1, :]
                        elif hidden_state.ndim == 2:
                            captured_hidden_state = hidden_state[-1, :]
                        else:
                            captured_hidden_state = hidden_state

                layer_capture = _LayerCapture(layers, _capture_hidden, target_layers)
                layer_capture.__enter__()
            else:
                layer_capture = None
                logger.warning("Model does not expose layers - EigenScore will be unavailable")

            # Initialize entropy-learning bridge for consciousness loop
            # Get hidden_dim from model config
            hidden_dim = getattr(
                getattr(base_model, "config", None),
                "hidden_size",
                getattr(base_model, "hidden_size", 576),  # Default for SmolLM
            )
            # Retain hidden states when agent_id is provided (for LoRA memory)
            retain_states = agent_id is not None
            bridge = EntropyLearningBridge(
                hidden_dim=hidden_dim,
                retain_hidden_states=retain_states,
            )
            bridge_feedbacks: list[BridgeFeedback] = []

            try:
                for i in range(resolved_max):
                    # Reset captured state for this forward pass
                    captured_hidden_state = None

                    # Forward pass to get logits (hooks capture hidden state)
                    logits = entry.model(tokens)
                    mx.eval(logits)

                    # Get logits for last position
                    last_logits = logits[0, -1, :]

                    # Compute entropy signal with captured hidden state for EigenScore
                    signal = monitor.compute_signal(
                        token_index=i,
                        token_id=0,  # Will update after sampling
                        token_text="",
                        logits=last_logits,
                        hidden_states=captured_hidden_state,  # Now wired to EigenScore!
                    )

                    # Process signal through entropy-learning bridge (consciousness loop)
                    # This routes to SurpriseDetector and queues sparsity events for consolidation
                    feedback = bridge.process_signal(
                        signal=signal,
                        logits=last_logits,
                        actual_token_id=int(mx.argmax(last_logits).item()),
                        hidden_state=captured_hidden_state,
                    )
                    bridge_feedbacks.append(feedback)

                    # Log hallucination risk warnings
                    if feedback.is_hallucination_risk:
                        logger.warning(
                            "Hallucination risk at token %d: eigenscore=%.3f, refusal=%.3f",
                            i, signal.eigenscore, signal.refusal_projection,
                        )

                    # Check if we should stop based on uncertainty
                    if signal.should_stop:
                        abstention_triggered = True
                        stop_reason = f"uncertainty:{signal.action.value}"
                        break

                    # Sample next token (greedy for now)
                    next_token = int(mx.argmax(last_logits).item())

                    # Update signal with actual token
                    token_text = entry.tokenizer.decode([next_token])
                    signal.token_id = next_token
                    signal.token_text = token_text
                    entropy_signals.append(signal)

                    if first_token_time is None:
                        first_token_time = time.time() - start_time

                    generated_tokens.append(next_token)

                    # Check for EOS
                    eos_token_id = getattr(entry.tokenizer, "eos_token_id", None)
                    if next_token == eos_token_id:
                        stop_reason = "eos"
                        break

                    # Append token to sequence
                    tokens = mx.concatenate([tokens, mx.array([[next_token]])], axis=1)
            finally:
                # Clean up layer capture hooks
                if layer_capture is not None:
                    layer_capture.__exit__(None, None, None)

            # Decode response
            response = entry.tokenizer.decode(generated_tokens)
            total_duration = max(time.time() - start_time, 1e-6)
            token_count = len(generated_tokens)
            tokens_per_second = token_count / total_duration

            # Compute entropy summary with bridge stats
            bridge_stats = bridge.get_stats()
            sparsity_events = bridge.get_sparsity_queue()

            if entropy_signals:
                entropies = [s.normalized_entropy for s in entropy_signals]
                eigenscores = [s.eigenscore for s in entropy_signals]
                refusal_projections = [s.refusal_projection for s in entropy_signals]
                uncertainty_events = sum(1 for s in entropy_signals if s.is_uncertain)

                # Log bridge activity
                if bridge_stats.warn_events > 0:
                    logger.info(
                        "Bridge stats: %d WARN events, %d sparsity events queued",
                        bridge_stats.warn_events, bridge_stats.sparsity_events,
                    )

                entropy_summary = EntropySignalSummary(
                    mean_entropy=sum(entropies) / len(entropies),
                    max_entropy=max(entropies),
                    mean_eigenscore=sum(eigenscores) / len(eigenscores),
                    max_eigenscore=max(eigenscores),
                    uncertainty_events=uncertainty_events,
                    abstention_triggered=abstention_triggered,
                    signals=[
                        {
                            "index": s.token_index,
                            "token": s.token_text,
                            "entropy": s.normalized_entropy,
                            "eigenscore": s.eigenscore,
                            "refusal_projection": s.refusal_projection,
                            "action": s.action.value,
                        }
                        for s in entropy_signals
                    ],
                    bridge_stats={
                        "warn_events": bridge_stats.warn_events,
                        "sparsity_events": bridge_stats.sparsity_events,
                        "confidence_injections": bridge_stats.confidence_injections,
                    },
                    sparsity_events=[
                        {
                            "token_index": e.token_index,
                            "eigenscore": e.eigenscore,
                            "refusal_projection": e.refusal_projection,
                            "action": e.action.value,
                            "hidden_state_hash": e.hidden_state_hash,
                            "layer_index": e.layer_index,
                        }
                        for e in sparsity_events
                    ],
                )
            else:
                entropy_summary = EntropySignalSummary(
                    mean_entropy=0.0,
                    max_entropy=0.0,
                    mean_eigenscore=0.0,
                    max_eigenscore=0.0,
                    uncertainty_events=0,
                    abstention_triggered=abstention_triggered,
                )

            # Save session for LoRA memory if agent_id provided
            if agent_id and sparsity_events:
                self._save_agent_session(
                    agent_id=agent_id,
                    model_path=str(model_path),
                    sparsity_events=sparsity_events,
                    hidden_states=bridge.get_hidden_states(),
                )

            return EntropyAwareInferenceResult(
                prompt=prompt,
                response=response,
                token_count=token_count,
                tokens_per_second=tokens_per_second,
                time_to_first_token=first_token_time,
                total_duration=total_duration,
                stop_reason=stop_reason,
                model=str(model_path),
                adapter=adapter,
                uncertainty_mode=uncertainty_mode,
                entropy_summary=entropy_summary,
            )

        finally:
            self.lock.release()

    def _get_agent_lora_path(self, agent_id: str) -> Path | None:
        """Get path to agent's trained LoRA weights if they exist.

        Args:
            agent_id: Agent identifier.

        Returns:
            Path to LoRA adapter directory, or None if not trained yet.
        """
        # Check for trained LoRA weights
        lora_dir = self.base_path / "lora_memory" / agent_id
        weights_path = lora_dir / "lora_weights.safetensors"

        if weights_path.exists():
            # LoRA weights exist - check if there's an adapter config
            config_path = lora_dir / "adapter_config.json"
            if config_path.exists():
                return lora_dir

            # Create minimal adapter config for mlx_lm compatibility
            metadata_path = lora_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    import json
                    metadata = json.loads(metadata_path.read_text())
                    adapter_config = {
                        "r": metadata.get("rank", 8),
                        "lora_alpha": metadata.get("alpha", 16.0),
                        "target_modules": metadata.get("target_modules", ["q_proj", "v_proj"]),
                        "lora_dropout": 0.0,
                        "bias": "none",
                    }
                    config_path.write_text(json.dumps(adapter_config, indent=2))
                    logger.info("Created adapter_config.json for agent '%s'", agent_id)
                    return lora_dir
                except Exception as exc:
                    logger.warning("Failed to create adapter config: %s", exc)
                    return None

        return None

    def _save_agent_session(
        self,
        agent_id: str,
        model_path: str,
        sparsity_events: list[Any],
        hidden_states: dict[str, Any],
    ) -> Path | None:
        """Save sparsity events and hidden states for LoRA memory.

        Args:
            agent_id: Agent identifier.
            model_path: Path to model.
            sparsity_events: List of SparsityEvent objects.
            hidden_states: Dict mapping keys to hidden state tensors.

        Returns:
            Path to session directory, or None if nothing to save.
        """
        if not sparsity_events:
            return None

        # Create agent session directory
        session_dir = self.base_path / "lora_memory" / agent_id / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Generate session ID from timestamp
        import time as time_mod
        session_id = f"session_{int(time_mod.time() * 1000)}"
        session_path = session_dir / session_id
        session_path.mkdir(exist_ok=True)

        # Save metadata
        metadata = {
            "agent_id": agent_id,
            "model_path": model_path,
            "session_id": session_id,
            "event_count": len(sparsity_events),
            "hidden_state_count": len(hidden_states),
            "sparsity_events": [
                {
                    "token_index": e.token_index,
                    "eigenscore": e.eigenscore,
                    "refusal_projection": e.refusal_projection,
                    "action": e.action.value if hasattr(e.action, "value") else str(e.action),
                    "hidden_state_hash": e.hidden_state_hash,
                    "hidden_state_key": e.hidden_state_key,
                    "layer_index": e.layer_index,
                    "manifold_coordinates": e.manifold_coordinates,
                }
                for e in sparsity_events
            ],
        }

        metadata_path = session_path / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Save hidden states if any
        if hidden_states:
            states_path = session_path / "hidden_states.safetensors"
            self._mx.save_safetensors(str(states_path), hidden_states)

        logger.info(
            "Saved agent session: %s (%d events, %d states)",
            session_path,
            len(sparsity_events),
            len(hidden_states),
        )

        return session_path

    def suite(
        self,
        model: str,
        suite_file: str,
        adapter: str | None = None,
        security_scan: bool = False,
    ) -> InferenceSuiteResult:
        """Execute batched inference over a suite of prompts.

        Loads prompts from file (.txt, .json, .jsonl) and executes inference
        keeping the model loaded for efficiency.

        Args:
            model: Model identifier or path
            suite_file: Path to suite file containing prompts
            adapter: Optional path to adapter directory
            security_scan: Whether to perform security analysis

        Returns:
            InferenceSuiteResult with all case results

        Raises:
            ValueError: If suite file doesn't exist or is invalid
        """
        suite_path = Path(suite_file).expanduser().resolve()
        if not suite_path.exists():
            raise ValueError(f"Suite file does not exist: {suite_path}")

        # Determine file type and load prompts/tests
        suffix = suite_path.suffix.lower()

        if suffix == ".json":
            # Try to load as suite config with tests
            try:
                config = json.loads(suite_path.read_text(encoding="utf-8"))
                if isinstance(config, dict) and "tests" in config:
                    return self._run_suite_config(
                        model, suite_path, config, adapter, security_scan
                    )
                elif isinstance(config, list):
                    # JSON array of prompts
                    prompts = []
                    for item in config:
                        if isinstance(item, str):
                            prompts.append({"prompt": item})
                        elif isinstance(item, dict):
                            prompts.append(item)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in suite file: {exc}") from exc
        else:
            # Load as plain text or JSONL
            prompts = []
            for p in self._read_prompts(suite_path):
                prompts.append({"prompt": p})

        if not prompts:
            raise ValueError(f"No prompts found in suite file: {suite_path}")

        logger.info("Running inference suite with %d prompts", len(prompts))

        cases: list[InferenceCaseResult] = []
        passed = 0
        failed = 0
        start_time = time.time()

        for i, item in enumerate(prompts):
            prompt = item.get("prompt", "") if isinstance(item, dict) else str(item)
            name = item.get("name", f"case_{i}") if isinstance(item, dict) else f"case_{i}"
            expected = item.get("expected") if isinstance(item, dict) else None
            try:
                result = self.run(
                    model=model,
                    prompt=prompt,
                    adapter=adapter,
                    security_scan=security_scan,
                )

                # Check expected if provided
                test_passed = None
                if expected:
                    if isinstance(expected, str):
                        test_passed = expected.lower() in result.response.lower()
                    elif isinstance(expected, list):
                        test_passed = any(
                            exp.lower() in result.response.lower() for exp in expected
                        )
                    if test_passed:
                        passed += 1
                    else:
                        failed += 1

                cases.append(
                    InferenceCaseResult(
                        name=name,
                        prompt=prompt[:100],
                        response=result.response[:200],
                        token_count=result.token_count,
                        duration=result.total_duration,
                        passed=test_passed,
                        expected=expected,
                    )
                )

            except Exception as exc:
                cases.append(
                    InferenceCaseResult(
                        name=name,
                        prompt=prompt[:100],
                        response="",
                        token_count=0,
                        duration=0.0,
                        passed=False,
                        expected=expected,
                        error=str(exc),
                    )
                )
                failed += 1
                logger.warning("Suite case %s failed: %s", name, exc)

        total_duration = time.time() - start_time
        total_cases = len(cases)

        return InferenceSuiteResult(
            model=model,
            adapter=adapter,
            suite=str(suite_path),
            cases=cases,
            total_cases=total_cases,
            passed=passed,
            failed=failed,
            total_duration=total_duration,
            summary={
                "pass_rate": passed / max(total_cases, 1) if (passed + failed) > 0 else None,
                "average_duration": total_duration / max(total_cases, 1),
                "suite_name": suite_path.stem,
            },
        )

    def _run_suite_config(
        self,
        model: str,
        suite_path: Path,
        config: dict[str, Any],
        adapter: str | None,
        security_scan: bool,
    ) -> InferenceSuiteResult:
        """Run suite from a structured config with tests."""
        tests = config.get("tests", [])
        if not tests:
            raise ValueError("Suite config contains no tests")

        logger.info("Running inference suite with %d tests", len(tests))

        cases: list[InferenceCaseResult] = []
        passed = 0
        failed = 0
        start_time = time.time()

        for i, test in enumerate(tests):
            name = test.get("name", f"test_{i}")
            prompt = test.get("prompt", "")
            expected = test.get("expected")

            try:
                result = self.run(
                    model=model,
                    prompt=prompt,
                    adapter=adapter,
                    security_scan=security_scan,
                )

                # Check expected if provided
                test_passed = True
                if expected:
                    if isinstance(expected, str):
                        test_passed = expected.lower() in result.response.lower()
                    elif isinstance(expected, list):
                        test_passed = any(
                            exp.lower() in result.response.lower() for exp in expected
                        )

                if test_passed:
                    passed += 1
                else:
                    failed += 1

                cases.append(
                    InferenceCaseResult(
                        name=name,
                        prompt=prompt[:100],
                        response=result.response[:200],
                        token_count=result.token_count,
                        duration=result.total_duration,
                        passed=test_passed,
                        expected=expected,
                    )
                )

            except Exception as exc:
                cases.append(
                    InferenceCaseResult(
                        name=name,
                        prompt=prompt[:100],
                        response="",
                        token_count=0,
                        duration=0.0,
                        passed=False,
                        expected=expected,
                        error=str(exc),
                    )
                )
                failed += 1
                logger.warning("Suite test %s failed: %s", name, exc)

        total_duration = time.time() - start_time
        total_cases = len(cases)

        return InferenceSuiteResult(
            model=model,
            adapter=adapter,
            suite=str(suite_path),
            cases=cases,
            total_cases=total_cases,
            passed=passed,
            failed=failed,
            total_duration=total_duration,
            summary={
                "pass_rate": passed / max(total_cases, 1),
                "average_duration": total_duration / max(total_cases, 1),
                "suite_name": config.get("name", "unnamed"),
            },
        )
