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

"""CUDA/PyTorch Inference Engine implementing HiddenStateEngine.

This adapter wraps PyTorch and HuggingFace Transformers for inference
on CUDA GPUs, implementing the HiddenStateEngine protocol for hexagonal
architecture compliance.

Usage:
    from modelcypher.adapters.cuda_inference import CUDAInferenceEngine

    engine = CUDAInferenceEngine()
    result = engine.infer("/path/to/model", "Hello, world!")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.ports.inference import HiddenStateEngine
from modelcypher.utils.locks import FileLock, FileLockError
from modelcypher.utils.paths import get_modelcypher_home
from modelcypher.utils.security import trust_remote_code_enabled, warn_trust_remote_code

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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


class _HiddenStateHook:
    """PyTorch forward hook for capturing hidden states."""

    def __init__(self, layer_index: int, capture: Callable[[int, Any], None]) -> None:
        self._layer_index = layer_index
        self._capture = capture

    def __call__(self, module: Any, input: Any, output: Any) -> None:
        self._capture(self._layer_index, output)


class CUDAInferenceEngine(HiddenStateEngine):
    """PyTorch/CUDA implementation of HiddenStateEngine.

    Provides inference and hidden state capture on NVIDIA GPUs using
    HuggingFace Transformers.

    Requires: pip install torch transformers safetensors
    """

    def __init__(
        self,
        base_path: Path | None = None,
    ) -> None:
        """Initialize CUDA inference engine.

        Args:
            base_path: Base directory for locks and caches.
        """
        self.base_path = base_path or get_modelcypher_home()
        self.lock = FileLock(self.base_path / "training.lock")
        self.device = "cuda"
        self._model_cache: dict[tuple[str, str | None], _ModelCacheEntry] = {}
        self._model_context_cache: dict[str, int] = {}
        self._torch = None
        self._available = False
        self._init_backend()
        if self._torch is not None:
            device_index = int(self._torch.cuda.current_device())
            self.device = f"cuda:{device_index}"

    def _init_backend(self) -> None:
        """Initialize PyTorch backend."""
        try:
            import torch

            self._torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available. ModelCypher requires GPU acceleration. "
                    "CPU fallback is not supported. Ensure CUDA is properly installed."
                )
            self._available = True
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch not available. Install with: pip install torch"
            ) from exc

    @property
    def available(self) -> bool:
        """Check if CUDA backend is available."""
        return self._available

    def _ensure_torch(self) -> None:
        """Ensure PyTorch is available."""
        if self._torch is None:
            raise RuntimeError(
                "PyTorch not available. Install with: pip install torch transformers"
            )

    def _load_model(self, model_path: Path, adapter: str | None) -> _ModelCacheEntry:
        """Load model and tokenizer, with caching."""
        self._ensure_torch()

        adapter_path = Path(adapter).expanduser().resolve() if adapter else None
        cache_key = (str(model_path), str(adapter_path) if adapter_path else None)
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers not available. Install: pip install transformers"
            ) from exc

        logger.info("Loading model from %s with CUDA backend...", model_path)

        warn_trust_remote_code(logger)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=trust_remote_code_enabled()
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=self._torch.bfloat16,
            device_map=self.device,
            trust_remote_code=trust_remote_code_enabled(),
        )

        if adapter_path:
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(adapter_path))
                logger.info("Loaded adapter from %s", adapter_path)
            except ImportError as exc:
                raise RuntimeError(
                    "peft is required to load adapters on CUDA. Install: pip install peft"
                ) from exc

        entry = _ModelCacheEntry(
            model=model,
            tokenizer=tokenizer,
            adapter_path=str(adapter_path) if adapter_path else None,
        )
        self._model_cache[cache_key] = entry
        return entry

    def _resolve_context_limit(self, model_path: Path, tokenizer: Any) -> int | None:
        """Resolve model context length from tokenizer or config."""
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
    def _context_from_config(model_path: Path) -> int | None:
        """Read context limit from config.json."""
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

    def _derive_max_tokens(
        self,
        model_path: Path,
        prompt: str,
        tokenizer: Any,
    ) -> int:
        """Derive max tokens from model context and prompt length."""
        context_limit = self._resolve_context_limit(model_path, tokenizer)
        if context_limit is None:
            return 0
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        available = context_limit - len(token_ids)
        return max(0, available)

    def _generate(
        self,
        model_path: Path,
        prompt: str,
        adapter: str | None,
    ) -> _GenerationResult:
        """Generate text using PyTorch model."""
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

        # Encode prompt
        inputs = entry.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        start = time.time()
        first_token_time: float | None = None

        # Generate without streaming (time-to-first-token unavailable)
        with self._torch.no_grad():
            outputs = entry.model.generate(
                **inputs,
                max_new_tokens=resolved_max_tokens,
                do_sample=False,  # Greedy decoding for determinism
                pad_token_id=entry.tokenizer.pad_token_id or entry.tokenizer.eos_token_id,
                use_cache=True,
            )

        duration = max(time.time() - start, 1e-6)

        # Decode only the generated tokens
        generated_ids = outputs[0][input_length:]
        text = entry.tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_count = len(generated_ids)
        tokens_per_second = float(token_count) / duration

        # Determine stop reason
        eos_token_id = entry.tokenizer.eos_token_id
        stop_reason = "stop"
        if token_count >= resolved_max_tokens:
            stop_reason = "length"
        elif token_count > 0 and eos_token_id is not None and generated_ids[-1].item() == eos_token_id:
            stop_reason = "stop"

        return _GenerationResult(
            text=text,
            token_count=token_count,
            tokens_per_second=tokens_per_second,
            time_to_first_token=first_token_time,
            total_duration=duration,
            stop_reason=stop_reason,
        )

    def infer(
        self,
        model: str,
        prompt: str,
    ) -> dict:
        """Run inference and return structured results.

        Args:
            model: Path to model directory
            prompt: Input prompt
        Returns:
            Dictionary with inference results
        """
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
        """Return hidden states keyed by layer index.

        Uses PyTorch forward hooks to capture hidden states during inference.

        Args:
            model: Path to model directory
            prompt: Input prompt
            adapter: Optional adapter path
            target_layers: Set of layer indices to capture (None = all)

        Returns:
            Dictionary mapping layer index to hidden state vector
        """
        self._ensure_torch()

        model_path = Path(model).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        try:
            entry = self._load_model(model_path, adapter)
            captured_states: dict[int, Any] = {}

            def capture(layer_index: int, output: Any) -> None:
                # Handle different output formats (tuple vs tensor)
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output
                # Get last token's hidden state
                captured_states[layer_index] = hidden_state[:, -1, :].detach()

            # Find transformer layers
            base_model = entry.model
            layers = None

            # Try common layer attribute names
            for attr in ("model.layers", "transformer.h", "gpt_neox.layers", "layers"):
                try:
                    parts = attr.split(".")
                    obj = base_model
                    for part in parts:
                        obj = getattr(obj, part)
                    if hasattr(obj, "__len__"):
                        layers = obj
                        break
                except AttributeError:
                    continue

            if layers is None:
                raise RuntimeError(
                    "Model does not expose transformer layers for capture."
                )

            num_layers = len(layers)
            if target_layers is None:
                target_layers = set(range(num_layers))

            # Register hooks
            hooks = []
            for idx, layer in enumerate(layers):
                if idx in target_layers:
                    hook = layer.register_forward_hook(
                        _HiddenStateHook(idx, capture)
                    )
                    hooks.append(hook)

            try:
                # Run forward pass
                inputs = entry.tokenizer(prompt, return_tensors="pt").to(self.device)
                with self._torch.no_grad():
                    _ = entry.model(**inputs)
            finally:
                # Remove hooks
                for hook in hooks:
                    hook.remove()

            # Convert to list format
            return {
                int(layer): state.float().cpu().reshape(-1).tolist()
                for layer, state in captured_states.items()
            }

        finally:
            self.lock.release()


def get_inference_engine() -> CUDAInferenceEngine:
    """Get the CUDA inference engine instance."""
    return CUDAInferenceEngine()


__all__ = ["CUDAInferenceEngine", "get_inference_engine"]
