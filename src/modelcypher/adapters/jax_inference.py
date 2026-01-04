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

"""JAX/TPU Inference Engine implementing HiddenStateEngine.

This adapter wraps JAX and HuggingFace Transformers for inference
on TPU/GPU, implementing the HiddenStateEngine protocol for hexagonal
architecture compliance.

Usage:
    from modelcypher.adapters.jax_inference import JAXInferenceEngine

    engine = JAXInferenceEngine()
    result = engine.infer("/path/to/model", "Hello, world!")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.ports.inference import HiddenStateEngine
from modelcypher.utils.locks import FileLock, FileLockError
from modelcypher.utils.paths import get_modelcypher_home

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


class JAXInferenceEngine(HiddenStateEngine):
    """JAX/Flax implementation of HiddenStateEngine.

    Provides inference and hidden state capture on TPU/GPU using
    HuggingFace Transformers with Flax backend.

    Requires: pip install jax jaxlib transformers flax
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize JAX inference engine.

        Args:
            base_path: Base directory for locks and caches.
        """
        self.base_path = base_path or get_modelcypher_home()
        self.lock = FileLock(self.base_path / "training.lock")
        self._model_cache: dict[tuple[str, str | None], _ModelCacheEntry] = {}
        self._model_context_cache: dict[str, int] = {}
        self._jax = None
        self._jnp = None
        self._available = False
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize JAX backend."""
        try:
            import jax
            import jax.numpy as jnp

            self._jax = jax
            self._jnp = jnp
            self._available = True
        except ImportError:
            self._jax = None
            self._jnp = None
            self._available = False
            logger.warning("JAX not available. Install with: pip install jax jaxlib")

    @property
    def available(self) -> bool:
        """Check if JAX backend is available."""
        return self._available

    def _ensure_jax(self) -> None:
        """Ensure JAX is available."""
        if self._jax is None:
            raise RuntimeError(
                "JAX not available. Install with: pip install jax jaxlib transformers flax"
            )

    def _load_model(self, model_path: Path, adapter: str | None) -> _ModelCacheEntry:
        """Load model and tokenizer, with caching."""
        self._ensure_jax()

        adapter_path = Path(adapter).expanduser().resolve() if adapter else None
        cache_key = (str(model_path), str(adapter_path) if adapter_path else None)
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "transformers with Flax not available. "
                "Install: pip install transformers flax"
            ) from exc

        logger.info("Loading model from %s with JAX backend...", model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )

        # Try Flax model first
        try:
            model = FlaxAutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning("Flax model loading failed, trying PyTorch conversion: %s", e)
            # Fall back to loading from PyTorch weights
            model = FlaxAutoModelForCausalLM.from_pretrained(
                str(model_path),
                from_pt=True,
                trust_remote_code=True,
            )

        if adapter_path:
            logger.warning(
                "Adapter loading not yet implemented for JAX. Ignoring adapter: %s",
                adapter_path,
            )

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

    def _resolve_max_tokens(
        self,
        model_path: Path,
        prompt: str,
        tokenizer: Any,
        max_tokens: int | None,
    ) -> int:
        """Resolve max_tokens based on context and prompt length."""
        context_limit = self._resolve_context_limit(model_path, tokenizer)
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_length = len(token_ids)

        if context_limit is None:
            if max_tokens is None:
                raise ValueError(
                    "max_tokens is required when the model context length cannot be resolved."
                )
            return max_tokens

        available = context_limit - prompt_length
        if available <= 0:
            raise ValueError("Prompt length exceeds model context length.")

        if max_tokens is not None:
            if max_tokens > available:
                raise ValueError(
                    f"Requested max_tokens ({max_tokens}) exceeds available context ({available})."
                )
            return max_tokens

        return available

    def _generate(
        self,
        model_path: Path,
        prompt: str,
        max_tokens: int | None,
        adapter: str | None,
    ) -> _GenerationResult:
        """Generate text using Flax model."""
        entry = self._load_model(model_path, adapter)
        resolved_max_tokens = self._resolve_max_tokens(
            model_path, prompt, entry.tokenizer, max_tokens
        )

        # Encode prompt
        inputs = entry.tokenizer(prompt, return_tensors="jax")
        input_length = inputs.input_ids.shape[1]

        start = time.time()

        # Generate using Flax model
        # Note: Flax models use .generate() similar to PyTorch
        outputs = entry.model.generate(
            inputs.input_ids,
            max_new_tokens=resolved_max_tokens,
            do_sample=False,  # Greedy decoding for determinism
            pad_token_id=entry.tokenizer.pad_token_id or entry.tokenizer.eos_token_id,
        )

        duration = max(time.time() - start, 1e-6)

        # Convert JAX array to a Python list for decoding.
        output_ids = outputs.sequences[0]
        generated_ids = output_ids[input_length:]
        text = entry.tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_count = len(generated_ids)
        tokens_per_second = float(token_count) / duration

        # Determine stop reason
        eos_token_id = entry.tokenizer.eos_token_id
        stop_reason = "stop"
        if token_count >= resolved_max_tokens:
            stop_reason = "length"
        elif token_count > 0 and eos_token_id is not None and int(generated_ids[-1]) == eos_token_id:
            stop_reason = "stop"

        return _GenerationResult(
            text=text,
            token_count=token_count,
            tokens_per_second=tokens_per_second,
            time_to_first_token=None,
            total_duration=duration,
            stop_reason=stop_reason,
        )

    def infer(
        self,
        model: str,
        prompt: str,
        max_tokens: int | None = None,
    ) -> dict:
        """Run inference and return structured results.

        Args:
            model: Path to model directory
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

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
                max_tokens=max_tokens,
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

        Uses Flax model's output_hidden_states to capture hidden states.

        Args:
            model: Path to model directory
            prompt: Input prompt
            adapter: Optional adapter path
            target_layers: Set of layer indices to capture (None = all)

        Returns:
            Dictionary mapping layer index to hidden state vector
        """
        self._ensure_jax()

        model_path = Path(model).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Training is running; inference is locked") from exc

        try:
            entry = self._load_model(model_path, adapter)

            # Encode prompt
            inputs = entry.tokenizer(prompt, return_tensors="jax")

            # Run forward pass with hidden states output
            outputs = entry.model(
                inputs.input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract hidden states
            hidden_states = outputs.hidden_states  # Tuple of layer outputs

            if hidden_states is None:
                raise RuntimeError("Model does not output hidden states.")

            num_layers = len(hidden_states)
            if target_layers is None:
                target_layers = set(range(num_layers))

            # Get last token's hidden state for each layer
            result: dict[int, list[float]] = {}
            for layer_idx in target_layers:
                if layer_idx < num_layers:
                    # Shape: (batch, seq_len, hidden_dim)
                    layer_state = hidden_states[layer_idx]
                    # Get last token, convert to float32, flatten to list
                    last_token_state = layer_state[0, -1, :]
                    result[layer_idx] = (
                        self._jnp.asarray(last_token_state, dtype=self._jnp.float32)
                        .reshape(-1)
                        .tolist()
                    )

            return result

        finally:
            self.lock.release()


def get_inference_engine() -> JAXInferenceEngine:
    """Get the JAX inference engine instance."""
    return JAXInferenceEngine()


__all__ = ["JAXInferenceEngine", "get_inference_engine"]
