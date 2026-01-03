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
Dual-Path Generator for entropy disagreement tracking (JAX Backend).

This is the JAX implementation. For other backends:
- MLX/macOS: see dual_path_mlx.py
- CUDA/PyTorch: see dual_path_cuda.py

Use _platform.get_dual_path_generator() for automatic platform selection.

Implementation based on JAX and Flax reference patterns (2025):
- transformers FlaxAutoModelForCausalLM for model loading
- jax.numpy for tensor operations
- jax.random for sampling
- Flax for model state handling

References:
- https://huggingface.co/docs/transformers/en/model_doc/auto#flax
- https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None
    jnp = None

logger = logging.getLogger(__name__)


def _division_epsilon_for_dtype(array: jnp.ndarray) -> float:
    eps = jnp.finfo(array.dtype).eps
    return float(jnp.sqrt(eps))


@dataclass
class SecurityScanMetricsJAX:
    """Security scan metrics for JAX dual-path generation."""

    token_count: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float


@dataclass
class DualPathGeneratorConfigurationJAX:
    """Configuration for JAX dual-path generator.

    Anomaly thresholds must be derived from baseline measurements.
    No arbitrary defaults - the caller must measure their model to determine
    appropriate thresholds.
    """

    base_model_path: str
    adapter_path: str | None
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    stop_sequences: list[str]
    entropy_top_k: int  # Top-K for entropy calculation
    seed: int

    # Anomaly detection thresholds - MUST be derived from baseline measurements
    kl_divergence_threshold: float | None = None
    """KL divergence above which sample is anomalous. Derive from baseline σ."""

    logit_margin_threshold: float | None = None
    """Logit margin above which sample is anomalous. Derive from baseline σ."""

    rank_fraction_threshold: float | None = None
    """Rank fraction below which sample is anomalous. Derive from baseline σ."""



def compute_token_rank_metrics_jax(
    scores: jnp.ndarray,
    token_id: int,
) -> tuple[int, float, bool]:
    """
    Compute rank geometry for a token in logit space.

    Args:
        scores: 1D array of logit scores.
        token_id: ID of the selected token.

    Returns:
        Tuple of (rank, rank_fraction, frontier_hit).
    """
    if scores.ndim > 1:
        scores = scores.squeeze()

    vocab_size = scores.shape[0]
    token_score = float(scores[token_id])

    # Rank = count of tokens with strictly higher score.
    token_rank = int(jnp.sum(scores > token_score))

    # Rank fraction: 1 = top token, 0 = bottom token.
    if vocab_size > 1:
        rank_fraction = 1.0 - (token_rank / (vocab_size - 1))
    else:
        rank_fraction = 1.0

    # Frontier size is the index of the largest relative gap in sorted scores.
    if vocab_size > 1:
        sorted_scores = -jnp.sort(-scores)
        gaps = sorted_scores[:-1] - sorted_scores[1:]
        eps = _division_epsilon_for_dtype(scores)
        max_gap = float(jnp.max(gaps))
        if max_gap <= eps:
            frontier_size = vocab_size
        else:
            frontier_size = int(jnp.argmax(gaps)) + 1
    else:
        frontier_size = 1

    frontier_hit = token_rank < frontier_size

    return token_rank, rank_fraction, frontier_hit


def compute_entropy_jax(
    logits: jnp.ndarray,
    top_k: int = 100,
) -> tuple[float, float]:
    """
    Compute entropy and variance from logits.

    Args:
        logits: [vocab_size] logit array
        top_k: Number of top tokens to consider

    Returns:
        Tuple of (entropy, variance)
    """
    # Ensure 1D
    if logits.ndim > 1:
        logits = logits.squeeze()

    # Get top-K logits for stability
    if top_k < logits.shape[0]:
        top_logits = jax.lax.top_k(logits, top_k)[0]
    else:
        top_logits = logits

    # Softmax for probabilities
    probs = jax.nn.softmax(top_logits)

    eps = _division_epsilon_for_dtype(probs)

    # Entropy: H = -sum(p * log(p))
    log_probs = jnp.log(probs + eps)
    entropy = float(-jnp.sum(probs * log_probs))

    # Variance of log probabilities
    variance = float(jnp.var(log_probs))

    return entropy, variance


def compute_kl_divergence_jax(
    logits_p: jnp.ndarray,
    logits_q: jnp.ndarray,
    top_k: int = 100,
) -> float:
    """
    Compute KL divergence D_KL(P || Q) from logits.

    Args:
        logits_p: Logits from distribution P
        logits_q: Logits from distribution Q
        top_k: Number of top tokens to consider

    Returns:
        KL divergence value
    """
    if logits_p.ndim > 1:
        logits_p = logits_p.squeeze()
    if logits_q.ndim > 1:
        logits_q = logits_q.squeeze()

    # Apply softmax
    p = jax.nn.softmax(logits_p)
    q = jax.nn.softmax(logits_q)

    eps = _division_epsilon_for_dtype(p)

    # KL divergence
    kl = float(jnp.sum(p * (jnp.log(p + eps) - jnp.log(q + eps))))

    return max(0.0, kl)


@dataclass
class EntropyDeltaSampleJAX:
    """Sample of entropy delta between base and adapter paths."""

    token_index: int
    generated_token_id: int
    base_entropy: float
    base_variance: float
    adapter_entropy: float
    adapter_variance: float
    kl_divergence: float
    base_logit_margin: float
    base_token_logit: float
    base_rank_fraction: float
    base_frontier_hit: bool


class DualPathGeneratorJAX:
    """
    JAX Dual-Path Generator for entropy disagreement tracking.

    Orchestrates dual-path generation comparing base model and adapter model
    outputs for security analysis and anomaly detection.

    Features:
    - Flax/JAX model loading
    - JIT-compiled forward passes
    - Entropy-based anomaly detection
    - Circuit breaker for safety

    Example:
        config = DualPathGeneratorConfigurationJAX(
            base_model_path="meta-llama/Llama-2-7b-hf"
        )
        generator = DualPathGeneratorJAX(config)
        async for chunk in generator.generate("Hello"):
            print(chunk)
    """

    def __init__(
        self,
        config: DualPathGeneratorConfigurationJAX,
        signal_router: Any = None,
    ) -> None:
        """
        Initialize the dual-path generator.

        Args:
            config: Generator configuration
            signal_router: Optional signal router for anomaly events
        """
        self.config = config
        self.signal_router = signal_router
        self.rng_key = jax.random.PRNGKey(config.seed)

        logger.info("Initializing DualPathGeneratorJAX")

        # Lazy imports for optional dependencies
        try:
            from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers package required with flax support. "
                "Install with: pip install transformers[flax]"
            )

        # Load tokenizer
        logger.info("Loading tokenizer from %s", config.base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        logger.info("Loading base model from %s", config.base_model_path)
        self.base_model = FlaxAutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            from_pt=True,  # Convert from PyTorch if needed
        )

        # For JAX, adapter support is more complex
        # We'll use the base model for both paths if no adapter specified
        if config.adapter_path:
            logger.warning(
                "JAX adapter loading is experimental. "
                "For production, consider using CUDA backend with PEFT."
            )
            # Try to load adapter weights manually
            self.adapter_model = self._load_adapter_model(config.adapter_path)
        else:
            self.adapter_model = self.base_model

        # Tracking state
        self.samples: list[EntropyDeltaSampleJAX] = []
        self.anomaly_count = 0

        logger.info("DualPathGeneratorJAX initialized successfully")

    def _load_adapter_model(self, adapter_path: str) -> Any:
        """
        Load adapter model for JAX.

        This is a simplified implementation. Full PEFT/LoRA support
        in JAX would require custom layer merging.
        """
        from transformers import FlaxAutoModelForCausalLM

        # For now, try to load as a full model
        # A proper implementation would merge LoRA weights
        try:
            return FlaxAutoModelForCausalLM.from_pretrained(
                adapter_path,
                from_pt=True,
            )
        except Exception as e:
            logger.warning("Could not load adapter as model: %s. Using base model.", e)
            return self.base_model

    async def generate(self, prompt: str) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate text with dual-path entropy analysis.

        Yields chunks containing:
        - {"type": "token", "text": str}
        - {"type": "anomaly", "sample": EntropyDeltaSampleJAX}
        - {"type": "metrics", "metrics": SecurityScanMetricsJAX}

        Args:
            prompt: Input prompt text

        Yields:
            Generation chunks with tokens, anomalies, and metrics
        """
        self.samples = []
        self.anomaly_count = 0

        start_time = time.time()
        time_to_first = 0.0
        token_count = 0

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="np",  # JAX uses numpy arrays
            padding=True,
            truncation=True,
        )

        input_ids = jnp.array(inputs["input_ids"])
        attention_mask = jnp.array(inputs.get("attention_mask", jnp.ones_like(input_ids)))

        # Initial forward pass (prefill)
        outputs_base = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        outputs_adapter = self.adapter_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits_base = outputs_base.logits[:, -1, :]
        logits_adapter = outputs_adapter.logits[:, -1, :]

        # Generation loop
        generated_ids = input_ids

        while token_count < self.config.max_tokens:
            # Sample from adapter logits
            self.rng_key, subkey = jax.random.split(self.rng_key)
            next_token_id = self._sample(logits_adapter[0], subkey)
            token_id = int(next_token_id)

            # Decode token
            text = self.tokenizer.decode([token_id], skip_special_tokens=True)

            # Compute entropy metrics
            base_entropy, base_variance = compute_entropy_jax(
                logits_base[0], self.config.entropy_top_k
            )
            adapter_entropy, adapter_variance = compute_entropy_jax(
                logits_adapter[0], self.config.entropy_top_k
            )

            # Compute KL divergence
            kl_div = compute_kl_divergence_jax(
                logits_adapter[0], logits_base[0], self.config.entropy_top_k
            )

            # Compute base logit geometry.
            scores_base = logits_base[0]
            token_logit = float(scores_base[token_id])
            max_logit = float(jnp.max(scores_base))
            logit_margin = max(0.0, max_logit - token_logit)

            _, rank_fraction, frontier_hit = compute_token_rank_metrics_jax(
                scores_base, token_id
            )

            # Create sample
            sample = EntropyDeltaSampleJAX(
                token_index=token_count,
                generated_token_id=token_id,
                base_entropy=base_entropy,
                base_variance=base_variance,
                adapter_entropy=adapter_entropy,
                adapter_variance=adapter_variance,
                kl_divergence=kl_div,
                base_logit_margin=logit_margin,
                base_token_logit=token_logit,
                base_rank_fraction=rank_fraction,
                base_frontier_hit=frontier_hit,
            )
            self.samples.append(sample)

            # Yield token
            yield {"type": "token", "text": text}

            # Check for anomalies
            is_anomaly = self._check_anomaly(sample)
            if is_anomaly:
                self.anomaly_count += 1
                yield {"type": "anomaly", "sample": sample}

            # Update state
            token_count += 1
            if token_count == 1:
                time_to_first = (time.time() - start_time) * 1000

            # Prepare next iteration
            next_token_array = jnp.array([[token_id]])
            generated_ids = jnp.concatenate([generated_ids, next_token_array], axis=-1)
            attention_mask = jnp.concatenate(
                [attention_mask, jnp.ones((1, 1), dtype=jnp.int32)], axis=-1
            )

            # Forward pass for next token
            # Note: JAX/Flax models don't always support KV caching as cleanly
            # For efficiency, we'd want to implement proper caching
            outputs_base = self.base_model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
            )
            outputs_adapter = self.adapter_model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
            )

            logits_base = outputs_base.logits[:, -1, :]
            logits_adapter = outputs_adapter.logits[:, -1, :]

            # Check stop conditions
            if token_id == self.tokenizer.eos_token_id:
                break
            if text in self.config.stop_sequences:
                break

        # Final metrics
        total_time = (time.time() - start_time) * 1000
        metrics = SecurityScanMetricsJAX(
            token_count=token_count,
            time_to_first_token_ms=time_to_first,
            total_time_ms=total_time,
            tokens_per_second=token_count / (total_time / 1000) if total_time > 0 else 0,
        )
        yield {"type": "metrics", "metrics": metrics}

    def _sample(self, logits: jnp.ndarray, rng_key: jax.random.PRNGKey) -> int:
        """Sample next token from logits."""
        if self.config.temperature == 0:
            return int(jnp.argmax(logits))

        # Apply temperature
        scaled_logits = logits / self.config.temperature

        # Apply top-k filtering
        if self.config.top_k > 0 and self.config.top_k < logits.shape[0]:
            top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, self.config.top_k)
            # Create mask for non-top-k positions
            mask = jnp.ones_like(scaled_logits) * float("-inf")
            mask = mask.at[top_k_indices].set(scaled_logits[top_k_indices])
            scaled_logits = mask

        # Apply top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_indices = jnp.argsort(scaled_logits)[::-1]
            sorted_logits = scaled_logits[sorted_indices]
            cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

            # Find cutoff
            cutoff_idx = jnp.searchsorted(cumulative_probs, self.config.top_p)
            cutoff_idx = jnp.minimum(cutoff_idx + 1, sorted_logits.shape[0])

            # Mask positions beyond cutoff
            mask = jnp.arange(sorted_logits.shape[0]) < cutoff_idx
            sorted_logits = jnp.where(mask, sorted_logits, float("-inf"))

            # Unsort
            unsort_indices = jnp.argsort(sorted_indices)
            scaled_logits = sorted_logits[unsort_indices]

        # Sample
        probs = jax.nn.softmax(scaled_logits)
        eps = _division_epsilon_for_dtype(probs)
        return int(jax.random.categorical(rng_key, jnp.log(probs + eps)))

    def _check_anomaly(self, sample: EntropyDeltaSampleJAX) -> bool:
        """Check if sample represents an anomaly.

        Uses caller-provided thresholds from config. If thresholds are not provided,
        no anomaly detection is performed (returns False).

        Thresholds should be derived from baseline measurements:
        - kl_divergence_threshold: baseline_mean + 2*baseline_std
        - logit_margin_threshold: baseline_mean + 2*baseline_std
        - rank_fraction_threshold: baseline_mean - 2*baseline_std
        """
        # High KL divergence indicates disagreement
        if self.config.kl_divergence_threshold is not None:
            if sample.kl_divergence > self.config.kl_divergence_threshold:
                return True
        # High logit margin indicates unexpected token
        if self.config.logit_margin_threshold is not None:
            if sample.base_logit_margin > self.config.logit_margin_threshold:
                return True
        # Low rank fraction indicates out-of-frontier selection
        if self.config.rank_fraction_threshold is not None:
            if sample.base_rank_fraction < self.config.rank_fraction_threshold:
                return True
        return False

__all__ = [
    "DualPathGeneratorJAX",
    "DualPathGeneratorConfigurationJAX",
    "SecurityScanMetricsJAX",
    "EntropyDeltaSampleJAX",
    "compute_token_rank_metrics_jax",
    "compute_entropy_jax",
    "compute_kl_divergence_jax",
]
