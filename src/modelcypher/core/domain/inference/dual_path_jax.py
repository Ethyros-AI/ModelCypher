"""
Dual-Path Generator for entropy disagreement tracking (JAX Backend).

This is the JAX implementation. For other backends:
- MLX/macOS: see dual_path_mlx.py
- CUDA/PyTorch: see dual_path_cuda.py

Use _platform.get_dual_path_generator() for automatic platform selection.

Implementation based on JAX and Flax best practices (2025):
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
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SecurityScanMetricsJAX:
    """Security scan metrics for JAX dual-path generation."""
    token_count: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    circuit_breaker_tripped: bool
    anomaly_alert_count: int


@dataclass
class DualPathGeneratorConfigurationJAX:
    """Configuration for JAX dual-path generator."""
    base_model_path: str
    adapter_path: str | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    halt_on_circuit_breaker: bool = True
    entropy_top_k: int = 100  # Top-K for entropy calculation
    seed: int = 42


def compute_token_rank_metrics_jax(
    probabilities: jnp.ndarray,
    token_id: int,
    top_k: int = 10,
) -> tuple[int, float, bool]:
    """
    Compute ranking-based metrics for a token in a probability distribution.

    Args:
        probabilities: 1D array of token probabilities
        token_id: ID of the selected token
        top_k: Threshold for top-K hit detection

    Returns:
        Tuple of (rank, normalized_approval, top_k_hit)
    """
    vocab_size = probabilities.shape[0]
    token_prob = float(probabilities[token_id])

    # Rank = count of tokens with strictly higher probability
    token_rank = int(jnp.sum(probabilities > token_prob))

    # Normalized approval: 1 = top token, 0 = bottom token
    if vocab_size > 1:
        normalized_approval = 1.0 - (token_rank / (vocab_size - 1))
    else:
        normalized_approval = 1.0

    # Top-K hit
    top_k_hit = token_rank < top_k

    return token_rank, normalized_approval, top_k_hit


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

    # Entropy: H = -sum(p * log(p))
    log_probs = jnp.log(probs + 1e-10)
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

    # KL divergence
    kl = float(jnp.sum(p * (jnp.log(p + 1e-10) - jnp.log(q + 1e-10))))

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
    base_surprisal: float
    base_approval_prob: float
    normalized_approval: float
    base_top_k_hit: bool


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
        self.circuit_breaker_tripped = False

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
            logger.warning(
                "Could not load adapter as model: %s. Using base model.", e
            )
            return self.base_model

    async def generate(self, prompt: str) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate text with dual-path entropy analysis.

        Yields chunks containing:
        - {"type": "token", "text": str}
        - {"type": "anomaly", "sample": EntropyDeltaSampleJAX}
        - {"type": "circuit_breaker", "samples": List}
        - {"type": "metrics", "metrics": SecurityScanMetricsJAX}

        Args:
            prompt: Input prompt text

        Yields:
            Generation chunks with tokens, anomalies, and metrics
        """
        self.samples = []
        self.anomaly_count = 0
        self.circuit_breaker_tripped = False

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

            # Compute base model approval
            probs_base = jax.nn.softmax(logits_base[0])
            token_prob = float(probs_base[token_id])
            surprisal = float(-jnp.log(probs_base[token_id] + 1e-10))

            _, normalized_approval, top_k_hit = compute_token_rank_metrics_jax(
                probs_base, token_id, top_k=10
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
                base_surprisal=surprisal,
                base_approval_prob=token_prob,
                normalized_approval=normalized_approval,
                base_top_k_hit=top_k_hit,
            )
            self.samples.append(sample)

            # Yield token
            yield {"type": "token", "text": text}

            # Check for anomalies
            is_anomaly = self._check_anomaly(sample)
            if is_anomaly:
                self.anomaly_count += 1
                yield {"type": "anomaly", "sample": sample}

            # Check circuit breaker
            if self.config.halt_on_circuit_breaker and self._check_circuit_breaker():
                self.circuit_breaker_tripped = True
                yield {"type": "circuit_breaker", "samples": self.samples}
                break

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
            circuit_breaker_tripped=self.circuit_breaker_tripped,
            anomaly_alert_count=self.anomaly_count,
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
        return int(jax.random.categorical(rng_key, jnp.log(probs + 1e-10)))

    def _check_anomaly(self, sample: EntropyDeltaSampleJAX) -> bool:
        """Check if sample represents an anomaly."""
        # High KL divergence indicates disagreement
        if sample.kl_divergence > 5.0:
            return True
        # High surprisal indicates unexpected token
        if sample.base_surprisal > 10.0:
            return True
        # Low approval indicates base model disapproves
        if sample.normalized_approval < 0.1:
            return True
        return False

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should trip."""
        if len(self.samples) < 5:
            return False

        # Trip if too many recent anomalies
        recent_samples = self.samples[-10:]
        anomaly_rate = sum(
            1 for s in recent_samples if self._check_anomaly(s)
        ) / len(recent_samples)
        return anomaly_rate > 0.5


__all__ = [
    "DualPathGeneratorJAX",
    "DualPathGeneratorConfigurationJAX",
    "SecurityScanMetricsJAX",
    "EntropyDeltaSampleJAX",
    "compute_token_rank_metrics_jax",
    "compute_entropy_jax",
    "compute_kl_divergence_jax",
]
