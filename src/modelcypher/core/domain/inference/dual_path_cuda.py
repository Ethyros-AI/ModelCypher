"""
Dual-Path Generator for entropy disagreement tracking (CUDA/PyTorch Backend).

This is the PyTorch/CUDA implementation. For other backends:
- MLX/macOS: see dual_path_mlx.py
- JAX/TPU: see dual_path_jax.py

Use _platform.get_dual_path_generator() for automatic platform selection.

Implementation based on PyTorch 2.9 and Transformers 4.x (2025):
- transformers.AutoModelForCausalLM for model loading
- peft.PeftModel for LoRA adapter support
- torch.no_grad() for inference efficiency
- torch.amp.autocast for mixed precision

References:
- https://huggingface.co/docs/transformers/en/main_classes/text_generation
- https://huggingface.co/docs/peft/en/quicktour
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SecurityScanMetricsCUDA:
    """Security scan metrics for CUDA dual-path generation."""
    token_count: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    circuit_breaker_tripped: bool
    anomaly_alert_count: int


@dataclass
class DualPathGeneratorConfigurationCUDA:
    """Configuration for CUDA dual-path generator."""
    base_model_path: str
    adapter_path: str | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    halt_on_circuit_breaker: bool = True
    device: str = "cuda:0"
    dtype: str = "float16"  # float16, bfloat16, float32
    entropy_top_k: int = 100  # Top-K for entropy calculation


def compute_token_rank_metrics_cuda(
    probabilities: torch.Tensor,
    token_id: int,
    top_k: int = 10,
) -> tuple[int, float, bool]:
    """
    Compute ranking-based metrics for a token in a probability distribution.

    Args:
        probabilities: 1D tensor of token probabilities
        token_id: ID of the selected token
        top_k: Threshold for top-K hit detection

    Returns:
        Tuple of (rank, normalized_approval, top_k_hit)
    """
    vocab_size = probabilities.shape[0]
    token_prob = probabilities[token_id].item()

    # Rank = count of tokens with strictly higher probability
    token_rank = int((probabilities > token_prob).sum().item())

    # Normalized approval: 1 = top token, 0 = bottom token
    if vocab_size > 1:
        normalized_approval = 1.0 - (token_rank / (vocab_size - 1))
    else:
        normalized_approval = 1.0

    # Top-K hit
    top_k_hit = token_rank < top_k

    return token_rank, normalized_approval, top_k_hit


def compute_entropy_cuda(
    logits: torch.Tensor,
    top_k: int = 100,
) -> tuple[float, float]:
    """
    Compute entropy and variance from logits.

    Args:
        logits: [vocab_size] logit tensor
        top_k: Number of top tokens to consider

    Returns:
        Tuple of (entropy, variance)
    """
    # Get top-K logits for stability
    if logits.dim() > 1:
        logits = logits.squeeze()

    if top_k < logits.shape[0]:
        top_logits, _ = torch.topk(logits, top_k)
    else:
        top_logits = logits

    # Softmax for probabilities
    probs = F.softmax(top_logits, dim=-1)

    # Entropy: H = -sum(p * log(p))
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs).item()

    # Variance of log probabilities
    variance = torch.var(log_probs).item()

    return entropy, variance


def compute_kl_divergence_cuda(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
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
    if logits_p.dim() > 1:
        logits_p = logits_p.squeeze()
    if logits_q.dim() > 1:
        logits_q = logits_q.squeeze()

    # Apply softmax
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)

    # KL divergence
    kl = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).item()

    return max(0.0, kl)


@dataclass
class EntropyDeltaSampleCUDA:
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


class DualPathGeneratorCUDA:
    """
    CUDA Dual-Path Generator for entropy disagreement tracking.

    Orchestrates dual-path generation comparing base model and adapter model
    outputs for security analysis and anomaly detection.

    Features:
    - Transformers model loading with device placement
    - PEFT LoRA adapter support
    - Mixed precision inference
    - Entropy-based anomaly detection
    - Circuit breaker for safety

    Example:
        config = DualPathGeneratorConfigurationCUDA(
            base_model_path="meta-llama/Llama-2-7b-hf",
            adapter_path="./my_adapter",
            device="cuda:0"
        )
        generator = DualPathGeneratorCUDA(config)
        async for chunk in generator.generate("Hello"):
            print(chunk)
    """

    def __init__(
        self,
        config: DualPathGeneratorConfigurationCUDA,
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
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(config.dtype, torch.float16)

        logger.info("Initializing DualPathGeneratorCUDA on %s", self.device)

        # Lazy imports for optional dependencies
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        # Load tokenizer
        logger.info("Loading tokenizer from %s", config.base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        logger.info("Loading base model from %s", config.base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.base_model.eval()

        # Load adapter model if specified
        if config.adapter_path:
            logger.info("Loading adapter from %s", config.adapter_path)
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    "peft package required for adapter support. "
                    "Install with: pip install peft"
                )

            # Load a separate instance with the adapter
            adapter_base = AutoModelForCausalLM.from_pretrained(
                config.base_model_path,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.adapter_model = PeftModel.from_pretrained(
                adapter_base,
                config.adapter_path,
            )
            self.adapter_model.eval()
        else:
            # No adapter = both paths use same model
            self.adapter_model = self.base_model

        # Tracking state
        self.samples: list[EntropyDeltaSampleCUDA] = []
        self.anomaly_count = 0
        self.circuit_breaker_tripped = False

        logger.info("DualPathGeneratorCUDA initialized successfully")

    async def generate(self, prompt: str) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate text with dual-path entropy analysis.

        Yields chunks containing:
        - {"type": "token", "text": str}
        - {"type": "anomaly", "sample": EntropyDeltaSampleCUDA}
        - {"type": "circuit_breaker", "samples": List}
        - {"type": "metrics", "metrics": SecurityScanMetricsCUDA}

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
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # Initialize past key values for caching
        past_base = None
        past_adapter = None

        with torch.no_grad():
            # Initial forward pass (prefill)
            outputs_base = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            outputs_adapter = self.adapter_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            past_base = outputs_base.past_key_values
            past_adapter = outputs_adapter.past_key_values
            logits_base = outputs_base.logits[:, -1, :]
            logits_adapter = outputs_adapter.logits[:, -1, :]

            # Generation loop
            generated_ids = input_ids.clone()

            while token_count < self.config.max_tokens:
                # Sample from adapter logits
                next_token_id = self._sample(logits_adapter)
                token_id = next_token_id.item()

                # Decode token
                text = self.tokenizer.decode([token_id], skip_special_tokens=True)

                # Compute entropy metrics
                base_entropy, base_variance = compute_entropy_cuda(
                    logits_base[0], self.config.entropy_top_k
                )
                adapter_entropy, adapter_variance = compute_entropy_cuda(
                    logits_adapter[0], self.config.entropy_top_k
                )

                # Compute KL divergence
                kl_div = compute_kl_divergence_cuda(
                    logits_adapter[0], logits_base[0], self.config.entropy_top_k
                )

                # Compute base model approval
                probs_base = F.softmax(logits_base[0], dim=-1)
                token_prob = probs_base[token_id].item()
                surprisal = -torch.log(probs_base[token_id] + 1e-10).item()

                _, normalized_approval, top_k_hit = compute_token_rank_metrics_cuda(
                    probs_base, token_id, top_k=10
                )

                # Create sample
                sample = EntropyDeltaSampleCUDA(
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
                generated_ids = torch.cat(
                    [generated_ids, next_token_id.unsqueeze(0)], dim=-1
                )

                # Forward pass for next token
                outputs_base = self.base_model(
                    input_ids=next_token_id.unsqueeze(0),
                    past_key_values=past_base,
                    use_cache=True,
                )
                outputs_adapter = self.adapter_model(
                    input_ids=next_token_id.unsqueeze(0),
                    past_key_values=past_adapter,
                    use_cache=True,
                )

                past_base = outputs_base.past_key_values
                past_adapter = outputs_adapter.past_key_values
                logits_base = outputs_base.logits[:, -1, :]
                logits_adapter = outputs_adapter.logits[:, -1, :]

                # Check stop conditions
                if token_id == self.tokenizer.eos_token_id:
                    break
                if text in self.config.stop_sequences:
                    break

        # Final metrics
        total_time = (time.time() - start_time) * 1000
        metrics = SecurityScanMetricsCUDA(
            token_count=token_count,
            time_to_first_token_ms=time_to_first,
            total_time_ms=total_time,
            tokens_per_second=token_count / (total_time / 1000) if total_time > 0 else 0,
            circuit_breaker_tripped=self.circuit_breaker_tripped,
            anomaly_alert_count=self.anomaly_count,
        )
        yield {"type": "metrics", "metrics": metrics}

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token from logits."""
        if self.config.temperature == 0:
            return torch.argmax(logits, dim=-1)

        # Apply temperature
        scaled_logits = logits / self.config.temperature

        # Apply top-k filtering
        if self.config.top_k > 0:
            indices_to_remove = scaled_logits < torch.topk(scaled_logits, self.config.top_k)[0][..., -1, None]
            scaled_logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            scaled_logits[indices_to_remove] = float("-inf")

        # Sample
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _check_anomaly(self, sample: EntropyDeltaSampleCUDA) -> bool:
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
        anomaly_rate = sum(1 for s in recent_samples if self._check_anomaly(s)) / len(recent_samples)
        return anomaly_rate > 0.5


__all__ = [
    "DualPathGeneratorCUDA",
    "DualPathGeneratorConfigurationCUDA",
    "SecurityScanMetricsCUDA",
    "EntropyDeltaSampleCUDA",
    "compute_token_rank_metrics_cuda",
    "compute_entropy_cuda",
    "compute_kl_divergence_cuda",
]
