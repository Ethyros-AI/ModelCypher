"""
JAX Evaluation Engine for Model and LoRA Checkpoint Assessment.

This is the JAX/Flax implementation. For other backends:
- MLX/macOS: see evaluation_mlx.py
- CUDA/PyTorch: see evaluation_cuda.py

Use _platform.get_evaluation_engine() for automatic platform selection.

Implementation based on JAX best practices (2025):
- jax.jit for optimized forward passes
- optax.softmax_cross_entropy for loss computation
- JAX numpy operations for accuracy computation
- Pure functional evaluation loops

References:
- https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html
- https://optax.readthedocs.io/en/latest/api/losses.html
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

logger = logging.getLogger(__name__)


class EvaluationMetricJAX(str, Enum):
    """Available evaluation metrics."""
    LOSS = "loss"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    BITS_PER_CHARACTER = "bpc"


@dataclass
class EvaluationConfigJAX:
    """Configuration for evaluation runs."""
    metrics: List[EvaluationMetricJAX] = field(
        default_factory=lambda: [EvaluationMetricJAX.LOSS, EvaluationMetricJAX.PERPLEXITY]
    )
    batch_size: int = 4
    sequence_length: int = 512
    max_samples: Optional[int] = None

    @classmethod
    def default(cls) -> "EvaluationConfigJAX":
        return cls()


@dataclass
class EvaluationProgressJAX:
    """Progress update during evaluation."""
    samples_processed: int
    total_samples: int
    current_metric: Optional[float] = None

    @property
    def percentage(self) -> float:
        return self.samples_processed / max(self.total_samples, 1)


@dataclass
class EvaluationResultJAX:
    """Result of an evaluation run."""
    metrics: Dict[EvaluationMetricJAX, float]
    samples_evaluated: int
    tokens_evaluated: int
    duration_seconds: float

    @property
    def loss(self) -> Optional[float]:
        return self.metrics.get(EvaluationMetricJAX.LOSS)

    @property
    def perplexity(self) -> Optional[float]:
        return self.metrics.get(EvaluationMetricJAX.PERPLEXITY)

    @property
    def accuracy(self) -> Optional[float]:
        return self.metrics.get(EvaluationMetricJAX.ACCURACY)


@dataclass
class EvaluationBatchJAX:
    """A single evaluation batch."""
    inputs: jnp.ndarray      # [batch, seq_len] int32
    targets: jnp.ndarray     # [batch, seq_len] int32
    mask: jnp.ndarray        # [batch, seq_len] float32
    valid_token_counts: List[int]


class EvaluationErrorJAX(Exception):
    """Evaluation failed."""
    pass


# =============================================================================
# Evaluation Engine
# =============================================================================

class EvaluationEngineJAX:
    """
    JAX Evaluation Engine.

    Computes loss, perplexity, and accuracy using JIT-compiled
    batch processing for TPU/GPU efficiency.

    Features (matching MLX parity):
    - Batched loss/perplexity computation
    - Next-token prediction accuracy
    - Progress callbacks
    - JIT-compiled metric computation
    """

    def __init__(self, config: Optional[EvaluationConfigJAX] = None) -> None:
        self.config = config or EvaluationConfigJAX.default()

    def evaluate(
        self,
        apply_fn: Callable,
        params: Dict[str, Any],
        batches: Iterator[EvaluationBatchJAX],
        total_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[EvaluationProgressJAX], None]] = None,
    ) -> EvaluationResultJAX:
        """
        Evaluate a model on a dataset.

        Args:
            apply_fn: Model forward function: apply_fn(params, inputs) -> logits
            params: Model parameters (JAX pytree)
            batches: Iterator of evaluation batches
            total_samples: Total sample count for progress
            progress_callback: Callback for progress updates

        Returns:
            EvaluationResultJAX with computed metrics
        """
        start_time = time.time()

        total_loss = 0.0
        total_correct = 0.0
        total_tokens = 0
        samples_processed = 0

        needs_loss = (
            EvaluationMetricJAX.LOSS in self.config.metrics or
            EvaluationMetricJAX.PERPLEXITY in self.config.metrics or
            EvaluationMetricJAX.BITS_PER_CHARACTER in self.config.metrics
        )
        needs_accuracy = EvaluationMetricJAX.ACCURACY in self.config.metrics

        # JIT-compile the evaluation step
        @jax.jit
        def eval_step(params, inputs, targets, mask):
            logits = apply_fn(params, inputs)
            return logits

        for batch in batches:
            # Forward pass
            logits = eval_step(params, batch.inputs, batch.targets, batch.mask)

            batch_tokens = sum(batch.valid_token_counts)
            total_tokens += batch_tokens

            if needs_loss:
                batch_loss = self._compute_batch_loss(logits, batch.targets, batch.mask)
                total_loss += batch_loss

            if needs_accuracy:
                batch_correct = self._compute_batch_accuracy(logits, batch.targets, batch.mask)
                total_correct += batch_correct

            samples_processed += batch.inputs.shape[0]

            if progress_callback and total_samples:
                progress_callback(EvaluationProgressJAX(
                    samples_processed=samples_processed,
                    total_samples=total_samples,
                ))

        if total_tokens == 0:
            raise EvaluationErrorJAX("Evaluation produced zero tokens. Check dataset format.")

        # Compute final metrics
        metrics: Dict[EvaluationMetricJAX, float] = {}

        if needs_loss:
            avg_loss = total_loss / total_tokens
            if EvaluationMetricJAX.LOSS in self.config.metrics:
                metrics[EvaluationMetricJAX.LOSS] = avg_loss
            if EvaluationMetricJAX.PERPLEXITY in self.config.metrics:
                metrics[EvaluationMetricJAX.PERPLEXITY] = math.exp(avg_loss)
            if EvaluationMetricJAX.BITS_PER_CHARACTER in self.config.metrics:
                metrics[EvaluationMetricJAX.BITS_PER_CHARACTER] = avg_loss / math.log(2)

        if needs_accuracy:
            metrics[EvaluationMetricJAX.ACCURACY] = total_correct / total_tokens

        duration = time.time() - start_time
        logger.info(
            "Evaluation completed: %d samples, %d tokens in %.2fs",
            samples_processed,
            total_tokens,
            duration,
        )

        return EvaluationResultJAX(
            metrics=metrics,
            samples_evaluated=samples_processed,
            tokens_evaluated=total_tokens,
            duration_seconds=duration,
        )

    def _compute_batch_loss(
        self,
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> float:
        """
        Compute sum of negative log-likelihood for a batch.

        Args:
            logits: [batch, seq, vocab] model outputs
            targets: [batch, seq] target token IDs
            mask: [batch, seq] attention mask

        Returns:
            Sum of NLL values
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Cross-entropy without reduction
        nll = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits_flat,
            labels=targets_flat,
        )

        # Apply mask
        nll = nll * mask_flat
        total_nll = float(jnp.sum(nll))

        return total_nll

    def _compute_batch_accuracy(
        self,
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> float:
        """
        Compute sum of correct predictions for a batch.

        Args:
            logits: [batch, seq, vocab] model outputs
            targets: [batch, seq] target token IDs
            mask: [batch, seq] attention mask

        Returns:
            Count of correct predictions
        """
        # Argmax predictions
        predictions = jnp.argmax(logits, axis=-1)

        # Check equality
        correct = (predictions == targets).astype(jnp.float32) * mask

        total_correct = float(jnp.sum(correct))
        return total_correct


# =============================================================================
# Batch Iterator
# =============================================================================

class DatasetBatchIteratorJAX:
    """
    JAX-optimized dataset batch iterator.

    Iterates over a dataset producing padded evaluation batches
    as JAX arrays.
    """

    def __init__(
        self,
        texts: List[str],
        tokenize_fn: Callable[[str], List[int]],
        sequence_length: int = 512,
        batch_size: int = 4,
        max_samples: Optional[int] = None,
    ) -> None:
        self.texts = texts
        self.tokenize = tokenize_fn
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.cursor = 0

    def __iter__(self) -> Iterator[EvaluationBatchJAX]:
        self.cursor = 0
        return self

    def __next__(self) -> EvaluationBatchJAX:
        if self.cursor >= len(self.texts):
            raise StopIteration

        if self.max_samples and self.cursor >= self.max_samples:
            raise StopIteration

        inputs_list: List[List[int]] = []
        targets_list: List[List[int]] = []
        mask_list: List[List[float]] = []
        valid_counts: List[int] = []

        while len(inputs_list) < self.batch_size and self.cursor < len(self.texts):
            if self.max_samples and self.cursor >= self.max_samples:
                break

            text = self.texts[self.cursor]
            self.cursor += 1

            tokens = self.tokenize(text)
            if len(tokens) < 2:
                continue

            # Truncate to sequence length
            tokens = tokens[:self.sequence_length]
            if len(tokens) < 2:
                continue

            valid_count = len(tokens) - 1
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            # Pad to sequence_length - 1
            pad_len = (self.sequence_length - 1) - len(input_tokens)
            if pad_len > 0:
                input_tokens = input_tokens + [0] * pad_len
                target_tokens = target_tokens + [0] * pad_len

            # Create mask
            mask = [1.0] * valid_count + [0.0] * pad_len

            inputs_list.append(input_tokens)
            targets_list.append(target_tokens)
            mask_list.append(mask)
            valid_counts.append(valid_count)

        if not inputs_list:
            raise StopIteration

        return EvaluationBatchJAX(
            inputs=jnp.array(inputs_list, dtype=jnp.int32),
            targets=jnp.array(targets_list, dtype=jnp.int32),
            mask=jnp.array(mask_list, dtype=jnp.float32),
            valid_token_counts=valid_counts,
        )

    def __len__(self) -> int:
        """Approximate number of batches."""
        effective_samples = min(len(self.texts), self.max_samples or len(self.texts))
        return (effective_samples + self.batch_size - 1) // self.batch_size


# =============================================================================
# LoRA Checkpoint Evaluation
# =============================================================================

def evaluate_lora_checkpoint_jax(
    apply_fn: Callable,
    params: Dict[str, Any],
    lora_params: Dict[str, Dict[str, jnp.ndarray]],
    checkpoint_path: Path,
    texts: List[str],
    tokenize_fn: Callable[[str], List[int]],
    config: Optional[EvaluationConfigJAX] = None,
    progress_callback: Optional[Callable[[EvaluationProgressJAX], None]] = None,
) -> EvaluationResultJAX:
    """
    Evaluate a LoRA checkpoint against a dataset.

    Args:
        apply_fn: Model forward function with LoRA
        params: Base model parameters
        lora_params: LoRA adapter parameters
        checkpoint_path: Path to LoRA checkpoint
        texts: List of evaluation texts
        tokenize_fn: Tokenization function
        config: Evaluation configuration
        progress_callback: Optional progress callback

    Returns:
        EvaluationResultJAX with metrics
    """
    from .lora_jax import load_lora_adapters_jax

    config = config or EvaluationConfigJAX.default()

    # Load checkpoint
    loaded_lora = load_lora_adapters_jax(checkpoint_path)

    # Merge loaded LoRA into current lora_params
    for key, value in loaded_lora.items():
        lora_params[key] = value

    logger.info("Loaded LoRA adapters from checkpoint")

    # Create batch iterator
    iterator = DatasetBatchIteratorJAX(
        texts=texts,
        tokenize_fn=tokenize_fn,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        max_samples=config.max_samples,
    )

    # Evaluate
    engine = EvaluationEngineJAX(config)
    return engine.evaluate(
        apply_fn=apply_fn,
        params=params,
        batches=iter(iterator),
        total_samples=min(len(texts), config.max_samples or len(texts)),
        progress_callback=progress_callback,
    )


__all__ = [
    "EvaluationMetricJAX",
    "EvaluationConfigJAX",
    "EvaluationProgressJAX",
    "EvaluationResultJAX",
    "EvaluationBatchJAX",
    "EvaluationEngineJAX",
    "EvaluationErrorJAX",
    "DatasetBatchIteratorJAX",
    "evaluate_lora_checkpoint_jax",
]
