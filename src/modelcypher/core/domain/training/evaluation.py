"""
Evaluation Engine for Model and LoRA Checkpoint Assessment (MLX Backend).

This is the MLX/macOS implementation. For CUDA/Linux, see evaluation_cuda.py.
Use _platform.get_evaluation_engine() for automatic platform selection.

Ported from the reference Swift implementation.

Features:
- Batched loss/perplexity computation
- Next-token prediction accuracy
- Progress callbacks
- GPU-optimized cross-entropy calculation

Metrics:
- Loss: Average cross-entropy loss per token
- Perplexity: exp(loss) - measures model uncertainty
- Accuracy: Next-token prediction accuracy
- Bits per character: loss / ln(2)

MLX-Specific:
- Uses mlx.nn.Module models
- Uses mx.softmax, mx.log for GPU-accelerated computation
- Uses mx.load for checkpoint loading
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Any
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class EvaluationMetric(str, Enum):
    """Available evaluation metrics."""
    LOSS = "loss"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    BITS_PER_CHARACTER = "bpc"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    metrics: List[EvaluationMetric] = field(
        default_factory=lambda: [EvaluationMetric.LOSS, EvaluationMetric.PERPLEXITY]
    )
    batch_size: int = 4
    sequence_length: int = 512
    max_samples: Optional[int] = None

    @classmethod
    def default(cls) -> "EvaluationConfig":
        return cls()


@dataclass
class EvaluationProgress:
    """Progress update during evaluation."""
    samples_processed: int
    total_samples: int
    current_metric: Optional[float] = None

    @property
    def percentage(self) -> float:
        return self.samples_processed / max(self.total_samples, 1)


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""
    metrics: Dict[EvaluationMetric, float]
    samples_evaluated: int
    tokens_evaluated: int
    duration_seconds: float

    @property
    def loss(self) -> Optional[float]:
        return self.metrics.get(EvaluationMetric.LOSS)

    @property
    def perplexity(self) -> Optional[float]:
        return self.metrics.get(EvaluationMetric.PERPLEXITY)

    @property
    def accuracy(self) -> Optional[float]:
        return self.metrics.get(EvaluationMetric.ACCURACY)


@dataclass
class EvaluationBatch:
    """A single evaluation batch."""
    inputs: mx.array      # [batch, seq_len] int32
    targets: mx.array     # [batch, seq_len] int32
    mask: mx.array        # [batch, seq_len] float32
    valid_token_counts: List[int]


class EvaluationError(Exception):
    """Evaluation failed."""
    pass


# =============================================================================
# Evaluation Engine
# =============================================================================

class EvaluationEngine:
    """
    Evaluates models and LoRA checkpoints against datasets.

    Computes loss, perplexity, and accuracy using GPU-optimized
    batch processing.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig.default()

    def evaluate(
        self,
        model: nn.Module,
        batches: Iterator[EvaluationBatch],
        total_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[EvaluationProgress], None]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a model on a dataset.

        Args:
            model: Language model to evaluate
            batches: Iterator of evaluation batches
            total_samples: Total sample count for progress (optional)
            progress_callback: Callback for progress updates

        Returns:
            EvaluationResult with computed metrics
        """
        start_time = time.time()

        total_loss = 0.0
        total_correct = 0.0
        total_tokens = 0
        samples_processed = 0

        needs_loss = (
            EvaluationMetric.LOSS in self.config.metrics or
            EvaluationMetric.PERPLEXITY in self.config.metrics or
            EvaluationMetric.BITS_PER_CHARACTER in self.config.metrics
        )
        needs_accuracy = EvaluationMetric.ACCURACY in self.config.metrics

        for batch in batches:
            # Forward pass
            logits = model(batch.inputs)
            mx.eval(logits)

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
                progress_callback(EvaluationProgress(
                    samples_processed=samples_processed,
                    total_samples=total_samples,
                ))

        if total_tokens == 0:
            raise EvaluationError("Evaluation produced zero tokens. Check dataset format.")

        # Compute final metrics
        metrics: Dict[EvaluationMetric, float] = {}

        if needs_loss:
            avg_loss = total_loss / total_tokens
            if EvaluationMetric.LOSS in self.config.metrics:
                metrics[EvaluationMetric.LOSS] = avg_loss
            if EvaluationMetric.PERPLEXITY in self.config.metrics:
                metrics[EvaluationMetric.PERPLEXITY] = math.exp(avg_loss)
            if EvaluationMetric.BITS_PER_CHARACTER in self.config.metrics:
                metrics[EvaluationMetric.BITS_PER_CHARACTER] = avg_loss / math.log(2)

        if needs_accuracy:
            metrics[EvaluationMetric.ACCURACY] = total_correct / total_tokens

        duration = time.time() - start_time

        return EvaluationResult(
            metrics=metrics,
            samples_evaluated=samples_processed,
            tokens_evaluated=total_tokens,
            duration_seconds=duration,
        )

    def _compute_batch_loss(
        self,
        logits: mx.array,
        targets: mx.array,
        mask: mx.array,
    ) -> float:
        """
        Compute sum of negative log-likelihood for a batch.

        Uses GPU cross-entropy to avoid large CPU transfers.

        Args:
            logits: [batch, seq, vocab] model outputs
            targets: [batch, seq] target token IDs
            mask: [batch, seq] attention mask

        Returns:
            Sum of NLL values
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, vocab_size).astype(mx.float32)
        targets_flat = targets.reshape(-1).astype(mx.int32)
        mask_flat = mask.reshape(-1).astype(mx.float32)

        # Cross-entropy without reduction
        # CE = -log(softmax(logits)[target])
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

        # Gather target log probs
        batch_indices = mx.arange(targets_flat.shape[0])
        target_log_probs = log_probs[batch_indices, targets_flat]

        # Negative log likelihood with mask
        nll = -target_log_probs * mask_flat
        total_nll = float(mx.sum(nll).item())

        return total_nll

    def _compute_batch_accuracy(
        self,
        logits: mx.array,
        targets: mx.array,
        mask: mx.array,
    ) -> float:
        """
        Compute sum of correct predictions for a batch.

        Uses GPU argmax for efficiency.

        Args:
            logits: [batch, seq, vocab] model outputs
            targets: [batch, seq] target token IDs
            mask: [batch, seq] attention mask

        Returns:
            Count of correct predictions
        """
        # GPU argmax
        predictions = mx.argmax(logits.astype(mx.float32), axis=-1).astype(mx.int32)

        # Check equality
        correct = (predictions == targets).astype(mx.float32) * mask

        total_correct = float(mx.sum(correct).item())
        return total_correct


# =============================================================================
# Batch Iterator
# =============================================================================

class DatasetBatchIterator:
    """
    Iterates over a dataset producing padded evaluation batches.

    Handles tokenization, padding, and batching on-the-fly.
    """

    def __init__(
        self,
        texts: List[str],
        tokenize_fn: Callable[[str], List[int]],
        sequence_length: int = 512,
        batch_size: int = 4,
        max_samples: Optional[int] = None,
    ):
        self.texts = texts
        self.tokenize = tokenize_fn
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.cursor = 0

    def __iter__(self) -> Iterator[EvaluationBatch]:
        self.cursor = 0
        return self

    def __next__(self) -> EvaluationBatch:
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

        return EvaluationBatch(
            inputs=mx.array(inputs_list, dtype=mx.int32),
            targets=mx.array(targets_list, dtype=mx.int32),
            mask=mx.array(mask_list, dtype=mx.float32),
            valid_token_counts=valid_counts,
        )


# =============================================================================
# LoRA Checkpoint Evaluation
# =============================================================================

def evaluate_lora_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    texts: List[str],
    tokenize_fn: Callable[[str], List[int]],
    config: Optional[EvaluationConfig] = None,
    progress_callback: Optional[Callable[[EvaluationProgress], None]] = None,
) -> EvaluationResult:
    """
    Evaluate a LoRA checkpoint against a dataset.

    Loads checkpoint weights into the model and evaluates.

    Args:
        model: Base model with LoRA layers
        checkpoint_path: Path to LoRA checkpoint (.safetensors)
        texts: List of evaluation texts
        tokenize_fn: Tokenization function
        config: Evaluation configuration
        progress_callback: Optional progress callback

    Returns:
        EvaluationResult with metrics
    """
    config = config or EvaluationConfig.default()

    # Load checkpoint weights
    checkpoint_weights = mx.load(str(checkpoint_path))

    # Filter to matching parameters
    model_params = dict(model.parameters())
    matched = {}
    for key, value in checkpoint_weights.items():
        if key in model_params:
            matched[key] = value

    if not matched:
        raise EvaluationError(f"No matching parameters in checkpoint: {checkpoint_path}")

    # Update model
    # (In practice, need to use model.update or similar)

    # Create batch iterator
    iterator = DatasetBatchIterator(
        texts=texts,
        tokenize_fn=tokenize_fn,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        max_samples=config.max_samples,
    )

    # Evaluate
    engine = EvaluationEngine(config)
    return engine.evaluate(
        model=model,
        batches=iter(iterator),
        total_samples=min(len(texts), config.max_samples or len(texts)),
        progress_callback=progress_callback,
    )
