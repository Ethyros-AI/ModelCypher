"""
CUDA Evaluation Engine for Model and LoRA Checkpoint Assessment.

This is the PyTorch/CUDA implementation. For other backends:
- MLX/macOS: see evaluation_mlx.py
- JAX/TPU: see evaluation_jax.py

Use _platform.get_evaluation_engine() for automatic platform selection.

Implementation based on PyTorch 2.x best practices (2025):
- torch.no_grad() context for inference efficiency
- torch.nn.functional.cross_entropy for loss computation
- torch.compile() optional for performance
- Proper device placement for GPU acceleration

References:
- https://pytorch.org/docs/stable/notes/autocast.html
- https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class EvaluationMetricCUDA(str, Enum):
    """Available evaluation metrics."""
    LOSS = "loss"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    BITS_PER_CHARACTER = "bpc"


@dataclass
class EvaluationConfigCUDA:
    """Configuration for evaluation runs."""
    metrics: list[EvaluationMetricCUDA] = field(
        default_factory=lambda: [EvaluationMetricCUDA.LOSS, EvaluationMetricCUDA.PERPLEXITY]
    )
    batch_size: int = 4
    sequence_length: int = 512
    max_samples: int | None = None

    @classmethod
    def default(cls) -> "EvaluationConfigCUDA":
        return cls()


@dataclass
class EvaluationProgressCUDA:
    """Progress update during evaluation."""
    samples_processed: int
    total_samples: int
    current_metric: float | None = None

    @property
    def percentage(self) -> float:
        return self.samples_processed / max(self.total_samples, 1)


@dataclass
class EvaluationResultCUDA:
    """Result of an evaluation run."""
    metrics: dict[EvaluationMetricCUDA, float]
    samples_evaluated: int
    tokens_evaluated: int
    duration_seconds: float

    @property
    def loss(self) -> float | None:
        return self.metrics.get(EvaluationMetricCUDA.LOSS)

    @property
    def perplexity(self) -> float | None:
        return self.metrics.get(EvaluationMetricCUDA.PERPLEXITY)

    @property
    def accuracy(self) -> float | None:
        return self.metrics.get(EvaluationMetricCUDA.ACCURACY)


@dataclass
class EvaluationBatchCUDA:
    """A single evaluation batch."""
    inputs: torch.Tensor      # [batch, seq_len] int64
    targets: torch.Tensor     # [batch, seq_len] int64
    mask: torch.Tensor        # [batch, seq_len] float32
    valid_token_counts: list[int]


class EvaluationErrorCUDA(Exception):
    """Evaluation failed."""
    pass


# =============================================================================
# Evaluation Engine
# =============================================================================

class EvaluationEngineCUDA:
    """
    CUDA Evaluation Engine (PyTorch backend).

    Computes loss, perplexity, and accuracy using GPU-optimized
    batch processing with automatic mixed precision support.

    Features (matching MLX parity):
    - Batched loss/perplexity computation
    - Next-token prediction accuracy
    - Progress callbacks
    - GPU-optimized cross-entropy calculation
    """

    def __init__(
        self,
        config: EvaluationConfigCUDA | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.config = config or EvaluationConfigCUDA.default()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def evaluate(
        self,
        model: nn.Module,
        batches: Iterator[EvaluationBatchCUDA],
        total_samples: int | None = None,
        progress_callback: Callable[[EvaluationProgressCUDA], None] | None = None,
    ) -> EvaluationResultCUDA:
        """
        Evaluate a model on a dataset.

        Args:
            model: PyTorch language model to evaluate
            batches: Iterator of evaluation batches
            total_samples: Total sample count for progress
            progress_callback: Callback for progress updates

        Returns:
            EvaluationResultCUDA with computed metrics
        """
        start_time = time.time()

        # Move model to device and set eval mode
        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_tokens = 0
        samples_processed = 0

        needs_loss = (
            EvaluationMetricCUDA.LOSS in self.config.metrics or
            EvaluationMetricCUDA.PERPLEXITY in self.config.metrics or
            EvaluationMetricCUDA.BITS_PER_CHARACTER in self.config.metrics
        )
        needs_accuracy = EvaluationMetricCUDA.ACCURACY in self.config.metrics

        with torch.no_grad():
            for batch in batches:
                # Move batch to device
                inputs = batch.inputs.to(self.device)
                targets = batch.targets.to(self.device)
                mask = batch.mask.to(self.device)

                # Forward pass
                logits = model(inputs)

                batch_tokens = sum(batch.valid_token_counts)
                total_tokens += batch_tokens

                if needs_loss:
                    batch_loss = self._compute_batch_loss(logits, targets, mask)
                    total_loss += batch_loss

                if needs_accuracy:
                    batch_correct = self._compute_batch_accuracy(logits, targets, mask)
                    total_correct += batch_correct

                samples_processed += inputs.shape[0]

                if progress_callback and total_samples:
                    progress_callback(EvaluationProgressCUDA(
                        samples_processed=samples_processed,
                        total_samples=total_samples,
                    ))

        if total_tokens == 0:
            raise EvaluationErrorCUDA("Evaluation produced zero tokens. Check dataset format.")

        # Compute final metrics
        metrics: dict[EvaluationMetricCUDA, float] = {}

        if needs_loss:
            avg_loss = total_loss / total_tokens
            if EvaluationMetricCUDA.LOSS in self.config.metrics:
                metrics[EvaluationMetricCUDA.LOSS] = avg_loss
            if EvaluationMetricCUDA.PERPLEXITY in self.config.metrics:
                metrics[EvaluationMetricCUDA.PERPLEXITY] = math.exp(avg_loss)
            if EvaluationMetricCUDA.BITS_PER_CHARACTER in self.config.metrics:
                metrics[EvaluationMetricCUDA.BITS_PER_CHARACTER] = avg_loss / math.log(2)

        if needs_accuracy:
            metrics[EvaluationMetricCUDA.ACCURACY] = total_correct / total_tokens

        duration = time.time() - start_time
        logger.info(
            "Evaluation completed: %d samples, %d tokens in %.2fs",
            samples_processed,
            total_tokens,
            duration,
        )

        return EvaluationResultCUDA(
            metrics=metrics,
            samples_evaluated=samples_processed,
            tokens_evaluated=total_tokens,
            duration_seconds=duration,
        )

    def _compute_batch_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        Compute sum of negative log-likelihood for a batch.

        Uses PyTorch cross-entropy for GPU efficiency.

        Args:
            logits: [batch, seq, vocab] model outputs
            targets: [batch, seq] target token IDs
            mask: [batch, seq] attention mask

        Returns:
            Sum of NLL values
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for cross-entropy: [batch*seq, vocab] and [batch*seq]
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Cross-entropy without reduction
        # Note: PyTorch cross_entropy expects [N, C] logits and [N] targets
        nll = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        # Apply mask
        nll = nll * mask_flat
        total_nll = float(nll.sum().item())

        return total_nll

    def _compute_batch_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
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
        # GPU argmax
        predictions = torch.argmax(logits, dim=-1)

        # Check equality
        correct = (predictions == targets).float() * mask

        total_correct = float(correct.sum().item())
        return total_correct


# =============================================================================
# Batch Iterator
# =============================================================================

class DatasetBatchIteratorCUDA:
    """
    CUDA-optimized dataset batch iterator.

    Iterates over a dataset producing padded evaluation batches
    as PyTorch tensors on the specified device.
    """

    def __init__(
        self,
        texts: list[str],
        tokenize_fn: Callable[[str], list[int]],
        sequence_length: int = 512,
        batch_size: int = 4,
        max_samples: int | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.texts = texts
        self.tokenize = tokenize_fn
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cursor = 0

    def __iter__(self) -> Iterator[EvaluationBatchCUDA]:
        self.cursor = 0
        return self

    def __next__(self) -> EvaluationBatchCUDA:
        if self.cursor >= len(self.texts):
            raise StopIteration

        if self.max_samples and self.cursor >= self.max_samples:
            raise StopIteration

        inputs_list: list[list[int]] = []
        targets_list: list[list[int]] = []
        mask_list: list[list[float]] = []
        valid_counts: list[int] = []

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

        return EvaluationBatchCUDA(
            inputs=torch.tensor(inputs_list, dtype=torch.long, device=self.device),
            targets=torch.tensor(targets_list, dtype=torch.long, device=self.device),
            mask=torch.tensor(mask_list, dtype=torch.float32, device=self.device),
            valid_token_counts=valid_counts,
        )

    def __len__(self) -> int:
        """Approximate number of batches."""
        effective_samples = min(len(self.texts), self.max_samples or len(self.texts))
        return (effective_samples + self.batch_size - 1) // self.batch_size


# =============================================================================
# LoRA Checkpoint Evaluation
# =============================================================================

def evaluate_lora_checkpoint_cuda(
    model: nn.Module,
    checkpoint_path: Path,
    texts: list[str],
    tokenize_fn: Callable[[str], list[int]],
    config: EvaluationConfigCUDA | None = None,
    progress_callback: Callable[[EvaluationProgressCUDA], None] | None = None,
    device: str = "cuda:0",
) -> EvaluationResultCUDA:
    """
    Evaluate a LoRA checkpoint against a dataset.

    Loads checkpoint weights into the model and evaluates.

    Args:
        model: Base PyTorch model with LoRA layers
        checkpoint_path: Path to LoRA checkpoint (.safetensors)
        texts: List of evaluation texts
        tokenize_fn: Tokenization function
        config: Evaluation configuration
        progress_callback: Optional progress callback
        device: CUDA device to use

    Returns:
        EvaluationResultCUDA with metrics
    """
    config = config or EvaluationConfigCUDA.default()

    # Load checkpoint weights using safetensors
    checkpoint_weights = load_file(str(checkpoint_path), device=device)

    # Apply weights to model (load matching keys)
    model_state = model.state_dict()
    matched_count = 0

    for key, value in checkpoint_weights.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                matched_count += 1
            else:
                logger.warning(
                    "Shape mismatch for %s: model %s vs checkpoint %s",
                    key,
                    model_state[key].shape,
                    value.shape,
                )

    if matched_count == 0:
        raise EvaluationErrorCUDA(
            f"No matching parameters in checkpoint: {checkpoint_path}"
        )

    model.load_state_dict(model_state, strict=False)
    logger.info("Loaded %d parameters from checkpoint", matched_count)

    # Create batch iterator
    iterator = DatasetBatchIteratorCUDA(
        texts=texts,
        tokenize_fn=tokenize_fn,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        max_samples=config.max_samples,
        device=device,
    )

    # Evaluate
    engine = EvaluationEngineCUDA(config, device=device)
    return engine.evaluate(
        model=model,
        batches=iter(iterator),
        total_samples=min(len(texts), config.max_samples or len(texts)),
        progress_callback=progress_callback,
    )


__all__ = [
    "EvaluationMetricCUDA",
    "EvaluationConfigCUDA",
    "EvaluationProgressCUDA",
    "EvaluationResultCUDA",
    "EvaluationBatchCUDA",
    "EvaluationEngineCUDA",
    "EvaluationErrorCUDA",
    "DatasetBatchIteratorCUDA",
    "evaluate_lora_checkpoint_cuda",
]
