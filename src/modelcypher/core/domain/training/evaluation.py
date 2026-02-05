# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Evaluation Engine - ONE implementation using Backend protocol.

Computes loss, perplexity, and accuracy for models and LoRA checkpoints.
All tensor operations go through Backend - no framework imports here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from modelcypher.core.domain.geometry.numerical_stability import (
    exp_scalar,
    log_scalar,
    safe_log_epsilon,
    precision_dtype,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class EvaluationMetric(str, Enum):
    """Metrics that can be computed during evaluation."""

    LOSS = "loss"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    BITS_PER_CHARACTER = "bpc"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    metrics: list[EvaluationMetric] = field(
        default_factory=lambda: [EvaluationMetric.LOSS, EvaluationMetric.PERPLEXITY]
    )
    batch_size: int = 4
    sequence_length: int = 512
    max_samples: int | None = None

    @classmethod
    def default(cls) -> "EvaluationConfig":
        return cls()


@dataclass
class EvaluationProgress:
    """Progress update during evaluation."""

    samples_processed: int
    total_samples: int
    current_metric: float | None = None

    @property
    def percentage(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return self.samples_processed / self.total_samples


@dataclass
class TrainingEvaluationResult:
    """Result of an evaluation run."""

    metrics: dict[EvaluationMetric, float]
    samples_evaluated: int
    tokens_evaluated: int
    duration_seconds: float

    @property
    def loss(self) -> float | None:
        return self.metrics.get(EvaluationMetric.LOSS)

    @property
    def perplexity(self) -> float | None:
        return self.metrics.get(EvaluationMetric.PERPLEXITY)

    @property
    def accuracy(self) -> float | None:
        return self.metrics.get(EvaluationMetric.ACCURACY)


@dataclass
class EvaluationBatch:
    """A single evaluation batch."""

    inputs: Any  # Backend Array [batch, seq_len]
    targets: Any  # Backend Array [batch, seq_len]
    mask: Any  # Backend Array [batch, seq_len]
    valid_token_counts: list[int]


class EvaluationError(Exception):
    """Evaluation failed."""


class EvaluationEngine:
    """Evaluates models using Backend protocol.

    All tensor operations go through self._backend.
    No framework imports in this file.
    """

    def __init__(
        self,
        backend: "Backend",
        config: EvaluationConfig | None = None,
    ):
        self._backend = backend
        self.config = config or EvaluationConfig.default()

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        dataset_path: Path,
        progress_callback: Callable[[EvaluationProgress], None] | None = None,
    ) -> TrainingEvaluationResult:
        """Evaluate model on dataset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            dataset_path: Path to evaluation dataset (JSONL)
            progress_callback: Optional progress updates

        Returns:
            TrainingEvaluationResult with computed metrics
        """
        b = self._backend
        start_time = time.time()

        # Load dataset
        samples = self._load_dataset(dataset_path)
        if self.config.max_samples:
            samples = samples[: self.config.max_samples]

        if not samples:
            raise EvaluationError(f"No samples found in {dataset_path}")

        total_samples = len(samples)
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        samples_processed = 0

        # Process in batches
        for batch in self._create_batches(samples, tokenizer):
            # Forward pass
            logits = self._forward(model, batch.inputs)

            # Compute loss
            batch_loss, batch_tokens = self._compute_loss(
                logits, batch.targets, batch.mask
            )
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens

            # Compute accuracy if requested
            if EvaluationMetric.ACCURACY in self.config.metrics:
                correct = self._compute_accuracy(logits, batch.targets, batch.mask)
                correct_predictions += correct

            samples_processed += len(batch.valid_token_counts)

            if progress_callback:
                avg_loss = total_loss / max(total_tokens, 1)
                progress_callback(
                    EvaluationProgress(
                        samples_processed=samples_processed,
                        total_samples=total_samples,
                        current_metric=avg_loss,
                    )
                )

        # Compute final metrics
        duration = time.time() - start_time
        avg_loss = total_loss / max(total_tokens, 1)

        metrics: dict[EvaluationMetric, float] = {}

        if EvaluationMetric.LOSS in self.config.metrics:
            metrics[EvaluationMetric.LOSS] = avg_loss

        if EvaluationMetric.PERPLEXITY in self.config.metrics:
            metrics[EvaluationMetric.PERPLEXITY] = exp_scalar(avg_loss, b)

        if EvaluationMetric.ACCURACY in self.config.metrics:
            metrics[EvaluationMetric.ACCURACY] = correct_predictions / max(
                total_tokens, 1
            )

        if EvaluationMetric.BITS_PER_CHARACTER in self.config.metrics:
            ln2 = log_scalar(2.0, b)
            metrics[EvaluationMetric.BITS_PER_CHARACTER] = avg_loss / ln2

        return TrainingEvaluationResult(
            metrics=metrics,
            samples_evaluated=samples_processed,
            tokens_evaluated=total_tokens,
            duration_seconds=duration,
        )

    def _forward(self, model: Any, inputs: "Array") -> "Array":
        """Run forward pass through model."""
        logits = model(inputs)
        # Force evaluation on lazy backends
        self._backend.eval(logits)
        return logits

    def _compute_loss(
        self,
        logits: "Array",
        targets: "Array",
        mask: "Array",
    ) -> tuple[float, int]:
        """Compute cross-entropy loss.

        Returns:
            Tuple of (average_loss, token_count)
        """
        b = self._backend

        # logits: [batch, seq, vocab]
        # targets: [batch, seq]
        # mask: [batch, seq]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_targets = targets[:, 1:]
        shift_mask = mask[:, 1:]

        # Flatten
        batch_size = int(shift_logits.shape[0])
        seq_len = int(shift_logits.shape[1])
        vocab_size = int(shift_logits.shape[2])

        flat_logits = b.reshape(shift_logits, (batch_size * seq_len, vocab_size))
        flat_targets = b.reshape(shift_targets, (batch_size * seq_len,))
        flat_mask = b.reshape(shift_mask, (batch_size * seq_len,))

        # Cross-entropy loss
        log_probs = b.log_softmax(flat_logits, axis=-1)

        # Gather target log probs
        target_log_probs = self._gather_along_axis(log_probs, flat_targets, axis=1)

        # Apply mask and sum
        masked_loss = -target_log_probs * flat_mask
        total_loss = float(b.to_scalar(b.sum(masked_loss)))
        token_count = int(b.to_scalar(b.sum(flat_mask)))

        avg_loss = total_loss / max(token_count, 1)
        return avg_loss, token_count

    def _gather_along_axis(
        self, arr: "Array", indices: "Array", axis: int
    ) -> "Array":
        """Gather values along axis (like take_along_axis)."""
        b = self._backend
        # For 2D array with axis=1, indices select columns per row
        n_rows = int(arr.shape[0])
        row_indices = b.arange(n_rows)
        return arr[row_indices, indices]

    def _compute_accuracy(
        self,
        logits: "Array",
        targets: "Array",
        mask: "Array",
    ) -> int:
        """Compute number of correct next-token predictions."""
        b = self._backend

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_targets = targets[:, 1:]
        shift_mask = mask[:, 1:]

        # Get predictions
        predictions = b.argmax(shift_logits, axis=-1)

        # Count correct (masked)
        correct = (predictions == shift_targets) * shift_mask
        return int(b.to_scalar(b.sum(correct)))

    def _load_dataset(self, path: Path) -> list[str]:
        """Load text samples from JSONL dataset."""
        import json

        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text") or data.get("content") or ""
                    if text:
                        samples.append(text)
                except json.JSONDecodeError:
                    continue
        return samples

    def _create_batches(
        self,
        samples: list[str],
        tokenizer: Any,
    ) -> Iterator[EvaluationBatch]:
        """Create batches from samples."""
        b = self._backend
        batch_size = self.config.batch_size
        seq_len = self.config.sequence_length

        current_batch: list[list[int]] = []
        valid_counts: list[int] = []

        for sample in samples:
            # Tokenize
            tokens = b.encode_tokens(tokenizer, sample)
            if len(tokens) < 2:
                continue

            # Truncate or use as-is
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]

            current_batch.append(tokens)
            valid_counts.append(len(tokens))

            if len(current_batch) >= batch_size:
                yield self._finalize_batch(current_batch, valid_counts, seq_len)
                current_batch = []
                valid_counts = []

        # Final partial batch
        if current_batch:
            yield self._finalize_batch(current_batch, valid_counts, seq_len)

    def _finalize_batch(
        self,
        token_lists: list[list[int]],
        valid_counts: list[int],
        seq_len: int,
    ) -> EvaluationBatch:
        """Pad and convert batch to arrays."""
        b = self._backend

        batch_size = len(token_lists)
        max_len = max(len(t) for t in token_lists)
        padded_len = min(max_len, seq_len)

        # Pad sequences
        padded = []
        masks = []
        for tokens in token_lists:
            pad_len = padded_len - len(tokens)
            if pad_len > 0:
                tokens = tokens + [0] * pad_len
            else:
                tokens = tokens[:padded_len]
            padded.append(tokens)

            # Mask: 1 for real tokens, 0 for padding
            mask = [1.0] * min(len(token_lists[0]), padded_len)
            mask.extend([0.0] * max(0, padded_len - len(mask)))
            masks.append(mask[:padded_len])

        inputs = b.array(padded, dtype="int32")
        targets = b.array(padded, dtype="int32")  # Same for LM
        mask = b.array(masks, dtype="float32")

        return EvaluationBatch(
            inputs=inputs,
            targets=targets,
            mask=mask,
            valid_token_counts=valid_counts,
        )


__all__ = [
    "EvaluationMetric",
    "EvaluationConfig",
    "EvaluationProgress",
    "TrainingEvaluationResult",
    "EvaluationBatch",
    "EvaluationError",
    "EvaluationEngine",
]
