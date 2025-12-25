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

"""Training dataset pipeline for JSONL files."""

import json
import logging
from pathlib import Path
from typing import Iterator

import mlx.core as mx

logger = logging.getLogger(__name__)


class TrainingDataset:
    """Iterable dataset for training from JSONL files."""

    def __init__(
        self,
        path: str,
        tokenizer: any,
        max_seq_length: int = 512,
        batch_size: int = 4,
    ):
        """Initialize training dataset.

        Parameters
        ----------
        path : str
            Path to JSONL dataset file.
        tokenizer : any
            Tokenizer for encoding text.
        max_seq_length : int
            Maximum sequence length for truncation.
        batch_size : int
            Batch size for training.
        """
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self._samples = self._load_samples()
        logger.info("Dataset: Loaded %d samples from %s", len(self._samples), self.path)

    def _load_samples(self) -> list[str]:
        samples = []
        if not self.path.exists():
            logger.warning("Dataset file not found: %s", self.path)
            return []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "text" in obj:
                        samples.append(obj["text"])
                    elif "messages" in obj:
                        # Convert chat format to text using tokenizer's template
                        if hasattr(self.tokenizer, "apply_chat_template"):
                            text = self.tokenizer.apply_chat_template(
                                obj["messages"], tokenize=False, add_generation_prompt=False
                            )
                            samples.append(text)
                        else:
                            # Fallback: join message contents
                            text = "\n".join(msg.get("content", "") for msg in obj["messages"])
                            samples.append(text)
                except json.JSONDecodeError:
                    continue
        return samples

    def __len__(self) -> int:
        return len(self._samples) // self.batch_size

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array]]:
        """Iterate over batched and tokenized training data.

        Yields
        ------
        tuple of (mx.array, mx.array)
            Input IDs and labels for each batch.
        """
        # Shuffle samples at start of each iteration
        # Shuffle indices using MLX random
        n = len(self._samples)
        indices = mx.random.permutation(mx.arange(n))

        # Get pad token id - try multiple common attribute names
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        if pad_id is None:
            pad_id = 0

        for i in range(0, len(self._samples), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_texts = [self._samples[idx] for idx in batch_indices]

            # Tokenize each text individually using encode()
            encoded_batch = []
            for text in batch_texts:
                tokens = self.tokenizer.encode(text)
                # Truncate if needed
                if len(tokens) > self.max_seq_length:
                    tokens = tokens[: self.max_seq_length]
                encoded_batch.append(tokens)

            # Pad to same length
            max_len = min(max(len(t) for t in encoded_batch), self.max_seq_length)
            padded_batch = []
            for tokens in encoded_batch:
                if len(tokens) < max_len:
                    tokens = tokens + [pad_id] * (max_len - len(tokens))
                padded_batch.append(tokens)

            input_ids = mx.array(padded_batch, dtype=mx.int32)

            # For causal language modeling, labels are input_ids shifted by 1
            # We pad with -100 (or a mask) if needed, but for simple MLX training
            # we can just shift.
            # Most MLX training examples use labels = input_ids[:, 1:]
            # and inputs = input_ids[:, :-1]
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]

            yield inputs, labels
