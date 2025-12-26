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

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)


class TrainingDataset(Iterable[tuple[object, object]]):
    """Lightweight dataset wrapper for local training adapters.

    Produces batches of (input_ids, target_ids). If a batch tokenizer is
    available, outputs MLX arrays for direct consumption by MLX training.
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: object,
        batch_size: int = 4,
        sequence_length: int = 1024,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset not found: {self.dataset_path}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if sequence_length <= 1:
            raise ValueError("sequence_length must be > 1")

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.seed = seed
        self._samples: list[str] = self._load_text_samples()
        self._use_batch_tokenizer = callable(self.tokenizer)
        self._sequences: list[list[int]] = []
        if not self._use_batch_tokenizer:
            self._sequences = self._encode_sequences(self._samples)

    def __len__(self) -> int:
        if self._use_batch_tokenizer:
            if not self._samples:
                return 0
            return (len(self._samples) + self.batch_size - 1) // self.batch_size
        if not self._sequences:
            return 0
        return (len(self._sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[object, object]]:
        if self._use_batch_tokenizer:
            return self._iter_batch_tokens()
        return self._iter_preencoded()

    def _iter_batch_tokens(self) -> Iterator[tuple[object, object]]:
        samples = list(self._samples)
        if self.shuffle and len(samples) > 1:
            import random

            random.Random(self.seed).shuffle(samples)

        for idx in range(0, len(samples), self.batch_size):
            batch_samples = samples[idx : idx + self.batch_size]
            input_ids = self._tokenize_batch(batch_samples)
            if input_ids is None:
                continue
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            yield inputs, labels

    def _iter_preencoded(self) -> Iterator[tuple[object, object]]:
        sequences = list(self._sequences)
        if self.shuffle and len(sequences) > 1:
            import random

            random.Random(self.seed).shuffle(sequences)

        for idx in range(0, len(sequences), self.batch_size):
            batch = sequences[idx : idx + self.batch_size]
            inputs = [seq[:-1] for seq in batch]
            targets = [seq[1:] for seq in batch]
            yield inputs, targets

    def _tokenize_batch(self, texts: list[str]):
        kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.sequence_length,
        }
        try:
            payload = self.tokenizer(texts, **kwargs)
        except TypeError:
            payload = self.tokenizer(texts)

        input_ids = payload.get("input_ids") if isinstance(payload, dict) else payload
        if input_ids is None:
            return None

        try:
            import mlx.core as mx

            return mx.array(input_ids)
        except Exception:
            return input_ids

    def _encode_sequences(self, samples: list[str]) -> list[list[int]]:
        sequences: list[list[int]] = []
        max_len = self.sequence_length + 1
        for text in samples:
            token_ids = self._encode_text(text)
            if len(token_ids) < 2:
                continue
            for start in range(0, len(token_ids) - 1, self.sequence_length):
                chunk = token_ids[start : start + max_len]
                if len(chunk) < 2:
                    continue
                if len(chunk) < max_len:
                    pad_token = self._pad_token_id()
                    if pad_token is not None:
                        chunk = chunk + [pad_token] * (max_len - len(chunk))
                sequences.append(chunk)
        return sequences

    def _encode_text(self, text: str) -> list[int]:
        if not hasattr(self.tokenizer, "encode"):
            return []
        try:
            encoded = self.tokenizer.encode(text)
            if isinstance(encoded, list):
                return encoded
            if hasattr(encoded, "ids"):
                return list(encoded.ids)
        except Exception as exc:
            logger.warning("Tokenizer encode failed: %s", exc)
        return []

    def _pad_token_id(self) -> int | None:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token = getattr(self.tokenizer, "pad_token", None)
            if pad_token is not None and hasattr(self.tokenizer, "encode"):
                encoded = self.tokenizer.encode(pad_token)
                if isinstance(encoded, list) and encoded:
                    return int(encoded[0])
                if hasattr(encoded, "ids") and encoded.ids:
                    return int(encoded.ids[0])
        return pad_token_id

    def _load_text_samples(self) -> list[str]:
        samples: list[str] = []
        try:
            with self.dataset_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    text = self._extract_text(line)
                    if text:
                        samples.append(text)
        except Exception as exc:
            logger.warning("Failed to read dataset %s: %s", self.dataset_path, exc)
        return samples

    def _extract_text(self, line: str) -> str | None:
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    if "text" in payload and isinstance(payload["text"], str):
                        return payload["text"].strip()
                    messages = payload.get("messages")
                    if isinstance(messages, list):
                        if hasattr(self.tokenizer, "apply_chat_template"):
                            return self.tokenizer.apply_chat_template(messages)
                        return " ".join(
                            msg.get("content", "")
                            for msg in messages
                            if isinstance(msg, dict)
                        ).strip()
                    for key in ("prompt", "completion", "content"):
                        value = payload.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
                    return None
            except json.JSONDecodeError:
                return line.strip() if line.strip() else None
        return line.strip() if line.strip() else None
