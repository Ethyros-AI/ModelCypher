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

"""Vocabulary analysis utilities for embedding matrices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class TokenizerType(str, Enum):
    BPE = "bpe"
    SENTENCEPIECE = "sentencepiece"
    WORDPIECE = "wordpiece"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VocabularyStats:
    vocab_size: int
    hidden_dim: int
    embedding_mean_norm: float
    embedding_std: float
    tokenizer_type: TokenizerType
    special_token_count: int
    has_tie_weights: bool
    mean_token_length: float | None = None
    bpe_merge_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "embedding_mean_norm": self.embedding_mean_norm,
            "embedding_std": self.embedding_std,
            "tokenizer_type": self.tokenizer_type.value,
            "special_token_count": self.special_token_count,
            "has_tie_weights": self.has_tie_weights,
        }
        if self.mean_token_length is not None:
            payload["mean_token_length"] = self.mean_token_length
        if self.bpe_merge_count is not None:
            payload["bpe_merge_count"] = self.bpe_merge_count
        return payload


@dataclass(frozen=True)
class VocabularyAlignment:
    alignment_score: float
    vocab_overlap_ratio: float | None
    dimension_ratio: float
    requires_projection: bool
    requires_vocab_mapping: bool
    shared_token_count: int
    source_only_tokens: int
    target_only_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "alignment_score": self.alignment_score,
            "vocab_overlap_ratio": self.vocab_overlap_ratio,
            "dimension_ratio": self.dimension_ratio,
            "requires_projection": self.requires_projection,
            "requires_vocab_mapping": self.requires_vocab_mapping,
            "shared_token_count": self.shared_token_count,
            "source_only_tokens": self.source_only_tokens,
            "target_only_tokens": self.target_only_tokens,
        }


class VocabularyAnalyzer:
    """Analyze vocabulary embedding geometry and overlap."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze_embeddings(
        self,
        embeddings: "Array",
        tokenizer_config: dict[str, object] | None = None,
    ) -> VocabularyStats:
        backend = self._backend
        array = backend.array(embeddings)
        backend.eval(array)

        if array.ndim != 2:
            raise ValueError("Expected 2D embedding matrix")

        norms = backend.norm(array, axis=1)
        mean_norm = backend.mean(norms)
        std_norm = backend.std(norms)
        backend.eval(mean_norm, std_norm)

        tokenizer_type = self._detect_tokenizer_type(tokenizer_config or {})
        special_token_count = self._count_special_tokens(tokenizer_config or {})

        has_tie_weights = bool((tokenizer_config or {}).get("tie_word_embeddings", False))

        mean_token_length = None
        if tokenizer_config and isinstance(tokenizer_config.get("vocab"), dict):
            vocab = tokenizer_config["vocab"]
            lengths = [len(token) for token in vocab]
            if lengths:
                mean_token_length = sum(lengths) / float(len(lengths))

        bpe_merge_count = None
        if tokenizer_config:
            merges = tokenizer_config.get("merges")
            if isinstance(merges, list):
                bpe_merge_count = len(merges)

        return VocabularyStats(
            vocab_size=int(array.shape[0]),
            hidden_dim=int(array.shape[1]),
            embedding_mean_norm=float(backend.to_numpy(mean_norm).item()),
            embedding_std=float(backend.to_numpy(std_norm).item()),
            tokenizer_type=tokenizer_type,
            special_token_count=special_token_count,
            has_tie_weights=has_tie_weights,
            mean_token_length=mean_token_length,
            bpe_merge_count=bpe_merge_count,
        )

    def analyze_alignment(
        self,
        source: VocabularyStats,
        target: VocabularyStats,
        source_vocab: dict[str, int] | None = None,
        target_vocab: dict[str, int] | None = None,
    ) -> VocabularyAlignment:
        dimension_ratio = (
            source.hidden_dim / float(target.hidden_dim)
            if target.hidden_dim
            else 0.0
        )
        requires_projection = source.hidden_dim != target.hidden_dim

        shared_token_count = 0
        source_only = source.vocab_size
        target_only = target.vocab_size
        overlap_ratio: float | None = None
        requires_vocab_mapping = True

        if source_vocab is not None and target_vocab is not None:
            shared, source_only_set, target_only_set = self.compute_token_overlap(
                source_vocab, target_vocab
            )
            shared_token_count = len(shared)
            source_only = len(source_only_set)
            target_only = len(target_only_set)
            total_unique = shared_token_count + source_only + target_only
            overlap_ratio = (
                shared_token_count / float(total_unique) if total_unique > 0 else 0.0
            )
            requires_vocab_mapping = total_unique != shared_token_count

        if overlap_ratio is not None:
            alignment_score = overlap_ratio
        else:
            denom = max(source.hidden_dim, target.hidden_dim)
            alignment_score = (
                min(source.hidden_dim, target.hidden_dim) / float(denom)
                if denom
                else 0.0
            )

        return VocabularyAlignment(
            alignment_score=float(alignment_score),
            vocab_overlap_ratio=overlap_ratio,
            dimension_ratio=float(dimension_ratio),
            requires_projection=requires_projection,
            requires_vocab_mapping=requires_vocab_mapping,
            shared_token_count=shared_token_count,
            source_only_tokens=source_only,
            target_only_tokens=target_only,
        )

    def compute_token_overlap(
        self,
        source_vocab: dict[str, int],
        target_vocab: dict[str, int],
    ) -> tuple[set[str], set[str], set[str]]:
        source_tokens = set(source_vocab.keys())
        target_tokens = set(target_vocab.keys())
        shared = source_tokens.intersection(target_tokens)
        source_only = source_tokens.difference(target_tokens)
        target_only = target_tokens.difference(source_tokens)
        return shared, source_only, target_only

    def _detect_tokenizer_type(self, config: dict[str, object]) -> TokenizerType:
        tokenizer_class = str(config.get("tokenizer_class", "")).lower()
        model_type = str(config.get("model_type", "")).lower()

        model_config = config.get("model")
        if isinstance(model_config, dict):
            model_type = str(model_config.get("type", model_type)).lower()

        if "gpt" in tokenizer_class or "bpe" in model_type:
            return TokenizerType.BPE
        if model_type in {"llama", "mistral", "gemma"}:
            return TokenizerType.SENTENCEPIECE
        if model_type in {"bert", "roberta", "wordpiece"}:
            return TokenizerType.WORDPIECE
        return TokenizerType.UNKNOWN

    def _count_special_tokens(self, config: dict[str, object]) -> int:
        special_keys = [
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "cls_token",
            "sep_token",
            "mask_token",
        ]
        count = 0
        for key in special_keys:
            if config.get(key):
                count += 1

        added_tokens = config.get("added_tokens")
        if isinstance(added_tokens, list):
            for token in added_tokens:
                if isinstance(token, dict) and token.get("special") is True:
                    count += 1
        return count
