from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from modelcypher.core.domain.agents.computational_gate_atlas import ComputationalGateInventory
from modelcypher.core.domain.agents.semantic_prime_frames import SemanticPrimeFrames
from modelcypher.core.domain.agents.semantic_prime_multilingual import (
    SemanticPrimeMultilingualInventoryLoader,
)
from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorExtractionConfig:
    use_enriched_primes: bool = True
    include_computational_gates: bool = True
    max_polyglot_texts_per_language: int = 2


class AnchorExtractorError(RuntimeError):
    pass


class AnchorExtractor:
    def extract(
        self,
        model_path: str,
        weights: dict[str, Any],
        config: AnchorExtractionConfig | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        cfg = config or AnchorExtractionConfig()
        tokenizer = self._load_tokenizer(model_path)
        embedding_key, embedding = self.token_embedding_matrix(weights)
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim != 2:
            raise AnchorExtractorError(
                f"Token embedding weight must be 2D. {embedding_key} shape={embedding.shape}"
            )

        if embedding.shape[0] < embedding.shape[1]:
            embedding = embedding.T

        vocab = embedding.shape[0]
        anchors: dict[str, np.ndarray] = {}
        confidence: dict[str, float] = {}

        if cfg.use_enriched_primes:
            anchors.update(self._enriched_prime_anchors(tokenizer, embedding, vocab, confidence, cfg))
        else:
            anchors.update(self._basic_prime_anchors(tokenizer, embedding, vocab, confidence))

        if cfg.include_computational_gates:
            anchors.update(self._computational_gate_anchors(tokenizer, embedding, vocab, confidence))

        if not anchors:
            raise AnchorExtractorError(
                f"Unable to derive anchors from token embeddings ({embedding_key})."
            )

        return anchors, confidence

    @staticmethod
    def token_embedding_matrix(weights: dict[str, Any]) -> tuple[str, np.ndarray]:
        preferred_suffixes = [
            "embed_tokens.weight",
            "tok_embeddings.weight",
            "token_embedding.weight",
            "wte.weight",
            "lm_head.weight",
        ]
        for suffix in preferred_suffixes:
            for key, value in weights.items():
                if key.endswith(suffix):
                    arr = np.asarray(value)
                    if arr.dtype in (np.float16, np.float32, np.float64):
                        return key, arr

        scored: list[tuple[str, np.ndarray, int]] = []
        for key, value in weights.items():
            arr = np.asarray(value)
            if arr.ndim != 2:
                continue
            if arr.dtype not in (np.float16, np.float32, np.float64):
                continue
            max_dim = max(arr.shape[0], arr.shape[1])
            min_dim = min(arr.shape[0], arr.shape[1])
            if max_dim < 8192 or min_dim < 256 or min_dim > 16384:
                continue
            score = 0
            lower = key.lower()
            if "embed" in lower:
                score += 100
            if "tok" in lower:
                score += 80
            if "wte" in lower:
                score += 80
            if "lm_head" in lower:
                score += 50
            if max_dim >= 32000:
                score += 20
            if max_dim >= 100000:
                score += 10
            score += min(30, max_dim // 4000)
            scored.append((key, arr, score))

        if not scored:
            raise AnchorExtractorError("Unable to locate token embedding weights in the model parameters.")

        scored.sort(key=lambda item: (item[2], item[1].size))
        key, arr, _ = scored[-1]
        return key, arr

    @staticmethod
    def normalize_anchor_matrix(matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2:
            return matrix
        mean = matrix.mean(axis=0, keepdims=True)
        centered = matrix - mean
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return centered / norms

    @staticmethod
    def _load_tokenizer(model_path: str) -> Tokenizer:
        path = Path(model_path).expanduser().resolve()
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.exists():
            raise AnchorExtractorError(f"Tokenizer not found at: {path}")
        return Tokenizer.from_file(str(path))

    @staticmethod
    def _confidence_for_token_count(count: int) -> float:
        if count <= 0:
            return 0.0
        if count == 1:
            return 1.0
        if count == 2:
            return 0.85
        if count == 3:
            return 0.70
        return max(0.3, 1.0 - 0.15 * float(count))

    def _enriched_prime_anchors(
        self,
        tokenizer: Tokenizer,
        embedding: np.ndarray,
        vocab: int,
        confidence: dict[str, float],
        cfg: AnchorExtractionConfig,
    ) -> dict[str, np.ndarray]:
        primes = SemanticPrimeFrames.enriched()
        polyglot = SemanticPrimeMultilingualInventoryLoader.global_diverse()
        polyglot_by_id: dict[str, list[str]] = {}
        for prime in polyglot.primes:
            texts: list[str] = []
            for bucket in prime.languages:
                texts.extend(bucket.texts[: cfg.max_polyglot_texts_per_language])
            polyglot_by_id[prime.id] = texts

        anchors: dict[str, np.ndarray] = {}
        for prime in primes:
            core_text = f" {prime.word}"
            core_ids = tokenizer.encode(core_text, add_special_tokens=False).ids
            core_valid = [token_id for token_id in core_ids if 0 <= token_id < vocab]
            if len(core_valid) > 3:
                logger.warning(
                    "Skipping prime '%s' (core word fragmented into %s tokens)",
                    prime.id,
                    len(core_valid),
                )
                continue
            if not core_valid:
                continue

            texts: list[str] = [core_text]
            texts.extend([f" {text}" for text in prime.frames])
            if prime.contrast:
                texts.append(f" {prime.contrast}")
            texts.extend([f" {text}" for text in prime.exemplars])
            for text in polyglot_by_id.get(prime.id, []):
                texts.append(f" {text}")

            seen: set[str] = set()
            unique = [text for text in texts if not (text in seen or seen.add(text))]

            vectors: list[np.ndarray] = []
            for text in unique:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                valid = [token_id for token_id in ids if 0 <= token_id < vocab]
                if not valid:
                    continue
                vectors.append(embedding[valid].mean(axis=0))

            if not vectors:
                continue

            anchor_id = f"prime:{prime.id}"
            anchors[anchor_id] = np.mean(np.stack(vectors, axis=0), axis=0)
            confidence[anchor_id] = self._confidence_for_token_count(len(core_valid))

        return anchors

    def _basic_prime_anchors(
        self,
        tokenizer: Tokenizer,
        embedding: np.ndarray,
        vocab: int,
        confidence: dict[str, float],
    ) -> dict[str, np.ndarray]:
        primes = SemanticPrimeInventory.english2014()
        anchors: dict[str, np.ndarray] = {}

        for prime in primes:
            text = f" {prime.canonical_english}"
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            valid = [token_id for token_id in ids if 0 <= token_id < vocab]
            if len(valid) > 3:
                logger.warning(
                    "Skipping prime '%s' (fragmented into %s tokens)",
                    prime.id,
                    len(valid),
                )
                continue
            if not valid:
                continue

            anchor_id = f"prime:{prime.id}"
            anchors[anchor_id] = embedding[valid].mean(axis=0)
            confidence[anchor_id] = self._confidence_for_token_count(len(valid))

        return anchors

    def _computational_gate_anchors(
        self,
        tokenizer: Tokenizer,
        embedding: np.ndarray,
        vocab: int,
        confidence: dict[str, float],
    ) -> dict[str, np.ndarray]:
        gates = ComputationalGateInventory.probe_gates()
        anchors: dict[str, np.ndarray] = {}

        for gate in gates:
            texts: list[str] = []
            gate_name = gate.name.lower().replace("_", " ")
            texts.append(f"{gate_name}: {gate.description}")
            texts.append(gate_name)
            examples = [example.strip() for example in gate.examples + gate.polyglot_examples]
            examples = [example for example in examples if example]
            if examples:
                texts.extend(examples)

            seen: set[str] = set()
            unique = [text for text in texts if not (text in seen or seen.add(text))]

            vectors: list[np.ndarray] = []
            token_counts: list[int] = []
            for text in unique:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                valid = [token_id for token_id in ids if 0 <= token_id < vocab]
                if not valid:
                    continue
                vectors.append(embedding[valid].mean(axis=0))
                token_counts.append(len(valid))

            if not vectors:
                continue

            anchor_id = f"gate:{gate.id}"
            anchors[anchor_id] = np.mean(np.stack(vectors, axis=0), axis=0)

            avg_tokens = float(sum(token_counts)) / float(max(1, len(token_counts)))
            if avg_tokens <= 5:
                gate_confidence = 1.0
            elif avg_tokens <= 15:
                gate_confidence = 0.85
            elif avg_tokens <= 30:
                gate_confidence = 0.7
            else:
                gate_confidence = max(0.5, 1.0 - 0.02 * avg_tokens)

            confidence[anchor_id] = gate_confidence

        return anchors
