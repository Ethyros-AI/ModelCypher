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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.ports.backend import Array
from modelcypher.core.domain.agents.computational_gate_atlas import ComputationalGateInventory
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean
from modelcypher.core.domain.agents.semantic_prime_frames import SemanticPrimeFrames
from modelcypher.core.domain.agents.semantic_prime_multilingual import (
    SemanticPrimeMultilingualInventoryLoader,
)
from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.use_cases.quantization_utils import (
    QuantizationConfig,
    dequantize_if_needed,
    quantization_hint_for_key,
)
from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorExtractionConfig:
    use_enriched_primes: bool = True
    include_computational_gates: bool = True
    max_polyglot_texts_per_language: int = 2
    use_unified_atlas: bool = True  # Default: use all probes from UnifiedAtlasInventory


class AnchorExtractorError(RuntimeError):
    pass


class AnchorExtractor:
    def _frechet_mean_of_embeddings(
        self,
        vectors: list[np.ndarray],
        backend: Backend | None = None,
    ) -> np.ndarray:
        """Compute Fréchet mean of embedding vectors.

        Embeddings live in curved representation space. Fréchet (Karcher)
        mean provides the proper geometric center.

        Parameters
        ----------
        vectors : list[np.ndarray]
            List of embedding vectors (numpy arrays)
        backend : Backend | None
            Backend for computation (uses default if None)

        Returns
        -------
        np.ndarray
            Fréchet mean as numpy array
        """
        if not vectors:
            raise ValueError("Cannot compute Fréchet mean of empty vector list")

        if len(vectors) == 1:
            return vectors[0]

        b = backend or get_default_backend()
        stacked = np.stack(vectors, axis=0)
        points = b.array(stacked)
        mean = frechet_mean(points, backend=b)
        b.eval(mean)
        return b.to_numpy(mean)

    def extract(
        self,
        model_path: str,
        weights: dict[str, Any],
        config: AnchorExtractionConfig | None = None,
        quantization: QuantizationConfig | None = None,
        backend: Backend | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        cfg = config or AnchorExtractionConfig()
        tokenizer = self._load_tokenizer(model_path)
        embedding_key, embedding = self.token_embedding_matrix(weights)
        if backend is not None:
            hint = quantization_hint_for_key(embedding_key, quantization)
            embedding = dequantize_if_needed(embedding, embedding_key, weights, backend, hint=hint)
        else:
            embedding = np.asarray(embedding)
            if embedding.dtype not in (np.float16, np.float32, np.float64):
                raise AnchorExtractorError(
                    f"Token embedding weight is quantized; provide backend to dequantize ({embedding_key})."
                )
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

        if cfg.use_unified_atlas:
            # Use all 321 probes from the unified atlas (supersedes individual atlas calls)
            anchors.update(self._unified_atlas_anchors(tokenizer, embedding, vocab, confidence))
        else:
            # Legacy mode: use subset of atlas sources
            if cfg.use_enriched_primes:
                anchors.update(
                    self._enriched_prime_anchors(tokenizer, embedding, vocab, confidence, cfg)
                )
            else:
                anchors.update(self._basic_prime_anchors(tokenizer, embedding, vocab, confidence))

            if cfg.include_computational_gates:
                anchors.update(
                    self._computational_gate_anchors(tokenizer, embedding, vocab, confidence)
                )

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
                    if arr.dtype.kind in {"f", "i", "u"}:
                        return key, arr

        scored: list[tuple[str, np.ndarray, int]] = []
        for key, value in weights.items():
            arr = np.asarray(value)
            if arr.ndim != 2:
                continue
            if arr.dtype.kind not in {"f", "i", "u"}:
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
            raise AnchorExtractorError(
                "Unable to locate token embedding weights in the model parameters."
            )

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
                # Use Fréchet mean for token embeddings (curvature is inherent)
                token_vecs = [embedding[tid] for tid in valid]
                vectors.append(self._frechet_mean_of_embeddings(token_vecs))

            if not vectors:
                continue

            anchor_id = f"prime:{prime.id}"
            # Fréchet mean of phrase embeddings (curvature is inherent in HD space)
            anchors[anchor_id] = self._frechet_mean_of_embeddings(vectors)
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
            # Fréchet mean of token embeddings (curvature is inherent)
            token_vecs = [embedding[tid] for tid in valid]
            anchors[anchor_id] = self._frechet_mean_of_embeddings(token_vecs)
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
                # Fréchet mean for token embeddings (curvature is inherent)
                token_vecs = [embedding[tid] for tid in valid]
                vectors.append(self._frechet_mean_of_embeddings(token_vecs))
                token_counts.append(len(valid))

            if not vectors:
                continue

            anchor_id = f"gate:{gate.id}"
            # Fréchet mean of phrase embeddings (curvature is inherent in HD space)
            anchors[anchor_id] = self._frechet_mean_of_embeddings(vectors)

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

    def _unified_atlas_anchors(
        self,
        tokenizer: Tokenizer,
        embedding: np.ndarray,
        vocab: int,
        confidence: dict[str, float],
    ) -> dict[str, np.ndarray]:
        """Extract anchors from all 321 unified atlas probes.

        Uses the complete UnifiedAtlasInventory which includes:
        - SEQUENCE_INVARIANT: 68 probes (Fibonacci, Lucas, Primes, Catalan, etc.)
        - SEMANTIC_PRIME: 65 probes (Wierzbicka's Natural Semantic Metalanguage)
        - COMPUTATIONAL_GATE: 76 probes (control flow, data types, functions)
        - EMOTION_CONCEPT: 32 probes (Plutchik wheel + dyads)
        - TEMPORAL_CONCEPT: 25 probes (tense, duration, causality, lifecycle)
        - SOCIAL_CONCEPT: 25 probes (power, kinship, formality, status)
        - MORAL_CONCEPT: 30 probes (Haidt's Moral Foundations Theory)

        Total: 321 probes for cross-domain triangulation.
        """
        probes = UnifiedAtlasInventory.all_probes()
        anchors: dict[str, np.ndarray] = {}

        for probe in probes:
            vectors: list[np.ndarray] = []
            for text in probe.support_texts:
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                valid = [tid for tid in ids if 0 <= tid < vocab]
                if valid:
                    # Fréchet mean for token embeddings (curvature is inherent)
                    token_vecs = [embedding[tid] for tid in valid]
                    vectors.append(self._frechet_mean_of_embeddings(token_vecs))

            if vectors:
                # Fréchet mean of phrase embeddings (curvature is inherent in HD space)
                anchors[probe.probe_id] = self._frechet_mean_of_embeddings(vectors)
                confidence[probe.probe_id] = probe.cross_domain_weight

        logger.info(
            "Extracted %d anchors from UnifiedAtlasInventory (%d probes available)",
            len(anchors),
            len(probes),
        )
        return anchors
