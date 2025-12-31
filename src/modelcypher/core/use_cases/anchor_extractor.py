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
from modelcypher.core.domain.agents.computational_gate_atlas import ComputationalGateInventory
from modelcypher.core.domain.agents.semantic_prime_frames import SemanticPrimeFrames
from modelcypher.core.domain.agents.semantic_prime_multilingual import (
    SemanticPrimeMultilingualInventoryLoader,
)
from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean
from modelcypher.core.use_cases.quantization_utils import (
    QuantizationConfig,
    dequantize_if_needed,
    quantization_hint_for_key,
)
from modelcypher.ports.backend import Array, Backend

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
        vectors: list[Array],
        backend: Backend | None = None,
    ) -> Array:
        """Compute Fréchet mean of embedding vectors.

        Embeddings live in curved representation space. Fréchet (Karcher)
        mean provides the proper geometric center.

        Parameters
        ----------
        vectors : list[Array]
            List of embedding vectors (backend arrays)
        backend : Backend | None
            Backend for computation (uses default if None)

        Returns
        -------
        Array
            Fréchet mean as backend array
        """
        if not vectors:
            raise ValueError("Cannot compute Fréchet mean of empty vector list")

        b = backend or get_default_backend()
        if len(vectors) == 1:
            return b.array(vectors[0])

        stacked = b.stack([b.array(vec) for vec in vectors], axis=0)
        mean = frechet_mean(stacked, backend=b)
        b.eval(mean)
        return mean

    def extract(
        self,
        model_path: str,
        weights: dict[str, Any],
        config: AnchorExtractionConfig | None = None,
        quantization: QuantizationConfig | None = None,
        backend: Backend | None = None,
    ) -> tuple[dict[str, Array], dict[str, float]]:
        cfg = config or AnchorExtractionConfig()
        b = backend or get_default_backend()
        tokenizer = self._load_tokenizer(model_path)
        embedding_key, embedding = self.token_embedding_matrix(weights, backend=b)
        hint = quantization_hint_for_key(embedding_key, quantization)
        embedding = dequantize_if_needed(embedding, embedding_key, weights, b, hint=hint)
        embedding = b.astype(b.array(embedding), "float32")
        b.eval(embedding)

        if embedding.ndim != 2:
            raise AnchorExtractorError(
                f"Token embedding weight must be 2D. {embedding_key} shape={embedding.shape}"
            )

        if embedding.shape[0] < embedding.shape[1]:
            embedding = b.transpose(embedding)
            b.eval(embedding)

        vocab = int(embedding.shape[0])
        anchors: dict[str, Array] = {}
        confidence: dict[str, float] = {}

        if cfg.use_unified_atlas:
            # Use all probes from the unified atlas (supersedes individual atlas calls)
            anchors.update(
                self._unified_atlas_anchors(tokenizer, embedding, vocab, confidence, b)
            )
        else:
            # Legacy mode: use subset of atlas sources
            if cfg.use_enriched_primes:
                anchors.update(
                    self._enriched_prime_anchors(tokenizer, embedding, vocab, confidence, cfg, b)
                )
            else:
                anchors.update(
                    self._basic_prime_anchors(tokenizer, embedding, vocab, confidence, b)
                )

            if cfg.include_computational_gates:
                anchors.update(
                    self._computational_gate_anchors(tokenizer, embedding, vocab, confidence, b)
                )

        if not anchors:
            raise AnchorExtractorError(
                f"Unable to derive anchors from token embeddings ({embedding_key})."
            )

        return anchors, confidence

    @staticmethod
    def token_embedding_matrix(
        weights: dict[str, Any],
        backend: Backend | None = None,
    ) -> tuple[str, Array]:
        """Extract the token embedding matrix from model weights.

        Prefers input embeddings (embed_tokens) over output embeddings (lm_head)
        because input embeddings encode *semantic similarity* while output embeddings
        encode *contextual similarity* (Bertolotti & Cazzola, ICML 2024). For semantic
        anchor extraction, input embeddings produce more meaningful clusters.

        Args:
            weights: Model weights dictionary.

        Returns:
            Tuple of (key_name, embedding_matrix).
        """
        # Priority order: input embeddings encode semantic similarity (Bertolotti 2024)
        preferred_suffixes = [
            "embed_tokens.weight",
            "tok_embeddings.weight",
            "token_embedding.weight",
            "wte.weight",
            "lm_head.weight",  # Fallback: contextual similarity, less ideal for anchors
        ]
        b = backend or get_default_backend()

        def _dtype_is_numeric(arr: Array) -> bool:
            dtype_name = str(getattr(arr, "dtype", "")).lower()
            return any(tag in dtype_name for tag in ("float", "int", "uint", "bfloat"))

        for suffix in preferred_suffixes:
            for key, value in weights.items():
                if not key.endswith(suffix):
                    continue
                try:
                    arr = b.array(value)
                except Exception:
                    continue
                if _dtype_is_numeric(arr):
                    return key, arr

        scored: list[tuple[str, Array, int, int]] = []
        for key, value in weights.items():
            try:
                arr = b.array(value)
            except Exception:
                continue
            if arr.ndim != 2:
                continue
            if not _dtype_is_numeric(arr):
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
            size = 1
            for dim in arr.shape:
                size *= int(dim)
            scored.append((key, arr, score, size))

        if not scored:
            raise AnchorExtractorError(
                "Unable to locate token embedding weights in the model parameters."
            )

        scored.sort(key=lambda item: (item[2], item[3]))
        key, arr, _, _ = scored[-1]
        return key, arr

    @staticmethod
    def normalize_anchor_matrix(
        matrix: Array,
        backend: Backend | None = None,
    ) -> Array:
        if matrix.ndim != 2:
            return matrix
        b = backend or get_default_backend()
        mean = b.mean(matrix, axis=0, keepdims=True)
        centered = matrix - mean
        norms = b.norm(centered, axis=1, keepdims=True)
        norms = b.maximum(norms, b.array(1e-6))
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
        embedding: Array,
        vocab: int,
        confidence: dict[str, float],
        cfg: AnchorExtractionConfig,
        backend: Backend,
    ) -> dict[str, Array]:
        primes = SemanticPrimeFrames.enriched()
        polyglot = SemanticPrimeMultilingualInventoryLoader.global_diverse()
        polyglot_by_id: dict[str, list[str]] = {}
        for prime in polyglot.primes:
            texts: list[str] = []
            for bucket in prime.languages:
                texts.extend(bucket.texts[: cfg.max_polyglot_texts_per_language])
            polyglot_by_id[prime.id] = texts

        anchors: dict[str, Array] = {}
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

            vectors: list[Array] = []
            for text in unique:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                valid = [token_id for token_id in ids if 0 <= token_id < vocab]
                if not valid:
                    continue
                # Use Fréchet mean for token embeddings (curvature is inherent)
                token_vecs = [embedding[tid] for tid in valid]
                vectors.append(self._frechet_mean_of_embeddings(token_vecs, backend=backend))

            if not vectors:
                continue

            anchor_id = f"prime:{prime.id}"
            # Fréchet mean of phrase embeddings (curvature is inherent in HD space)
            anchors[anchor_id] = self._frechet_mean_of_embeddings(vectors, backend=backend)
            confidence[anchor_id] = self._confidence_for_token_count(len(core_valid))

        return anchors

    def _basic_prime_anchors(
        self,
        tokenizer: Tokenizer,
        embedding: Array,
        vocab: int,
        confidence: dict[str, float],
        backend: Backend,
    ) -> dict[str, Array]:
        primes = SemanticPrimeInventory.english2014()
        anchors: dict[str, Array] = {}

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
            anchors[anchor_id] = self._frechet_mean_of_embeddings(
                token_vecs, backend=backend
            )
            confidence[anchor_id] = self._confidence_for_token_count(len(valid))

        return anchors

    def _computational_gate_anchors(
        self,
        tokenizer: Tokenizer,
        embedding: Array,
        vocab: int,
        confidence: dict[str, float],
        backend: Backend,
    ) -> dict[str, Array]:
        gates = ComputationalGateInventory.probe_gates()
        anchors: dict[str, Array] = {}

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

            vectors: list[Array] = []
            token_counts: list[int] = []
            for text in unique:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                valid = [token_id for token_id in ids if 0 <= token_id < vocab]
                if not valid:
                    continue
                # Fréchet mean for token embeddings (curvature is inherent)
                token_vecs = [embedding[tid] for tid in valid]
                vectors.append(self._frechet_mean_of_embeddings(token_vecs, backend=backend))
                token_counts.append(len(valid))

            if not vectors:
                continue

            anchor_id = f"gate:{gate.id}"
            # Fréchet mean of phrase embeddings (curvature is inherent in HD space)
            anchors[anchor_id] = self._frechet_mean_of_embeddings(vectors, backend=backend)

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
        embedding: Array,
        vocab: int,
        confidence: dict[str, float],
        backend: Backend,
    ) -> dict[str, Array]:
        """Extract anchors from all unified atlas probes.

        Uses the complete UnifiedAtlasInventory which includes:
        - SEQUENCE_INVARIANT: 70 probes (Fibonacci, Lucas, Primes, Catalan, etc.)
        - SEMANTIC_PRIME: 65 probes (Wierzbicka's Natural Semantic Metalanguage)
        - COMPUTATIONAL_GATE: 76 probes (control flow, data types, functions)
        - EMOTION_CONCEPT: 32 probes (Plutchik wheel + dyads)
        - TEMPORAL_CONCEPT: 25 probes (tense, duration, causality, lifecycle)
        - SPATIAL_CONCEPT: 23 probes (vertical, lateral, depth, mass, furniture)
        - SOCIAL_CONCEPT: 25 probes (power, kinship, formality, status)
        - MORAL_CONCEPT: 30 probes (Haidt's Moral Foundations Theory)
        - CONCEPTUAL_GENEALOGY: 29 probes (etymology + lineage)
        - METAPHOR_INVARIANT: 14 probes (cross-cultural semantic anchors)
        - SYNTAX_CONCEPT: 24 probes (parts of speech, morphology, word order)
        - SAFETY_ETHICS: 34 probes (consent, autonomy, coercion, boundaries)

        See UnifiedAtlasInventory.total_probe_count() for current total.
        """
        probes = UnifiedAtlasInventory.all_probes()
        anchors: dict[str, Array] = {}

        for probe in probes:
            vectors: list[Array] = []
            for text in probe.support_texts:
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                valid = [tid for tid in ids if 0 <= tid < vocab]
                if valid:
                    # Fréchet mean for token embeddings (curvature is inherent)
                    token_vecs = [embedding[tid] for tid in valid]
                    vectors.append(self._frechet_mean_of_embeddings(token_vecs, backend=backend))

            if vectors:
                # Fréchet mean of phrase embeddings (curvature is inherent in HD space)
                anchors[probe.probe_id] = self._frechet_mean_of_embeddings(
                    vectors, backend=backend
                )
                confidence[probe.probe_id] = probe.cross_domain_weight

        logger.info(
            "Extracted %d anchors from UnifiedAtlasInventory (%d probes available)",
            len(anchors),
            len(probes),
        )
        return anchors
