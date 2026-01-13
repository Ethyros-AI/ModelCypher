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

"""
Dimensional Alignment Measurement.

Measures alignment quality at each dimension:
- 1D: Tokenization (discrete - Jaccard overlap, sequence agreement)
- 2D: Embedding space (CKA on post-embed_tokens)
- 3D: Structural (CKA on pre/post LayerNorm)
- 4D+: Transformer (CKA on hidden/intermediate)

Same geodesic math applies wherever continuous representations exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class DimensionalAlignment:
    """Results of per-dimension alignment measurement."""

    # 1D: Tokenization (discrete)
    vocab_overlap: float  # |V_src ∩ V_tgt| / |V_tgt|
    vocab_jaccard: float  # |shared| / |union|
    sequence_agreement: float  # fraction of texts with same token count
    shared_token_count: int
    source_vocab_size: int
    target_vocab_size: int

    # 2D: Embedding
    embedding_cka: float | None  # CKA on post-embed_tokens

    # 3D: LayerNorm / Structural
    layernorm_cka: float | None  # CKA on pre/post LayerNorm

    # 4D+: Transformer
    hidden_cka_mean: float | None  # Mean CKA over all hidden layers
    hidden_cka_min: float | None  # Min CKA (worst layer)
    intermediate_cka_mean: float | None  # Mean CKA over intermediate

    def summary(self) -> str:
        """Generate human-readable alignment summary."""
        lines = [
            "DIMENSIONAL ALIGNMENT REPORT",
            "=" * 40,
            "",
            f"1D (Tokenization):",
            f"  Vocabulary overlap: {self.vocab_overlap:.1%}",
            f"  Jaccard similarity: {self.vocab_jaccard:.3f}",
            f"  Sequence agreement: {self.sequence_agreement:.1%}",
            f"  Shared tokens: {self.shared_token_count:,}",
            "",
        ]

        # Geodesic CKA is a diagnostic; 1.0 indicates strong overlap.
        if self.embedding_cka is not None:
            lines.extend([
                f"2D (Embedding):",
                f"  CKA: {self.embedding_cka:.6f}",
                "",
            ])

        if self.layernorm_cka is not None:
            lines.extend([
                f"3D (LayerNorm):",
                f"  Pre/Post CKA: {self.layernorm_cka:.6f}",
                "",
            ])

        if self.hidden_cka_mean is not None:
            lines.extend([
                f"4D+ (Transformer):",
                f"  Hidden CKA (mean): {self.hidden_cka_mean:.6f}",
                f"  Hidden CKA (min): {self.hidden_cka_min:.6f}" if self.hidden_cka_min else "",
            ])
            if self.intermediate_cka_mean is not None:
                lines.append(f"  Intermediate CKA (mean): {self.intermediate_cka_mean:.4f}")

        return "\n".join(lines)


# =============================================================================
# 1D: TOKENIZATION ALIGNMENT
# =============================================================================


def measure_1d_alignment(
    source_tokenizer: Any,
    target_tokenizer: Any,
    test_texts: list[str] | None = None,
) -> dict[str, float | int]:
    """
    Measure 1D tokenization alignment (discrete space).

    CKA doesn't apply to discrete symbols. Instead we measure:
    - Vocabulary overlap: fraction of target vocab present in source
    - Jaccard similarity: intersection / union
    - Sequence agreement: fraction of texts with same token count

    Args:
        source_tokenizer: Source model's tokenizer
        target_tokenizer: Target model's tokenizer
        test_texts: Optional texts for sequence agreement test

    Returns:
        Dict with vocab_overlap, vocab_jaccard, sequence_agreement, counts
    """
    # Get vocabularies
    src_vocab = set(source_tokenizer.get_vocab().keys())
    tgt_vocab = set(target_tokenizer.get_vocab().keys())

    shared = src_vocab & tgt_vocab
    union = src_vocab | tgt_vocab

    # Vocab metrics
    vocab_overlap = len(shared) / len(tgt_vocab) if tgt_vocab else 0.0
    vocab_jaccard = len(shared) / len(union) if union else 0.0

    # Sequence agreement (same token count for same text)
    sequence_agreement = 0.0
    if test_texts:
        agreements = 0
        for text in test_texts:
            src_tokens = source_tokenizer.encode(text, add_special_tokens=False)
            tgt_tokens = target_tokenizer.encode(text, add_special_tokens=False)
            if len(src_tokens) == len(tgt_tokens):
                agreements += 1
        sequence_agreement = agreements / len(test_texts)

    return {
        "vocab_overlap": vocab_overlap,
        "vocab_jaccard": vocab_jaccard,
        "sequence_agreement": sequence_agreement,
        "shared_token_count": len(shared),
        "source_vocab_size": len(src_vocab),
        "target_vocab_size": len(tgt_vocab),
    }


# =============================================================================
# 3D: LAYERNORM ALIGNMENT
# =============================================================================


def measure_3d_alignment(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    backend: "Backend",
) -> dict[str, float]:
    """
    Measure 3D LayerNorm alignment (structural space).

    LayerNorm is element-wise: y = (x - μ) / σ * w + b
    We measure CKA between pre-LN and post-LN activations to verify
    that LayerNorm preserves the geometric structure.

    Args:
        model: The model to probe
        tokenizer: Tokenizer for the model
        texts: Probe texts
        backend: Backend for computation

    Returns:
        Dict with layernorm_cka
    """
    from modelcypher.core.domain.geometry.cka import (
        compute_cka,
        HSICEstimator,
    )
    from modelcypher.core.domain.geometry.numerical_stability import (
        machine_epsilon,
        sqrt_scalar,
    )

    pre_ln_acts: list["Array"] = []
    post_ln_acts: list["Array"] = []

    try:
        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            return {"layernorm_cka": None}

        inner = model.model
        layer = inner.layers[0]  # Use first layer for measurement

        for text in texts[:50]:  # Sample 50 texts
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)
            input_ids = backend.array([token_ids], dtype="int32")

            # Get embedding
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                continue

            # Pre-LayerNorm: raw hidden state (after embedding or previous layer)
            pre_ln = backend.mean(h, axis=(0, 1))
            backend.eval(pre_ln)
            pre_ln_acts.append(pre_ln)

            # Post-LayerNorm
            if hasattr(layer, "input_layernorm"):
                h_post_ln = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_post_ln = layer.ln_1(h)
            else:
                h_post_ln = h

            post_ln = backend.mean(h_post_ln, axis=(0, 1))
            backend.eval(post_ln)
            post_ln_acts.append(post_ln)

        if len(pre_ln_acts) < 2:
            return {"layernorm_cka": None}

        # Compute CKA between pre and post LayerNorm
        pre_stacked = backend.stack(pre_ln_acts, axis=0)
        post_stacked = backend.stack(post_ln_acts, axis=0)
        pre_stacked = backend.astype(pre_stacked, "float32")
        post_stacked = backend.astype(post_stacked, "float32")
        backend.eval(pre_stacked, post_stacked)

        cka_result = compute_cka(
            pre_stacked,
            post_stacked,
            backend=backend,
            estimator=HSICEstimator.AUTO,
        )

        # Primary metric: geodesic RBF CKA (manifold-aware)
        geodesic_cka = cka_result.cka if cka_result.is_valid else None
        precision = sqrt_scalar(machine_epsilon(backend, pre_stacked), backend)

        if geodesic_cka is not None:
            geodesic_deviation = abs(1.0 - geodesic_cka)
            # Log deviations as info (LayerNorm may change structure)
            if geodesic_deviation > precision:
                logger.info(
                    "3D ALIGNMENT: Geodesic CKA deviation %.2e > precision %.2e (LayerNorm structure change).",
                    geodesic_deviation,
                    precision,
                )
        else:
            geodesic_deviation = None

        return {
            "layernorm_cka": geodesic_cka,
            "layernorm_geodesic_cka": geodesic_cka,
            "layernorm_geodesic_deviation": geodesic_deviation,
            "layernorm_precision_threshold": precision,
        }

    except Exception as e:
        logger.warning("3D alignment measurement failed: %s", e)
        return {"layernorm_cka": None}


# =============================================================================
# FULL DIMENSIONAL ALIGNMENT
# =============================================================================


def measure_dimensional_alignment(
    source_tokenizer: Any,
    target_tokenizer: Any,
    probe_metrics: dict[str, Any],
    test_texts: list[str] | None = None,
) -> DimensionalAlignment:
    """
    Compile full dimensional alignment report from probe metrics.

    Args:
        source_tokenizer: Source model tokenizer
        target_tokenizer: Target model tokenizer
        probe_metrics: Metrics from probe stage (contains 2D, 4D+ CKA)
        test_texts: Optional texts for 1D sequence agreement test

    Returns:
        DimensionalAlignment with all metrics
    """
    # 1D: Tokenization
    one_d = measure_1d_alignment(source_tokenizer, target_tokenizer, test_texts)

    # 2D: Embedding (from probe stage)
    embedding_cka = probe_metrics.get("embedding_cka")

    # 3D: LayerNorm (TODO: would need model to compute)
    layernorm_cka = probe_metrics.get("layernorm_cka")

    # 4D+: Transformer (from probe stage)
    layer_ckas = [
        v for k, v in probe_metrics.items()
        if k.startswith("layer_") and k.endswith("_cka") and isinstance(v, (int, float))
    ]
    hidden_cka_mean = sum(layer_ckas) / len(layer_ckas) if layer_ckas else None
    hidden_cka_min = min(layer_ckas) if layer_ckas else None

    inter_ckas = [
        v for k, v in probe_metrics.items()
        if "intermediate" in k and "cka" in k.lower() and isinstance(v, (int, float))
    ]
    intermediate_cka_mean = sum(inter_ckas) / len(inter_ckas) if inter_ckas else None

    return DimensionalAlignment(
        vocab_overlap=one_d["vocab_overlap"],
        vocab_jaccard=one_d["vocab_jaccard"],
        sequence_agreement=one_d["sequence_agreement"],
        shared_token_count=one_d["shared_token_count"],
        source_vocab_size=one_d["source_vocab_size"],
        target_vocab_size=one_d["target_vocab_size"],
        embedding_cka=embedding_cka,
        layernorm_cka=layernorm_cka,
        hidden_cka_mean=hidden_cka_mean,
        hidden_cka_min=hidden_cka_min,
        intermediate_cka_mean=intermediate_cka_mean,
    )


__all__ = [
    "DimensionalAlignment",
    "measure_1d_alignment",
    "measure_3d_alignment",
    "measure_dimensional_alignment",
]
