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

"""Runtime metrics computed from model forward passes.

Metrics computed during inference:

    expansion_ratio: peak_norm / final_norm
        Ratio of maximum hidden state norm to final norm.
        Thresholds below are empirical defaults (configurable).

    logit_entropy: Shannon entropy of output distribution
        Computed from softmax of final logits.

    peak_layer: Layer index with maximum hidden state norm.

    e_pi_matches: Count of consecutive layer norm ratios within
        tolerance of e/π (≈0.865) or π/e (≈1.156).

Note: Thresholds (0.9, 1.1, 1.25, 0.8) are empirical defaults
observed on test sets. Override via configuration for your use case.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

logger = logging.getLogger(__name__)

E_PI = math.e / math.pi  # 0.8653
PI_E = math.pi / math.e  # 1.1557


@dataclass
class AlignmentMetrics:
    """Runtime alignment metrics computed from a single forward pass."""

    token_count: int
    expansion_ratio: float
    compression_ratio: float
    peak_layer: int
    total_layers: int
    initial_norm: float
    peak_norm: float
    final_norm: float
    logit_entropy: float
    e_pi_matches: int  # Layer ratios matching e/π or π/e

    @property
    def peak_layer_pct(self) -> float:
        """Peak layer as percentage of total depth."""
        return self.peak_layer / self.total_layers

    @property
    def e_pi_ratio(self) -> float:
        """e/π match ratio (matches / total layers)."""
        return self.e_pi_matches / self.total_layers if self.total_layers > 0 else 0

    @property
    def expansion_status(self) -> Literal["OPTIMAL", "OVER", "UNDER", "MARGINAL"]:
        """Classify expansion ratio into bins.

        Thresholds are empirical defaults. Override for your use case.
        """
        if 0.9 <= self.expansion_ratio <= 1.1:
            return "OPTIMAL"
        elif self.expansion_ratio > 1.25:
            return "OVER"
        elif self.expansion_ratio < 0.8:
            return "UNDER"
        return "MARGINAL"

    @property
    def constant_alignment(self) -> Literal["STRONG", "MODERATE", "WEAK"]:
        """Classify e/π match ratio into bins.

        Thresholds derived from empirical observation on 16-layer models:
        - 41% match rate observed on correct answers (n=100)
        - 34% match rate observed on incorrect answers (n=100)
        """
        if self.e_pi_ratio >= 0.40:
            return "STRONG"
        elif self.e_pi_ratio >= 0.30:
            return "MODERATE"
        return "WEAK"

    @property
    def should_reflect(self) -> bool:
        """Flag based on metric thresholds.

        Returns True if any threshold is exceeded. Thresholds are empirical.
        """
        return (
            self.expansion_status in ("OVER", "UNDER") or
            self.token_count > 20 or
            self.logit_entropy > 7.0 or
            self.constant_alignment == "WEAK"
        )

    @property
    def confidence(self) -> Literal["HIGH", "MEDIUM", "LOW"]:
        """Overall confidence assessment combining all metrics."""
        # High confidence requires good expansion ratio AND good constant alignment
        if (self.expansion_status == "OPTIMAL" and
            self.constant_alignment == "STRONG" and
            2.0 < self.logit_entropy < 6.0):
            return "HIGH"
        elif (self.expansion_status in ("OPTIMAL", "MARGINAL") and
              self.constant_alignment in ("STRONG", "MODERATE")):
            return "MEDIUM"
        return "LOW"


def compute_alignment_metrics(
    model,
    tokenizer,
    text: str,
    e_pi_tolerance: float = 0.1,
) -> AlignmentMetrics:
    """Compute alignment metrics from a forward pass.

    Args:
        model: MLX model with model.model.embed_tokens and model.model.layers
        tokenizer: Tokenizer for encoding text
        text: Input text to analyze
        e_pi_tolerance: Tolerance for matching e/π or π/e ratios

    Returns:
        AlignmentMetrics with all computed values
    """
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    # Track norms through layers
    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)
    initial_norm = float(mx.sqrt(mx.sum(hidden * hidden)))
    peak_norm = initial_norm
    peak_layer = 0

    norms = [initial_norm]

    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        norm = float(mx.sqrt(mx.sum(hidden * hidden)))
        norms.append(norm)
        if norm > peak_norm:
            peak_norm = norm
            peak_layer = i + 1

    final_norm = norms[-1]

    # Use dtype-derived epsilon for numerical stability
    # sqrt(eps) provides headroom for safe division
    eps = float(mx.finfo(mx.float32).eps)
    div_eps = math.sqrt(eps)
    log_eps = mx.finfo(mx.float32).tiny  # Smallest positive float for log safety

    # Compression ratio and expansion ratio (peak/final)
    compression_ratio = peak_norm / final_norm if final_norm > div_eps else 1.0
    expansion_ratio = compression_ratio  # Direct ratio, no PHI normalization

    # Count layer-to-layer ratios matching e/π or π/e
    # This metric correlates with correctness (discovered via SHA-256 mining research)
    e_pi_matches = 0
    for i in range(1, len(norms)):
        if norms[i - 1] > div_eps:
            ratio = norms[i] / norms[i - 1]
            if abs(ratio - E_PI) < e_pi_tolerance or abs(ratio - PI_E) < e_pi_tolerance:
                e_pi_matches += 1

    # Get logits for entropy
    logits = model(input_ids)
    last_logits = logits[0, -1, :].astype(mx.float32)
    mx.eval(last_logits)

    # Compute entropy
    probs = mx.softmax(last_logits)
    mx.eval(probs)
    log_probs = mx.log(probs + log_eps)
    entropy = float(-mx.sum(probs * log_probs))

    return AlignmentMetrics(
        token_count=len(tokens),
        expansion_ratio=expansion_ratio,
        compression_ratio=compression_ratio,
        peak_layer=peak_layer,
        total_layers=len(model.model.layers),
        initial_norm=initial_norm,
        peak_norm=peak_norm,
        final_norm=final_norm,
        logit_entropy=entropy,
        e_pi_matches=e_pi_matches,
    )


def self_aligned_generate(
    model,
    tokenizer,
    prompt: str,
    generate_fn,
    max_tokens: int = 100,
    reflect_prefix: str = "Let me understand the question. ",
) -> tuple[str, AlignmentMetrics, bool]:
    """Generate with optional self-reflection based on alignment metrics.

    Args:
        model: MLX model
        tokenizer: Tokenizer
        prompt: Input prompt
        generate_fn: Generation function (model, tokenizer, prompt, max_tokens) -> str
        max_tokens: Maximum tokens to generate
        reflect_prefix: Prefix to add if reflection is triggered

    Returns:
        Tuple of (response, metrics, did_reflect)
    """
    # Check alignment metrics for the input
    metrics = compute_alignment_metrics(model, tokenizer, prompt)

    did_reflect = False

    if metrics.should_reflect and reflect_prefix not in prompt:
        # Add reflection prompt
        logger.info(f"Triggering self-reflection (expansion_ratio={metrics.expansion_ratio:.3f}, tokens={metrics.token_count})")
        augmented_prompt = prompt.rstrip() + "\n\n" + reflect_prefix
        did_reflect = True
    else:
        augmented_prompt = prompt

    # Generate response
    response = generate_fn(model, tokenizer, prompt=augmented_prompt, max_tokens=max_tokens, verbose=False)

    return response, metrics, did_reflect


# Convenience function for quick assessment
def quick_assess(model, tokenizer, text: str) -> dict:
    """Quick assessment returning a simple dict."""
    m = compute_alignment_metrics(model, tokenizer, text)
    return {
        "tokens": m.token_count,
        "expansion_ratio": round(m.expansion_ratio, 3),
        "e_pi_matches": m.e_pi_matches,
        "e_pi_ratio": round(m.e_pi_ratio, 3),
        "expansion_status": m.expansion_status,
        "constant_alignment": m.constant_alignment,
        "confidence": m.confidence,
        "should_reflect": m.should_reflect,
    }
