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

"""Runtime Self-Alignment: Metrics the model can compute about itself.

These metrics allow a model to assess its own processing quality at runtime
and potentially trigger self-correction (e.g., request clarification, reflect).

Key Metrics:
    comp/φ: Compression ratio normalized by golden ratio
        - 0.9-1.1: Optimal processing
        - >1.25: Over-expansion (confusion)
        - <0.8: Under-expansion (shallow/hallucination)

    logit_entropy: Uncertainty in next token prediction
        - Low: Very confident (may be hallucinating)
        - High: Uncertain (may need reflection)

    peak_layer: Where maximum activation occurs
        - Optimal: 50-70% through network
        - Early: May be shallow processing
        - Late: May be over-processing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

logger = logging.getLogger(__name__)

PHI = 1.618033988749895


@dataclass
class AlignmentMetrics:
    """Runtime alignment metrics computed from a single forward pass."""

    token_count: int
    comp_phi: float
    compression_ratio: float
    peak_layer: int
    total_layers: int
    initial_norm: float
    peak_norm: float
    final_norm: float
    logit_entropy: float

    @property
    def peak_layer_pct(self) -> float:
        """Peak layer as percentage of total depth."""
        return self.peak_layer / self.total_layers

    @property
    def phi_status(self) -> Literal["OPTIMAL", "OVER", "UNDER", "MARGINAL"]:
        """Assess comp/φ quality."""
        if 0.9 <= self.comp_phi <= 1.1:
            return "OPTIMAL"
        elif self.comp_phi > 1.25:
            return "OVER"
        elif self.comp_phi < 0.8:
            return "UNDER"
        return "MARGINAL"

    @property
    def should_reflect(self) -> bool:
        """Recommend self-reflection based on metrics."""
        # Suggest reflection if:
        # 1. Processing is sub-optimal
        # 2. Input is long (may need question extraction)
        # 3. Entropy is very high (uncertain)
        return (
            self.phi_status in ("OVER", "UNDER") or
            self.token_count > 20 or
            self.logit_entropy > 7.0
        )

    @property
    def confidence(self) -> Literal["HIGH", "MEDIUM", "LOW"]:
        """Overall confidence assessment."""
        if self.phi_status == "OPTIMAL" and 2.0 < self.logit_entropy < 6.0:
            return "HIGH"
        elif self.phi_status in ("OPTIMAL", "MARGINAL"):
            return "MEDIUM"
        return "LOW"


def compute_alignment_metrics(model, tokenizer, text: str) -> AlignmentMetrics:
    """Compute alignment metrics from a forward pass.

    Args:
        model: MLX model with model.model.embed_tokens and model.model.layers
        tokenizer: Tokenizer for encoding text
        text: Input text to analyze

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

    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        norm = float(mx.sqrt(mx.sum(hidden * hidden)))
        if norm > peak_norm:
            peak_norm = norm
            peak_layer = i + 1

    final_norm = float(mx.sqrt(mx.sum(hidden * hidden)))

    # Compression ratio
    compression_ratio = peak_norm / final_norm if final_norm > 1e-10 else 1.0
    comp_phi = compression_ratio / PHI

    # Get logits for entropy
    logits = model(input_ids)
    last_logits = logits[0, -1, :].astype(mx.float32)
    mx.eval(last_logits)

    # Compute entropy
    probs = mx.softmax(last_logits)
    mx.eval(probs)
    log_probs = mx.log(probs + 1e-10)
    entropy = float(-mx.sum(probs * log_probs))

    return AlignmentMetrics(
        token_count=len(tokens),
        comp_phi=comp_phi,
        compression_ratio=compression_ratio,
        peak_layer=peak_layer,
        total_layers=len(model.model.layers),
        initial_norm=initial_norm,
        peak_norm=peak_norm,
        final_norm=final_norm,
        logit_entropy=entropy,
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
        logger.info(f"Triggering self-reflection (comp/φ={metrics.comp_phi:.3f}, tokens={metrics.token_count})")
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
        "comp_phi": round(m.comp_phi, 3),
        "status": m.phi_status,
        "confidence": m.confidence,
        "should_reflect": m.should_reflect,
    }
