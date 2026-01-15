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
Dual-Path Generator for entropy disagreement tracking (MLX Backend).

This is the MLX/macOS implementation. For other backends:
- CUDA/PyTorch: see dual_path_cuda.py
- JAX/TPU: see dual_path_jax.py

Use modelcypher.infrastructure.dual_path_factory.get_dual_path_generator_class()
for automatic platform selection.

NOTE: This module has infrastructure dependencies (mlx_lm for model loading)
that cannot be fully abstracted via the Backend protocol. The model loading
and forward passes remain MLX-specific until a full inference abstraction
layer is implemented.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

# Infrastructure dependencies (MLX-specific model loading)
# These cannot be abstracted via Backend protocol
# from mlx_lm import load  # Moved to lazy import inside DualPathGenerator
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

# Import our ported modules
from modelcypher.core.domain.inference.entropy_dynamics import (
    EntropyDeltaSample,
    EntropyDeltaTracker,
    LogitDivergenceCalculator,
    LogitEntropyCalculator,
)


def compute_token_rank_metrics(
    scores: "Array",
    token_id: int,
    backend: "Backend | None" = None,
) -> tuple[int, float, bool]:
    """Compute rank geometry for a token in logit space.

    Ranking is defined by ordering in logit space (monotone under softmax),
    and the frontier boundary is derived from the largest relative logit gap.

    Args:
        scores: 1D array of logit scores.
        token_id: ID of the selected token.
        backend: Backend for array operations (auto-detected if None).

    Returns:
        Tuple of (rank, rank_fraction, frontier_hit) where:
        - rank: 0-indexed rank of the token (0 = highest score).
        - rank_fraction: 1.0 = top token, 0.0 = bottom token.
        - frontier_hit: True if token is in the derived top frontier.
    """
    b = backend or get_default_backend()

    if scores.ndim > 1:
        scores = b.squeeze(scores)

    vocab_size = scores.shape[0]
    token_score = scores[token_id]

    # Rank = count of tokens with strictly higher score.
    rank_arr = b.sum(b.astype(scores > token_score, "float32"))
    b.eval(rank_arr)
    token_rank = int(b.to_scalar(rank_arr))

    # Rank fraction: 1 = top token, 0 = bottom token.
    if vocab_size > 1:
        rank_fraction = 1.0 - (token_rank / (vocab_size - 1))
    else:
        rank_fraction = 1.0

    # Frontier size is the index of the largest relative gap in sorted scores.
    if vocab_size > 1:
        sorted_scores = -b.sort(-scores)
        gaps = sorted_scores[:-1] - sorted_scores[1:]
        eps = division_epsilon(b, scores)
        max_gap_arr = b.max(gaps)
        b.eval(max_gap_arr)
        max_gap = float(b.to_scalar(max_gap_arr))
        if max_gap <= eps:
            frontier_size = vocab_size
        else:
            frontier_index_arr = b.argmax(gaps)
            b.eval(frontier_index_arr)
            frontier_index = int(b.to_scalar(frontier_index_arr))
            frontier_size = frontier_index + 1
    else:
        frontier_size = 1

    frontier_hit = token_rank < frontier_size

    return token_rank, rank_fraction, frontier_hit


@dataclass
class SecurityScanMetrics:
    token_count: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float


class DualPathGenerator:
    """
    Orchestrates dual-path generation with entropy disagreement tracking.

    Maintains concept of "Base Model" and "Adapter Model" (via hot-swapping or separate instances).
    Ideally uses one model and toggles LoRA adapters if supported by MLX-LM,
    or maintains two model instances if memory permits.

    For strict 1:1 parity with Swift `DualPathGenerator`, this attempts to use the
    Single-Model Hot-Swap approach if possible, or falls back to just running
    separate forward passes if we can manage the state.

    NOTE: Model loading and forward passes use MLX-LM infrastructure directly.
    Only math operations (softmax, argmax, log) use the Backend protocol.
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop_sequences: list[str],
        signal_router: Any = None,  # placeholder for signal system
        backend: "Backend | None" = None,
    ):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.stop_sequences = stop_sequences
        self._backend = backend or get_default_backend()
        source = Path(base_model_path).name
        self.delta_tracker = EntropyDeltaTracker(source=source, router=signal_router)

        # Load model(s)
        # Note: In a real app we might inject the loaded model.
        # Here we assume paths are provided.
        # MLX-LM loading handling:
        # We load the BASE model.
        # For the ADAPTER path, we need to apply adapters.
        from mlx_lm import load

        logger.info(f"Loading model from {base_model_path}")
        self.model, self.tokenizer = load(base_model_path)

        # If adapter path is present, we need a way to apply it.
        # MLX-LM supports adapters via `load(..., adapter_path=...)`
        # But we need DYNAMIC switching for "Dual Path".
        # We can implement this by manually applying LoRA layers or loading a second model instance.
        # Loading a second instance is safer for parity and easier to implement first.
        # Swift implementation used hot-swap.
        # Let's try loading a SECOND model for the "Adapter" path if memory allows,
        # or just fail if implementation complexity of hot-swap in Python is too high for this turn.
        # A 1:1 port of the Swift `applyAdapter` / `detachAdapter` logic requires access to the LoRA layers in Python.

        self.adapter_model = None
        if adapter_path:
            logger.info(f"Loading adapter model from {adapter_path}")
            # In MLX-LM, loading with adapter_path fuses? Or returns LoRA model?
            from mlx_lm import load

            self.adapter_model, _ = load(base_model_path, adapter_path=adapter_path)
        else:
            self.adapter_model = self.model  # If no adapter, both paths are same (degenerate case)

        self.entropy_calc = LogitEntropyCalculator()

    async def generate(self, prompt: str) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generates text while performing dual-path analysis.
        Yields chunks: token, anomaly, metrics.
        """
        b = self._backend

        # 1. Tokenize
        prompt_tokens = b.array(self.tokenizer.encode(prompt))

        start_time = time.time()
        time_to_first = 0.0
        token_count = 0

        # Start tracking session
        correlation_id = uuid.uuid4()
        self.delta_tracker.start_session(correlation_id)

        # We need manual generation loop to intercept logits
        # MLX-LM `generate` is high level. We use `generate_step` or manual loop.

        # Internal state
        tokens = b.tolist(prompt_tokens)

        # Cache for both models
        # MLX-LM make_cache equivalent?
        # We'll use the simplified loop for now.

        cache_base = None
        cache_adapter = None

        # Initial forward pass (prefill)
        # Base Path
        logits_base, cache_base = self.model(prompt_tokens[None], cache=cache_base)
        # Adapter Path
        logits_adapter, cache_adapter = self.adapter_model(prompt_tokens[None], cache=cache_adapter)

        # Logits handling for next token...
        # We need to sample from ADAPTER logits, but analyze BOTH.

        curr_logits_adapter = logits_adapter[:, -1, :]
        curr_logits_base = logits_base[:, -1, :]

        while token_count < self.max_tokens:
            # Analyze
            # Compute Entropy/Divergence
            # (Synchronous in Python, unlike Swift actor)
            analysis_start = time.perf_counter()

            # This logic mirrors `process` in Swift's DualPathLogitProcessor

            # 1. Sample from Adapter
            # temp/top_p logic
            token_tensor = self._sample(curr_logits_adapter)
            token_id = token_tensor.item()

            # 2. Decode
            text = self.tokenizer.decode([token_id])

            # 3. Security Analysis
            # Compute base entropy/variance
            base_entropy, base_variance = self.entropy_calc.compute(curr_logits_base)

            # Compute adapter entropy/variance
            adapter_entropy, adapter_variance = self.entropy_calc.compute(curr_logits_adapter)

            # Compute KL
            # Need LogitDivergenceCalculator (assuming it was ported as static or class)
            div_calc = LogitDivergenceCalculator()
            kl = div_calc.kl_divergence(curr_logits_adapter, curr_logits_base)

            # Record Delta
            # We call delta_tracker.record_step(...)
            # Note: Swift logic accumulates `PendingEntropyData` then sends to actor.
            # Python is simpler.
            base_top_idx = b.argmax(curr_logits_base)
            adapter_top_idx = b.argmax(curr_logits_adapter)
            b.eval(base_top_idx, adapter_top_idx)
            base_top_token = int(b.to_scalar(base_top_idx))
            adapter_top_token = int(b.to_scalar(adapter_top_idx))

            # Compute base logits for rank geometry.
            scores_base = b.squeeze(curr_logits_base)
            token_logit_arr = b.take(scores_base, b.array([token_id]), axis=0)
            token_logit_arr = b.squeeze(token_logit_arr)
            max_logit_arr = b.max(scores_base)
            b.eval(token_logit_arr, max_logit_arr)
            token_logit = float(b.to_scalar(token_logit_arr))
            max_logit = float(b.to_scalar(max_logit_arr))
            logit_margin = max(0.0, max_logit - token_logit)

            # Compute rank geometry for the generated token.
            _, rank_fraction, frontier_hit = compute_token_rank_metrics(
                scores_base, token_id, backend=b
            )

            sample = EntropyDeltaSample(
                token_index=token_count,
                generated_token=token_id,
                base_entropy=base_entropy,
                base_logit_variance=base_variance,
                base_top_token=base_top_token,
                adapter_entropy=adapter_entropy,
                adapter_logit_variance=adapter_variance,
                adapter_top_token=adapter_top_token,
                latency_ms=(time.perf_counter() - analysis_start) * 1000,
                kl_divergence_adapter_to_base=kl,
                base_logit_margin=logit_margin,
                base_token_logit=token_logit,
                base_rank_fraction=rank_fraction,
                base_frontier_hit=frontier_hit,
            )

            self.delta_tracker.record_step(sample)

            # Yield token
            yield {"type": "token", "text": text}

            # Prepare next step
            tokens.append(token_id)
            token_count += 1
            if token_count == 1:
                time_to_first = (time.time() - start_time) * 1000

            input_tensor = b.array([[token_id]])

            logits_base, cache_base = self.model(input_tensor, cache=cache_base)
            logits_adapter, cache_adapter = self.adapter_model(input_tensor, cache=cache_adapter)

            curr_logits_base = logits_base[:, -1, :]
            curr_logits_adapter = logits_adapter[:, -1, :]

            # Check output stop
            if text in self.stop_sequences:
                break

        total_time = (time.time() - start_time) * 1000
        metrics = SecurityScanMetrics(
            token_count=token_count,
            time_to_first_token_ms=time_to_first,
            total_time_ms=total_time,
            tokens_per_second=token_count / (total_time / 1000),
        )
        yield {"type": "metrics", "metrics": metrics}

    def _sample(self, logits: "Array") -> "Array":
        """Simple sampling (greedy or temperature)."""
        b = self._backend

        if self.temperature == 0:
            return b.argmax(logits, axis=-1)

        # Apply temp
        scaled_logits = logits / self.temperature
        # Use backend's random_categorical - required for proper sampling
        if not hasattr(b, "random_categorical"):
            raise NotImplementedError(
                f"Backend {type(b).__name__} does not support random_categorical. "
                "Sampling requires this operation. Use a backend that supports it, "
                "or set temperature=0.0 for greedy decoding."
            )
        return b.random_categorical(scaled_logits)
