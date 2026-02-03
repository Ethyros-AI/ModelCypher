# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 17: SOAR Curriculum - Structural Quality Metrics
#
# Computes geometric "structural quality" metrics for problems
# without solving them. Based on SOAR paper finding:
# "Structural quality matters more than solution correctness."

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.fisher_information import (
    compute_empirical_fisher_diagonal,
    FisherInformationResult,
)
from modelcypher.core.domain.geometry.mode_connectivity import (
    analyze_mode_connectivity,
    ModeConnectivityResult,
)
from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class StructuralMetrics:
    """Structural quality metrics for a single problem or problem set."""

    # Fisher-based metrics
    fisher_mean: float  # Mean Fisher score across dimensions
    fisher_trace: float  # Total curvature (trace of FIM)
    fisher_effective_rank: float  # How many dimensions matter (participation ratio)

    # CKA-based metrics (if comparing problem sets)
    cka_similarity: float = 1.0  # CKA between problem activations and reference

    # Barrier-based metrics (mode connectivity)
    barrier_height: float = 0.0  # Max divergence along activation path
    barrier_normalized: float = 0.0  # Normalized barrier

    # Combined score (higher = better structural quality)
    structural_quality_score: float = 0.0


def compute_activation_fisher(
    activations: "Array",
    backend: "Backend | None" = None,
) -> FisherInformationResult:
    """Compute Fisher Information diagonal from activations.

    This measures which dimensions the model "uses" when processing
    these activations. High Fisher = dimension is important.

    Args:
        activations: Activation tensor [n_samples, d]
        backend: Optional backend

    Returns:
        FisherInformationResult with diagonal and statistics
    """
    b = backend or get_default_backend()
    return compute_empirical_fisher_diagonal(activations, b)


def compute_cka_compatibility(
    source_activations: "Array",
    target_activations: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Compute CKA similarity between two sets of activations.

    Higher CKA = more compatible representations.

    Args:
        source_activations: Reference activations [n_samples, d1]
        target_activations: Problem activations [n_samples, d2]
        backend: Optional backend

    Returns:
        CKA score in [0, 1] (1 = identical structure)
    """
    b = backend or get_default_backend()

    # Center activations
    source_centered = source_activations - b.mean(source_activations, axis=0, keepdims=True)
    target_centered = target_activations - b.mean(target_activations, axis=0, keepdims=True)
    b.eval(source_centered, target_centered)

    return compute_linear_cka_from_activations(source_centered, target_centered, b)


def compute_activation_barrier(
    source_activations: "Array",
    target_activations: "Array",
    n_steps: int = 11,
    backend: "Backend | None" = None,
) -> tuple[float, float]:
    """Compute CKA-based barrier along activation interpolation path.

    This measures how much the representation diverges from source
    as we interpolate toward target. High barrier = incompatible problems.

    Args:
        source_activations: Starting activations [n_samples, d]
        target_activations: Ending activations [n_samples, d]
        n_steps: Number of interpolation points
        backend: Optional backend

    Returns:
        (barrier_height, normalized_barrier) tuple
    """
    b = backend or get_default_backend()

    # Center for CKA
    source_mean = b.mean(source_activations, axis=0, keepdims=True)
    target_mean = b.mean(target_activations, axis=0, keepdims=True)
    source_centered = source_activations - source_mean
    target_centered = target_activations - target_mean
    b.eval(source_centered, target_centered)

    # Compute CKA loss (1 - CKA) at interpolation points
    losses = []
    t_values = [i / (n_steps - 1) for i in range(n_steps)]

    for t in t_values:
        # Interpolate activations
        interpolated = (1 - t) * source_centered + t * target_centered
        b.eval(interpolated)

        # CKA divergence from source
        cka = compute_linear_cka_from_activations(source_centered, interpolated, b)
        losses.append(1.0 - cka)

    # Barrier = max loss
    barrier_height = max(losses)

    # Normalized by endpoint losses
    source_loss = losses[0]  # Should be ~0
    target_loss = losses[-1]
    normalized = barrier_height / max(target_loss, 1e-8) if target_loss > 1e-8 else 0.0

    return barrier_height, normalized


def compute_structural_metrics(
    activations: "Array",
    reference_activations: "Array | None" = None,
    backend: "Backend | None" = None,
) -> StructuralMetrics:
    """Compute all structural quality metrics for a set of activations.

    Args:
        activations: Activations from processing problems [n_samples, d]
        reference_activations: Optional reference for comparison [n_ref, d]
        backend: Optional backend

    Returns:
        StructuralMetrics with all measurements
    """
    b = backend or get_default_backend()

    # Fisher metrics
    fisher_result = compute_activation_fisher(activations, b)

    fisher_mean = fisher_result.mean_fim
    fisher_trace = fisher_result.trace_fim
    fisher_effective_rank = fisher_result.effective_rank

    # CKA similarity (if reference provided)
    cka_similarity = 1.0
    barrier_height = 0.0
    barrier_normalized = 0.0

    if reference_activations is not None:
        # CKA and barrier require same sample count
        # If different, use cosine similarity as proxy
        n_samples = int(activations.shape[0])
        n_ref = int(reference_activations.shape[0])

        if n_samples == n_ref and n_samples > 1:
            # Standard CKA path
            cka_similarity = compute_cka_compatibility(reference_activations, activations, b)
            barrier_height, barrier_normalized = compute_activation_barrier(
                reference_activations, activations, n_steps=11, backend=b
            )
        else:
            # Use mean activation cosine similarity as proxy
            # This handles single-sample activations
            act_mean = b.mean(activations, axis=0)
            ref_mean = b.mean(reference_activations, axis=0)
            b.eval(act_mean, ref_mean)

            # Cosine similarity
            dot = float(b.sum(act_mean * ref_mean))
            norm_act = float(b.sqrt(b.sum(act_mean * act_mean)))
            norm_ref = float(b.sqrt(b.sum(ref_mean * ref_mean)))
            cka_similarity = max(0.0, dot / max(norm_act * norm_ref, 1e-10))

            # Use 1 - cosine_sim as proxy for barrier
            barrier_height = 1.0 - cka_similarity
            barrier_normalized = barrier_height

    # Compute combined structural quality score
    # INSIGHT from exp17 v1: Problems too similar to reference don't teach much!
    # The "Goldilocks zone" is moderate difference - not too easy, not too hard.
    #
    # Higher quality = better for learning:
    # - Moderate CKA similarity (0.85-0.95 ideal, not 0.99+)
    # - Moderate barrier (some challenge, but not confusing)
    # - Lower Fisher on training data (model needs to learn, not just recall)
    #
    # Score components:
    # 1. Goldilocks CKA: peaks at ~0.9, penalizes both <0.7 and >0.98
    # 2. Productive difficulty: moderate barrier is good
    # 3. Learning opportunity: low Fisher = model doesn't already know this

    # Goldilocks CKA: bell curve centered at 0.9
    cka_goldilocks = 1.0 - abs(cka_similarity - 0.90) * 5.0  # Peaks at 0.9, drops off both sides
    cka_goldilocks = max(0.0, min(1.0, cka_goldilocks))

    # Productive difficulty: barrier in [0.02, 0.10] is ideal
    # Too low = nothing to learn, too high = confusing
    if barrier_height < 0.02:
        barrier_score = barrier_height / 0.02  # Ramps up to optimal
    elif barrier_height <= 0.10:
        barrier_score = 1.0  # Optimal zone
    else:
        barrier_score = max(0.0, 1.0 - (barrier_height - 0.10) * 5)  # Drops off after

    # Learning opportunity: lower Fisher = model doesn't already know this
    # Inverse relationship - low Fisher is GOOD for learning
    fisher_learning = 1.0 - min(fisher_mean * 100, 1.0)  # Lower is better

    quality_score = (
        0.4 * cka_goldilocks +      # Moderate similarity (not too easy)
        0.3 * barrier_score +        # Productive difficulty
        0.3 * fisher_learning        # Learning opportunity
    )

    return StructuralMetrics(
        fisher_mean=fisher_mean,
        fisher_trace=fisher_trace,
        fisher_effective_rank=fisher_effective_rank,
        cka_similarity=cka_similarity,
        barrier_height=barrier_height,
        barrier_normalized=barrier_normalized,
        structural_quality_score=quality_score,
    )


def collect_problem_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    max_length: int = 64,
    model_path: str | None = None,
) -> "Array":
    """Collect activations from a model for a list of prompts.

    Args:
        model: The model (with .model.layers attribute)
        tokenizer: Tokenizer for encoding
        prompts: List of problem prompts
        layer_idx: Which layer to collect from
        max_length: Max sequence length
        model_path: Optional model path for architecture detection

    Returns:
        Activations tensor [n_prompts, hidden_dim]
    """
    from modelcypher.adapters.model_architecture import get_model_architecture

    backend = get_default_backend()
    arch = get_model_architecture(model, model_path=model_path)
    activations_list = []

    for prompt in prompts:
        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        input_ids = mx.array([tokens])

        # Forward pass with hook to capture activations
        # Simple approach: just call the layer directly
        hidden = arch.embed_module(input_ids)
        mx.eval(hidden)

        for i, layer in enumerate(arch.layers):
            if i > layer_idx:
                break
            # Standard transformer forward
            hidden = layer(hidden)
            mx.eval(hidden)

        # Mean pool over sequence
        pooled = mx.mean(hidden, axis=1)  # [1, d]
        mx.eval(pooled)
        activations_list.append(pooled)

    # Stack all activations
    all_activations = mx.concatenate(activations_list, axis=0)
    mx.eval(all_activations)

    return all_activations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test with synthetic data
    backend = get_default_backend()
    backend.random_seed(42)

    source = backend.random_normal((50, 128))
    target = backend.random_normal((50, 128))

    metrics = compute_structural_metrics(source, target, backend)
    print(f"\nStructural Metrics:")
    print(f"  Fisher mean: {metrics.fisher_mean:.6f}")
    print(f"  Fisher variance: {metrics.fisher_variance:.6f}")
    print(f"  Fisher effective rank: {metrics.fisher_effective_rank}")
    print(f"  CKA similarity: {metrics.cka_similarity:.4f}")
    print(f"  Barrier height: {metrics.barrier_height:.4f}")
    print(f"  Quality score: {metrics.structural_quality_score:.4f}")
