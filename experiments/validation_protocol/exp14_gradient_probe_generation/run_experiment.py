# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experiment 14: Gradient-Based Probe Generation

Goal: Generate probes that maximize activation in orthogonal directions.

Approach:
1. Start with a token sequence
2. Compute activation at target layer
3. Project activation onto orthogonal complement of current probe subspace
4. Maximize ||orthogonal_component|| via gradient ascent on embeddings
5. Discretize to nearest tokens

This closes the loop: given any probe subspace, we can generate new probes
that increase rank, until rank = hidden_dim.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_numerical_rank(activations, backend) -> int:
    """Compute numerical rank from SVD."""
    import mlx.core as mx

    acts_f32 = activations.astype(mx.float32)
    mx.eval(acts_f32)

    _, S, _ = mx.linalg.svd(acts_f32, stream=mx.cpu)
    mx.eval(S)

    s_values = S.tolist()
    s_max = max(s_values) if s_values else 0.0

    eps = 1.19209e-07
    threshold = s_max * (eps ** 0.5)

    rank = sum(1 for s in s_values if s > threshold)
    return rank


def compute_orthogonal_basis(probe_activations, rank):
    """Compute basis for orthogonal complement of probe subspace."""
    import mlx.core as mx

    probe_f32 = probe_activations.astype(mx.float32)
    mx.eval(probe_f32)

    # Covariance in hidden_dim space
    cov = probe_f32.T @ probe_f32
    mx.eval(cov)

    # Eigendecomposition
    eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
    mx.eval(eigvals, eigvecs)

    # Eigenvectors corresponding to smallest eigenvalues form the null space
    # Take d - rank eigenvectors with smallest eigenvalues
    hidden_dim = int(probe_activations.shape[1])
    null_rank = hidden_dim - rank

    if null_rank <= 0:
        return None  # Already full rank

    # Eigenvectors are in ascending eigenvalue order
    # First null_rank are the null space
    U_null = eigvecs[:, :null_rank]
    mx.eval(U_null)

    return U_null


def get_layer_activation(model, input_ids, layer_idx):
    """Get activation at specific layer for given input."""
    import mlx.core as mx

    inner = model.model if hasattr(model, "model") else model
    if not hasattr(inner, "layers"):
        raise RuntimeError("Model structure not compatible")

    # Get embeddings
    if hasattr(inner, "embed_tokens"):
        h = inner.embed_tokens(input_ids)
    elif hasattr(inner, "wte"):
        h = inner.wte(input_ids)
    else:
        raise RuntimeError("Cannot find embedding layer")

    # Forward through layers
    for idx, layer in enumerate(inner.layers):
        if idx > layer_idx:
            break
        result = layer(h)
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result

        if idx == layer_idx:
            # Mean pool over sequence
            pooled = mx.mean(h, axis=(0, 1))
            return pooled

    return None


def generate_orthogonal_probe(
    model,
    tokenizer,
    layer_idx: int,
    U_null,
    seed_tokens=None,
    n_steps: int = 50,
    lr: float = 0.1,
):
    """Generate a probe that activates in the null space of current probes.

    Uses gradient ascent on continuous token embeddings, then discretizes.
    """
    import mlx.core as mx

    inner = model.model if hasattr(model, "model") else model

    # Get embedding matrix
    if hasattr(inner, "embed_tokens"):
        embed_weight = inner.embed_tokens.weight
    elif hasattr(inner, "wte"):
        embed_weight = inner.wte.weight
    else:
        raise RuntimeError("Cannot find embedding layer")

    vocab_size = embed_weight.shape[0]
    embed_dim = embed_weight.shape[1]

    # Initialize with seed tokens or random
    seq_len = 10
    if seed_tokens is not None:
        init_ids = mx.array(seed_tokens[:seq_len])
        if init_ids.shape[0] < seq_len:
            pad = mx.zeros((seq_len - init_ids.shape[0],), dtype=mx.int32)
            init_ids = mx.concatenate([init_ids, pad])
    else:
        init_ids = mx.random.randint(100, vocab_size - 100, (seq_len,))

    mx.eval(init_ids)

    # Get initial embeddings
    init_embeds = embed_weight[init_ids]  # [seq_len, embed_dim]
    mx.eval(init_embeds)

    # Optimize embeddings directly
    embeds = init_embeds

    for step in range(n_steps):
        # Define objective: maximize projection onto null space
        def objective(e):
            # Reshape for model input: [1, seq_len, embed_dim]
            h = e[None, :, :]

            # Forward through layers (simplified - just use first few)
            for idx, layer in enumerate(inner.layers):
                if idx > layer_idx:
                    break
                result = layer(h)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result

            # Pool to get activation vector
            activation = mx.mean(h, axis=(0, 1))  # [hidden_dim]

            # Project onto null space
            proj = U_null @ (U_null.T @ activation)

            # Return negative norm (we want to maximize)
            return -mx.sqrt(mx.sum(proj ** 2) + 1e-8)

        # Compute gradient
        loss, grad = mx.value_and_grad(objective)(embeds)
        mx.eval(loss, grad)

        # Gradient descent (actually ascent since objective is negated)
        embeds = embeds - lr * grad
        mx.eval(embeds)

        if step % 10 == 0:
            logger.debug(f"Step {step}: null_proj_norm = {-float(loss):.4f}")

    # Discretize: find nearest tokens for each position
    final_ids = []
    for pos in range(seq_len):
        embed_pos = embeds[pos]  # [embed_dim]
        # Distance to all vocab embeddings
        dists = mx.sum((embed_weight - embed_pos[None, :]) ** 2, axis=1)
        mx.eval(dists)
        nearest_id = int(mx.argmin(dists))
        final_ids.append(nearest_id)

    # Decode to text
    try:
        text = tokenizer.decode(final_ids)
    except Exception:
        text = "<decode-failed>"

    return final_ids, text


def run_experiment(
    model_path: str,
    layer_idx: int,
    initial_probes: int = 100,
    n_generated: int = 20,
    output_path: str | None = None,
):
    """Run gradient-based probe generation experiment."""
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.probe_loader import load_all_probes
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    backend = get_default_backend()
    logger.info("Backend: %s", type(backend).__name__)

    # Load model
    logger.info("Loading model from %s", model_path)
    model, tokenizer = mlx_load(model_path)

    # Get dimensions
    hidden_dim = 0
    n_layers = 0
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
        first_layer = model.model.layers[0]
        if hasattr(first_layer, "input_layernorm"):
            hidden_dim = first_layer.input_layernorm.weight.shape[0]

    logger.info("Hidden dimension: %d", hidden_dim)
    logger.info("Target layer: %d", layer_idx)

    # Load probes
    probes = load_all_probes()
    from modelcypher.core.use_cases.merge.stages.probe_helpers import _select_probe_text
    probe_texts = [_select_probe_text(p) for p in probes if _select_probe_text(p)][:initial_probes]

    logger.info("Using %d initial probes", len(probe_texts))

    # Collect initial activations
    logger.info("=" * 60)
    logger.info("PHASE 1: Collecting initial probe activations")
    logger.info("=" * 60)

    activation_provider = MLXActivationProvider()
    batch_result = activation_provider.collect_probe_activations_batch(
        model, tokenizer, probe_texts
    )

    layer_activations = []
    for probe_idx in range(len(probe_texts)):
        if layer_idx in batch_result.hidden[probe_idx]:
            layer_activations.append(batch_result.hidden[probe_idx][layer_idx])

    probe_activations = mx.stack(layer_activations, axis=0)
    mx.eval(probe_activations)

    initial_rank = compute_numerical_rank(probe_activations, backend)
    logger.info(f"Initial rank: {initial_rank}/{hidden_dim} ({100*initial_rank/hidden_dim:.1f}%)")

    if initial_rank >= hidden_dim:
        logger.info("Already at full rank!")
        return

    # Compute null space basis
    logger.info("=" * 60)
    logger.info("PHASE 2: Computing null space basis")
    logger.info("=" * 60)

    U_null = compute_orthogonal_basis(probe_activations, initial_rank)
    if U_null is None:
        logger.info("No null space (full rank)")
        return

    null_dim = int(U_null.shape[1])
    logger.info(f"Null space dimension: {null_dim}")

    # Generate probes
    logger.info("=" * 60)
    logger.info("PHASE 3: Generating probes via gradient optimization")
    logger.info("=" * 60)

    generated_activations = []
    generated_texts = []

    for i in range(n_generated):
        logger.info(f"Generating probe {i+1}/{n_generated}...")

        # Use random seed each time
        token_ids, text = generate_orthogonal_probe(
            model, tokenizer, layer_idx, U_null,
            n_steps=30, lr=0.05
        )

        # Get activation for generated probe
        input_ids = mx.array([token_ids])
        activation = get_layer_activation(model, input_ids, layer_idx)

        if activation is not None:
            mx.eval(activation)
            generated_activations.append(activation)
            generated_texts.append(text)
            logger.info(f"  Generated: {repr(text[:50])}...")

    if not generated_activations:
        logger.error("Failed to generate any probes")
        return

    # Measure rank increase
    logger.info("=" * 60)
    logger.info("PHASE 4: Measuring rank increase")
    logger.info("=" * 60)

    generated_stack = mx.stack(generated_activations, axis=0)
    mx.eval(generated_stack)

    combined = mx.concatenate([probe_activations, generated_stack], axis=0)
    mx.eval(combined)

    final_rank = compute_numerical_rank(combined, backend)
    rank_increase = final_rank - initial_rank

    logger.info(f"Initial rank: {initial_rank}/{hidden_dim}")
    logger.info(f"Final rank: {final_rank}/{hidden_dim}")
    logger.info(f"Rank increase: +{rank_increase} from {n_generated} generated probes")
    logger.info(f"Efficiency: {rank_increase/n_generated:.2f} rank per probe")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if rank_increase > 0:
        logger.info("✓ Gradient-based generation INCREASES rank")
        logger.info("  This validates the closed-form probe generation approach")
    else:
        logger.info("? No rank increase - optimization may need tuning")

    # Save results
    if output_path:
        result = {
            "model_path": model_path,
            "layer_idx": layer_idx,
            "hidden_dim": hidden_dim,
            "initial_rank": initial_rank,
            "final_rank": final_rank,
            "rank_increase": rank_increase,
            "n_generated": n_generated,
            "generated_texts": generated_texts,
        }
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 14: Gradient-Based Probe Generation"
    )
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--initial-probes", type=int, default=100)
    parser.add_argument("--n-generated", type=int, default=20)
    parser.add_argument("--output", type=str,
                        default="experiments/validation_protocol/exp14_gradient_probe_generation/results.json")

    args = parser.parse_args()

    run_experiment(
        model_path=args.model,
        layer_idx=args.layer,
        initial_probes=args.initial_probes,
        n_generated=args.n_generated,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
