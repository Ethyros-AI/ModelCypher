# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experiment 11: SAE Feature Coverage

Question: Can SAEs identify which dimensions our probes miss?

Protocol:
1. Collect activations at target layer (e.g., SmolLM layer 15 - the 26% coverage layer)
2. Train SAE on those activations (using diverse text corpus)
3. Run all probes through SAE, identify dormant features
4. Measure: n_dormant / n_total_features

Expected outcome:
- Dormant features should correspond to ~74% of dimensions (the unmapped space)

Success criteria:
- Dormant feature ratio correlates with rank deficiency
- SAE reveals what concepts our probes miss
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from sae import SAEConfig, SparseAutoencoder, train_sae, find_dormant_features

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from SAE feature coverage experiment."""

    model_path: str
    layer_idx: int
    hidden_dim: int

    # Probe coverage (from prior experiments)
    n_probes: int
    numerical_rank: int
    rank_coverage: float  # rank / hidden_dim

    # SAE feature coverage
    sae_hidden_dim: int
    n_dormant_features: int
    n_active_features: int
    dormant_ratio: float  # n_dormant / sae_hidden_dim

    # Training metadata
    sae_recon_loss: float
    sae_sparsity: float

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "layer_idx": self.layer_idx,
            "hidden_dim": self.hidden_dim,
            "n_probes": self.n_probes,
            "numerical_rank": self.numerical_rank,
            "rank_coverage": self.rank_coverage,
            "sae_hidden_dim": self.sae_hidden_dim,
            "n_dormant_features": self.n_dormant_features,
            "n_active_features": self.n_active_features,
            "dormant_ratio": self.dormant_ratio,
            "sae_recon_loss": self.sae_recon_loss,
            "sae_sparsity": self.sae_sparsity,
            "analysis": {
                "rank_deficiency": 1.0 - self.rank_coverage,
                "dormant_ratio": self.dormant_ratio,
                "correlation_hypothesis": "dormant_ratio should correlate with rank_deficiency",
            },
        }


def compute_numerical_rank(activations, backend) -> int:
    """Compute numerical rank from SVD."""
    import mlx.core as mx

    acts_f32 = activations.astype(mx.float32)
    mx.eval(acts_f32)

    _, S, _ = mx.linalg.svd(acts_f32, stream=mx.cpu)
    mx.eval(S)

    s_values = S.tolist()
    s_max = max(s_values) if s_values else 0.0

    # Threshold: sqrt(machine_epsilon) * max_singular_value
    eps = 1.19209e-07  # float32 machine epsilon
    threshold = s_max * (eps ** 0.5)

    rank = sum(1 for s in s_values if s > threshold)
    return rank


def collect_diverse_activations(model, tokenizer, layer_idx: int, n_samples: int = 10000):
    """Collect activations from diverse text for SAE training.

    Uses random text generation to get broad activation coverage.
    """
    import mlx.core as mx
    import random

    logger.info("Collecting diverse activations for SAE training...")

    # Generate diverse prompts
    diverse_prompts = []

    # Categories of text to generate
    categories = [
        # Simple patterns
        "The quick brown fox",
        "A B C D E F G H",
        "1 2 3 4 5 6 7 8",
        "red green blue yellow",

        # Technical
        "def function(x): return x * 2",
        "SELECT * FROM users WHERE",
        "import numpy as np",
        "<html><body>Hello</body></html>",

        # Languages
        "Bonjour, comment allez-vous?",
        "Hola, cómo estás?",
        "你好世界",
        "こんにちは",

        # Math/Science
        "E = mc^2 is Einstein's",
        "The derivative of x^2 is",
        "Hydrogen has atomic number",
        "The mitochondria is the",

        # Conversational
        "Hello! How are you today?",
        "I think the weather is",
        "What do you think about",
        "Can you help me with",

        # Narrative
        "Once upon a time there was",
        "In the beginning, there was",
        "The story begins with a",
        "She walked down the",

        # Random combinations
        "xyz123 abc456 def789",
        "!@#$%^&*()_+-=[]{}",
        "aaaaa bbbbb ccccc ddddd",
    ]

    # Expand with variations
    for base in categories:
        diverse_prompts.append(base)
        # Add length variations
        diverse_prompts.append(base + " " + base)
        diverse_prompts.append(base[:len(base)//2] if len(base) > 5 else base)

    # Add random token sequences
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50000
    for _ in range(100):
        # Random tokens decoded back to text
        random_ids = [random.randint(100, min(vocab_size-1, 1000)) for _ in range(random.randint(5, 20))]
        try:
            text = tokenizer.decode(random_ids)
            if text and len(text) > 3:
                diverse_prompts.append(text)
        except Exception:
            pass

    # Limit to n_samples
    if len(diverse_prompts) > n_samples:
        diverse_prompts = diverse_prompts[:n_samples]

    logger.info(f"Generated {len(diverse_prompts)} diverse prompts")

    # Collect activations
    all_activations = []

    inner = model.model if hasattr(model, "model") else model
    if not hasattr(inner, "layers"):
        raise RuntimeError("Model structure not compatible")

    for text in diverse_prompts:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens.ids)

            input_ids = mx.array([token_ids])

            # Get embeddings
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(input_ids)
            elif hasattr(inner, "wte"):
                h = inner.wte(input_ids)
            else:
                continue

            # Forward through layers up to target
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
                    mx.eval(pooled)
                    all_activations.append(pooled)
                    break

        except Exception as e:
            logger.debug(f"Skipping prompt due to error: {e}")
            continue

    if not all_activations:
        raise RuntimeError("Failed to collect any activations")

    # Stack into matrix
    activations = mx.stack(all_activations, axis=0)
    mx.eval(activations)

    logger.info(f"Collected {activations.shape[0]} activation vectors of dim {activations.shape[1]}")

    return activations


def run_experiment(
    model_path: str,
    layer_idx: int,
    max_probes: int = 500,
    sae_expansion: int = 8,
    sae_epochs: int = 5,
    output_path: str | None = None,
) -> ExperimentResult:
    """Run SAE feature coverage experiment."""
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
        if hasattr(first_layer, "input_layernorm") and hasattr(first_layer.input_layernorm, "weight"):
            hidden_dim = first_layer.input_layernorm.weight.shape[0]

    logger.info("Hidden dimension: %d", hidden_dim)
    logger.info("Number of layers: %d", n_layers)
    logger.info("Target layer: %d", layer_idx)

    if layer_idx >= n_layers:
        raise ValueError(f"Layer index {layer_idx} >= n_layers {n_layers}")

    # Phase 1: Collect diverse activations for SAE training
    logger.info("=" * 60)
    logger.info("PHASE 1: Collecting diverse activations for SAE training")
    logger.info("=" * 60)

    diverse_activations = collect_diverse_activations(
        model, tokenizer, layer_idx, n_samples=500
    )

    # Phase 2: Train SAE
    logger.info("=" * 60)
    logger.info("PHASE 2: Training Sparse Autoencoder")
    logger.info("=" * 60)

    sae_config = SAEConfig(
        input_dim=hidden_dim,
        expansion_factor=sae_expansion,
        num_epochs=sae_epochs,
    )

    logger.info(f"SAE config: input_dim={sae_config.input_dim}, "
                f"hidden_dim={sae_config.hidden_dim}, "
                f"expansion={sae_expansion}x")

    sae = train_sae(diverse_activations, sae_config, verbose=True)

    # Compute final reconstruction loss
    _, recon_loss, l1_loss = sae.loss(diverse_activations)
    mx.eval(recon_loss, l1_loss)
    final_recon = float(recon_loss)
    final_l1 = float(l1_loss)

    logger.info(f"Final SAE: recon_loss={final_recon:.4f}, sparsity={final_l1:.4f}")

    # Phase 3: Collect probe activations
    logger.info("=" * 60)
    logger.info("PHASE 3: Collecting probe activations")
    logger.info("=" * 60)

    probes = load_all_probes()
    logger.info("Total probes available: %d", len(probes))

    from modelcypher.core.use_cases.merge.stages.probe_helpers import _select_probe_text
    valid_probes = []
    for probe in probes:
        text = _select_probe_text(probe)
        if text:
            valid_probes.append((probe, text))

    if max_probes and max_probes < len(valid_probes):
        valid_probes = valid_probes[:max_probes]

    logger.info("Using %d probes", len(valid_probes))

    # Collect activations
    activation_provider = MLXActivationProvider()
    probe_texts = [text for _, text in valid_probes]
    batch_result = activation_provider.collect_probe_activations_batch(
        model, tokenizer, probe_texts
    )

    # Extract activations for target layer
    layer_activations = []
    for probe_idx in range(len(valid_probes)):
        if layer_idx in batch_result.hidden[probe_idx]:
            act = batch_result.hidden[probe_idx][layer_idx]
            layer_activations.append(act)

    if not layer_activations:
        raise RuntimeError(f"No activations collected for layer {layer_idx}")

    probe_activations = mx.stack(layer_activations, axis=0)
    mx.eval(probe_activations)

    logger.info(f"Collected {probe_activations.shape[0]} probe activations")

    # Phase 4: Compute numerical rank
    logger.info("=" * 60)
    logger.info("PHASE 4: Computing numerical rank of probe activations")
    logger.info("=" * 60)

    numerical_rank = compute_numerical_rank(probe_activations, backend)
    rank_coverage = numerical_rank / hidden_dim

    logger.info(f"Numerical rank: {numerical_rank}/{hidden_dim} ({100*rank_coverage:.1f}%)")

    # Phase 5: Find dormant SAE features
    logger.info("=" * 60)
    logger.info("PHASE 5: Finding dormant SAE features")
    logger.info("=" * 60)

    dormant_mask, max_activations, n_dormant, n_total = find_dormant_features(
        sae, probe_activations, threshold=0.01
    )

    n_active = n_total - n_dormant
    dormant_ratio = n_dormant / n_total

    logger.info(f"SAE features: {n_total} total")
    logger.info(f"  Active (activated by probes): {n_active} ({100*n_active/n_total:.1f}%)")
    logger.info(f"  Dormant (never activated): {n_dormant} ({100*dormant_ratio:.1f}%)")

    # Phase 6: Decoder Rank Analysis (Experiment 12)
    logger.info("=" * 60)
    logger.info("PHASE 6: Decoder Rank Analysis (Active Features)")
    logger.info("=" * 60)

    # Extract decoder columns for active features
    active_mask = ~dormant_mask
    mx.eval(active_mask)

    # Get indices of active features
    active_indices = []
    for i in range(n_total):
        if not bool(dormant_mask[i]):
            active_indices.append(i)

    logger.info(f"Active features: {len(active_indices)}")

    # Extract decoder columns for active features
    # W_dec shape: [input_dim, hidden_dim] = [576, 4608]
    decoder_active_cols = []
    for idx in active_indices[:min(len(active_indices), 1000)]:  # Limit for memory
        col = sae.W_dec[:, idx]
        decoder_active_cols.append(col)

    if decoder_active_cols:
        decoder_active = mx.stack(decoder_active_cols, axis=1)  # [576, n_active]
        mx.eval(decoder_active)

        # Compute rank of active decoder columns
        decoder_rank = compute_numerical_rank(decoder_active.T, backend)
        logger.info(f"Rank of active decoder columns: {decoder_rank}/{hidden_dim}")
        logger.info(f"Rank of probe activations: {numerical_rank}/{hidden_dim}")

        if decoder_rank > numerical_rank + 10:
            logger.info("✓ SAE finds MORE dimensions than probes span")
            logger.info("  → Dormant features may access unmapped dimensions")
        else:
            logger.info("≈ SAE active features span same subspace as probes")
            logger.info("  → Unmapped dimensions may be inaccessible via any input")
    else:
        decoder_rank = 0
        logger.info("No active features found")

    # Phase 7: Orthogonal Feature Analysis
    logger.info("=" * 60)
    logger.info("PHASE 7: Orthogonal Feature Analysis")
    logger.info("=" * 60)

    # Compute probe subspace via SVD
    probe_f32 = probe_activations.astype(mx.float32)
    mx.eval(probe_f32)
    U_probe, S_probe, _ = mx.linalg.svd(probe_f32, stream=mx.cpu)
    mx.eval(U_probe, S_probe)

    # Keep only significant directions (rank)
    U_probe_truncated = U_probe[:, :numerical_rank]  # [n_probes, rank]
    mx.eval(U_probe_truncated)

    # For a sample of features, compute orthogonal component
    orthogonal_scores = []
    sample_size = min(500, n_total)
    feature_indices = list(range(0, n_total, n_total // sample_size))[:sample_size]

    for feat_idx in feature_indices:
        # Get decoder column for this feature
        d_i = sae.W_dec[:, feat_idx]  # [hidden_dim]

        # Project onto probe subspace
        # First need to transpose probe activations for correct dimensions
        # U_probe is [n_probes, rank], probe_activations is [n_probes, hidden_dim]
        # We need to find basis vectors in hidden_dim space

        # Compute SVD of probe_activations.T to get hidden_dim basis
        _, _, Vh_probe = mx.linalg.svd(probe_f32.T, stream=mx.cpu)
        mx.eval(Vh_probe)
        # Vh_probe is [min(hidden_dim, n_probes), n_probes]
        # Left singular vectors of probe_f32.T = right singular vectors of probe_f32
        # These span the column space of probe_f32 in hidden_dim

        # Actually, let's compute probe_activations @ probe_activations.T for Gram
        # and get eigenvectors in hidden_dim space
        # probe_activations: [n_probes, hidden_dim]
        # We want basis vectors in hidden_dim space that span the probe subspace

        # Covariance in hidden_dim space
        cov = probe_f32.T @ probe_f32  # [hidden_dim, hidden_dim]
        mx.eval(cov)

        # Eigendecomposition
        eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
        mx.eval(eigvals, eigvecs)

        # Take top 'rank' eigenvectors (they're returned in ascending order)
        U_hidden = eigvecs[:, -numerical_rank:]  # [hidden_dim, rank]
        mx.eval(U_hidden)

        # Project d_i onto probe subspace
        proj = U_hidden @ (U_hidden.T @ d_i)
        mx.eval(proj)

        # Orthogonal component
        orth = d_i - proj
        mx.eval(orth)

        # Norm of orthogonal component
        orth_norm = float(mx.sqrt(mx.sum(orth ** 2)))
        d_norm = float(mx.sqrt(mx.sum(d_i ** 2)))

        if d_norm > 1e-8:
            orth_ratio = orth_norm / d_norm
        else:
            orth_ratio = 0.0

        orthogonal_scores.append((feat_idx, orth_ratio))

        # Only compute once for the basis (it's the same for all features)
        break

    # Re-run with cached basis
    orthogonal_scores = []
    for feat_idx in feature_indices:
        d_i = sae.W_dec[:, feat_idx]

        proj = U_hidden @ (U_hidden.T @ d_i)
        orth = d_i - proj

        orth_norm = float(mx.sqrt(mx.sum(orth ** 2)))
        d_norm = float(mx.sqrt(mx.sum(d_i ** 2)))

        if d_norm > 1e-8:
            orth_ratio = orth_norm / d_norm
        else:
            orth_ratio = 0.0

        orthogonal_scores.append((feat_idx, orth_ratio))

    # Sort by orthogonal ratio (descending)
    orthogonal_scores.sort(key=lambda x: x[1], reverse=True)

    # Report top orthogonal features
    logger.info(f"Analyzed {len(orthogonal_scores)} features for orthogonality to probe subspace")
    logger.info(f"Probe subspace rank: {numerical_rank}")
    logger.info("")
    logger.info("Top features accessing unmapped dimensions:")
    for i, (feat_idx, orth_ratio) in enumerate(orthogonal_scores[:10]):
        logger.info(f"  Feature {feat_idx}: {100*orth_ratio:.1f}% orthogonal to probe subspace")

    # Count features with significant orthogonal component
    high_orth_count = sum(1 for _, r in orthogonal_scores if r > 0.5)
    med_orth_count = sum(1 for _, r in orthogonal_scores if 0.1 < r <= 0.5)
    low_orth_count = sum(1 for _, r in orthogonal_scores if r <= 0.1)

    logger.info("")
    logger.info(f"Orthogonality distribution (of {len(orthogonal_scores)} sampled):")
    logger.info(f"  High (>50% orthogonal): {high_orth_count} ({100*high_orth_count/len(orthogonal_scores):.1f}%)")
    logger.info(f"  Medium (10-50%): {med_orth_count} ({100*med_orth_count/len(orthogonal_scores):.1f}%)")
    logger.info(f"  Low (<10%): {low_orth_count} ({100*low_orth_count/len(orthogonal_scores):.1f}%)")

    # Phase 8: Summary Analysis
    logger.info("=" * 60)
    logger.info("SUMMARY ANALYSIS")
    logger.info("=" * 60)

    rank_deficiency = 1.0 - rank_coverage

    logger.info(f"Rank deficiency (1 - rank/d): {100*rank_deficiency:.1f}%")
    logger.info(f"Dormant feature ratio: {100*dormant_ratio:.1f}%")
    logger.info(f"Features with high orthogonality (>50%): {high_orth_count}/{len(orthogonal_scores)}")

    if high_orth_count > 0:
        logger.info("")
        logger.info("✓ SAE identifies features accessing unmapped dimensions")
        logger.info("  → These features can guide probe generation")

    # Build result
    result = ExperimentResult(
        model_path=model_path,
        layer_idx=layer_idx,
        hidden_dim=hidden_dim,
        n_probes=len(valid_probes),
        numerical_rank=numerical_rank,
        rank_coverage=rank_coverage,
        sae_hidden_dim=n_total,
        n_dormant_features=n_dormant,
        n_active_features=n_active,
        dormant_ratio=dormant_ratio,
        sae_recon_loss=final_recon,
        sae_sparsity=final_l1,
    )

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 11: SAE Feature Coverage"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Model to analyze",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Layer index to analyze (default: 15 for SmolLM's 26% coverage layer)",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=500,
        help="Maximum number of probes to use",
    )
    parser.add_argument(
        "--sae-expansion",
        type=int,
        default=8,
        help="SAE expansion factor (hidden_dim = input_dim * expansion)",
    )
    parser.add_argument(
        "--sae-epochs",
        type=int,
        default=5,
        help="Number of SAE training epochs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/validation_protocol/exp11_sae_feature_coverage/results.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    run_experiment(
        model_path=args.model,
        layer_idx=args.layer,
        max_probes=args.max_probes,
        sae_expansion=args.sae_expansion,
        sae_epochs=args.sae_epochs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
