# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experiment 13: Targeted Probe Generation

Question: Can we increase numerical rank by adding probes that activate orthogonal SAE features?

Theorem to validate:
  If feature f has decoder column d_f orthogonal to probe subspace U,
  AND we find input x such that SAE encodes x with high activation on f,
  THEN adding x to probe set increases rank.

Protocol:
1. Train SAE, compute probe subspace U
2. Identify features orthogonal to U (from Exp 11/12)
3. Search existing probe corpus for probes that activate orthogonal features
4. Measure: Does adding these probes increase rank?

This validates that orthogonality predicts rank-increasing probes.
The search is over existing probes - no generation yet.
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

# Import SAE from exp11
sys.path.insert(0, str(Path(__file__).parent.parent / "exp11_sae_feature_coverage"))
from sae import SAEConfig, SparseAutoencoder, train_sae

if TYPE_CHECKING:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from targeted probe generation experiment."""

    model_path: str
    layer_idx: int
    hidden_dim: int

    # Initial state
    initial_n_probes: int
    initial_rank: int
    initial_coverage: float

    # After adding targeted probes
    final_n_probes: int
    final_rank: int
    final_coverage: float

    # Improvement
    rank_increase: int
    probes_added: int
    efficiency: float  # rank_increase / probes_added

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "layer_idx": self.layer_idx,
            "hidden_dim": self.hidden_dim,
            "initial": {
                "n_probes": self.initial_n_probes,
                "rank": self.initial_rank,
                "coverage": self.initial_coverage,
            },
            "final": {
                "n_probes": self.final_n_probes,
                "rank": self.final_rank,
                "coverage": self.final_coverage,
            },
            "improvement": {
                "rank_increase": self.rank_increase,
                "probes_added": self.probes_added,
                "efficiency": self.efficiency,
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


def compute_probe_subspace_basis(probe_activations, rank):
    """Compute orthonormal basis for probe subspace in hidden_dim space."""
    import mlx.core as mx

    probe_f32 = probe_activations.astype(mx.float32)
    mx.eval(probe_f32)

    # Covariance in hidden_dim space
    cov = probe_f32.T @ probe_f32  # [hidden_dim, hidden_dim]
    mx.eval(cov)

    # Eigendecomposition
    eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
    mx.eval(eigvals, eigvecs)

    # Take top 'rank' eigenvectors (returned in ascending order)
    U_hidden = eigvecs[:, -rank:]  # [hidden_dim, rank]
    mx.eval(U_hidden)

    return U_hidden


def compute_feature_orthogonality(sae, U_hidden):
    """Compute orthogonality of each SAE feature to probe subspace.

    Returns list of (feature_idx, orthogonality_ratio) sorted descending.
    """
    import mlx.core as mx

    n_features = sae.config.hidden_dim
    orthogonal_scores = []

    for feat_idx in range(n_features):
        d_i = sae.W_dec[:, feat_idx]

        # Project onto probe subspace
        proj = U_hidden @ (U_hidden.T @ d_i)
        orth = d_i - proj

        orth_norm = float(mx.sqrt(mx.sum(orth ** 2)))
        d_norm = float(mx.sqrt(mx.sum(d_i ** 2)))

        if d_norm > 1e-8:
            orth_ratio = orth_norm / d_norm
        else:
            orth_ratio = 0.0

        orthogonal_scores.append((feat_idx, orth_ratio))

    # Sort by orthogonality (descending)
    orthogonal_scores.sort(key=lambda x: x[1], reverse=True)
    return orthogonal_scores


def find_probes_activating_feature(sae, all_probe_activations, feature_idx, top_k=10):
    """Find probes that maximally activate a specific SAE feature.

    Returns list of (probe_idx, activation_value) sorted descending.
    """
    import mlx.core as mx

    # Encode all probes
    encoded = sae.encode(all_probe_activations)  # [n_probes, n_features]
    mx.eval(encoded)

    # Get activation for target feature across all probes
    feature_activations = encoded[:, feature_idx]  # [n_probes]
    mx.eval(feature_activations)

    # Find top activating probes
    n_probes = int(all_probe_activations.shape[0])
    activations_list = [(i, float(feature_activations[i])) for i in range(n_probes)]
    activations_list.sort(key=lambda x: x[1], reverse=True)

    return activations_list[:top_k]


def run_experiment(
    model_path: str,
    layer_idx: int,
    initial_probes: int = 100,
    target_probes: int = 50,
    sae_epochs: int = 10,
    output_path: str | None = None,
) -> ExperimentResult:
    """Run targeted probe generation experiment."""
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

    # Load ALL probes
    probes = load_all_probes()
    logger.info("Total probes available: %d", len(probes))

    from modelcypher.core.use_cases.merge.stages.probe_helpers import _select_probe_text
    all_probe_texts = []
    for probe in probes:
        text = _select_probe_text(probe)
        if text:
            all_probe_texts.append(text)

    # Limit probes for faster iteration
    max_total_probes = 1000
    if len(all_probe_texts) > max_total_probes:
        all_probe_texts = all_probe_texts[:max_total_probes]

    logger.info("Using %d probes (limited for speed)", len(all_probe_texts))

    # Collect activations for probes
    logger.info("=" * 60)
    logger.info("PHASE 1: Collecting activations for probes")
    logger.info("=" * 60)

    activation_provider = MLXActivationProvider()
    batch_result = activation_provider.collect_probe_activations_batch(
        model, tokenizer, all_probe_texts
    )

    # Extract activations for target layer
    all_layer_activations = []
    valid_probe_indices = []
    for probe_idx in range(len(all_probe_texts)):
        if layer_idx in batch_result.hidden[probe_idx]:
            act = batch_result.hidden[probe_idx][layer_idx]
            all_layer_activations.append(act)
            valid_probe_indices.append(probe_idx)

    all_probe_activations = mx.stack(all_layer_activations, axis=0)
    mx.eval(all_probe_activations)
    logger.info(f"Collected {all_probe_activations.shape[0]} probe activations")

    # Phase 2: Train SAE on diverse activations
    logger.info("=" * 60)
    logger.info("PHASE 2: Training SAE")
    logger.info("=" * 60)

    sae_config = SAEConfig(
        input_dim=hidden_dim,
        expansion_factor=8,
        num_epochs=sae_epochs,
    )

    # Use first 500 probes for diverse training data
    train_activations = all_probe_activations[:500]
    sae = train_sae(train_activations, sae_config, verbose=True)

    # Phase 3: Establish baseline with initial probes
    logger.info("=" * 60)
    logger.info("PHASE 3: Baseline with %d random probes", initial_probes)
    logger.info("=" * 60)

    # Use first N probes as initial set
    initial_activations = all_probe_activations[:initial_probes]
    mx.eval(initial_activations)

    initial_rank = compute_numerical_rank(initial_activations, backend)
    initial_coverage = initial_rank / hidden_dim

    logger.info(f"Initial rank: {initial_rank}/{hidden_dim} ({100*initial_coverage:.1f}%)")

    # Phase 4: Compute probe subspace and find orthogonal features
    logger.info("=" * 60)
    logger.info("PHASE 4: Finding orthogonal SAE features")
    logger.info("=" * 60)

    U_hidden = compute_probe_subspace_basis(initial_activations, initial_rank)
    orthogonal_features = compute_feature_orthogonality(sae, U_hidden)

    logger.info("Top 10 orthogonal features:")
    for feat_idx, orth_ratio in orthogonal_features[:10]:
        logger.info(f"  Feature {feat_idx}: {100*orth_ratio:.1f}% orthogonal")

    # Phase 5: Find probes that activate orthogonal features
    logger.info("=" * 60)
    logger.info("PHASE 5: Finding probes that activate orthogonal features")
    logger.info("=" * 60)

    # For each highly orthogonal feature, find probes that activate it
    # Use probes NOT in the initial set
    remaining_activations = all_probe_activations[initial_probes:]
    remaining_indices = valid_probe_indices[initial_probes:]

    # Find probes that activate top orthogonal features
    targeted_probe_indices = set()
    for feat_idx, orth_ratio in orthogonal_features[:100]:  # Check top 100 orthogonal features
        if orth_ratio < 0.5:
            break  # Only consider highly orthogonal features

        # Find probes activating this feature (among remaining probes)
        # Encode remaining probes
        encoded_remaining = sae.encode(remaining_activations)
        mx.eval(encoded_remaining)

        feature_acts = encoded_remaining[:, feat_idx]
        mx.eval(feature_acts)

        # Find top activators
        for i in range(int(remaining_activations.shape[0])):
            if float(feature_acts[i]) > 0.1:  # Significant activation
                targeted_probe_indices.add(initial_probes + i)

        if len(targeted_probe_indices) >= target_probes:
            break

    targeted_probe_indices = sorted(list(targeted_probe_indices))[:target_probes]
    logger.info(f"Found {len(targeted_probe_indices)} probes activating orthogonal features")
    logger.info(f"DEBUG: targeted indices range: {min(targeted_probe_indices)} to {max(targeted_probe_indices)}")

    # Phase 6: Add targeted probes and measure rank increase
    logger.info("=" * 60)
    logger.info("PHASE 6: Measuring rank increase")
    logger.info("=" * 60)

    # DEBUG: Verify activation shapes and consistency
    logger.info(f"DEBUG: all_probe_activations shape: {all_probe_activations.shape}")
    logger.info(f"DEBUG: initial_activations shape: {initial_activations.shape}")
    logger.info(f"DEBUG: targeted_probe_indices: {targeted_probe_indices[:5]}...")

    # Verify initial rank computation is consistent
    initial_rank_check = compute_numerical_rank(all_probe_activations[:initial_probes], backend)
    logger.info(f"DEBUG: initial_rank_check: {initial_rank_check}")

    # Get targeted activations directly from the stacked array
    # MLX requires array indexing, not list
    targeted_idx_array = mx.array(targeted_probe_indices)
    targeted_activations = all_probe_activations[targeted_idx_array]
    mx.eval(targeted_activations)
    logger.info(f"DEBUG: targeted_activations shape: {targeted_activations.shape}")

    targeted_rank_only = compute_numerical_rank(targeted_activations, backend)
    logger.info(f"DEBUG: targeted probes alone have rank: {targeted_rank_only}")

    # Combine using concatenate instead of re-stacking
    combined_activations = mx.concatenate([
        all_probe_activations[:initial_probes],
        targeted_activations
    ], axis=0)
    mx.eval(combined_activations)
    logger.info(f"DEBUG: combined_activations shape: {combined_activations.shape}")

    final_rank = compute_numerical_rank(combined_activations, backend)
    final_coverage = final_rank / hidden_dim
    final_n_probes = initial_probes + len(targeted_probe_indices)

    rank_increase = final_rank - initial_rank
    probes_added = len(targeted_probe_indices)
    efficiency = rank_increase / probes_added if probes_added > 0 else 0.0

    logger.info(f"Initial: {initial_rank}/{hidden_dim} ({100*initial_coverage:.1f}%) with {initial_probes} probes")
    logger.info(f"Final: {final_rank}/{hidden_dim} ({100*final_coverage:.1f}%) with {final_n_probes} probes")
    logger.info(f"Rank increase: +{rank_increase} from {probes_added} targeted probes")
    logger.info(f"Efficiency: {efficiency:.2f} rank per probe")

    # Phase 7: Compare to random probe selection
    logger.info("=" * 60)
    logger.info("PHASE 7: Comparison to random selection")
    logger.info("=" * 60)

    # Add same number of random probes (not targeting orthogonal features)
    import random
    random.seed(42)
    random_indices = random.sample(range(initial_probes, int(all_probe_activations.shape[0])), probes_added)
    random_idx_array = mx.array(random_indices)
    random_targeted = all_probe_activations[random_idx_array]
    random_activations = mx.concatenate([
        all_probe_activations[:initial_probes],
        random_targeted
    ], axis=0)
    mx.eval(random_activations)

    random_rank = compute_numerical_rank(random_activations, backend)
    random_increase = random_rank - initial_rank

    logger.info(f"Random selection: rank {random_rank} (change: {random_increase:+d}) from {probes_added} probes")
    logger.info(f"Targeted selection: rank {final_rank} (change: {rank_increase:+d}) from {probes_added} probes")

    if rank_increase > random_increase:
        improvement_factor = rank_increase / random_increase if random_increase > 0 else float('inf')
        logger.info(f"✓ VALIDATED: Targeting orthogonal features is {improvement_factor:.1f}x more effective")
    else:
        logger.info("? Targeting did not outperform random selection")

    # Build result
    result = ExperimentResult(
        model_path=model_path,
        layer_idx=layer_idx,
        hidden_dim=hidden_dim,
        initial_n_probes=initial_probes,
        initial_rank=initial_rank,
        initial_coverage=initial_coverage,
        final_n_probes=final_n_probes,
        final_rank=final_rank,
        final_coverage=final_coverage,
        rank_increase=rank_increase,
        probes_added=probes_added,
        efficiency=efficiency,
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
        description="Experiment 13: Targeted Probe Generation"
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
        help="Layer index to analyze",
    )
    parser.add_argument(
        "--initial-probes",
        type=int,
        default=100,
        help="Number of initial probes for baseline",
    )
    parser.add_argument(
        "--target-probes",
        type=int,
        default=50,
        help="Number of targeted probes to add",
    )
    parser.add_argument(
        "--sae-epochs",
        type=int,
        default=10,
        help="SAE training epochs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/validation_protocol/exp13_targeted_probe_generation/results.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    run_experiment(
        model_path=args.model,
        layer_idx=args.layer,
        initial_probes=args.initial_probes,
        target_probes=args.target_probes,
        sae_epochs=args.sae_epochs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
