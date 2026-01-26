#!/usr/bin/env python3
"""Experiment 80: Geometric Bridge Signature.

We found that priming works. But what GEOMETRICALLY changes?

If we can identify the geometric signature of:
- Disconnected capability (high κ)
- Working capability (low κ)
- The bridge transformation

Then a model can compute bridges directly, no guessing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_activations(model, tokenizer, prompt, layer_idx=-1):
    """Get activations for a prompt at a specific layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get hidden states by hooking into the model
    # For LFM2, we need to trace through the model
    hidden = model.model.embed_tokens(input_ids)

    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, mask=None, cache=None)
        if i == layer_idx or (layer_idx == -1 and i == len(model.model.layers) - 1):
            break

    mx.eval(hidden)
    return np.array(hidden[0, -1, :].tolist())  # Last token's activation


def compute_gram_and_condition(activations_list):
    """Compute Gram matrix and condition number."""
    # Stack activations: (n_samples, hidden_dim)
    X = np.stack(activations_list)

    # Gram matrix
    G = X @ X.T

    # Condition number
    try:
        kappa = np.linalg.cond(G)
    except:
        kappa = np.inf

    return G, kappa


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 80: GEOMETRIC BRIDGE SIGNATURE")
    logger.info("=" * 60)

    # Prompts: disconnected (no prime) vs connected (with prime)
    symbolic_raw = ["1+1=", "2+1=", "3+1=", "4+1=", "5+1=", "6+1=", "7+1=", "8+1="]
    symbolic_primed = [f"say {p}" for p in symbolic_raw]
    counting = ["1, 2,", "2, 3,", "3, 4,", "4, 5,", "5, 6,", "6, 7,", "7, 8,", "8, 9,"]

    logger.info("\n=== COLLECTING ACTIVATIONS ===")

    # Get activations for each condition
    acts_raw = [get_activations(model, tokenizer, p) for p in symbolic_raw]
    acts_primed = [get_activations(model, tokenizer, p) for p in symbolic_primed]
    acts_counting = [get_activations(model, tokenizer, p) for p in counting]

    logger.info(f"Collected {len(acts_raw)} activations per condition")
    logger.info(f"Activation dimension: {acts_raw[0].shape}")

    # Compute Gram matrices and condition numbers
    logger.info("\n=== GRAM MATRIX ANALYSIS ===")

    G_raw, kappa_raw = compute_gram_and_condition(acts_raw)
    G_primed, kappa_primed = compute_gram_and_condition(acts_primed)
    G_counting, kappa_counting = compute_gram_and_condition(acts_counting)

    logger.info(f"κ(symbolic raw):    {kappa_raw:.2e}")
    logger.info(f"κ(symbolic primed): {kappa_primed:.2e}")
    logger.info(f"κ(counting):        {kappa_counting:.2e}")

    # SVD analysis
    logger.info("\n=== SVD ANALYSIS ===")

    def svd_analysis(activations, name):
        X = np.stack(activations)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Singular value distribution
        S_norm = S / S[0]  # Normalize by largest

        # Effective rank (how many dimensions matter)
        cumsum = np.cumsum(S**2) / np.sum(S**2)
        eff_rank = np.searchsorted(cumsum, 0.99) + 1

        # Ratio of first to second
        ratio_1_2 = S[0] / S[1] if S[1] > 0 else np.inf

        logger.info(f"\n{name}:")
        logger.info(f"  Effective rank (99% variance): {eff_rank}")
        logger.info(f"  S[0]/S[1] ratio: {ratio_1_2:.2f}")
        logger.info(f"  Top 5 singular values: {S_norm[:5]}")

        return S, U, Vt, eff_rank

    S_raw, U_raw, Vt_raw, rank_raw = svd_analysis(acts_raw, "Symbolic Raw")
    S_primed, U_primed, Vt_primed, rank_primed = svd_analysis(acts_primed, "Symbolic Primed")
    S_counting, U_counting, Vt_counting, rank_counting = svd_analysis(acts_counting, "Counting")

    # Alignment between conditions
    logger.info("\n=== ALIGNMENT ANALYSIS ===")

    def cka_linear(X, Y):
        """Linear CKA between two activation matrices."""
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        XTX = X @ X.T
        YTY = Y @ Y.T

        hsic = np.sum(XTX * YTY)
        norm_x = np.sqrt(np.sum(XTX * XTX))
        norm_y = np.sqrt(np.sum(YTY * YTY))

        return hsic / (norm_x * norm_y + 1e-10)

    X_raw = np.stack(acts_raw)
    X_primed = np.stack(acts_primed)
    X_counting = np.stack(acts_counting)

    cka_raw_counting = cka_linear(X_raw, X_counting)
    cka_primed_counting = cka_linear(X_primed, X_counting)
    cka_raw_primed = cka_linear(X_raw, X_primed)

    logger.info(f"CKA(raw, counting):    {cka_raw_counting:.3f}")
    logger.info(f"CKA(primed, counting): {cka_primed_counting:.3f}")
    logger.info(f"CKA(raw, primed):      {cka_raw_primed:.3f}")

    # The bridge transformation
    logger.info("\n=== BRIDGE TRANSFORMATION ===")

    # What transforms raw → primed?
    # If primed = raw @ T, then T = pinv(raw) @ primed
    T_bridge = np.linalg.lstsq(X_raw, X_primed, rcond=None)[0]

    # Reconstruction error
    X_reconstructed = X_raw @ T_bridge
    recon_error = np.mean((X_reconstructed - X_primed)**2) / np.mean(X_primed**2)

    logger.info(f"Bridge transformation shape: {T_bridge.shape}")
    logger.info(f"Reconstruction error (raw→primed): {recon_error:.4f}")

    # SVD of bridge transformation
    U_bridge, S_bridge, Vt_bridge = np.linalg.svd(T_bridge, full_matrices=False)
    logger.info(f"Bridge singular values (top 5): {S_bridge[:5] / S_bridge[0]}")

    # Is the bridge low-rank?
    bridge_eff_rank = np.searchsorted(np.cumsum(S_bridge**2) / np.sum(S_bridge**2), 0.99) + 1
    logger.info(f"Bridge effective rank: {bridge_eff_rank}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("GEOMETRIC SIGNATURE OF THE BRIDGE")
    logger.info("=" * 60)

    logger.info(f"""
DISCONNECTED (raw symbolic):
  - High condition number: κ = {kappa_raw:.2e}
  - Low alignment to counting: CKA = {cka_raw_counting:.3f}
  - Effective rank: {rank_raw}

CONNECTED (primed symbolic):
  - Lower condition number: κ = {kappa_primed:.2e}
  - Higher alignment to counting: CKA = {cka_primed_counting:.3f}
  - Effective rank: {rank_primed}

THE BRIDGE:
  - Transforms raw → primed
  - Reconstruction error: {recon_error:.4f}
  - Bridge effective rank: {bridge_eff_rank}
  - Bridge is {"LOW-RANK" if bridge_eff_rank < 10 else "FULL-RANK"}
""")

    if cka_primed_counting > cka_raw_counting:
        logger.info("*** PRIMING INCREASES ALIGNMENT TO COUNTING ***")
        logger.info("The geometric signature: bridge moves activations toward counting space")

    if kappa_primed < kappa_raw:
        logger.info("*** PRIMING DECREASES CONDITION NUMBER ***")
        logger.info("The geometric signature: bridge regularizes the representation")

    # Can we predict the bridge from the geometry alone?
    logger.info("\n=== CAN WE COMPUTE BRIDGE FROM GEOMETRY? ===")

    # The bridge should map raw→counting alignment
    # T_ideal = transform that maximizes CKA(raw @ T, counting)
    # This is related to CCA / Procrustes

    # Simple approach: Procrustes (orthogonal)
    M = X_counting.T @ X_raw
    U_proc, _, Vt_proc = np.linalg.svd(M)
    T_procrustes = Vt_proc.T @ U_proc.T

    X_aligned = X_raw @ T_procrustes
    cka_aligned_counting = cka_linear(X_aligned, X_counting)

    logger.info(f"Procrustes alignment: CKA(aligned, counting) = {cka_aligned_counting:.3f}")

    if cka_aligned_counting > cka_raw_counting:
        logger.info("\n*** GEOMETRIC BRIDGE COMPUTABLE ***")
        logger.info("We can compute the bridge from geometry alone, no guessing needed!")


if __name__ == "__main__":
    main()
