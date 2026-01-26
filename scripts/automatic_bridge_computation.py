#!/usr/bin/env python3
"""Experiment 82: Automatic Bridge Computation.

Can we compute effective bridges from geometry alone, without trying primes?

The key insight from Phase 9.5:
- Priming creates a transformation in activation space
- This transformation can be computed via Procrustes alignment
- If we have a reference "working" capability, we can compute the bridge

This experiment tests: Given ONLY geometry (no manual prime discovery),
can we compute a bridge that achieves similar accuracy to discovered primes?
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_activations(model, tokenizer, prompt: str, layer_idx: int = -1) -> np.ndarray:
    """Get activations for a prompt at a specific layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)

    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, mask=None, cache=None)
        if i == layer_idx or (layer_idx == -1 and i == len(model.model.layers) - 1):
            break

    mx.eval(hidden)
    return np.array(hidden[0, -1, :].tolist())


def compute_kappa(activations: np.ndarray) -> float:
    """Compute condition number of Gram matrix."""
    G = activations @ activations.T
    try:
        return float(np.linalg.cond(G))
    except:
        return float('inf')


def compute_procrustes(A_source: np.ndarray, A_target: np.ndarray) -> np.ndarray:
    """Compute orthogonal Procrustes alignment T such that A_source @ T ≈ A_target."""
    # Center the data
    A_source_centered = A_source - A_source.mean(axis=0)
    A_target_centered = A_target - A_target.mean(axis=0)

    # SVD of cross-covariance
    M = A_target_centered.T @ A_source_centered
    U, S, Vt = np.linalg.svd(M)

    # Orthogonal transform
    T = Vt.T @ U.T

    return T


def compute_least_squares_bridge(A_source: np.ndarray, A_target: np.ndarray) -> np.ndarray:
    """Compute least squares bridge T such that A_source @ T ≈ A_target."""
    T, residuals, rank, s = np.linalg.lstsq(A_source, A_target, rcond=None)
    return T


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two activation matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XTX = X @ X.T
    YTY = Y @ Y.T

    hsic = np.sum(XTX * YTY)
    norm_x = np.sqrt(np.sum(XTX * XTX))
    norm_y = np.sqrt(np.sum(YTY * YTY))

    return float(hsic / (norm_x * norm_y + 1e-10))


def evaluate_accuracy(model, tokenizer, prime: str, problems: List[Tuple[str, str]]) -> float:
    """Evaluate accuracy on a problem set with optional prime."""
    import mlx.core as mx

    correct = 0
    for problem, expected in problems:
        prompt = f"{prime} {problem}" if prime else problem

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_token = int(np.argmax(probs))
        predicted = tokenizer.decode([top_token]).strip()

        if expected in predicted or predicted == expected:
            correct += 1

    return correct / len(problems) if problems else 0.0


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 82: AUTOMATIC BRIDGE COMPUTATION")
    logger.info("=" * 60)

    # Define capability pairs: disconnected → reference (working)
    # We use counting as reference since we found it has low κ
    # We use arithmetic (high κ, but works with priming) as disconnected

    # Raw arithmetic prompts (disconnected)
    arith_prompts = ["1+1=", "2+1=", "3+1=", "4+1=", "5+1=", "6+1=", "7+1=", "8+1="]
    arith_problems = [("1+1=", "2"), ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"),
                     ("5+1=", "6"), ("6+1=", "7"), ("7+1=", "8"), ("8+1=", "9")]

    # Primed arithmetic prompts (working - what we want to achieve)
    prime = "Arithmetic means calculating numbers."
    arith_primed_prompts = [f"{prime} {p}" for p in arith_prompts]

    # Counting prompts (reference - naturally working)
    counting_prompts = ["1, 2,", "2, 3,", "3, 4,", "4, 5,", "5, 6,", "6, 7,", "7, 8,", "8, 9,"]

    logger.info("\n=== COLLECTING ACTIVATIONS ===")

    # Get activations
    acts_raw = np.stack([get_activations(model, tokenizer, p) for p in arith_prompts])
    acts_primed = np.stack([get_activations(model, tokenizer, p) for p in arith_primed_prompts])
    acts_counting = np.stack([get_activations(model, tokenizer, p) for p in counting_prompts])

    logger.info(f"Raw arithmetic: shape {acts_raw.shape}, κ={compute_kappa(acts_raw):.2e}")
    logger.info(f"Primed arithmetic: shape {acts_primed.shape}, κ={compute_kappa(acts_primed):.2e}")
    logger.info(f"Counting: shape {acts_counting.shape}, κ={compute_kappa(acts_counting):.2e}")

    # Baseline accuracies
    logger.info("\n=== BASELINE ACCURACIES ===")
    acc_raw = evaluate_accuracy(model, tokenizer, "", arith_problems)
    acc_primed = evaluate_accuracy(model, tokenizer, prime, arith_problems)
    logger.info(f"Raw arithmetic: {acc_raw:.0%}")
    logger.info(f"Primed arithmetic: {acc_primed:.0%}")

    # Compute bridges
    logger.info("\n=== COMPUTING BRIDGES ===")

    # Bridge 1: Procrustes from raw → primed (direct)
    T_direct = compute_procrustes(acts_raw, acts_primed)
    acts_bridged_direct = acts_raw @ T_direct

    # Bridge 2: Procrustes from raw → counting (cross-domain)
    T_counting = compute_procrustes(acts_raw, acts_counting)
    acts_bridged_counting = acts_raw @ T_counting

    # Bridge 3: Least squares from raw → primed
    T_lstsq = compute_least_squares_bridge(acts_raw, acts_primed)
    acts_bridged_lstsq = acts_raw @ T_lstsq

    logger.info(f"Direct bridge shape: {T_direct.shape}")
    logger.info(f"Counting bridge shape: {T_counting.shape}")
    logger.info(f"Lstsq bridge shape: {T_lstsq.shape}")

    # Analyze bridges
    logger.info("\n=== BRIDGE ANALYSIS ===")

    def analyze_bridge(T: np.ndarray, name: str):
        U, S, Vt = np.linalg.svd(T, full_matrices=False)

        # Effective rank (99% variance)
        cumsum = np.cumsum(S**2) / np.sum(S**2)
        eff_rank = int(np.searchsorted(cumsum, 0.99) + 1)

        # Frobenius norm
        frob_norm = float(np.linalg.norm(T, 'fro'))

        # Spectral norm (largest singular value)
        spectral_norm = float(S[0])

        logger.info(f"\n{name}:")
        logger.info(f"  Effective rank (99% var): {eff_rank}")
        logger.info(f"  Frobenius norm: {frob_norm:.4f}")
        logger.info(f"  Spectral norm: {spectral_norm:.4f}")
        logger.info(f"  Top 5 singular values: {S[:5] / S[0]}")

        return eff_rank, S

    rank_direct, S_direct = analyze_bridge(T_direct, "Direct (raw→primed)")
    rank_counting, S_counting = analyze_bridge(T_counting, "Counting (raw→counting)")
    rank_lstsq, S_lstsq = analyze_bridge(T_lstsq, "Lstsq (raw→primed)")

    # CKA alignment after bridge
    logger.info("\n=== ALIGNMENT AFTER BRIDGE ===")

    cka_raw_primed = cka_linear(acts_raw, acts_primed)
    cka_bridged_direct = cka_linear(acts_bridged_direct, acts_primed)
    cka_bridged_counting = cka_linear(acts_bridged_counting, acts_counting)
    cka_bridged_lstsq = cka_linear(acts_bridged_lstsq, acts_primed)

    logger.info(f"CKA(raw, primed): {cka_raw_primed:.4f}")
    logger.info(f"CKA(bridged_direct, primed): {cka_bridged_direct:.4f}")
    logger.info(f"CKA(bridged_counting, counting): {cka_bridged_counting:.4f}")
    logger.info(f"CKA(bridged_lstsq, primed): {cka_bridged_lstsq:.4f}")

    # κ after bridge
    logger.info("\n=== κ AFTER BRIDGE ===")

    kappa_raw = compute_kappa(acts_raw)
    kappa_bridged_direct = compute_kappa(acts_bridged_direct)
    kappa_bridged_counting = compute_kappa(acts_bridged_counting)
    kappa_bridged_lstsq = compute_kappa(acts_bridged_lstsq)

    logger.info(f"κ(raw): {kappa_raw:.2e}")
    logger.info(f"κ(bridged_direct): {kappa_bridged_direct:.2e}")
    logger.info(f"κ(bridged_counting): {kappa_bridged_counting:.2e}")
    logger.info(f"κ(bridged_lstsq): {kappa_bridged_lstsq:.2e}")

    # Key question: Can we reconstruct what priming does?
    logger.info("\n=== RECONSTRUCTION QUALITY ===")

    # How close is bridged to primed?
    reconstruction_error_direct = np.mean((acts_bridged_direct - acts_primed)**2) / np.mean(acts_primed**2)
    reconstruction_error_lstsq = np.mean((acts_bridged_lstsq - acts_primed)**2) / np.mean(acts_primed**2)

    logger.info(f"Reconstruction error (Procrustes): {reconstruction_error_direct:.4f}")
    logger.info(f"Reconstruction error (Lstsq): {reconstruction_error_lstsq:.4f}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: CAN WE COMPUTE BRIDGES FROM GEOMETRY?")
    logger.info("=" * 60)

    logger.info(f"""
BASELINE:
  Raw accuracy: {acc_raw:.0%}
  Primed accuracy: {acc_primed:.0%}
  Priming gain: +{acc_primed - acc_raw:.0%}

BRIDGE PROPERTIES:
  Direct bridge (raw→primed):
    - Effective rank: {rank_direct}
    - κ reduction: {kappa_raw:.2e} → {kappa_bridged_direct:.2e}
    - Reconstruction error: {reconstruction_error_direct:.4f}

  Counting bridge (raw→counting):
    - Effective rank: {rank_counting}
    - κ after: {kappa_bridged_counting:.2e}

  Lstsq bridge (raw→primed):
    - Effective rank: {rank_lstsq}
    - Reconstruction error: {reconstruction_error_lstsq:.4f}

KEY FINDINGS:
  - CKA(bridged, target) = {cka_bridged_direct:.4f} (Procrustes)
  - CKA(bridged, target) = {cka_bridged_lstsq:.4f} (Lstsq)
  - Bridge is {"LOW-RANK" if rank_direct < 20 else "FULL-RANK"} (rank {rank_direct})
""")

    if reconstruction_error_lstsq < 0.1:
        logger.info("*** BRIDGE CAN RECONSTRUCT PRIMING TRANSFORMATION ***")
        logger.info("Geometry contains the information needed to compute the bridge!")
    elif reconstruction_error_lstsq < 0.5:
        logger.info("*** BRIDGE PARTIALLY RECONSTRUCTS PRIMING ***")
        logger.info("Some information is captured, but not all.")
    else:
        logger.info("*** BRIDGE DOES NOT RECONSTRUCT PRIMING ***")
        logger.info("The transformation is more complex than linear bridge.")

    if kappa_bridged_direct < kappa_raw * 0.5:
        logger.info(f"\n*** κ REDUCED BY {(1 - kappa_bridged_direct/kappa_raw)*100:.0f}% ***")
    else:
        logger.info(f"\n*** κ NOT SIGNIFICANTLY REDUCED ***")

    # Save results
    output_path = "data/experiments/automatic_bridge_computation.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "baseline": {
            "accuracy_raw": float(acc_raw),
            "accuracy_primed": float(acc_primed),
            "kappa_raw": float(kappa_raw),
        },
        "bridges": {
            "direct_procrustes": {
                "effective_rank": int(rank_direct),
                "reconstruction_error": float(reconstruction_error_direct),
                "kappa_after": float(kappa_bridged_direct),
                "cka_after": float(cka_bridged_direct),
            },
            "counting_procrustes": {
                "effective_rank": int(rank_counting),
                "kappa_after": float(kappa_bridged_counting),
                "cka_after": float(cka_bridged_counting),
            },
            "lstsq": {
                "effective_rank": int(rank_lstsq),
                "reconstruction_error": float(reconstruction_error_lstsq),
                "kappa_after": float(kappa_bridged_lstsq),
                "cka_after": float(cka_bridged_lstsq),
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
