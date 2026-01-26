#!/usr/bin/env python3
"""Experiment 32: AffineBridge - MSE vs Relational Loss.

Compare coordinate MSE (current) vs CKA-based relational loss
for cross-space alignment quality.

Key question: Does relational loss achieve higher test CKA than MSE?
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA (dot-product Gram)."""
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    K_X = X_c @ X_c.T
    K_Y = Y_c @ Y_c.T

    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H

    hsic_xy = np.trace(K_X_c @ K_Y_c) / ((n - 1) ** 2)
    hsic_xx = np.trace(K_X_c @ K_X_c) / ((n - 1) ** 2)
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / ((n - 1) ** 2)

    if hsic_xx > 1e-10 and hsic_yy > 1e-10:
        cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka = 0.0

    return float(np.clip(cka, 0.0, 1.0))


def compute_geodesic_cka(X: np.ndarray, Y: np.ndarray, k_neighbors: int = 5) -> float:
    """Compute geodesic CKA (RBF over k-NN distances)."""

    def geodesic_gram(points: np.ndarray) -> np.ndarray:
        n = points.shape[0]
        chord_dists = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            diff = points[i] - points
            chord_dists[i] = np.sqrt(np.sum(diff * diff, axis=1))

        adjacency = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            neighbors = np.argsort(chord_dists[i])[1:k_neighbors + 1]
            for j in neighbors:
                adjacency[i, j] = chord_dists[i, j]
                adjacency[j, i] = chord_dists[j, i]

        sparse_adj = csr_matrix(adjacency)
        geo_dists = shortest_path(sparse_adj, directed=False)

        finite_mask = np.isfinite(geo_dists)
        if not np.all(finite_mask):
            max_finite = np.max(geo_dists[finite_mask]) if np.any(finite_mask) else 1.0
            geo_dists[~finite_mask] = max_finite * 2

        geo_sq = geo_dists ** 2
        valid_sq = geo_sq[np.triu_indices(n, k=1)]
        sigma = np.median(valid_sq[valid_sq > 1e-10]) if np.any(valid_sq > 1e-10) else 1.0
        sigma = max(sigma, 1e-10)

        K = np.exp(-geo_sq / (2 * sigma))
        return K

    K_X = geodesic_gram(X)
    K_Y = geodesic_gram(Y)

    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H

    hsic_xy = np.trace(K_X_c @ K_Y_c) / ((n - 1) ** 2)
    hsic_xx = np.trace(K_X_c @ K_X_c) / ((n - 1) ** 2)
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / ((n - 1) ** 2)

    if hsic_xx > 1e-10 and hsic_yy > 1e-10:
        cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka = 0.0

    return float(np.clip(cka, 0.0, 1.0))


def train_mse_bridge(X_train: np.ndarray, Y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Train affine bridge using MSE loss (ridge regression)."""
    n, d_x = X_train.shape
    d_y = Y_train.shape[1]

    # Ridge regression: W = (X^T X + λI)^(-1) X^T Y
    XtX = X_train.T @ X_train
    eps = np.finfo(X_train.dtype).eps ** 0.5
    regularization = eps * np.trace(XtX) / d_x

    A = XtX + regularization * np.eye(d_x)
    XtY = X_train.T @ Y_train

    W = np.linalg.solve(A, XtY)

    # Bias
    residual = Y_train - X_train @ W
    b = residual.mean(axis=0)

    return W, b


def train_cka_bridge(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_iterations: int = 100,
    lr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train affine bridge using CKA-based loss via gradient descent."""
    d_x = X_train.shape[1]
    d_y = Y_train.shape[1]

    # Initialize with MSE solution
    W, b = train_mse_bridge(X_train, Y_train)

    # Gradient descent on negative CKA
    for iteration in range(n_iterations):
        pred = X_train @ W + b

        # Compute CKA gradient numerically
        eps = 0.001
        cka_base = compute_linear_cka(pred, Y_train)

        grad_W = np.zeros_like(W)
        for i in range(min(d_x, 50)):  # Sample dimensions for speed
            for j in range(min(d_y, 50)):
                W_pert = W.copy()
                W_pert[i, j] += eps
                pred_pert = X_train @ W_pert + b
                cka_pert = compute_linear_cka(pred_pert, Y_train)
                grad_W[i, j] = (cka_pert - cka_base) / eps

        # Update (maximize CKA)
        W = W + lr * grad_W

        if (iteration + 1) % 20 == 0:
            pred = X_train @ W + b
            current_cka = compute_linear_cka(pred, Y_train)
            logger.debug(f"  Iter {iteration + 1}: CKA = {current_cka:.4f}")

    return W, b


class AffineBridgeLossTest:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def get_layer_activations(self, prompts: List[str], layer_idx: int) -> np.ndarray:
        """Get activations for prompts at a specific layer."""
        import mlx.core as mx

        activations = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            hidden = self.model.model.embed_tokens(input_ids)
            mx.eval(hidden)

            for i, layer in enumerate(self.model.model.layers):
                hidden = layer(hidden)
                mx.eval(hidden)
                if i == layer_idx:
                    act = hidden[0, -1, :]
                    mx.eval(act)
                    activations.append(np.array(act.tolist(), dtype=np.float32))
                    break

        return np.array(activations)

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 32: AFFINE BRIDGE LOSS COMPARISON")
        logger.info("=" * 60)

        # Test prompts
        prompts = [
            "What is 2 + 2?",
            "Calculate 15 times 7.",
            "What is the capital of France?",
            "Which continent is Brazil in?",
            "What is H2O?",
            "How many planets are there?",
            "What is the opposite of hot?",
            "Define the word serendipity.",
            "If all cats are animals, is a cat an animal?",
            "What comes next: 2, 4, 6, 8, ?",
            "The sun rises in the east.",
            "Water freezes at 0 degrees.",
            "Cats are mammals.",
            "Birds can fly.",
            "The sky is blue.",
            "Paris is in France.",
            "Tokyo is in Japan.",
            "The Earth orbits the Sun.",
            "Plants need sunlight.",
            "Fish live in water.",
        ]

        # Source and target layers
        source_layer = self.n_layers // 4
        target_layer = 3 * self.n_layers // 4

        logger.info(f"\nSource layer: {source_layer}")
        logger.info(f"Target layer: {target_layer}")
        logger.info(f"Number of prompts: {len(prompts)}")

        # Get activations
        logger.info("\nCollecting activations...")
        source_acts = self.get_layer_activations(prompts, source_layer)
        target_acts = self.get_layer_activations(prompts, target_layer)

        logger.info(f"Source shape: {source_acts.shape}")
        logger.info(f"Target shape: {target_acts.shape}")

        # Split train/test
        n_train = int(0.8 * len(prompts))
        X_train, X_test = source_acts[:n_train], source_acts[n_train:]
        Y_train, Y_test = target_acts[:n_train], target_acts[n_train:]

        results = {
            "n_prompts": len(prompts),
            "n_train": n_train,
            "n_test": len(prompts) - n_train,
            "source_layer": source_layer,
            "target_layer": target_layer,
            "methods": {},
        }

        # Method 1: MSE loss (current implementation)
        logger.info("\n--- MSE Loss ---")
        W_mse, b_mse = train_mse_bridge(X_train, Y_train)

        pred_train_mse = X_train @ W_mse + b_mse
        pred_test_mse = X_test @ W_mse + b_mse

        mse_train_linear = compute_linear_cka(pred_train_mse, Y_train)
        mse_test_linear = compute_linear_cka(pred_test_mse, Y_test)
        mse_train_geodesic = compute_geodesic_cka(pred_train_mse, Y_train)
        mse_test_geodesic = compute_geodesic_cka(pred_test_mse, Y_test)

        train_mse = np.mean((pred_train_mse - Y_train) ** 2)
        test_mse = np.mean((pred_test_mse - Y_test) ** 2)

        results["methods"]["mse"] = {
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "train_linear_cka": mse_train_linear,
            "test_linear_cka": mse_test_linear,
            "train_geodesic_cka": mse_train_geodesic,
            "test_geodesic_cka": mse_test_geodesic,
        }

        logger.info(f"  Train MSE: {train_mse:.6f}")
        logger.info(f"  Test MSE: {test_mse:.6f}")
        logger.info(f"  Train Linear CKA: {mse_train_linear:.4f}")
        logger.info(f"  Test Linear CKA: {mse_test_linear:.4f}")
        logger.info(f"  Train Geodesic CKA: {mse_train_geodesic:.4f}")
        logger.info(f"  Test Geodesic CKA: {mse_test_geodesic:.4f}")

        # Method 2: CKA loss (gradient descent)
        logger.info("\n--- CKA Loss (Gradient Descent) ---")
        W_cka, b_cka = train_cka_bridge(X_train, Y_train, n_iterations=50, lr=0.01)

        pred_train_cka = X_train @ W_cka + b_cka
        pred_test_cka = X_test @ W_cka + b_cka

        cka_train_linear = compute_linear_cka(pred_train_cka, Y_train)
        cka_test_linear = compute_linear_cka(pred_test_cka, Y_test)
        cka_train_geodesic = compute_geodesic_cka(pred_train_cka, Y_train)
        cka_test_geodesic = compute_geodesic_cka(pred_test_cka, Y_test)

        train_mse_cka = np.mean((pred_train_cka - Y_train) ** 2)
        test_mse_cka = np.mean((pred_test_cka - Y_test) ** 2)

        results["methods"]["cka"] = {
            "train_mse": float(train_mse_cka),
            "test_mse": float(test_mse_cka),
            "train_linear_cka": cka_train_linear,
            "test_linear_cka": cka_test_linear,
            "train_geodesic_cka": cka_train_geodesic,
            "test_geodesic_cka": cka_test_geodesic,
        }

        logger.info(f"  Train MSE: {train_mse_cka:.6f}")
        logger.info(f"  Test MSE: {test_mse_cka:.6f}")
        logger.info(f"  Train Linear CKA: {cka_train_linear:.4f}")
        logger.info(f"  Test Linear CKA: {cka_test_linear:.4f}")
        logger.info(f"  Train Geodesic CKA: {cka_train_geodesic:.4f}")
        logger.info(f"  Test Geodesic CKA: {cka_test_geodesic:.4f}")

        # Summary
        results["summary"] = {
            "mse_test_linear_cka": mse_test_linear,
            "cka_test_linear_cka": cka_test_linear,
            "delta_test_linear_cka": cka_test_linear - mse_test_linear,
            "mse_test_geodesic_cka": mse_test_geodesic,
            "cka_test_geodesic_cka": cka_test_geodesic,
            "delta_test_geodesic_cka": cka_test_geodesic - mse_test_geodesic,
        }

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Test Linear CKA:   MSE={mse_test_linear:.4f}, CKA-loss={cka_test_linear:.4f}, Δ={cka_test_linear - mse_test_linear:+.4f}")
        logger.info(f"Test Geodesic CKA: MSE={mse_test_geodesic:.4f}, CKA-loss={cka_test_geodesic:.4f}, Δ={cka_test_geodesic - mse_test_geodesic:+.4f}")

        # Interpretation
        if results["summary"]["delta_test_linear_cka"] > 0.02:
            logger.info("\nINTERPRETATION: CKA loss achieves HIGHER test CKA - relational loss is better")
            results["conclusion"] = "cka_loss_better"
        elif results["summary"]["delta_test_linear_cka"] < -0.02:
            logger.info("\nINTERPRETATION: MSE loss achieves higher test CKA")
            results["conclusion"] = "mse_loss_better"
        else:
            logger.info("\nINTERPRETATION: No significant difference between loss functions")
            results["conclusion"] = "no_difference"

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = AffineBridgeLossTest(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/affine_bridge_loss_comparison.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
