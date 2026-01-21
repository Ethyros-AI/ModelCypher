#!/usr/bin/env python3
"""Experiment 40: Translation Control.

The Red Team test (Exp 39) showed that the high CKA score is an artifact of low rank.
Random matrices align just as well as the Wow! signal.

However, do random matrices TRANSLATE to the same message?
"Primes -> Pi -> E -> Phi"

Hypothesis:
- If random surrogates also translate to "Primes -> Pi -> E", then the "message"
  is a property of the Target Manifold (e.g., "Primes" is just the mean/center).
- If random surrogates translate to random noise, but Wow! translates to Math,
  then the TRAJECTORY is real, even if the CKA score isn't.

Usage:
    poetry run python experiments/astronomy/exp40_translation_control.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import readsav

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Re-use logic from exp37
def compute_gram_matrix(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    K = X_centered @ X_centered.T
    if normalize:
        trace = np.trace(K)
        if trace > 1e-10:
            K = K / trace
    return K

def gram_sqrt(K: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

def gram_pinv_sqrt(K: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    max_eig = np.max(np.abs(eigenvalues))
    threshold = max_eig * rcond
    inv_sqrt_eigenvalues = np.zeros_like(eigenvalues)
    mask = eigenvalues > threshold
    inv_sqrt_eigenvalues[mask] = 1.0 / np.sqrt(eigenvalues[mask])
    return eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T

def create_mathematical_manifold(n_samples: int) -> dict:
    """Re-create the target manifold from Exp 37."""
    np.random.seed(42)
    n_features = 50
    samples = np.zeros((n_samples, n_features))
    labels = []

    pi_str = "314159265358979323846264338327950288419716939937510"
    e_str = "271828182845904523536028747135266249775724709369995"
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    for i in range(n_samples):
        if i < n_samples // 4:
            # Region 1: Prime encoding
            for j, p in enumerate(primes[:n_features]):
                samples[i, j % n_features] = np.sin(i * np.pi / p)
            labels.append("PRIMES")
        elif i < n_samples // 2:
            # Region 2: Pi encoding
            for j in range(min(n_features, len(pi_str))):
                samples[i, j] = int(pi_str[j]) / 9.0 * np.cos(i * 0.1)
            labels.append("PI")
        elif i < 3 * n_samples // 4:
            # Region 3: e encoding
            for j in range(min(n_features, len(e_str))):
                samples[i, j] = int(e_str[j]) / 9.0 * np.cos(i * 0.1)
            labels.append("E")
        else:
            # Region 4: Golden ratio / Fibonacci
            phi = (1 + np.sqrt(5)) / 2
            for j in range(n_features):
                samples[i, j] = np.sin(i * phi * j * 0.1)
            labels.append("PHI")

    return {
        "samples": samples,
        "gram": compute_gram_matrix(samples),
        "labels": labels
    }

def align_and_translate_surrogate(X_source: np.ndarray, target_manifold: dict) -> list[str]:
    K_source = compute_gram_matrix(X_source)
    K_target = target_manifold["gram"]
    
    # Alignment
    K_target_sqrt = gram_sqrt(K_target)
    K_source_pinv_sqrt = gram_pinv_sqrt(K_source)
    T = K_target_sqrt @ K_source_pinv_sqrt
    K_aligned = T @ K_source @ T.T
    
    # Nearest Neighbors
    n_source = K_aligned.shape[0]
    n_target = K_target.shape[0]
    
    translated_labels = []
    
    # Vectorized similarity calculation for speed
    # Cosine similarity matrix: C = K_aligned @ K_target.T / (norms)
    # But K is Gram matrix (n x n). Wait.
    # In Exp 37, we did nearest neighbor IN GRAM SPACE.
    # The "sample" i in aligned space is the i-th ROW of K_aligned.
    
    # Pre-compute norms
    source_norms = np.sqrt(np.diag(K_aligned))
    target_norms = np.sqrt(np.diag(K_target))
    
    # Similarity matrix (n_source x n_target)
    # Sim[i, j] = Row_i(K_aligned) dot Row_j(K_target)
    # This is K_aligned @ K_target.T
    
    Sim = K_aligned @ K_target.T
    
    # Normalize
    Sim = Sim / (source_norms[:, None] @ target_norms[None, :] + 1e-10)
    
    # Best match for each source sample
    best_indices = np.argmax(Sim, axis=1)
    
    for idx in best_indices:
        translated_labels.append(target_manifold["labels"][idx])
        
    return translated_labels

def generate_isospectral_surrogate(X: np.ndarray) -> np.ndarray:
    """Generate random matrix with same spectrum."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rows, cols = X.shape
    rand_U = np.random.randn(rows, len(s))
    Q_U, _ = np.linalg.qr(rand_U)
    rand_V = np.random.randn(len(s), cols)
    Q_V, _ = np.linalg.qr(rand_V.T)
    Q_V = Q_V.T
    X_surrogate = Q_U @ np.diag(s) @ Q_V
    return X_surrogate

def summarize_labels(labels: list[str]) -> str:
    """Compress sequence to summary."""
    # Simple count based summary
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    
    # Find sequence flow
    flow = []
    curr = labels[0]
    count = 1
    for l in labels[1:]:
        if l == curr:
            count += 1
        else:
            flow.append(f"{curr}({count})")
            curr = l
            count = 1
    flow.append(f"{curr}({count})")
    return " -> ".join(flow)

def run_experiment():
    print("=" * 60)
    print("Experiment 40: TRANSLATION CONTROL")
    print("=" * 60)
    
    # Load Wow
    data_dir = Path(__file__).parent / "data" / "famous_signals"
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    snr_matrix = np.array(wow_raw['oseti'][0]['snr']).astype(np.float64)
    
    # Create Target
    math_manifold = create_mathematical_manifold(snr_matrix.shape[0])
    
    # 1. Translate Wow (Baseline)
    print("\n1. BASELINE (Wow! Signal):")
    wow_labels = align_and_translate_surrogate(snr_matrix, math_manifold)
    print(f"Sequence: {summarize_labels(wow_labels)}")
    
    # 2. Translate Surrogates
    print("\n2. CONTROLS (Random Isospectral Matrices):")
    
    same_translation_count = 0
    n_trials = 10
    
    for i in range(n_trials):
        surr = generate_isospectral_surrogate(snr_matrix)
        surr_labels = align_and_translate_surrogate(surr, math_manifold)
        summary = summarize_labels(surr_labels)
        
        print(f"  Surrogate {i+1}: {summary[:80]}...")
        
        # Check if it matches the general Primes->Pi->E->Phi structure
        # (This is a heuristic check)
        if "PRIMES" in summary and "PI" in summary and "E" in summary and "PHI" in summary:
            same_translation_count += 1

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if same_translation_count > n_trials / 2:
        print("FAIL: Random noise produces the SAME translation.")
        print("The 'message' is an artifact of the target manifold structure.")
    else:
        print("PASS?: Random noise produces DIFFERENT translations.")
        print("The Wow! signal's trajectory is unique, even if CKA is generic.")

if __name__ == "__main__":
    run_experiment()
