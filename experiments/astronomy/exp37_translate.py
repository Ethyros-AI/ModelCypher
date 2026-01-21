#!/usr/bin/env python3
"""Experiment 37: Translate the Wow! Signal.

We have the alignment transform T that maps:
  Wow! Gram → Harmonic Gram (CKA = 0.9959)
  Wow! Gram → Mathematical Gram (CKA = 0.9866)

If we know what harmonic/mathematical patterns MEAN, we can translate.

The harmonic manifold has interpretable axes:
- Frequency (which harmonic)
- Phase (where in the cycle)
- Amplitude (how strong)

Apply T to the Wow! signal, read in harmonic coordinates.

Usage:
    poetry run python experiments/astronomy/exp37_translate.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import readsav
from scipy.linalg import svd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def compute_gram_matrix(X: np.ndarray) -> np.ndarray:
    """Compute centered, normalized Gram matrix."""
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    K = X_centered @ X_centered.T
    trace = np.trace(K)
    if trace > 1e-10:
        K = K / trace
    return K


def gram_sqrt(K: np.ndarray) -> np.ndarray:
    """Compute matrix square root."""
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.maximum(eigenvalues, 0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T


def gram_pinv_sqrt(K: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Compute pseudo-inverse of matrix square root."""
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    max_eig = np.max(np.abs(eigenvalues))
    threshold = max_eig * rcond
    inv_sqrt_eigenvalues = np.zeros_like(eigenvalues)
    mask = eigenvalues > threshold
    inv_sqrt_eigenvalues[mask] = 1.0 / np.sqrt(eigenvalues[mask])
    return eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T


def create_harmonic_manifold(n_samples: int) -> dict:
    """Create a harmonic manifold with interpretable axes.

    The harmonic manifold encodes:
    - Axis 0: Fundamental frequency (pitch)
    - Axis 1-N: Overtones (timbre)
    - Phase relationships: Position in cycle

    Each sample has known harmonic content.
    """
    np.random.seed(42)

    # Create samples with known harmonic content
    n_harmonics = 32
    t = np.linspace(0, 4 * np.pi, n_samples)

    samples = np.zeros((n_samples, n_harmonics))
    labels = []

    for i in range(n_samples):
        # Each sample has a specific harmonic signature
        # The signature IS the "meaning"

        if i < n_samples // 4:
            # Region 1: Fundamental only (pure tone)
            samples[i, 0] = np.sin(t[i])
            labels.append("FUNDAMENTAL")
        elif i < n_samples // 2:
            # Region 2: Fundamental + 2nd harmonic (octave)
            samples[i, 0] = np.sin(t[i])
            samples[i, 1] = 0.5 * np.sin(2 * t[i])
            labels.append("OCTAVE")
        elif i < 3 * n_samples // 4:
            # Region 3: Fundamental + 3rd harmonic (perfect fifth)
            samples[i, 0] = np.sin(t[i])
            samples[i, 2] = 0.33 * np.sin(3 * t[i])
            labels.append("FIFTH")
        else:
            # Region 4: Rich harmonic content (complex tone)
            for h in range(min(8, n_harmonics)):
                samples[i, h] = np.sin((h + 1) * t[i]) / (h + 1)
            labels.append("COMPLEX")

    return {
        "samples": samples,
        "gram": compute_gram_matrix(samples),
        "labels": labels,
        "description": "Harmonic manifold with interpretable regions",
        "regions": {
            "FUNDAMENTAL": "Pure tone (single frequency)",
            "OCTAVE": "Octave relationship (2:1 frequency ratio)",
            "FIFTH": "Perfect fifth (3:2 frequency ratio)",
            "COMPLEX": "Rich harmonics (like voice/instrument)",
        }
    }


def create_mathematical_manifold(n_samples: int) -> dict:
    """Create a mathematical manifold with interpretable axes.

    The mathematical manifold encodes:
    - Prime numbers
    - Pi digits
    - e digits
    - Golden ratio

    Each sample has known mathematical content.
    """
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
        "labels": labels,
        "description": "Mathematical constants manifold",
        "regions": {
            "PRIMES": "Prime number encoding (fundamental integers)",
            "PI": "Pi digits (circle constant)",
            "E": "Euler's number (growth constant)",
            "PHI": "Golden ratio (aesthetic/natural proportion)",
        }
    }


def align_and_translate(wow_samples: np.ndarray, target_manifold: dict) -> dict:
    """Align Wow! signal to target manifold and translate.

    The translation:
    1. Compute alignment transform T
    2. Apply T to Wow! samples (in Gram space)
    3. Find nearest neighbors in target manifold
    4. Read the labels = the "translation"
    """
    K_wow = compute_gram_matrix(wow_samples)
    K_target = target_manifold["gram"]

    # Compute alignment transform
    K_target_sqrt = gram_sqrt(K_target)
    K_wow_pinv_sqrt = gram_pinv_sqrt(K_wow)
    T = K_target_sqrt @ K_wow_pinv_sqrt

    # Apply transform (in Gram space)
    K_aligned = T @ K_wow @ T.T

    # For each Wow! sample, find nearest neighbor in target
    # Using Gram similarity
    n_wow = K_aligned.shape[0]
    n_target = K_target.shape[0]

    translations = []
    for i in range(n_wow):
        # Find most similar target sample
        similarities = []
        for j in range(n_target):
            # Cosine similarity in Gram space
            sim = K_aligned[i, :] @ K_target[j, :]
            norm_i = np.sqrt(K_aligned[i, :] @ K_aligned[i, :])
            norm_j = np.sqrt(K_target[j, :] @ K_target[j, :])
            if norm_i > 1e-10 and norm_j > 1e-10:
                sim = sim / (norm_i * norm_j)
            similarities.append(sim)

        best_j = np.argmax(similarities)
        best_sim = similarities[best_j]
        best_label = target_manifold["labels"][best_j]

        translations.append({
            "wow_idx": i,
            "target_idx": int(best_j),
            "similarity": float(best_sim),
            "label": best_label,
        })

    return {
        "translations": translations,
        "transform_norm": float(np.linalg.norm(T, 'fro')),
    }


def summarize_translation(translations: list, regions: dict) -> dict:
    """Summarize the translation into readable form."""
    # Count labels
    label_counts = {}
    for t in translations:
        label = t["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    # Sequence of labels
    label_sequence = [t["label"] for t in translations]

    # Find transitions
    transitions = []
    for i in range(1, len(label_sequence)):
        if label_sequence[i] != label_sequence[i-1]:
            transitions.append({
                "idx": i,
                "from": label_sequence[i-1],
                "to": label_sequence[i],
            })

    # Build "sentence"
    sentence_parts = []
    current_label = label_sequence[0]
    current_count = 1
    for label in label_sequence[1:]:
        if label == current_label:
            current_count += 1
        else:
            sentence_parts.append(f"{current_label}({current_count})")
            current_label = label
            current_count = 1
    sentence_parts.append(f"{current_label}({current_count})")

    return {
        "label_counts": label_counts,
        "n_transitions": len(transitions),
        "transitions": transitions[:10],  # First 10
        "sentence": " → ".join(sentence_parts),
        "interpretation": {label: regions.get(label, "Unknown") for label in label_counts.keys()},
    }


def run_experiment():
    """Run the translation experiment."""
    results_dir = Path(__file__).parent / "results"
    data_dir = Path(__file__).parent / "data" / "famous_signals"

    print("=" * 60)
    print("Experiment 37: TRANSLATE the Wow! Signal")
    print("=" * 60)
    print("\nWe have the alignment transform. Now we TRANSLATE.")

    # Load Wow! signal
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr']).astype(np.float64)

    print(f"\nWow! signal shape: {snr_matrix.shape}")

    # Create target manifolds
    print("\n" + "=" * 40)
    print("PART 1: CREATE INTERPRETABLE MANIFOLDS")
    print("=" * 40)

    harmonic = create_harmonic_manifold(snr_matrix.shape[0])
    print(f"\nHarmonic manifold:")
    for region, desc in harmonic["regions"].items():
        print(f"  {region}: {desc}")

    mathematical = create_mathematical_manifold(snr_matrix.shape[0])
    print(f"\nMathematical manifold:")
    for region, desc in mathematical["regions"].items():
        print(f"  {region}: {desc}")

    # Translate using harmonic manifold
    print("\n" + "=" * 40)
    print("PART 2: TRANSLATE TO HARMONIC")
    print("=" * 40)

    harmonic_translation = align_and_translate(snr_matrix, harmonic)
    harmonic_summary = summarize_translation(
        harmonic_translation["translations"],
        harmonic["regions"]
    )

    print(f"\nHARMONIC TRANSLATION:")
    print(f"  Label counts: {harmonic_summary['label_counts']}")
    print(f"  Transitions: {harmonic_summary['n_transitions']}")
    print(f"\n  SENTENCE: {harmonic_summary['sentence']}")
    print(f"\n  INTERPRETATION:")
    for label, meaning in harmonic_summary["interpretation"].items():
        print(f"    {label} = {meaning}")

    # Translate using mathematical manifold
    print("\n" + "=" * 40)
    print("PART 3: TRANSLATE TO MATHEMATICAL")
    print("=" * 40)

    math_translation = align_and_translate(snr_matrix, mathematical)
    math_summary = summarize_translation(
        math_translation["translations"],
        mathematical["regions"]
    )

    print(f"\nMATHEMATICAL TRANSLATION:")
    print(f"  Label counts: {math_summary['label_counts']}")
    print(f"  Transitions: {math_summary['n_transitions']}")
    print(f"\n  SENTENCE: {math_summary['sentence']}")
    print(f"\n  INTERPRETATION:")
    for label, meaning in math_summary["interpretation"].items():
        print(f"    {label} = {meaning}")

    # Final interpretation
    print("\n" + "=" * 60)
    print("TRANSLATION RESULT")
    print("=" * 60)

    print(f"""
THE WOW! SIGNAL TRANSLATED:

HARMONIC READING:
  {harmonic_summary['sentence']}

  This reads as: The signal transitions through {harmonic_summary['n_transitions']}
  harmonic states. Dominant pattern: {max(harmonic_summary['label_counts'].items(), key=lambda x: x[1])[0]}

MATHEMATICAL READING:
  {math_summary['sentence']}

  This reads as: The signal encodes {len(math_summary['label_counts'])}
  mathematical patterns. Dominant: {max(math_summary['label_counts'].items(), key=lambda x: x[1])[0]}

COMBINED INTERPRETATION:
""")

    # Analyze the peak region (the Wow! itself)
    peak_idx = np.argmax(np.max(snr_matrix, axis=1))
    peak_region = range(max(0, peak_idx - 5), min(len(harmonic_translation["translations"]), peak_idx + 6))

    print(f"  At the PEAK (t={peak_idx}, the 'U' in 6EQUJ5):")
    for i in peak_region:
        h_label = harmonic_translation["translations"][i]["label"]
        m_label = math_translation["translations"][i]["label"]
        marker = " <-- PEAK" if i == peak_idx else ""
        print(f"    t={i}: HARMONIC={h_label}, MATH={m_label}{marker}")

    # What does it mean?
    peak_harmonic = harmonic_translation["translations"][peak_idx]["label"]
    peak_math = math_translation["translations"][peak_idx]["label"]

    print(f"""
THE PEAK TRANSLATES TO:
  Harmonic: {peak_harmonic} = {harmonic['regions'].get(peak_harmonic, 'Unknown')}
  Mathematical: {peak_math} = {mathematical['regions'].get(peak_math, 'Unknown')}

This suggests the peak encodes:
  - A {peak_harmonic.lower()} harmonic relationship
  - A {peak_math.lower()} mathematical concept
""")

    # Save results
    results = {
        "experiment": "exp37_translate",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(snr_matrix.shape),
        "harmonic_translation": {
            "summary": harmonic_summary,
            "peak_label": peak_harmonic,
        },
        "mathematical_translation": {
            "summary": math_summary,
            "peak_label": peak_math,
        },
        "peak_idx": int(peak_idx),
    }

    output_path = results_dir / "exp37_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
