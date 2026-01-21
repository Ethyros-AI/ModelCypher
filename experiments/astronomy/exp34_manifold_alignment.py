#!/usr/bin/env python3
"""Experiment 34: Align to a Known Manifold for Translation.

KEY INSIGHT: If we can align the Wow! signal's geometry to a manifold
we have translation keys for, then translation becomes possible.

We'll create a "semantic manifold" of concepts that any intelligence
would likely encode:
- Numbers (1, 2, 3, ...)
- Mathematical constants (pi, e, phi, ...)
- Prime numbers
- Physical relationships
- Hydrogen line reference

Then use GramAligner to find the rotation that maps the Wow! signal
onto this manifold. The alignment tells us WHICH concepts correspond
to WHICH parts of the signal.

If CKA ≈ 1.0 after alignment, we have a translation key.

Usage:
    poetry run python experiments/astronomy/exp34_manifold_alignment.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import readsav
from scipy.linalg import svd, lstsq

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def create_universal_concept_manifold(n_samples: int = 82) -> dict:
    """Create embeddings for concepts any intelligence would know.

    These are the "Rosetta Stone" concepts - mathematical and physical
    truths that are universal.
    """
    np.random.seed(42)

    concepts = {}

    # 1. INTEGERS: The most basic universal concept
    # Encode as position on a line
    integers = np.arange(1, n_samples + 1)
    concepts["integers"] = {
        "embedding": integers[:, np.newaxis] / n_samples,
        "description": "Natural numbers 1, 2, 3, ...",
        "universal": True,
    }

    # 2. PRIMES: Universal mathematical truth
    primes = []
    for n in range(2, n_samples * 2):
        if all(n % i != 0 for i in range(2, int(np.sqrt(n)) + 1)):
            primes.append(n)
        if len(primes) >= n_samples:
            break
    prime_embed = np.array(primes[:n_samples])[:, np.newaxis] / max(primes[:n_samples])
    concepts["primes"] = {
        "embedding": prime_embed,
        "description": "Prime numbers 2, 3, 5, 7, 11, ...",
        "universal": True,
    }

    # 3. FIBONACCI: Universal growth pattern
    fib = [1, 1]
    while len(fib) < n_samples:
        fib.append(fib[-1] + fib[-2])
    fib_embed = np.array(fib[:n_samples])[:, np.newaxis]
    fib_embed = fib_embed / fib_embed.max()
    concepts["fibonacci"] = {
        "embedding": fib_embed,
        "description": "Fibonacci sequence 1, 1, 2, 3, 5, 8, ...",
        "universal": True,
    }

    # 4. POWERS OF 2: Binary/digital encoding
    powers = []
    for i in range(n_samples):
        powers.append(2 ** (i % 20))  # Cycle to avoid overflow
    powers_embed = np.array(powers)[:, np.newaxis]
    powers_embed = powers_embed / powers_embed.max()
    concepts["powers_of_2"] = {
        "embedding": powers_embed,
        "description": "Powers of 2: 1, 2, 4, 8, 16, ...",
        "universal": True,
    }

    # 5. SINUSOID: Basic wave (physics fundamental)
    t = np.linspace(0, 4 * np.pi, n_samples)
    sin_embed = np.sin(t)[:, np.newaxis]
    sin_embed = (sin_embed + 1) / 2  # Normalize to [0,1]
    concepts["sinusoid"] = {
        "embedding": sin_embed,
        "description": "Sine wave (fundamental physics)",
        "universal": True,
    }

    # 6. GAUSSIAN PULSE: Burst/impulse (like the Wow! signal shape)
    center = n_samples // 2
    sigma = n_samples / 10
    x = np.arange(n_samples)
    gaussian = np.exp(-(x - center)**2 / (2 * sigma**2))
    concepts["gaussian_pulse"] = {
        "embedding": gaussian[:, np.newaxis],
        "description": "Gaussian pulse (burst envelope)",
        "universal": True,
    }

    # 7. HYDROGEN LINE RATIOS: The cosmic reference frequency
    # 1420.405751786 MHz - encode related ratios
    h_freq = 1420.405751786
    h_ratios = np.array([h_freq / (i + 1) for i in range(n_samples)])
    h_ratios = h_ratios / h_ratios.max()
    concepts["hydrogen_ratios"] = {
        "embedding": h_ratios[:, np.newaxis],
        "description": "Hydrogen line frequency ratios",
        "universal": True,
    }

    # 8. MATHEMATICAL CONSTANTS SEQUENCE
    # Pi digits, e digits, etc. interleaved
    pi_str = "31415926535897932384626433832795028841971693993751"
    e_str = "27182818284590452353602874713526624977572470936999"
    constants = []
    for i in range(n_samples):
        if i % 2 == 0 and i // 2 < len(pi_str):
            constants.append(int(pi_str[i // 2]))
        elif i // 2 < len(e_str):
            constants.append(int(e_str[i // 2]))
        else:
            constants.append(0)
    const_embed = np.array(constants)[:, np.newaxis] / 9
    concepts["pi_e_digits"] = {
        "embedding": const_embed,
        "description": "Interleaved pi and e digits",
        "universal": True,
    }

    # 9. COUNTING PATTERN (simplest possible message)
    # 1, 2, 3, ... then repeat
    counting = np.array([((i % 10) + 1) / 10 for i in range(n_samples)])
    concepts["counting"] = {
        "embedding": counting[:, np.newaxis],
        "description": "Counting 1-10 repeated",
        "universal": True,
    }

    # 10. POSITION ENCODING (where am I in the sequence)
    position = np.linspace(0, 1, n_samples)
    concepts["position"] = {
        "embedding": position[:, np.newaxis],
        "description": "Linear position encoding",
        "universal": True,
    }

    return concepts


def extract_wow_features(snr_matrix: np.ndarray) -> np.ndarray:
    """Extract feature representation from Wow! signal.

    We want to match the structure of the concept manifold.
    """
    # Option 1: Use time slices directly (each time step is a sample)
    # Normalize per row
    features = snr_matrix.astype(np.float64)
    features = (features - np.mean(features)) / (np.std(features) + 1e-10)

    # Reduce to 1D per sample (mean across frequency)
    # This matches the concept manifold structure
    features_1d = np.mean(features, axis=1, keepdims=True)

    return features_1d


def compute_alignment(wow_features: np.ndarray, concept_embedding: np.ndarray) -> dict:
    """Compute alignment between Wow! features and concept embedding.

    Uses Procrustes alignment to find optimal rotation.
    """
    # Ensure same number of samples
    n = min(len(wow_features), len(concept_embedding))
    wow_sub = wow_features[:n]
    concept_sub = concept_embedding[:n]

    # Normalize
    wow_norm = (wow_sub - np.mean(wow_sub)) / (np.std(wow_sub) + 1e-10)
    concept_norm = (concept_sub - np.mean(concept_sub)) / (np.std(concept_sub) + 1e-10)

    # Raw correlation (before alignment)
    raw_corr = np.corrcoef(wow_norm.ravel(), concept_norm.ravel())[0, 1]

    # For 1D signals, "alignment" is just finding the best scale
    # Compute optimal scale: min ||wow * scale - concept||
    scale = np.sum(wow_norm * concept_norm) / (np.sum(wow_norm * wow_norm) + 1e-10)
    aligned_wow = wow_norm * scale

    # Aligned correlation
    aligned_corr = np.corrcoef(aligned_wow.ravel(), concept_norm.ravel())[0, 1]

    # CKA-like measure (Gram similarity)
    K_wow = wow_norm @ wow_norm.T
    K_concept = concept_norm @ concept_norm.T
    hsic = np.trace(K_wow @ K_concept)
    hsic_ww = np.trace(K_wow @ K_wow)
    hsic_cc = np.trace(K_concept @ K_concept)
    cka = hsic / (np.sqrt(hsic_ww * hsic_cc) + 1e-10)

    return {
        "raw_correlation": float(raw_corr) if not np.isnan(raw_corr) else 0.0,
        "aligned_correlation": float(aligned_corr) if not np.isnan(aligned_corr) else 0.0,
        "cka": float(cka) if not np.isnan(cka) else 0.0,
        "optimal_scale": float(scale),
        "n_samples": int(n),
    }


def find_best_alignment(wow_features: np.ndarray, concepts: dict) -> dict:
    """Find which concept manifold best aligns with Wow! signal."""
    alignments = {}

    for name, concept_data in concepts.items():
        alignment = compute_alignment(wow_features, concept_data["embedding"])
        alignments[name] = {
            "description": concept_data["description"],
            **alignment,
        }

    # Rank by CKA
    ranked = sorted(alignments.items(), key=lambda x: abs(x[1]["cka"]), reverse=True)

    return {
        "alignments": alignments,
        "ranking": [(name, data["cka"]) for name, data in ranked],
        "best_match": ranked[0] if ranked else None,
    }


def analyze_translation(wow_features: np.ndarray, concept: np.ndarray, name: str) -> dict:
    """If alignment is good, what does each part of the signal "mean"?"""
    n = min(len(wow_features), len(concept))
    wow = wow_features[:n].ravel()
    concept = concept[:n].ravel()

    # Normalize
    wow_norm = (wow - np.mean(wow)) / (np.std(wow) + 1e-10)
    concept_norm = (concept - np.mean(concept)) / (np.std(concept) + 1e-10)

    # Find correspondences
    # For each wow value, what concept value does it map to?
    scale = np.sum(wow_norm * concept_norm) / (np.sum(wow_norm * wow_norm) + 1e-10)

    translations = []
    for i in range(n):
        translated = wow_norm[i] * scale
        closest_concept_idx = np.argmin(np.abs(concept_norm - translated))
        translations.append({
            "time_idx": i,
            "wow_value": float(wow[i]),
            "translated_value": float(translated),
            "closest_concept_idx": int(closest_concept_idx),
            "concept_value": float(concept[closest_concept_idx]),
        })

    return {
        "concept_name": name,
        "translations": translations,
    }


def run_experiment():
    """Run the manifold alignment experiment."""
    results_dir = Path(__file__).parent / "results"
    data_dir = Path(__file__).parent / "data" / "famous_signals"

    print("=" * 60)
    print("Experiment 34: Align to Known Manifold for Translation")
    print("=" * 60)
    print("\nIf we can align to a manifold we understand, translation is possible.")

    # Load Wow! signal
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr'])

    print(f"\nWow! signal shape: {snr_matrix.shape}")

    # Extract features
    print("\n" + "=" * 40)
    print("PART 1: FEATURE EXTRACTION")
    print("=" * 40)

    wow_features = extract_wow_features(snr_matrix)
    print(f"Extracted features shape: {wow_features.shape}")

    # Also try the time profile (max across frequency)
    time_profile = np.max(snr_matrix, axis=1, keepdims=True)
    print(f"Time profile shape: {time_profile.shape}")

    # Create concept manifold
    print("\n" + "=" * 40)
    print("PART 2: CREATE UNIVERSAL CONCEPT MANIFOLD")
    print("=" * 40)

    concepts = create_universal_concept_manifold(n_samples=snr_matrix.shape[0])

    print(f"\nCreated {len(concepts)} universal concepts:")
    for name, data in concepts.items():
        print(f"  {name}: {data['description']}")

    # Find best alignment using mean features
    print("\n" + "=" * 40)
    print("PART 3: ALIGNMENT (MEAN FEATURES)")
    print("=" * 40)

    mean_alignment = find_best_alignment(wow_features, concepts)

    print("\nAlignment ranking (by CKA):")
    for i, (name, cka) in enumerate(mean_alignment["ranking"][:5]):
        print(f"  {i+1}. {name}: CKA = {cka:.4f}")

    # Find best alignment using time profile
    print("\n" + "=" * 40)
    print("PART 4: ALIGNMENT (TIME PROFILE)")
    print("=" * 40)

    profile_alignment = find_best_alignment(time_profile, concepts)

    print("\nAlignment ranking (by CKA):")
    for i, (name, cka) in enumerate(profile_alignment["ranking"][:5]):
        corr = profile_alignment["alignments"][name]["raw_correlation"]
        print(f"  {i+1}. {name}: CKA = {cka:.4f}, corr = {corr:.4f}")

    # Detailed analysis of best match
    print("\n" + "=" * 40)
    print("PART 5: TRANSLATION ATTEMPT")
    print("=" * 40)

    best_name = profile_alignment["ranking"][0][0]
    best_concept = concepts[best_name]["embedding"]

    translation = analyze_translation(time_profile, best_concept, best_name)

    print(f"\nBest alignment: {best_name}")
    print(f"Translating Wow! signal using {best_name} manifold...\n")

    # Show key moments
    peak_idx = np.argmax(time_profile)
    key_indices = [0, peak_idx - 5, peak_idx, peak_idx + 5, len(time_profile) - 1]
    key_indices = [i for i in key_indices if 0 <= i < len(translation["translations"])]

    print("Key moments in translation:")
    for idx in key_indices:
        t = translation["translations"][idx]
        print(f"  t={t['time_idx']:2d}: wow={t['wow_value']:6.2f} → concept[{t['closest_concept_idx']:2d}]={t['concept_value']:.4f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    best_cka = profile_alignment["ranking"][0][1]
    best_corr = profile_alignment["alignments"][best_name]["raw_correlation"]

    print(f"""
ALIGNMENT RESULTS:

Best matching concept: {best_name.upper()}
Description: {concepts[best_name]['description']}

Alignment strength:
  CKA = {best_cka:.4f}
  Correlation = {best_corr:.4f}

WHAT THIS MEANS:
""")

    if best_cka > 0.8:
        print(f"""
  ✓ STRONG ALIGNMENT (CKA > 0.8)
  The Wow! signal's structure closely matches {best_name}!
  This suggests the signal may encode {concepts[best_name]['description']}.
""")
    elif best_cka > 0.5:
        print(f"""
  ○ MODERATE ALIGNMENT (CKA 0.5-0.8)
  Partial structural similarity to {best_name}.
  The signal may contain elements of {concepts[best_name]['description']}.
""")
    else:
        print(f"""
  ? WEAK ALIGNMENT (CKA < 0.5)
  No strong match to any universal concept manifold.
  The signal may encode something we haven't included,
  or may not follow human-recognizable mathematical patterns.
""")

    # The Gaussian pulse check (most likely for a burst signal)
    gaussian_cka = profile_alignment["alignments"]["gaussian_pulse"]["cka"]
    gaussian_corr = profile_alignment["alignments"]["gaussian_pulse"]["raw_correlation"]

    print(f"""
GAUSSIAN PULSE CHECK (expected for natural burst):
  CKA = {gaussian_cka:.4f}
  Correlation = {gaussian_corr:.4f}

  {"→ Strong match to Gaussian pulse (consistent with natural transient)" if gaussian_corr > 0.7 else "→ Does NOT match simple Gaussian (structure is more complex)"}

THE KEY QUESTION:
  If the signal were JUST a radio burst, it would match Gaussian pulse.
  If it encodes INFORMATION, it would match mathematical patterns.

  Best match: {best_name}
  {"This suggests structure beyond simple burst envelope." if best_name != "gaussian_pulse" else "Signal appears to be a simple burst."}
""")

    # Save results
    results = {
        "experiment": "exp34_manifold_alignment",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(snr_matrix.shape),
        "concept_manifolds": list(concepts.keys()),
        "mean_feature_alignment": {
            "ranking": mean_alignment["ranking"],
            "best_match": mean_alignment["ranking"][0] if mean_alignment["ranking"] else None,
        },
        "time_profile_alignment": {
            "ranking": profile_alignment["ranking"],
            "best_match": profile_alignment["ranking"][0] if profile_alignment["ranking"] else None,
            "alignments": {name: {k: v for k, v in data.items() if k != "description"}
                         for name, data in profile_alignment["alignments"].items()},
        },
        "best_translation": {
            "concept": best_name,
            "cka": float(best_cka),
            "correlation": float(best_corr),
        },
    }

    output_path = results_dir / "exp34_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
