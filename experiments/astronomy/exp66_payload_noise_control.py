"""
Experiment 66: Payload Noise Control

exp65 found that the Period 12 "double e" payload maps to "EXCITEMENT" (1.0).
Is this real, or does any 82x12 low-rank matrix map to "EXCITEMENT"?

This experiment:
1. Generates random motifs of shape (82, 12).
2. Matches their spectral profile (eigenvalues) to the real payload.
3. Projects them onto the semantic manifold.
4. Checks if "EXCITEMENT" or "EMOTIONS" appear significantly.

If random motifs map to "EXCITEMENT", the finding is an artifact.
If random motifs map to "NOISE/CHAOS", the finding is real.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import (
    load_model,
    build_semantic_manifold,
    project_signal_to_manifold,
    load_wow_signal,
    SEMANTIC_CATEGORIES
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def generate_surrogate_motifs(n_trials, target_shape, target_spectrum):
    """Generate random matrices with the same singular values as the target."""
    surrogates = []
    for _ in range(n_trials):
        # Generate random matrix
        H = np.random.randn(*target_shape)
        U, _, Vh = linalg.svd(H, full_matrices=False)
        
        # Force spectrum to match target
        # Pad target spectrum if needed
        s_new = np.zeros(len(U[0]))
        k = min(len(target_spectrum), len(s_new))
        s_new[:k] = target_spectrum[:k]
        
        # Reconstruct
        H_new = U @ np.diag(s_new) @ Vh
        surrogates.append(H_new)
    return surrogates

def get_payload_spectrum():
    """Get the singular values of the real payload."""
    wow = load_wow_signal()
    period = 12
    n_time = wow.shape[1]
    n_segments = n_time // period
    segments = [wow[:, i*period:(i+1)*period] for i in range(n_segments)]
    stacked = np.stack(segments, axis=0)
    mean_motif = np.mean(stacked, axis=0)
    
    # Normalize
    mean_motif = (mean_motif - np.mean(mean_motif)) / (np.std(mean_motif) + 1e-10)
    
    _, S, _ = linalg.svd(mean_motif, full_matrices=False)
    return mean_motif.shape, S

def main():
    print("=" * 60)
    print("Experiment 66: Payload Noise Control")
    print("=" * 60)
    
    # 1. Get Payload Properties
    shape, spectrum = get_payload_spectrum()
    print(f"   Payload Shape: {shape}")
    print(f"   Top Eigenvalues: {[f'{s:.2f}' for s in spectrum[:5]]}")
    
    # 2. Load Semantic Manifold
    print("\n2. Loading Semantic Highway...")
    model_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    model, tokenizer, n_layers = load_model(model_path)
    semantic_data, semantic_activations = build_semantic_manifold(model, tokenizer, n_layers // 2)
    
    # 3. Run Controls
    n_trials = 20
    print(f"\n3. Running {n_trials} Rank-Matched Controls...")
    
    controls = generate_surrogate_motifs(n_trials, shape, spectrum)
    
    excitement_scores = []
    top_cats = []
    
    for i, ctrl in enumerate(controls):
        top_matches, sims, _ = project_signal_to_manifold(
            ctrl, semantic_activations, semantic_data, n_components=8
        )
        
        # Check for "excitement"
        excitement_score = 0
        for m in top_matches:
            if m['label'] == 'excitement':
                excitement_score = m['similarity']
                break
        
        excitement_scores.append(excitement_score)
        top_cats.append(top_matches[0]['category'])
        
        if i % 5 == 0:
            print(f"   Trial {i}: Top Cat = {top_matches[0]['category']}, Excitement = {excitement_score:.4f}")
            
    # 4. Statistics
    mean_excitement = np.mean(excitement_scores)
    std_excitement = np.std(excitement_scores)
    
    # Real payload score (from exp65)
    real_excitement = 1.0000 
    
    z_score = (real_excitement - mean_excitement) / (std_excitement + 1e-10)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"   Real Payload 'Excitement': {real_excitement:.4f}")
    print(f"   Random Control Mean:       {mean_excitement:.4f} ± {std_excitement:.4f}")
    print(f"   Z-Score:                   {z_score:+.2f}")
    
    print("\n   Top Categories in Random Trials:")
    from collections import Counter
    print(Counter(top_cats))
    
    if z_score > 3:
        print("\n   CONCLUSION: The 'Excitement' mapping is STATISTICALLY SIGNIFICANT.")
        print("   It is NOT an artifact of the spectral profile.")
    else:
        print("\n   CONCLUSION: The mapping is indistinguishable from noise.")
        
    # Save
    with open(RESULTS_DIR / "exp66_results.json", "w") as f:
        json.dump({
            "experiment": "exp66_payload_noise_control",
            "timestamp": datetime.now().isoformat(),
            "real_excitement": real_excitement,
            "control_stats": {
                "mean": float(mean_excitement),
                "std": float(std_excitement),
                "z_score": float(z_score)
            },
            "control_categories": top_cats
        }, f, indent=2)

if __name__ == "__main__":
    main()
