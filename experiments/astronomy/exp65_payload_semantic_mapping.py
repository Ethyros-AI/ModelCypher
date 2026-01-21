"""
Experiment 65: Semantic Mapping of the Period 12 Payload

exp64 verified:
1. Period 12 motif has "double e" eigenstructure (S0/S1≈e, S1/S2≈e).
2. Segments are structurally related (Procrustes alignment > 0.3 vs ~0 raw).

Hypothesis:
If the header (phi/pi) maps to "MATHEMATICAL" generally,
does the payload (e/e) map to specific concepts like "GROWTH", "CHANGE", or "EXPONENTIAL"?

This experiment:
1. Reconstructs the 'clean' Period 12 motif.
2. Projects this SPECIFIC motif onto the semantic highway (using exp42 machinery).
3. Checks if the top matching concepts shift from 'abstract math' to 'growth/change'.
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

def get_period12_motif():
    """Extract and process the Period 12 motif."""
    wow = load_wow_signal()
    period = 12
    n_time = wow.shape[1]
    n_segments = n_time // period
    segments = []
    
    for i in range(n_segments):
        start = i * period
        end = start + period
        seg = wow[:, start:end]
        segments.append(seg)
        
    stacked = np.stack(segments, axis=0)
    mean_motif = np.mean(stacked, axis=0)
    
    # Normalize
    mean_motif = (mean_motif - np.mean(mean_motif)) / (np.std(mean_motif) + 1e-10)
    
    return mean_motif

def main():
    print("=" * 60)
    print("Experiment 65: Semantic Mapping of Period 12 Payload")
    print("=" * 60)
    
    # 1. Get the payload motif
    print("\n1. Extracting Period 12 'Double e' Motif...")
    motif = get_period12_motif()
    print(f"   Motif shape: {motif.shape}")
    
    # 2. Load Semantic Highway
    print("\n2. Loading Semantic Highway...")
    model_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    model, tokenizer, n_layers = load_model(model_path)
    
    # Use bottleneck layer (same as exp42)
    layer_idx = n_layers // 2
    semantic_data, semantic_activations = build_semantic_manifold(model, tokenizer, layer_idx)
    
    # 3. Project Motif
    print("\n3. Projecting Motif to Semantic Manifold...")
    
    # We need to project the motif. 
    # Important: The motif is 82x12. The original signal was 82x50.
    # The Gram matrix size will differ (12x12 vs 50x50), but the projection 
    # logic in exp42 compares *spectral signatures* (eigenvalues), which are scale-invariant.
    # So this should work directly.
    
    top_matches, similarities, features = project_signal_to_manifold(
        motif, semantic_activations, semantic_data, n_components=8
    )
    
    print("\n   TOP 20 CONCEPT MATCHES FOR PAYLOAD:")
    print("   " + "-" * 50)
    for i, match in enumerate(top_matches):
        print(f"   {i+1:2d}. [{match['category']:10s}] {match['label']:15s} = {match['similarity']:.4f}")
        
    # Analyze Categories
    print("\n   CATEGORY RANKINGS:")
    cat_sims = {}
    cat_counts = {}
    
    for i, d in enumerate(semantic_data):
        cat = d["category"]
        if cat not in cat_sims:
            cat_sims[cat] = 0.0
            cat_counts[cat] = 0
        cat_sims[cat] += similarities[i]
        cat_counts[cat] += 1
        
    sorted_cats = sorted([(c, cat_sims[c]/cat_counts[c]) for c in cat_sims], 
                        key=lambda x: x[1], reverse=True)
    
    for cat, score in sorted_cats:
        print(f"   {cat:12s}: {score:.4f}")
        
    # Check specifically for "growth" related concepts vs "math"
    print("\n   Specific Concept Checks:")
    targets = ["exponential", "growth", "change", "evolution", "pi", "prime", "mathematics"]
    
    print(f"   {'Concept':15s} {'Rank':<5} {'Score':<6}")
    print("   " + "-"*30)
    
    # Create a quick lookup
    lookup = {m['label']: (i+1, m['similarity']) for i, m in enumerate(top_matches)}
    # We need to look in the full list for lower ranked ones
    full_lookup = {}
    sorted_indices = np.argsort(similarities)[::-1]
    for rank, idx in enumerate(sorted_indices):
        label = semantic_data[idx]['label']
        full_lookup[label] = (rank+1, similarities[idx])
        
    for t in targets:
        # Fuzzy match
        found = False
        for label, (rank, score) in full_lookup.items():
            if t in label.lower():
                print(f"   {label:15s} #{rank:<4} {score:.4f}")
                found = True
        if not found:
            print(f"   {t:15s} --    --")

    # Save
    output = {
        "experiment": "exp65_payload_semantics",
        "timestamp": datetime.now().isoformat(),
        "top_matches": top_matches,
        "category_rankings": {c: s for c, s in sorted_cats}
    }
    
    with open(RESULTS_DIR / "exp65_results.json", "w") as f:
        json.dump(output, f, indent=2)
        
if __name__ == "__main__":
    main()
