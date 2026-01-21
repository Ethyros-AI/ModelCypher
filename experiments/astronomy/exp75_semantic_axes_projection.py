"""
Experiment 75: Semantic Axes Projection

Objective:
Project the Wow! signal onto the fundamental axes of the Semantic Manifold
defined in the Dimensional Hierarchy research.

Axes:
1. PC1: Abstract \u2194 Concrete
2. PC2: Animate \u2194 Static
3. PC3: Natural \u2194 Artificial

Method:
1. Load the LLM (SmolLM-135M).
2. Construct the 3 axes using contrastive concept pairs.
3. Map the Wow! signal into this embedding space.
   - We use the projection weights from exp42 (Gram alignment) to create 
     a weighted centroid of semantic concepts that represents the signal.
4. Measure the signal's position on these axes.

Hypothesis:
- If it's a beacon/structure: Expected [Abstract, Static, Artificial]
- If it's a natural phenomenon: Expected [Concrete, Dynamic, Natural]
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

# Define axes using specific probe pairs to define the vector direction
AXIS_DEFINITIONS = {
    "PC1_Abstract_Concrete": {
        "positive": ["philosophy", "mathematics", "love", "justice", "infinity", "theory", "concept"],
        "negative": ["rock", "tree", "car", "apple", "house", "dog", "water"]
    },
    "PC2_Animate_Static": {
        "positive": ["running", "dancing", "thinking", "living", "growing", "eating", "fighting"],
        "negative": ["stone", "statue", "mountain", "table", "wall", "dead", "frozen"]
    },
    "PC3_Natural_Artificial": {
        "positive": ["tree", "river", "sun", "cloud", "flower", "animal", "ocean"],
        "negative": ["computer", "robot", "engine", "plastic", "algorithm", "factory", "concrete"]
    }
}

def get_embedding(model, tokenizer, text, layer_idx):
    """Helper to get single embedding."""
    import mlx.core as mx
    
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    
    inner = model.model
    if hasattr(inner, "embed_tokens"):
        h = inner.embed_tokens(input_ids)
    else:
        h = inner.wte(input_ids)
        
    for idx, layer in enumerate(inner.layers):
        if idx > layer_idx:
            break
        h = layer(h)
        if isinstance(h, tuple): h = h[0]
            
    # Mean pool
    return np.array(mx.mean(h, axis=(0,1))).astype(np.float64)

def construct_axes(model, tokenizer, layer_idx):
    """Build the coordinate system vectors."""
    axes = {}
    print("\nConstructing Semantic Axes...")
    
    for name, poles in AXIS_DEFINITIONS.items():
        pos_vecs = []
        neg_vecs = []
        
        for p in poles["positive"]:
            v = get_embedding(model, tokenizer, p, layer_idx)
            pos_vecs.append(v)
            
        for n in poles["negative"]:
            v = get_embedding(model, tokenizer, n, layer_idx)
            neg_vecs.append(v)
            
        # Axis vector points from Negative to Positive
        pos_centroid = np.mean(pos_vecs, axis=0)
        neg_centroid = np.mean(neg_vecs, axis=0)
        axis_vec = pos_centroid - neg_centroid
        
        # Normalize
        axis_vec = axis_vec / np.linalg.norm(axis_vec)
        axes[name] = axis_vec
        print(f"   Defined {name}")
        
    return axes

def main():
    print("=" * 60)
    print("Experiment 75: Semantic Axes Projection")
    print("=" * 60)
    
    # 1. Load Model
    print("\n1. Loading Model...")
    model_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    model, tokenizer, n_layers = load_model(model_path)
    layer_idx = n_layers // 2
    
    # 2. Build Axes
    axes = construct_axes(model, tokenizer, layer_idx)
    
    # 3. Load Wow! Signal and Manifold
    print("\n2. Mapping Signal to Embedding Space...")
    wow = load_wow_signal()
    semantic_data, semantic_activations = build_semantic_manifold(model, tokenizer, layer_idx)
    
    # Get projection weights (similarities)
    # We use the signal's structure to find which concepts it activates
    top_matches, similarities, _ = project_signal_to_manifold(
        wow, semantic_activations, semantic_data, n_components=10
    )
    
    # 4. Construct Signal Vector
    # Weighted sum of semantic concepts based on structural similarity
    # V_signal = sum(similarity_i * V_concept_i)
    
    print("\n3. Synthesizing Signal Vector...")
    signal_vec = np.zeros_like(semantic_activations[0])
    total_weight = 0
    
    # Use only positive similarities (activations) to build the vector
    # or use all? Usually cosine sim is [-1, 1].
    # exp42 similarities are computed via absolute loading matching, so they are [0, 1].
    
    for i, sim in enumerate(similarities):
        if sim > 0:
            signal_vec += sim * semantic_activations[i]
            total_weight += sim
            
    signal_vec = signal_vec / total_weight
    signal_vec = signal_vec / np.linalg.norm(signal_vec)
    
    # 5. Project onto Axes
    print("\n4. Projecting onto Axes...")
    results = {}
    
    print("\n   COORDINATES (Range -1.0 to +1.0):")
    print("   ---------------------------------")
    
    for name, axis in axes.items():
        # Dot product = projection
        score = np.dot(signal_vec, axis)
        results[name] = float(score)
        
        # Interpret
        pole = "Neutral"
        if abs(score) > 0.05: # Threshold
            if "Abstract_Concrete" in name:
                pole = "ABSTRACT" if score > 0 else "CONCRETE"
            elif "Animate_Static" in name:
                pole = "ANIMATE" if score > 0 else "STATIC"
            elif "Natural_Artificial" in name:
                pole = "NATURAL" if score > 0 else "ARTIFICIAL"
                
        print(f"   {name:<25}: {score:+.4f} --> {pole}")

    # 6. Verify "Excitement" Payload (from exp66)
    # Let's project the "Period 12 Payload" specifically
    print("\n5. Checking Period 12 Payload Position...")
    
    # Reconstruct payload motif
    period = 12
    n_segments = wow.shape[1] // period
    segments = [wow[:, i*period:(i+1)*period] for i in range(n_segments)]
    motif = np.mean(np.stack(segments), axis=0)
    
    # Project motif
    _, motif_sims, _ = project_signal_to_manifold(motif, semantic_activations, semantic_data, n_components=8)
    
    motif_vec = np.zeros_like(signal_vec)
    motif_weight = 0
    for i, sim in enumerate(motif_sims):
        if sim > 0:
            motif_vec += sim * semantic_activations[i]
            motif_weight += sim
    motif_vec = motif_vec / motif_weight
    motif_vec = motif_vec / np.linalg.norm(motif_vec)
    
    for name, axis in axes.items():
        score = np.dot(motif_vec, axis)
        print(f"   Payload on {name:<14}: {score:+.4f}")

    # Save
    with open(RESULTS_DIR / "exp75_results.json", "w") as f:
        json.dump({
            "experiment": "exp75_semantic_axes",
            "timestamp": datetime.now().isoformat(),
            "global_coordinates": results
        }, f, indent=2)

if __name__ == "__main__":
    main()
