"""
Experiment 42: Map Wow! Signal to the Semantic Highway

The key insight: embeddings are translation layers (on/off ramps).
The INVARIANT structure lives in the middle layers - the semantic highway.

If an intelligence sent a message, they'd encode it in the invariant
geometric structure - the shape that ANY intelligence converges to.

This experiment:
1. Builds the semantic highway manifold from LLM middle layers
2. Projects the Wow! signal's eigenstructure onto that manifold
3. Finds where on the invariant structure the signal lands
4. The location IS the message
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.io import readsav
from scipy import linalg

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Semantic categories from visualize_semantic_highway.py
SEMANTIC_CATEGORIES = {
    "PRIMES": {
        "probes": [
            ("I", "I am here."),
            ("YOU", "You are there."),
            ("SOMEONE", "Someone is coming."),
            ("SOMETHING", "Something happened."),
            ("GOOD", "This is good."),
            ("BAD", "This is bad."),
            ("BIG", "It is big."),
            ("SMALL", "It is small."),
            ("THINK", "I think this."),
            ("KNOW", "I know this."),
            ("WANT", "I want this."),
            ("FEEL", "I feel this."),
            ("SEE", "I see this."),
            ("HEAR", "I hear this."),
            ("SAY", "I say this."),
            ("DO", "I do this."),
            ("HAPPEN", "It happened."),
            ("MOVE", "It moves."),
            ("LIVE", "People live."),
            ("DIE", "People die."),
            ("NOW", "It is now."),
            ("BEFORE", "Before this."),
            ("AFTER", "After this."),
            ("HERE", "It is here."),
            ("ABOVE", "It is above."),
            ("BELOW", "It is below."),
            ("NOT", "It is not."),
            ("MAYBE", "Maybe it is."),
            ("CAN", "I can do this."),
            ("BECAUSE", "Because of this."),
        ],
    },
    "ABSTRACT": {
        "probes": [
            ("love", "Love is a powerful emotion."),
            ("justice", "Justice requires fairness."),
            ("freedom", "Freedom means independence."),
            ("truth", "Truth is hard to find."),
            ("beauty", "Beauty is subjective."),
            ("wisdom", "Wisdom comes with experience."),
            ("courage", "Courage overcomes fear."),
            ("hope", "Hope sustains us."),
            ("faith", "Faith guides belief."),
            ("honor", "Honor demands integrity."),
            ("democracy", "Democracy requires participation."),
            ("philosophy", "Philosophy seeks wisdom."),
            ("ethics", "Ethics guides behavior."),
            ("morality", "Morality defines right and wrong."),
            ("consciousness", "Consciousness is awareness."),
        ],
    },
    "CONCRETE": {
        "probes": [
            ("apple", "The red apple sits on the table."),
            ("chair", "A wooden chair has four legs."),
            ("table", "The table is made of oak."),
            ("book", "The book has many pages."),
            ("car", "The car drives fast."),
            ("house", "The house has a roof."),
            ("tree", "The tree grows tall."),
            ("rock", "The rock is heavy."),
            ("water", "Water flows downhill."),
            ("fire", "Fire burns hot."),
            ("mountain", "The mountain is tall."),
            ("river", "The river flows to the sea."),
            ("bird", "The bird flies south."),
            ("dog", "The dog runs fast."),
            ("cat", "The cat sleeps often."),
        ],
    },
    "ACTIONS": {
        "probes": [
            ("running", "She is running fast."),
            ("jumping", "He is jumping high."),
            ("swimming", "They are swimming in the pool."),
            ("writing", "I am writing a letter."),
            ("reading", "She is reading a book."),
            ("eating", "We are eating dinner."),
            ("sleeping", "The baby is sleeping."),
            ("dancing", "They are dancing together."),
            ("singing", "She is singing beautifully."),
            ("building", "Workers are building a house."),
            ("destroying", "The storm is destroying everything."),
            ("creating", "Artists are creating art."),
            ("learning", "Students are learning math."),
            ("teaching", "Teachers are teaching history."),
            ("healing", "Doctors are healing patients."),
        ],
    },
    "EMOTIONS": {
        "probes": [
            ("happiness", "Happiness fills the room."),
            ("sadness", "Sadness overwhelms her."),
            ("anger", "Anger consumes him."),
            ("fear", "Fear grips the crowd."),
            ("surprise", "Surprise lit up her face."),
            ("disgust", "Disgust showed clearly."),
            ("joy", "Joy spreads easily."),
            ("grief", "Grief takes time."),
            ("anxiety", "Anxiety builds slowly."),
            ("excitement", "Excitement grows."),
            ("jealousy", "Jealousy poisons relationships."),
            ("pride", "Pride swelled in his chest."),
            ("shame", "Shame colored her cheeks."),
            ("guilt", "Guilt weighed heavily."),
            ("contentment", "Contentment settled in."),
        ],
    },
    "SCIENTIFIC": {
        "probes": [
            ("photosynthesis", "Photosynthesis converts light to energy."),
            ("gravity", "Gravity pulls objects down."),
            ("evolution", "Evolution shapes species."),
            ("electricity", "Electricity powers devices."),
            ("magnetism", "Magnetism attracts metals."),
            ("chemistry", "Chemistry studies matter."),
            ("biology", "Biology studies life."),
            ("physics", "Physics explains motion."),
            ("mathematics", "Mathematics is precise."),
            ("algorithm", "The algorithm processes data."),
            ("quantum", "Quantum mechanics is strange."),
            ("relativity", "Relativity bends spacetime."),
            ("entropy", "Entropy always increases."),
            ("thermodynamics", "Thermodynamics governs heat."),
            ("genetics", "Genetics determines traits."),
        ],
    },
    "SOCIAL": {
        "probes": [
            ("family", "Family gathers for dinner."),
            ("friend", "A friend listens carefully."),
            ("enemy", "The enemy approaches."),
            ("stranger", "A stranger walked by."),
            ("community", "The community came together."),
            ("society", "Society shapes behavior."),
            ("culture", "Culture varies widely."),
            ("tradition", "Tradition guides practice."),
            ("marriage", "Marriage joins two people."),
            ("childhood", "Childhood shapes personality."),
            ("leadership", "Leadership requires vision."),
            ("cooperation", "Cooperation achieves more."),
            ("conflict", "Conflict arises often."),
            ("negotiation", "Negotiation seeks compromise."),
            ("celebration", "Celebration brings joy."),
        ],
    },
    "TEMPORAL": {
        "probes": [
            ("yesterday", "Yesterday was rainy."),
            ("tomorrow", "Tomorrow will be sunny."),
            ("ancient", "Ancient civilizations built pyramids."),
            ("modern", "Modern technology advances."),
            ("future", "The future is uncertain."),
            ("past", "The past cannot change."),
            ("present", "The present moment matters."),
            ("eternal", "Some things seem eternal."),
            ("temporary", "This is only temporary."),
            ("sudden", "A sudden change occurred."),
            ("gradual", "Gradual change is lasting."),
            ("beginning", "Every story has a beginning."),
            ("ending", "Every story has an ending."),
            ("duration", "The duration was long."),
            ("moment", "A moment of silence."),
        ],
    },
    "SPATIAL": {
        "probes": [
            ("inside", "It is inside the box."),
            ("outside", "Go outside to play."),
            ("near", "Stay near the door."),
            ("far", "The star is far away."),
            ("left", "Turn left at the corner."),
            ("right", "The door is on the right."),
            ("up", "Look up at the sky."),
            ("down", "Climb down the ladder."),
            ("center", "Stand in the center."),
            ("edge", "Walk along the edge."),
            ("surface", "The surface is smooth."),
            ("depth", "The depth is unknown."),
            ("height", "The height is impressive."),
            ("width", "The width is narrow."),
            ("distance", "The distance is great."),
        ],
    },
    # Additional probes relevant to a cosmic message
    "MATHEMATICAL": {
        "probes": [
            ("one", "The number one is unity."),
            ("two", "Two things together."),
            ("three", "Three points define a plane."),
            ("pi", "Pi is approximately 3.14159."),
            ("prime", "Prime numbers are indivisible."),
            ("sequence", "The sequence follows a pattern."),
            ("pattern", "The pattern repeats itself."),
            ("ratio", "The ratio is constant."),
            ("infinity", "Infinity has no end."),
            ("zero", "Zero means nothing."),
            ("fibonacci", "The Fibonacci sequence appears in nature."),
            ("exponential", "Exponential growth is fast."),
            ("logarithm", "The logarithm compresses scale."),
            ("symmetry", "Symmetry is balanced."),
            ("dimension", "The dimension is measurable."),
        ],
    },
    "COSMIC": {
        "probes": [
            ("star", "The star shines brightly."),
            ("galaxy", "The galaxy contains billions of stars."),
            ("universe", "The universe is vast."),
            ("light", "Light travels fast."),
            ("signal", "The signal was received."),
            ("message", "The message was decoded."),
            ("transmission", "The transmission was clear."),
            ("radio", "Radio waves carry information."),
            ("frequency", "The frequency determines the pitch."),
            ("wavelength", "The wavelength is measurable."),
            ("spectrum", "The spectrum shows all colors."),
            ("origin", "The origin is unknown."),
            ("direction", "The direction points north."),
            ("arrival", "The arrival was unexpected."),
            ("contact", "Contact was established."),
        ],
    },
}


def load_wow_signal():
    """Load the Wow! signal from IDL .sav file."""
    data_path = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"
    if not data_path.exists():
        raise FileNotFoundError(f"Wow! signal not found at {data_path}")

    data = readsav(str(data_path))
    oseti = data['oseti'][0]
    signal = oseti['SNR']
    signal = signal.astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal


def load_model(path: str):
    """Load a model and return model, tokenizer, n_layers."""
    from mlx_lm import load
    import mlx.core as mx

    model, tokenizer = load(path)
    mx.eval(model.parameters())

    inner = model.model if hasattr(model, "model") else model
    n_layers = len(inner.layers)

    return model, tokenizer, n_layers


def get_layer_activation(model, tokenizer, text: str, layer_idx: int):
    """Get mean-pooled activation at specific layer."""
    import mlx.core as mx

    inner = model.model if hasattr(model, "model") else model

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    if hasattr(inner, "embed_tokens"):
        h = inner.embed_tokens(input_ids)
    elif hasattr(inner, "wte"):
        h = inner.wte(input_ids)
    else:
        return None

    for idx, layer in enumerate(inner.layers):
        if idx > layer_idx:
            break
        result = layer(h)
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result

    # Cast to float32 for stability, then to numpy float64
    h = h.astype(mx.float32)
    pooled = mx.mean(h, axis=(0, 1))
    mx.eval(pooled)
    result = np.array(pooled, dtype=np.float64)
    # Handle any NaN/Inf values
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def build_semantic_manifold(model, tokenizer, layer_idx: int):
    """Build the semantic highway manifold from LLM middle layers."""
    print(f"   Building semantic manifold at layer {layer_idx}...")

    all_data = []
    all_activations = []

    for category, info in SEMANTIC_CATEGORIES.items():
        print(f"      Collecting {category}...")
        for label, probe in info["probes"]:
            act = get_layer_activation(model, tokenizer, probe, layer_idx)
            if act is not None:
                all_data.append({
                    "category": category,
                    "label": label,
                    "probe": probe,
                })
                all_activations.append(act)

    activations = np.stack(all_activations)  # [n_concepts, hidden_dim]
    print(f"   Manifold shape: {activations.shape}")

    return all_data, activations


def project_signal_to_manifold(signal, semantic_activations, semantic_data, n_components=10):
    """
    Project the signal's eigenstructure onto the semantic manifold.

    NEW APPROACH: Compare geometric structure via Gram matrices.
    The Gram matrix K = X @ X.T encodes the relational geometry - which concepts
    are "near" which other concepts. This is the INVARIANT structure.

    We find which semantic concept's local neighborhood best matches the
    signal's internal structure.
    """
    print("   Projecting signal to manifold via Gram alignment...")

    # 1. Normalize activations to have unit norm (for cosine similarity)
    semantic_norms = np.linalg.norm(semantic_activations, axis=1, keepdims=True)
    semantic_unit = semantic_activations / (semantic_norms + 1e-8)

    # 2. Signal structure: Each time slice is a point in frequency space
    # Normalize signal rows (time slices)
    signal_row_norms = np.linalg.norm(signal, axis=1, keepdims=True)
    signal_unit = signal / (signal_row_norms + 1e-8)

    # 3. Compute Gram matrices (relational structure)
    G_semantic = semantic_unit @ semantic_unit.T  # [150, 150] - how concepts relate
    G_signal = signal_unit @ signal_unit.T  # [82, 82] - how time slices relate

    print(f"   Semantic Gram: {G_semantic.shape}")
    print(f"   Signal Gram: {G_signal.shape}")

    # 4. The signal's Gram matrix encodes its geometric structure
    # We want to find which semantic concepts have similar structure
    # Use the eigenspectrum of each Gram matrix as a geometric signature

    _, S_semantic, _ = linalg.svd(G_semantic, full_matrices=False)
    _, S_signal, _ = linalg.svd(G_signal, full_matrices=False)

    # Normalize eigenspectra
    S_semantic_norm = S_semantic / S_semantic.sum()
    S_signal_norm = S_signal / S_signal.sum()

    # Match dimensions for comparison (truncate to shorter)
    k = min(len(S_semantic_norm), len(S_signal_norm), n_components * 3)
    S_semantic_k = S_semantic_norm[:k]
    S_signal_k = S_signal_norm[:k]

    # Spectral similarity
    spectral_sim = 1.0 - np.sqrt(np.sum((S_semantic_k - S_signal_k) ** 2))
    print(f"   Overall spectral similarity: {spectral_sim:.4f}")

    # 5. Alternative: For each concept, compute how well its local structure
    # matches the signal's global structure
    # Use the concept's row in the Gram matrix as its "context vector"

    # For each concept i, its context is G_semantic[i, :] - how it relates to all others
    # The signal's "context" is the mean correlation pattern

    signal_context = G_signal.mean(axis=0)  # Average relationship pattern [82]
    signal_context_norm = signal_context / (np.linalg.norm(signal_context) + 1e-8)

    # 6. Project signal into semantic space via learned mapping
    # Use top k eigenvectors of semantic Gram as basis
    U_sem, S_sem, Vt_sem = linalg.svd(G_semantic, full_matrices=False)
    k = min(n_components, len(S_sem))

    # The top eigenvectors define the "directions" in semantic space
    semantic_basis = U_sem[:, :k]  # [150, k]

    # Project signal's average structure into semantic space
    # This requires matching dimensions - use SVD of G_signal
    U_sig, S_sig, Vt_sig = linalg.svd(G_signal, full_matrices=False)

    # The signal's dominant eigenvectors encode its structure
    # Match the spectral signatures
    signal_spectrum = S_sig[:k] / S_sig[:k].sum() if S_sig[:k].sum() > 0 else np.ones(k) / k
    semantic_spectrum = S_sem[:k] / S_sem[:k].sum() if S_sem[:k].sum() > 0 else np.ones(k) / k

    # 7. For each concept, compute similarity based on:
    # a) How strongly it participates in each semantic eigenvector (weighted by spectral match)
    similarities = np.zeros(len(semantic_data))

    for i in range(len(semantic_data)):
        # Concept i's loading on each principal direction
        concept_loadings = semantic_basis[i, :]

        # Weight by how well each direction matches the signal's spectrum
        spectral_weight = np.exp(-np.abs(semantic_spectrum - signal_spectrum))
        weighted_loading = np.sum(np.abs(concept_loadings) * spectral_weight)

        similarities[i] = weighted_loading

    # Normalize
    similarities = similarities / (np.max(np.abs(similarities)) + 1e-8)

    # 8. Find top matches
    sorted_idx = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_idx[:20]:
        results.append({
            "label": semantic_data[idx]["label"],
            "category": semantic_data[idx]["category"],
            "probe": semantic_data[idx]["probe"],
            "similarity": float(similarities[idx]),
        })

    signal_features = S_signal_k  # Return the signal's eigenspectrum as features
    return results, similarities, signal_features


def compute_category_distribution(similarities, semantic_data):
    """Compute how similarity distributes across categories."""
    category_sims = {}
    category_counts = {}

    for i, d in enumerate(semantic_data):
        cat = d["category"]
        if cat not in category_sims:
            category_sims[cat] = 0.0
            category_counts[cat] = 0
        category_sims[cat] += similarities[i]
        category_counts[cat] += 1

    # Mean similarity per category
    category_means = {cat: category_sims[cat] / category_counts[cat] for cat in category_sims}

    return category_means


def run_random_baseline(n_trials, semantic_activations, semantic_data, signal_shape):
    """Run random noise baseline for comparison."""
    print(f"   Running {n_trials} random noise trials...")

    all_category_means = {cat: [] for cat in SEMANTIC_CATEGORIES.keys()}

    for _ in range(n_trials):
        # Generate random noise with same shape as signal
        noise = np.random.randn(*signal_shape)
        _, similarities, _ = project_signal_to_manifold(
            noise, semantic_activations, semantic_data, n_components=10
        )
        cat_means = compute_category_distribution(similarities, semantic_data)
        for cat, mean in cat_means.items():
            all_category_means[cat].append(mean)

    # Compute statistics
    baseline_stats = {}
    for cat in all_category_means:
        values = np.array(all_category_means[cat])
        baseline_stats[cat] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    return baseline_stats


def main():
    print("=" * 60)
    print("Experiment 42: Map Wow! Signal to Semantic Highway")
    print("=" * 60)

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    signal = load_wow_signal()
    print(f"   Signal shape: {signal.shape}")

    # Load model
    print("\n2. Loading LLM...")
    model_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    model, tokenizer, n_layers = load_model(model_path)
    print(f"   Model loaded: SmolLM-135M ({n_layers} layers)")

    # Build semantic manifold at bottleneck layer
    print("\n3. Building semantic manifold...")
    bottleneck_layer = n_layers // 2
    print(f"   Using bottleneck layer: {bottleneck_layer}")
    semantic_data, semantic_activations = build_semantic_manifold(model, tokenizer, bottleneck_layer)

    # Project signal to manifold
    print("\n4. Projecting signal to semantic manifold...")
    top_matches, similarities, signal_features = project_signal_to_manifold(
        signal, semantic_activations, semantic_data
    )

    print("\n   TOP 20 NEAREST CONCEPTS:")
    print("   " + "-" * 50)
    for i, match in enumerate(top_matches):
        print(f"   {i+1:2d}. [{match['category']:10s}] {match['label']:15s} = {match['similarity']:.4f}")

    # Category distribution
    print("\n5. Category distribution...")
    cat_means = compute_category_distribution(similarities, semantic_data)
    sorted_cats = sorted(cat_means.items(), key=lambda x: x[1], reverse=True)
    print("\n   CATEGORY RANKINGS:")
    for cat, mean in sorted_cats:
        print(f"   {cat:12s}: {mean:.4f}")

    # Random baseline
    print("\n6. Computing random noise baseline...")
    baseline_stats = run_random_baseline(20, semantic_activations, semantic_data, signal.shape)

    # Compute z-scores for categories
    print("\n   CATEGORY Z-SCORES (signal vs random):")
    z_scores = {}
    for cat, mean in cat_means.items():
        baseline_mean = baseline_stats[cat]["mean"]
        baseline_std = baseline_stats[cat]["std"]
        z = (mean - baseline_mean) / (baseline_std + 1e-8)
        z_scores[cat] = {
            "signal_mean": float(mean),
            "baseline_mean": float(baseline_mean),
            "baseline_std": float(baseline_std),
            "z_score": float(z),
        }

    sorted_z = sorted(z_scores.items(), key=lambda x: x[1]["z_score"], reverse=True)
    for cat, stats in sorted_z:
        direction = "ABOVE" if stats["z_score"] > 0 else "BELOW"
        print(f"   {cat:12s}: z={stats['z_score']:+.2f} ({direction} baseline)")

    # Save results
    results = {
        "experiment": "exp42_semantic_highway_mapping",
        "timestamp": datetime.now().isoformat(),
        "model": "SmolLM-135M",
        "bottleneck_layer": bottleneck_layer,
        "n_layers": n_layers,
        "signal_shape": list(signal.shape),
        "n_semantic_concepts": len(semantic_data),
        "top_matches": top_matches,
        "category_distribution": {cat: float(mean) for cat, mean in sorted_cats},
        "category_z_scores": z_scores,
        "signal_eigenvalues": {
            "spectrum": list(np.linalg.svd(signal, compute_uv=False)[:10]),
            "participation_ratio": float(np.sum(np.linalg.svd(signal, compute_uv=False) ** 2) ** 2 /
                                         np.sum(np.linalg.svd(signal, compute_uv=False) ** 4)),
        },
    }

    output_path = RESULTS_DIR / "exp42_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n7. Results saved to {output_path}")

    # Key finding summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    top_cat = sorted_cats[0][0]
    top_z = sorted_z[0]
    print(f"\nHighest category alignment: {top_cat}")
    print(f"Strongest z-score: {top_z[0]} (z={top_z[1]['z_score']:.2f})")
    print(f"\nTop 5 concept matches:")
    for i, match in enumerate(top_matches[:5]):
        print(f"  {i+1}. {match['label']} ({match['category']})")

    return results


if __name__ == "__main__":
    main()
