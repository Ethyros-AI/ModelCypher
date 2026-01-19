#!/usr/bin/env python3
"""NSM Primes vs Derived Concepts at the Highway.

Hypothesis: Semantic primes (Wierzbicka's NSM) are the "stars" of conceptual
space - they have more connections, more weight. At the bottleneck, primes
should project more strongly onto the principal components because the
highway IS the primes.

Test: Do prime probes project more strongly onto bottleneck PCs than
derived/composite concept probes?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# NSM Semantic Primes (Wierzbicka's ~65 universal concepts)
# Organized by category, with simple probe sentences
NSM_PRIMES = {
    # Substantives
    "I": ["I am here.", "I think.", "I want something."],
    "YOU": ["You are there.", "You know this.", "You can do it."],
    "SOMEONE": ["Someone is coming.", "Someone said this.", "Someone wants it."],
    "SOMETHING": ["Something happened.", "Something is here.", "Something moved."],
    "PEOPLE": ["People live here.", "People think differently.", "People want things."],
    "BODY": ["The body moves.", "My body feels.", "A body has parts."],

    # Determiners
    "THIS": ["This is good.", "This thing here.", "This happened now."],
    "THE_SAME": ["The same thing again.", "It is the same.", "They are the same."],
    "OTHER": ["The other one.", "Something other.", "Other people think."],

    # Quantifiers
    "ONE": ["One thing.", "One person came.", "There is one."],
    "TWO": ["Two things.", "Two people.", "There are two."],
    "SOME": ["Some people know.", "Some things are good.", "Some came."],
    "ALL": ["All people.", "All things.", "All of them."],
    "MANY": ["Many things.", "Many people.", "There are many."],

    # Evaluators
    "GOOD": ["This is good.", "Good things happen.", "It feels good."],
    "BAD": ["This is bad.", "Bad things happen.", "It feels bad."],

    # Descriptors
    "BIG": ["A big thing.", "It is big.", "Something big."],
    "SMALL": ["A small thing.", "It is small.", "Something small."],

    # Mental predicates
    "THINK": ["I think this.", "People think.", "Think about it."],
    "KNOW": ["I know this.", "People know things.", "Know the truth."],
    "WANT": ["I want this.", "People want things.", "Want something."],
    "FEEL": ["I feel this.", "People feel.", "Feel something."],
    "SEE": ["I see this.", "People see things.", "See it now."],
    "HEAR": ["I hear this.", "People hear sounds.", "Hear it now."],

    # Speech
    "SAY": ["I say this.", "People say things.", "Say something."],
    "WORDS": ["These are words.", "Words have meaning.", "Say words."],
    "TRUE": ["This is true.", "True things.", "It is true."],

    # Actions/Events
    "DO": ["I do this.", "People do things.", "Do something."],
    "HAPPEN": ["It happened.", "Things happen.", "Something happened."],
    "MOVE": ["It moves.", "Things move.", "Move now."],

    # Existence/Possession
    "THERE_IS": ["There is something.", "There is a thing.", "There is one."],
    "HAVE": ["I have this.", "People have things.", "Have something."],

    # Life/Death
    "LIVE": ["People live.", "Things live.", "Live here."],
    "DIE": ["People die.", "Things die.", "It died."],

    # Time
    "WHEN": ["When it happened.", "When you came.", "When is it."],
    "NOW": ["It is now.", "Now is the time.", "Do it now."],
    "BEFORE": ["Before this.", "It was before.", "Before now."],
    "AFTER": ["After this.", "It was after.", "After now."],
    "A_LONG_TIME": ["A long time ago.", "For a long time.", "A long time passed."],
    "A_SHORT_TIME": ["A short time ago.", "For a short time.", "A short time passed."],

    # Space
    "WHERE": ["Where is it.", "Where you are.", "Where it happened."],
    "HERE": ["It is here.", "Here now.", "Come here."],
    "ABOVE": ["It is above.", "Above this.", "Look above."],
    "BELOW": ["It is below.", "Below this.", "Look below."],
    "FAR": ["It is far.", "Far from here.", "Go far."],
    "NEAR": ["It is near.", "Near here.", "Come near."],
    "SIDE": ["On this side.", "The other side.", "Side by side."],
    "INSIDE": ["It is inside.", "Inside this.", "Go inside."],

    # Logical
    "NOT": ["It is not.", "Not this.", "Not now."],
    "MAYBE": ["Maybe it is.", "Maybe not.", "Maybe so."],
    "CAN": ["I can do this.", "People can.", "Can it happen."],
    "BECAUSE": ["Because of this.", "It happened because.", "Because I want."],
    "IF": ["If this happens.", "If you want.", "If it is true."],

    # Intensifier
    "VERY": ["Very good.", "Very big.", "Very much."],
    "MORE": ["More of this.", "More than that.", "Want more."],

    # Similarity
    "LIKE": ["Like this.", "It is like that.", "Something like it."],
}

# Derived/Composite concepts (built from primes)
# These should require more dimensions to represent
DERIVED_CONCEPTS = {
    # Emotions (complex combinations of FEEL + evaluators + mental states)
    "JEALOUSY": [
        "She felt jealous of her friend's success.",
        "Jealousy consumed his thoughts.",
        "The jealousy was overwhelming.",
    ],
    "NOSTALGIA": [
        "Nostalgia for the old days.",
        "She felt nostalgic about her childhood.",
        "A wave of nostalgia hit him.",
    ],
    "AMBIVALENCE": [
        "He felt ambivalent about the decision.",
        "Her ambivalence was evident.",
        "Ambivalence paralyzed him.",
    ],
    "SCHADENFREUDE": [
        "He felt schadenfreude at his rival's failure.",
        "A guilty sense of schadenfreude.",
        "Schadenfreude is a complex emotion.",
    ],

    # Abstract social concepts
    "DEMOCRACY": [
        "Democracy requires participation.",
        "The democracy was fragile.",
        "They fought for democracy.",
    ],
    "BUREAUCRACY": [
        "The bureaucracy was slow.",
        "Bureaucracy creates inefficiency.",
        "Navigate the bureaucracy.",
    ],
    "CAPITALISM": [
        "Capitalism drives innovation.",
        "The capitalism was unregulated.",
        "Critique of capitalism.",
    ],

    # Technical/Scientific
    "PHOTOSYNTHESIS": [
        "Photosynthesis converts light to energy.",
        "Plants perform photosynthesis.",
        "The process of photosynthesis.",
    ],
    "THERMODYNAMICS": [
        "Thermodynamics governs energy transfer.",
        "The laws of thermodynamics.",
        "Thermodynamics is fundamental.",
    ],
    "QUANTUM_ENTANGLEMENT": [
        "Quantum entanglement connects particles.",
        "The mystery of quantum entanglement.",
        "Measuring quantum entanglement.",
    ],

    # Complex actions
    "PROCRASTINATION": [
        "Procrastination delayed the project.",
        "He struggled with procrastination.",
        "Overcoming procrastination is hard.",
    ],
    "COLLABORATION": [
        "Collaboration improved the outcome.",
        "The collaboration was successful.",
        "Effective collaboration requires trust.",
    ],

    # Philosophical
    "EXISTENTIALISM": [
        "Existentialism emphasizes individual existence.",
        "The philosophy of existentialism.",
        "Existentialism questions meaning.",
    ],
    "EPISTEMOLOGY": [
        "Epistemology studies knowledge.",
        "The epistemology of science.",
        "Epistemological questions arise.",
    ],
}


def load_model(path: str):
    """Load a model."""
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

    pooled = mx.mean(h, axis=(0, 1))
    mx.eval(pooled)
    return pooled


def get_activations(model, tokenizer, probes: list[str], layer_idx: int):
    """Get activations for probes."""
    import mlx.core as mx

    activations = []
    for probe in probes:
        act = get_layer_activation(model, tokenizer, probe, layer_idx)
        if act is not None:
            activations.append(act)

    if not activations:
        return None

    stacked = mx.stack(activations, axis=0)
    stacked = stacked.astype(mx.float32)
    mx.eval(stacked)
    return np.array(stacked)


def analyze_primes_vs_derived(model, tokenizer, n_layers: int, model_name: str):
    """Compare prime vs derived concept representation at bottleneck."""

    bottleneck_layer = n_layers // 2
    logger.info(f"Analyzing {model_name} at bottleneck layer {bottleneck_layer}/{n_layers}")

    # Collect prime probes
    prime_probes = []
    prime_labels = []
    for concept, probes in NSM_PRIMES.items():
        prime_probes.extend(probes)
        prime_labels.extend([concept] * len(probes))

    # Collect derived probes
    derived_probes = []
    derived_labels = []
    for concept, probes in DERIVED_CONCEPTS.items():
        derived_probes.extend(probes)
        derived_labels.extend([concept] * len(probes))

    logger.info(f"Prime probes: {len(prime_probes)}, Derived probes: {len(derived_probes)}")

    # Get activations
    prime_acts = get_activations(model, tokenizer, prime_probes, bottleneck_layer)
    derived_acts = get_activations(model, tokenizer, derived_probes, bottleneck_layer)

    if prime_acts is None or derived_acts is None:
        return None

    # Combine for joint PCA
    all_acts = np.vstack([prime_acts, derived_acts])
    n_primes = len(prime_acts)
    n_derived = len(derived_acts)

    # Compute Gram matrix
    G = all_acts @ all_acts.T

    # SVD
    U, S, Vt = np.linalg.svd(G, full_matrices=False)

    # Effective rank
    threshold = S[0] * 3.45e-4
    effective_rank = int(np.sum(S > threshold))
    logger.info(f"Gram effective rank: {effective_rank}")

    # Measure projection strength onto top PCs
    # For each probe, compute |projection onto PC_i|

    prime_projections = []
    derived_projections = []

    for pc in range(min(5, effective_rank + 2)):
        # Prime projections onto this PC
        prime_proj = np.abs(U[:n_primes, pc])
        derived_proj = np.abs(U[n_primes:, pc])

        prime_projections.append({
            "mean": float(np.mean(prime_proj)),
            "std": float(np.std(prime_proj)),
            "max": float(np.max(prime_proj)),
        })
        derived_projections.append({
            "mean": float(np.mean(derived_proj)),
            "std": float(np.std(derived_proj)),
            "max": float(np.max(derived_proj)),
        })

    # Compute "centrality" - how much of the variance is captured by top PCs
    # For primes vs derived

    # Reconstruction error with k PCs
    def reconstruction_error(U_subset, S, k):
        """Fraction of variance NOT explained by top k PCs."""
        if k >= len(S):
            return 0.0
        explained = np.sum(S[:k])
        total = np.sum(S)
        return 1.0 - (explained / total)

    # Compute per-group explained variance
    # Project each group onto the shared PCs and see how much variance is captured

    prime_U = U[:n_primes, :]
    derived_U = U[n_primes:, :]

    # Variance explained by top k PCs for each group
    prime_var_explained = []
    derived_var_explained = []

    for k in [1, 2, 3, 5, 10]:
        if k > len(S):
            continue

        # For primes: how much of their variance is in top k PCs
        prime_in_topk = np.sum(prime_U[:, :k] ** 2) / np.sum(prime_U ** 2)
        derived_in_topk = np.sum(derived_U[:, :k] ** 2) / np.sum(derived_U ** 2)

        prime_var_explained.append({"k": k, "var_explained": float(prime_in_topk)})
        derived_var_explained.append({"k": k, "var_explained": float(derived_in_topk)})

        logger.info(f"Top {k} PCs: Primes={prime_in_topk:.3f}, Derived={derived_in_topk:.3f}")

    # Statistical test: do primes have higher projection onto top PCs?
    prime_top3_proj = np.mean(np.abs(prime_U[:, :3]), axis=1)
    derived_top3_proj = np.mean(np.abs(derived_U[:, :3]), axis=1)

    # Cohen's d for projection strength
    pooled_std = np.sqrt((np.std(prime_top3_proj)**2 + np.std(derived_top3_proj)**2) / 2)
    cohens_d = (np.mean(prime_top3_proj) - np.mean(derived_top3_proj)) / (pooled_std + 1e-10)

    logger.info(f"Prime vs Derived top-3 PC projection: Cohen's d = {cohens_d:.3f}")
    logger.info(f"  Primes mean: {np.mean(prime_top3_proj):.4f}")
    logger.info(f"  Derived mean: {np.mean(derived_top3_proj):.4f}")

    return {
        "model": model_name,
        "bottleneck_layer": bottleneck_layer,
        "n_layers": n_layers,
        "n_prime_probes": n_primes,
        "n_derived_probes": n_derived,
        "gram_effective_rank": effective_rank,
        "singular_values": [float(s) for s in S[:10]],
        "prime_projections": prime_projections,
        "derived_projections": derived_projections,
        "prime_var_explained": prime_var_explained,
        "derived_var_explained": derived_var_explained,
        "cohens_d_top3_projection": float(cohens_d),
        "prime_top3_mean": float(np.mean(prime_top3_proj)),
        "derived_top3_mean": float(np.mean(derived_top3_proj)),
    }


def main():
    models = {
        "SmolLM-135M": str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M"),
        "LFM2-350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        "Qwen2.5-0.5B": "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
    }

    results = {"models": {}}

    for name, path in models.items():
        logger.info("=" * 60)
        logger.info(f"ANALYZING {name}")
        logger.info("=" * 60)

        try:
            model, tokenizer, n_layers = load_model(path)
            result = analyze_primes_vs_derived(model, tokenizer, n_layers, name)
            if result:
                results["models"][name] = result
        except Exception as e:
            logger.error(f"Failed to analyze {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Do Primes Have More Weight at the Highway?")
    logger.info("=" * 60)

    for model_name, model_results in results["models"].items():
        d = model_results["cohens_d_top3_projection"]
        prime_mean = model_results["prime_top3_mean"]
        derived_mean = model_results["derived_top3_mean"]

        interpretation = "PRIMES STRONGER" if d > 0.2 else ("DERIVED STRONGER" if d < -0.2 else "NO DIFFERENCE")

        logger.info(f"\n{model_name}:")
        logger.info(f"  Cohen's d: {d:.3f} ({interpretation})")
        logger.info(f"  Prime mean projection: {prime_mean:.4f}")
        logger.info(f"  Derived mean projection: {derived_mean:.4f}")

        # Variance explained comparison
        for pve, dve in zip(model_results["prime_var_explained"],
                           model_results["derived_var_explained"]):
            k = pve["k"]
            logger.info(f"  Top {k} PCs: Primes={pve['var_explained']:.1%}, Derived={dve['var_explained']:.1%}")

    # Save
    output_path = Path(__file__).parent / "nsm_primes_highway_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
