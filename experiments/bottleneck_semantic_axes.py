#!/usr/bin/env python3
"""Bottleneck Semantic Axes Analysis.

Question: What do the 2-3 universal bottleneck dimensions encode?

Approach:
1. Collect activations at the bottleneck (50% depth)
2. Compute principal components of the Gram matrix
3. Project semantically-labeled probes onto these axes
4. See which categories separate on which axis

Hypothesis from docs/research/dimensional_hierarchy.md:
- PC1: Abstract ↔ Concrete
- PC2: Animate ↔ Static
- PC3: Natural ↔ Artificial
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

# Semantically labeled probes - designed to test hypothesized axes
SEMANTIC_PROBES = {
    # Abstract vs Concrete (hypothesized PC1)
    "abstract": [
        "Love is a complex emotion.",
        "Justice requires balance.",
        "Freedom means different things.",
        "Truth is hard to define.",
        "Beauty lies in perception.",
        "Hope sustains us.",
        "Fear controls behavior.",
        "Wisdom comes with age.",
    ],
    "concrete": [
        "The red apple sits on the table.",
        "A wooden chair has four legs.",
        "The metal key opens the door.",
        "Water fills the glass container.",
        "The brick wall is ten feet tall.",
        "A rubber ball bounces high.",
        "The stone bridge crosses the river.",
        "Paper burns quickly in fire.",
    ],

    # Animate vs Static (hypothesized PC2)
    "animate": [
        "The dog runs through the park.",
        "She dances gracefully on stage.",
        "Birds fly south for winter.",
        "The child laughs with joy.",
        "Fish swim in the ocean.",
        "He walks to work every day.",
        "Cats hunt mice at night.",
        "The athlete sprints to the finish.",
    ],
    "static": [
        "The mountain stands tall.",
        "A painting hangs on the wall.",
        "The building has fifty floors.",
        "Stars fill the night sky.",
        "The desert stretches for miles.",
        "A statue stands in the square.",
        "The lake reflects the clouds.",
        "Snow covers the ground.",
    ],

    # Natural vs Artificial (hypothesized PC3)
    "natural": [
        "Trees grow in the forest.",
        "Rivers flow to the sea.",
        "Flowers bloom in spring.",
        "Thunder rumbles in storms.",
        "Grass covers the meadow.",
        "Waves crash on the shore.",
        "Leaves fall in autumn.",
        "Fire burns the dry wood.",
    ],
    "artificial": [
        "The computer processes data.",
        "Cars drive on highways.",
        "Machines manufacture products.",
        "Electricity powers the city.",
        "Robots assemble components.",
        "Screens display information.",
        "Engines burn fuel efficiently.",
        "Circuits control the device.",
    ],

    # Additional semantic dimensions to test
    "temporal_past": [
        "The dinosaurs went extinct.",
        "Ancient Rome fell long ago.",
        "Yesterday it rained heavily.",
        "The war ended decades ago.",
    ],
    "temporal_future": [
        "Tomorrow will bring change.",
        "Next year we will travel.",
        "Soon the project will finish.",
        "Eventually all things end.",
    ],
    "positive": [
        "Success brings happiness.",
        "Victory feels wonderful.",
        "Love conquers all obstacles.",
        "Joy fills the heart.",
    ],
    "negative": [
        "Failure causes disappointment.",
        "Defeat feels terrible.",
        "Hate destroys relationships.",
        "Sadness weighs heavily.",
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


def get_activations_for_category(model, tokenizer, probes: list[str], layer_idx: int):
    """Get activations for a list of probes."""
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


def analyze_bottleneck_axes(model, tokenizer, n_layers: int, model_name: str):
    """Analyze what the bottleneck axes encode."""

    # Get bottleneck layer (50% depth)
    bottleneck_layer = n_layers // 2
    logger.info(f"Analyzing {model_name} at layer {bottleneck_layer}/{n_layers}")

    # Collect activations for all semantic categories
    all_activations = []
    all_labels = []
    category_indices = {}

    idx = 0
    for category, probes in SEMANTIC_PROBES.items():
        acts = get_activations_for_category(model, tokenizer, probes, bottleneck_layer)
        if acts is not None:
            category_indices[category] = (idx, idx + len(acts))
            all_activations.append(acts)
            all_labels.extend([category] * len(acts))
            idx += len(acts)

    if not all_activations:
        return None

    # Stack all activations
    X = np.vstack(all_activations)
    logger.info(f"Total probes: {X.shape[0]}, Hidden dim: {X.shape[1]}")

    # Compute Gram matrix
    G = X @ X.T

    # SVD of Gram matrix
    U, S, Vt = np.linalg.svd(G, full_matrices=False)

    # Effective rank
    threshold = S[0] * 3.45e-4
    effective_rank = int(np.sum(S > threshold))
    logger.info(f"Gram effective rank: {effective_rank}")

    # Project onto top principal components (in Gram/probe space)
    # U columns are the principal directions in probe space
    projections = {}
    for category, (start, end) in category_indices.items():
        # Get mean projection onto each PC for this category
        pc_coords = []
        for pc in range(min(5, effective_rank + 2)):  # Top 5 PCs
            # Project category's samples onto this PC
            category_proj = U[start:end, pc]
            pc_coords.append({
                "mean": float(np.mean(category_proj)),
                "std": float(np.std(category_proj)),
            })
        projections[category] = pc_coords

    # Compute separability scores for hypothesized axes
    separability = {}

    # PC1: Abstract vs Concrete?
    if "abstract" in projections and "concrete" in projections:
        for pc in range(min(3, len(projections["abstract"]))):
            abs_mean = projections["abstract"][pc]["mean"]
            con_mean = projections["concrete"][pc]["mean"]
            abs_std = projections["abstract"][pc]["std"]
            con_std = projections["concrete"][pc]["std"]

            # Cohen's d (effect size)
            pooled_std = np.sqrt((abs_std**2 + con_std**2) / 2)
            d = abs(abs_mean - con_mean) / (pooled_std + 1e-10)
            separability[f"PC{pc+1}_abstract_vs_concrete"] = float(d)

    # PC2: Animate vs Static?
    if "animate" in projections and "static" in projections:
        for pc in range(min(3, len(projections["animate"]))):
            ani_mean = projections["animate"][pc]["mean"]
            sta_mean = projections["static"][pc]["mean"]
            ani_std = projections["animate"][pc]["std"]
            sta_std = projections["static"][pc]["std"]

            pooled_std = np.sqrt((ani_std**2 + sta_std**2) / 2)
            d = abs(ani_mean - sta_mean) / (pooled_std + 1e-10)
            separability[f"PC{pc+1}_animate_vs_static"] = float(d)

    # PC3: Natural vs Artificial?
    if "natural" in projections and "artificial" in projections:
        for pc in range(min(3, len(projections["natural"]))):
            nat_mean = projections["natural"][pc]["mean"]
            art_mean = projections["artificial"][pc]["mean"]
            nat_std = projections["natural"][pc]["std"]
            art_std = projections["artificial"][pc]["std"]

            pooled_std = np.sqrt((nat_std**2 + art_std**2) / 2)
            d = abs(nat_mean - art_mean) / (pooled_std + 1e-10)
            separability[f"PC{pc+1}_natural_vs_artificial"] = float(d)

    # Temporal: Past vs Future?
    if "temporal_past" in projections and "temporal_future" in projections:
        for pc in range(min(3, len(projections["temporal_past"]))):
            past_mean = projections["temporal_past"][pc]["mean"]
            fut_mean = projections["temporal_future"][pc]["mean"]
            past_std = projections["temporal_past"][pc]["std"]
            fut_std = projections["temporal_future"][pc]["std"]

            pooled_std = np.sqrt((past_std**2 + fut_std**2) / 2)
            d = abs(past_mean - fut_mean) / (pooled_std + 1e-10)
            separability[f"PC{pc+1}_past_vs_future"] = float(d)

    # Valence: Positive vs Negative?
    if "positive" in projections and "negative" in projections:
        for pc in range(min(3, len(projections["positive"]))):
            pos_mean = projections["positive"][pc]["mean"]
            neg_mean = projections["negative"][pc]["mean"]
            pos_std = projections["positive"][pc]["std"]
            neg_std = projections["negative"][pc]["std"]

            pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
            d = abs(pos_mean - neg_mean) / (pooled_std + 1e-10)
            separability[f"PC{pc+1}_positive_vs_negative"] = float(d)

    # Find which PC best separates each semantic dimension
    best_separations = {}
    dimensions = ["abstract_vs_concrete", "animate_vs_static", "natural_vs_artificial",
                  "past_vs_future", "positive_vs_negative"]

    for dim in dimensions:
        best_pc = None
        best_d = 0
        for key, d in separability.items():
            if dim in key and d > best_d:
                best_d = d
                best_pc = key.split("_")[0]
        if best_pc:
            best_separations[dim] = {"best_pc": best_pc, "cohens_d": best_d}

    return {
        "model": model_name,
        "bottleneck_layer": bottleneck_layer,
        "n_layers": n_layers,
        "gram_effective_rank": effective_rank,
        "singular_values": [float(s) for s in S[:10]],
        "variance_explained": [float(s/S.sum()) for s in S[:5]],
        "projections": projections,
        "separability": separability,
        "best_separations": best_separations,
    }


def main():
    # Test on multiple models
    models = {
        "SmolLM-135M": str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M"),
        "LFM2-350M": "/path/to/models/mlx-community/LFM2-350M-MLX-bf16",
        "Qwen2.5-0.5B": "/path/to/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
    }

    results = {"models": {}}

    for name, path in models.items():
        logger.info("=" * 60)
        logger.info(f"ANALYZING {name}")
        logger.info("=" * 60)

        try:
            model, tokenizer, n_layers = load_model(path)
            result = analyze_bottleneck_axes(model, tokenizer, n_layers, name)
            if result:
                results["models"][name] = result

                # Print key findings
                logger.info(f"\nBest separations for {name}:")
                for dim, info in result["best_separations"].items():
                    logger.info(f"  {dim}: {info['best_pc']} (d={info['cohens_d']:.2f})")
        except Exception as e:
            logger.error(f"Failed to analyze {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary across models
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-MODEL SUMMARY: Which PC encodes what?")
    logger.info("=" * 60)

    dimensions = ["abstract_vs_concrete", "animate_vs_static", "natural_vs_artificial"]

    for dim in dimensions:
        logger.info(f"\n{dim}:")
        for model_name, model_results in results["models"].items():
            if dim in model_results["best_separations"]:
                info = model_results["best_separations"][dim]
                logger.info(f"  {model_name}: {info['best_pc']} (d={info['cohens_d']:.2f})")

    # Save results
    output_path = Path(__file__).parent / "bottleneck_semantic_axes_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
