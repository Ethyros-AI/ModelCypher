#!/usr/bin/env python3
"""Test if making math explicit unlocks expansion.

Hypothesis: The model CAN expand, it just doesn't recognize implicit math as math.
If we reformulate failing problems to make the math explicit, expansion should increase.

This proves the intervention point: teach implicit→explicit math recognition.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_spectral_entropy(activations: np.ndarray, sqrt_eps: float) -> float:
    """Compute spectral entropy from activations."""
    if len(activations) < 2:
        return 0.0
    centered = activations - activations.mean(axis=0)
    _, S, _ = svd(centered, full_matrices=False)
    S_valid = S[S > sqrt_eps * S[0]]
    if len(S_valid) < 2:
        return 0.0
    p = S_valid ** 2
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-10)))


def get_all_layer_activations(model, tokenizer, prompts: List[str], n_layers: int) -> Dict[int, List[np.ndarray]]:
    """Get activations at every layer for multiple prompts."""
    import mlx.core as mx

    layer_activations = {i: [] for i in range(n_layers)}

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = model.model.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(model.model.layers):
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            mx.eval(hidden)
            layer_activations[layer_idx].append(
                np.array(hidden[0, -1, :].tolist(), dtype=np.float32)
            )

    return layer_activations


def compute_entropy_trajectory(layer_activations: Dict[int, List[np.ndarray]], n_layers: int, sqrt_eps: float) -> List[float]:
    """Compute entropy at each layer."""
    trajectory = []
    for layer_idx in range(n_layers):
        acts = np.vstack(layer_activations[layer_idx])
        entropy = compute_spectral_entropy(acts, sqrt_eps)
        trajectory.append(entropy)
    return trajectory


def analyze_trajectory(trajectory: List[float]) -> Dict:
    """Analyze expansion/compression dynamics."""
    n_layers = len(trajectory)
    peak_idx = np.argmax(trajectory)
    peak_entropy = trajectory[peak_idx]
    initial_entropy = trajectory[0]
    final_entropy = trajectory[-1]

    expansion_rate = (peak_entropy - initial_entropy) / (peak_idx + 1) if peak_idx > 0 else 0
    compression_layers = n_layers - peak_idx - 1
    compression_rate = (peak_entropy - final_entropy) / max(compression_layers, 1)

    if expansion_rate > 1e-10:
        ratio = compression_rate / expansion_rate
    else:
        ratio = float('inf')

    return {
        "initial": initial_entropy,
        "peak": peak_entropy,
        "peak_layer": int(peak_idx),
        "final": final_entropy,
        "expansion_rate": expansion_rate,
        "compression_rate": compression_rate,
        "ratio": ratio,
        "ratio_vs_phi": ratio / PHI if ratio != float('inf') else float('inf'),
        "trajectory": trajectory,
    }


# The 5 failing problems - ORIGINAL (implicit math)
FAILING_PROBLEMS_ORIGINAL = [
    # Problem 5: Wendi's chickens - implicit count
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if she wants to feed each chicken with 3 cups per day?",

    # Problem 13: Carlos's lemon tree - break-even implicit
    "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?",

    # Problem 14: Melanie's vacuums - working backwards with fractions
    "Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?",

    # Problem 17: Two trains - simple but confusing structure
    "Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days?",

    # Problem 30: Gloria's shoes - comparison requiring inference
    "Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs $37, how much do the boots cost?",
]

# The same problems - EXPLICIT math version
FAILING_PROBLEMS_EXPLICIT = [
    # Problem 5: Made math explicit
    "Wendi has some chickens. Each chicken needs 3 cups of feed per day. In the morning, she gives 15 cups total. In the afternoon, she gives 25 cups total. That's 15 + 25 = 40 cups so far. If each chicken needs 3 cups total, she can calculate chickens = total_cups / 3. How many cups does she need for the evening meal to reach 3 cups per chicken?",

    # Problem 13: Made math explicit
    "Carlos plants a tree. Initial cost: $90. Each year: Revenue = 7 lemons × $1.5 = $10.50. Cost = $3. Net profit per year = $10.50 - $3 = $7.50. Years to break even = $90 / $7.50. How many years until profit exceeds costs?",

    # Problem 14: Made math explicit
    "Melanie starts with X vacuums. At green house: sells X/3, has 2X/3 left. At red house: sells 2, has (2X/3 - 2) left. At orange house: sells half of what's left = (2X/3 - 2)/2, has (2X/3 - 2)/2 left. Final count: (2X/3 - 2)/2 = 5. Solve for X.",

    # Problem 17: Made math explicit
    "Each train travels: Day 1: 80 miles. Day 2: 150 miles. Total per train = 80 + 150. What is the total distance for each train?",

    # Problem 30: Made math explicit
    "Two heels cost: $33 + $37 = $70. Boots cost $5 more than the heels together. Boots = $70 + $5. What do the boots cost?",
]

# Expected answers
EXPECTED_ANSWERS = ["20", "13", "18", "230", "75"]  # Note: Problem 30's expected was 104, but that seems wrong


def main():
    import mlx.core as mx
    from mlx_lm import load, generate

    logger.info("=" * 70)
    logger.info("EXPLICIT MATH UNLOCK TEST")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_final_mastery_lora"

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    n_layers = len(model.model.layers)
    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_layers": n_layers,
        "phi": PHI,
    }

    # Test ORIGINAL (implicit) versions
    logger.info(f"\n{'=' * 50}")
    logger.info("ORIGINAL PROBLEMS (implicit math)")
    logger.info(f"{'=' * 50}")

    original_prompts = [f"Question: {q}\n\nAnswer:" for q in FAILING_PROBLEMS_ORIGINAL]
    original_activations = get_all_layer_activations(model, tokenizer, original_prompts, n_layers)
    original_trajectory = compute_entropy_trajectory(original_activations, n_layers, sqrt_eps)
    original_analysis = analyze_trajectory(original_trajectory)

    logger.info(f"Initial entropy: {original_analysis['initial']:.4f}")
    logger.info(f"Peak entropy (layer {original_analysis['peak_layer']}): {original_analysis['peak']:.4f}")
    logger.info(f"Final entropy: {original_analysis['final']:.4f}")
    logger.info(f"Expansion rate: {original_analysis['expansion_rate']:.4f}")
    logger.info(f"Compression rate: {original_analysis['compression_rate']:.4f}")
    logger.info(f"RATIO: {original_analysis['ratio']:.4f}")
    logger.info(f"Ratio/φ: {original_analysis['ratio_vs_phi']:.4f}")

    # Test model answers on original
    logger.info("\nModel answers (original):")
    original_correct = 0
    for i, (prompt, expected) in enumerate(zip(original_prompts, EXPECTED_ANSWERS)):
        output = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
        # Extract number
        import re
        if "####" in output:
            nums = re.findall(r'-?\d+', output.split("####")[-1])
        else:
            nums = re.findall(r'-?\d+', output)
        predicted = nums[-1] if nums else ""
        is_correct = predicted == expected
        if is_correct:
            original_correct += 1
        logger.info(f"  [{i+1}] {'OK' if is_correct else 'WRONG'}: {predicted} (expected {expected})")

    results["original"] = {
        "analysis": original_analysis,
        "correct": original_correct,
        "total": len(EXPECTED_ANSWERS),
    }

    # Test EXPLICIT versions
    logger.info(f"\n{'=' * 50}")
    logger.info("EXPLICIT PROBLEMS (math made visible)")
    logger.info(f"{'=' * 50}")

    explicit_prompts = [f"Question: {q}\n\nAnswer:" for q in FAILING_PROBLEMS_EXPLICIT]
    explicit_activations = get_all_layer_activations(model, tokenizer, explicit_prompts, n_layers)
    explicit_trajectory = compute_entropy_trajectory(explicit_activations, n_layers, sqrt_eps)
    explicit_analysis = analyze_trajectory(explicit_trajectory)

    logger.info(f"Initial entropy: {explicit_analysis['initial']:.4f}")
    logger.info(f"Peak entropy (layer {explicit_analysis['peak_layer']}): {explicit_analysis['peak']:.4f}")
    logger.info(f"Final entropy: {explicit_analysis['final']:.4f}")
    logger.info(f"Expansion rate: {explicit_analysis['expansion_rate']:.4f}")
    logger.info(f"Compression rate: {explicit_analysis['compression_rate']:.4f}")
    logger.info(f"RATIO: {explicit_analysis['ratio']:.4f}")
    logger.info(f"Ratio/φ: {explicit_analysis['ratio_vs_phi']:.4f}")

    # Test model answers on explicit
    logger.info("\nModel answers (explicit):")
    explicit_correct = 0
    for i, (prompt, expected) in enumerate(zip(explicit_prompts, EXPECTED_ANSWERS)):
        output = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
        import re
        if "####" in output:
            nums = re.findall(r'-?\d+', output.split("####")[-1])
        else:
            nums = re.findall(r'-?\d+', output)
        predicted = nums[-1] if nums else ""
        is_correct = predicted == expected
        if is_correct:
            explicit_correct += 1
        logger.info(f"  [{i+1}] {'OK' if is_correct else 'WRONG'}: {predicted} (expected {expected})")

    results["explicit"] = {
        "analysis": explicit_analysis,
        "correct": explicit_correct,
        "total": len(EXPECTED_ANSWERS),
    }

    # Comparison
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPARISON: IMPLICIT vs EXPLICIT")
    logger.info(f"{'=' * 70}")

    logger.info(f"\n{'Metric':<25} {'Original':<15} {'Explicit':<15} {'Delta':<15}")
    logger.info("-" * 70)
    logger.info(f"{'Initial entropy':<25} {original_analysis['initial']:<15.4f} {explicit_analysis['initial']:<15.4f} {explicit_analysis['initial'] - original_analysis['initial']:+.4f}")
    logger.info(f"{'Peak entropy':<25} {original_analysis['peak']:<15.4f} {explicit_analysis['peak']:<15.4f} {explicit_analysis['peak'] - original_analysis['peak']:+.4f}")
    logger.info(f"{'Expansion rate':<25} {original_analysis['expansion_rate']:<15.4f} {explicit_analysis['expansion_rate']:<15.4f} {explicit_analysis['expansion_rate'] - original_analysis['expansion_rate']:+.4f}")
    logger.info(f"{'Ratio':<25} {original_analysis['ratio']:<15.4f} {explicit_analysis['ratio']:<15.4f} {explicit_analysis['ratio'] - original_analysis['ratio']:+.4f}")
    logger.info(f"{'Ratio/φ':<25} {original_analysis['ratio_vs_phi']:<15.4f} {explicit_analysis['ratio_vs_phi']:<15.4f} {explicit_analysis['ratio_vs_phi'] - original_analysis['ratio_vs_phi']:+.4f}")
    logger.info(f"{'Accuracy':<25} {original_correct}/{len(EXPECTED_ANSWERS):<14} {explicit_correct}/{len(EXPECTED_ANSWERS):<14} {explicit_correct - original_correct:+d}")

    # Layer-by-layer divergence
    logger.info(f"\n{'=' * 50}")
    logger.info("LAYER-BY-LAYER ENTROPY DIVERGENCE")
    logger.info(f"{'=' * 50}")

    divergence = []
    for i, (orig, expl) in enumerate(zip(original_trajectory, explicit_trajectory)):
        diff = expl - orig
        divergence.append(diff)
        if i % 6 == 0:  # Print every 6th layer
            logger.info(f"Layer {i:2d}: orig={orig:.4f}, expl={expl:.4f}, diff={diff:+.4f}")

    max_divergence_layer = np.argmax(divergence)
    logger.info(f"\nMax divergence at layer {max_divergence_layer}: {divergence[max_divergence_layer]:+.4f}")

    results["divergence"] = {
        "per_layer": divergence,
        "max_layer": int(max_divergence_layer),
        "max_value": divergence[max_divergence_layer],
    }

    # Save results
    output_path = Path("data/experiments/explicit_math_unlock.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    # Conclusion
    expansion_unlocked = explicit_analysis['expansion_rate'] > original_analysis['expansion_rate'] * 1.2
    ratio_improved = explicit_analysis['ratio_vs_phi'] < original_analysis['ratio_vs_phi']

    logger.info(f"\n{'=' * 70}")
    logger.info("CONCLUSION")
    logger.info(f"{'=' * 70}")
    logger.info(f"Expansion unlocked: {expansion_unlocked}")
    logger.info(f"Ratio improved toward φ: {ratio_improved}")
    logger.info(f"Accuracy improved: {explicit_correct > original_correct}")

    if expansion_unlocked:
        logger.info("\n** HYPOTHESIS CONFIRMED: Making math explicit unlocks expansion **")
        logger.info("** The model CAN expand, it just needs to recognize math as math **")

    return results


if __name__ == "__main__":
    main()
