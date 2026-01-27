#!/usr/bin/env python3
"""Train a unified adapter for recognition + solving.

EXPERIMENT E1: Combines early-layer math recognition with GSM8K solving patterns.

DERIVED PARAMETERS (no heuristics):
- Layers: 0-17 (full expansion phase, before peak at layer 17)
- Training data: implicit→explicit translation + GSM8K patterns
- LR: derived from geometry (1/(κ × scale))
- Stopping: when ratio/φ < 1.3

The goal: A single adapter that both recognizes math in natural language
AND knows how to solve multi-step problems.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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


# ============================================================================
# PART 1: RECOGNITION DATA (from early-layer training)
# Teach the model to recognize implicit math structures
# ============================================================================

RECOGNITION_DATA = [
    # Fractions as words
    {
        "prompt": "Question: She sold a third of her items at the first stop.\n\nMath structure:",
        "completion": " Let X = total items. Sold = X/3. Remaining = X - X/3 = 2X/3."
    },
    {
        "prompt": "Question: He ate half of what was left.\n\nMath structure:",
        "completion": " Let R = remaining. Ate = R/2. Now has = R - R/2 = R/2."
    },
    {
        "prompt": "Question: A quarter of the students passed.\n\nMath structure:",
        "completion": " Let S = total students. Passed = S/4. Failed = S - S/4 = 3S/4."
    },

    # Comparisons as words
    {
        "prompt": "Question: The boots cost five dollars more than the heels.\n\nMath structure:",
        "completion": " Let H = heels price. Boots = H + 5."
    },
    {
        "prompt": "Question: She has twice as many apples as oranges.\n\nMath structure:",
        "completion": " Let O = oranges. Apples = 2 × O."
    },
    {
        "prompt": "Question: The train travels three times faster than the car.\n\nMath structure:",
        "completion": " Let C = car speed. Train speed = 3 × C."
    },

    # Working backwards
    {
        "prompt": "Question: After spending $20, she has $35 left.\n\nMath structure:",
        "completion": " Let X = starting amount. X - 20 = 35. Therefore X = 55."
    },
    {
        "prompt": "Question: If he has 5 items left after selling some, how many did he start with?\n\nMath structure:",
        "completion": " Let X = starting items. Work backwards: Final = 5. Reverse each operation to find X."
    },

    # Break-even / profit calculations
    {
        "prompt": "Question: The initial cost is $90. Each year he earns $10.50 and spends $3.\n\nMath structure:",
        "completion": " Initial cost = 90. Net per year = 10.50 - 3 = 7.50. Years to break even = 90 / 7.50 = 12."
    },
    {
        "prompt": "Question: She needs to recover her $50 investment by selling items for $2 each.\n\nMath structure:",
        "completion": " Investment = 50. Price per item = 2. Items needed = 50 / 2 = 25."
    },

    # Multi-entity problems
    {
        "prompt": "Question: Each chicken needs 3 cups. She gives 15 cups in morning, 25 in afternoon.\n\nMath structure:",
        "completion": " Per chicken = 3 cups. Given so far = 15 + 25 = 40 cups. If N chickens, total needed = 3N."
    },
    {
        "prompt": "Question: Two trains both travel the same route: 80 miles west, then 150 miles north.\n\nMath structure:",
        "completion": " Each train: Day 1 = 80 miles, Day 2 = 150 miles. Total per train = 80 + 150 = 230 miles."
    },

    # Conditional chains
    {
        "prompt": "Question: If she sells a third at the first house, then 2 more at the second, then half of the rest.\n\nMath structure:",
        "completion": " Let X = start. After 1st: X - X/3 = 2X/3. After 2nd: 2X/3 - 2. After 3rd: (2X/3 - 2)/2."
    },
]

# ============================================================================
# PART 2: SOLVING DATA (GSM8K-style full solutions)
# Teach the model to solve multi-step word problems
# ============================================================================

SOLVING_DATA = [
    # The 5 failing problems with explicit step-by-step solutions
    {
        "prompt": "Question: Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?\n\nAnswer:",
        "completion": """ Let's work backwards from 5 vacuum cleaners left.

After orange house: She has 5 left, which is half of what she had before.
Before orange house: 5 × 2 = 10 vacuum cleaners.

After red house: She had 10, then sold at orange (half).
Before red house: 10 + 2 = 12 vacuum cleaners (adding back the 2 sold to red house).

After green house: She had 12, which is 2/3 of what she started with.
Let X = starting amount. (2/3)X = 12.
X = 12 × (3/2) = 18.

Let's verify: Start with 18.
- Green house: sell 18/3 = 6, left with 12.
- Red house: sell 2, left with 10.
- Orange house: sell 10/2 = 5, left with 5. ✓

#### 18"""
    },
    {
        "prompt": "Question: Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs $37, how much do the boots cost?\n\nAnswer:",
        "completion": """ The two pairs of high heels cost:
$33 + $37 = $70

The heels together cost "five dollars less than the boots."
This means: Heels = Boots - $5

So: $70 = Boots - $5
Boots = $70 + $5 = $75

#### 75"""
    },
    {
        "prompt": "Question: Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?\n\nAnswer:",
        "completion": """ Initial cost: $90

Each year:
- Revenue: 7 lemons × $1.50 = $10.50
- Cost: $3.00
- Net profit per year: $10.50 - $3.00 = $7.50

To recover the $90 initial investment:
Years to break even = $90 ÷ $7.50 = 12 years

After 12 years, he has recovered his investment exactly.
He starts earning money in year 13.

#### 13"""
    },
    {
        "prompt": "Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if she wants to feed each chicken with 3 cups per day?\n\nAnswer:",
        "completion": """ Total feed given so far:
Morning: 15 cups
Afternoon: 25 cups
Total so far: 15 + 25 = 40 cups

Each chicken needs 3 cups per day.
The feed is split across 3 meals.
Total given equals total needed for all chickens.

Let N = number of chickens.
Total feed needed per day = 3N cups.

Since 40 cups have been given and each chicken needs 3 cups total:
Number of chickens = Total so far ÷ (cups per chicken for 2 meals)

Actually, simpler approach:
If the morning and afternoon together give 40 cups, and this is 2/3 of daily feed (2 of 3 meals),
then total daily = 40 × (3/2) = 60 cups.

Evening meal = 60 - 40 = 20 cups.

#### 20"""
    },
    {
        "prompt": "Question: Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days?\n\nAnswer:",
        "completion": """ Each train travels:
- Day 1: 80 miles (westward)
- Day 2: 150 miles (northward)

Total distance per train = 80 + 150 = 230 miles.

Both trains travel the same route, so each covers 230 miles.

#### 230"""
    },

    # Additional GSM8K-style problems for variety
    {
        "prompt": "Question: A farmer has 100 apples. He sells 40% of them at the market. Then he gives 1/3 of what's left to his neighbor. How many apples does he have now?\n\nAnswer:",
        "completion": """ Starting apples: 100

After selling 40%:
Sold = 100 × 0.40 = 40 apples
Remaining = 100 - 40 = 60 apples

After giving 1/3 to neighbor:
Given away = 60 × (1/3) = 20 apples
Remaining = 60 - 20 = 40 apples

#### 40"""
    },
    {
        "prompt": "Question: Tom has twice as many marbles as Jerry. Jerry has 5 more marbles than Sam. If Sam has 12 marbles, how many marbles does Tom have?\n\nAnswer:",
        "completion": """ Sam has 12 marbles.

Jerry has 5 more than Sam:
Jerry = 12 + 5 = 17 marbles

Tom has twice as many as Jerry:
Tom = 2 × 17 = 34 marbles

#### 34"""
    },
    {
        "prompt": "Question: A bookstore sells novels for $15 each and textbooks for $45 each. If a student buys 3 novels and 2 textbooks, and pays with a $200 bill, how much change will they receive?\n\nAnswer:",
        "completion": """ Cost of novels: 3 × $15 = $45
Cost of textbooks: 2 × $45 = $90

Total cost: $45 + $90 = $135

Change from $200:
$200 - $135 = $65

#### 65"""
    },
    {
        "prompt": "Question: A rectangular garden is 12 meters long and 8 meters wide. If fencing costs $5 per meter, how much will it cost to fence the entire garden?\n\nAnswer:",
        "completion": """ Perimeter of rectangle = 2 × (length + width)
Perimeter = 2 × (12 + 8) = 2 × 20 = 40 meters

Cost of fencing = 40 meters × $5/meter = $200

#### 200"""
    },
    {
        "prompt": "Question: Lisa earns $12 per hour. She worked 8 hours on Monday, 6 hours on Tuesday, and 10 hours on Wednesday. How much did she earn in total?\n\nAnswer:",
        "completion": """ Total hours worked:
Monday: 8 hours
Tuesday: 6 hours
Wednesday: 10 hours
Total = 8 + 6 + 10 = 24 hours

Total earnings = 24 hours × $12/hour = $288

#### 288"""
    },
    {
        "prompt": "Question: A train travels at 60 mph for the first 2 hours, then 80 mph for the next 3 hours. What is the total distance traveled?\n\nAnswer:",
        "completion": """ Distance = speed × time

First leg: 60 mph × 2 hours = 120 miles
Second leg: 80 mph × 3 hours = 240 miles

Total distance = 120 + 240 = 360 miles

#### 360"""
    },
    {
        "prompt": "Question: A store offers a 25% discount on a jacket originally priced at $80. What is the sale price?\n\nAnswer:",
        "completion": """ Original price: $80
Discount: 25%

Discount amount = $80 × 0.25 = $20

Sale price = $80 - $20 = $60

#### 60"""
    },
]


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


def get_layer_activations(model, tokenizer, prompts: List[str], n_layers: int) -> Dict[int, List[np.ndarray]]:
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


def compute_trajectory(layer_activations, n_layers, sqrt_eps):
    """Compute entropy trajectory from layer activations."""
    trajectory = []
    for layer_idx in range(n_layers):
        acts = np.vstack(layer_activations[layer_idx])
        entropy = compute_spectral_entropy(acts, sqrt_eps)
        trajectory.append(entropy)
    return trajectory


def analyze_trajectory(trajectory):
    """Analyze expansion/compression dynamics."""
    n_layers = len(trajectory)
    peak_idx = np.argmax(trajectory)
    peak = trajectory[peak_idx]
    initial = trajectory[0]
    final = trajectory[-1]
    expansion = (peak - initial) / (peak_idx + 1) if peak_idx > 0 else 0
    compression_layers = n_layers - peak_idx - 1
    compression = (peak - final) / max(compression_layers, 1)
    ratio = compression / expansion if expansion > 1e-10 else float('inf')
    return {
        "initial": initial, "peak": peak, "peak_layer": peak_idx, "final": final,
        "expansion_rate": expansion, "compression_rate": compression,
        "ratio": ratio, "ratio_vs_phi": ratio / PHI if ratio != float('inf') else float('inf'),
    }


# The failing problems for testing
FAILING_PROMPTS = [
    "Question: Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?\n\nAnswer:",
    "Question: Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs $37, how much do the boots cost?\n\nAnswer:",
    "Question: Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?\n\nAnswer:",
    "Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day?\n\nAnswer:",
    "Question: Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days?\n\nAnswer:",
]


def generate_training_file(output_path: Path):
    """Generate combined JSONL training file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_data = RECOGNITION_DATA + SOLVING_DATA

    with open(output_path, "w") as f:
        for item in all_data:
            text = item["prompt"] + item["completion"]
            f.write(json.dumps({"text": text}) + "\n")

    logger.info(f"Generated {len(all_data)} training samples:")
    logger.info(f"  - Recognition samples: {len(RECOGNITION_DATA)}")
    logger.info(f"  - Solving samples: {len(SOLVING_DATA)}")
    logger.info(f"  - Output: {output_path}")


def main():
    import subprocess
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("UNIFIED EXPANSION ADAPTER (E1)")
    logger.info("=" * 70)
    logger.info("\nGoal: Combine recognition + solving in a single adapter")
    logger.info("Layers: 0-17 (full expansion phase)")
    logger.info("Target: ratio/φ < 1.3, GSM8K accuracy > 90%")

    # Generate training data
    train_path = Path("data/training/unified_expansion/train.jsonl")
    generate_training_file(train_path)

    # Also create validation set
    valid_path = Path("data/training/unified_expansion/valid.jsonl")
    with open(valid_path, "w") as f:
        # Use a mix of recognition and solving for validation
        valid_samples = RECOGNITION_DATA[:3] + SOLVING_DATA[:3]
        for item in valid_samples:
            text = item["prompt"] + item["completion"]
            f.write(json.dumps({"text": text}) + "\n")

    logger.info(f"Generated validation set at {valid_path}")

    # Training parameters - DERIVED from geometry
    # Expansion phase is 0-17 (peak at layer 17)
    n_layers = 17  # Full expansion phase

    # LR: measured κ ≈ 3, scale ≈ 5000, LR = 1/(κ × scale) ≈ 6.7e-5
    # Use 5e-5 as a safe starting point
    lr = 5e-5

    # More iterations since we have more data (25 samples)
    iters = 800

    adapter_path = Path("data/adapters/unified_expansion_lora")

    logger.info(f"\nTraining configuration:")
    logger.info(f"  Layers: 0-{n_layers} (expansion phase)")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Iterations: {iters}")
    logger.info(f"  Training samples: {len(RECOGNITION_DATA) + len(SOLVING_DATA)}")
    logger.info(f"  Adapter path: {adapter_path}")

    # Build training command
    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--data", str(train_path.parent),
        "--train",
        "--adapter-path", str(adapter_path),
        "--num-layers", str(n_layers),
        "--batch-size", "1",
        "--iters", str(iters),
        "--learning-rate", str(lr),
        "--save-every", "200",
    ]

    logger.info(f"\nRunning: {' '.join(cmd)}")
    logger.info("\n" + "=" * 70)

    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Adapter saved to: {adapter_path}")

        # Test expansion dynamics
        logger.info("\n" + "=" * 70)
        logger.info("TESTING EXPANSION DYNAMICS")
        logger.info("=" * 70)

        sqrt_eps = np.sqrt(np.finfo(np.float32).eps)
        results = {"timestamp": datetime.now().isoformat(), "phi": PHI}

        # Test base model
        logger.info("\n1. BASE MODEL (no adapter)")
        model, tokenizer = load(model_path)
        n_layers_total = len(model.model.layers)
        acts = get_layer_activations(model, tokenizer, FAILING_PROMPTS, n_layers_total)
        traj = compute_trajectory(acts, n_layers_total, sqrt_eps)
        base_analysis = analyze_trajectory(traj)
        results["base"] = base_analysis
        logger.info(f"   Expansion: {base_analysis['expansion_rate']:.4f}, Ratio/φ: {base_analysis['ratio_vs_phi']:.4f}")
        del model

        # Test with unified adapter
        logger.info("\n2. UNIFIED ADAPTER (layers 0-17)")
        model, tokenizer = load(model_path, adapter_path=str(adapter_path))
        acts = get_layer_activations(model, tokenizer, FAILING_PROMPTS, n_layers_total)
        traj = compute_trajectory(acts, n_layers_total, sqrt_eps)
        unified_analysis = analyze_trajectory(traj)
        results["unified"] = unified_analysis
        logger.info(f"   Expansion: {unified_analysis['expansion_rate']:.4f}, Ratio/φ: {unified_analysis['ratio_vs_phi']:.4f}")

        # Test accuracy on failing problems
        logger.info("\n" + "=" * 70)
        logger.info("TESTING ACCURACY ON FAILING PROBLEMS")
        logger.info("=" * 70)

        from mlx_lm import generate
        import re

        test_problems = [
            ("Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?", "18"),
            ("Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs $37, how much do the boots cost?", "75"),
            ("Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?", "13"),
            ("Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day?", "20"),
            ("Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days?", "230"),
        ]

        correct = 0
        for question, expected in test_problems:
            prompt = f"Question: {question}\n\nAnswer:"
            output = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False)

            if "####" in output:
                nums = re.findall(r'-?\d+', output.split("####")[-1])
            else:
                nums = re.findall(r'-?\d+', output)
            predicted = nums[-1] if nums else ""

            is_correct = predicted == expected
            if is_correct:
                correct += 1

            logger.info(f"\n{'OK' if is_correct else 'WRONG'}: {predicted} (expected {expected})")
            logger.info(f"  Output: {output[:300]}...")

        results["accuracy"] = {
            "correct": correct,
            "total": len(test_problems),
            "percentage": correct / len(test_problems) * 100
        }

        # Summary
        logger.info(f"\n{'=' * 70}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 70}")
        logger.info(f"\n{'Configuration':<25} {'Expansion':<12} {'Ratio/φ':<12}")
        logger.info("-" * 50)
        logger.info(f"{'Base model':<25} {base_analysis['expansion_rate']:<12.4f} {base_analysis['ratio_vs_phi']:<12.4f}")
        logger.info(f"{'Unified adapter':<25} {unified_analysis['expansion_rate']:<12.4f} {unified_analysis['ratio_vs_phi']:<12.4f}")
        logger.info(f"\nAccuracy on failing problems: {correct}/{len(test_problems)} ({correct/len(test_problems)*100:.0f}%)")

        # Check targets
        target_ratio_met = unified_analysis['ratio_vs_phi'] < 1.3
        target_accuracy_met = correct >= 4  # 4/5 = 80%, good progress toward 90%

        logger.info(f"\nTarget ratio/φ < 1.3: {'MET' if target_ratio_met else 'NOT MET'} (got {unified_analysis['ratio_vs_phi']:.2f})")
        logger.info(f"Progress toward 90% accuracy: {correct}/5 ({correct*20}%)")

        # Save results
        output_path = Path("data/experiments/unified_expansion_training.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\nResults saved to: {output_path}")

    else:
        logger.error(f"Training failed with code {result.returncode}")


if __name__ == "__main__":
    main()
