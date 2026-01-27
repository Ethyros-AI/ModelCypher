#!/usr/bin/env python3
"""Find where LFM2-350M math breaks down.

Test progressively harder math to identify the failure boundary.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.neighbors import NearestNeighbors

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


# Progressive difficulty levels
MATH_LEVEL_1 = [  # Single-step arithmetic
    ("What is 5 + 3?", "8"),
    ("What is 12 - 7?", "5"),
    ("What is 4 × 6?", "24"),
    ("What is 20 ÷ 4?", "5"),
    ("What is 15 + 8?", "23"),
]

MATH_LEVEL_2 = [  # Two-step arithmetic
    ("What is 5 + 3 × 2?", "11"),
    ("What is (8 + 4) ÷ 3?", "4"),
    ("If you have 15 and add 7, then subtract 9, what do you get?", "13"),
    ("What is 6 × 4 + 5?", "29"),
    ("What is 100 - 35 - 25?", "40"),
]

MATH_LEVEL_3 = [  # Word problems with 2 steps
    ("John has $50. He buys a book for $12 and a pen for $3. How much does he have left?", "35"),
    ("A farmer has 24 chickens and 18 ducks. He sells 10 birds. How many does he have?", "32"),
    ("Lisa reads 15 pages per day. How many pages does she read in a week?", "105"),
    ("A store had 80 apples. They sold 35 and received 20 more. How many now?", "65"),
    ("Tom earns $8 per hour. He works 6 hours and spends $20. How much does he have?", "28"),
]

MATH_LEVEL_4 = [  # Multi-step reasoning
    ("A train travels at 60 km/h for 2 hours, then 80 km/h for 1.5 hours. What's the total distance?", "240"),
    ("If 3 workers can build a wall in 12 days, how many days for 6 workers?", "6"),
    ("A rectangle has perimeter 24cm and width 4cm. What's the length?", "8"),
    ("John is twice as old as Mary. Mary is 15. How old will John be in 5 years?", "35"),
    ("A shirt costs $40 after 20% discount. What was the original price?", "50"),
]

MATH_LEVEL_5 = [  # Complex reasoning
    ("If 5 machines make 5 widgets in 5 minutes, how many widgets do 100 machines make in 100 minutes?", "2000"),
    ("A snail climbs 3 feet up a wall each day but slides down 2 feet each night. How many days to reach 10 feet?", "8"),
    ("In a room of 23 people, what's the probability at least 2 share a birthday? (approximate %)", "50"),
    ("A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much is the ball?", "0.05"),
    ("If you fold a paper in half 10 times, how many layers?", "1024"),
]

MATH_LEVEL_6 = [  # GSM8K-style multi-step
    ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins for friends daily. She sells the rest for $2 each. How much does she make per day?", "18"),
    ("A farmer has 52 cows. He buys 17 more, then sells a quarter of all his cows. How many remain?", "51"),
    ("Beth has 72 marbles. She gives 1/3 to Ann, then half of what's left to Carl. How many does Beth have?", "24"),
    ("A pool fills at 3 gallons/minute and drains at 1 gallon/minute. Starting empty, how many gallons after 2 hours?", "240"),
    ("Train A leaves at 8am at 60mph. Train B leaves at 9am at 80mph same direction. How many hours until B catches A?", "3"),
]


def compute_intrinsic_dimension_twonn(X: np.ndarray) -> float:
    if len(X) < 10:
        return float('nan')
    k = min(3, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nn.kneighbors(X)
    d1, d2 = distances[:, 1], distances[:, 2]
    valid = d1 > 1e-10
    if valid.sum() < 5:
        return float('nan')
    mu = d2[valid] / d1[valid]
    mu = mu[mu > 1]
    if len(mu) < 5:
        return float('nan')
    return float(len(np.log(mu)) / np.sum(np.log(mu)))


def get_geometric_signature(model, tokenizer, prompt: str) -> Dict:
    import mlx.core as mx
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    trajectory = []
    emb_np = np.array(hidden[0].tolist())
    trajectory.append(compute_intrinsic_dimension_twonn(emb_np))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        act_np = np.array(hidden[0].tolist())
        trajectory.append(compute_intrinsic_dimension_twonn(act_np))

    traj = np.array(trajectory)
    valid = traj[~np.isnan(traj)]
    n_layers = len(model.model.layers)

    if len(valid) > 2:
        peak_idx = np.nanargmax(traj)
        peak_dim = traj[peak_idx]
        initial_dim = traj[0] if not np.isnan(traj[0]) else valid[0]
        final_dim = traj[-1] if not np.isnan(traj[-1]) else valid[-1]
        compression_ratio = peak_dim / final_dim if final_dim > 0.1 else float('nan')
        compression_vs_phi = compression_ratio / PHI if not np.isnan(compression_ratio) else float('nan')
        expansion_ratio = peak_dim / initial_dim if initial_dim > 0.1 else float('nan')
    else:
        peak_idx = -1
        compression_vs_phi = float('nan')
        expansion_ratio = float('nan')

    return {
        "peak_layer_pct": peak_idx / n_layers * 100 if peak_idx >= 0 else float('nan'),
        "compression_vs_phi": compression_vs_phi,
        "expansion_ratio": expansion_ratio,
    }


def check_answer(output: str, expected: str) -> bool:
    output_clean = output.lower().replace(",", "").replace("$", "")
    expected_clean = expected.lower().replace(",", "").replace("$", "")

    # Check exact match first
    if expected_clean in output_clean:
        return True

    # Extract numbers from output
    nums = re.findall(r'-?\d+\.?\d*', output_clean)

    # Try numeric match
    try:
        expected_num = float(expected_clean)
        if nums:
            for num in nums:
                try:
                    if abs(float(num) - expected_num) < 0.01:
                        return True
                except ValueError:
                    continue
    except ValueError:
        pass  # expected is not a number

    return False


def run_level(model, tokenizer, level_name: str, problems: List) -> Dict:
    from mlx_lm import generate

    results = []
    for question, expected in problems:
        prompt = f"Question: {question}\n\nAnswer (give just the number):"

        try:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
        except Exception as e:
            output = f"ERROR: {e}"

        is_correct = check_answer(output, expected)

        try:
            geo = get_geometric_signature(model, tokenizer, prompt)
        except:
            geo = {"peak_layer_pct": float('nan'), "compression_vs_phi": float('nan'), "expansion_ratio": float('nan')}

        results.append({
            "question": question,
            "expected": expected,
            "output": output[:150],
            "correct": is_correct,
            **geo,
        })

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) * 100

    correct_phi = [r["compression_vs_phi"] for r in results if r["correct"] and not np.isnan(r["compression_vs_phi"])]
    wrong_phi = [r["compression_vs_phi"] for r in results if not r["correct"] and not np.isnan(r["compression_vs_phi"])]

    return {
        "level": level_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "correct_comp_phi": np.mean(correct_phi) if correct_phi else float('nan'),
        "wrong_comp_phi": np.mean(wrong_phi) if wrong_phi else float('nan'),
        "details": results,
    }


def main():
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("LFM2-350M MATH DIFFICULTY PROGRESSION")
    logger.info("Finding where the geometry breaks")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "levels": [],
    }

    levels = [
        ("Level 1: Single-step", MATH_LEVEL_1),
        ("Level 2: Two-step arithmetic", MATH_LEVEL_2),
        ("Level 3: 2-step word problems", MATH_LEVEL_3),
        ("Level 4: Multi-step reasoning", MATH_LEVEL_4),
        ("Level 5: Complex reasoning", MATH_LEVEL_5),
        ("Level 6: GSM8K-style", MATH_LEVEL_6),
    ]

    for level_name, problems in levels:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"{level_name}")
        logger.info("=" * 50)

        level_result = run_level(model, tokenizer, level_name, problems)
        results["levels"].append(level_result)

        logger.info(f"  Accuracy: {level_result['accuracy']:.0f}% ({level_result['correct']}/{level_result['total']})")
        if not np.isnan(level_result['correct_comp_phi']):
            logger.info(f"  Correct comp/φ: {level_result['correct_comp_phi']:.2f}")
        if not np.isnan(level_result['wrong_comp_phi']):
            logger.info(f"  Wrong comp/φ: {level_result['wrong_comp_phi']:.2f}")

        # Show failures
        failures = [r for r in level_result["details"] if not r["correct"]]
        for f in failures:
            logger.info(f"  ✗ {f['question'][:50]}...")
            logger.info(f"    Expected: {f['expected']}, Got: {f['output'][:40]}...")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DIFFICULTY PROGRESSION SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Level':<30} {'Accuracy':<12} {'Correct φ':<12} {'Wrong φ'}")
    logger.info("-" * 70)

    for level in results["levels"]:
        acc = f"{level['accuracy']:.0f}%"
        cor = f"{level['correct_comp_phi']:.2f}" if not np.isnan(level['correct_comp_phi']) else "N/A"
        wrg = f"{level['wrong_comp_phi']:.2f}" if not np.isnan(level['wrong_comp_phi']) else "N/A"
        logger.info(f"{level['level']:<30} {acc:<12} {cor:<12} {wrg}")

    # Find the breakdown point
    logger.info("\n" + "=" * 70)
    logger.info("BREAKDOWN ANALYSIS")
    logger.info("=" * 70)

    for i, level in enumerate(results["levels"]):
        if level["accuracy"] < 80:
            logger.info(f"\n⚠️  BREAKDOWN at {level['level']}: {level['accuracy']:.0f}%")
            if not np.isnan(level['wrong_comp_phi']):
                logger.info(f"   Wrong answers use comp/φ = {level['wrong_comp_phi']:.2f}")
                if level['wrong_comp_phi'] > 1.5:
                    logger.info("   → Model is template-matching when it should compute")
                elif level['wrong_comp_phi'] < 0.7:
                    logger.info("   → Model under-compresses (not reaching answer)")
            break
    else:
        logger.info("\n✓ No major breakdown - model handles all levels reasonably")

    # Save
    output_path = Path("data/experiments/lfm2_350m_math_difficulty.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
