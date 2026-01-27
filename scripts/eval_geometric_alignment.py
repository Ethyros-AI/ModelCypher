#!/usr/bin/env python3
"""Evaluate geometric alignment training results.

Compare base model vs trained adapter on:
1. Accuracy by difficulty level
2. Geometric properties (comp/φ, peak timing)
3. Correlation between geometry and correctness
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


# Test problems across all difficulty levels
TEST_PROBLEMS = [
    # Level 1
    ("What is 8 + 5?", "13", 1),
    ("What is 15 - 7?", "8", 1),
    ("What is 9 × 3?", "27", 1),

    # Level 2
    ("What is 4 + 3 × 5?", "19", 2),
    ("What is (10 - 2) × 3?", "24", 2),
    ("What is 25 ÷ 5 + 3?", "8", 2),

    # Level 3
    ("Mark has 25 toys. He gives 8 to his sister. How many does he have?", "17", 3),
    ("A pen costs $3. How much for 7 pens?", "21", 3),
    ("There are 30 students. 12 are girls. How many are boys?", "18", 3),

    # Level 4
    ("A store has 150 items. It sells 45 and receives 30 more. How many now?", "135", 4),
    ("Tom works 6 hours at $12/hour. He spends $25. What's left?", "47", 4),
    ("A rectangle is 12cm by 5cm. What's the perimeter?", "34", 4),

    # Level 5
    ("A $60 item is 25% off. What's the sale price?", "45", 5),
    ("4 workers finish a job in 8 days. How many days for 2 workers?", "16", 5),
    ("A tank fills at 6 gal/min, drains at 2 gal/min. Net in 30 minutes?", "120", 5),

    # Level 6 (GSM8K-style)
    ("Janet has 20 eggs. She uses 5 for breakfast and 3 for baking. She sells the rest at $2 each. How much does she make?", "24", 6),
    ("A farmer has 80 sheep. He sells 1/4, then buys 15 more. How many now?", "75", 6),
    ("Train A goes 50mph. Train B goes 70mph and leaves 2 hours later. How many hours until B catches A?", "5", 6),
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


def get_metrics(model, tokenizer, prompt: str) -> Dict:
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

        comp_ratio = peak_dim / final_dim if final_dim > 0.1 else float('nan')
        comp_phi = comp_ratio / PHI if not np.isnan(comp_ratio) else float('nan')
        peak_pct = peak_idx / n_layers * 100
    else:
        peak_pct = float('nan')
        comp_phi = float('nan')

    return {"peak_pct": peak_pct, "comp_phi": comp_phi}


def evaluate_model(model, tokenizer, problems: List, model_name: str) -> Dict:
    from mlx_lm import generate

    results = []

    for question, expected, difficulty in problems:
        prompt = f"Question: {question}\n\nAnswer:"

        try:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
        except:
            output = "ERROR"

        # Check answer
        nums = re.findall(r'-?\d+\.?\d*', output.replace(",", ""))
        is_correct = False
        if nums:
            for num in nums:
                try:
                    if abs(float(num) - float(expected)) < 0.1:
                        is_correct = True
                        break
                except:
                    pass

        # Get geometry
        try:
            metrics = get_metrics(model, tokenizer, prompt)
        except:
            metrics = {"peak_pct": float('nan'), "comp_phi": float('nan')}

        results.append({
            "question": question[:40],
            "expected": expected,
            "output": output[:50],
            "correct": is_correct,
            "difficulty": difficulty,
            "peak_pct": metrics["peak_pct"],
            "comp_phi": metrics["comp_phi"],
        })

    # Aggregate
    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)

    # By difficulty
    by_diff = {}
    for d in sorted(set(r["difficulty"] for r in results)):
        d_results = [r for r in results if r["difficulty"] == d]
        d_correct = sum(1 for r in d_results if r["correct"])
        d_comp_phi = [r["comp_phi"] for r in d_results if not np.isnan(r["comp_phi"])]

        by_diff[d] = {
            "accuracy": d_correct / len(d_results) * 100 if d_results else 0,
            "correct": d_correct,
            "total": len(d_results),
            "avg_comp_phi": np.mean(d_comp_phi) if d_comp_phi else float('nan'),
        }

    # Correct vs incorrect geometry
    correct_phi = [r["comp_phi"] for r in results if r["correct"] and not np.isnan(r["comp_phi"])]
    incorrect_phi = [r["comp_phi"] for r in results if not r["correct"] and not np.isnan(r["comp_phi"])]

    return {
        "model": model_name,
        "total_accuracy": total_correct / total * 100,
        "total_correct": total_correct,
        "total": total,
        "by_difficulty": by_diff,
        "correct_comp_phi": np.mean(correct_phi) if correct_phi else float('nan'),
        "incorrect_comp_phi": np.mean(incorrect_phi) if incorrect_phi else float('nan'),
        "details": results,
    }


def main():
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("GEOMETRIC ALIGNMENT EVALUATION")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    adapter_path = "data/adapters/geometric_alignment_lora"

    results = {"timestamp": datetime.now().isoformat()}

    # Evaluate base model
    logger.info("\n" + "-" * 50)
    logger.info("BASE MODEL (no adapter)")
    logger.info("-" * 50)

    model, tokenizer = load(model_path)
    base_results = evaluate_model(model, tokenizer, TEST_PROBLEMS, "base")
    results["base"] = base_results

    logger.info(f"Overall: {base_results['total_accuracy']:.0f}% ({base_results['total_correct']}/{base_results['total']})")
    logger.info(f"Correct comp/φ: {base_results['correct_comp_phi']:.2f}")
    logger.info(f"Incorrect comp/φ: {base_results['incorrect_comp_phi']:.2f}")

    # Evaluate with adapter
    logger.info("\n" + "-" * 50)
    logger.info("WITH GEOMETRIC ALIGNMENT ADAPTER")
    logger.info("-" * 50)

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    adapted_results = evaluate_model(model, tokenizer, TEST_PROBLEMS, "adapted")
    results["adapted"] = adapted_results

    logger.info(f"Overall: {adapted_results['total_accuracy']:.0f}% ({adapted_results['total_correct']}/{adapted_results['total']})")
    logger.info(f"Correct comp/φ: {adapted_results['correct_comp_phi']:.2f}")
    logger.info(f"Incorrect comp/φ: {adapted_results['incorrect_comp_phi']:.2f}")

    # Comparison
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON BY DIFFICULTY")
    logger.info("=" * 70)

    logger.info(f"\n{'Diff':<6} {'Base Acc':<12} {'Adapt Acc':<12} {'Base φ':<10} {'Adapt φ':<10} {'Change'}")
    logger.info("-" * 70)

    for d in sorted(base_results["by_difficulty"].keys()):
        base_d = base_results["by_difficulty"][d]
        adapt_d = adapted_results["by_difficulty"].get(d, {"accuracy": 0, "avg_comp_phi": float('nan')})

        base_acc = f"{base_d['accuracy']:.0f}%"
        adapt_acc = f"{adapt_d['accuracy']:.0f}%"
        base_phi = f"{base_d['avg_comp_phi']:.2f}" if not np.isnan(base_d['avg_comp_phi']) else "N/A"
        adapt_phi = f"{adapt_d['avg_comp_phi']:.2f}" if not np.isnan(adapt_d['avg_comp_phi']) else "N/A"

        acc_change = adapt_d['accuracy'] - base_d['accuracy']
        change = f"+{acc_change:.0f}%" if acc_change >= 0 else f"{acc_change:.0f}%"

        logger.info(f"{d:<6} {base_acc:<12} {adapt_acc:<12} {base_phi:<10} {adapt_phi:<10} {change}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    acc_improvement = adapted_results['total_accuracy'] - base_results['total_accuracy']
    logger.info(f"\nAccuracy: {base_results['total_accuracy']:.0f}% → {adapted_results['total_accuracy']:.0f}% ({'+' if acc_improvement >= 0 else ''}{acc_improvement:.0f}%)")

    if not np.isnan(base_results['correct_comp_phi']) and not np.isnan(adapted_results['correct_comp_phi']):
        phi_change = adapted_results['correct_comp_phi'] - base_results['correct_comp_phi']
        closer_to_1 = abs(adapted_results['correct_comp_phi'] - 1.0) < abs(base_results['correct_comp_phi'] - 1.0)
        logger.info(f"Correct comp/φ: {base_results['correct_comp_phi']:.2f} → {adapted_results['correct_comp_phi']:.2f} {'(closer to 1.0!)' if closer_to_1 else ''}")

    # Key insight
    logger.info("\n" + "-" * 50)

    if adapted_results['total_accuracy'] > base_results['total_accuracy']:
        logger.info("✓ GEOMETRIC TRAINING IMPROVED ACCURACY")
    elif adapted_results['total_accuracy'] == base_results['total_accuracy']:
        logger.info("= ACCURACY UNCHANGED")
    else:
        logger.info("✗ ACCURACY DECREASED - need to adjust training")

    if not np.isnan(adapted_results['correct_comp_phi']):
        if abs(adapted_results['correct_comp_phi'] - 1.0) < 0.1:
            logger.info("✓ COMPRESSION/φ NEAR IDEAL (1.0)")
        elif abs(adapted_results['correct_comp_phi'] - 1.0) < abs(base_results['correct_comp_phi'] - 1.0):
            logger.info("✓ COMPRESSION/φ MOVED TOWARD IDEAL")

    # Save
    output_path = Path("data/experiments/geometric_alignment_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
