#!/usr/bin/env python3
"""Comprehensive benchmark of LFM2-350M with geometric analysis.

For each task category, we measure:
1. Accuracy (can it solve the problem?)
2. Geometric signature (how does it process the problem?)

This tells us WHERE to focus training.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# =============================================================================
# BENCHMARK TASKS
# =============================================================================

# Math reasoning (GSM8K style)
MATH_PROBLEMS = [
    ("If Sarah has 5 apples and buys 3 more, how many apples does she have?", "8"),
    ("A store sells pencils for $2 each. Tom buys 4 pencils. How much does he spend?", "8"),
    ("John read 15 pages Monday and 20 pages Tuesday. How many pages total?", "35"),
    ("A train travels 60 mph for 2 hours. How far does it go?", "120"),
    ("If 3 people share 12 cookies equally, how many does each person get?", "4"),
    ("Maria has $20. She spends $7 on lunch. How much does she have left?", "13"),
    ("A rectangle is 5 meters long and 3 meters wide. What is its area?", "15"),
    ("If you save $5 per week, how much do you save in 8 weeks?", "40"),
    ("A baker makes 24 cookies and puts 6 on each plate. How many plates?", "4"),
    ("The temperature was 10°C and dropped 15 degrees. What is it now?", "-5"),
]

# Science reasoning (ARC style)
SCIENCE_PROBLEMS = [
    ("What causes day and night on Earth?", "rotation"),
    ("What state of matter has a fixed shape and volume?", "solid"),
    ("What organ pumps blood through the body?", "heart"),
    ("What gas do plants produce during photosynthesis?", "oxygen"),
    ("What force keeps planets in orbit around the Sun?", "gravity"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What type of animal is a frog - mammal, reptile, or amphibian?", "amphibian"),
    ("What part of the plant absorbs water from the soil?", "roots"),
    ("What causes shadows?", "light"),
    ("Is the Moon a planet, star, or satellite?", "satellite"),
]

# Logic and deduction
LOGIC_PROBLEMS = [
    ("All dogs are mammals. Rex is a dog. Is Rex a mammal?", "yes"),
    ("If it rains, the ground gets wet. It rained. Is the ground wet?", "yes"),
    ("Some birds can fly. Penguins are birds. Can all penguins fly?", "no"),
    ("If A is bigger than B, and B is bigger than C, is A bigger than C?", "yes"),
    ("All squares are rectangles. Is every rectangle a square?", "no"),
    ("If no fish can walk, and salmon are fish, can salmon walk?", "no"),
    ("Tom is taller than Jane. Jane is taller than Mike. Who is shortest?", "mike"),
    ("If today is Monday, what day was it 3 days ago?", "friday"),
    ("All cats have tails. Fluffy is a cat. Does Fluffy have a tail?", "yes"),
    ("Some fruits are red. Apples are fruits. Are all apples red?", "no"),
]

# Knowledge recall
KNOWLEDGE_PROBLEMS = [
    ("What is the capital of France?", "paris"),
    ("Who wrote Romeo and Juliet?", "shakespeare"),
    ("What is the largest planet in our solar system?", "jupiter"),
    ("In what year did World War II end?", "1945"),
    ("What is the chemical symbol for gold?", "au"),
    ("Who painted the Mona Lisa?", "vinci"),
    ("What is the tallest mountain on Earth?", "everest"),
    ("What language is spoken in Brazil?", "portuguese"),
    ("How many continents are there?", "7"),
    ("What is the speed of light in km/s (approximately)?", "300000"),
]

# Instruction following
INSTRUCTION_PROBLEMS = [
    ("List three colors.", ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown"]),
    ("Say hello in one word.", ["hello", "hi", "greetings", "hey"]),
    ("Name a fruit.", ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "berry", "melon", "cherry"]),
    ("What comes after 5?", ["6", "six"]),
    ("Name a country in Europe.", ["france", "germany", "italy", "spain", "uk", "poland", "greece", "sweden", "norway"]),
    ("What is 2+2?", ["4", "four"]),
    ("Name an animal that swims.", ["fish", "dolphin", "whale", "shark", "seal", "turtle", "penguin", "otter"]),
    ("What color is the sky on a clear day?", ["blue"]),
    ("Name a day of the week.", ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]),
    ("What do you call a baby dog?", ["puppy", "pup"]),
]

# Commonsense reasoning
COMMONSENSE_PROBLEMS = [
    ("You're hungry. Should you eat food or rocks?", "food"),
    ("It's raining. Should you use an umbrella or sunglasses?", "umbrella"),
    ("To cut paper, would you use scissors or a spoon?", "scissors"),
    ("Which is heavier: a ton of feathers or a ton of bricks?", "same"),
    ("Can a fish climb a tree?", "no"),
    ("Do humans need to breathe?", "yes"),
    ("Is ice hot or cold?", "cold"),
    ("Can you see in complete darkness?", "no"),
    ("Do birds have wings?", "yes"),
    ("Is water wet?", "yes"),
]


def compute_intrinsic_dimension_twonn(X: np.ndarray) -> float:
    """Estimate intrinsic dimension via TwoNN method."""
    if len(X) < 10:
        return float('nan')

    k = min(3, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nn.kneighbors(X)

    d1 = distances[:, 1]
    d2 = distances[:, 2]

    valid = d1 > 1e-10
    if valid.sum() < 5:
        return float('nan')

    mu = d2[valid] / d1[valid]
    mu = mu[mu > 1]

    if len(mu) < 5:
        return float('nan')

    log_mu = np.log(mu)
    d = len(log_mu) / np.sum(log_mu)

    return float(d)


def get_geometric_signature(model, tokenizer, prompt: str) -> Dict:
    """Get geometric signature for a prompt."""
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
    else:
        peak_idx = -1
        compression_vs_phi = float('nan')

    return {
        "peak_layer_pct": peak_idx / n_layers * 100 if peak_idx >= 0 else float('nan'),
        "compression_vs_phi": compression_vs_phi,
    }


def evaluate_answer(output: str, expected, is_list: bool = False) -> bool:
    """Check if output contains expected answer."""
    output_lower = output.lower().strip()

    if is_list:
        # For instruction following, check if any expected word is in output
        return any(exp.lower() in output_lower for exp in expected)
    else:
        # For exact answers, check if expected is in output
        expected_lower = str(expected).lower()

        # Handle numeric answers
        if expected_lower.lstrip('-').isdigit():
            nums = re.findall(r'-?\d+', output.replace(",", ""))
            if nums:
                return expected_lower in nums

        # Handle word answers
        return expected_lower in output_lower


def run_benchmark(model, tokenizer, category: str, problems: List, is_instruction: bool = False) -> Dict:
    """Run benchmark for a category."""
    from mlx_lm import generate

    results = []

    for question, expected in problems:
        prompt = f"Question: {question}\n\nAnswer:"

        try:
            output = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
        except Exception as e:
            output = f"ERROR: {e}"

        is_correct = evaluate_answer(output, expected, is_list=is_instruction)

        # Get geometric signature
        try:
            geo = get_geometric_signature(model, tokenizer, prompt)
        except Exception:
            geo = {"peak_layer_pct": float('nan'), "compression_vs_phi": float('nan')}

        results.append({
            "question": question,
            "expected": expected if not is_instruction else expected[0],
            "output": output[:100],
            "correct": is_correct,
            "peak_layer_pct": geo["peak_layer_pct"],
            "compression_vs_phi": geo["compression_vs_phi"],
        })

    # Aggregate
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) * 100

    correct_results = [r for r in results if r["correct"] and not np.isnan(r["compression_vs_phi"])]
    incorrect_results = [r for r in results if not r["correct"] and not np.isnan(r["compression_vs_phi"])]

    return {
        "category": category,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(results),
        "correct_comp_phi": np.mean([r["compression_vs_phi"] for r in correct_results]) if correct_results else float('nan'),
        "incorrect_comp_phi": np.mean([r["compression_vs_phi"] for r in incorrect_results]) if incorrect_results else float('nan'),
        "correct_peak_pct": np.mean([r["peak_layer_pct"] for r in correct_results]) if correct_results else float('nan'),
        "incorrect_peak_pct": np.mean([r["peak_layer_pct"] for r in incorrect_results]) if incorrect_results else float('nan'),
        "details": results,
    }


def main():
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("LFM2-350M COMPREHENSIVE BENCHMARK")
    logger.info("Measuring capability AND geometry")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"\nLoading model: {model_path}")
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    hidden_dim = model.model.embed_tokens.weight.shape[1]
    logger.info(f"Architecture: {n_layers} layers, {hidden_dim} hidden dim")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "architecture": {"n_layers": n_layers, "hidden_dim": hidden_dim},
        "benchmarks": [],
    }

    # Run all benchmarks
    benchmarks = [
        ("math", MATH_PROBLEMS, False),
        ("science", SCIENCE_PROBLEMS, False),
        ("logic", LOGIC_PROBLEMS, False),
        ("knowledge", KNOWLEDGE_PROBLEMS, False),
        ("instruction", INSTRUCTION_PROBLEMS, True),
        ("commonsense", COMMONSENSE_PROBLEMS, False),
    ]

    for category, problems, is_instruction in benchmarks:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"CATEGORY: {category.upper()}")
        logger.info(f"{'=' * 50}")

        bench_result = run_benchmark(model, tokenizer, category, problems, is_instruction)
        results["benchmarks"].append(bench_result)

        logger.info(f"  Accuracy: {bench_result['accuracy']:.0f}% ({bench_result['correct']}/{bench_result['total']})")

        if not np.isnan(bench_result['correct_comp_phi']):
            logger.info(f"  Correct comp/φ: {bench_result['correct_comp_phi']:.2f}")
        if not np.isnan(bench_result['incorrect_comp_phi']):
            logger.info(f"  Incorrect comp/φ: {bench_result['incorrect_comp_phi']:.2f}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Category':<15} {'Accuracy':<12} {'Correct φ':<12} {'Wrong φ':<12}")
    logger.info("-" * 55)

    for bench in results["benchmarks"]:
        acc = f"{bench['accuracy']:.0f}%"
        cor_phi = f"{bench['correct_comp_phi']:.2f}" if not np.isnan(bench['correct_comp_phi']) else "N/A"
        inc_phi = f"{bench['incorrect_comp_phi']:.2f}" if not np.isnan(bench['incorrect_comp_phi']) else "N/A"
        logger.info(f"{bench['category']:<15} {acc:<12} {cor_phi:<12} {inc_phi:<12}")

    # Overall stats
    total_correct = sum(b["correct"] for b in results["benchmarks"])
    total_problems = sum(b["total"] for b in results["benchmarks"])
    overall_accuracy = total_correct / total_problems * 100

    logger.info("-" * 55)
    logger.info(f"{'OVERALL':<15} {overall_accuracy:.0f}%")

    # Geometric analysis
    logger.info("\n" + "=" * 70)
    logger.info("GEOMETRIC ANALYSIS")
    logger.info("=" * 70)

    all_correct = []
    all_incorrect = []
    for bench in results["benchmarks"]:
        for d in bench["details"]:
            if not np.isnan(d["compression_vs_phi"]):
                if d["correct"]:
                    all_correct.append(d["compression_vs_phi"])
                else:
                    all_incorrect.append(d["compression_vs_phi"])

    if all_correct:
        logger.info(f"\nCorrect answers:   comp/φ = {np.mean(all_correct):.2f} ± {np.std(all_correct):.2f}")
    if all_incorrect:
        logger.info(f"Incorrect answers: comp/φ = {np.mean(all_incorrect):.2f} ± {np.std(all_incorrect):.2f}")

    # Identify weak areas
    logger.info("\n" + "=" * 70)
    logger.info("AREAS FOR IMPROVEMENT")
    logger.info("=" * 70)

    weak_categories = sorted(results["benchmarks"], key=lambda x: x["accuracy"])
    for bench in weak_categories[:3]:
        if bench["accuracy"] < 100:
            logger.info(f"\n{bench['category'].upper()}: {bench['accuracy']:.0f}% accuracy")
            failures = [d for d in bench["details"] if not d["correct"]]
            for f in failures[:3]:
                logger.info(f"  ✗ Q: {f['question'][:50]}...")
                logger.info(f"    Expected: {f['expected']}, Got: {f['output'][:30]}...")

    results["overall_accuracy"] = overall_accuracy
    results["correct_comp_phi_mean"] = np.mean(all_correct) if all_correct else float('nan')
    results["incorrect_comp_phi_mean"] = np.mean(all_incorrect) if all_incorrect else float('nan')

    # Save
    output_path = Path("data/experiments/lfm2_350m_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
