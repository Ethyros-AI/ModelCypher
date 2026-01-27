#!/usr/bin/env python3
"""Experiment: Test if harder problems require more dimensional expansion.

Prediction: Problems requiring more computation should show:
1. Higher peak dimension (more exploration needed)
2. Later peak layer (more processing time)
3. More expansion ratio (bigger dimensional journey)

Difficulty proxies:
- Number of reasoning steps in solution
- Answer magnitude (log scale)
- Problem word count
- Number of operations needed
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr
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


def estimate_problem_difficulty(question: str, full_solution: str, answer: str) -> Dict:
    """Estimate problem difficulty from multiple proxies."""

    # 1. Word count in question
    word_count = len(question.split())

    # 2. Number count in question (more numbers = more to track)
    numbers_in_question = re.findall(r'\d+\.?\d*', question)
    n_numbers = len(numbers_in_question)

    # 3. Answer magnitude (log scale)
    try:
        ans_val = abs(float(answer.replace(',', '')))
        answer_magnitude = np.log10(ans_val + 1)
    except:
        answer_magnitude = 0

    # 4. Reasoning steps (count sentences/lines in solution)
    # GSM8K solutions have step-by-step reasoning
    steps = [s.strip() for s in full_solution.split('\n') if s.strip() and not s.strip().startswith('####')]
    n_steps = len(steps)

    # 5. Operations count (count +, -, *, /, =)
    ops = re.findall(r'[+\-*/=]', full_solution)
    n_operations = len(ops)

    # 6. Sentence count in question
    sentences = [s.strip() for s in question.split('.') if s.strip()]
    n_sentences = len(sentences)

    # 7. Has comparison/relation words (makes tracking harder)
    relation_words = ['more', 'less', 'than', 'twice', 'half', 'each', 'per', 'total', 'left', 'remaining']
    n_relations = sum(1 for w in relation_words if w in question.lower())

    # Composite difficulty score (weighted sum, normalized)
    # Higher = harder
    difficulty_score = (
        0.2 * (word_count / 50) +           # Normalize by typical length
        0.1 * (n_numbers / 5) +             # Normalize by typical count
        0.1 * answer_magnitude +            # Already log scale
        0.3 * (n_steps / 5) +               # Key indicator
        0.2 * (n_operations / 10) +         # Normalize by typical count
        0.1 * n_relations                   # Extra complexity
    )

    return {
        "word_count": word_count,
        "n_numbers": n_numbers,
        "answer_magnitude": answer_magnitude,
        "n_steps": n_steps,
        "n_operations": n_operations,
        "n_sentences": n_sentences,
        "n_relations": n_relations,
        "difficulty_score": difficulty_score,
    }


def analyze_problem(
    model,
    tokenizer,
    question: str,
    full_solution: str,
    expected: str,
) -> Dict:
    """Analyze a problem: difficulty metrics + dimensional trajectory."""
    import mlx.core as mx
    from mlx_lm import generate

    # Get difficulty metrics
    difficulty = estimate_problem_difficulty(question, full_solution, expected)

    # Generate answer
    prompt = f"Question: {question}\n\nAnswer:"
    output = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False)

    # Extract predicted answer
    if "####" in output:
        answer_part = output.split("####")[-1].replace(",", "").replace("$", "").strip()
        nums = re.findall(r'-?\d+\.?\d*', answer_part)
        if nums:
            try:
                num_val = float(nums[0])
                predicted = str(int(num_val)) if num_val == int(num_val) else nums[0]
            except ValueError:
                predicted = nums[0] if nums else ""
        else:
            predicted = ""
    else:
        nums = re.findall(r'-?\d+', output.replace(",", ""))
        predicted = nums[-1] if nums else ""

    is_correct = predicted == expected

    # Get dimensional trajectory
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    dim_trajectory = []

    # Embedding layer
    emb_np = np.array(hidden[0].tolist())
    dim_trajectory.append(compute_intrinsic_dimension_twonn(emb_np))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)

        act_np = np.array(hidden[0].tolist())
        dim_trajectory.append(compute_intrinsic_dimension_twonn(act_np))

    # Compute dimensional metrics
    dim_traj = np.array(dim_trajectory)
    valid_dims = dim_traj[~np.isnan(dim_traj)]

    if len(valid_dims) > 2:
        peak_idx = np.nanargmax(dim_traj)
        peak_dim = dim_traj[peak_idx]
        initial_dim = dim_traj[0] if not np.isnan(dim_traj[0]) else valid_dims[0]
        final_dim = dim_traj[-1] if not np.isnan(dim_traj[-1]) else valid_dims[-1]

        expansion_ratio = peak_dim / initial_dim if initial_dim > 0.1 else float('nan')
        compression_ratio = peak_dim / final_dim if final_dim > 0.1 else float('nan')
    else:
        peak_idx = -1
        peak_dim = float('nan')
        initial_dim = float('nan')
        final_dim = float('nan')
        expansion_ratio = float('nan')
        compression_ratio = float('nan')

    return {
        "question": question[:200],
        "expected": expected,
        "predicted": predicted,
        "is_correct": is_correct,
        "difficulty": difficulty,
        "dimensional": {
            "peak_layer": int(peak_idx),
            "peak_dim": peak_dim,
            "initial_dim": initial_dim,
            "final_dim": final_dim,
            "expansion_ratio": expansion_ratio,
            "compression_ratio": compression_ratio,
            "compression_vs_phi": compression_ratio / PHI if not np.isnan(compression_ratio) else float('nan'),
        },
    }


def main():
    import mlx.core as mx
    from mlx_lm import load
    from modelcypher.core.use_cases.curriculum import BenchmarkLoader

    logger.info("=" * 70)
    logger.info("DIFFICULTY VS EXPANSION EXPERIMENT")
    logger.info("Testing: Harder problems need more dimensional expansion")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/unified_expansion_lora"

    logger.info(f"\nLoading model with adapter: {adapter_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    # Load GSM8K with full solutions
    logger.info("\nLoading GSM8K...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        gsm_data = list(ds)[:30]  # First 30 problems
        logger.info(f"Loaded {len(gsm_data)} problems from HuggingFace")
    except Exception as e:
        logger.error(f"Could not load GSM8K: {e}")
        return

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "adapter": adapter_path,
        "problems": [],
    }

    for i, item in enumerate(gsm_data):
        question = item["question"]
        full_solution = item["answer"]

        # Extract final answer
        if "####" in full_solution:
            expected = full_solution.split("####")[-1].strip()
        else:
            expected = full_solution.strip()

        logger.info(f"\n[{i+1}/{len(gsm_data)}] Analyzing problem...")

        analysis = analyze_problem(model, tokenizer, question, full_solution, expected)
        results["problems"].append(analysis)

        diff = analysis["difficulty"]
        dim = analysis["dimensional"]
        status = "OK" if analysis["is_correct"] else "WRONG"

        logger.info(f"  {status} | Difficulty: {diff['difficulty_score']:.2f} | "
                   f"Steps: {diff['n_steps']} | Expansion: {dim['expansion_ratio']:.1f}x | "
                   f"Peak layer: {dim['peak_layer']}")

    # Correlation analysis
    logger.info("\n" + "=" * 70)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 70)

    # Extract arrays for correlation
    valid_problems = [p for p in results["problems"]
                     if not np.isnan(p["dimensional"]["expansion_ratio"])]

    if len(valid_problems) < 5:
        logger.error("Not enough valid problems for correlation analysis")
        return

    difficulty_scores = np.array([p["difficulty"]["difficulty_score"] for p in valid_problems])
    n_steps = np.array([p["difficulty"]["n_steps"] for p in valid_problems])
    n_operations = np.array([p["difficulty"]["n_operations"] for p in valid_problems])
    word_counts = np.array([p["difficulty"]["word_count"] for p in valid_problems])

    expansion_ratios = np.array([p["dimensional"]["expansion_ratio"] for p in valid_problems])
    peak_dims = np.array([p["dimensional"]["peak_dim"] for p in valid_problems])
    peak_layers = np.array([p["dimensional"]["peak_layer"] for p in valid_problems])
    compression_vs_phi = np.array([p["dimensional"]["compression_vs_phi"] for p in valid_problems])

    is_correct = np.array([1 if p["is_correct"] else 0 for p in valid_problems])

    logger.info("\nDifficulty ↔ Dimensional Metrics (Spearman correlation):")
    logger.info("-" * 60)

    correlations = {}

    # Difficulty score correlations
    r, p = spearmanr(difficulty_scores, expansion_ratios)
    logger.info(f"Difficulty ↔ Expansion ratio:  r={r:+.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")
    correlations["difficulty_vs_expansion"] = {"r": r, "p": p}

    r, p = spearmanr(difficulty_scores, peak_dims)
    logger.info(f"Difficulty ↔ Peak dimension:   r={r:+.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")
    correlations["difficulty_vs_peak_dim"] = {"r": r, "p": p}

    r, p = spearmanr(difficulty_scores, peak_layers)
    logger.info(f"Difficulty ↔ Peak layer:       r={r:+.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")
    correlations["difficulty_vs_peak_layer"] = {"r": r, "p": p}

    # Steps correlations (most direct difficulty measure)
    logger.info("\nSteps ↔ Dimensional Metrics:")
    logger.info("-" * 60)

    r, p = spearmanr(n_steps, expansion_ratios)
    logger.info(f"N_steps ↔ Expansion ratio:     r={r:+.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")
    correlations["steps_vs_expansion"] = {"r": r, "p": p}

    r, p = spearmanr(n_steps, peak_dims)
    logger.info(f"N_steps ↔ Peak dimension:      r={r:+.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")
    correlations["steps_vs_peak_dim"] = {"r": r, "p": p}

    r, p = spearmanr(n_steps, peak_layers)
    logger.info(f"N_steps ↔ Peak layer:          r={r:+.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")
    correlations["steps_vs_peak_layer"] = {"r": r, "p": p}

    # Operations correlations
    logger.info("\nOperations ↔ Dimensional Metrics:")
    logger.info("-" * 60)

    r, p = spearmanr(n_operations, expansion_ratios)
    logger.info(f"N_ops ↔ Expansion ratio:       r={r:+.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")
    correlations["ops_vs_expansion"] = {"r": r, "p": p}

    r, p = spearmanr(n_operations, peak_dims)
    logger.info(f"N_ops ↔ Peak dimension:        r={r:+.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")
    correlations["ops_vs_peak_dim"] = {"r": r, "p": p}

    # Compression/φ vs correctness
    logger.info("\nCompression/φ Analysis:")
    logger.info("-" * 60)

    correct_compression = compression_vs_phi[is_correct == 1]
    incorrect_compression = compression_vs_phi[is_correct == 0]

    if len(correct_compression) > 0:
        logger.info(f"Correct answers:   compression/φ = {np.mean(correct_compression):.3f} ± {np.std(correct_compression):.3f}")
    if len(incorrect_compression) > 0:
        logger.info(f"Incorrect answers: compression/φ = {np.mean(incorrect_compression):.3f} ± {np.std(incorrect_compression):.3f}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("HYPOTHESIS TEST RESULTS")
    logger.info("=" * 70)

    sig_correlations = [(k, v) for k, v in correlations.items() if v["p"] < 0.05]

    if sig_correlations:
        logger.info("\nSignificant correlations found:")
        for k, v in sig_correlations:
            direction = "positive" if v["r"] > 0 else "negative"
            logger.info(f"  {k}: {direction} (r={v['r']:.3f}, p={v['p']:.4f})")
    else:
        logger.info("\nNo significant correlations at p < 0.05")

    # Test the specific prediction
    logger.info("\n" + "-" * 60)
    logger.info("PREDICTION: Harder problems require more expansion")
    logger.info("-" * 60)

    diff_exp_corr = correlations.get("difficulty_vs_expansion", {"r": 0, "p": 1})
    steps_exp_corr = correlations.get("steps_vs_expansion", {"r": 0, "p": 1})

    if diff_exp_corr["r"] > 0 and diff_exp_corr["p"] < 0.1:
        logger.info("✓ SUPPORTED: Higher difficulty → higher expansion ratio")
    elif steps_exp_corr["r"] > 0 and steps_exp_corr["p"] < 0.1:
        logger.info("✓ PARTIALLY SUPPORTED: More steps → higher expansion ratio")
    else:
        logger.info("✗ NOT SUPPORTED in current sample")
        logger.info("  Possible explanations:")
        logger.info("  - Template matching bypasses expansion for recognized patterns")
        logger.info("  - Sample size too small for detection")
        logger.info("  - Difficulty proxy doesn't capture computational complexity")

    results["correlations"] = correlations
    results["summary"] = {
        "n_problems": len(valid_problems),
        "accuracy": np.mean(is_correct) * 100,
        "mean_difficulty": np.mean(difficulty_scores),
        "mean_expansion": np.mean(expansion_ratios),
        "mean_compression_vs_phi": np.nanmean(compression_vs_phi),
    }

    # Save results
    output_path = Path("data/experiments/difficulty_vs_expansion.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
