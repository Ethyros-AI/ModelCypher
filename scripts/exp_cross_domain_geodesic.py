#!/usr/bin/env python3
"""Experiment: Test if cross-domain transfer shares geodesic structure.

Hypothesis: The math adapter improved ARC-Challenge by 6% because the
dimensional trajectory (geodesic structure) is domain-independent.

If true:
1. Correct answers across domains should have similar dimensional signatures
2. The φ compression ratio should appear in both math and science
3. Peak layers should be similar for problems of similar difficulty
4. The "structure of thinking" is universal, not domain-specific
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
from scipy.stats import spearmanr, ks_2samp, ttest_ind
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


def get_dimensional_trajectory(model, tokenizer, prompt: str) -> Dict:
    """Get dimensional trajectory through all layers."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    trajectory = []

    # Embedding layer
    emb_np = np.array(hidden[0].tolist())
    trajectory.append(compute_intrinsic_dimension_twonn(emb_np))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)

        act_np = np.array(hidden[0].tolist())
        trajectory.append(compute_intrinsic_dimension_twonn(act_np))

    # Compute metrics
    traj = np.array(trajectory)
    valid = traj[~np.isnan(traj)]

    if len(valid) > 2:
        peak_idx = np.nanargmax(traj)
        peak_dim = traj[peak_idx]
        initial_dim = traj[0] if not np.isnan(traj[0]) else valid[0]
        final_dim = traj[-1] if not np.isnan(traj[-1]) else valid[-1]

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
        "trajectory": trajectory,
        "peak_layer": int(peak_idx),
        "peak_dim": peak_dim,
        "initial_dim": initial_dim,
        "final_dim": final_dim,
        "expansion_ratio": expansion_ratio,
        "compression_ratio": compression_ratio,
        "compression_vs_phi": compression_ratio / PHI if not np.isnan(compression_ratio) else float('nan'),
    }


def evaluate_gsm8k(model, tokenizer, limit: int = 20) -> List[Dict]:
    """Evaluate GSM8K problems with dimensional analysis."""
    from mlx_lm import generate

    logger.info("Loading GSM8K...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        data = list(ds)[:limit]
    except Exception as e:
        logger.error(f"Could not load GSM8K: {e}")
        return []

    results = []
    for i, item in enumerate(data):
        question = item["question"]
        full_solution = item["answer"]

        if "####" in full_solution:
            expected = full_solution.split("####")[-1].strip()
        else:
            expected = full_solution.strip()

        prompt = f"Question: {question}\n\nAnswer:"
        output = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)

        # Extract answer
        if "####" in output:
            answer_part = output.split("####")[-1].replace(",", "").replace("$", "").strip()
            nums = re.findall(r'-?\d+\.?\d*', answer_part)
            if nums:
                try:
                    num_val = float(nums[0])
                    predicted = str(int(num_val)) if num_val == int(num_val) else nums[0]
                except:
                    predicted = nums[0] if nums else ""
            else:
                predicted = ""
        else:
            nums = re.findall(r'-?\d+', output.replace(",", ""))
            predicted = nums[-1] if nums else ""

        is_correct = predicted == expected

        # Get dimensional trajectory
        traj_data = get_dimensional_trajectory(model, tokenizer, prompt)

        results.append({
            "domain": "math",
            "question": question[:150],
            "expected": expected,
            "predicted": predicted,
            "is_correct": is_correct,
            **traj_data,
        })

        status = "OK" if is_correct else "WRONG"
        logger.info(f"  [GSM8K {i+1}/{limit}] {status} | Peak L{traj_data['peak_layer']} | "
                   f"Comp/φ: {traj_data['compression_vs_phi']:.2f}")

    return results


def evaluate_arc_challenge(model, tokenizer, limit: int = 20) -> List[Dict]:
    """Evaluate ARC-Challenge problems with dimensional analysis."""
    from mlx_lm import generate

    logger.info("Loading ARC-Challenge...")
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        data = list(ds)[:limit]
    except Exception as e:
        logger.error(f"Could not load ARC-Challenge: {e}")
        return []

    results = []
    for i, item in enumerate(data):
        question = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]

        choice_labels = choices["label"]
        choice_texts = choices["text"]

        formatted_choices = "\n".join(
            f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)
        )

        answer_idx = choice_labels.index(answer_key) if answer_key in choice_labels else 0
        expected = choice_texts[answer_idx]

        prompt = f"Question: {question}\n{formatted_choices}\n\nThe answer is:"
        output = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)

        # Check if answer matches
        output_lower = output.lower()
        predicted = ""
        is_correct = False

        for label, text in zip(choice_labels, choice_texts):
            if text.lower() in output_lower or f"{label}." in output or f"({label})" in output:
                predicted = text
                if text == expected:
                    is_correct = True
                break

        # Get dimensional trajectory
        traj_data = get_dimensional_trajectory(model, tokenizer, prompt)

        results.append({
            "domain": "science",
            "question": question[:150],
            "expected": expected,
            "predicted": predicted,
            "is_correct": is_correct,
            **traj_data,
        })

        status = "OK" if is_correct else "WRONG"
        logger.info(f"  [ARC {i+1}/{limit}] {status} | Peak L{traj_data['peak_layer']} | "
                   f"Comp/φ: {traj_data['compression_vs_phi']:.2f}")

    return results


def main():
    import mlx.core as mx
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("CROSS-DOMAIN GEODESIC STRUCTURE EXPERIMENT")
    logger.info("Testing: Do math and science share dimensional structure?")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/unified_expansion_lora"

    logger.info(f"\nLoading model with adapter: {adapter_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    # Evaluate both domains
    logger.info("\n" + "=" * 50)
    logger.info("MATH DOMAIN (GSM8K)")
    logger.info("=" * 50)
    math_results = evaluate_gsm8k(model, tokenizer, limit=20)

    logger.info("\n" + "=" * 50)
    logger.info("SCIENCE DOMAIN (ARC-Challenge)")
    logger.info("=" * 50)
    science_results = evaluate_arc_challenge(model, tokenizer, limit=20)

    # Combine results
    all_results = math_results + science_results

    # Analysis
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-DOMAIN COMPARISON")
    logger.info("=" * 70)

    # Extract metrics by domain and correctness
    def get_metrics(results: List[Dict], correct_only: bool = None):
        filtered = results
        if correct_only is not None:
            filtered = [r for r in results if r["is_correct"] == correct_only]

        return {
            "n": len(filtered),
            "peak_layers": [r["peak_layer"] for r in filtered],
            "peak_dims": [r["peak_dim"] for r in filtered if not np.isnan(r["peak_dim"])],
            "compression_vs_phi": [r["compression_vs_phi"] for r in filtered
                                   if not np.isnan(r["compression_vs_phi"])],
            "expansion_ratios": [r["expansion_ratio"] for r in filtered
                                if not np.isnan(r["expansion_ratio"])],
        }

    math_correct = get_metrics(math_results, correct_only=True)
    math_wrong = get_metrics(math_results, correct_only=False)
    science_correct = get_metrics(science_results, correct_only=True)
    science_wrong = get_metrics(science_results, correct_only=False)

    # Summary stats
    logger.info("\n--- Accuracy ---")
    math_acc = len([r for r in math_results if r["is_correct"]]) / len(math_results) * 100
    science_acc = len([r for r in science_results if r["is_correct"]]) / len(science_results) * 100
    logger.info(f"Math (GSM8K):        {math_acc:.0f}%")
    logger.info(f"Science (ARC-Chal):  {science_acc:.0f}%")

    logger.info("\n--- Compression/φ by Domain & Correctness ---")
    if math_correct["compression_vs_phi"]:
        logger.info(f"Math Correct:    {np.mean(math_correct['compression_vs_phi']):.3f} ± {np.std(math_correct['compression_vs_phi']):.3f}")
    if math_wrong["compression_vs_phi"]:
        logger.info(f"Math Wrong:      {np.mean(math_wrong['compression_vs_phi']):.3f} ± {np.std(math_wrong['compression_vs_phi']):.3f}")
    if science_correct["compression_vs_phi"]:
        logger.info(f"Science Correct: {np.mean(science_correct['compression_vs_phi']):.3f} ± {np.std(science_correct['compression_vs_phi']):.3f}")
    if science_wrong["compression_vs_phi"]:
        logger.info(f"Science Wrong:   {np.mean(science_wrong['compression_vs_phi']):.3f} ± {np.std(science_wrong['compression_vs_phi']):.3f}")

    logger.info("\n--- Peak Layer by Domain ---")
    if math_correct["peak_layers"]:
        logger.info(f"Math:    {np.mean(math_correct['peak_layers']):.1f} ± {np.std(math_correct['peak_layers']):.1f}")
    if science_correct["peak_layers"]:
        logger.info(f"Science: {np.mean(science_correct['peak_layers']):.1f} ± {np.std(science_correct['peak_layers']):.1f}")

    # Statistical tests
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL TESTS")
    logger.info("=" * 70)

    results_summary = {"tests": []}

    # Test 1: Do correct answers across domains have similar compression/φ?
    if math_correct["compression_vs_phi"] and science_correct["compression_vs_phi"]:
        stat, p = ttest_ind(math_correct["compression_vs_phi"], science_correct["compression_vs_phi"])
        logger.info(f"\nT-test: Math correct vs Science correct (compression/φ)")
        logger.info(f"  t={stat:.3f}, p={p:.4f}")
        if p > 0.05:
            logger.info(f"  ✓ NOT significantly different - shared structure!")
        else:
            logger.info(f"  ✗ Significantly different")
        results_summary["tests"].append({
            "name": "correct_compression_cross_domain",
            "t": stat, "p": p,
            "same_structure": p > 0.05
        })

    # Test 2: Do correct answers share peak layer distribution?
    if math_correct["peak_layers"] and science_correct["peak_layers"]:
        stat, p = ks_2samp(math_correct["peak_layers"], science_correct["peak_layers"])
        logger.info(f"\nKS-test: Math correct vs Science correct (peak layer)")
        logger.info(f"  D={stat:.3f}, p={p:.4f}")
        if p > 0.05:
            logger.info(f"  ✓ Same distribution - shared structure!")
        else:
            logger.info(f"  ✗ Different distribution")
        results_summary["tests"].append({
            "name": "correct_peak_layer_cross_domain",
            "D": stat, "p": p,
            "same_structure": p > 0.05
        })

    # Test 3: Correct vs Wrong within each domain (sanity check)
    logger.info("\n--- Within-Domain: Correct vs Wrong ---")

    if math_correct["compression_vs_phi"] and math_wrong["compression_vs_phi"]:
        stat, p = ttest_ind(math_correct["compression_vs_phi"], math_wrong["compression_vs_phi"])
        logger.info(f"Math: correct vs wrong compression/φ: t={stat:.3f}, p={p:.4f}")

    if science_correct["compression_vs_phi"] and science_wrong["compression_vs_phi"]:
        stat, p = ttest_ind(science_correct["compression_vs_phi"], science_wrong["compression_vs_phi"])
        logger.info(f"Science: correct vs wrong compression/φ: t={stat:.3f}, p={p:.4f}")

    # Test 4: Correlation of metrics across domains
    logger.info("\n--- Cross-Domain Metric Correlations ---")

    all_correct = [r for r in all_results if r["is_correct"]]
    if len(all_correct) > 5:
        is_math = np.array([1 if r["domain"] == "math" else 0 for r in all_correct])
        comp_phi = np.array([r["compression_vs_phi"] for r in all_correct
                            if not np.isnan(r["compression_vs_phi"])])
        peak_layers = np.array([r["peak_layer"] for r in all_correct])

        # Does domain predict compression/φ? (should NOT if structure is shared)
        if len(comp_phi) == len(is_math):
            r, p = spearmanr(is_math, comp_phi)
            logger.info(f"Domain ↔ Compression/φ: r={r:.3f}, p={p:.4f}")
            if abs(r) < 0.3 and p > 0.05:
                logger.info(f"  ✓ Weak/no correlation - structure is domain-independent!")
            else:
                logger.info(f"  Domain affects compression ratio")

    # Final verdict
    logger.info("\n" + "=" * 70)
    logger.info("HYPOTHESIS VERDICT")
    logger.info("=" * 70)

    shared_evidence = sum(1 for t in results_summary["tests"] if t.get("same_structure", False))
    total_tests = len(results_summary["tests"])

    if shared_evidence >= total_tests / 2:
        logger.info(f"\n✓ SUPPORTED: Cross-domain transfer shares geodesic structure")
        logger.info(f"  {shared_evidence}/{total_tests} tests show shared dimensional structure")
        logger.info(f"  The math adapter works for science because the STRUCTURE is universal")
    else:
        logger.info(f"\n✗ NOT SUPPORTED: Domains have different geodesic structure")
        logger.info(f"  Only {shared_evidence}/{total_tests} tests show shared structure")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "adapter": adapter_path,
        "math_results": math_results,
        "science_results": science_results,
        "summary": {
            "math_accuracy": math_acc,
            "science_accuracy": science_acc,
            "math_compression_phi": np.mean(math_correct["compression_vs_phi"]) if math_correct["compression_vs_phi"] else None,
            "science_compression_phi": np.mean(science_correct["compression_vs_phi"]) if science_correct["compression_vs_phi"] else None,
        },
        "tests": results_summary["tests"],
    }

    output_path = Path("data/experiments/cross_domain_geodesic.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
