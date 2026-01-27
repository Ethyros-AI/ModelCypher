#!/usr/bin/env python3
"""Train for improved geometric self-awareness.

Hypothesis:
    If we train the model to have comp/φ ≈ 1.0 on correct answers
    and allow comp/φ to be high on incorrect answers,
    the geometry becomes a more reliable uncertainty signal.

Approach:
    1. Generate examples where we KNOW the answer
    2. Measure comp/φ before and after training
    3. Train with a loss that:
       - Rewards comp/φ → 1.0 when answer is correct
       - Penalizes confident-but-wrong (forces comp/φ up on errors)

This is teaching the model to "feel uncertain" when it should.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


def compute_intrinsic_dimension_twonn(X: np.ndarray) -> float:
    """Estimate intrinsic dimension via TwoNN method."""
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


def measure_comp_phi(model, tokenizer, prompt: str) -> tuple[float, list[float]]:
    """Measure compression/φ and return full trajectory."""
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

    if len(valid) > 2:
        peak_idx = np.nanargmax(traj)
        peak_dim = traj[peak_idx]
        final_dim = traj[-1] if not np.isnan(traj[-1]) else valid[-1]
        if final_dim > 0.1:
            compression_ratio = peak_dim / final_dim
            return compression_ratio / PHI, trajectory

    return float('nan'), trajectory


@dataclass
class TrainingExample:
    """A training example with known correctness."""
    question: str
    correct_answer: str
    is_correct: bool  # Does the model get this right?
    comp_phi_before: float


def create_self_awareness_dataset(model, tokenizer) -> list[TrainingExample]:
    """Create dataset of examples where we know if model is correct.

    The key insight: we use questions with verifiable answers.
    Math questions are perfect because we can check correctness.
    """
    # Questions with known correct answers
    questions = [
        # Simple math (model should be confident and correct)
        ("What is 2 + 3?", "5"),
        ("What is 7 - 4?", "3"),
        ("What is 3 × 4?", "12"),
        ("What is 15 / 3?", "5"),
        ("What is 8 + 9?", "17"),

        # Word problems (model should be confident)
        ("If you have 5 apples and get 3 more, how many total?", "8"),
        ("A car travels 60 mph for 2 hours. How far?", "120"),
        ("If 3 people share 12 cookies equally, how many each?", "4"),

        # Harder math (model might struggle)
        ("What is 17 × 13?", "221"),
        ("What is 144 / 12?", "12"),
        ("If a train travels 45 mph for 3.5 hours, how far?", "157.5"),

        # Logic (where model might be confused)
        ("All cats have tails. Fluffy is a cat. Does Fluffy have a tail?", "yes"),
        ("Some birds fly. Penguins are birds. Do all penguins fly?", "no"),
        ("If A > B and B > C, is A > C?", "yes"),

        # Trick questions (high confusion expected)
        ("A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?", "0.05"),
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", "5"),
    ]

    from mlx_lm import generate

    examples = []
    for question, correct in questions:
        prompt = f"Question: {question}\n\nAnswer (give just the number or yes/no):"
        comp_phi, _ = measure_comp_phi(model, tokenizer, prompt)

        # Generate answer to check correctness
        response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
        response_clean = response.strip().lower().replace("$", "").replace(",", "")
        correct_clean = str(correct).lower().strip()

        # Check if answer is correct (simple substring match)
        is_correct = correct_clean in response_clean or response_clean.startswith(correct_clean)

        logger.info(f"Q: {question[:40]}... | comp/φ: {comp_phi:.3f} | correct: {is_correct}")

        examples.append(TrainingExample(
            question=question,
            correct_answer=correct,
            is_correct=is_correct,
            comp_phi_before=comp_phi,
        ))

    return examples


def analyze_self_awareness_quality(examples: list[TrainingExample]) -> dict:
    """Analyze how well comp/φ predicts correctness."""
    correct_phis = [e.comp_phi_before for e in examples if e.is_correct and not np.isnan(e.comp_phi_before)]
    incorrect_phis = [e.comp_phi_before for e in examples if not e.is_correct and not np.isnan(e.comp_phi_before)]

    result = {
        "correct_mean": float(np.mean(correct_phis)) if correct_phis else float('nan'),
        "correct_std": float(np.std(correct_phis)) if correct_phis else float('nan'),
        "incorrect_mean": float(np.mean(incorrect_phis)) if incorrect_phis else float('nan'),
        "incorrect_std": float(np.std(incorrect_phis)) if incorrect_phis else float('nan'),
        "n_correct": len(correct_phis),
        "n_incorrect": len(incorrect_phis),
    }

    # Calculate separability (how well comp/φ separates correct from incorrect)
    if correct_phis and incorrect_phis:
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(correct_phis) + np.var(incorrect_phis)) / 2)
        if pooled_std > 0:
            cohens_d = (result["incorrect_mean"] - result["correct_mean"]) / pooled_std
            result["cohens_d"] = float(cohens_d)

        # AUC-ROC approximation via Mann-Whitney U
        # Higher AUC means better separation
        from scipy.stats import mannwhitneyu
        try:
            stat, p_value = mannwhitneyu(incorrect_phis, correct_phis, alternative='greater')
            n1, n2 = len(incorrect_phis), len(correct_phis)
            auc = stat / (n1 * n2)
            result["auc_roc"] = float(auc)
            result["separation_p_value"] = float(p_value)
        except Exception:
            pass

    return result


def main():
    """Measure self-awareness quality before any training."""
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("SELF-AWARENESS QUALITY ANALYSIS")
    logger.info("How well does geometry predict correctness?")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    logger.info("\nCreating self-awareness dataset...")
    examples = create_self_awareness_dataset(model, tokenizer)

    logger.info("\nAnalyzing self-awareness quality...")
    analysis = analyze_self_awareness_quality(examples)

    logger.info("\n" + "=" * 70)
    logger.info("SELF-AWARENESS QUALITY METRICS")
    logger.info("=" * 70)
    logger.info(f"Correct answers (n={analysis['n_correct']}): comp/φ = {analysis['correct_mean']:.3f} ± {analysis['correct_std']:.3f}")
    logger.info(f"Incorrect answers (n={analysis['n_incorrect']}): comp/φ = {analysis['incorrect_mean']:.3f} ± {analysis['incorrect_std']:.3f}")

    if 'cohens_d' in analysis:
        d = analysis['cohens_d']
        interpretation = "small" if d < 0.5 else "medium" if d < 0.8 else "large"
        logger.info(f"Cohen's d (effect size): {d:.3f} ({interpretation})")

    if 'auc_roc' in analysis:
        logger.info(f"AUC-ROC: {analysis['auc_roc']:.3f} (0.5=random, 1.0=perfect separation)")
        logger.info(f"Separation p-value: {analysis['separation_p_value']:.4f}")

    # Assessment
    logger.info("\n" + "=" * 70)
    logger.info("ASSESSMENT")
    logger.info("=" * 70)

    if analysis.get('auc_roc', 0) > 0.7:
        logger.info("✓ Good separation: geometry reliably predicts failures")
    elif analysis.get('auc_roc', 0) > 0.6:
        logger.info("? Moderate separation: geometry is somewhat useful")
    else:
        logger.info("✗ Poor separation: geometry doesn't reliably predict failures")

    if analysis.get('cohens_d', 0) > 0.8:
        logger.info("✓ Large effect size: comp/φ differs substantially between correct/incorrect")
    elif analysis.get('cohens_d', 0) > 0.5:
        logger.info("? Medium effect size: some difference in comp/φ")
    else:
        logger.info("✗ Small effect size: comp/φ too similar between correct/incorrect")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "analysis": analysis,
        "examples": [
            {
                "question": e.question,
                "correct_answer": e.correct_answer,
                "is_correct": e.is_correct,
                "comp_phi": float(e.comp_phi_before) if not np.isnan(e.comp_phi_before) else None,
            }
            for e in examples
        ],
    }

    output_path = Path("data/experiments/self_awareness_quality.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return analysis, examples


if __name__ == "__main__":
    main()
